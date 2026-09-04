from __future__ import annotations

import csv
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import time
from typing import Mapping

import music21
import numpy as np
import psutil
import torch

from next_token_experiment.data.preprocess import prepare_corpus
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.experiment.storage import write_csv

from .artifact_audit import audit_run
from .classical_models import FiniteGlobalHMM, GlobalHDPHMM
from .config import LearningCurveConfig, build_default_learning_curve_config
from .resource_monitor import ResourceMonitor
from .runner import (
    _build_hardware_manifest,
    _build_transformer_dataloaders,
    _build_transformer_model,
    _transformer_summary_row,
)
from .vomm import VariableOrderMarkovModel


EXPECTED_MODELS = ("finite_hmm", "hdp_hmm", "transformer", "vomm")
EXPECTED_CACHE_SHA256 = "F42F9D7AB8550A4C366CFCF410C3CF67C85FAD46F5C4F54818403DEEC328E144"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _seed_key(value: object) -> tuple[int, object]:
    try:
        return 0, int(str(value))
    except ValueError:
        return 1, str(value)


def select_source_configurations(
    source_run: str | Path,
    *,
    fraction: float,
) -> dict[str, dict[str, object]]:
    rows = [
        row
        for row in _read_csv(Path(source_run) / "results_raw.csv")
        if math.isclose(float(row["frac"]), fraction)
    ]
    selected: dict[str, dict[str, object]] = {}
    for model in EXPECTED_MODELS:
        candidates = [row for row in rows if row["model"] == model]
        if not candidates:
            raise ValueError(f"source run has no {model} row for fraction {fraction}")
        row = min(
            candidates,
            key=lambda item: (_seed_key(item["data_seed"]), _seed_key(item["model_seed"])),
        )
        model_seed: object = row["model_seed"]
        try:
            model_seed = int(str(model_seed))
        except ValueError:
            pass
        selected[model] = {
            "data_seed": int(row["data_seed"]),
            "model_seed": model_seed,
            "hyperparameters": json.loads(row["hyperparams_json"]),
        }
    return selected


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "fit_seconds",
        "evaluation_seconds",
        "peak_process_memory_bytes",
        "peak_gpu_memory_bytes",
    )
    summaries = []
    for model in EXPECTED_MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        summary: dict[str, object] = {"model": model, "repetitions": len(model_rows)}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in model_rows
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            summary.update(
                {
                    f"{metric}_median": median(values) if values else None,
                    f"{metric}_min": min(values) if values else None,
                    f"{metric}_max": max(values) if values else None,
                }
            )
        summaries.append(summary)
    return summaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_benchmark_artifacts(
    output_dir: str | Path,
    *,
    rows: list[dict[str, object]],
    environment: dict[str, object],
    config: dict[str, object],
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "resource_benchmark_raw.csv"
    summary_path = output / "resource_benchmark_summary.csv"
    environment_path = output / "resource_benchmark_environment.json"
    config_path = output / "resource_benchmark_config.json"
    write_csv(raw_path, rows)
    write_csv(summary_path, _summarize(rows))
    _write_json(environment_path, environment)
    _write_json(config_path, config)

    repetitions = int(config["repetitions"])
    counts = {model: sum(row["model"] == model for row in rows) for model in EXPECTED_MODELS}
    process_memory_ok = all(
        row.get("peak_process_memory_status") == "measured"
        and int(row.get("peak_process_memory_bytes") or 0) > 0
        for row in rows
    )
    gpu_status_ok = all(
        row.get("peak_gpu_memory_status")
        == ("measured" if str(row.get("device", "")).startswith("cuda") else "not_applicable")
        for row in rows
    )
    paths = (raw_path, summary_path, environment_path, config_path)
    audit = {
        "status": (
            "passed"
            if counts == {model: repetitions for model in EXPECTED_MODELS}
            and process_memory_ok
            and gpu_status_ok
            else "failed"
        ),
        "coverage": {
            "models": counts,
            "repetitions": repetitions,
            "expected_rows": repetitions * len(EXPECTED_MODELS),
            "observed_rows": len(rows),
        },
        "process_memory_status": "passed" if process_memory_ok else "failed",
        "gpu_memory_status": "passed" if gpu_status_ok else "failed",
        "files": [
            {"relative_path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths
        ],
    }
    _write_json(output / "resource_benchmark_audit.json", audit)
    return audit


def _coerce_like(current: object, value: object) -> object:
    if is_dataclass(current) and isinstance(value, Mapping):
        return replace(
            current,
            **{
                field.name: _coerce_like(getattr(current, field.name), value[field.name])
                for field in fields(current)
                if field.name in value
            },
        )
    if isinstance(current, tuple) and isinstance(value, list):
        if current and isinstance(current[0], tuple):
            return tuple(tuple(item) for item in value)
        return tuple(value)
    return value


def load_source_config(source_run: str | Path) -> LearningCurveConfig:
    payload = json.loads((Path(source_run) / "config.json").read_text(encoding="utf-8"))
    config = build_default_learning_curve_config()
    return _coerce_like(config, payload)  # type: ignore[return-value]


def _piece_ids(path: Path, *, fraction: float | None = None) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fraction is None:
        return list(payload["piece_ids"])
    entry = next(
        item for item in payload["fractions"] if math.isclose(float(item["frac"]), fraction)
    )
    return list(entry["piece_ids"])


def _nvidia_temperature() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _run_family(
    model_name: str,
    *,
    config: LearningCurveConfig,
    selected: dict[str, object],
    train_pieces: list,
    validation_pieces: list,
    test_pieces: list,
) -> dict[str, object]:
    tokenizer = build_tokenizer(config.experiment.representation)
    bos_token_id = tokenizer.bos_token_id
    max_context_length = config.experiment.windows.max_context_length
    model_seed = selected["model_seed"]
    hyperparameters = selected["hyperparameters"]
    use_cuda = model_name == "transformer" and torch.cuda.is_available()
    temperature_start = _nvidia_temperature() if use_cuda else None

    if model_name == "vomm":
        model = VariableOrderMarkovModel(
            max_order=int(hyperparameters["selected_order"]),
            vocabulary_size=bos_token_id + 1,
            bos_token_id=bos_token_id,
        )
        fit_args = ([list(piece.tokens) for piece in train_pieces],)
        eval_args = (test_pieces, max_context_length)
        fit_kwargs = eval_kwargs = {}
    elif model_name == "finite_hmm":
        model = FiniteGlobalHMM(
            candidate_num_states=(int(hyperparameters["selected_states"]),),
            max_iterations=config.finite_hmm_max_iterations,
            tolerance=config.finite_hmm_tolerance,
            seed=int(model_seed),
            vocab_size=bos_token_id + 1,
        )
        fit_args = (train_pieces, validation_pieces)
        eval_args = (test_pieces,)
        fit_kwargs = eval_kwargs = {
            "bos_token_id": bos_token_id,
            "max_context_length": max_context_length,
        }
    elif model_name == "hdp_hmm":
        model = GlobalHDPHMM(
            truncation_level=config.hdp_truncation_level,
            n_iters=config.hdp_n_iters,
            burn_in=config.hdp_burn_in,
            hyperparameter_grid=(
                (
                    float(hyperparameters["alpha"]),
                    float(hyperparameters["alpha0"]),
                    float(hyperparameters["gamma"]),
                ),
            ),
            seed=int(model_seed),
            vocab_size=bos_token_id + 1,
        )
        fit_args = (train_pieces, validation_pieces)
        eval_args = (test_pieces,)
        fit_kwargs = eval_kwargs = {
            "bos_token_id": bos_token_id,
            "max_context_length": max_context_length,
        }
    elif model_name == "transformer":
        model = _build_transformer_model(config, int(model_seed))
        dataloaders = _build_transformer_dataloaders(
            config,
            train_pieces=train_pieces,
            validation_pieces=validation_pieces,
            test_pieces=test_pieces,
        )
        fit_args = (dataloaders["train"], dataloaders["validation"])
        eval_args = (dataloaders["test"],)
        fit_kwargs = eval_kwargs = {}
    else:
        raise ValueError(f"unknown model {model_name}")

    with ResourceMonitor(use_cuda=use_cuda) as monitor:
        fit_start = time.perf_counter()
        fit_result = model.fit(*fit_args, **fit_kwargs)
        fit_seconds = time.perf_counter() - fit_start
        evaluation_start = time.perf_counter()
        evaluation = model.evaluate(*eval_args, **eval_kwargs)
        evaluation_seconds = time.perf_counter() - evaluation_start
    resources = monitor.measurement()
    if model_name == "transformer":
        summary = _transformer_summary_row(fit_result, evaluation)
        test_nll = summary["test_nll_per_token"]
        device = summary["device"]
    else:
        summary = evaluation["summary"]
        test_nll = summary["test_nll_per_token"]
        device = "cpu"
    return {
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "test_nll": test_nll,
        "device": device,
        "gpu_temperature_c_start": temperature_start,
        "gpu_temperature_c_end": _nvidia_temperature() if use_cuda else None,
        **resources,
    }


def run_resource_benchmark(
    *,
    source_run: str | Path,
    fraction: float,
    split_seed: int,
    repetitions: int,
    output_dir: str | Path,
    corpus_root: str | Path | None = None,
    corpus_cache: str | Path | None = None,
    n_workers: int = 6,
) -> dict[str, object]:
    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"benchmark output directory is not empty: {output}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    source_audit = audit_run(source)
    if source_audit["status"] != "passed":
        raise ValueError(f"source run audit is {source_audit['status']}, not passed")

    config = load_source_config(source)
    if config.split_seed != split_seed:
        raise ValueError(f"source split seed is {config.split_seed}, requested {split_seed}")
    repo_root = source.parents[2]
    resolved_corpus_root = Path(corpus_root) if corpus_root else repo_root / config.experiment.corpus.root_dir
    resolved_cache = Path(corpus_cache) if corpus_cache else repo_root / "artifacts" / "corpus_cache_3000.jsonl"
    cache_hash = _sha256(resolved_cache)
    if cache_hash != EXPECTED_CACHE_SHA256:
        raise ValueError(f"unexpected corpus cache SHA-256: {cache_hash}")
    config = replace(
        config,
        experiment=replace(
            config.experiment,
            corpus=replace(config.experiment.corpus, root_dir=str(resolved_corpus_root)),
        ),
    )
    preparation = prepare_corpus(
        config.experiment,
        max_files=3000,
        n_workers=n_workers,
        cache_path=resolved_cache,
        sample_seed=7,
    )
    by_id = {piece.piece_id: piece for piece in preparation.pieces}
    split_root = source / "splits"
    selected = select_source_configurations(source, fraction=fraction)
    train_ids = _piece_ids(
        split_root / f"train_fractions_seed{selected['finite_hmm']['data_seed']}.json",
        fraction=fraction,
    )
    validation_ids = _piece_ids(split_root / "val_pieces.json")
    test_ids = _piece_ids(split_root / "test_pieces.json")
    missing = sorted((set(train_ids) | set(validation_ids) | set(test_ids)) - set(by_id))
    if missing:
        raise ValueError(f"prepared corpus is missing {len(missing)} source split pieces")
    train_pieces = [by_id[piece_id] for piece_id in train_ids]
    validation_pieces = [by_id[piece_id] for piece_id in validation_ids]
    test_pieces = [by_id[piece_id] for piece_id in test_ids]

    rows = []
    for model_name in EXPECTED_MODELS:
        for repetition in range(1, repetitions + 1):
            row = _run_family(
                model_name,
                config=config,
                selected=selected[model_name],
                train_pieces=train_pieces,
                validation_pieces=validation_pieces,
                test_pieces=test_pieces,
            )
            rows.append(
                {
                    "model": model_name,
                    "repetition": repetition,
                    "data_fraction": fraction,
                    "split_seed": split_seed,
                    "model_seed": selected[model_name]["model_seed"],
                    **row,
                }
            )
            print(f"[resource-benchmark] {model_name} repetition {repetition}/{repetitions}", flush=True)

    hardware = _build_hardware_manifest(
        target_device=config.experiment.hardware.target_device,
        precision=config.experiment.hardware.precision,
    )
    environment = {
        **hardware,
        "platform": platform.platform(),
        "psutil_version": psutil.__version__,
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": os.cpu_count(),
    }
    benchmark_config = {
        "source_run": source.name,
        "source_audit_status": source_audit["status"],
        "fraction": fraction,
        "split_seed": split_seed,
        "repetitions": repetitions,
        "corpus_sample_seed": 7,
        "max_files": 3000,
        "n_workers": n_workers,
        "corpus_cache_sha256": cache_hash,
        "prepared_pieces": len(preparation.pieces),
        "exclusions": len(preparation.exclusions),
        "configurations": selected,
    }
    return write_benchmark_artifacts(
        output,
        rows=rows,
        environment=environment,
        config=benchmark_config,
    )
