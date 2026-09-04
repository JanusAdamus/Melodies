"""Benchmark repetido de los modelos seleccionados y auditoría de sus recursos.

El módulo no reabre la comparación de configuraciones. Toma las decisiones de
una corrida auditada, repite el ajuste y la evaluación con la misma partición y
registra tiempo, RAM del árbol de procesos y, cuando corresponde, memoria CUDA.
"""

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
from typing import Mapping


EXPECTED_MODELS = ("finite_hmm", "hdp_hmm", "transformer", "vomm")
EXPECTED_CORPUS_COUNTS = {
    "entries": 3000,
    "prepared_pieces": 2933,
    "exclusions": 67,
    "events": 693754,
    "tokens": 693754,
}
_SUMMARY_METRICS = (
    "selection_wall_clock_s",
    "evaluation_wall_clock_s",
    "total_protocol_wall_clock_s",
    "selection_peak_process_tree_rss_bytes",
    "evaluation_peak_process_tree_rss_bytes",
    "selection_peak_cuda_allocated_bytes",
    "selection_peak_cuda_reserved_bytes",
    "evaluation_peak_cuda_allocated_bytes",
    "evaluation_peak_cuda_reserved_bytes",
    "test_nll",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _seed_key(value: object) -> tuple[int, object]:
    try:
        return 0, int(str(value))
    except ValueError:
        return 1, str(value)


def select_source_configurations(
    source_run: str | Path, *, fraction: float
) -> dict[str, dict[str, object]]:
    """Recupera una configuración ya seleccionada por familia.

    Si existen varias semillas, se escoge la primera de forma determinista. El
    benchmark mide recursos; no usa el test para volver a elegir modelos.
    """

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
            key=lambda item: (
                _seed_key(item.get("data_seed")),
                _seed_key(item.get("model_seed")),
            ),
        )
        model_seed: object = row.get("model_seed", "deterministic")
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
    summaries: list[dict[str, object]] = []
    for model in EXPECTED_MODELS:
        model_rows = [row for row in rows if row.get("model") == model]
        summary: dict[str, object] = {"model": model, "repetitions": len(model_rows)}
        for metric in _SUMMARY_METRICS:
            values: list[float] = []
            for row in model_rows:
                value = row.get(metric)
                if value in {None, ""}:
                    continue
                number = float(value)
                if math.isfinite(number):
                    values.append(number)
            summary.update(
                {
                    f"{metric}_median": median(values) if values else None,
                    f"{metric}_min": min(values) if values else None,
                    f"{metric}_max": max(values) if values else None,
                }
            )
        summaries.append(summary)
    return summaries


def _resource_rows_are_complete(rows: list[dict[str, object]]) -> bool:
    for row in rows:
        if row.get("resource_measurement_condition") != "isolated":
            return False
        if row.get("resource_cost_usable") not in {True, 1, "true", "True", "1"}:
            return False
        device = str(row.get("device", "")).lower()
        if row.get("model") == "transformer" and not device.startswith("cuda"):
            return False
        for phase in ("selection", "evaluation"):
            if row.get(f"{phase}_resource_status") != "measured":
                return False
            if int(row.get(f"{phase}_peak_process_tree_rss_bytes") or 0) <= 0:
                return False
            if device.startswith("cuda"):
                if row.get(f"{phase}_cuda_resource_status") != "measured":
                    return False
                if int(row.get(f"{phase}_peak_cuda_allocated_bytes") or 0) <= 0:
                    return False
    return True


def write_benchmark_artifacts(
    output_dir: str | Path,
    *,
    rows: list[dict[str, object]],
    environment: dict[str, object],
    config: dict[str, object],
    corpus_fingerprint: dict[str, object],
    expected_corpus_fingerprint: str,
) -> dict[str, object]:
    """Escribe los resultados y un dictamen verificable del benchmark."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "resource_benchmark_raw.csv"
    summary_path = output / "resource_benchmark_summary.csv"
    environment_path = output / "resource_benchmark_environment.json"
    config_path = output / "resource_benchmark_config.json"
    _write_csv(raw_path, rows)
    _write_csv(summary_path, _summarize(rows))
    _write_json(environment_path, environment)
    _write_json(config_path, config)

    repetitions = int(config["repetitions"])
    expected_keys = {
        (model, repetition)
        for model in EXPECTED_MODELS
        for repetition in range(1, repetitions + 1)
    }
    try:
        observed_keys = [(str(row["model"]), int(row["repetition"])) for row in rows]
    except (KeyError, TypeError, ValueError):
        observed_keys = []
    coverage_ok = (
        len(rows) == len(expected_keys)
        and len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
    )
    observed_fingerprint = str(corpus_fingerprint.get("sha256", "")).upper()
    expected_fingerprint = expected_corpus_fingerprint.upper()
    fingerprint_ok = (
        corpus_fingerprint.get("algorithm") == "sha256-canonical-corpus-v1"
        and observed_fingerprint == expected_fingerprint
    )
    fingerprint_counts = corpus_fingerprint.get("counts", {})
    counts_ok = isinstance(fingerprint_counts, Mapping) and all(
        fingerprint_counts.get(key) == expected
        for key, expected in EXPECTED_CORPUS_COUNTS.items()
    )
    resource_ok = _resource_rows_are_complete(rows)
    paths = (raw_path, summary_path, environment_path, config_path)
    counts = {model: sum(row.get("model") == model for row in rows) for model in EXPECTED_MODELS}
    audit = {
        "schema_version": 1,
        "status": (
            "passed"
            if coverage_ok and fingerprint_ok and counts_ok and resource_ok
            else "failed"
        ),
        "coverage": {
            "models": counts,
            "repetitions": repetitions,
            "expected_rows": len(expected_keys),
            "observed_rows": len(rows),
        },
        "coverage_status": "passed" if coverage_ok else "failed",
        "resource_status": "passed" if resource_ok else "failed",
        "corpus_fingerprint_status": "passed" if fingerprint_ok else "failed",
        "corpus_counts_status": "passed" if counts_ok else "failed",
        "corpus_fingerprint": corpus_fingerprint,
        "expected_corpus_fingerprint": expected_fingerprint,
        "files": [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
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


def load_source_config(source_run: str | Path):
    from .config import build_default_learning_curve_config

    payload = json.loads((Path(source_run) / "config.json").read_text(encoding="utf-8"))
    return _coerce_like(build_default_learning_curve_config(), payload)


def _piece_ids(path: Path, *, fraction: float | None = None) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if fraction is None:
        return list(payload["piece_ids"])
    entry = next(
        item for item in payload["fractions"] if math.isclose(float(item["frac"]), fraction)
    )
    return list(entry["piece_ids"])


def _load_cached_preparation(cache_path: Path, representation_config):
    """Reconstruye el corpus tokenizado sin depender de sus rutas originales."""

    from next_token_experiment.data.tokenizer import build_tokenizer
    from next_token_experiment.schemas import (
        CorpusPreparationResult,
        ExclusionRecord,
        PreparedPiece,
    )

    vocabulary = list(build_tokenizer(representation_config).musical_vocabulary)
    pieces = []
    exclusions = []
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            kind = payload.pop("kind", None)
            if kind == "piece":
                pieces.append(PreparedPiece(vocabulary=vocabulary, **payload))
            elif kind == "exclusion":
                exclusions.append(ExclusionRecord(**payload))
            else:
                raise ValueError(f"cache line {line_number} has invalid kind: {kind!r}")
    return CorpusPreparationResult(
        pieces=sorted(pieces, key=lambda item: item.piece_id),
        exclusions=sorted(exclusions, key=lambda item: item.piece_id),
    )


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


def _observation_fields(prefix: str, observation) -> dict[str, object]:
    return {
        f"{prefix}_wall_clock_s": observation.wall_clock_s,
        f"{prefix}_peak_process_rss_bytes": observation.peak_process_rss_bytes,
        f"{prefix}_peak_process_tree_rss_bytes": observation.peak_process_tree_rss_bytes,
        f"{prefix}_peak_cuda_allocated_bytes": observation.peak_cuda_allocated_bytes,
        f"{prefix}_peak_cuda_reserved_bytes": observation.peak_cuda_reserved_bytes,
        f"{prefix}_resource_status": observation.process_memory_status,
        f"{prefix}_cuda_resource_status": observation.cuda_memory_status,
    }


def _run_family(
    model_name: str,
    *,
    config,
    selected: dict[str, object],
    train_pieces: list,
    validation_pieces: list,
    test_pieces: list,
    measurement_condition: str,
) -> dict[str, object]:
    import torch

    from next_token_experiment.data.tokenizer import build_tokenizer

    from .classical_models import FiniteGlobalHMM, GlobalHDPHMM
    from .resource_monitor import ResourceMonitor
    from .runner import (
        _build_transformer_dataloaders,
        _build_transformer_model,
        _transformer_summary_row,
    )
    from .vomm import VariableOrderMarkovModel

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
        fit_protocol = "source_selected_configuration"
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
        fit_protocol = "source_selected_configuration"
    elif model_name == "hdp_hmm":
        model = GlobalHDPHMM(
            truncation_level=config.hdp_truncation_level,
            n_iters=config.hdp_n_iters,
            burn_in=config.hdp_burn_in,
            hyperparameter_grid=((
                float(hyperparameters["alpha"]),
                float(hyperparameters["alpha0"]),
                float(hyperparameters["gamma"]),
            ),),
            seed=int(model_seed),
            vocab_size=bos_token_id + 1,
        )
        fit_args = (train_pieces, validation_pieces)
        eval_args = (test_pieces,)
        fit_kwargs = eval_kwargs = {
            "bos_token_id": bos_token_id,
            "max_context_length": max_context_length,
        }
        fit_protocol = "source_selected_configuration"
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
        fit_protocol = "fixed_architecture_with_early_stopping"
    else:
        raise ValueError(f"unknown model {model_name}")

    selection_monitor = ResourceMonitor(measure_cuda=use_cuda)
    with selection_monitor:
        fit_result = model.fit(*fit_args, **fit_kwargs)
    evaluation_monitor = ResourceMonitor(measure_cuda=use_cuda)
    with evaluation_monitor:
        evaluation = model.evaluate(*eval_args, **eval_kwargs)
    if selection_monitor.result is None or evaluation_monitor.result is None:
        raise RuntimeError("resource monitor produced no observation")

    if model_name == "transformer":
        summary = _transformer_summary_row(fit_result, evaluation)
        test_nll = summary["test_nll_per_token"]
        device = summary["device"]
        training = {
            "best_epoch": summary.get("best_epoch"),
            "epochs_completed": summary.get("epochs_completed"),
            "early_stopped": summary.get("early_stopped"),
        }
    else:
        test_nll = evaluation["summary"]["test_nll_per_token"]
        device = "cpu"
        training = {}
    return {
        **_observation_fields("selection", selection_monitor.result),
        **_observation_fields("evaluation", evaluation_monitor.result),
        "total_protocol_wall_clock_s": (
            selection_monitor.result.wall_clock_s + evaluation_monitor.result.wall_clock_s
        ),
        "test_nll": test_nll,
        "device": device,
        "fit_protocol": fit_protocol,
        "resource_measurement_condition": measurement_condition,
        "resource_cost_usable": measurement_condition == "isolated",
        "resource_sample_interval_s": max(
            selection_monitor.result.sample_interval_s,
            evaluation_monitor.result.sample_interval_s,
        ),
        "gpu_temperature_c_start": temperature_start,
        "gpu_temperature_c_end": _nvidia_temperature() if use_cuda else None,
        **training,
    }


def run_resource_benchmark(
    *,
    source_run: str | Path,
    fraction: float,
    split_seed: int,
    repetitions: int,
    output_dir: str | Path,
    expected_corpus_fingerprint: str,
    corpus_cache: str | Path | None = None,
    measurement_condition: str = "isolated",
) -> dict[str, object]:
    """Ejecuta las cuatro familias en serie y conserva evidencia auditable."""

    import psutil

    from .artifact_audit import audit_run
    from .corpus_fingerprint import fingerprint_corpus_cache
    from .runner import _build_hardware_manifest

    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"benchmark output directory is not empty: {output}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if measurement_condition not in {"isolated", "contended", "unknown"}:
        raise ValueError("invalid measurement condition")
    source_audit = audit_run(source)
    if source_audit["status"] != "passed":
        raise ValueError(f"source run audit is {source_audit['status']}, not passed")

    config = load_source_config(source)
    if config.split_seed != split_seed:
        raise ValueError(f"source split seed is {config.split_seed}, requested {split_seed}")
    repository_root = Path(__file__).resolve().parents[1]
    resolved_cache = (
        Path(corpus_cache).resolve()
        if corpus_cache
        else repository_root / "artifacts" / "corpus_cache_3000.jsonl"
    )
    corpus_fingerprint = fingerprint_corpus_cache(resolved_cache)
    if str(corpus_fingerprint["sha256"]).upper() != expected_corpus_fingerprint.upper():
        raise ValueError("the canonical corpus fingerprint does not match the expected value")
    mismatches = {
        key: {"expected": value, "observed": corpus_fingerprint["counts"].get(key)}
        for key, value in EXPECTED_CORPUS_COUNTS.items()
        if corpus_fingerprint["counts"].get(key) != value
    }
    if mismatches:
        raise ValueError(f"unexpected cached corpus counts: {mismatches}")

    preparation = _load_cached_preparation(resolved_cache, config.experiment.representation)
    prepared_counts = {
        "prepared_pieces": len(preparation.pieces),
        "exclusions": len(preparation.exclusions),
        "events": sum(piece.n_events for piece in preparation.pieces),
        "tokens": sum(len(piece.tokens) for piece in preparation.pieces),
    }
    prepared_mismatches = {
        key: {"expected": EXPECTED_CORPUS_COUNTS[key], "observed": value}
        for key, value in prepared_counts.items()
        if value != EXPECTED_CORPUS_COUNTS[key]
    }
    if prepared_mismatches:
        raise ValueError(f"unexpected prepared corpus counts: {prepared_mismatches}")

    by_id = {piece.piece_id: piece for piece in preparation.pieces}
    split_root = source / "splits"
    selected = select_source_configurations(source, fraction=fraction)
    selected_architecture = selected["transformer"]["hyperparameters"].get("architecture")
    configured_architecture = config.experiment.transformer.architecture
    if selected_architecture != configured_architecture:
        raise ValueError(
            "source transformer architecture does not match its saved configuration: "
            f"{selected_architecture!r} != {configured_architecture!r}"
        )
    data_seeds = {entry["data_seed"] for entry in selected.values()}
    if len(data_seeds) != 1:
        raise ValueError(f"selected source rows use different data seeds: {data_seeds}")
    data_seed = int(next(iter(data_seeds)))
    train_ids = _piece_ids(
        split_root / f"train_fractions_seed{data_seed}.json", fraction=fraction
    )
    validation_ids = _piece_ids(split_root / "val_pieces.json")
    test_ids = _piece_ids(split_root / "test_pieces.json")
    missing = sorted((set(train_ids) | set(validation_ids) | set(test_ids)) - set(by_id))
    if missing:
        raise ValueError(f"prepared corpus is missing {len(missing)} source split pieces")
    train_pieces = [by_id[piece_id] for piece_id in train_ids]
    validation_pieces = [by_id[piece_id] for piece_id in validation_ids]
    test_pieces = [by_id[piece_id] for piece_id in test_ids]

    rows: list[dict[str, object]] = []
    for model_name in EXPECTED_MODELS:
        for repetition in range(1, repetitions + 1):
            measurement = _run_family(
                model_name,
                config=config,
                selected=selected[model_name],
                train_pieces=train_pieces,
                validation_pieces=validation_pieces,
                test_pieces=test_pieces,
                measurement_condition=measurement_condition,
            )
            rows.append(
                {
                    "model": model_name,
                    "repetition": repetition,
                    "frac": fraction,
                    "split_seed": split_seed,
                    "data_seed": data_seed,
                    "model_seed": selected[model_name]["model_seed"],
                    **measurement,
                }
            )
            print(
                f"[resource-benchmark] {model_name} repetition {repetition}/{repetitions}",
                flush=True,
            )

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
        "data_seed": data_seed,
        "repetitions": repetitions,
        "measurement_condition": measurement_condition,
        "corpus_sample_seed": 7,
        "max_files": 3000,
        "prepared_counts": prepared_counts,
        "configurations": selected,
        "transformer_fit_protocol": "fixed_architecture_with_early_stopping",
    }
    return write_benchmark_artifacts(
        output,
        rows=rows,
        environment=environment,
        config=benchmark_config,
        corpus_fingerprint=corpus_fingerprint,
        expected_corpus_fingerprint=expected_corpus_fingerprint,
    )
