from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
import json
import math
from numbers import Integral
from pathlib import Path
from statistics import mean, pstdev
import sys
import time

import matplotlib

matplotlib.use("Agg")  # El runner solo escribe PNG; sin esto Tk falla en headless.

import matplotlib.pyplot as plt
import pandas as pd

from next_token_experiment.data.dataset import WindowedSequenceDataset, build_dataloaders
from next_token_experiment.data.preprocess import build_preprocessing_report, prepare_corpus
from next_token_experiment.data.tokenizer import build_tokenizer
from next_token_experiment.experiment.storage import ensure_directory, write_csv
from next_token_experiment.models.small_transformer import (
    SmallTransformerNextTokenModel,
    SmallTransformerStudySpec,
)
from next_token_experiment.schemas import PreparedPiece

from .classical_models import FiniteGlobalHMM, GlobalHDPHMM
from .config import LearningCurveConfig
from .decision import pareto_front
from .splits import FixedSplits, build_fixed_splits, build_nested_training_subsets
from .statistics import pairwise_model_comparisons
from .structural_metrics import adjusted_rand_index, boundary_f1, normalized_mutual_information
from .vomm import select_vomm_by_validation


REQUIRED_STRUCTURAL_COLUMNS = ("piece_id", "event_index", "segment_label", "boundary")

# VOMM selection is deterministic in the stochastic model seed: it depends only on the
# training subset (data_seed, fraction) and validation set. It is therefore scored once
# per (data_seed, fraction) and labelled with this sentinel instead of being recomputed
# per model_seed, which would only fabricate identical duplicate rows.
_VOMM_DETERMINISTIC_SEED = "deterministic"


def _guard_fresh_run_directory(run_directory: Path) -> None:
    """Refuse to write into an existing, non-empty run directory.

    Reusing a run name would interleave new artifacts with stale ones from a previous
    run, so fail before writing anything and ask for a different name.
    """

    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"Run directory {run_directory} already exists and is not empty; "
            "choose a different run_name to avoid mixing artifacts from separate runs."
        )


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


_CHECKPOINT_ROW_KEYS = (
    "raw_rows",
    "piece_metric_rows",
    "protocol_evidence",
    "structural_predictions",
)


def _load_checkpoint(path: Path) -> tuple[set[str], dict[str, list[dict[str, object]]]]:
    """Rehidrata las filas que dejo una corrida interrumpida.

    La unidad es una celda (data_seed, fraction): al reanudar se saltan las celdas ya
    completas en vez de reajustar sus modelos. Una linea truncada por una muerte a mitad
    de escritura se descarta, y esa celda se recalcula.
    """

    completed: set[str] = set()
    rows: dict[str, list[dict[str, object]]] = {key: [] for key in _CHECKPOINT_ROW_KEYS}
    if not path.exists():
        return completed, rows

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            print(
                f"[runner] checkpoint: descarto la linea {number}, incompleta; "
                "esa celda se recalcula",
                file=sys.stderr,
                flush=True,
            )
            continue
        completed.add(str(payload["cell"]))
        for key in _CHECKPOINT_ROW_KEYS:
            rows[key].extend(payload.get(key, []))
    return completed, rows


def _append_checkpoint(path: Path, cell: str, rows: Mapping[str, list[dict[str, object]]]) -> None:
    """Persiste una celda terminada. Append, nunca reescritura del archivo entero."""

    payload = {"cell": cell, **{key: _json_safe(rows[key]) for key in _CHECKPOINT_ROW_KEYS}}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    safe_payload = _json_safe(payload)
    target.write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _build_transformer_model(
    config: LearningCurveConfig,
    model_seed: int,
    *,
    piece_metadata_by_id: dict[str, dict[str, object]] | None = None,
) -> SmallTransformerNextTokenModel:
    tokenizer = build_tokenizer(config.experiment.representation)
    transformer = config.experiment.transformer
    spec = SmallTransformerStudySpec(
        architecture=transformer.architecture,
        n_layers=transformer.n_layers,
        d_model=transformer.d_model,
        n_heads=transformer.n_heads,
        ff_dim=transformer.ff_dim,
        dropout=transformer.dropout,
        learning_rate=transformer.learning_rate,
        weight_decay=transformer.weight_decay,
        batch_size=transformer.batch_size,
        max_epochs=transformer.max_epochs,
        early_stopping_patience=transformer.early_stopping_patience,
        gradient_accumulation_steps=transformer.gradient_accumulation_steps,
        grad_clip_norm=transformer.grad_clip_norm,
        label_smoothing=transformer.label_smoothing,
        lr_scheduler_factor=transformer.lr_scheduler_factor,
        lr_scheduler_patience=transformer.lr_scheduler_patience,
        min_learning_rate=transformer.min_learning_rate,
        tie_input_output_embeddings=transformer.tie_input_output_embeddings,
        use_relative_position_bias=transformer.use_relative_position_bias,
        relative_attention_num_buckets=transformer.relative_attention_num_buckets,
        relative_attention_max_distance=transformer.relative_attention_max_distance,
        generation_num_prompts=1,
        generation_prompt_length=8,
        generation_max_new_tokens=8,
        generation_temperature=1.0,
        generation_top_k=4,
    )
    return SmallTransformerNextTokenModel(
        spec=spec,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_context_length=config.experiment.windows.max_context_length,
        hardware=config.experiment.hardware,
        seed=model_seed,
        piece_metadata_by_id=piece_metadata_by_id,
    )


def _build_transformer_dataloaders(
    config: LearningCurveConfig,
    train_pieces,
    validation_pieces,
    test_pieces,
):
    tokenizer = build_tokenizer(config.experiment.representation)
    train_dataset = WindowedSequenceDataset(
        pieces=train_pieces,
        tokenizer=tokenizer,
        split="train",
        max_context_length=config.experiment.windows.max_context_length,
        stride=config.experiment.windows.train_stride,
        min_window_length=config.experiment.windows.min_window_length,
    )
    validation_dataset = WindowedSequenceDataset(
        pieces=validation_pieces,
        tokenizer=tokenizer,
        split="validation",
        max_context_length=config.experiment.windows.max_context_length,
        stride=config.experiment.windows.eval_stride,
        min_window_length=config.experiment.windows.min_window_length,
    )
    test_dataset = WindowedSequenceDataset(
        pieces=test_pieces,
        tokenizer=tokenizer,
        split="test",
        max_context_length=config.experiment.windows.max_context_length,
        stride=config.experiment.windows.eval_stride,
        min_window_length=config.experiment.windows.min_window_length,
    )
    return build_dataloaders(
        tokenizer=tokenizer,
        config=config.experiment,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
    )


def _transformer_summary_row(fit_result: dict, eval_result: dict) -> dict[str, object]:
    summary = eval_result["summary"]
    fit_summary = fit_result["summary"]
    return {
        "validation_nll_per_token": fit_summary["best_validation_nll"],
        "test_nll_per_token": summary["nll_per_token"],
        "test_perplexity": summary["perplexity"],
        "test_accuracy": summary.get("accuracy"),
        "test_brier_score": summary.get("brier_score"),
        "n_params": summary["parameter_count"],
        "train_time_sec": sum(
            row["train_wall_clock_s"] + row["validation_wall_clock_s"]
            for row in fit_result["train_log"]
        ),
        "evaluation_wall_clock_s": summary["eval_wall_clock_s"],
        "device": summary.get("runtime", {}).get("device", "unknown"),
    }


def _require_canonical_work_ids(pieces: list[PreparedPiece]) -> None:
    missing = [
        piece.piece_id
        for piece in pieces
        if not isinstance(piece.canonical_work_id, str) or not piece.canonical_work_id.strip()
    ]
    if missing:
        raise ValueError(f"Prepared pieces require nonempty canonical_work_id values: {missing}")


def _build_grouped_fixed_splits(
    pieces: list[PreparedPiece],
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> FixedSplits:
    _require_canonical_work_ids(pieces)
    groups: dict[str, list[PreparedPiece]] = {}
    for piece in sorted(pieces, key=lambda item: item.piece_id):
        groups.setdefault(piece.canonical_work_id, []).append(piece)
    representatives = [members[0] for _, members in sorted(groups.items())]
    representative_splits = build_fixed_splits(
        representatives,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    def expand(selected: list[PreparedPiece]) -> list[PreparedPiece]:
        selected_groups = {piece.canonical_work_id for piece in selected}
        return sorted(
            [piece for group_id in selected_groups for piece in groups[group_id]],
            key=lambda item: item.piece_id,
        )

    fixed_splits = FixedSplits(
        test_pieces=expand(representative_splits.test_pieces),
        validation_pieces=expand(representative_splits.validation_pieces),
        train_pool_pieces=expand(representative_splits.train_pool_pieces),
    )
    group_sets = [
        {piece.canonical_work_id for piece in fixed_splits.train_pool_pieces},
        {piece.canonical_work_id for piece in fixed_splits.validation_pieces},
        {piece.canonical_work_id for piece in fixed_splits.test_pieces},
    ]
    if any(left & right for index, left in enumerate(group_sets) for right in group_sets[index + 1 :]):
        raise ValueError("Canonical work groups must not cross split boundaries.")
    return fixed_splits


def _split_payload(pieces: list[PreparedPiece]) -> dict[str, object]:
    canonical_work_ids = sorted({piece.canonical_work_id for piece in pieces})
    return {
        "n_pieces": len(pieces),
        "piece_ids": [piece.piece_id for piece in pieces],
        "n_canonical_groups": len(canonical_work_ids),
        "canonical_work_ids": canonical_work_ids,
        "n_tokens": sum(len(piece.tokens) for piece in pieces),
    }


def _coverage_plan(pieces: list[PreparedPiece]) -> list[dict[str, object]]:
    return [
        {
            "piece_id": piece.piece_id,
            "canonical_work_id": piece.canonical_work_id,
            "n_events": len(piece.tokens),
            "expected_event_indices": list(range(len(piece.tokens))),
        }
        for piece in pieces
    ]


def _build_execution_plan(
    config: LearningCurveConfig,
    fixed_splits: FixedSplits,
    nested_by_seed: dict[int, list[tuple[float, list[PreparedPiece]]]],
) -> dict[str, object]:
    tokenizer = build_tokenizer(config.experiment.representation)
    run_rows = sum(len(subsets) for subsets in nested_by_seed.values()) * len(config.model_seeds)
    fits_by_family: dict[str, dict[str, object]] = {
        "finite_hmm": {
            "run_rows": run_rows,
            "candidate_fits": run_rows * len(config.finite_hmm_states),
            "workload_class": "lightweight_or_classical",
        },
        "hdp_hmm": {
            "run_rows": run_rows,
            "candidate_fits": run_rows * len(config.hdp_hyperparameter_grid),
            "workload_class": "lightweight_or_classical",
        },
        "transformer": {
            "run_rows": run_rows,
            "candidate_fits": run_rows,
            "workload_class": "neural",
        },
    }
    if config.include_vomm_control:
        fits_by_family["vomm"] = {
            "run_rows": run_rows,
            "candidate_fits": run_rows * len(config.vomm_candidate_orders),
            "workload_class": "lightweight_or_classical",
        }

    nested_payload = []
    for data_seed, subsets in sorted(nested_by_seed.items()):
        nested_payload.append(
            {
                "data_seed": data_seed,
                "fractions": [
                    {"frac": fraction, **_split_payload(pieces)}
                    for fraction, pieces in subsets
                ],
            }
        )

    artifact_names = (
        "config.json",
        "preprocessing_report.json",
        "exclusions.csv",
        "splits/*.json",
        "execution_plan.json",
        "results_raw.csv",
        "results_summary.csv",
        "piece_metrics_raw.csv",
        "pairwise_comparisons.json",
        "engineering_costs.csv",
        "protocol_audit.json",
        "structural_evaluation.json",
        "pareto_summary.json",
        "learning_curve.png",
    )
    return {
        "status": "planned_no_evidence",
        "claims_evidence": False,
        "mode": "plan_only",
        "splits": {
            "train_pool": _split_payload(fixed_splits.train_pool_pieces),
            "validation": _split_payload(fixed_splits.validation_pieces),
            "test": _split_payload(fixed_splits.test_pieces),
        },
        "nested_training_subsets": nested_payload,
        "fits_by_family": fits_by_family,
        "seeds": {
            "split_seed": config.split_seed,
            "data_seeds": list(config.data_seeds),
            "model_seeds": list(config.model_seeds),
        },
        "context_and_reset_policy": {
            "max_context_length": config.experiment.windows.max_context_length,
            "evaluation_slices": "non_overlapping_with_exact_positive_tail",
            "reset": "BOS at every validation/test slice boundary",
            "bos_is_target": False,
        },
        "common_predictive_support": {
            "musical_token_ids": list(range(tokenizer.musical_vocab_size)),
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "pad_in_scoring_support": False,
            "support_size": tokenizer.bos_token_id + 1,
        },
        "exact_expected_coverage": {
            "validation": _coverage_plan(fixed_splits.validation_pieces),
            "test": _coverage_plan(fixed_splits.test_pieces),
        },
        "workload_distinction": {
            "lightweight_or_classical": [
                model for model in ("finite_hmm", "hdp_hmm", "vomm") if model in fits_by_family
            ],
            "neural": ["transformer"],
            "note": "Fit counts describe planned work, not measured runtime or evidence.",
        },
        "artifact_plan": [
            {
                "path": name,
                "plan_only_status": "written" if name in {
                    "config.json",
                    "preprocessing_report.json",
                    "exclusions.csv",
                    "splits/*.json",
                    "execution_plan.json",
                } else "not_written_until_execution",
            }
            for name in artifact_names
        ],
    }


def _parse_boundary(value: object, *, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, float) and math.isfinite(value) and value in {0.0, 1.0}:
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
    raise ValueError(f"structural annotation boundary must be boolean at CSV row {row_number}")


def _load_structural_annotations(path_value: str | None) -> dict[str, object] | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"structural annotations file does not exist: {path}")
    frame = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_STRUCTURAL_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "structural annotation CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    records: list[dict[str, object]] = []
    seen_coordinates: set[tuple[str, int]] = set()
    for row_offset, row in frame.iterrows():
        row_number = int(row_offset) + 2
        piece_id = str(row["piece_id"]).strip()
        segment_label = str(row["segment_label"]).strip()
        if not piece_id or piece_id.lower() == "nan":
            raise ValueError(f"structural annotation piece_id is empty at CSV row {row_number}")
        if not segment_label or segment_label.lower() == "nan":
            raise ValueError(f"structural annotation segment_label is empty at CSV row {row_number}")
        event_value = row["event_index"]
        try:
            event_number = float(event_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"structural annotation event_index is invalid at CSV row {row_number}") from exc
        if not math.isfinite(event_number) or not event_number.is_integer() or event_number < 0:
            raise ValueError(f"structural annotation event_index is invalid at CSV row {row_number}")
        event_index = int(event_number)
        coordinate = (piece_id, event_index)
        if coordinate in seen_coordinates:
            raise ValueError(f"duplicate structural annotation coordinate: {coordinate}")
        seen_coordinates.add(coordinate)
        records.append(
            {
                "piece_id": piece_id,
                "event_index": event_index,
                "segment_label": segment_label,
                "boundary": _parse_boundary(row["boundary"], row_number=row_number),
            }
        )
    records.sort(key=lambda item: (str(item["piece_id"]), int(item["event_index"])))
    return {"path": str(path), "records": records}


def _build_piece_metric_rows(
    model_name: str,
    fraction: float,
    data_seed: int,
    model_seed: int | str,
    piece_metrics: list[dict[str, object]],
    evaluated_pieces: list[PreparedPiece],
) -> list[dict[str, object]]:
    pieces_by_id = {piece.piece_id: piece for piece in evaluated_pieces}
    rows = []
    for item in piece_metrics:
        piece_id = str(item.get("piece_id", ""))
        if piece_id not in pieces_by_id:
            raise ValueError(f"Model {model_name} returned metrics for unknown piece_id={piece_id!r}.")
        piece = pieces_by_id[piece_id]
        if not piece.canonical_work_id.strip():
            raise ValueError(f"PreparedPiece {piece_id!r} has an empty canonical_work_id.")
        row = dict(item)
        row["piece_id"] = piece.piece_id
        row["canonical_work_id"] = piece.canonical_work_id
        row["model"] = model_name
        row["frac"] = fraction
        row["data_seed"] = data_seed
        row["model_seed"] = model_seed
        rows.append(row)
    return rows


def _append_protocol_evidence(
    *,
    output_path: Path,
    evidence: list[dict[str, object]],
    model_name: str,
    fraction: float,
    data_seed: int,
    model_seed: int | str,
    evaluated_pieces: list[PreparedPiece],
    piece_metrics: list[dict[str, object]],
) -> None:
    metrics_by_piece = {str(item.get("piece_id", "")): item for item in piece_metrics}
    expected_piece_ids = {piece.piece_id for piece in evaluated_pieces}
    unexpected_piece_ids = sorted(set(metrics_by_piece) - expected_piece_ids)
    failed = bool(unexpected_piece_ids)

    for piece in evaluated_pieces:
        expected = list(range(len(piece.tokens)))
        metric = metrics_by_piece.get(piece.piece_id)
        raw_scored = None if metric is None else metric.get("scored_event_indices")
        invalid_values: list[object] = []
        scored: list[int] = []
        if isinstance(raw_scored, (list, tuple)):
            for value in raw_scored:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    invalid_values.append(value)
                else:
                    scored.append(int(value))
        else:
            invalid_values.append(raw_scored)

        counts = Counter(scored)
        duplicates = sorted(index for index, count in counts.items() if count > 1)
        out_of_range = sorted(index for index in scored if index < 0 or index >= len(piece.tokens))
        scored_in_range = {index for index in scored if 0 <= index < len(piece.tokens)}
        omissions = sorted(set(expected) - scored_in_range)
        count_mismatch = len(scored) != len(expected)
        order_mismatch = scored != expected
        item_failed = bool(
            metric is None
            or invalid_values
            or duplicates
            or out_of_range
            or omissions
            or count_mismatch
            or order_mismatch
        )
        failed = failed or item_failed
        entry: dict[str, object] = {
            "model": model_name,
            "frac": fraction,
            "data_seed": data_seed,
            "model_seed": model_seed,
            "piece_id": piece.piece_id,
            "canonical_work_id": piece.canonical_work_id,
            "status": "failed" if item_failed else "passed",
            "expected_count": len(expected),
            "scored_count": len(scored),
            "count_mismatch": count_mismatch,
            "order_mismatch": order_mismatch,
        }
        if item_failed:
            # Los indices solo informan cuando algo fallo. En una corrida que pasa,
            # expected y scored son ambos list(range(n)): identicos y sin informacion.
            entry.update(
                {
                    "expected_event_indices": expected,
                    "scored_event_indices": scored,
                    "duplicate_event_indices": duplicates,
                    "omitted_event_indices": omissions,
                    "out_of_range_event_indices": out_of_range,
                    "invalid_event_indices": invalid_values,
                }
            )
        evidence.append(entry)

    print(
        f"[runner] {model_name} frac={fraction} data_seed={data_seed} "
        f"model_seed={model_seed} done ({len(evaluated_pieces)} test pieces)",
        file=sys.stderr,
        flush=True,
    )

    # El audit completo se escribe una sola vez al cerrar la corrida. Escribirlo aqui en
    # cada bloque reescribia una lista que crece, con costo cuadratico. Solo se adelanta
    # la escritura cuando algo fallo, para que el diagnostico sobreviva a la excepcion.
    if failed:
        _write_json(
            output_path,
            {
                "status": "failed",
                "policy": "every musical test event must be exposed by model evaluation exactly once",
                "unexpected_piece_ids": unexpected_piece_ids,
                "evidence": evidence,
            },
        )
        raise ValueError(f"protocol coverage violation for model {model_name}")


def _aggregate_summary(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["model"]), float(row["frac"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (model, fraction), rows in sorted(grouped.items()):
        perplexities = [float(row["test_ppl"]) for row in rows]
        nlls = [float(row["test_nll"]) for row in rows]
        train_tokens = [int(row["n_train_tokens"]) for row in rows]
        summary_rows.append(
            {
                "model": model,
                "frac": fraction,
                "mean_n_train_tokens": mean(train_tokens),
                "mean_test_ppl": mean(perplexities),
                "std_test_ppl": pstdev(perplexities) if len(perplexities) > 1 else 0.0,
                "mean_test_nll": mean(nlls),
                "std_test_nll": pstdev(nlls) if len(nlls) > 1 else 0.0,
                "runs": len(rows),
            }
        )
    return summary_rows


def _plot_learning_curve(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        return
    plt.figure(figsize=(8, 5))
    frame = pd.DataFrame(summary_rows)
    for model_name, group in frame.groupby("model"):
        group = group.sort_values("mean_n_train_tokens")
        plt.plot(group["mean_n_train_tokens"], group["mean_test_ppl"], marker="o", label=model_name)
        plt.fill_between(
            group["mean_n_train_tokens"],
            group["mean_test_ppl"] - group["std_test_ppl"],
            group["mean_test_ppl"] + group["std_test_ppl"],
            alpha=0.2,
        )
    plt.xscale("log")
    plt.xlabel("Tokens de entrenamiento")
    plt.ylabel("Perplejidad de test = exp(NLL)")
    plt.title("Curva predictiva compartida por familia")
    plt.legend()
    plt.tight_layout()
    ensure_directory(output_path.parent)
    plt.savefig(output_path)
    plt.close()


def _engineering_cost_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "model",
        "data_seed",
        "model_seed",
        "frac",
        "n_train_pieces",
        "n_train_tokens",
        "fit_wall_clock_s",
        "evaluation_wall_clock_s",
        "n_params",
        "selected_states",
        "effective_states",
        "count_table_size",
        "device",
    )
    rows = []
    for raw in raw_rows:
        row = {field: raw.get(field) for field in fields}
        row.update(
            {
                "peak_memory_bytes": None,
                "peak_memory_status": "not_measured_reliably",
                "energy_joules": None,
                "energy_status": "not_measured_reliably",
                "cost_scope": "separate_units_not_collapsed",
            }
        )
        rows.append(row)
    return rows


def _collect_structural_predictions(
    target: list[dict[str, object]],
    model_name: str,
    evaluation: Mapping[str, object],
    *,
    fraction: float,
    data_seed: int,
    model_seed: object,
) -> None:
    predictions = evaluation.get("structural_predictions")
    if not isinstance(predictions, list):
        return
    for prediction in predictions:
        if isinstance(prediction, Mapping):
            target.append(
                {
                    **dict(prediction),
                    "model": model_name,
                    "frac": fraction,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                }
            )


def _build_structural_evaluation(
    reference: dict[str, object] | None,
    predictions: list[dict[str, object]],
    *,
    tolerance: int,
) -> dict[str, object]:
    if reference is None:
        return {
            "status": "not_evaluated",
            "reason": "missing_structural_annotations_input",
            "missing_input": "structural_annotations_path",
            "model_metrics": [],
        }
    if not predictions:
        return {
            "status": "not_evaluated",
            "reason": "missing_inferred_structure_artifact",
            "reference_annotations": {
                "path": reference["path"],
                "n_rows": len(reference["records"]),
                "required_columns": list(REQUIRED_STRUCTURAL_COLUMNS),
            },
            "missing_artifact": "per-model inferred segment labels and boundaries",
            "model_metrics": [],
        }

    reference_by_piece: dict[str, list[dict[str, object]]] = {}
    for record in reference["records"]:
        reference_by_piece.setdefault(str(record["piece_id"]), []).append(record)
    piece_metrics: list[dict[str, object]] = []
    for prediction in predictions:
        piece_id = str(prediction.get("piece_id", ""))
        records = reference_by_piece.get(piece_id)
        labels = prediction.get("segment_labels")
        boundaries = prediction.get("boundary_indices")
        if not records or not isinstance(labels, list) or not isinstance(boundaries, list):
            continue
        reference_labels = [record["segment_label"] for record in records]
        if len(reference_labels) != len(labels):
            continue
        reference_boundaries = [int(record["event_index"]) for record in records if record["boundary"]]
        boundary_scores = boundary_f1(reference_boundaries, boundaries, tolerance)
        piece_metrics.append(
            {
                "model": prediction["model"],
                "piece_id": piece_id,
                "boundary": boundary_scores,
                "normalized_mutual_information": normalized_mutual_information(reference_labels, labels),
                "adjusted_rand_index": adjusted_rand_index(reference_labels, labels),
            }
        )
    if not piece_metrics:
        return {
            "status": "not_evaluated",
            "reason": "no_comparable_reference_and_inferred_arrays",
            "model_metrics": [],
        }

    model_metrics = []
    for model_name in sorted({str(item["model"]) for item in piece_metrics}):
        rows = [item for item in piece_metrics if item["model"] == model_name]
        model_metrics.append(
            {
                "model": model_name,
                "n_pieces": len(rows),
                "mean_boundary_f1": mean(float(item["boundary"]["f1"]) for item in rows),
                "mean_nmi": mean(float(item["normalized_mutual_information"]) for item in rows),
                "mean_ari": mean(float(item["adjusted_rand_index"]) for item in rows),
            }
        )
    return {
        "status": "ok",
        "boundary_tolerance": tolerance,
        "piece_metrics": piece_metrics,
        "model_metrics": model_metrics,
    }


def _build_pareto_summary(
    raw_rows: list[dict[str, object]],
    structural_evaluation: dict[str, object],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        if float(row["frac"]) == 1.0:
            grouped.setdefault(str(row["model"]), []).append(row)
    predictive_cost_rows = []
    for model_name, rows in sorted(grouped.items()):
        values = {
            "model": model_name,
            "mean_test_nll": mean(float(row["test_nll"]) for row in rows),
            "mean_fit_wall_clock_s": mean(float(row["fit_wall_clock_s"]) for row in rows),
            "mean_evaluation_wall_clock_s": mean(
                float(row["evaluation_wall_clock_s"]) for row in rows
            ),
            "runs": len(rows),
        }
        if all(math.isfinite(float(value)) for key, value in values.items() if key.startswith("mean_")):
            predictive_cost_rows.append(values)
    partial_axes = (
        "mean_test_nll",
        "mean_fit_wall_clock_s",
        "mean_evaluation_wall_clock_s",
    )
    partial_payload = {
        "status": "ok" if predictive_cost_rows else "not_evaluated",
        "scope": "predictive_cost_partial_frontier",
        "axes": list(partial_axes),
        "candidates": predictive_cost_rows,
        "frontier": pareto_front(predictive_cost_rows, minimize=partial_axes, maximize=())
        if predictive_cost_rows
        else [],
    }

    if structural_evaluation.get("status") != "ok":
        full_payload = {
            "status": "not_evaluated",
            "reason": "structural_measurements_unavailable",
            "scope": "full_three_axis_frontier",
            "frontier": [],
        }
    else:
        structural_by_model = {
            str(row["model"]): row
            for row in structural_evaluation.get("model_metrics", [])
        }
        missing_models = sorted(set(grouped) - set(structural_by_model))
        full_rows = [
            {**row, "mean_boundary_f1": structural_by_model[row["model"]]["mean_boundary_f1"]}
            for row in predictive_cost_rows
            if row["model"] in structural_by_model
        ]
        if missing_models:
            full_payload = {
                "status": "incomparable",
                "reason": "missing_structural_measurements_for_models",
                "missing_models": missing_models,
                "scope": "full_three_axis_frontier",
                "frontier": [],
            }
        else:
            full_payload = {
                "status": "ok" if full_rows else "incomparable",
                "scope": "full_three_axis_frontier",
                "axes": [*partial_axes, "mean_boundary_f1"],
                "frontier": pareto_front(
                    full_rows,
                    minimize=partial_axes,
                    maximize=("mean_boundary_f1",),
                ) if full_rows else [],
            }
    return {
        "predictive_cost_partial_frontier": partial_payload,
        "full_three_axis_frontier": full_payload,
    }


def _write_split_artifacts(
    output_root: Path,
    fixed_splits: FixedSplits,
    nested_by_seed: dict[int, list[tuple[float, list[PreparedPiece]]]],
) -> None:
    split_root = ensure_directory(output_root / "splits")
    _write_json(split_root / "test_pieces.json", _split_payload(fixed_splits.test_pieces))
    _write_json(split_root / "val_pieces.json", _split_payload(fixed_splits.validation_pieces))
    for data_seed, nested_subsets in sorted(nested_by_seed.items()):
        _write_json(
            split_root / f"train_fractions_seed{data_seed}.json",
            {
                "data_seed": data_seed,
                "fractions": [
                    {"frac": fraction, **_split_payload(subset)}
                    for fraction, subset in nested_subsets
                ],
            },
        )


def run_learning_curve_experiment(
    config: LearningCurveConfig,
    *,
    run_name: str = "learning_curve",
    max_files: int | None = None,
    plan_only: bool = False,
    n_workers: int | None = None,
    corpus_cache_path: str | Path | None = None,
    corpus_sample_seed: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    run_directory = Path(config.results_root) / run_name
    if not resume:
        _guard_fresh_run_directory(run_directory)
    output_root = ensure_directory(run_directory)
    preparation = prepare_corpus(
        config.experiment,
        max_files=max_files,
        n_workers=n_workers,
        cache_path=corpus_cache_path,
        sample_seed=corpus_sample_seed,
    )
    fixed_splits = _build_grouped_fixed_splits(
        preparation.pieces,
        test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )
    nested_by_seed = {
        data_seed: build_nested_training_subsets(
            fixed_splits.train_pool_pieces,
            fractions=config.train_fractions,
            data_seed=data_seed,
        )
        for data_seed in config.data_seeds
    }

    _write_json(output_root / "config.json", asdict(config))
    _write_json(output_root / "preprocessing_report.json", build_preprocessing_report(preparation))
    write_csv(output_root / "exclusions.csv", [item.__dict__ for item in preparation.exclusions])
    _write_split_artifacts(output_root, fixed_splits, nested_by_seed)
    structural_reference = _load_structural_annotations(config.structural_annotations_path)

    if plan_only:
        execution_plan = _build_execution_plan(config, fixed_splits, nested_by_seed)
        _write_json(output_root / "execution_plan.json", execution_plan)
        return {
            "status": "planned_no_evidence",
            "output_root": str(output_root),
            "n_prepared_pieces": len(preparation.pieces),
            "n_exclusions": len(preparation.exclusions),
            "n_runs": 0,
            "execution_plan": str(output_root / "execution_plan.json"),
        }

    tokenizer = build_tokenizer(config.experiment.representation)
    bos_token_id = tokenizer.bos_token_id
    comparison_vocabulary_size = bos_token_id + 1
    max_context_length = config.experiment.windows.max_context_length
    test_piece_metadata = {
        piece.piece_id: {
            "title": piece.title,
            "composer": piece.composer,
            "canonical_work_id": piece.canonical_work_id,
        }
        for piece in fixed_splits.test_pieces
    }

    checkpoint_path = output_root / "checkpoint.jsonl"
    completed_cells, restored = _load_checkpoint(checkpoint_path) if resume else (set(), None)
    raw_rows: list[dict[str, object]] = list(restored["raw_rows"]) if restored else []
    piece_metric_rows: list[dict[str, object]] = list(restored["piece_metric_rows"]) if restored else []
    protocol_evidence: list[dict[str, object]] = list(restored["protocol_evidence"]) if restored else []
    structural_predictions: list[dict[str, object]] = list(restored["structural_predictions"]) if restored else []
    protocol_path = output_root / "protocol_audit.json"

    total_cells = sum(len(subsets) for subsets in nested_by_seed.values())
    if completed_cells:
        print(
            f"[runner] reanudando: {len(completed_cells)} de {total_cells} celdas ya hechas",
            file=sys.stderr,
            flush=True,
        )

    for data_seed, nested_subsets in sorted(nested_by_seed.items()):
        for fraction, train_pieces in nested_subsets:
            cell = f"data_seed={data_seed},frac={fraction}"
            if cell in completed_cells:
                continue
            marks = {
                "raw_rows": len(raw_rows),
                "piece_metric_rows": len(piece_metric_rows),
                "protocol_evidence": len(protocol_evidence),
                "structural_predictions": len(structural_predictions),
            }
            n_train_tokens = sum(len(piece.tokens) for piece in train_pieces)
            n_train_pieces = len(train_pieces)

            if config.include_vomm_control:
                vomm = select_vomm_by_validation(
                    train_sequences=[list(piece.tokens) for piece in train_pieces],
                    validation_pieces=fixed_splits.validation_pieces,
                    candidate_orders=config.vomm_candidate_orders,
                    vocabulary_size=comparison_vocabulary_size,
                    bos_token_id=bos_token_id,
                    max_context_length=max_context_length,
                )
                vomm_eval = vomm.evaluate(fixed_splits.test_pieces, max_context_length)
                vomm_summary = vomm_eval["summary"]
                _append_protocol_evidence(
                    output_path=protocol_path,
                    evidence=protocol_evidence,
                    model_name="vomm",
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=_VOMM_DETERMINISTIC_SEED,
                    evaluated_pieces=fixed_splits.test_pieces,
                    piece_metrics=vomm_eval["piece_metrics"],
                )
                _collect_structural_predictions(
                    structural_predictions,
                    "vomm",
                    vomm_eval,
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=_VOMM_DETERMINISTIC_SEED,
                )
                raw_rows.append(
                    {
                        "model": "vomm",
                        "data_seed": data_seed,
                        "model_seed": _VOMM_DETERMINISTIC_SEED,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": vomm_summary["n_params"],
                        "selected_states": None,
                        "effective_states": None,
                        "count_table_size": vomm_summary["count_table_size"],
                        "device": "cpu",
                        "val_ppl": math.exp(float(vomm_summary["validation_nll_per_token"])),
                        "test_ppl": vomm_summary["test_perplexity"],
                        "test_nll": vomm_summary["test_nll_per_token"],
                        "test_accuracy": vomm_summary["accuracy"],
                        "test_brier_score": vomm_summary["brier_score"],
                        "train_time_sec": vomm_summary["train_time_sec"],
                        "fit_wall_clock_s": vomm_summary["train_time_sec"],
                        "evaluation_wall_clock_s": vomm_summary["evaluation_wall_clock_s"],
                        "hyperparams_json": json.dumps({"selected_order": vomm.selected_order}),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows(
                        "vomm",
                        fraction,
                        data_seed,
                        _VOMM_DETERMINISTIC_SEED,
                        vomm_eval["piece_metrics"],
                        fixed_splits.test_pieces,
                    )
                )

            for model_seed in config.model_seeds:
                finite_hmm = FiniteGlobalHMM(
                    candidate_num_states=config.finite_hmm_states,
                    max_iterations=config.finite_hmm_max_iterations,
                    tolerance=config.finite_hmm_tolerance,
                    seed=model_seed,
                    vocab_size=comparison_vocabulary_size,
                )
                finite_hmm.fit(
                    train_pieces,
                    fixed_splits.validation_pieces,
                    bos_token_id=bos_token_id,
                    max_context_length=max_context_length,
                )
                finite_eval_start = time.perf_counter()
                finite_eval = finite_hmm.evaluate(
                    fixed_splits.test_pieces,
                    bos_token_id=bos_token_id,
                    max_context_length=max_context_length,
                )
                finite_eval_elapsed = time.perf_counter() - finite_eval_start
                finite_summary = finite_eval["summary"]
                _append_protocol_evidence(
                    output_path=protocol_path,
                    evidence=protocol_evidence,
                    model_name="finite_hmm",
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=model_seed,
                    evaluated_pieces=fixed_splits.test_pieces,
                    piece_metrics=finite_eval["piece_metrics"],
                )
                _collect_structural_predictions(
                    structural_predictions,
                    "finite_hmm",
                    finite_eval,
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=model_seed,
                )
                raw_rows.append(
                    {
                        "model": "finite_hmm",
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": finite_summary["n_params"],
                        "selected_states": finite_summary["selected_states"],
                        "effective_states": None,
                        "count_table_size": None,
                        "device": "cpu",
                        "val_ppl": math.exp(float(finite_summary["validation_nll_per_token"])),
                        "test_ppl": finite_summary["test_perplexity"],
                        "test_nll": finite_summary["test_nll_per_token"],
                        "train_time_sec": finite_summary["train_time_sec"],
                        "fit_wall_clock_s": finite_summary["train_time_sec"],
                        "evaluation_wall_clock_s": finite_eval_elapsed,
                        "hyperparams_json": json.dumps({"selected_states": finite_summary["selected_states"]}),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows(
                        "finite_hmm",
                        fraction,
                        data_seed,
                        model_seed,
                        finite_eval["piece_metrics"],
                        fixed_splits.test_pieces,
                    )
                )

                hdp_hmm = GlobalHDPHMM(
                    truncation_level=config.hdp_truncation_level,
                    n_iters=config.hdp_n_iters,
                    burn_in=config.hdp_burn_in,
                    hyperparameter_grid=config.hdp_hyperparameter_grid,
                    seed=model_seed,
                    vocab_size=comparison_vocabulary_size,
                )
                hdp_fit = hdp_hmm.fit(
                    train_pieces,
                    fixed_splits.validation_pieces,
                    bos_token_id=bos_token_id,
                    max_context_length=max_context_length,
                )
                hdp_eval_start = time.perf_counter()
                hdp_eval = hdp_hmm.evaluate(
                    fixed_splits.test_pieces,
                    bos_token_id=bos_token_id,
                    max_context_length=max_context_length,
                )
                hdp_eval_elapsed = time.perf_counter() - hdp_eval_start
                hdp_summary = hdp_eval["summary"]
                _append_protocol_evidence(
                    output_path=protocol_path,
                    evidence=protocol_evidence,
                    model_name="hdp_hmm",
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=model_seed,
                    evaluated_pieces=fixed_splits.test_pieces,
                    piece_metrics=hdp_eval["piece_metrics"],
                )
                _collect_structural_predictions(
                    structural_predictions,
                    "hdp_hmm",
                    hdp_eval,
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=model_seed,
                )
                raw_rows.append(
                    {
                        "model": "hdp_hmm",
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": hdp_summary["n_params"],
                        "selected_states": None,
                        "effective_states": hdp_summary["effective_states"],
                        "count_table_size": None,
                        "device": "cpu",
                        "val_ppl": math.exp(float(hdp_summary["validation_nll_per_token"])),
                        "test_ppl": hdp_summary["test_perplexity"],
                        "test_nll": hdp_summary["test_nll_per_token"],
                        "train_time_sec": hdp_summary["train_time_sec"],
                        "fit_wall_clock_s": hdp_summary["train_time_sec"],
                        "evaluation_wall_clock_s": hdp_eval_elapsed,
                        "hyperparams_json": json.dumps(hdp_fit["selected_hyperparameters"]),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows(
                        "hdp_hmm",
                        fraction,
                        data_seed,
                        model_seed,
                        hdp_eval["piece_metrics"],
                        fixed_splits.test_pieces,
                    )
                )

                transformer_model = _build_transformer_model(
                    config,
                    model_seed,
                    piece_metadata_by_id=test_piece_metadata,
                )
                dataloaders = _build_transformer_dataloaders(
                    config,
                    train_pieces=train_pieces,
                    validation_pieces=fixed_splits.validation_pieces,
                    test_pieces=fixed_splits.test_pieces,
                )
                transformer_fit_start = time.perf_counter()
                transformer_fit = transformer_model.fit(
                    dataloaders["train"],
                    dataloaders["validation"],
                )
                transformer_fit_wall_clock_s = time.perf_counter() - transformer_fit_start
                transformer_eval = transformer_model.evaluate(dataloaders["test"])
                transformer_summary = _transformer_summary_row(transformer_fit, transformer_eval)
                _append_protocol_evidence(
                    output_path=protocol_path,
                    evidence=protocol_evidence,
                    model_name="transformer",
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=model_seed,
                    evaluated_pieces=fixed_splits.test_pieces,
                    piece_metrics=transformer_eval["piece_metrics"],
                )
                _collect_structural_predictions(
                    structural_predictions,
                    "transformer",
                    transformer_eval,
                    fraction=fraction,
                    data_seed=data_seed,
                    model_seed=model_seed,
                )
                raw_rows.append(
                    {
                        "model": "transformer",
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "frac": fraction,
                        "n_train_pieces": n_train_pieces,
                        "n_train_tokens": n_train_tokens,
                        "n_params": transformer_summary["n_params"],
                        "selected_states": None,
                        "effective_states": None,
                        "count_table_size": None,
                        "device": transformer_summary["device"],
                        "val_ppl": math.exp(float(transformer_summary["validation_nll_per_token"])),
                        "test_ppl": transformer_summary["test_perplexity"],
                        "test_nll": transformer_summary["test_nll_per_token"],
                        "test_accuracy": transformer_summary["test_accuracy"],
                        "test_brier_score": transformer_summary["test_brier_score"],
                        "train_time_sec": transformer_summary["train_time_sec"],
                        "fit_wall_clock_s": transformer_fit_wall_clock_s,
                        "evaluation_wall_clock_s": transformer_summary["evaluation_wall_clock_s"],
                        "hyperparams_json": json.dumps(
                            {"architecture": config.experiment.transformer.architecture}
                        ),
                    }
                )
                piece_metric_rows.extend(
                    _build_piece_metric_rows(
                        "transformer",
                        fraction,
                        data_seed,
                        model_seed,
                        transformer_eval["piece_metrics"],
                        fixed_splits.test_pieces,
                    )
                )

            _append_checkpoint(
                checkpoint_path,
                cell,
                {
                    "raw_rows": raw_rows[marks["raw_rows"] :],
                    "piece_metric_rows": piece_metric_rows[marks["piece_metric_rows"] :],
                    "protocol_evidence": protocol_evidence[marks["protocol_evidence"] :],
                    "structural_predictions": structural_predictions[
                        marks["structural_predictions"] :
                    ],
                },
            )
            completed_cells.add(cell)
            print(
                f"[runner] celda {cell} lista ({len(completed_cells)}/{total_cells})",
                file=sys.stderr,
                flush=True,
            )

    summary_rows = _aggregate_summary(raw_rows)
    write_csv(output_root / "results_raw.csv", raw_rows)
    write_csv(output_root / "results_summary.csv", summary_rows)
    write_csv(output_root / "piece_metrics_raw.csv", piece_metric_rows)
    write_csv(output_root / "engineering_costs.csv", _engineering_cost_rows(raw_rows))
    _plot_learning_curve(summary_rows, output_root / "learning_curve.png")

    pairwise_payload = pairwise_model_comparisons(
        piece_metric_rows,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.bootstrap_seed,
    )
    _write_json(output_root / "pairwise_comparisons.json", pairwise_payload)
    structural_payload = _build_structural_evaluation(
        structural_reference,
        structural_predictions,
        tolerance=config.boundary_tolerance,
    )
    _write_json(output_root / "structural_evaluation.json", structural_payload)
    pareto_payload = _build_pareto_summary(raw_rows, structural_payload)
    _write_json(output_root / "pareto_summary.json", pareto_payload)
    _write_json(
        protocol_path,
        {
            "status": "passed",
            "policy": "every musical test event must be exposed by model evaluation exactly once",
            "unexpected_piece_ids": [],
            "evidence": protocol_evidence,
        },
    )

    artifacts = {
        "results_raw": str(output_root / "results_raw.csv"),
        "results_summary": str(output_root / "results_summary.csv"),
        "piece_metrics_raw": str(output_root / "piece_metrics_raw.csv"),
        "pairwise_comparisons": str(output_root / "pairwise_comparisons.json"),
        "engineering_costs": str(output_root / "engineering_costs.csv"),
        "protocol_audit": str(protocol_path),
        "structural_evaluation": str(output_root / "structural_evaluation.json"),
        "pareto_summary": str(output_root / "pareto_summary.json"),
        "learning_curve": str(output_root / "learning_curve.png"),
    }
    summary = {
        "status": "completed",
        "output_root": str(output_root),
        "n_prepared_pieces": len(preparation.pieces),
        "n_exclusions": len(preparation.exclusions),
        "n_runs": len(raw_rows),
        "pairwise_comparisons": pairwise_payload,
        "artifacts": artifacts,
    }
    # run_summary.json deja el resumen en disco para que la auditoría de
    # artefactos pueda contrastar archivos y cifras sin re-ejecutar nada.
    _write_json(output_root / "run_summary.json", {key: value for key, value in summary.items() if key != "pairwise_comparisons"})
    return summary
