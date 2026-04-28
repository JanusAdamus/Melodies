from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable

from ..experiment.runner import run_small_transformer_experiment
from ..experiment.storage import ensure_directory, write_csv, write_json
from ..profiles import build_profile_config


DEFAULT_MANIFEST_PATH = Path(__file__).with_name("canonical_runs.json")


@dataclass(frozen=True)
class BenchmarkRunSpec:
    run_id: str
    profile: str
    purpose: str
    comparison_group: str
    expected_budget_class: str
    run_name: str
    representation: str | None = None
    context_length: int | None = None
    max_files: int | None = None
    max_windows: dict[str, int] | None = None
    seed: int | None = None
    notes: tuple[str, ...] = ()
    config_overrides: dict[str, Any] | None = None


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    return _load_json(target)


def _apply_dataclass_overrides(instance: Any, overrides: dict[str, Any]) -> Any:
    if not overrides:
        return instance
    if not is_dataclass(instance):
        raise TypeError(f"Cannot apply dataclass overrides to non-dataclass instance: {type(instance)!r}")

    allowed_fields = {field.name for field in fields(instance)}
    updates: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in allowed_fields:
            raise KeyError(f"Unknown override field '{key}' for {type(instance).__name__}")
        current_value = getattr(instance, key)
        if isinstance(value, dict) and is_dataclass(current_value):
            updates[key] = _apply_dataclass_overrides(current_value, value)
        else:
            updates[key] = value
    return replace(instance, **updates)


def _apply_config_overrides(config: Any, overrides: dict[str, Any] | None, seed: int | None = None) -> Any:
    updated = config
    if overrides:
        updated = _apply_dataclass_overrides(updated, overrides)
    if seed is not None:
        updated = replace(updated, split=replace(updated.split, seed=int(seed)))
    return updated


def load_benchmark_run_specs(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> list[BenchmarkRunSpec]:
    payload = _load_json(manifest_path)
    return [
        BenchmarkRunSpec(
            run_id=str(item["run_id"]),
            profile=str(item["profile"]),
            purpose=str(item["purpose"]),
            comparison_group=str(item["comparison_group"]),
            expected_budget_class=str(item["expected_budget_class"]),
            run_name=str(item["run_name"]),
            representation=item.get("representation"),
            context_length=item.get("context_length"),
            max_files=item.get("max_files"),
            max_windows=item.get("max_windows"),
            seed=item.get("seed"),
            notes=tuple(item.get("notes", [])),
            config_overrides=item.get("config_overrides"),
        )
        for item in payload
    ]


def resolve_benchmark_suite(
    run_specs: Iterable[BenchmarkRunSpec],
    *,
    run_ids: set[str] | None = None,
    comparison_group: str | None = None,
    only_smoke: bool = False,
    only_full: bool = False,
) -> list[BenchmarkRunSpec]:
    resolved = []
    for spec in run_specs:
        if run_ids is not None and spec.run_id not in run_ids:
            continue
        if comparison_group is not None and spec.comparison_group != comparison_group:
            continue
        if only_smoke and spec.expected_budget_class != "smoke":
            continue
        if only_full and spec.expected_budget_class != "full":
            continue
        resolved.append(spec)
    return resolved


def _flatten_notes(notes: tuple[str, ...]) -> str:
    return " | ".join(note.strip() for note in notes if note.strip())


def _extract_summary_row(spec: BenchmarkRunSpec, result_dir: str | Path) -> dict[str, Any]:
    result_root = Path(result_dir)
    config_payload = _read_json_if_exists(result_root / "config.json") or {}
    data_payload = _read_json_if_exists(result_root / "data_summary.json") or {}
    fit_summary = _read_json_if_exists(result_root / "transformer" / "fit_summary.json") or {}
    test_summary = _read_json_if_exists(result_root / "transformer" / "test_summary.json") or {}
    validation_slice_metrics = _read_json_if_exists(result_root / "transformer" / "validation_slice_metrics.json") or {}
    test_slice_metrics = _read_json_if_exists(result_root / "transformer" / "test_slice_metrics.json") or {}

    dataset = data_payload.get("dataset", {})
    runtime = test_summary.get("runtime", data_payload.get("runtime", {}))
    timings = data_payload.get("timings", {})
    transformer = config_payload.get("transformer", {})
    representation = config_payload.get("representation", {})
    windows = config_payload.get("windows", {})

    return {
        "run_id": spec.run_id,
        "run_name": spec.run_name,
        "profile": spec.profile,
        "comparison_group": spec.comparison_group,
        "budget_class": spec.expected_budget_class,
        "representation": representation.get("primary", spec.representation),
        "context_length": windows.get("max_context_length", spec.context_length),
        "n_layers": transformer.get("n_layers"),
        "d_model": transformer.get("d_model"),
        "n_heads": transformer.get("n_heads"),
        "attention_implementation": transformer.get("attention_implementation"),
        "use_relative_position_bias": transformer.get("use_relative_position_bias"),
        "n_train_pieces": dataset.get("n_train_pieces"),
        "n_validation_pieces": dataset.get("n_validation_pieces"),
        "n_test_pieces": dataset.get("n_test_pieces"),
        "n_train_windows": dataset.get("n_train_windows"),
        "n_validation_windows": dataset.get("n_validation_windows"),
        "n_test_windows": dataset.get("n_test_windows"),
        "nll_per_token": test_summary.get("nll_per_token"),
        "perplexity": test_summary.get("perplexity"),
        "accuracy": test_summary.get("accuracy"),
        "top_3_accuracy": test_summary.get("top_3_accuracy"),
        "top_5_accuracy": test_summary.get("top_5_accuracy"),
        "best_validation_nll": fit_summary.get("best_validation_nll"),
        "best_validation_perplexity": fit_summary.get("best_validation_perplexity"),
        "fit_wall_clock_s": timings.get("fit_wall_clock_s"),
        "evaluation_wall_clock_s": timings.get("evaluation_wall_clock_s"),
        "total_wall_clock_s": timings.get("total_wall_clock_s"),
        "parameter_count": test_summary.get("parameter_count", fit_summary.get("parameter_count")),
        "device": runtime.get("device"),
        "actual_precision": runtime.get("actual_precision"),
        "attention_implementation_effective": runtime.get("attention_implementation_effective"),
        "flash_attention_candidate": runtime.get("flash_attention_candidate"),
        "seed": runtime.get("seed", spec.seed),
        "result_dir": str(result_root),
        "status": "completed",
        "notes": _flatten_notes(spec.notes),
        "validation_slices_recorded": sorted(validation_slice_metrics.keys()) if isinstance(validation_slice_metrics, dict) else [],
        "test_slices_recorded": sorted(test_slice_metrics.keys()) if isinstance(test_slice_metrics, dict) else [],
    }


def _build_markdown_summary(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run_id",
        "profile",
        "representation",
        "context_length",
        "attention_implementation",
        "nll_per_token",
        "accuracy",
        "top_3_accuracy",
        "top_5_accuracy",
        "total_wall_clock_s",
        "parameter_count",
        "status",
    ]
    lines = [
        "# Transformer Benchmark Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def _build_status_rows(resolved_specs: Iterable[BenchmarkRunSpec], rows_by_run_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    statuses = []
    for spec in resolved_specs:
        existing = rows_by_run_id.get(spec.run_id)
        statuses.append(
            {
                "run_id": spec.run_id,
                "profile": spec.profile,
                "run_name": spec.run_name,
                "comparison_group": spec.comparison_group,
                "budget_class": spec.expected_budget_class,
                "status": "completed" if existing is not None else "missing",
                "result_dir": existing.get("result_dir") if existing else "",
                "notes": _flatten_notes(spec.notes),
            }
        )
    return statuses


def _resolve_output_root(results_root: str | Path) -> Path:
    return ensure_directory(Path(results_root) / "benchmark_suite")


def collect_benchmark_suite(
    resolved_specs: Iterable[BenchmarkRunSpec],
    *,
    results_root: str | Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in resolved_specs:
        result_dir = Path(results_root) / spec.run_name
        config_path = result_dir / "config.json"
        test_summary_path = result_dir / "transformer" / "test_summary.json"
        if config_path.exists() and test_summary_path.exists():
            rows.append(_extract_summary_row(spec, result_dir))

    output_root = _resolve_output_root(results_root)
    rows_by_run_id = {row["run_id"]: row for row in rows}
    status_rows = _build_status_rows(resolved_specs, rows_by_run_id)
    manifest_payload = [asdict(spec) for spec in resolved_specs]

    write_json(output_root / "run_manifest_resolved.json", {"runs": manifest_payload})
    write_json(output_root / "run_status.json", {"runs": status_rows})
    write_json(output_root / "summary.json", {"runs": rows})
    write_csv(output_root / "summary.csv", rows)
    (output_root / "summary.md").write_text(_build_markdown_summary(rows), encoding="utf-8")

    return {
        "output_root": str(output_root),
        "manifest": manifest_payload,
        "status_rows": status_rows,
        "summary_rows": rows,
    }


def execute_benchmark_suite(
    resolved_specs: Iterable[BenchmarkRunSpec],
    *,
    corpus_root: str | None = None,
    results_root: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    resolved_specs = list(resolved_specs)
    executed: list[dict[str, Any]] = []
    for spec in resolved_specs:
        config = build_profile_config(
            spec.profile,
            corpus_root=corpus_root,
            results_root=results_root,
        )
        config = _apply_config_overrides(config, spec.config_overrides, seed=spec.seed)
        if dry_run:
            executed.append(
                {
                    "run_id": spec.run_id,
                    "status": "dry_run",
                    "profile": spec.profile,
                    "run_name": spec.run_name,
                    "max_files": spec.max_files,
                    "max_windows": spec.max_windows or {},
                    "resolved_config": config.to_dict(),
                }
            )
            continue

        result = run_small_transformer_experiment(
            config=config,
            run_name=spec.run_name,
            max_files=spec.max_files,
            max_windows_per_split=spec.max_windows or None,
        )
        executed.append(
            {
                "run_id": spec.run_id,
                "status": "completed",
                "profile": spec.profile,
                "run_name": spec.run_name,
                "result_dir": result["result_dir"],
            }
        )

    collected = collect_benchmark_suite(resolved_specs, results_root=results_root)
    collected["executed_runs"] = executed
    return collected
