"""Escenarios monetarios reproducibles a partir de tiempos medidos."""

from __future__ import annotations

import csv
from datetime import date
import json
import math
from pathlib import Path
from statistics import mean
from typing import Mapping


def _finite_nonnegative(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _device_class(value: object) -> str:
    normalized = str(value).strip().lower()
    return "gpu" if any(token in normalized for token in ("cuda", "gpu", "mps")) else "cpu"


def _load_tariffs(path: Path) -> tuple[str, dict[str, dict[str, object]], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("tariff file must contain an object")
    currency = str(payload.get("currency", "")).strip()
    if not currency:
        raise ValueError("tariff file must declare currency")

    tariffs: dict[str, dict[str, object]] = {}
    tariff_items = payload.get("tariffs", [])
    if not isinstance(tariff_items, list):
        raise ValueError("tariffs must be a list")
    for item in tariff_items:
        if not isinstance(item, Mapping):
            raise ValueError("each tariff must be an object")
        device = str(item.get("device_class", "")).lower()
        if device not in {"cpu", "gpu"}:
            raise ValueError("device_class must be cpu or gpu")
        if device in tariffs:
            raise ValueError(f"duplicate tariff for {device}")
        rate = _finite_nonnegative(item.get("hourly_rate"))
        if rate is None:
            raise ValueError(f"invalid hourly_rate for {device}")
        source = str(item.get("source", "")).strip()
        observed_on = str(item.get("observed_on", "")).strip()
        if not source or not observed_on:
            raise ValueError(f"tariff {device} must declare source and observed_on")
        try:
            date.fromisoformat(observed_on)
        except ValueError as error:
            raise ValueError(f"tariff {device} observed_on must be YYYY-MM-DD") from error
        region = str(item.get("region", "")).strip()
        scope = str(item.get("scope", "")).strip()
        billing_unit = str(item.get("billing_unit", "")).strip()
        ownership_model = str(item.get("ownership_model", "")).strip()
        if not region:
            raise ValueError(f"tariff {device} must declare region")
        if not scope:
            raise ValueError(f"tariff {device} must declare scope")
        if billing_unit != "device_hour":
            raise ValueError(f"tariff {device} billing_unit must be device_hour")
        if ownership_model not in {"owned_equipment", "remote_service"}:
            raise ValueError(
                f"tariff {device} ownership_model must be owned_equipment or remote_service"
            )
        tariffs[device] = {
            "hourly_rate": rate,
            "source": source,
            "observed_on": observed_on,
            "region": region,
            "scope": scope,
            "billing_unit": billing_unit,
            "ownership_model": ownership_model,
        }
    if set(tariffs) != {"cpu", "gpu"}:
        raise ValueError("tariff file must provide one cpu and one gpu rate")

    scenarios = []
    scenario_items = payload.get("scenarios", [])
    if not isinstance(scenario_items, list):
        raise ValueError("scenarios must be a list")
    scenario_names: set[str] = set()
    for item in scenario_items:
        if not isinstance(item, Mapping):
            raise ValueError("each scenario must be an object")
        name = str(item.get("name", "")).strip()
        repetitions = _finite_nonnegative(item.get("evaluation_repetitions"))
        if not name or repetitions is None or int(repetitions) != repetitions:
            raise ValueError("scenario needs a name and integer evaluation_repetitions")
        if name in scenario_names:
            raise ValueError(f"duplicate scenario name: {name}")
        scenario_names.add(name)
        scenarios.append({"name": name, "evaluation_repetitions": int(repetitions)})
    if not scenarios:
        scenarios = [{"name": "selection_and_fit", "evaluation_repetitions": 0}]
    return currency, tariffs, scenarios


def _read_eligible_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = []
    for row in rows:
        fraction = _finite_nonnegative(row.get("frac"))
        if fraction is None or abs(fraction - 1.0) > 1e-9:
            continue
        condition = str(row.get("resource_measurement_condition", "unknown"))
        if condition != "isolated" or not _truthy(row.get("resource_cost_usable")):
            continue
        selection = _finite_nonnegative(row.get("selection_wall_clock_s"))
        evaluation = _finite_nonnegative(row.get("evaluation_wall_clock_s"))
        if selection is None or evaluation is None:
            continue
        eligible.append(
            {
                "model": str(row.get("model")),
                "device_class": _device_class(row.get("device")),
                "selection_wall_clock_s": selection,
                "evaluation_wall_clock_s": evaluation,
                "selection_peak_process_tree_rss_bytes": _finite_nonnegative(
                    row.get("selection_peak_process_tree_rss_bytes")
                ),
                "selection_peak_cuda_allocated_bytes": _finite_nonnegative(
                    row.get("selection_peak_cuda_allocated_bytes")
                ),
            }
        )
    return eligible


def build_cost_scenarios(
    engineering_costs_path: str | Path,
    tariff_path: str | Path,
) -> dict[str, object]:
    cost_file = Path(engineering_costs_path)
    tariff_file = Path(tariff_path)
    currency, tariffs, scenarios = _load_tariffs(tariff_file)
    rows = _read_eligible_rows(cost_file)
    if not rows:
        raise ValueError(
            "no isolated full-fraction rows with measured selection and evaluation times"
        )

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    observations = []
    for model, model_rows in sorted(grouped.items()):
        devices = {str(row["device_class"]) for row in model_rows}
        if len(devices) != 1:
            raise ValueError(f"model {model} mixes device classes: {sorted(devices)}")
        device = devices.pop()
        selection_s = mean(float(row["selection_wall_clock_s"]) for row in model_rows)
        evaluation_s = mean(float(row["evaluation_wall_clock_s"]) for row in model_rows)
        rate = float(tariffs[device]["hourly_rate"])
        scenario_costs = []
        for scenario in scenarios:
            repetitions = int(scenario["evaluation_repetitions"])
            total_seconds = selection_s + repetitions * evaluation_s
            scenario_costs.append(
                {
                    "name": scenario["name"],
                    "evaluation_repetitions": repetitions,
                    "total_wall_clock_s": total_seconds,
                    "estimated_cost": total_seconds / 3600.0 * rate,
                }
            )
        observations.append(
            {
                "model": model,
                "device_class": device,
                "n_rows": len(model_rows),
                "mean_selection_wall_clock_s": selection_s,
                "mean_evaluation_wall_clock_s": evaluation_s,
                "max_selection_process_tree_rss_bytes": max(
                    (
                        float(row["selection_peak_process_tree_rss_bytes"])
                        for row in model_rows
                        if row["selection_peak_process_tree_rss_bytes"] is not None
                    ),
                    default=None,
                ),
                "max_selection_cuda_allocated_bytes": max(
                    (
                        float(row["selection_peak_cuda_allocated_bytes"])
                        for row in model_rows
                        if row["selection_peak_cuda_allocated_bytes"] is not None
                    ),
                    default=None,
                ),
                "hourly_rate": rate,
                "scenario_costs": scenario_costs,
            }
        )

    break_even = []
    for scenario in scenarios:
        repetitions = int(scenario["evaluation_repetitions"])
        gpu_rows = [row for row in observations if row["device_class"] == "gpu"]
        cpu_rows = [row for row in observations if row["device_class"] == "cpu"]
        for gpu in gpu_rows:
            gpu_seconds = float(gpu["mean_selection_wall_clock_s"]) + repetitions * float(
                gpu["mean_evaluation_wall_clock_s"]
            )
            if gpu_seconds <= 0:
                continue
            for cpu in cpu_rows:
                cpu_seconds = float(cpu["mean_selection_wall_clock_s"]) + repetitions * float(
                    cpu["mean_evaluation_wall_clock_s"]
                )
                break_even.append(
                    {
                        "scenario": scenario["name"],
                        "gpu_model": gpu["model"],
                        "cpu_model": cpu["model"],
                        "max_gpu_to_cpu_hourly_rate_ratio_for_gpu_to_cost_less": (
                            cpu_seconds / gpu_seconds
                        ),
                    }
                )

    return {
        "status": "computed_from_isolated_measurements",
        "currency": currency,
        "tariffs": tariffs,
        "source_engineering_costs": cost_file.name,
        "observations": observations,
        "break_even_rate_ratios": break_even,
        "limitations": [
            "Tariffs are scenarios, not expenses observed during the original experiment.",
            "Development, storage, maintenance and hardware amortization are excluded unless represented in the tariff.",
            "Energy use and environmental impact are not estimated.",
        ],
    }


def write_cost_scenarios(
    engineering_costs_path: str | Path,
    tariff_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    report = build_cost_scenarios(engineering_costs_path, tariff_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
