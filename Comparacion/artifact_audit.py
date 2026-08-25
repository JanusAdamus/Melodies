"""Auditoría de sólo lectura de los artefactos de una corrida de comparación.

El objetivo es que cada cifra publicada pueda remontarse a un archivo primario
con su hash, o que la ausencia del archivo quede registrada sin sustitutos
fabricados. Este módulo nunca escribe dentro de la carpeta auditada.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

MANIFEST_FILENAME = "artifact_manifest.json"
AUDIT_FILENAME = "artifact_audit.json"

#: Artefactos que una corrida terminada debe contener.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "results_raw.csv",
    "results_summary.csv",
    "piece_metrics_raw.csv",
    "engineering_costs.csv",
    "pairwise_comparisons.json",
    "protocol_audit.json",
    "structural_evaluation.json",
    "pareto_summary.json",
    "config.json",
    "preprocessing_report.json",
    "exclusions.csv",
    "checkpoint.jsonl",
)

#: Artefactos deseables cuya ausencia se registra, pero no invalida la corrida.
OPTIONAL_ARTIFACTS: tuple[str, ...] = (
    "run_summary.json",
    "learning_curve.png",
    "hardware_manifest.json",
    "denominator_audit.json",
    "canonicalization_audit.json",
)

_JSON_ARTIFACTS: tuple[str, ...] = tuple(
    name for name in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS if name.endswith(".json")
)

_SUMMARY_TOLERANCE = 1e-6

_STATUS_ORDER = {"passed": 0, "incomplete": 1, "failed": 2}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(run_dir: str | Path) -> dict[str, object]:
    """Devuelve rutas relativas, tamaños y SHA-256 de todo lo que hay en la corrida."""

    root = Path(run_dir)
    files: list[dict[str, object]] = []
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _hash_file(path),
                }
            )
    return {"run_name": root.name, "n_files": len(files), "files": files}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: object) -> float | None:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _worst(statuses: Iterable[str]) -> str:
    return max(statuses, key=lambda status: _STATUS_ORDER[status], default="passed")


def _check_presence(root: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]
    return {
        "name": "artifacts_present",
        "status": "incomplete" if missing else "passed",
        "missing": missing,
        "expected": list(REQUIRED_ARTIFACTS),
    }


def _check_json(root: Path) -> dict[str, object]:
    unreadable: dict[str, str] = {}
    for name in _JSON_ARTIFACTS:
        path = root / name
        if not path.exists():
            continue
        try:
            _read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            unreadable[name] = str(error)
    return {
        "name": "json_parsable",
        "status": "failed" if unreadable else "passed",
        "unreadable": unreadable,
    }


def _expected_cells(config: Mapping[str, object] | None) -> int | None:
    if not config:
        return None
    seeds = config.get("data_seeds")
    fractions = config.get("train_fractions")
    if not isinstance(seeds, Sequence) or not isinstance(fractions, Sequence):
        return None
    return len(seeds) * len(fractions)


def _completed_cells(root: Path) -> tuple[int | None, list[str]]:
    path = root / "checkpoint.jsonl"
    if not path.exists():
        return None, []
    cells: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        cell = payload.get("cell")
        if isinstance(cell, str) and cell not in cells:
            cells.append(cell)
    return len(cells), cells


def _check_cells(root: Path, config: Mapping[str, object] | None) -> dict[str, object]:
    completed, cells = _completed_cells(root)
    expected = _expected_cells(config)
    protocol_status = None
    protocol_path = root / "protocol_audit.json"
    if protocol_path.exists():
        try:
            payload = _read_json(protocol_path)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            protocol_status = payload.get("status")

    status = "passed"
    if completed is None or expected is None:
        status = "incomplete"
    elif completed < expected:
        status = "incomplete"
    elif completed > expected:
        status = "failed"
    if protocol_status is not None and protocol_status != "passed":
        status = _worst([status, "incomplete"])

    return {
        "name": "cells_completed",
        "status": status,
        "completed_cells": completed,
        "expected_cells": expected,
        "cells": cells,
        "protocol_audit_status": protocol_status,
    }


def _summary_key(row: Mapping[str, str]) -> tuple[str, float] | None:
    fraction = _float(row.get("frac"))
    if fraction is None:
        return None
    return str(row.get("model")), fraction


def _check_summary(root: Path) -> dict[str, object]:
    raw_path = root / "results_raw.csv"
    summary_path = root / "results_summary.csv"
    if not raw_path.exists() or not summary_path.exists():
        return {
            "name": "summary_matches_raw",
            "status": "incomplete",
            "reason": "missing_inputs",
            "mismatches": [],
        }

    grouped: dict[tuple[str, float], list[float]] = {}
    for row in _read_csv(raw_path):
        key = _summary_key(row)
        value = _float(row.get("test_nll"))
        if key is None or value is None:
            continue
        grouped.setdefault(key, []).append(value)

    mismatches: list[dict[str, object]] = []
    seen: set[tuple[str, float]] = set()
    for row in _read_csv(summary_path):
        key = _summary_key(row)
        if key is None:
            continue
        seen.add(key)
        values = grouped.get(key)
        if not values:
            mismatches.append({"key": list(key), "reason": "no_raw_rows"})
            continue
        reported_mean = _float(row.get("mean_test_nll"))
        expected_mean = sum(values) / len(values)
        if reported_mean is None or abs(reported_mean - expected_mean) > _SUMMARY_TOLERANCE:
            mismatches.append(
                {
                    "key": list(key),
                    "reason": "mean_test_nll",
                    "reported": reported_mean,
                    "recomputed": expected_mean,
                }
            )
        reported_runs = row.get("runs", row.get("n_runs"))
        runs = _float(reported_runs)
        if runs is not None and int(runs) != len(values):
            mismatches.append(
                {
                    "key": list(key),
                    "reason": "runs",
                    "reported": int(runs),
                    "recomputed": len(values),
                }
            )

    for key in sorted(set(grouped) - seen):
        mismatches.append({"key": list(key), "reason": "missing_summary_row"})

    return {
        "name": "summary_matches_raw",
        "status": "failed" if mismatches else "passed",
        "mismatches": mismatches,
    }


def _check_config(root: Path, config: Mapping[str, object] | None) -> dict[str, object]:
    raw_path = root / "results_raw.csv"
    if config is None or not raw_path.exists():
        return {
            "name": "config_matches_rows",
            "status": "incomplete",
            "reason": "missing_inputs",
        }

    rows = _read_csv(raw_path)
    observed_seeds = {value for value in (_float(row.get("data_seed")) for row in rows) if value is not None}
    observed_fractions = {value for value in (_float(row.get("frac")) for row in rows) if value is not None}
    configured_seeds = {
        value
        for value in (_float(item) for item in config.get("data_seeds", []) or [])
        if value is not None
    }
    configured_fractions = {
        value
        for value in (_float(item) for item in config.get("train_fractions", []) or [])
        if value is not None
    }

    missing_seeds = sorted(configured_seeds - observed_seeds)
    missing_fractions = sorted(configured_fractions - observed_fractions)
    extra_seeds = sorted(observed_seeds - configured_seeds)
    extra_fractions = sorted(observed_fractions - configured_fractions)

    if extra_seeds or extra_fractions:
        status = "failed"
    elif missing_seeds or missing_fractions:
        status = "incomplete"
    else:
        status = "passed"

    return {
        "name": "config_matches_rows",
        "status": status,
        "missing_data_seeds": missing_seeds,
        "missing_fractions": missing_fractions,
        "unconfigured_data_seeds": extra_seeds,
        "unconfigured_fractions": extra_fractions,
    }


def _check_run_summary(root: Path) -> dict[str, object]:
    """Informativa: las corridas anteriores a esta auditoría no escribían run_summary.json."""

    path = root / "run_summary.json"
    if not path.exists():
        return {
            "name": "run_summary_present",
            "status": "incomplete",
            "reason": "run_summary_not_written_by_this_run",
            "advisory": True,
        }
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as error:
        return {
            "name": "run_summary_present",
            "status": "failed",
            "reason": str(error),
            "advisory": True,
        }

    if not isinstance(payload, Mapping):
        return {
            "name": "run_summary_present",
            "status": "failed",
            "reason": "not_a_mapping",
            "advisory": True,
        }

    artifacts = payload.get("artifacts", {})
    missing = [
        name
        for name, location in (artifacts.items() if isinstance(artifacts, Mapping) else [])
        if not (root / Path(str(location)).name).exists()
    ]
    return {
        "name": "run_summary_present",
        "status": "failed" if missing else "passed",
        "missing_artifacts": missing,
        "status_field": payload.get("status"),
        "advisory": True,
    }


def _counts(root: Path, checks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    raw_path = root / "results_raw.csv"
    rows = _read_csv(raw_path) if raw_path.exists() else []
    cells_check = next(check for check in checks if check["name"] == "cells_completed")
    return {
        "raw_rows": len(rows),
        "models": len({row.get("model") for row in rows}),
        "fractions": len({row.get("frac") for row in rows}),
        "data_seeds": len({row.get("data_seed") for row in rows}),
        "model_seeds": len({row.get("model_seed") for row in rows}),
        "completed_cells": cells_check["completed_cells"],
        "expected_cells": cells_check["expected_cells"],
    }


def audit_run(run_dir: str | Path) -> dict[str, object]:
    """Audita una corrida sin modificarla y devuelve el informe."""

    root = Path(run_dir)
    if not root.is_dir():
        return {
            "run_name": root.name,
            "run_dir": str(root),
            "status": "incomplete",
            "reason": "original_artifacts_not_found",
            "checks": [],
            "counts": {},
        }

    config: Mapping[str, object] | None = None
    config_path = root / "config.json"
    if config_path.exists():
        try:
            loaded = _read_json(config_path)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, Mapping):
            config = loaded

    checks = [
        _check_presence(root),
        _check_json(root),
        _check_cells(root, config),
        _check_summary(root),
        _check_config(root, config),
        _check_run_summary(root),
    ]
    return {
        "run_name": root.name,
        "run_dir": str(root),
        "status": _worst(
            str(check["status"]) for check in checks if not check.get("advisory", False)
        ),
        "checks": checks,
        "counts": _counts(root, checks),
    }


def write_audit_reports(run_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Escribe manifiesto y auditoría fuera de la corrida original."""

    root = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("los informes no pueden escribirse dentro de la corrida auditada")
    output.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(root)
    audit = audit_run(root)
    manifest_path = output / MANIFEST_FILENAME
    audit_path = output / AUDIT_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "status": audit["status"],
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
        "counts": audit["counts"],
    }
