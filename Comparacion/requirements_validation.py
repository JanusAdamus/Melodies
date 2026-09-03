"""Validación reproducible de requisitos de ingeniería contra evidencia local."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from .evidence_package import verify_evidence_package


_STATUS_RANK = {"passed": 0, "partial": 1, "failed": 2}


def _resolve(context_file: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (context_file.parent / path).resolve()


def _field(payload: object, dotted: str) -> object:
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _package_root(context: Mapping[str, object], context_file: Path) -> Path | None:
    value = context.get("evidence_package")
    return _resolve(context_file, value) if value else None


def _check_rule(
    rule: Mapping[str, object],
    context: Mapping[str, object],
    context_file: Path,
) -> dict[str, object]:
    kind = str(rule.get("type", ""))
    missing_status = str(rule.get("missing_status", "partial"))
    if missing_status not in _STATUS_RANK:
        raise ValueError(f"invalid missing_status: {missing_status}")

    if kind == "context_file_exists":
        key = str(rule["context_key"])
        value = context.get(key)
        if not value:
            return {"status": missing_status, "reason": f"missing context key {key}"}
        path = _resolve(context_file, value)
        return {
            "status": "passed" if path.is_file() else missing_status,
            "evidence": str(path),
            "reason": None if path.is_file() else "file not found",
        }

    if kind == "package_verified":
        root = _package_root(context, context_file)
        if root is None or not root.is_dir():
            return {"status": missing_status, "reason": "evidence package not found"}
        report = verify_evidence_package(root)
        return {
            "status": "passed" if report["status"] == "passed" else "failed",
            "evidence": str(root),
            "details": report,
        }

    if kind in {"files_in_each_run", "json_field_in_each_run"}:
        root = _package_root(context, context_file)
        if root is None or not (root / "package_manifest.json").is_file():
            return {"status": missing_status, "reason": "evidence package not found"}
        manifest = json.loads((root / "package_manifest.json").read_text(encoding="utf-8"))
        run_names = [str(item["name"]) for item in manifest.get("runs", [])]
        if not run_names:
            return {"status": "failed", "reason": "evidence package has no runs"}
        relative = str(rule["relative_path"])
        missing = []
        mismatches = []
        for name in run_names:
            path = root / "runs" / name / relative
            if not path.is_file():
                missing.append(name)
                continue
            if kind == "json_field_in_each_run":
                payload = json.loads(path.read_text(encoding="utf-8"))
                actual = _field(payload, str(rule["field"]))
                allowed = rule.get("allowed", [rule.get("expected")])
                if actual not in allowed:
                    mismatches.append({"run": name, "actual": actual})
        if mismatches:
            status = "failed"
        elif missing:
            status = missing_status
        else:
            status = "passed"
        return {
            "status": status,
            "evidence": relative,
            "missing_runs": missing,
            "mismatches": mismatches,
        }

    if kind == "test_report_passed":
        value = context.get("test_report")
        if not value:
            return {"status": missing_status, "reason": "test_report not declared"}
        path = _resolve(context_file, value)
        if not path.is_file():
            return {"status": missing_status, "reason": "test_report not found"}
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing_tests = [
            fragment
            for fragment in rule.get("required_test_fragments", [])
            if not any(fragment in test_id for test_id in payload.get("test_ids", []))
        ]
        tests_run = payload.get("tests_run")
        test_ids = payload.get("test_ids")
        if (
            payload.get("status") != "passed"
            or not isinstance(tests_run, int)
            or tests_run <= 0
            or not isinstance(test_ids, list)
            or not test_ids
        ):
            status = "failed"
        elif missing_tests:
            status = missing_status
        else:
            status = "passed"
        return {
            "status": status,
            "evidence": str(path),
            "tests_run": tests_run,
            "missing_test_fragments": missing_tests,
        }

    if kind == "resource_costs_complete":
        value = context.get("resource_costs")
        if not value:
            return {"status": missing_status, "reason": "resource_costs not declared"}
        path = _resolve(context_file, value)
        if not path.is_file():
            return {"status": missing_status, "reason": "resource_costs not found"}
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        eligible = [
            row
            for row in rows
            if str(row.get("frac")) in {"1", "1.0"}
            and str(row.get("resource_measurement_condition")) == "isolated"
            and str(row.get("resource_cost_usable", "")).lower() in {"true", "1"}
        ]
        models = {str(row.get("model")) for row in eligible}
        required_models = set(rule.get("required_models", []))
        incomplete = []
        for row in eligible:
            required_fields = [
                "selection_wall_clock_s",
                "evaluation_wall_clock_s",
                "selection_peak_process_tree_rss_bytes",
                "evaluation_peak_process_tree_rss_bytes",
            ]
            device = str(row.get("device", "")).lower()
            if "cuda" in device or "gpu" in device:
                required_fields.extend(
                    [
                        "selection_peak_cuda_allocated_bytes",
                        "evaluation_peak_cuda_allocated_bytes",
                    ]
                )
            if any(not str(row.get(field, "")).strip() for field in required_fields):
                incomplete.append(str(row.get("model")))
        status = (
            "passed"
            if required_models.issubset(models) and not incomplete
            else missing_status
        )
        return {
            "status": status,
            "evidence": str(path),
            "eligible_models": sorted(models),
            "missing_models": sorted(required_models - models),
            "incomplete_models": sorted(set(incomplete)),
        }

    if kind == "cost_scenarios_computed":
        value = context.get("cost_scenarios")
        if not value:
            return {"status": missing_status, "reason": "cost_scenarios not declared"}
        path = _resolve(context_file, value)
        if not path.is_file():
            return {"status": missing_status, "reason": "cost_scenarios not found"}
        payload = json.loads(path.read_text(encoding="utf-8"))
        complete = (
            payload.get("status") == "computed_from_isolated_measurements"
            and bool(payload.get("observations"))
            and set(payload.get("tariffs", {})) == {"cpu", "gpu"}
        )
        return {
            "status": "passed" if complete else "failed",
            "evidence": str(path),
        }

    raise ValueError(f"unsupported requirement rule: {kind}")


def _worst(statuses: list[str]) -> str:
    return max(statuses, key=lambda item: _STATUS_RANK[item], default="partial")


def validate_requirements(
    requirements_path: str | Path,
    context_path: str | Path,
) -> dict[str, object]:
    requirements_file = Path(requirements_path).resolve()
    context_file = Path(context_path).resolve()
    specification = json.loads(requirements_file.read_text(encoding="utf-8"))
    context = json.loads(context_file.read_text(encoding="utf-8"))
    if not isinstance(specification, Mapping):
        raise ValueError("requirement register must be an object")
    if not isinstance(context, Mapping):
        raise ValueError("validation context must be an object")

    requirements = specification.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("requirement register must contain non-empty requirements")

    results = []
    seen_ids: set[str] = set()
    for item in requirements:
        if not isinstance(item, Mapping):
            raise ValueError("each requirement must be an object")
        requirement_id = str(item.get("id", "")).strip()
        if not requirement_id:
            raise ValueError("each requirement must declare an id")
        if requirement_id in seen_ids:
            raise ValueError(f"duplicate requirement id: {requirement_id}")
        seen_ids.add(requirement_id)
        rules = item.get("checks")
        if not isinstance(rules, list) or not rules:
            raise ValueError(f"requirement {requirement_id} must declare checks")
        if not all(isinstance(rule, Mapping) for rule in rules):
            raise ValueError(f"requirement {requirement_id} has an invalid check")
        checks = [
            _check_rule(rule, context, context_file) for rule in rules
        ]
        status = _worst([str(check["status"]) for check in checks])
        ceiling = item.get("status_ceiling")
        if ceiling in _STATUS_RANK and _STATUS_RANK[str(ceiling)] > _STATUS_RANK[status]:
            status = str(ceiling)
        results.append(
            {
                "id": requirement_id,
                "type": item.get("type", "unspecified"),
                "statement": item["statement"],
                "decision": item.get("decision"),
                "status": status,
                "checks": checks,
            }
        )

    counts = {status: sum(row["status"] == status for row in results) for status in _STATUS_RANK}
    overall = "failed" if counts["failed"] else ("partial" if counts["partial"] else "passed")
    return {
        "schema_version": 1,
        "status": overall,
        "counts": counts,
        "requirements": results,
    }


def write_requirement_validation(
    requirements_path: str | Path,
    context_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    report = validate_requirements(requirements_path, context_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "validation_matrix.json"
    md_path = output / "validation_matrix.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Matriz de validación de requisitos",
        "",
        "| ID | Tipo | Requisito | Estado |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["requirements"]:
        statement = str(row["statement"]).replace("|", "\\|")
        lines.append(f"| {row['id']} | {row['type']} | {statement} | {row['status']} |")
    lines.extend(
        [
            "",
            "La matriz se genera desde evidencia estructurada. Un estado parcial no equivale a cumplimiento.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        **report,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }
