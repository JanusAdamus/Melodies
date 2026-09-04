from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Mapping


SOURCE_PACKAGE_FILES = (
    "config.json",
    "preprocessing_report.json",
    "exclusions.csv",
    "results_raw.csv",
    "results_summary.csv",
    "piece_metrics_raw.csv",
    "pairwise_comparisons.json",
    "engineering_costs.csv",
    "protocol_audit.json",
    "structural_evaluation.json",
    "pareto_summary.json",
    "finite_hmm_grid_audit.json",
    "hardware_manifest.json",
)
SENSITIVITY_PACKAGE_FILES = (
    "config.json",
    "results_raw.csv",
    "results_summary.csv",
    "pairwise_comparisons.json",
    "engineering_costs.csv",
    "protocol_audit.json",
    "finite_hmm_grid_audit.json",
    "hdp_chain_diagnostics.json",
)
BENCHMARK_PACKAGE_FILES = (
    "resource_benchmark_raw.csv",
    "resource_benchmark_summary.csv",
    "resource_benchmark_environment.json",
    "resource_benchmark_config.json",
    "resource_benchmark_audit.json",
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _read_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _json_status(path: Path, expected: str = "passed") -> bool:
    payload = _read_json(path)
    return isinstance(payload, Mapping) and payload.get("status") == expected


def _check_r5(root: Path) -> tuple[str, list[dict[str, object]]]:
    benchmark = root / "artifacts" / "resource_benchmark" / "final_fit_split7"
    audit_ok = _json_status(benchmark / "resource_benchmark_audit.json")
    memory_ok = False
    raw_path = benchmark / "resource_benchmark_raw.csv"
    if raw_path.is_file():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            "peak_process_memory_bytes",
            "peak_process_memory_status",
        }
        memory_ok = bool(rows) and required.issubset(rows[0]) and all(
            row["peak_process_memory_status"] == "measured"
            and int(row["peak_process_memory_bytes"]) > 0
            for row in rows
        )
    scenario = _read_json(root / "artifacts" / "economic_cost_scenario.json")
    scenario_ok = (
        isinstance(scenario, Mapping)
        and scenario.get("status") == "documented"
        and bool(scenario.get("as_of_date"))
        and bool(scenario.get("rates"))
    )
    checks = [
        {"name": "final_fit_benchmark_audit", "passed": audit_ok},
        {"name": "peak_process_memory", "passed": memory_ok},
        {"name": "dated_economic_scenario", "passed": scenario_ok},
    ]
    return ("passed" if all(check["passed"] for check in checks) else "partial"), checks


def validate_engineering_requirements(
    repo_root: str | Path,
    *,
    requirements_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repo_root)
    spec_path = Path(requirements_path) if requirements_path else root / "docs" / "engineering-requirements.json"
    payload = _read_json(spec_path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("requirements"), list):
        raise ValueError(f"invalid requirements file: {spec_path}")

    source = root / "artifacts" / "Comparacion" / "tesis_3000_gpu_20260823_1941"
    audits = root / "artifacts" / "Comparacion" / "audits" / source.name
    package = root / "artifacts" / "reproducibility" / "r4-r5-evidence"
    r4_checks = [
        {
            "name": "reproducibility_package_manifest",
            "passed": _json_status(package / "package_manifest.json"),
        },
        {
            "name": "clean_clone_regeneration",
            "passed": _json_status(package / "clean_clone_verification.json"),
        },
    ]
    r4_status = "passed" if all(check["passed"] for check in r4_checks) else "partial"
    r5_status, r5_checks = _check_r5(root)
    defaults: dict[str, tuple[str, list[dict[str, object]]]] = {
        "R1": (
            "passed" if _json_status(audits / "denominator_audit.json") and (source / "splits").is_dir() else "partial",
            [{"name": "denominators_and_splits", "passed": _json_status(audits / "denominator_audit.json") and (source / "splits").is_dir()}],
        ),
        "R2": (
            "passed" if _json_status(source / "protocol_audit.json") else "partial",
            [{"name": "protocol_coverage", "passed": _json_status(source / "protocol_audit.json")}],
        ),
        "R3": (
            "passed" if (source / "config.json").is_file() and (source / "results_raw.csv").is_file() else "partial",
            [{"name": "selection_config_and_rows", "passed": (source / "config.json").is_file() and (source / "results_raw.csv").is_file()}],
        ),
        "R4": (r4_status, r4_checks),
        "R5": (r5_status, r5_checks),
        "R6": (
            "not_evaluated"
            if isinstance(_read_json(source / "structural_evaluation.json"), Mapping)
            and _read_json(source / "structural_evaluation.json").get("status") == "not_evaluated"
            else "partial",
            [{"name": "structural_status_declared", "passed": (source / "structural_evaluation.json").is_file()}],
        ),
        "R7": (
            "passed" if _json_status(root / "artifacts" / "test_verification.json") else "partial",
            [{"name": "test_verification", "passed": _json_status(root / "artifacts" / "test_verification.json")}],
        ),
    }
    requirements = []
    for requirement in payload["requirements"]:
        identifier = str(requirement["id"])
        status, checks = defaults.get(identifier, ("partial", []))
        requirements.append({**dict(requirement), "status": status, "checks": checks})
    return {
        "status": "passed" if all(item["status"] in {"passed", "not_evaluated"} for item in requirements) else "partial",
        "requirements": requirements,
    }


def write_validation_reports(
    repo_root: str | Path,
    *,
    requirements_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repo_root)
    result = validate_engineering_requirements(root, requirements_path=requirements_path)
    output_json = root / "artifacts" / "requirements_validation.json"
    output_md = root / "artifacts" / "requirements_validation.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Engineering requirements validation",
        "",
        f"Overall status: **{result['status']}**",
        "",
        "| Requirement | Status | Description |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {item['id']} | {item['status']} | {item.get('description', '')} |"
        for item in result["requirements"]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, str) and _WINDOWS_ABSOLUTE.match(value):
        return "<redacted-absolute-path>"
    return value


def _copy_safe(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".json":
        payload = _read_json(source)
        if payload is None:
            raise ValueError(f"invalid JSON artifact: {source}")
        target.write_text(
            json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return
    if source.suffix.lower() in {".csv", ".md", ".txt"}:
        text = source.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:\\(?:Users|Melodies)\\", text, flags=re.IGNORECASE):
            raise ValueError(f"personal absolute path in artifact: {source}")
        target.write_text(text, encoding="utf-8")
        return
    shutil.copy2(source, target)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_reproducibility_package(
    *,
    source_run: str | Path,
    benchmark_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    source = Path(source_run)
    benchmark = Path(benchmark_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"package output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_PACKAGE_FILES:
        path = source / name
        if path.is_file():
            _copy_safe(path, output / "source_run" / name)
    split_root = source / "splits"
    if split_root.is_dir():
        for path in sorted(split_root.glob("*.json")):
            _copy_safe(path, output / "source_run" / "splits" / path.name)
    if benchmark.is_dir():
        for name in BENCHMARK_PACKAGE_FILES:
            path = benchmark / name
            if path.is_file():
                _copy_safe(path, output / "resource_benchmark" / path.name)
    audit_root = source.parent / "audits" / source.name
    if audit_root.is_dir():
        for path in sorted(audit_root.glob("*.json")):
            _copy_safe(path, output / "source_audit" / path.name)
    for sensitivity in sorted(source.parent.glob("sens_*")):
        if not sensitivity.is_dir():
            continue
        for name in SENSITIVITY_PACKAGE_FILES:
            path = sensitivity / name
            if path.is_file():
                _copy_safe(path, output / "sensitivities" / sensitivity.name / name)
        sensitivity_audit = source.parent / "audits" / sensitivity.name
        if sensitivity_audit.is_dir():
            for path in sorted(sensitivity_audit.glob("*.json")):
                _copy_safe(
                    path,
                    output / "sensitivities" / sensitivity.name / "audit" / path.name,
                )
    artifact_root = source.parent.parent
    for path in sorted(artifact_root.glob("diagnostico_*.json")):
        _copy_safe(path, output / "diagnostics" / path.name)
    (output / "REGENERATION.md").write_text(
        "# Regeneration\n\n"
        "Run both unittest suites, verify the documented corpus-cache SHA-256, "
        "then regenerate tables and figures with `scripts/figuras_tesis.py`.\n",
        encoding="utf-8",
    )
    files = [
        {
            "relative_path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _hash_file(path),
        }
        for path in sorted(item for item in output.rglob("*") if item.is_file())
    ]
    missing_benchmark = [
        name
        for name in BENCHMARK_PACKAGE_FILES
        if not (output / "resource_benchmark" / name).is_file()
    ]
    manifest = {
        "status": "partial" if missing_benchmark else "passed",
        "contains_corpus": False,
        "missing_benchmark_artifacts": missing_benchmark,
        "files": files,
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
