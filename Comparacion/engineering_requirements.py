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
EXPECTED_BENCHMARK_MODELS = ("finite_hmm", "hdp_hmm", "transformer", "vomm")
EVIDENCE_PACKAGE_FILES = (
    "cache_reconstruction_verification.json",
    "economic_cost_scenario.json",
    "requirements_validation.json",
    "requirements_validation.md",
    "test_verification.json",
)
REQUIRED_PACKAGE_PATHS = (
    "source_run/config.json",
    "source_run/preprocessing_report.json",
    "source_run/results_raw.csv",
    "source_run/results_summary.csv",
    "source_run/piece_metrics_raw.csv",
    "source_run/pairwise_comparisons.json",
    "source_run/engineering_costs.csv",
    "source_run/protocol_audit.json",
    "source_audit/artifact_audit.json",
    "source_audit/denominator_audit.json",
    "resource_benchmark/resource_benchmark_raw.csv",
    "resource_benchmark/resource_benchmark_summary.csv",
    "resource_benchmark/resource_benchmark_environment.json",
    "resource_benchmark/resource_benchmark_config.json",
    "resource_benchmark/resource_benchmark_audit.json",
    "evidence/economic_cost_scenario.json",
    "evidence/requirements_validation.json",
    "evidence/requirements_validation.md",
    "evidence/test_verification.json",
    "REGENERATION.md",
)
_ABSOLUTE_PATH = re.compile(
    r"(?i)(?:^|[\s\"'=({\[,;])(?:[A-Z]:[\\/]|\\\\|/(?:home|users|mnt|media|volumes|tmp)(?:/|$))"
)
_CORPUS_PATH = re.compile(
    r"(?i)(?:^|[\s\"'=({\[,;])(?=[^\r\n,;\"']*[\\/])[^\r\n,;\"']*(?:pdmx|mxl)[^\r\n,;\"']*"
)


def _contains_forbidden_path(value: str) -> bool:
    return bool(_ABSOLUTE_PATH.search(value) or _CORPUS_PATH.search(value))


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


def _listed_files_are_valid(root: Path, payload: Mapping[str, object]) -> bool:
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return False
    resolved_root = root.resolve()
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            return False
        relative = item.get("relative_path")
        if not isinstance(relative, str) or relative in seen:
            return False
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return False
        target = root / relative_path
        try:
            resolved_target = target.resolve(strict=True)
        except OSError:
            return False
        if resolved_root not in resolved_target.parents or not resolved_target.is_file():
            return False
        if item.get("size_bytes") != resolved_target.stat().st_size:
            return False
        if item.get("sha256") != _hash_file(resolved_target):
            return False
        seen.add(relative)
    return True


def _package_manifest_is_valid(package: Path) -> bool:
    payload = _read_json(package / "package_manifest.json")
    listed_paths = {
        item.get("relative_path")
        for item in payload.get("files", [])
        if isinstance(item, Mapping)
    } if isinstance(payload, Mapping) else set()
    required_paths = set(REQUIRED_PACKAGE_PATHS) | {
        "source_run/splits/test_pieces.json",
        "source_run/splits/val_pieces.json",
    }
    train_splits = list((package / "source_run" / "splits").glob("train_fractions_seed*.json"))
    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and path.name not in {"package_manifest.json", "clean_clone_verification.json"}
    }
    forbidden_content = any(
        "corpus_cache" in path.name.lower()
        or path.suffix.lower() in {".mxl", ".musicxml"}
        or (
            path.suffix.lower() in {".json", ".csv", ".md", ".txt"}
            and _contains_forbidden_path(path.read_text(encoding="utf-8"))
        )
        for path in package.rglob("*")
        if path.is_file()
    )
    return bool(
        isinstance(payload, Mapping)
        and payload.get("status") == "passed"
        and payload.get("contains_corpus") is False
        and payload.get("missing_required_artifacts") == []
        and listed_paths == actual_paths
        and required_paths.issubset(listed_paths)
        and train_splits
        and all(path.relative_to(package).as_posix() in listed_paths for path in train_splits)
        and _listed_files_are_valid(package, payload)
        and _json_status(package / "source_audit" / "artifact_audit.json")
        and _json_status(package / "source_audit" / "denominator_audit.json", expected="ok")
        and _benchmark_audit_is_valid(package / "resource_benchmark")
        and _json_status(package / "evidence" / "test_verification.json")
        and _json_status(
            package / "evidence" / "economic_cost_scenario.json",
            expected="documented",
        )
        and not forbidden_content
    )


def _benchmark_audit_is_valid(benchmark: Path) -> bool:
    payload = _read_json(benchmark / "resource_benchmark_audit.json")
    if not isinstance(payload, Mapping):
        return False
    coverage = payload.get("coverage")
    files = payload.get("files")
    listed_paths = {
        item.get("relative_path")
        for item in files
        if isinstance(item, Mapping)
    } if isinstance(files, list) else set()
    expected_files = set(BENCHMARK_PACKAGE_FILES) - {"resource_benchmark_audit.json"}
    models = coverage.get("models") if isinstance(coverage, Mapping) else None
    repetitions = coverage.get("repetitions") if isinstance(coverage, Mapping) else None
    expected_rows = repetitions * len(EXPECTED_BENCHMARK_MODELS) if isinstance(repetitions, int) else None
    try:
        with (benchmark / "resource_benchmark_raw.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        observed_keys = [(row["model"], int(row["repetition"])) for row in rows]
        expected_keys = {
            (model, repetition)
            for model in EXPECTED_BENCHMARK_MODELS
            for repetition in range(1, int(repetitions) + 1)
        }
        rows_ok = (
            len(observed_keys) == len(set(observed_keys))
            and set(observed_keys) == expected_keys
            and all(
                row.get("peak_process_memory_status") == "measured"
                and int(row.get("peak_process_memory_bytes") or 0) > 0
                and row.get("peak_gpu_memory_status")
                == (
                    "measured"
                    if row.get("device", "").startswith("cuda")
                    else "not_applicable"
                )
                and (
                    not row.get("device", "").startswith("cuda")
                    or int(row.get("peak_gpu_memory_bytes") or 0) > 0
                )
                for row in rows
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        rows_ok = False
    return bool(
        payload.get("status") == "passed"
        and payload.get("coverage_status") == "passed"
        and payload.get("process_memory_status") == "passed"
        and payload.get("gpu_memory_status") == "passed"
        and isinstance(coverage, Mapping)
        and isinstance(repetitions, int)
        and repetitions > 0
        and models == {model: repetitions for model in EXPECTED_BENCHMARK_MODELS}
        and coverage.get("expected_rows") == expected_rows
        and coverage.get("expected_rows") == coverage.get("observed_rows")
        and expected_files.issubset(listed_paths)
        and _listed_files_are_valid(benchmark, payload)
        and rows_ok
    )


def _declared_evidence_exists(root: Path, requirement: Mapping[str, object]) -> bool:
    evidence = requirement.get("evidence")
    if evidence is None:
        return True
    if not isinstance(evidence, list) or not evidence:
        return False
    resolved_root = root.resolve()
    for item in evidence:
        if not isinstance(item, str):
            return False
        path = Path(item)
        if path.is_absolute() or ".." in path.parts:
            return False
        target = (root / path).resolve()
        if resolved_root != target and resolved_root not in target.parents:
            return False
        if not target.exists():
            return False
    return True


def _denominator_audit_is_valid(path: Path) -> bool:
    payload = _read_json(path)
    per_model = payload.get("per_model") if isinstance(payload, Mapping) else None
    return bool(
        isinstance(payload, Mapping)
        and payload.get("status") == "ok"
        and isinstance(payload.get("n_scored_files"), int)
        and payload["n_scored_files"] > 0
        and isinstance(payload.get("n_canonical_works"), int)
        and payload["n_canonical_works"] > 0
        and payload.get("unscored_test_files") == []
        and isinstance(per_model, Mapping)
        and set(EXPECTED_BENCHMARK_MODELS).issubset(per_model)
        and isinstance(payload.get("pairs"), list)
        and bool(payload.get("pairs"))
    )


def _protocol_audit_is_valid(path: Path) -> bool:
    payload = _read_json(path)
    evidence = payload.get("evidence") if isinstance(payload, Mapping) else None
    return bool(
        isinstance(payload, Mapping)
        and payload.get("status") == "passed"
        and payload.get("unexpected_piece_ids") == []
        and isinstance(evidence, list)
        and evidence
        and all(
            isinstance(item, Mapping)
            and item.get("status") == "passed"
            and item.get("count_mismatch") is False
            and item.get("order_mismatch") is False
            and item.get("expected_count") == item.get("scored_count")
            for item in evidence
        )
    )


def _selection_evidence_is_valid(source: Path) -> bool:
    config = _read_json(source / "config.json")
    try:
        with (source / "results_raw.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    required_columns = {"model", "frac", "data_seed", "model_seed", "hyperparams_json"}
    return bool(
        isinstance(config, Mapping)
        and isinstance(config.get("train_fractions"), list)
        and config.get("train_fractions")
        and isinstance(config.get("data_seeds"), list)
        and config.get("data_seeds")
        and isinstance(config.get("model_seeds"), list)
        and config.get("model_seeds")
        and rows
        and required_columns.issubset(rows[0])
    )


def _test_verification_is_valid(path: Path) -> bool:
    payload = _read_json(path)
    suites = payload.get("suites") if isinstance(payload, Mapping) else None
    return bool(
        isinstance(payload, Mapping)
        and payload.get("status") == "passed"
        and payload.get("diff_check") == "passed"
        and isinstance(payload.get("commit"), str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", payload["commit"])
        and isinstance(suites, list)
        and len(suites) >= 2
        and all(
            isinstance(suite, Mapping)
            and isinstance(suite.get("tests"), int)
            and suite["tests"] > 0
            and suite.get("failures") == 0
            and suite.get("errors") == 0
            for suite in suites
        )
    )


def _check_r5(root: Path) -> tuple[str, list[dict[str, object]]]:
    benchmark = root / "artifacts" / "resource_benchmark" / "final_fit_split7"
    audit_ok = _benchmark_audit_is_valid(benchmark)
    memory_ok = False
    raw_path = benchmark / "resource_benchmark_raw.csv"
    if raw_path.is_file():
        try:
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
        except (OSError, KeyError, TypeError, ValueError):
            memory_ok = False
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
            "passed": _package_manifest_is_valid(package),
        },
        {
            "name": "clean_clone_regeneration",
            "passed": _json_status(package / "clean_clone_verification.json"),
        },
    ]
    r4_status = "passed" if all(check["passed"] for check in r4_checks) else "partial"
    r5_status, r5_checks = _check_r5(root)
    denominator_ok = _denominator_audit_is_valid(audits / "denominator_audit.json")
    protocol_ok = _protocol_audit_is_valid(source / "protocol_audit.json")
    selection_ok = _selection_evidence_is_valid(source)
    tests_ok = _test_verification_is_valid(root / "artifacts" / "test_verification.json")
    defaults: dict[str, tuple[str, list[dict[str, object]]]] = {
        "R1": (
            "passed" if denominator_ok and (source / "splits").is_dir() else "partial",
            [{"name": "denominators_and_splits", "passed": denominator_ok and (source / "splits").is_dir()}],
        ),
        "R2": (
            "passed" if protocol_ok else "partial",
            [{"name": "protocol_coverage", "passed": protocol_ok}],
        ),
        "R3": (
            "passed" if selection_ok else "partial",
            [{"name": "selection_config_and_rows", "passed": selection_ok}],
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
            "passed" if tests_ok else "partial",
            [{"name": "test_verification", "passed": tests_ok}],
        ),
    }
    requirements = []
    for requirement in payload["requirements"]:
        identifier = str(requirement["id"])
        status, checks = defaults.get(identifier, ("partial", []))
        evidence_ok = _declared_evidence_exists(root, requirement)
        checks = [*checks, {"name": "declared_evidence_exists", "passed": evidence_ok}]
        if not evidence_ok:
            status = "partial"
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
    if isinstance(value, str) and _contains_forbidden_path(value):
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
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(
                ["<redacted-path>" if _contains_forbidden_path(cell) else cell for cell in row]
                for row in rows
            )
        return
    if source.suffix.lower() in {".md", ".txt"}:
        text = source.read_text(encoding="utf-8")
        if _contains_forbidden_path(text):
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
    for name in EVIDENCE_PACKAGE_FILES:
        path = artifact_root / name
        if path.is_file():
            _copy_safe(path, output / "evidence" / name)
    (output / "REGENERATION.md").write_text(
        "# Regeneration\n\n"
        "From a clean checkout, run:\n\n"
        "```powershell\n"
        "python -m pip install -r requirements.txt\n"
        "python -m unittest discover -s tests -v\n"
        "python -m unittest discover -s next_token_experiment/tests -v\n"
        "$expected = 'F42F9D7AB8550A4C366CFCF410C3CF67C85FAD46F5C4F54818403DEEC328E144'\n"
        "$actual = (Get-FileHash artifacts/corpus_cache_3000.jsonl -Algorithm SHA256).Hash\n"
        "if ($actual -ne $expected) { throw \"Unexpected corpus cache SHA-256: $actual\" }\n"
        "Copy-Item -Recurse package/source_run artifacts/Comparacion/tesis_3000_gpu_20260823_1941\n"
        "Copy-Item -Recurse package/source_audit artifacts/Comparacion/audits/tesis_3000_gpu_20260823_1941\n"
        "Copy-Item -Recurse package/sensitivities/* artifacts/Comparacion/\n"
        "python scripts/figuras_tesis.py\n"
        "```\n\n"
        "Run `scripts/run_resource_benchmark.py --help` for the final-fit benchmark "
        "arguments. The benchmark refuses to fit if the cache hash or corpus counts differ.\n",
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
    required_paths = list(REQUIRED_PACKAGE_PATHS)
    split_dir = output / "source_run" / "splits"
    if not (split_dir / "test_pieces.json").is_file():
        required_paths.append("source_run/splits/test_pieces.json")
    if not (split_dir / "val_pieces.json").is_file():
        required_paths.append("source_run/splits/val_pieces.json")
    if not list(split_dir.glob("train_fractions_seed*.json")):
        required_paths.append("source_run/splits/train_fractions_seed*.json")
    for sensitivity in sorted(source.parent.glob("sens_*")):
        if sensitivity.is_dir():
            for suffix in ("config.json", "results_summary.csv", "audit/artifact_audit.json"):
                required_paths.append(f"sensitivities/{sensitivity.name}/{suffix}")
    missing_required = sorted(
        relative for relative in set(required_paths) if not (output / relative).is_file()
    )
    required_statuses_ok = (
        _json_status(output / "source_audit" / "artifact_audit.json")
        and _json_status(output / "source_audit" / "denominator_audit.json", expected="ok")
        and _benchmark_audit_is_valid(output / "resource_benchmark")
        and _json_status(output / "evidence" / "test_verification.json")
        and _json_status(
            output / "evidence" / "economic_cost_scenario.json",
            expected="documented",
        )
    )
    manifest = {
        "status": "passed" if not missing_required and required_statuses_ok else "partial",
        "contains_corpus": False,
        "missing_benchmark_artifacts": missing_benchmark,
        "missing_required_artifacts": missing_required,
        "required_statuses_ok": required_statuses_ok,
        "files": files,
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
