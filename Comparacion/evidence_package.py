"""Construcción y verificación de paquetes públicos de evidencia experimental."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
from typing import Iterable, Mapping
import zipfile

from .artifact_audit import OPTIONAL_ARTIFACTS, REQUIRED_ARTIFACTS, audit_run


PACKAGE_MANIFEST = "package_manifest.json"
VERIFICATION_REPORT = "verification_report.json"

_EXTRA_FILES = (
    "canonicalization_audit.json",
    "denominator_audit.json",
    "finite_hmm_grid_audit.json",
    "hdp_chain_diagnostics.json",
    "training_exposure_audit.json",
)
_TEXT_EXTENSIONS = {".csv", ".json", ".jsonl", ".md", ".txt"}
_SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_PATH_PATTERN = re.compile(r"(?i)[a-z]:[\\/][^\s\"']+")
_POSIX_LOCAL_PATH_PATTERN = re.compile(r"/(?:home|users|tmp|var/tmp)/[^\s\"']+", re.IGNORECASE)
_PROHIBITED_PATTERNS = (
    re.compile(r"(?i)[a-z]:[\\/]"),
    re.compile(r"(?i)c:[\\/]users[\\/]"),
    re.compile(r"/(?:home|users|tmp|var/tmp)/", re.IGNORECASE),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_run_name(value: object) -> str:
    name = str(value).strip()
    if not _SAFE_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"invalid public run name: {value!r}")
    return name


def _safe_relative_path(value: object) -> str:
    text = str(value).strip()
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if (
        not text
        or windows.is_absolute()
        or posix.is_absolute()
        or ".." in windows.parts
        or ".." in posix.parts
        or ":" in text
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return posix.as_posix()


def _looks_absolute_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _sanitize_string(value: str) -> str:
    if _looks_absolute_path(value):
        return "<redacted:absolute-path>"
    sanitized = _WINDOWS_PATH_PATTERN.sub("<redacted:absolute-path>", value)
    return _POSIX_LOCAL_PATH_PATTERN.sub("<redacted:absolute-path>", sanitized)


def _sanitize_value(value: object) -> object:
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _write_sanitized(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        target.write_text(
            json.dumps(_sanitize_value(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return
    if suffix == ".jsonl":
        lines = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL in {source}:{line_number}") from error
            lines.append(json.dumps(_sanitize_value(payload), ensure_ascii=False))
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            fields = list(reader.fieldnames or [])
            rows = [
                {field: _sanitize_string(str(row.get(field, ""))) for field in fields}
                for row in reader
            ]
        with target.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return
    if suffix in {".md", ".txt"}:
        target.write_text(
            _sanitize_string(source.read_text(encoding="utf-8")), encoding="utf-8"
        )
        return
    shutil.copyfile(source, target)


def _candidate_files(run_dir: Path) -> Iterable[Path]:
    names = (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS, *_EXTRA_FILES)
    seen: set[Path] = set()
    for name in names:
        path = run_dir / name
        if path.is_file() and path not in seen:
            if path.is_symlink():
                raise ValueError(f"symbolic links are not allowed in evidence: {path}")
            seen.add(path)
            yield path
    for pattern in ("splits/*.json", "hdp_chain_trace_seed*.csv"):
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                if path.is_symlink():
                    raise ValueError(f"symbolic links are not allowed in evidence: {path}")
                seen.add(path)
                yield path


def _manifest_entries(root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {PACKAGE_MANIFEST, VERIFICATION_REPORT}:
            continue
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _write_deterministic_zip(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes())


def export_evidence_package(
    registry_path: str | Path,
    output_dir: str | Path,
    *,
    archive_path: str | Path | None = None,
) -> dict[str, object]:
    registry_file = Path(registry_path).resolve()
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    runs = registry.get("runs") if isinstance(registry, Mapping) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError("registry must contain a non-empty runs list")

    output = Path(output_dir).resolve()
    archive = Path(archive_path).resolve() if archive_path else output.with_suffix(".zip")
    if archive == output or output in archive.parents:
        raise ValueError("archive must be outside the evidence package directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evidence output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    public_runs: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for item in runs:
        if not isinstance(item, Mapping):
            raise ValueError("each run registry entry must be an object")
        name = _safe_run_name(item.get("name"))
        if name in seen_names:
            raise ValueError(f"duplicate public run name: {name}")
        seen_names.add(name)
        source = Path(str(item.get("path", "")))
        if not source.is_absolute():
            source = (registry_file.parent / source).resolve()
        audit = audit_run(source)
        if audit.get("status") != "passed":
            raise ValueError(f"run {name} did not pass artifact audit: {audit.get('status')}")

        target_run = output / "runs" / name
        for source_file in _candidate_files(source):
            relative = source_file.relative_to(source)
            _write_sanitized(source_file, target_run / relative)

        audit_target = output / "audits" / name / "artifact_audit.json"
        audit_target.parent.mkdir(parents=True, exist_ok=True)
        audit_target.write_text(
            json.dumps(_sanitize_value(audit), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        public_runs.append(
            {
                "name": name,
                "role": str(item.get("role", "unspecified")),
                "cost_usable": bool(item.get("cost_usable", True)),
            }
        )

    dictionary = {
        "purpose": "Evidence needed to verify the predictive comparison reported in the thesis.",
        "units": {
            "test_nll": "natural-log units per evaluated event",
            "test_ppl": "exp(test_nll)",
            "*_wall_clock_s": "seconds",
            "*_bytes": "bytes",
        },
        "scope": "Derived metrics and run metadata; no third-party scores or model weights.",
    }
    (output / "data_dictionary.json").write_text(
        json.dumps(dictionary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Paquete de evidencia de la comparación\n\n"
        "Contiene métricas derivadas, configuración y auditorías. No contiene "
        "partituras, caches del corpus ni pesos de modelos. Verifique el paquete "
        "antes de utilizarlo.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "runs": public_runs,
        "files": _manifest_entries(output),
    }
    (output / PACKAGE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    verification = verify_evidence_package(output)
    if verification["status"] != "passed":
        raise ValueError(f"exported package did not verify: {verification['issues']}")
    (output / VERIFICATION_REPORT).write_text(
        json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    _write_deterministic_zip(output, archive)
    return {
        "status": "passed",
        "output_dir": str(output),
        "archive_path": str(archive),
        "archive_sha256": _sha256(archive),
        "n_runs": len(public_runs),
        "n_files": len(manifest["files"]),
    }


def verify_evidence_package(package_dir: str | Path) -> dict[str, object]:
    root = Path(package_dir).resolve()
    issues: list[str] = []
    manifest_path = root / PACKAGE_MANIFEST
    if not manifest_path.is_file():
        return {"status": "failed", "issues": [f"missing {PACKAGE_MANIFEST}"]}

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"status": "failed", "issues": [f"invalid manifest: {error}"]}
    if not isinstance(manifest, Mapping):
        return {"status": "failed", "issues": ["manifest must contain an object"]}

    expected: dict[str, Mapping[str, object]] = {}
    file_entries = manifest.get("files", [])
    if not isinstance(file_entries, list):
        issues.append("manifest files must be a list")
        file_entries = []
    for entry in file_entries:
        if not isinstance(entry, Mapping) or "relative_path" not in entry:
            issues.append("invalid file entry in manifest")
            continue
        try:
            relative = _safe_relative_path(entry["relative_path"])
        except ValueError:
            issues.append(f"unsafe manifest path: {entry.get('relative_path')}")
            continue
        if relative in expected:
            issues.append(f"duplicate manifest path: {relative}")
            continue
        expected[relative] = entry
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or not runs:
        issues.append("manifest has no runs")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {PACKAGE_MANIFEST, VERIFICATION_REPORT}
    }
    for relative in sorted(set(expected) - set(actual)):
        issues.append(f"missing file: {relative}")
    for relative in sorted(set(actual) - set(expected)):
        issues.append(f"unexpected file: {relative}")
    for relative in sorted(set(expected) & set(actual)):
        entry = expected[relative]
        if _sha256(actual[relative]) != entry.get("sha256"):
            issues.append(f"hash mismatch: {relative}")
        if actual[relative].suffix.lower() in _TEXT_EXTENSIONS:
            text = actual[relative].read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in _PROHIBITED_PATTERNS):
                issues.append(f"absolute local path found: {relative}")

    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, Mapping):
            issues.append("invalid run entry in manifest")
            continue
        try:
            name = _safe_run_name(run.get("name"))
        except ValueError:
            issues.append(f"unsafe run name in manifest: {run.get('name')}")
            continue
        for required in ("config.json", "results_raw.csv", "protocol_audit.json"):
            if not (root / "runs" / name / required).is_file():
                issues.append(f"missing core run file: {name}/{required}")
        audit_path = root / "audits" / name / "artifact_audit.json"
        if not audit_path.is_file():
            issues.append(f"missing packaged audit: {name}")
            continue
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"invalid packaged audit: {name}")
            continue
        if audit.get("status") != "passed":
            issues.append(f"packaged audit did not pass: {name}")

    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "n_expected_files": len(expected),
        "n_actual_files": len(actual),
    }
