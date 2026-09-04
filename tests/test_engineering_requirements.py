from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.engineering_requirements import (
    _artifact_audit_is_valid,
    _clean_clone_verification_is_valid,
    build_reproducibility_package,
    validate_engineering_requirements,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest_items(root: Path) -> list[dict[str, object]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in {"package_manifest.json", "clean_clone_verification.json"}
    ]


class EngineeringRequirementTests(unittest.TestCase):
    def test_status_only_audits_and_clean_clone_records_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit = root / "artifact_audit.json"
            clean_clone = root / "clean_clone_verification.json"
            _json(audit, {"status": "passed"})
            _json(clean_clone, {"status": "passed"})

            self.assertFalse(_artifact_audit_is_valid(audit))
            self.assertFalse(_clean_clone_verification_is_valid(clean_clone))

    def test_manifest_hash_mismatch_keeps_r4_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {"requirements": [{"id": "R4", "description": "Reproducibility"}]},
            )
            package = root / "artifacts" / "reproducibility" / "r4-r5-evidence"
            payload = package / "source_run" / "config.json"
            _json(payload, {"value": 1})
            _json(
                package / "package_manifest.json",
                {
                    "status": "passed",
                    "contains_corpus": False,
                    "missing_required_artifacts": [],
                    "files": [
                        {
                            "relative_path": "source_run/config.json",
                            "size_bytes": payload.stat().st_size,
                            "sha256": "0" * 64,
                        }
                    ],
                },
            )
            _json(package / "clean_clone_verification.json", {"status": "passed"})

            result = validate_engineering_requirements(root)

            self.assertEqual(result["requirements"][0]["status"], "partial")

    def test_declared_evidence_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {
                    "requirements": [
                        {
                            "id": "R7",
                            "description": "Tests",
                            "evidence": ["artifacts/does-not-exist.json"],
                        }
                    ]
                },
            )
            _json(root / "artifacts" / "test_verification.json", {"status": "passed"})

            result = validate_engineering_requirements(root)

            self.assertEqual(result["requirements"][0]["status"], "partial")

    def test_benchmark_audit_hash_mismatch_keeps_r5_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {"requirements": [{"id": "R5", "description": "Cost"}]},
            )
            benchmark = root / "artifacts" / "resource_benchmark" / "final_fit_split7"
            raw = benchmark / "resource_benchmark_raw.csv"
            raw.parent.mkdir(parents=True)
            raw.write_text(
                "model,peak_process_memory_bytes,peak_process_memory_status\n"
                "finite_hmm,1000,measured\n",
                encoding="utf-8",
            )
            _json(
                benchmark / "resource_benchmark_audit.json",
                {
                    "status": "passed",
                    "coverage_status": "passed",
                    "process_memory_status": "passed",
                    "gpu_memory_status": "passed",
                    "coverage": {"expected_rows": 1, "observed_rows": 1},
                    "files": [
                        {
                            "relative_path": raw.name,
                            "size_bytes": raw.stat().st_size,
                            "sha256": "0" * 64,
                        }
                    ],
                },
            )
            _json(
                root / "artifacts" / "economic_cost_scenario.json",
                {"status": "documented", "as_of_date": "2026-09-03", "rates": [{}]},
            )

            result = validate_engineering_requirements(root)

            self.assertEqual(result["requirements"][0]["status"], "partial")

    def test_denominator_audit_ok_status_satisfies_r1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {"requirements": [{"id": "R1", "description": "Denominators"}]},
            )
            source = (
                root
                / "artifacts"
                / "Comparacion"
                / "tesis_3000_gpu_20260823_1941"
            )
            (source / "splits").mkdir(parents=True)
            _json(
                root
                / "artifacts"
                / "Comparacion"
                / "audits"
                / source.name
                / "denominator_audit.json",
                {
                    "status": "ok",
                    "n_scored_files": 1,
                    "n_canonical_works": 1,
                    "unscored_test_files": [],
                    "per_model": {
                        "finite_hmm": {},
                        "hdp_hmm": {},
                        "transformer": {},
                        "vomm": {},
                    },
                    "pairs": [{}],
                },
            )

            result = validate_engineering_requirements(root)

            self.assertEqual(result["requirements"][0]["status"], "passed")

    def test_status_only_test_verification_does_not_satisfy_r7(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {
                    "requirements": [
                        {
                            "id": "R7",
                            "description": "Tests",
                            "evidence": ["artifacts/test_verification.json"],
                        }
                    ]
                },
            )
            _json(root / "artifacts" / "test_verification.json", {"status": "passed"})

            result = validate_engineering_requirements(root)

            self.assertEqual(result["requirements"][0]["status"], "partial")

    def test_package_without_complete_benchmark_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "results_raw.csv").write_text("model\nfinite_hmm\n", encoding="utf-8")

            manifest = build_reproducibility_package(
                source_run=source,
                benchmark_dir=root / "missing-benchmark",
                output_dir=root / "package",
            )

            self.assertEqual(manifest["status"], "partial")

    def test_package_with_benchmark_but_missing_source_evidence_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            benchmark = root / "benchmark"
            source.mkdir()
            benchmark.mkdir()
            (source / "results_raw.csv").write_text("model\nfinite_hmm\n", encoding="utf-8")
            for name in (
                "resource_benchmark_raw.csv",
                "resource_benchmark_summary.csv",
            ):
                (benchmark / name).write_text("model\nfinite_hmm\n", encoding="utf-8")
            for name in (
                "resource_benchmark_environment.json",
                "resource_benchmark_config.json",
                "resource_benchmark_audit.json",
            ):
                _json(benchmark / name, {"status": "passed"})

            manifest = build_reproducibility_package(
                source_run=source,
                benchmark_dir=benchmark,
                output_dir=root / "package",
            )

            self.assertEqual(manifest["status"], "partial")
            self.assertTrue(manifest["missing_required_artifacts"])

    def test_missing_package_and_benchmark_keep_r4_r5_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {
                    "requirements": [
                        {"id": f"R{number}", "description": f"Requirement {number}"}
                        for number in range(1, 8)
                    ]
                },
            )

            result = validate_engineering_requirements(root)

            statuses = {item["id"]: item["status"] for item in result["requirements"]}
            self.assertEqual(statuses["R4"], "partial")
            self.assertEqual(statuses["R5"], "partial")
            self.assertNotEqual(result["status"], "passed")

    def test_complete_r4_r5_evidence_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _json(
                root / "docs" / "engineering-requirements.json",
                {
                    "requirements": [
                        {"id": f"R{number}", "description": f"Requirement {number}"}
                        for number in range(1, 8)
                    ]
                },
            )
            package = root / "artifacts" / "reproducibility" / "r4-r5-evidence"
            for relative in (
                "source_run/config.json",
                "source_run/preprocessing_report.json",
                "source_run/results_raw.csv",
                "source_run/results_summary.csv",
                "source_run/piece_metrics_raw.csv",
                "source_run/pairwise_comparisons.json",
                "source_run/engineering_costs.csv",
                "source_run/protocol_audit.json",
                "source_run/hardware_manifest.json",
                "source_run/splits/test_pieces.json",
                "source_run/splits/val_pieces.json",
                "source_run/splits/train_fractions_seed1.json",
                "resource_benchmark/resource_benchmark_summary.csv",
                "resource_benchmark/resource_benchmark_environment.json",
                "resource_benchmark/resource_benchmark_config.json",
                "evidence/requirements_validation.json",
                "evidence/requirements_validation.md",
                "REGENERATION.md",
            ):
                path = package / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}" if path.suffix == ".json" else "value\n", encoding="utf-8")
            _json(
                package / "source_audit" / "artifact_audit.json",
                {
                    "status": "passed",
                    "checks": [{"status": "passed"}],
                    "counts": {
                        "raw_rows": 1,
                        "models": 4,
                        "completed_cells": 1,
                        "expected_cells": 1,
                    },
                },
            )
            _json(
                package / "source_audit" / "denominator_audit.json",
                {
                    "status": "ok",
                    "n_scored_files": 1,
                    "n_canonical_works": 1,
                    "unscored_test_files": [],
                    "per_model": {model: {} for model in ("finite_hmm", "hdp_hmm", "transformer", "vomm")},
                    "pairs": [{}],
                },
            )
            _json(
                package / "evidence" / "test_verification.json",
                {
                    "status": "passed",
                    "commit": "a" * 40,
                    "diff_check": "passed",
                    "suites": [
                        {"tests": 1, "failures": 0, "errors": 0},
                        {"tests": 1, "failures": 0, "errors": 0},
                    ],
                },
            )
            _json(
                package / "evidence" / "economic_cost_scenario.json",
                {
                    "status": "documented",
                    "as_of_date": "2026-09-03",
                    "rates": [{"usd_per_hour": 1.0, "source": "https://example.com"}],
                },
            )
            packaged_raw = package / "resource_benchmark" / "resource_benchmark_raw.csv"
            packaged_raw.write_text(
                "model,repetition,peak_process_memory_bytes,peak_process_memory_status,"
                "peak_gpu_memory_bytes,peak_gpu_memory_status,device\n"
                "finite_hmm,1,1000,measured,,not_applicable,cpu\n"
                "hdp_hmm,1,1000,measured,,not_applicable,cpu\n"
                "transformer,1,1000,measured,2000,measured,cuda\n"
                "vomm,1,1000,measured,,not_applicable,cpu\n",
                encoding="utf-8",
            )
            benchmark_files = [
                path
                for path in (package / "resource_benchmark").iterdir()
                if path.name != "resource_benchmark_audit.json"
            ]
            _json(
                package / "resource_benchmark" / "resource_benchmark_audit.json",
                {
                    "status": "passed",
                    "coverage_status": "passed",
                    "process_memory_status": "passed",
                    "gpu_memory_status": "passed",
                    "coverage": {
                        "models": {
                            "finite_hmm": 1,
                            "hdp_hmm": 1,
                            "transformer": 1,
                            "vomm": 1,
                        },
                        "repetitions": 1,
                        "expected_rows": 4,
                        "observed_rows": 4,
                    },
                    "files": [
                        {
                            "relative_path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                        }
                        for path in benchmark_files
                    ],
                },
            )
            _json(
                package / "clean_clone_verification.json",
                {
                    "status": "passed",
                    "commit": "a" * 40,
                    "corpus_cache_sha256": "F42F9D7AB8550A4C366CFCF410C3CF67C85FAD46F5C4F54818403DEEC328E144",
                    "corpus_counts": {
                        "entries": 3000,
                        "prepared_pieces": 2933,
                        "exclusions": 67,
                        "events": 693754,
                    },
                    "commands": [{"command": "python -m unittest", "exit_code": 0}],
                    "deterministic_products": [
                        {
                            "relative_path": "figure.png",
                            "expected_sha256": "b" * 64,
                            "actual_sha256": "b" * 64,
                        }
                    ],
                    "nondeterministic_variations": [],
                },
            )
            _json(
                package / "package_manifest.json",
                {
                    "status": "passed",
                    "contains_corpus": False,
                    "missing_required_artifacts": [],
                    "files": _manifest_items(package),
                },
            )
            benchmark = root / "artifacts" / "resource_benchmark" / "final_fit_split7"
            benchmark.mkdir(parents=True, exist_ok=True)
            with (benchmark / "resource_benchmark_raw.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "model",
                        "repetition",
                        "peak_process_memory_bytes",
                        "peak_process_memory_status",
                        "peak_gpu_memory_bytes",
                        "peak_gpu_memory_status",
                        "device",
                    ),
                )
                writer.writeheader()
                for model in ("finite_hmm", "hdp_hmm", "transformer", "vomm"):
                    is_gpu = model == "transformer"
                    writer.writerow(
                        {
                            "model": model,
                            "repetition": 1,
                            "peak_process_memory_bytes": 1000,
                            "peak_process_memory_status": "measured",
                            "peak_gpu_memory_bytes": 2000 if is_gpu else "",
                            "peak_gpu_memory_status": "measured" if is_gpu else "not_applicable",
                            "device": "cuda" if is_gpu else "cpu",
                        }
                    )
            raw = benchmark / "resource_benchmark_raw.csv"
            for name in (
                "resource_benchmark_summary.csv",
                "resource_benchmark_environment.json",
                "resource_benchmark_config.json",
            ):
                path = benchmark / name
                path.write_text("value\n" if path.suffix == ".csv" else "{}", encoding="utf-8")
            benchmark_files = [
                path
                for path in benchmark.iterdir()
                if path.name != "resource_benchmark_audit.json"
            ]
            _json(
                benchmark / "resource_benchmark_audit.json",
                {
                    "status": "passed",
                    "coverage_status": "passed",
                    "process_memory_status": "passed",
                    "gpu_memory_status": "passed",
                    "coverage": {
                        "models": {
                            "finite_hmm": 1,
                            "hdp_hmm": 1,
                            "transformer": 1,
                            "vomm": 1,
                        },
                        "repetitions": 1,
                        "expected_rows": 4,
                        "observed_rows": 4,
                    },
                    "files": [
                        {
                            "relative_path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                        }
                        for path in benchmark_files
                    ],
                },
            )
            _json(
                root / "artifacts" / "economic_cost_scenario.json",
                {
                    "status": "documented",
                    "as_of_date": "2026-09-03",
                    "rates": [
                        {"usd_per_hour": 1.0, "source": "https://example.com/pricing"}
                    ],
                },
            )

            result = validate_engineering_requirements(root)

            statuses = {item["id"]: item["status"] for item in result["requirements"]}
            self.assertEqual(statuses["R4"], "passed")
            self.assertEqual(statuses["R5"], "passed")

    def test_package_excludes_corpus_and_personal_paths_and_hashes_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            benchmark = root / "benchmark"
            output = root / "package"
            source.mkdir()
            benchmark.mkdir()
            (source / "results_raw.csv").write_text("model,test_nll\nfinite_hmm,1.0\n", encoding="utf-8")
            _json(
                source / "config.json",
                {
                    "results_root": r"C:\Users\name\artifacts",
                    "corpus_root": r"external\PDMX\mxl",
                    "split_seed": 7,
                },
            )
            (source / "corpus_cache_3000.jsonl").write_text("restricted", encoding="utf-8")
            (benchmark / "resource_benchmark_raw.csv").write_text(
                "model,fit_seconds\nfinite_hmm,1.0\n", encoding="utf-8"
            )
            (benchmark / "resource_benchmark_summary.csv").write_text(
                "model,fit_seconds_median\nfinite_hmm,1.0\n", encoding="utf-8"
            )
            for name in (
                "resource_benchmark_environment.json",
                "resource_benchmark_config.json",
                "resource_benchmark_audit.json",
            ):
                _json(benchmark / name, {"status": "passed"})

            manifest = build_reproducibility_package(
                source_run=source,
                benchmark_dir=benchmark,
                output_dir=output,
            )

            self.assertEqual(manifest["status"], "partial")
            self.assertFalse((output / "source_run" / "corpus_cache_3000.jsonl").exists())
            packaged_config = (output / "source_run" / "config.json").read_text(encoding="utf-8")
            self.assertNotIn("C:\\Users", packaged_config)
            self.assertNotIn("PDMX", packaged_config)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

    def test_package_redacts_embedded_corpus_paths_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            (source / "results_raw.csv").write_text(
                "model,source_path\nfinite_hmm,/srv/researcher/private/file.csv\n",
                encoding="utf-8",
            )

            build_reproducibility_package(
                source_run=source,
                benchmark_dir=root / "benchmark",
                output_dir=root / "package",
            )

            packaged = (root / "package" / "source_run" / "results_raw.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("<redacted-path>", packaged)
            self.assertNotIn("/srv/", packaged)

    def test_package_includes_source_audit_sensitivities_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            comparisons = root / "artifacts" / "Comparacion"
            source = comparisons / "source_run"
            sensitivity = comparisons / "sens_split17"
            benchmark = root / "benchmark"
            source.mkdir(parents=True)
            sensitivity.mkdir()
            benchmark.mkdir()
            (source / "results_raw.csv").write_text("model\nfinite_hmm\n", encoding="utf-8")
            (sensitivity / "results_summary.csv").write_text("model\nfinite_hmm\n", encoding="utf-8")
            _json(comparisons / "audits" / source.name / "artifact_audit.json", {"status": "passed"})
            _json(root / "artifacts" / "diagnostico_finite_hmm_k.json", {"status": "completed"})
            _json(root / "artifacts" / "cache_reconstruction_verification.json", {"status": "hash_mismatch"})
            _json(root / "artifacts" / "economic_cost_scenario.json", {"status": "documented"})

            build_reproducibility_package(
                source_run=source,
                benchmark_dir=benchmark,
                output_dir=root / "package",
            )

            package = root / "package"
            self.assertTrue((package / "source_audit" / "artifact_audit.json").is_file())
            self.assertTrue(
                (package / "sensitivities" / "sens_split17" / "results_summary.csv").is_file()
            )
            self.assertTrue(
                (package / "diagnostics" / "diagnostico_finite_hmm_k.json").is_file()
            )
            regeneration = (package / "REGENERATION.md").read_text(encoding="utf-8")
            self.assertIn("package/diagnostics", regeneration)
            self.assertIn("reproduced_tables", regeneration)
            self.assertTrue(
                (package / "evidence" / "cache_reconstruction_verification.json").is_file()
            )
            self.assertTrue(
                (package / "evidence" / "economic_cost_scenario.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
