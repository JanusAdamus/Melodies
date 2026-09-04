from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.engineering_requirements import (
    build_reproducibility_package,
    validate_engineering_requirements,
)


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class EngineeringRequirementTests(unittest.TestCase):
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
                {"status": "ok"},
            )

            result = validate_engineering_requirements(root)

            self.assertEqual(result["requirements"][0]["status"], "passed")

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
            _json(package / "package_manifest.json", {"status": "passed", "files": []})
            _json(package / "clean_clone_verification.json", {"status": "passed"})
            benchmark = root / "artifacts" / "resource_benchmark" / "final_fit_split7"
            _json(benchmark / "resource_benchmark_audit.json", {"status": "passed"})
            benchmark.mkdir(parents=True, exist_ok=True)
            with (benchmark / "resource_benchmark_raw.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "model",
                        "peak_process_memory_bytes",
                        "peak_process_memory_status",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "finite_hmm",
                        "peak_process_memory_bytes": 1000,
                        "peak_process_memory_status": "measured",
                    }
                )
            _json(
                root / "artifacts" / "economic_cost_scenario.json",
                {"status": "documented", "as_of_date": "2026-09-03", "rates": [{"usd_per_hour": 1.0}]},
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
            _json(source / "config.json", {"results_root": r"C:\Users\name\artifacts", "split_seed": 7})
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

            self.assertEqual(manifest["status"], "passed")
            self.assertFalse((output / "source_run" / "corpus_cache_3000.jsonl").exists())
            self.assertNotIn("C:\\Users", (output / "source_run" / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["files"]))

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


if __name__ == "__main__":
    unittest.main()
