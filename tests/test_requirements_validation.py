from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.evidence_package import export_evidence_package
from Comparacion.requirements_validation import validate_requirements
from tests.test_artifact_audit import _build_run


class RequirementValidationTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest().upper()

    def test_verified_package_and_existing_file_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            registry = root / "registry.json"
            registry.write_text(
                json.dumps({"runs": [{"name": "main", "path": str(run_dir)}]}),
                encoding="utf-8",
            )
            package = root / "package"
            export_evidence_package(registry, package)
            evidence = root / "practice.md"
            evidence.write_text("documented", encoding="utf-8")
            context = root / "context.json"
            context.write_text(
                json.dumps(
                    {
                        "evidence_package": str(package),
                        "engineering_practices": str(evidence),
                    }
                ),
                encoding="utf-8",
            )
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R1",
                                "statement": "package",
                                "checks": [{"type": "package_verified"}],
                            },
                            {
                                "id": "R2",
                                "statement": "practice",
                                "checks": [
                                    {
                                        "type": "context_file_exists",
                                        "context_key": "engineering_practices",
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = validate_requirements(requirements, context)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["passed"], 2)

    def test_missing_evidence_remains_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = root / "context.json"
            context.write_text("{}", encoding="utf-8")
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R4",
                                "statement": "public evidence",
                                "checks": [{"type": "package_verified"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = validate_requirements(requirements, context)

            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["requirements"][0]["status"], "partial")

    def test_status_ceiling_prevents_documentation_from_claiming_social_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / "social.md"
            protocol.write_text("protocol", encoding="utf-8")
            context = root / "context.json"
            context.write_text(
                json.dumps({"social_protocol": str(protocol)}), encoding="utf-8"
            )
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R9",
                                "statement": "social",
                                "status_ceiling": "partial",
                                "checks": [
                                    {
                                        "type": "context_file_exists",
                                        "context_key": "social_protocol",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = validate_requirements(requirements, context)

            self.assertEqual(report["requirements"][0]["status"], "partial")

    def test_empty_requirement_register_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requirements = root / "requirements.json"
            requirements.write_text('{"requirements": []}', encoding="utf-8")
            context = root / "context.json"
            context.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty requirements"):
                validate_requirements(requirements, context)

    def test_duplicate_requirement_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R1",
                                "statement": "first",
                                "checks": [{"type": "package_verified"}],
                            },
                            {
                                "id": "R1",
                                "statement": "second",
                                "checks": [{"type": "package_verified"}],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            context = root / "context.json"
            context.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate requirement id"):
                validate_requirements(requirements, context)

    def test_resource_requirement_stays_partial_without_evaluation_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            costs = root / "engineering_costs.csv"
            with costs.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "model",
                        "device",
                        "frac",
                        "selection_wall_clock_s",
                        "evaluation_wall_clock_s",
                        "selection_peak_process_tree_rss_bytes",
                        "evaluation_peak_process_tree_rss_bytes",
                        "resource_measurement_condition",
                        "resource_cost_usable",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "finite_hmm",
                        "device": "cpu",
                        "frac": "1.0",
                        "selection_wall_clock_s": "10",
                        "evaluation_wall_clock_s": "1",
                        "selection_peak_process_tree_rss_bytes": "1000",
                        "evaluation_peak_process_tree_rss_bytes": "",
                        "resource_measurement_condition": "isolated",
                        "resource_cost_usable": "true",
                    }
                )
            context = root / "context.json"
            context.write_text(
                json.dumps({"resource_costs": str(costs)}), encoding="utf-8"
            )
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R5",
                                "statement": "resources",
                                "checks": [
                                    {
                                        "type": "resource_costs_complete",
                                        "required_models": ["finite_hmm"],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = validate_requirements(requirements, context)

            self.assertEqual(report["status"], "partial")
            self.assertEqual(
                report["requirements"][0]["checks"][0]["incomplete_models"],
                ["finite_hmm"],
            )

    def test_resource_benchmark_audit_and_hashes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "resource_benchmark_raw.csv"
            raw.write_text("model,repetition\nfinite_hmm,1\n", encoding="utf-8")
            audit = root / "resource_benchmark_audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "coverage_status": "passed",
                        "resource_status": "passed",
                        "corpus_fingerprint_status": "passed",
                        "corpus_counts_status": "passed",
                        "coverage": {
                            "models": {"finite_hmm": 3},
                            "repetitions": 3,
                            "expected_rows": 3,
                            "observed_rows": 3,
                        },
                        "files": [
                            {
                                "relative_path": raw.name,
                                "sha256": self._sha256(raw),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context = root / "context.json"
            context.write_text(
                json.dumps({"resource_benchmark_audit": str(audit)}), encoding="utf-8"
            )
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R5",
                                "statement": "resources",
                                "checks": [
                                    {
                                        "type": "resource_benchmark_passed",
                                        "required_models": ["finite_hmm"],
                                        "minimum_repetitions": 3,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = validate_requirements(requirements, context)

            self.assertEqual(report["status"], "passed")

    def test_resource_benchmark_with_too_few_repetitions_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.csv"
            raw.write_text("ok\n1\n", encoding="utf-8")
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "coverage_status": "passed",
                        "resource_status": "passed",
                        "corpus_fingerprint_status": "passed",
                        "corpus_counts_status": "passed",
                        "coverage": {
                            "models": {"finite_hmm": 1},
                            "repetitions": 1,
                            "expected_rows": 1,
                            "observed_rows": 1,
                        },
                        "files": [
                            {"relative_path": raw.name, "sha256": self._sha256(raw)}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            context = root / "context.json"
            context.write_text(
                json.dumps({"resource_benchmark_audit": str(audit)}), encoding="utf-8"
            )
            requirements = root / "requirements.json"
            requirements.write_text(
                json.dumps(
                    {
                        "requirements": [
                            {
                                "id": "R5",
                                "statement": "resources",
                                "checks": [
                                    {
                                        "type": "resource_benchmark_passed",
                                        "required_models": ["finite_hmm"],
                                        "minimum_repetitions": 3,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = validate_requirements(requirements, context)

            self.assertEqual(report["status"], "partial")


if __name__ == "__main__":
    unittest.main()
