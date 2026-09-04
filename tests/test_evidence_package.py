from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from Comparacion.evidence_package import (
    export_evidence_package,
    verify_evidence_package,
)
from tests.test_artifact_audit import _build_run


class EvidencePackageTests(unittest.TestCase):
    def _registry(self, root: Path, run_dir: Path) -> Path:
        registry = root / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "name": "main",
                            "path": str(run_dir),
                            "role": "primary",
                            "cost_usable": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return registry

    def test_export_is_sanitized_and_self_verifying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["corpus_root"] = r"C:\Users\Example\PDMX"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            output = root / "public"
            report = export_evidence_package(self._registry(root, run_dir), output)

            self.assertEqual(report["status"], "passed")
            self.assertTrue(Path(report["archive_path"]).is_file())
            packaged_config = (output / "runs" / "main" / "config.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("Users", packaged_config)
            self.assertIn("<redacted:absolute-path>", packaged_config)
            self.assertEqual(verify_evidence_package(output)["status"], "passed")

    def test_incomplete_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            (run_dir / "results_raw.csv").unlink()

            with self.assertRaisesRegex(ValueError, "did not pass"):
                export_evidence_package(self._registry(root, run_dir), root / "public")

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            output = root / "public"
            export_evidence_package(self._registry(root, run_dir), output)
            (output / "runs" / "main" / "config.json").write_text("{}", encoding="utf-8")

            report = verify_evidence_package(output)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("hash mismatch" in issue for issue in report["issues"]))

    def test_manifest_without_runs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            package.mkdir()
            (package / "package_manifest.json").write_text(
                json.dumps({"runs": [], "files": []}), encoding="utf-8"
            )

            report = verify_evidence_package(package)

            self.assertEqual(report["status"], "failed")
            self.assertIn("manifest has no runs", report["issues"])

    def test_duplicate_public_run_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "runs": [
                            {"name": "main", "path": str(run_dir)},
                            {"name": "main", "path": str(run_dir)},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate public run name"):
                export_evidence_package(registry, root / "public")

    def test_archive_cannot_be_written_inside_the_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            output = root / "public"

            with self.assertRaisesRegex(ValueError, "archive must be outside"):
                export_evidence_package(
                    self._registry(root, run_dir),
                    output,
                    archive_path=output / "evidence.zip",
                )

    def test_verifier_rejects_unsafe_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "package"
            package.mkdir()
            (package / "package_manifest.json").write_text(
                json.dumps(
                    {
                        "runs": [{"name": "../outside"}],
                        "files": [
                            {
                                "relative_path": "../outside.txt",
                                "size_bytes": 0,
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = verify_evidence_package(package)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("unsafe" in issue for issue in report["issues"]))

    def test_supplemental_benchmark_is_packaged_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            benchmark = root / "benchmark"
            benchmark.mkdir()
            with (benchmark / "resource_benchmark_raw.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=("model", "repetition"))
                writer.writeheader()
                writer.writerow({"model": "finite_hmm", "repetition": 1})
            raw_hash = hashlib.sha256(
                (benchmark / "resource_benchmark_raw.csv").read_bytes()
            ).hexdigest().upper()
            (benchmark / "resource_benchmark_audit.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "files": [
                            {
                                "relative_path": "resource_benchmark_raw.csv",
                                "sha256": raw_hash,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "runs": [{"name": "main", "path": str(run_dir)}],
                        "supplemental_artifacts": [
                            {
                                "name": "resource-benchmark",
                                "path": str(benchmark),
                                "audit": "resource_benchmark_audit.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "public"

            export_evidence_package(registry, output)

            self.assertTrue(
                (
                    output
                    / "supplemental"
                    / "resource-benchmark"
                    / "resource_benchmark_raw.csv"
                ).is_file()
            )
            self.assertEqual(verify_evidence_package(output)["status"], "passed")

    def test_stale_supplemental_audit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = _build_run(root)
            benchmark = root / "benchmark"
            benchmark.mkdir()
            (benchmark / "raw.csv").write_text("changed\n", encoding="utf-8")
            (benchmark / "audit.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "files": [
                            {"relative_path": "raw.csv", "sha256": "0" * 64}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "runs": [{"name": "main", "path": str(run_dir)}],
                        "supplemental_artifacts": [
                            {
                                "name": "resource-benchmark",
                                "path": str(benchmark),
                                "audit": "audit.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inconsistent"):
                export_evidence_package(registry, root / "public")


if __name__ == "__main__":
    unittest.main()
