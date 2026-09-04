from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from Comparacion.resource_benchmark import (
    select_source_configurations,
    write_benchmark_artifacts,
)


class ResourceBenchmarkArtifactTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model in ("finite_hmm", "hdp_hmm", "transformer", "vomm"):
            for repetition in range(1, 4):
                uses_cuda = model == "transformer"
                rows.append(
                    {
                        "model": model,
                        "repetition": repetition,
                        "frac": 1.0,
                        "split_seed": 7,
                        "model_seed": 1 if model != "vomm" else "deterministic",
                        "device": "cuda" if uses_cuda else "cpu",
                        "selection_wall_clock_s": float(repetition),
                        "evaluation_wall_clock_s": 0.5,
                        "total_protocol_wall_clock_s": float(repetition) + 0.5,
                        "selection_peak_process_tree_rss_bytes": 1000 + repetition,
                        "selection_resource_status": "measured",
                        "selection_peak_cuda_allocated_bytes": 2000 if uses_cuda else None,
                        "selection_peak_cuda_reserved_bytes": 2500 if uses_cuda else None,
                        "selection_cuda_resource_status": "measured" if uses_cuda else "not_requested",
                        "evaluation_peak_process_tree_rss_bytes": 900 + repetition,
                        "evaluation_resource_status": "measured",
                        "evaluation_peak_cuda_allocated_bytes": 1500 if uses_cuda else None,
                        "evaluation_peak_cuda_reserved_bytes": 1800 if uses_cuda else None,
                        "evaluation_cuda_resource_status": "measured" if uses_cuda else "not_requested",
                        "resource_measurement_condition": "isolated",
                        "resource_cost_usable": True,
                        "test_nll": 1.0,
                        "fit_protocol": (
                            "fixed_architecture_with_early_stopping"
                            if uses_cuda
                            else "source_selected_configuration"
                        ),
                    }
                )
        return rows

    def test_writes_audited_three_repetition_benchmark(self) -> None:
        fingerprint = {
            "algorithm": "sha256-canonical-corpus-v1",
            "sha256": "A" * 64,
            "raw_file_sha256": "B" * 64,
            "counts": {
                "entries": 3000,
                "prepared_pieces": 2933,
                "exclusions": 67,
                "events": 693754,
                "tokens": 693754,
            },
            "excluded_fields": ["source_path", "path", "detail"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            audit = write_benchmark_artifacts(
                output,
                rows=self._rows(),
                environment={"python_version": "3.12"},
                config={"repetitions": 3, "measurement_condition": "isolated"},
                corpus_fingerprint=fingerprint,
                expected_corpus_fingerprint="A" * 64,
            )

            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["coverage"]["observed_rows"], 12)
            self.assertEqual(audit["corpus_fingerprint_status"], "passed")
            for name in (
                "resource_benchmark_raw.csv",
                "resource_benchmark_summary.csv",
                "resource_benchmark_environment.json",
                "resource_benchmark_config.json",
                "resource_benchmark_audit.json",
            ):
                self.assertTrue((output / name).is_file())
            with (output / "resource_benchmark_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                summary = list(csv.DictReader(handle))
            finite = next(row for row in summary if row["model"] == "finite_hmm")
            self.assertEqual(float(finite["selection_wall_clock_s_median"]), 2.0)

    def test_source_configuration_selection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            with (source / "results_raw.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "model",
                        "frac",
                        "data_seed",
                        "model_seed",
                        "hyperparams_json",
                    ),
                )
                writer.writeheader()
                for model in ("finite_hmm", "hdp_hmm", "transformer", "vomm"):
                    writer.writerow(
                        {
                            "model": model,
                            "frac": 1.0,
                            "data_seed": 2,
                            "model_seed": 2,
                            "hyperparams_json": json.dumps({"choice": "later"}),
                        }
                    )
                    writer.writerow(
                        {
                            "model": model,
                            "frac": 1.0,
                            "data_seed": 1,
                            "model_seed": 1,
                            "hyperparams_json": json.dumps({"choice": "first"}),
                        }
                    )

            selected = select_source_configurations(source, fraction=1.0)

        self.assertEqual(set(selected), {"finite_hmm", "hdp_hmm", "transformer", "vomm"})
        self.assertTrue(
            all(item["hyperparameters"]["choice"] == "first" for item in selected.values())
        )

    def test_rejects_duplicate_repetitions_and_fingerprint_mismatch(self) -> None:
        rows = self._rows()
        rows[-1]["repetition"] = 2
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit = write_benchmark_artifacts(
                temporary_directory,
                rows=rows,
                environment={},
                config={"repetitions": 3, "measurement_condition": "isolated"},
                corpus_fingerprint={
                    "algorithm": "sha256-canonical-corpus-v1",
                    "sha256": "A" * 64,
                    "counts": {},
                },
                expected_corpus_fingerprint="C" * 64,
            )

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["coverage_status"], "failed")
        self.assertEqual(audit["corpus_fingerprint_status"], "failed")

    def test_audit_rejects_missing_gpu_memory(self) -> None:
        rows = self._rows()
        transformer = next(row for row in rows if row["model"] == "transformer")
        transformer["selection_peak_cuda_allocated_bytes"] = None
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit = write_benchmark_artifacts(
                temporary_directory,
                rows=rows,
                environment={},
                config={"repetitions": 3, "measurement_condition": "isolated"},
                corpus_fingerprint={
                    "algorithm": "sha256-canonical-corpus-v1",
                    "sha256": "A" * 64,
                    "counts": {},
                },
                expected_corpus_fingerprint="A" * 64,
            )

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["resource_status"], "failed")

    def test_audit_rejects_transformer_cpu_fallback(self) -> None:
        rows = self._rows()
        for row in rows:
            if row["model"] == "transformer":
                row["device"] = "cpu"
                row["selection_cuda_resource_status"] = "not_requested"
                row["evaluation_cuda_resource_status"] = "not_requested"
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit = write_benchmark_artifacts(
                temporary_directory,
                rows=rows,
                environment={},
                config={"repetitions": 3, "measurement_condition": "isolated"},
                corpus_fingerprint={
                    "algorithm": "sha256-canonical-corpus-v1",
                    "sha256": "A" * 64,
                    "counts": {},
                },
                expected_corpus_fingerprint="A" * 64,
            )

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["resource_status"], "failed")

    def test_audit_rejects_unexpected_corpus_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audit = write_benchmark_artifacts(
                temporary_directory,
                rows=self._rows(),
                environment={},
                config={"repetitions": 3, "measurement_condition": "isolated"},
                corpus_fingerprint={
                    "algorithm": "sha256-canonical-corpus-v1",
                    "sha256": "A" * 64,
                    "counts": {"entries": 2999},
                },
                expected_corpus_fingerprint="A" * 64,
            )

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["corpus_counts_status"], "failed")


if __name__ == "__main__":
    unittest.main()
