from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Comparacion.config import build_default_learning_curve_config
from Comparacion.resource_benchmark import (
    _assert_corpus_invariants,
    run_resource_benchmark,
    select_source_configurations,
    write_benchmark_artifacts,
)


class ResourceBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _valid_rows() -> list[dict[str, object]]:
        rows = []
        for model in ("finite_hmm", "hdp_hmm", "transformer", "vomm"):
            for repetition, fit_seconds in enumerate((1.0, 2.0, 3.0), start=1):
                rows.append(
                    {
                        "model": model,
                        "repetition": repetition,
                        "fit_seconds": fit_seconds,
                        "evaluation_seconds": 0.5,
                        "peak_process_memory_bytes": 1000 + repetition,
                        "peak_process_memory_status": "measured",
                        "peak_gpu_memory_bytes": 2000 if model == "transformer" else None,
                        "peak_gpu_memory_status": "measured" if model == "transformer" else "not_applicable",
                        "device": "cuda" if model == "transformer" else "cpu",
                    }
                )
        return rows

    def test_selects_one_fixed_configuration_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory)
            rows = [
                ("finite_hmm", "2", 2, '{"selected_states": 48}'),
                ("finite_hmm", "1", 1, '{"selected_states": 24}'),
                ("hdp_hmm", "1", 1, '{"alpha": 8, "alpha0": 4, "gamma": 2}'),
                ("transformer", "1", 1, '{"architecture": "decoder_only"}'),
                ("vomm", "deterministic", 1, '{"selected_order": 2}'),
            ]
            with (source / "results_raw.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("model", "model_seed", "data_seed", "frac", "hyperparams_json"),
                )
                writer.writeheader()
                for model, model_seed, data_seed, hyperparameters in rows:
                    writer.writerow(
                        {
                            "model": model,
                            "model_seed": model_seed,
                            "data_seed": data_seed,
                            "frac": 1.0,
                            "hyperparams_json": hyperparameters,
                        }
                    )

            selected = select_source_configurations(source, fraction=1.0)

            self.assertEqual(set(selected), {"finite_hmm", "hdp_hmm", "transformer", "vomm"})
            self.assertEqual(selected["finite_hmm"]["model_seed"], 1)
            self.assertEqual(selected["finite_hmm"]["hyperparameters"]["selected_states"], 24)

    def test_writes_summary_environment_config_and_passing_audit(self) -> None:
        rows = self._valid_rows()

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = write_benchmark_artifacts(
                output,
                rows=rows,
                environment={"python_version": "3.12.10"},
                config={"repetitions": 3},
            )

            self.assertEqual(result["status"], "passed")
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
            self.assertEqual(float(finite["fit_seconds_median"]), 2.0)
            audit = json.loads(
                (output / "resource_benchmark_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(audit["coverage"]["observed_rows"], 12)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in audit["files"]))

    def test_audit_rejects_duplicate_repetitions_and_zero_gpu_peak(self) -> None:
        rows = self._valid_rows()
        rows[-1]["repetition"] = 2
        transformer = next(row for row in rows if row["model"] == "transformer")
        transformer["peak_gpu_memory_bytes"] = 0

        with tempfile.TemporaryDirectory() as temporary_directory:
            audit = write_benchmark_artifacts(
                temporary_directory,
                rows=rows,
                environment={"python_version": "3.12.10"},
                config={"repetitions": 3},
            )

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["coverage_status"], "failed")
        self.assertEqual(audit["gpu_memory_status"], "failed")

    def test_corpus_invariants_are_required_before_fitting(self) -> None:
        valid = {
            "n_prepared_pieces": 2933,
            "n_exclusions": 67,
            "n_total_tokens": 693754,
            "n_total_events": 693754,
        }

        _assert_corpus_invariants(valid)
        for field in valid:
            invalid = {**valid, field: valid[field] - 1}
            with self.subTest(field=field), self.assertRaises(ValueError):
                _assert_corpus_invariants(invalid)

    def test_invalid_cache_hash_stops_before_preparation_or_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "artifacts" / "Comparacion" / "source_run"
            source.mkdir(parents=True)
            (source / "config.json").write_text(
                json.dumps(asdict(build_default_learning_curve_config())),
                encoding="utf-8",
            )
            cache = root / "artifacts" / "corpus_cache_3000.jsonl"
            cache.write_text("wrong cache", encoding="utf-8")

            with (
                patch("Comparacion.resource_benchmark.audit_run", return_value={"status": "passed"}),
                patch("Comparacion.resource_benchmark.prepare_corpus") as prepare,
                self.assertRaisesRegex(ValueError, "unexpected corpus cache SHA-256"),
            ):
                run_resource_benchmark(
                    source_run=source,
                    fraction=1.0,
                    split_seed=7,
                    repetitions=3,
                    output_dir=root / "benchmark",
                    corpus_cache=cache,
                )

            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
