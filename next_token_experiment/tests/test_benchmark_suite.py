from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from next_token_experiment.benchmarks.suite import (
    BenchmarkRunSpec,
    collect_benchmark_suite,
    load_benchmark_run_specs,
    resolve_benchmark_suite,
)


class BenchmarkSuiteTests(unittest.TestCase):
    def test_load_and_resolve_canonical_runs(self) -> None:
        specs = load_benchmark_run_specs()
        self.assertEqual([spec.run_id for spec in specs[:4]], [
            "cpu_baseline_smoke",
            "cpu_baseline_full",
            "research_richer_events_smoke",
            "research_richer_events_full",
        ])

        smoke_specs = resolve_benchmark_suite(specs, only_smoke=True)
        self.assertTrue(all(spec.expected_budget_class == "smoke" for spec in smoke_specs))

        full_specs = resolve_benchmark_suite(specs, only_full=True)
        self.assertTrue(all(spec.expected_budget_class == "full" for spec in full_specs))

    def test_collect_benchmark_suite_builds_consolidated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir) / "results"
            run_dir = results_root / "cpu_baseline_smoke"
            transformer_dir = run_dir / "transformer"
            transformer_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "representation": {"primary": "pitch_class"},
                        "windows": {"max_context_length": 128},
                        "transformer": {
                            "n_layers": 3,
                            "d_model": 128,
                            "n_heads": 4,
                            "use_relative_position_bias": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "data_summary.json").write_text(
                json.dumps(
                    {
                        "dataset": {
                            "n_train_pieces": 4,
                            "n_validation_pieces": 1,
                            "n_test_pieces": 1,
                            "n_train_windows": 20,
                            "n_validation_windows": 4,
                            "n_test_windows": 4,
                        },
                        "runtime": {
                            "device": "cpu",
                            "actual_precision": "fp32",
                            "seed": 7,
                        },
                        "timings": {
                            "fit_wall_clock_s": 12.5,
                            "evaluation_wall_clock_s": 1.5,
                            "total_wall_clock_s": 15.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (transformer_dir / "fit_summary.json").write_text(
                json.dumps(
                    {
                        "best_validation_nll": 1.23,
                        "best_validation_perplexity": 3.42,
                        "parameter_count": 415872,
                    }
                ),
                encoding="utf-8",
            )
            (transformer_dir / "test_summary.json").write_text(
                json.dumps(
                    {
                        "nll_per_token": 1.11,
                        "perplexity": 3.03,
                        "accuracy": 0.55,
                        "top_3_accuracy": 0.72,
                        "top_5_accuracy": 0.84,
                        "parameter_count": 415872,
                        "runtime": {
                            "device": "cpu",
                            "actual_precision": "fp32",
                            "seed": 7,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (transformer_dir / "validation_slice_metrics.json").write_text(
                json.dumps({"piece_length": []}),
                encoding="utf-8",
            )
            (transformer_dir / "test_slice_metrics.json").write_text(
                json.dumps({"token_rarity": [], "composer": []}),
                encoding="utf-8",
            )

            specs = [
                BenchmarkRunSpec(
                    run_id="cpu_baseline_smoke",
                    profile="cpu_baseline",
                    purpose="test",
                    comparison_group="baseline_vs_research",
                    expected_budget_class="smoke",
                    run_name="cpu_baseline_smoke",
                    representation="pitch_class",
                    context_length=128,
                    max_files=8,
                    max_windows={"train": 64, "validation": 16, "test": 16},
                    seed=7,
                    notes=("test run",),
                )
            ]
            payload = collect_benchmark_suite(specs, results_root=results_root)

            summary_rows = payload["summary_rows"]
            self.assertEqual(len(summary_rows), 1)
            self.assertEqual(summary_rows[0]["run_id"], "cpu_baseline_smoke")
            self.assertEqual(summary_rows[0]["representation"], "pitch_class")
            self.assertEqual(summary_rows[0]["status"], "completed")
            self.assertEqual(summary_rows[0]["n_train_windows"], 20)
            self.assertIn("token_rarity", summary_rows[0]["test_slices_recorded"])

            benchmark_root = results_root / "benchmark_suite"
            self.assertTrue((benchmark_root / "summary.csv").exists())
            self.assertTrue((benchmark_root / "summary.json").exists())
            self.assertTrue((benchmark_root / "summary.md").exists())
            self.assertTrue((benchmark_root / "run_status.json").exists())


if __name__ == "__main__":
    unittest.main()
