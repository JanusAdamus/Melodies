"""T2 (auditoría 2026-10-26): la NLL de representaciones distintas no se compara.

Cada representación tiene su propio alfabeto, así que su NLL por token está en
otra escala. El resumen del benchmark agrupa por representación y nunca pone dos
representaciones en la misma tabla.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from next_token_experiment.benchmarks.suite import BenchmarkRunSpec, collect_benchmark_suite


def _write_run(results_root: Path, run_name: str, representation: str, nll: float) -> BenchmarkRunSpec:
    run_dir = results_root / run_name
    (run_dir / "transformer").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"representation": {"primary": representation}}), encoding="utf-8"
    )
    (run_dir / "transformer" / "test_summary.json").write_text(
        json.dumps({"nll_per_token": nll}), encoding="utf-8"
    )
    return BenchmarkRunSpec(
        run_id=run_name,
        profile="p",
        purpose="t",
        comparison_group="baseline_vs_research",
        expected_budget_class="smoke",
        run_name=run_name,
    )


class BenchmarkRepresentationTests(unittest.TestCase):
    def test_runs_with_different_representations_are_never_tabulated_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            specs = [
                _write_run(root, "pc_a", "pitch_class", 2.0),
                _write_run(root, "rich", "event_pitch_duration_metrical", 4.0),
                _write_run(root, "pc_b", "pitch_class", 2.1),
            ]

            payload = collect_benchmark_suite(specs, results_root=root)
            markdown = (root / "benchmark_suite" / "summary.md").read_text(encoding="utf-8")
            summary = json.loads((root / "benchmark_suite" / "summary.json").read_text(encoding="utf-8"))

        tables = [block for block in markdown.split("\n\n") if block.lstrip().startswith("|")]
        for table in tables:
            with self.subTest(table=table[:60]):
                has_pitch_class = "| pitch_class |" in table
                has_rich = "| event_pitch_duration_metrical |" in table
                self.assertFalse(has_pitch_class and has_rich)
        self.assertEqual(len(tables), 2)
        self.assertEqual(summary["nll_comparable_within"], "representation")
        self.assertEqual(
            sorted(summary["runs_by_representation"]),
            ["event_pitch_duration_metrical", "pitch_class"],
        )
        self.assertEqual(
            [row["run_id"] for row in summary["runs_by_representation"]["pitch_class"]],
            ["pc_a", "pc_b"],
        )
        self.assertEqual(payload["nll_comparable_within"], "representation")


if __name__ == "__main__":
    unittest.main()
