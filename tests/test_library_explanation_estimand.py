"""La lectura del corpus describe la figura de estados con el estimando que grafica (ocupación >= 2 %)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.analysis.library_visualization import build_analysis_explanation


class StateCountsExplanationTests(unittest.TestCase):
    def test_state_counts_figure_is_described_with_one_estimand(self) -> None:
        analysis = pd.DataFrame(
            {"finite_effective_states": [2.0, 2.5], "hdp_effective_states": [3.0, 2.0]}
        )
        line = next(
            row
            for row in build_analysis_explanation(analysis).splitlines()
            if "compare_state_counts_scatter.png" in row
        )
        self.assertNotIn("estados activos", line)
        self.assertIn("estados efectivos del HMM finito", line)
        self.assertIn("2 %", line)


if __name__ == "__main__":
    unittest.main()
