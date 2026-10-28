"""Las salidas del HDP no llaman 'activos' a dos estimandos distintos.

``n_active_states`` del resumen es el estimando del preregistro (media posterior del
número de estados con ocupación >= 2 %). ``result.active_states`` son los estados
distintos de la muestra de referencia, sin umbral, y alimentan la matriz
``transition_matrix_active``, los mapas de calor y la interpretación: lo dicen.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src.analysis import visualization
from src.analysis.diagnostics import build_hdp_tables, summarize_hdp_result
from src.analysis.interpretation import interpret_hdp_states
from src.models.hdp_hmm import TruncatedHDPHMM
from tests.test_finite_effective_states import _TOKENS_98_1_1, _sequence

REFERENCE_AXIS = "reference_sample_state"


def _fit():
    observations = _sequence(_TOKENS_98_1_1 * 4)
    observations.events.extend(SimpleNamespace(offset=float(index)) for index in range(observations.size))
    return TruncatedHDPHMM(n_states=20, n_iters=60, burn_in=30, seed=1).fit(observations)


class ReferenceStatesLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _fit()
        cls.tables = build_hdp_tables(pd.DataFrame(), cls.result)

    def test_the_case_separates_the_two_counts(self) -> None:
        # 98/1/1 %: la muestra de referencia visita estados con 1 % que no cuentan.
        self.assertGreater(len(self.result.active_states), self.result.effective_states)

    def test_summaries_name_the_reference_sample_count(self) -> None:
        for table in (self.result.summary_dataframe(), summarize_hdp_result(self.result), self.tables["summary"]):
            metrics = table.set_index("metric")["value"]
            self.assertAlmostEqual(float(metrics["n_active_states"]), self.result.effective_states)
            self.assertEqual(float(metrics["n_reference_states"]), float(len(self.result.active_states)))

    def test_active_transition_table_says_its_states_are_the_reference_sample(self) -> None:
        table = self.tables["transition_matrix_active"]
        self.assertEqual(list(table.index), self.result.active_state_labels)
        self.assertEqual(table.index.name, REFERENCE_AXIS)
        self.assertEqual(table.columns.name, REFERENCE_AXIS)

    def test_interpretation_reports_the_reference_share(self) -> None:
        table = interpret_hdp_states(self.result)
        self.assertEqual(table["state"].tolist(), self.result.active_states)
        usage = self.result.state_usage
        expected = [usage[state] / usage.sum() for state in self.result.active_states]
        self.assertEqual(table["reference_share"].tolist(), expected)

    def test_heatmap_titles_say_reference_sample(self) -> None:
        titles: list[str] = []

        def capture(*args, **kwargs):
            titles.extend(arg for arg in args if isinstance(arg, str) and "HDP" in arg)
            return "x"

        with patch.multiple(
            visualization,
            plot_heatmap=capture,
            plot_emission_heatmap=capture,
            plot_transition_graph=capture,
            plot_state_timeline=capture,
            plot_history=lambda *args, **kwargs: [],
        ), patch.object(visualization.Path, "mkdir"):
            visualization.generate_hdp_figures(self.result, "unused")
        heatmaps = [title for title in titles if "Timeline" not in title]
        self.assertEqual(len(heatmaps), 2)
        for title in heatmaps:
            self.assertIn("muestra de referencia", title)


if __name__ == "__main__":
    unittest.main()
