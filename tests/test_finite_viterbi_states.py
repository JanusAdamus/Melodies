"""Las salidas del HMM finito no llaman 'activos' a dos estimandos distintos.

Igual que en el HDP: ``n_active_states`` del resumen es el estimando del preregistro
(estados con ocupación >= 2 %), mientras que ``result.active_states`` son los estados
distintos de la ruta de Viterbi, sin umbral.  La tabla ``transition_matrix_active`` y
los mapas de calor usan estos últimos y deben decirlo.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from src.analysis import visualization
from src.analysis.diagnostics import build_finite_tables
from tests.test_finite_effective_states import _TOKENS_98_1_1, _sequence, _three_state_model

VITERBI_AXIS = "viterbi_state"


class FiniteViterbiStatesLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        observations = _sequence(_TOKENS_98_1_1)
        observations.events.extend(SimpleNamespace(offset=float(index)) for index in range(observations.size))
        cls.result = _three_state_model().fit_predict(observations)
        cls.tables = build_finite_tables(pd.DataFrame(), cls.result)

    def test_the_case_separates_the_two_counts(self) -> None:
        self.assertGreater(len(self.result.active_states), self.result.effective_states)

    def test_active_transition_table_says_its_states_are_the_viterbi_path(self) -> None:
        table = self.tables["transition_matrix_active"]
        self.assertEqual(len(table), len(self.result.active_states))
        self.assertEqual(table.index.name, VITERBI_AXIS)
        self.assertEqual(table.columns.name, VITERBI_AXIS)

    def test_heatmap_titles_say_viterbi(self) -> None:
        titles: list[str] = []

        def capture(*args, **kwargs):
            titles.extend(arg for arg in args if isinstance(arg, str) and "finito" in arg)
            return "x"

        with patch.multiple(
            visualization,
            plot_heatmap=capture,
            plot_emission_heatmap=capture,
            plot_transition_graph=capture,
            plot_state_timeline=capture,
        ), patch.object(visualization.Path, "mkdir"):
            visualization.generate_finite_figures(self.result, "unused")
        heatmaps = [title for title in titles if "Timeline" not in title]
        self.assertEqual(len(heatmaps), 2)
        for title in heatmaps:
            self.assertIn("ruta de Viterbi", title)
            self.assertNotIn("activa", title)


if __name__ == "__main__":
    unittest.main()
