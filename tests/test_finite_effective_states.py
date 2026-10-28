"""F2: el HMM finito reporta el mismo estimando de estados que el HDP-HMM.

Preregistro, "Número de estados": media posterior del número de estados con
ocupación >= 2 % de los eventos; al finito se le aplica el mismo conteo sobre su
posterior de estados. No se compara un K del HDP contra un K de Viterbi.
"""

from __future__ import annotations

from contextlib import ExitStack
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analysis.diagnostics import compare_models
from src.data.observations import ObservationSequence
from src.models.finite_hmm import HARMONIC_STATE_SPACE, FiniteChordHMM


def _sequence(tokens: list[int]) -> ObservationSequence:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    array = np.asarray(tokens, dtype=int)
    return ObservationSequence(
        observation_type="pitch_class",
        tokens=array,
        vocabulary=names,
        decoded=[names[token] for token in array],
        events=[],
        extra={},
    )


def _three_state_model(emission_noise: float = 1e-6) -> FiniteChordHMM:
    """Tres estados; el estado k emite casi siempre la clase de altura k."""

    emission = np.full((3, 12), emission_noise)
    for state in range(3):
        emission[state, state] = 1.0
    emission /= emission.sum(axis=1, keepdims=True)
    transition = np.full((3, 3), 0.05)
    np.fill_diagonal(transition, 0.9)
    return FiniteChordHMM(
        state_space=HARMONIC_STATE_SPACE[:3],
        initial_probs=np.full(3, 1.0 / 3.0),
        transition_matrix=transition,
        emission_matrix=emission,
    )


# 98 % de los eventos en el estado 0, 1 % en el 1 y 1 % en el 2.
_TOKENS_98_1_1 = [0] * 49 + [1] + [0] * 49 + [2]


class FiniteEffectiveStatesTests(unittest.TestCase):
    def test_states_below_two_percent_do_not_count(self) -> None:
        result = _three_state_model().fit_predict(_sequence(_TOKENS_98_1_1))
        self.assertEqual(len(result.active_states), 3)
        self.assertAlmostEqual(result.effective_states, 1.0)

    def test_compare_models_uses_the_preregistered_estimand_for_the_finite(self) -> None:
        result = _three_state_model().fit_predict(_sequence(_TOKENS_98_1_1))
        table = compare_models(result, None)
        self.assertAlmostEqual(float(table.loc[0, "effective_states"]), 1.0)

    def test_threshold_is_the_preregistered_two_percent(self) -> None:
        model = _three_state_model()
        self.assertEqual(model.occupancy_threshold, 0.02)
        tokens = [0] * 48 + [1, 1] + [0] * 48 + [2, 2]  # 96 % / 2 % / 2 %
        self.assertAlmostEqual(model.fit_predict(_sequence(tokens)).effective_states, 3.0)

    def test_it_is_a_posterior_mean_over_state_paths(self) -> None:
        # Con emisiones ruidosas la posterior de estados no es degenerada: la media
        # posterior queda estrictamente entre los conteos de las trayectorias.
        model = _three_state_model(emission_noise=0.5)
        tokens = [0] * 30 + [1] * 2 + [0] * 30 + [2] * 2 + [0] * 36
        result = model.fit_predict(_sequence(tokens))
        samples = result.occupied_state_samples
        self.assertEqual(len(samples), model.n_posterior_samples)
        self.assertAlmostEqual(result.effective_states, float(np.mean(samples)))
        self.assertGreater(len(set(samples)), 1)

    def test_same_seed_gives_the_same_estimate(self) -> None:
        model = _three_state_model(emission_noise=0.5)
        tokens = [0] * 30 + [1] * 2 + [0] * 30 + [2] * 2 + [0] * 36
        first = model.fit_predict(_sequence(tokens)).effective_states
        second = model.fit_predict(_sequence(tokens)).effective_states
        self.assertEqual(first, second)

    def test_library_batch_reports_the_finite_effective_states(self) -> None:
        from src.analysis import library_batch

        sequence = _sequence(_TOKENS_98_1_1)
        row = pd.Series({"path": "obra.musicxml", "title": "obra"})
        # La firma cambia entre ramas: en fix/auditoria-tesis recibe los eventos ya
        # extraídos; en la base, la ruta de la partitura y prefer_treble.
        parameters = inspect.signature(library_batch.analyze_catalog_piece).parameters
        kwargs = {"row": row, "obs_type": "pitch_class", "model": "finite_hmm", "hdp_params": {}}
        if "events" in parameters:
            kwargs["events"] = []
        else:
            kwargs["prefer_treble"] = False
        with ExitStack() as stack:
            for name in ("parse_score", "extract_events"):
                if hasattr(library_batch, name):
                    stack.enter_context(patch.object(library_batch, name, return_value=[]))
            stack.enter_context(patch.object(library_batch, "build_observation_sequence", return_value=sequence))
            stack.enter_context(patch.object(library_batch, "FiniteChordHMM", _three_state_model))
            record = library_batch.analyze_catalog_piece(**kwargs)
        self.assertAlmostEqual(record["finite_effective_states"], 1.0)


class SummaryLabelTests(unittest.TestCase):
    """El rótulo 'n_active_states' es el estimando del preregistro en ambos modelos."""

    def setUp(self) -> None:
        self.result = _three_state_model().fit_predict(_sequence(_TOKENS_98_1_1))

    @staticmethod
    def _metrics(table: pd.DataFrame) -> pd.Series:
        return table.set_index("metric")["value"]

    def _check(self, table: pd.DataFrame) -> None:
        metrics = self._metrics(table)
        # 98/1/1 %: Viterbi visita 3 estados, pero solo 1 supera el 2 %.
        self.assertAlmostEqual(float(metrics["n_active_states"]), self.result.effective_states)
        self.assertAlmostEqual(float(metrics["n_active_states"]), 1.0)
        self.assertEqual(float(metrics["n_viterbi_states"]), 3.0)

    def test_finite_summary_dataframe(self) -> None:
        self._check(self.result.summary_dataframe())

    def test_summarize_finite_result(self) -> None:
        from src.analysis.diagnostics import summarize_finite_result

        self._check(summarize_finite_result(self.result))

    def test_hdp_tables_use_the_same_estimand_under_the_label(self) -> None:
        from src.analysis.diagnostics import summarize_hdp_result
        from src.models.hdp_hmm import TruncatedHDPHMM

        hdp = TruncatedHDPHMM(n_states=6, n_iters=20, burn_in=10, seed=1).fit(_sequence(_TOKENS_98_1_1))
        for table in (hdp.summary_dataframe(), summarize_hdp_result(hdp)):
            self.assertAlmostEqual(float(self._metrics(table)["n_active_states"]), hdp.effective_states)

    def test_hdp_trace_active_states_is_the_same_estimand(self) -> None:
        """La traza por iteración ('active_states') cuenta ocupación >= 2 %, no estados distintos."""

        from Comparacion.hdp_diagnostics import build_chain_diagnostics
        from src.models.hdp_hmm import TruncatedHDPHMM

        burn_in = 30
        hdp = TruncatedHDPHMM(n_states=20, n_iters=60, burn_in=burn_in, seed=1).fit(
            _sequence(_TOKENS_98_1_1 * 4)
        )
        history = hdp.diagnostics.active_state_history
        # store_every=1: la muestra guardada s es la iteración burn_in + s.
        self.assertEqual(history[burn_in:], hdp.diagnostics.occupied_state_samples)

        # Es la tabla "diagnostics" de build_hdp_tables y la serie de hdp_active_states.png.
        table = hdp.diagnostics.to_dataframe()
        self.assertAlmostEqual(float(table["active_states"].iloc[burn_in:].mean()), hdp.effective_states)
        rows = hdp.diagnostics.trace_rows()
        self.assertEqual([row["active_states"] for row in rows], history)
        # 60 iteraciones: la ventana tardía de Comparacion es justo el tramo tras el burn-in.
        chain = build_chain_diagnostics({1: rows}, min_iterations=10)["chains"]["1"]
        self.assertAlmostEqual(float(chain["late_mean_active_states"]), hdp.effective_states)


if __name__ == "__main__":
    unittest.main()
