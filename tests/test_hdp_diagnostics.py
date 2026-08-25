from __future__ import annotations

import math
import unittest

from Comparacion.hdp_diagnostics import (
    build_chain_diagnostics,
    lag_one_autocorrelation,
)


def _stable_chain(seed: int, level: float, length: int = 40) -> list[dict[str, float]]:
    return [
        {
            "iteration": index + 1,
            "log_likelihood": level + (0.5 if index % 2 else -0.5),
            "active_states": 12,
            "beta_entropy": 2.0,
            "model_seed": seed,
        }
        for index in range(length)
    ]


def _drifting_chain(seed: int, length: int = 40) -> list[dict[str, float]]:
    return [
        {
            "iteration": index + 1,
            "log_likelihood": -1000.0 + 10.0 * index,
            "active_states": 5 + index // 4,
            "beta_entropy": 2.0,
            "model_seed": seed,
        }
        for index in range(length)
    ]


class LagOneAutocorrelationTests(unittest.TestCase):
    def test_alternating_series_is_negatively_correlated(self) -> None:
        values = [1.0, -1.0] * 10

        self.assertLess(lag_one_autocorrelation(values), 0.0)

    def test_constant_series_has_no_defined_autocorrelation(self) -> None:
        self.assertIsNone(lag_one_autocorrelation([2.0] * 10))

    def test_short_series_has_no_defined_autocorrelation(self) -> None:
        self.assertIsNone(lag_one_autocorrelation([1.0]))


class ChainDiagnosticsTests(unittest.TestCase):
    def test_stable_chains_that_agree_are_reported_as_stable(self) -> None:
        diagnostics = build_chain_diagnostics(
            {1: _stable_chain(1, -900.0), 2: _stable_chain(2, -900.2)},
            min_iterations=20,
        )

        self.assertEqual(diagnostics["status"], "ok")
        self.assertEqual(diagnostics["n_chains"], 2)
        self.assertTrue(diagnostics["chains"]["1"]["window_stability"])
        self.assertTrue(diagnostics["cross_chain"]["active_states_agree"])
        self.assertEqual(diagnostics["verdict"], "stable_within_the_observed_window")

    def test_drifting_chain_is_not_called_converged(self) -> None:
        diagnostics = build_chain_diagnostics(
            {1: _drifting_chain(1), 2: _drifting_chain(2)}, min_iterations=20
        )

        self.assertFalse(diagnostics["chains"]["1"]["window_stability"])
        self.assertEqual(diagnostics["verdict"], "drift_detected")
        self.assertFalse(diagnostics["convergence_claimed"])

    def test_disagreeing_chains_are_reported(self) -> None:
        diagnostics = build_chain_diagnostics(
            {1: _stable_chain(1, -900.0), 2: _stable_chain(2, -400.0)},
            min_iterations=20,
        )

        self.assertFalse(diagnostics["cross_chain"]["log_likelihood_agree"])
        self.assertEqual(diagnostics["verdict"], "chains_disagree")

    def test_short_chains_are_inconclusive(self) -> None:
        diagnostics = build_chain_diagnostics(
            {1: _stable_chain(1, -900.0, length=8)}, min_iterations=20
        )

        self.assertEqual(diagnostics["status"], "diagnostics_inconclusive")
        self.assertEqual(diagnostics["reason"], "chains_shorter_than_min_iterations")
        self.assertFalse(diagnostics["convergence_claimed"])

    def test_a_single_chain_cannot_show_cross_chain_agreement(self) -> None:
        diagnostics = build_chain_diagnostics({1: _stable_chain(1, -900.0)}, min_iterations=20)

        self.assertEqual(diagnostics["n_chains"], 1)
        self.assertIsNone(diagnostics["cross_chain"]["active_states_agree"])
        self.assertEqual(diagnostics["verdict"], "single_chain_no_cross_chain_evidence")

    def test_never_claims_convergence(self) -> None:
        diagnostics = build_chain_diagnostics(
            {1: _stable_chain(1, -900.0), 2: _stable_chain(2, -900.1)},
            min_iterations=20,
        )

        self.assertFalse(diagnostics["convergence_claimed"])
        self.assertIn("stability", diagnostics["policy"])

    def test_non_finite_values_are_excluded_from_the_windows(self) -> None:
        chain = _stable_chain(1, -900.0)
        chain[0]["log_likelihood"] = math.nan
        diagnostics = build_chain_diagnostics({1: chain}, min_iterations=20)

        self.assertEqual(diagnostics["chains"]["1"]["n_non_finite"], 1)


if __name__ == "__main__":
    unittest.main()
