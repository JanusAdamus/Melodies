"""C5 (auditoría 2026-10-26): Pareto con incertidumbre por eje y sin el tiempo de evaluación.

Reproduce ``scripts/auditoria_2026_10_26/comparacion/estadistica.py`` (parte 2):
con NLL, ajuste y evaluación como ejes, cada mejor-en-un-eje quedaba en la
frontera aunque perdiera por mucho en NLL, y los milisegundos de evaluación
decidían.
"""

from __future__ import annotations

import unittest

from Comparacion.decision import pareto_front
from Comparacion.runner import _build_pareto_summary


def _raw(model: str, nll: float, selected_fit: float, evaluation: float, device: str = "cpu") -> dict:
    return {
        "model": model,
        "frac": 1.0,
        "test_nll": nll,
        "fit_wall_clock_s": 10 * selected_fit,
        "selected_fit_wall_clock_s": selected_fit,
        "evaluation_wall_clock_s": evaluation,
        "device": device,
    }


def _headline(intervals: dict[str, tuple[float, float]]) -> dict:
    return {
        "estimates": [
            {"model": model, "frac": 1.0, "estimand": "token_weighted_nll", "bootstrap_95_ci": list(bounds)}
            for model, bounds in intervals.items()
        ]
    }


class IntervalDominanceTests(unittest.TestCase):
    def test_overlapping_intervals_are_not_a_strict_improvement(self) -> None:
        rows = [
            {"model": "a", "nll": 1.00, "nll_interval": [0.95, 1.10], "fit": 10.0},
            {"model": "b", "nll": 1.05, "nll_interval": [1.00, 1.15], "fit": 10.0},
        ]

        front = pareto_front(rows, minimize=("nll", "fit"), maximize=())

        self.assertEqual([row["model"] for row in front], ["a", "b"])

    def test_separated_interval_on_one_axis_and_overlap_elsewhere_dominates(self) -> None:
        rows = [
            {"model": "a", "nll": 1.00, "nll_interval": [0.95, 1.10], "fit": 10.0, "fit_interval": [9, 11]},
            {"model": "b", "nll": 1.05, "nll_interval": [1.00, 1.15], "fit": 100.0, "fit_interval": [90, 110]},
        ]

        front = pareto_front(rows, minimize=("nll", "fit"), maximize=())

        self.assertEqual([row["model"] for row in front], ["a"])

    def test_without_intervals_the_point_definition_is_unchanged(self) -> None:
        rows = [{"model": "a", "nll": 1.0, "fit": 1.0}, {"model": "b", "nll": 1.0, "fit": 2.0}]

        self.assertEqual([row["model"] for row in pareto_front(rows, minimize=("nll", "fit"), maximize=())], ["a"])


class PairedDominanceTests(unittest.TestCase):
    """C5 con la regla preregistrada: en NLL decide la diferencia pareada por obra."""

    def _rows(self) -> list[dict]:
        # Marginales que se traslapan (gran dispersión entre obras), pero «cheap» es
        # peor que «good» en todas las obras por 0.03: la diferencia pareada no cruza 0.
        return [
            {"model": "cheap", "nll": 2.04, "nll_interval": [1.909, 2.175], "fit": 1.0},
            {"model": "good", "nll": 2.01, "nll_interval": [1.882, 2.145], "fit": 10.0},
        ]

    def test_a_significantly_worse_model_does_not_dominate_through_overlapping_marginals(self) -> None:
        paired = {"nll": {("cheap", "good"): (0.030, 0.030)}}

        front = pareto_front(self._rows(), minimize=("nll", "fit"), maximize=(), paired_intervals=paired)

        self.assertEqual(sorted(row["model"] for row in front), ["cheap", "good"])

    def test_paired_interval_is_read_in_both_orientations(self) -> None:
        # Sólo se da good − cheap; la regla debe leer cheap − good = −(good − cheap).
        paired = {"nll": {("good", "cheap"): (-0.030, -0.030)}}

        front = pareto_front(self._rows(), minimize=("nll", "fit"), maximize=(), paired_intervals=paired)

        self.assertEqual(sorted(row["model"] for row in front), ["cheap", "good"])

    def test_paired_interval_crossing_zero_is_a_tie_on_that_axis(self) -> None:
        paired = {"nll": {("cheap", "good"): (-0.01, 0.07)}}

        front = pareto_front(self._rows(), minimize=("nll", "fit"), maximize=(), paired_intervals=paired)

        self.assertEqual([row["model"] for row in front], ["cheap"])

    def test_paired_interval_below_zero_is_strictly_better(self) -> None:
        rows = [
            {"model": "a", "nll": 2.00, "nll_interval": [1.8, 2.2], "fit": 10.0},
            {"model": "b", "nll": 2.01, "nll_interval": [1.8, 2.2], "fit": 10.0},
        ]
        paired = {"nll": {("a", "b"): (-0.02, -0.005)}}

        front = pareto_front(rows, minimize=("nll", "fit"), maximize=(), paired_intervals=paired)

        self.assertEqual([row["model"] for row in front], ["a"])


class RunnerParetoTests(unittest.TestCase):
    def test_evaluation_time_does_not_decide_and_selected_fit_is_the_cost_axis(self) -> None:
        raw = [
            _raw("transformer", 1.76, selected_fit=169.0, evaluation=5.0, device="cuda"),
            _raw("finite_hmm", 2.00, selected_fit=628.0, evaluation=1.0),
            _raw("hdp_hmm", 2.08, selected_fit=578.0, evaluation=0.9),
            _raw("vomm", 2.08, selected_fit=41.0, evaluation=3.0),
        ]
        headline = _headline(
            {
                "transformer": (1.74, 1.78),
                "finite_hmm": (1.97, 2.03),
                "hdp_hmm": (2.05, 2.11),
                "vomm": (2.05, 2.11),
            }
        )

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(partial["axes"], ["mean_test_nll", "mean_selected_fit_wall_clock_s"])
        self.assertIn("mean_evaluation_wall_clock_s", partial["reported_not_dominance_axes"])
        # hdp_hmm era «óptimo» sólo por 0.1 s de evaluación; ahora lo domina el transformador.
        self.assertEqual(sorted(row["model"] for row in partial["frontier"]), ["transformer", "vomm"])
        self.assertTrue(partial["devices_differ"])
        self.assertEqual(partial["devices"]["transformer"], ["cuda"])

    def test_runner_uses_the_paired_work_difference_for_nll_dominance(self) -> None:
        raw = [
            _raw("cheap", 2.04, selected_fit=1.0, evaluation=1.0),
            _raw("good", 2.01, selected_fit=10.0, evaluation=1.0),
        ]
        headline = _headline({"cheap": (1.909, 2.175), "good": (1.882, 2.145)})
        pairwise = {
            "comparisons": [
                {
                    "model_a": "cheap",
                    "model_b": "good",
                    "status": "ok",
                    "token_weighted_difference": 0.030,
                    "token_weighted_bootstrap_95_ci": [0.030, 0.030],
                }
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(sorted(row["model"] for row in partial["frontier"]), ["cheap", "good"])
        self.assertIn("paired", partial["dominance"])

    def test_times_within_the_tolerance_are_ties(self) -> None:
        raw = [_raw("a", 2.0, selected_fit=100.0, evaluation=1.0), _raw("b", 2.0, selected_fit=105.0, evaluation=1.0)]
        headline = _headline({"a": (1.9, 2.1), "b": (1.9, 2.1)})

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(sorted(row["model"] for row in partial["frontier"]), ["a", "b"])
        self.assertGreater(partial["time_relative_tolerance"], 0.0)

    def test_without_a_selected_fit_time_a_model_is_not_ranked_on_cost(self) -> None:
        raw = [_raw("a", 2.0, selected_fit=100.0, evaluation=1.0)]
        raw[0]["selected_fit_wall_clock_s"] = None

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, _headline({"a": (1.9, 2.1)}))[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(partial["candidates"], [])
        self.assertEqual(partial["excluded_models"], ["a"])


if __name__ == "__main__":
    unittest.main()
