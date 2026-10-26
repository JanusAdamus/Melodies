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
