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
    """Mismos intervalos para la media por obra (decide) y la ponderada (secundaria)."""

    return {
        "estimates": [
            {
                "model": model,
                "frac": 1.0,
                "estimand": estimand,
                "point_estimate": (bounds[0] + bounds[1]) / 2,
                "bootstrap_95_ci": list(bounds),
            }
            for model, bounds in intervals.items()
            for estimand in ("work_mean_nll", "token_weighted_nll")
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

        self.assertEqual(partial["axes"], ["work_mean_test_nll", "mean_selected_fit_wall_clock_s"])
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
                    "n_pairs": 40,
                    "mean_difference": 0.030,
                    "bootstrap_95_ci": [0.030, 0.030],
                    "work_mean_holm_ci": [0.030, 0.030],
                    "work_mean_holm_decision_ci": [0.030, 0.030],
                    "work_mean_holm_rejects_zero": True,
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


def _paired(model_a: str, model_b: str, *, raw, holm, rejects: bool, n_pairs: int = 40) -> dict:
    return {
        "model_a": model_a,
        "model_b": model_b,
        "status": "ok",
        "n_pairs": n_pairs,
        "mean_difference": (raw[0] + raw[1]) / 2,
        "bootstrap_95_ci": list(raw),
        # Holm tal cual: las pruebas de abajo comprueban que el runner lo ensancha si no rechaza.
        "work_mean_holm_ci": list(holm),
        "work_mean_holm_decision_ci": list(holm),
        "work_mean_holm_rejects_zero": rejects,
    }


class PreregisteredParetoRuleTests(unittest.TestCase):
    """Revisión adversarial de C5: Holm, n mínimo de obras y ciclos de dominancia."""

    def _fast_slow(self) -> tuple[list[dict], dict]:
        raw = [
            _raw("fast", 2.08, selected_fit=1.0, evaluation=1.0),
            _raw("slow", 2.00, selected_fit=10.0, evaluation=1.0),
        ]
        return raw, _headline({"fast": (1.9, 2.2), "slow": (1.85, 2.15)})

    def test_nll_dominance_uses_the_holm_adjusted_interval_not_the_plain_95(self) -> None:
        # Reproduce chk_holm2.py: el IC del 95 % excluye 0 pero el de Holm lo cruza.
        raw, headline = self._fast_slow()
        pairwise = {
            "comparisons": [
                _paired("fast", "slow", raw=(0.0014, 0.0843), holm=(-0.0064, 0.0939), rejects=False)
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual([row["model"] for row in partial["frontier"]], ["fast"])
        self.assertIn("holm", partial["nll_paired_interval"])

    def test_a_pair_not_rejected_by_holm_is_a_tie_even_if_its_interval_excludes_zero(self) -> None:
        # Holm por pasos: tras el primer no rechazo, los siguientes tampoco se rechazan.
        raw, headline = self._fast_slow()
        pairwise = {
            "comparisons": [_paired("fast", "slow", raw=(0.02, 0.05), holm=(0.01, 0.06), rejects=False)]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual([row["model"] for row in partial["frontier"]], ["fast"])

    def test_a_holm_rejection_still_decides_the_axis(self) -> None:
        raw, headline = self._fast_slow()
        pairwise = {
            "comparisons": [_paired("fast", "slow", raw=(0.02, 0.05), holm=(0.01, 0.06), rejects=True)]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(sorted(row["model"] for row in partial["frontier"]), ["fast", "slow"])

    def test_a_paired_interval_over_too_few_works_does_not_decide(self) -> None:
        # adv_pareto.py: con una obra en común el bootstrap colapsa a un punto.
        raw = [
            _raw("a", 2.00, selected_fit=10.0, evaluation=1.0),
            _raw("b", 2.01, selected_fit=10.0, evaluation=1.0),
        ]
        headline = _headline({"a": (1.0, 3.0), "b": (1.0, 3.0)})
        pairwise = {
            "comparisons": [
                _paired("a", "b", raw=(-0.00999, -0.00999), holm=(-0.00999, -0.00999), rejects=True, n_pairs=1)
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(sorted(row["model"] for row in partial["frontier"]), ["a", "b"])
        self.assertEqual(partial["nll_paired_pairs"], 0)
        self.assertGreater(partial["nll_paired_min_works"], 1)

    def test_a_dominance_cycle_never_empties_the_frontier(self) -> None:
        # adv_pareto.py: cada par sobre su propia intersección de obras da A>B>C>A.
        rows = [{"model": name, "nll": 2.0, "nll_interval": [1.9, 2.1], "fit": 10.0} for name in "ABC"]
        paired = {"nll": {("A", "B"): (-0.2, -0.1), ("B", "C"): (-0.2, -0.1), ("A", "C"): (0.1, 0.2)}}

        front = pareto_front(rows, minimize=("nll", "fit"), maximize=(), paired_intervals=paired)

        self.assertEqual([row["model"] for row in front], ["A", "B", "C"])

    def test_a_row_dominated_from_outside_its_cycle_still_leaves_the_frontier(self) -> None:
        rows = [{"model": name, "nll": 2.0, "nll_interval": [1.9, 2.1], "fit": 10.0} for name in "ABCD"]
        paired = {
            "nll": {
                ("A", "B"): (-0.2, -0.1),
                ("B", "C"): (-0.2, -0.1),
                ("A", "C"): (0.1, 0.2),
                ("D", "A"): (-0.2, -0.1),
                ("D", "B"): (-0.1, 0.1),
                ("D", "C"): (-0.1, 0.1),
            }
        }

        front = pareto_front(rows, minimize=("nll", "fit"), maximize=(), paired_intervals=paired)

        self.assertEqual([row["model"] for row in front], ["B", "C", "D"])

    def test_runner_reports_dominance_cycles(self) -> None:
        raw = [_raw(name, 2.0, selected_fit=10.0, evaluation=1.0) for name in "ABC"]
        headline = _headline({name: (1.9, 2.1) for name in "ABC"})
        pairwise = {
            "comparisons": [
                _paired("A", "B", raw=(-1.0, -0.99), holm=(-1.0, -0.99), rejects=True, n_pairs=20),
                _paired("A", "C", raw=(0.99, 1.0), holm=(0.99, 1.0), rejects=True, n_pairs=20),
                _paired("B", "C", raw=(-1.0, -0.99), holm=(-1.0, -0.99), rejects=True, n_pairs=20),
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(sorted(row["model"] for row in partial["frontier"]), ["A", "B", "C"])
        self.assertEqual(partial["dominance_cycles"], [["A", "B", "C"]])


class WorkMeanParetoTests(unittest.TestCase):
    """Aclaración 2: en la frontera también decide la media por obra, no la ponderada."""

    def test_the_work_mean_decision_interval_decides_not_the_token_weighted_one(self) -> None:
        raw = [
            _raw("cheap", 2.04, selected_fit=1.0, evaluation=1.0),
            _raw("good", 2.01, selected_fit=10.0, evaluation=1.0),
        ]
        headline = _headline({"cheap": (1.9, 2.2), "good": (1.85, 2.15)})
        pairwise = {
            "comparisons": [
                {
                    "model_a": "cheap",
                    "model_b": "good",
                    "status": "ok",
                    "n_pairs": 40,
                    # Ponderada: «cheap» es peor; por obra, empate.
                    "token_weighted_holm_ci": [0.02, 0.04],
                    "token_weighted_holm_decision_ci": [0.02, 0.04],
                    "token_weighted_holm_rejects_zero": True,
                    "work_mean_holm_ci": [-0.01, 0.03],
                    "work_mean_holm_decision_ci": [-0.01, 0.03],
                    "work_mean_holm_rejects_zero": False,
                }
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual([row["model"] for row in partial["frontier"]], ["cheap"])
        self.assertIn("work_mean", partial["nll_paired_interval"])

    def test_marginals_use_the_work_mean_estimate(self) -> None:
        raw = [
            _raw("cheap", 2.04, selected_fit=1.0, evaluation=1.0),
            _raw("good", 2.01, selected_fit=10.0, evaluation=1.0),
        ]
        headline = {
            "estimates": [
                # Ponderada separa a «good»; por obra los intervalos se traslapan.
                {"model": "cheap", "frac": 1.0, "estimand": "token_weighted_nll", "point_estimate": 2.04, "bootstrap_95_ci": [2.03, 2.05]},
                {"model": "good", "frac": 1.0, "estimand": "token_weighted_nll", "point_estimate": 2.01, "bootstrap_95_ci": [2.00, 2.02]},
                {"model": "cheap", "frac": 1.0, "estimand": "work_mean_nll", "point_estimate": 2.10, "bootstrap_95_ci": [1.9, 2.3]},
                {"model": "good", "frac": 1.0, "estimand": "work_mean_nll", "point_estimate": 2.08, "bootstrap_95_ci": [1.9, 2.3]},
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual([row["model"] for row in partial["frontier"]], ["cheap"])
        self.assertEqual(partial["axes"][0], "work_mean_test_nll")
        cheap = next(row for row in partial["candidates"] if row["model"] == "cheap")
        self.assertAlmostEqual(cheap["work_mean_test_nll"], 2.10)
        self.assertEqual(cheap["work_mean_test_nll_interval"], [1.9, 2.3])

    def test_pairs_without_paired_works_are_listed_as_not_deciding(self) -> None:
        raw = [_raw("a", 2.0, selected_fit=10.0, evaluation=1.0), _raw("b", 2.0, selected_fit=10.0, evaluation=1.0)]
        headline = _headline({"a": (1.9, 2.1), "b": (1.9, 2.1)})
        pairwise = {
            "comparisons": [
                {
                    "model_a": "a",
                    "model_b": "b",
                    "status": "unavailable",
                    "reason": "no_paired_finite_works",
                    "n_pairs": 0,
                    "work_mean_holm_decision_ci": None,
                    "work_mean_holm_rejects_zero": None,
                }
            ]
        }

        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, headline, pairwise)[
            "predictive_cost_partial_frontier"
        ]

        self.assertEqual(
            partial["nll_paired_pairs_skipped"],
            [{"pair": ["a", "b"], "reason": "no_paired_finite_works", "n_pairs": 0}],
        )

    def test_the_time_tolerance_band_is_declared_in_the_output(self) -> None:
        raw = [_raw("a", 2.0, selected_fit=100.0, evaluation=1.0)]
        partial = _build_pareto_summary(raw, {"status": "not_evaluated"}, _headline({"a": (1.9, 2.1)}))[
            "predictive_cost_partial_frontier"
        ]

        band = partial["time_interval"]
        self.assertEqual(band["relative_tolerance"], 0.10)
        self.assertIn("min", band["definition"])
        self.assertIn("max", band["definition"])
        # El preregistro sólo dice «la incertidumbre de cada medición de costo»: el
        # 10 % es una decisión de diseño y la salida no debe hacerla pasar por preregistrada.
        self.assertIn("not_in_preregistration", band["status"])
        # La documentación nombrada existe y declara la banda.
        from pathlib import Path

        doc = Path(__file__).resolve().parents[1] / band["documented_in"]
        self.assertTrue(doc.is_file(), doc)
        text = doc.read_text(encoding="utf-8")
        self.assertIn("PARETO_TIME_RELATIVE_TOLERANCE", text)
        self.assertIn("no está en el preregistro", text)
        candidate = partial["candidates"][0]
        self.assertAlmostEqual(candidate["mean_selected_fit_wall_clock_s_interval"][0], 90.0)
        self.assertAlmostEqual(candidate["mean_selected_fit_wall_clock_s_interval"][1], 110.0)


if __name__ == "__main__":
    unittest.main()
