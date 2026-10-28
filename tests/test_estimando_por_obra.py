"""Aclaración 2 al preregistro: decide la media por obra de la diferencia de NLL.

Cada obra pesa lo mismo; su IC es bootstrap por obra ajustado por Holm por pasos
sobre la familia preregistrada.  La diferencia ponderada por tokens es
secundaria.  Tras el primer no rechazo de Holm, el IC al nivel α/(m − r) puede
excluir 0 sin que la prueba se rechace: el IC que decide lo incluye.
"""

from __future__ import annotations

import unittest

import numpy as np

from Comparacion.statistics import _apply_bootstrap_holm, pairwise_model_comparisons
from tests.test_familia_holm import _four_model_rows


class WorkMeanHolmTests(unittest.TestCase):
    def test_every_comparison_carries_the_holm_adjusted_ci_of_the_work_mean(self) -> None:
        payload = pairwise_model_comparisons(_four_model_rows(), bootstrap_samples=300, seed=0)
        alphas = sorted(item["work_mean_holm_alpha"] for item in payload["comparisons"])
        self.assertEqual(alphas, sorted(0.05 / m for m in range(1, 7)))
        for item in payload["comparisons"]:
            with self.subTest(pair=(item["model_a"], item["model_b"])):
                self.assertEqual(item["work_mean_holm_status"], "ok")
                low, high = item["work_mean_holm_ci"]
                self.assertLessEqual(low, item["mean_difference"])
                self.assertGreaterEqual(high, item["mean_difference"])
                # El IC de Holm es al menos tan ancho como el del 95 % sin ajustar.
                plain_low, plain_high = item["bootstrap_95_ci"]
                self.assertLessEqual(low, plain_low + 1e-12)
                self.assertGreaterEqual(high, plain_high - 1e-12)
        family = payload["holm_family"]["work_mean_ci"]
        self.assertEqual(family["family_size"], 6)
        self.assertEqual(family["n_evaluable"], 6)

    def test_the_work_mean_is_the_declared_decision_estimand(self) -> None:
        payload = pairwise_model_comparisons(_four_model_rows(), bootstrap_samples=50, seed=0)
        decision = payload["decision"]
        self.assertEqual(decision["estimand"], "mean_over_works_of_work_nll_difference")
        self.assertEqual(decision["interval"], "work_mean_holm_decision_ci")
        self.assertEqual(decision["rejects_zero"], "work_mean_holm_rejects_zero")
        self.assertEqual(decision["secondary_estimand"], "token_weighted_nll_difference")

    def test_a_model_without_rows_keeps_the_family_and_is_not_evaluable(self) -> None:
        rows = [row for row in _four_model_rows() if row["model"] != "d"]
        payload = pairwise_model_comparisons(
            rows, bootstrap_samples=200, seed=0, expected_models=["a", "b", "c", "d"]
        )
        by_pair = {(item["model_a"], item["model_b"]): item for item in payload["comparisons"]}
        for pair in [("a", "d"), ("b", "d"), ("c", "d")]:
            with self.subTest(pair=pair):
                self.assertEqual(by_pair[pair]["work_mean_holm_status"], "not_evaluable")
                self.assertIsNone(by_pair[pair]["work_mean_holm_ci"])
                self.assertIsNone(by_pair[pair]["work_mean_holm_decision_ci"])
                self.assertIsNone(by_pair[pair]["work_mean_holm_rejects_zero"])
        alphas = sorted(
            item["work_mean_holm_alpha"]
            for item in payload["comparisons"]
            if item["work_mean_holm_alpha"] is not None
        )
        self.assertEqual(alphas, sorted([0.05 / 6, 0.05 / 5, 0.05 / 4]))
        self.assertEqual(
            payload["holm_family"]["work_mean_ci"]["not_evaluable"], [["a", "d"], ["b", "d"], ["c", "d"]]
        )


def _two_file_work_rows(nll_by_model: dict[str, tuple[float, float]]) -> list[dict]:
    """Una obra con dos archivos de 10 y 100 notas; otra obra de un archivo."""

    rows = []
    for model, (short_nll, long_nll) in nll_by_model.items():
        for piece, n_tokens, nll in (("w0_a", 10, short_nll), ("w0_b", 100, long_nll)):
            rows.append(
                {"model": model, "frac": 1.0, "canonical_work_id": "w0", "piece_id": piece,
                 "split_seed": 0, "data_seed": 0, "model_seed": 0,
                 "n_tokens": n_tokens, "test_nll": nll, "log_likelihood": -nll * n_tokens}
            )
        rows.append(
            {"model": model, "frac": 1.0, "canonical_work_id": "w1", "piece_id": "w1_a",
             "split_seed": 0, "data_seed": 0, "model_seed": 0,
             "n_tokens": 50, "test_nll": 2.0, "log_likelihood": -100.0}
        )
    return rows


class PerNoteNllOfAWorkTests(unittest.TestCase):
    """La NLL de una obra es su NLL por nota (Σll/Σn), no la media de sus archivos."""

    def test_the_paired_difference_uses_the_per_note_nll_of_each_work(self) -> None:
        rows = _two_file_work_rows({"a": (1.0, 3.0), "b": (2.0, 2.0)})
        payload = pairwise_model_comparisons(rows, bootstrap_samples=50, seed=0)
        (item,) = payload["comparisons"]
        per_note_a = (10 * 1.0 + 100 * 3.0) / 110
        # w0 difiere en per_note_a - 2; w1 no difiere; media sobre 2 obras.
        self.assertAlmostEqual(item["mean_difference"], (per_note_a - 2.0) / 2, places=12)

    def test_the_paired_estimand_matches_the_pareto_marginal(self) -> None:
        from Comparacion.statistics import headline_estimates

        rows = _two_file_work_rows({"a": (5.0, 1.0), "b": (1.0, 1.5)})
        paired = pairwise_model_comparisons(rows, bootstrap_samples=50, seed=0)["comparisons"][0]
        marginal = {
            item["model"]: item["point_estimate"]
            for item in headline_estimates(rows, bootstrap_samples=50, seed=0)["estimates"]
            if item["estimand"] == "work_mean_nll"
        }
        self.assertAlmostEqual(paired["mean_difference"], marginal["a"] - marginal["b"], places=12)
        # Mismo signo que la NLL por nota: A es mejor en w0 (Σll/Σn).
        self.assertLess(paired["mean_difference"], 0.0)

    def test_repeated_runs_of_a_work_are_averaged_after_the_per_note_ratio(self) -> None:
        rows = _two_file_work_rows({"a": (1.0, 3.0), "b": (2.0, 2.0)})
        again = [dict(row, model_seed=1) for row in rows]
        payload = pairwise_model_comparisons(rows + again, bootstrap_samples=50, seed=0)
        per_note_a = (10 * 1.0 + 100 * 3.0) / 110
        self.assertAlmostEqual(
            payload["comparisons"][0]["mean_difference"], (per_note_a - 2.0) / 2, places=12
        )

    def test_a_single_file_work_without_token_count_keeps_its_nll(self) -> None:
        rows = [
            {"model": m, "frac": 1.0, "canonical_work_id": w, "split_seed": 0, "data_seed": 0,
             "model_seed": 0, "test_nll": v}
            for m, w, v in (("a", "w0", 1.0), ("b", "w0", 2.0), ("a", "w1", 1.0), ("b", "w1", 3.0))
        ]
        payload = pairwise_model_comparisons(rows, bootstrap_samples=50, seed=0)
        self.assertAlmostEqual(payload["comparisons"][0]["mean_difference"], -1.5, places=12)


class HeadlineRoleTests(unittest.TestCase):
    def test_headline_estimates_mark_the_work_mean_as_the_one_that_decides(self) -> None:
        from Comparacion.statistics import headline_estimates

        payload = headline_estimates(_four_model_rows(), bootstrap_samples=50, seed=0)
        self.assertEqual(payload["decision_estimand"], "work_mean_nll")
        roles = {item["estimand"]: item["role"] for item in payload["estimates"]}
        self.assertEqual(roles["work_mean_nll"], "decides")
        self.assertEqual(roles["token_weighted_nll"], "secondary")
        self.assertEqual(roles["token_weighted_perplexity"], "secondary")


class DecisionIntervalAfterHolmStopsTests(unittest.TestCase):
    def _comparisons(self) -> tuple[list[dict], dict[int, np.ndarray]]:
        # p bootstrap 0.03 y 0.04 con m = 2: la primera no pasa α/2 = 0.025 y Holm
        # se detiene; la segunda, a α/1 = 0.05, tiene un IC que excluye 0.
        first = np.concatenate([np.full(15, -1.0), np.full(985, 1.0)])
        second = np.concatenate([np.full(20, -1.0), np.full(980, 1.0)])
        comparisons = [{"model_a": "a", "model_b": "b"}, {"model_a": "a", "model_b": "c"}]
        return comparisons, {0: first, 1: second}

    def test_the_decision_interval_includes_zero_when_holm_does_not_reject(self) -> None:
        comparisons, draws = self._comparisons()
        _apply_bootstrap_holm(comparisons, draws, prefix="work_mean")
        second = comparisons[1]
        self.assertFalse(second["work_mean_holm_rejects_zero"])
        # El IC al nivel α/(m − r) excluye 0, pero no es el que decide.
        self.assertGreater(second["work_mean_holm_ci"][0], 0.0)
        low, high = second["work_mean_holm_decision_ci"]
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.0)
        self.assertFalse(comparisons[0]["work_mean_holm_rejects_zero"])

    def test_a_rejected_pair_keeps_its_interval_for_the_decision(self) -> None:
        comparisons = [{"model_a": "a", "model_b": "b"}]
        draws = {0: np.full(1000, 0.5)}
        _apply_bootstrap_holm(comparisons, draws, prefix="work_mean")
        self.assertTrue(comparisons[0]["work_mean_holm_rejects_zero"])
        self.assertEqual(comparisons[0]["work_mean_holm_decision_ci"], comparisons[0]["work_mean_holm_ci"])

    def test_the_payload_says_which_interval_to_read(self) -> None:
        payload = pairwise_model_comparisons(_four_model_rows(), bootstrap_samples=50, seed=0)
        usage = payload["holm_family"]["ci_usage"]
        self.assertIn("decision_ci", usage)
        for item in payload["comparisons"]:
            self.assertIn("token_weighted_holm_decision_ci", item)


if __name__ == "__main__":
    unittest.main()
