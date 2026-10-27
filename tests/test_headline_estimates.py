"""C7 (auditoría 2026-10-26): cada cifra titular con el IC de su propio estimando.

La NLL titular ponderada por token es Σll/Σn; su IC sale de remuestrear obras y
recalcular ese cociente, no de la media por obra. La media por obra es otro
estimando y se reporta aparte, con su propio IC.
"""

from __future__ import annotations

import math
import unittest

from Comparacion.statistics import headline_estimates, pairwise_model_comparisons


def _row(model: str, work: str, n_tokens: int, nll: float, *, model_seed: int = 1, frac: float = 1.0):
    return {
        "model": model,
        "frac": frac,
        "split_seed": 7,
        "data_seed": 1,
        "model_seed": model_seed,
        "canonical_work_id": work,
        "piece_id": f"{work}.xml",
        "n_tokens": n_tokens,
        "nll_per_token": nll,
    }


def _estimate(payload: dict, model: str, estimand: str, frac: float = 1.0) -> dict:
    (entry,) = [
        item for item in payload["estimates"]
        if item["model"] == model and item["frac"] == frac and item["estimand"] == estimand
    ]
    return entry


class HeadlineEstimateTests(unittest.TestCase):
    def test_token_weighted_and_work_mean_are_distinct_estimands(self) -> None:
        rows = [_row("m", "short", 10, 1.0), _row("m", "long", 90, 2.0)]

        payload = headline_estimates(rows, bootstrap_samples=500, seed=0)

        token = _estimate(payload, "m", "token_weighted_nll")
        work = _estimate(payload, "m", "work_mean_nll")
        self.assertAlmostEqual(token["point_estimate"], 1.9)
        self.assertAlmostEqual(work["point_estimate"], 1.5)
        self.assertEqual(token["ci_method"], "work_cluster_bootstrap_of_ratio_sum_ll_over_sum_n")
        self.assertEqual(token["n_works"], 2)
        # Remuestrear obras sólo puede dar cocientes entre las NLL de las obras.
        low, high = token["bootstrap_95_ci"]
        self.assertGreaterEqual(low, 1.0)
        self.assertLessEqual(high, 2.0)

    def test_perplexity_interval_is_the_exponential_of_the_token_weighted_interval(self) -> None:
        rows = [_row("m", f"w{index}", 20 + index, 1.0 + 0.1 * index) for index in range(8)]

        payload = headline_estimates(rows, bootstrap_samples=500, seed=0)

        token = _estimate(payload, "m", "token_weighted_nll")
        perplexity = _estimate(payload, "m", "token_weighted_perplexity")
        self.assertAlmostEqual(perplexity["point_estimate"], math.exp(token["point_estimate"]))
        self.assertAlmostEqual(perplexity["bootstrap_95_ci"][0], math.exp(token["bootstrap_95_ci"][0]))
        self.assertAlmostEqual(perplexity["bootstrap_95_ci"][1], math.exp(token["bootstrap_95_ci"][1]))
        self.assertLess(token["bootstrap_95_ci"][0], token["point_estimate"])
        self.assertGreater(token["bootstrap_95_ci"][1], token["point_estimate"])

    def test_repeated_runs_are_averaged_within_a_work_not_counted_as_works(self) -> None:
        rows = [
            _row("m", "a", 10, 1.0, model_seed=1),
            _row("m", "a", 10, 3.0, model_seed=2),
            _row("m", "b", 30, 2.0, model_seed=1),
            _row("m", "b", 30, 2.0, model_seed=2),
        ]

        token = _estimate(headline_estimates(rows, bootstrap_samples=200, seed=0), "m", "token_weighted_nll")

        self.assertEqual(token["n_works"], 2)
        self.assertEqual(token["n_runs"], 2)
        self.assertAlmostEqual(token["point_estimate"], (10 * 2.0 + 30 * 2.0) / 40)

    def test_identical_works_collapse_the_interval(self) -> None:
        rows = [_row("m", f"w{index}", 25, 1.7) for index in range(5)]

        token = _estimate(headline_estimates(rows, bootstrap_samples=200, seed=0), "m", "token_weighted_nll")

        self.assertAlmostEqual(token["bootstrap_95_ci"][0], 1.7)
        self.assertAlmostEqual(token["bootstrap_95_ci"][1], 1.7)

    def test_each_fraction_gets_its_own_estimate(self) -> None:
        rows = [_row("m", "a", 10, 1.0, frac=0.5), _row("m", "a", 10, 2.0)]

        payload = headline_estimates(rows, bootstrap_samples=50, seed=0)

        self.assertAlmostEqual(_estimate(payload, "m", "token_weighted_nll", 0.5)["point_estimate"], 1.0)
        self.assertAlmostEqual(_estimate(payload, "m", "token_weighted_nll", 1.0)["point_estimate"], 2.0)


class PairwiseTokenWeightedTests(unittest.TestCase):
    def test_comparison_reports_the_token_weighted_difference_with_its_own_interval(self) -> None:
        rows = []
        for index in range(6):
            n_tokens = 10 + 20 * index
            rows.append(_row("a", f"w{index}", n_tokens, 1.0 + 0.05 * index))
            rows.append(_row("b", f"w{index}", n_tokens, 1.2 + 0.01 * index))

        comparison = pairwise_model_comparisons(rows, bootstrap_samples=300, seed=0)["comparisons"][0]

        total = sum(10 + 20 * index for index in range(6))
        expected = sum((10 + 20 * index) * (0.04 * index - 0.2) for index in range(6)) / total
        self.assertEqual(comparison["estimand"], "mean_over_works_of_work_nll_difference")
        self.assertEqual(comparison["token_weighted_estimand"], "token_weighted_nll_difference")
        self.assertAlmostEqual(comparison["token_weighted_difference"], expected)
        low, high = comparison["token_weighted_bootstrap_95_ci"]
        self.assertLessEqual(low, expected)
        self.assertGreaterEqual(high, expected)


class HolmAdjustedIntervalTests(unittest.TestCase):
    """Preregistro, sección B: la regla de decisión usa el IC «tras Holm»."""

    def _rows(self) -> list[dict]:
        import numpy as np

        rng = np.random.default_rng(3)
        rows = []
        for index in range(30):
            n_tokens = 20 + 3 * index
            base = 2.0 + 0.3 * rng.standard_normal()
            rows.append(_row("a", f"w{index}", n_tokens, base))
            rows.append(_row("b", f"w{index}", n_tokens, base + 0.5 + 0.01 * rng.standard_normal()))
            rows.append(_row("c", f"w{index}", n_tokens, base + 0.02 * rng.standard_normal()))
        return rows

    def _comparisons(self) -> list[dict]:
        return pairwise_model_comparisons(self._rows(), bootstrap_samples=2000, seed=0)["comparisons"]

    def test_each_comparison_carries_a_step_down_holm_interval(self) -> None:
        comparisons = self._comparisons()
        by_pair = {(item["model_a"], item["model_b"]): item for item in comparisons}

        alphas = sorted(item["token_weighted_holm_alpha"] for item in comparisons)
        self.assertEqual(alphas, sorted([0.05 / 3, 0.05 / 2, 0.05 / 1]))
        for item in comparisons:
            with self.subTest(pair=(item["model_a"], item["model_b"])):
                raw_low, raw_high = item["token_weighted_bootstrap_95_ci"]
                low, high = item["token_weighted_holm_ci"]
                self.assertLessEqual(low, raw_low)
                self.assertGreaterEqual(high, raw_high)
        self.assertTrue(by_pair[("a", "b")]["token_weighted_holm_rejects_zero"])
        self.assertTrue(by_pair[("b", "c")]["token_weighted_holm_rejects_zero"])
        self.assertFalse(by_pair[("a", "c")]["token_weighted_holm_rejects_zero"])
        self.assertEqual(by_pair[("a", "c")]["token_weighted_holm_alpha"], 0.05)

    def test_holm_steps_down_in_order_of_the_bootstrap_p_value(self) -> None:
        ordered = sorted(self._comparisons(), key=lambda item: item["token_weighted_holm_alpha"])

        decisions = [item["token_weighted_holm_rejects_zero"] for item in ordered]
        # Una vez que una comparación no se rechaza, ninguna posterior se rechaza.
        self.assertEqual(decisions, sorted(decisions, reverse=True))
        p_values = [item["token_weighted_bootstrap_p_value"] for item in ordered]
        self.assertEqual(p_values, sorted(p_values))

    def test_a_rejection_matches_a_holm_interval_that_excludes_zero(self) -> None:
        for item in self._comparisons():
            with self.subTest(pair=(item["model_a"], item["model_b"])):
                low, high = item["token_weighted_holm_ci"]
                if item["token_weighted_holm_rejects_zero"]:
                    self.assertTrue(low > 0.0 or high < 0.0)


if __name__ == "__main__":
    unittest.main()
