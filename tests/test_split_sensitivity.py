from __future__ import annotations

import unittest

from Comparacion.cli import build_parser
from Comparacion.statistics import (
    pairwise_model_comparisons,
    summarize_split_variation,
)


def _rows(split_seed: int, offset: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for work_index in range(4):
        for model, penalty in (("vomm", 0.0), ("finite_hmm", 0.2)):
            rows.append(
                {
                    "model": model,
                    "frac": 1.0,
                    "split_seed": split_seed,
                    "data_seed": 1,
                    "model_seed": 1,
                    "canonical_work_id": f"work-{work_index}",
                    "nll_per_token": 2.0 + penalty + offset + 0.05 * work_index,
                }
            )
    return rows


class SplitSeedCliTests(unittest.TestCase):
    def test_split_seed_flag_is_parsed(self) -> None:
        args = build_parser().parse_args(["--split-seed", "17"])

        self.assertEqual(args.split_seed, 17)


class SplitVariationTests(unittest.TestCase):
    def test_works_are_not_independent_observations_across_splits(self) -> None:
        rows = _rows(7, 0.0) + _rows(17, 0.1)
        summary = summarize_split_variation(rows)

        self.assertEqual(summary["n_split_seeds"], 2)
        self.assertEqual(summary["split_seeds"], [7, 17])
        # Cuatro obras vistas en dos particiones no son ocho observaciones.
        self.assertEqual(summary["n_distinct_works"], 4)
        self.assertEqual(summary["n_work_split_observations"], 8)
        self.assertEqual(summary["independence_unit"], "canonical_work_within_split_seed")

    def test_between_work_and_between_split_variation_are_reported_apart(self) -> None:
        rows = _rows(7, 0.0) + _rows(17, 0.1)
        summary = summarize_split_variation(rows)
        vomm = summary["models"]["vomm"]

        self.assertAlmostEqual(vomm["between_split_std"], 0.05, places=6)
        self.assertGreater(vomm["between_work_std"], 0.0)
        self.assertEqual(len(vomm["per_split_mean_nll"]), 2)

    def test_a_single_split_seed_reports_no_between_split_variation(self) -> None:
        summary = summarize_split_variation(_rows(7, 0.0))

        self.assertEqual(summary["n_split_seeds"], 1)
        self.assertIsNone(summary["models"]["vomm"]["between_split_std"])
        self.assertEqual(summary["status"], "single_split_seed")


class PairwiseAcrossSplitsTests(unittest.TestCase):
    def test_comparisons_are_computed_per_split_seed(self) -> None:
        rows = _rows(7, 0.0) + _rows(17, 0.1)
        payload = pairwise_model_comparisons(rows, bootstrap_samples=32, seed=1)

        self.assertEqual(payload["split_seeds"], [7, 17])
        self.assertEqual(payload["pooling"], "per_split_seed_never_pooled_across_splits")
        by_split = payload["comparisons_by_split_seed"]
        self.assertEqual(sorted(by_split), ["17", "7"])
        for split_seed, comparisons in by_split.items():
            with self.subTest(split_seed=split_seed):
                self.assertEqual(comparisons[0]["n_pairs"], 4)

    def test_rows_without_split_seed_keep_the_previous_shape(self) -> None:
        rows = [
            {key: value for key, value in row.items() if key != "split_seed"}
            for row in _rows(7, 0.0)
        ]
        payload = pairwise_model_comparisons(rows, bootstrap_samples=32, seed=1)

        self.assertEqual(payload["split_seeds"], [])
        self.assertEqual(payload["comparisons"][0]["n_pairs"], 4)


if __name__ == "__main__":
    unittest.main()
