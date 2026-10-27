"""La familia de Holm es la del preregistro y no se encoge cuando faltan datos.

Preregistro (sección B): «Holm sobre las seis comparaciones».  Si a una prueba le
faltan datos (por ejemplo, el n_tokens de una sola obra), esa prueba sigue
contando en la familia y se reporta como no evaluable; el tamaño de la familia y
sus pruebas quedan escritos en la salida.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Comparacion.statistics import pairwise_model_comparisons


def _row(model: str, work: str, n_tokens, nll: float) -> dict:
    row = {
        "model": model,
        "frac": 1.0,
        "split_seed": 7,
        "data_seed": 1,
        "model_seed": 1,
        "canonical_work_id": work,
        "piece_id": f"{work}.xml",
        "nll_per_token": nll,
    }
    if n_tokens is not None:
        row["n_tokens"] = n_tokens
    return row


def _four_model_rows(*, drop_tokens_of: tuple[str, str] | None = None) -> list[dict]:
    rng = np.random.default_rng(5)
    offsets = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.5}
    rows = []
    for index in range(30):
        base = 2.0 + 0.3 * rng.standard_normal()
        for model, offset in offsets.items():
            work = f"w{index}"
            n_tokens = None if drop_tokens_of == (model, work) else 20 + 3 * index
            rows.append(_row(model, work, n_tokens, base + offset + 0.01 * rng.standard_normal()))
    return rows


class TokenWeightedHolmFamilyTests(unittest.TestCase):
    def test_missing_tokens_of_one_work_does_not_shrink_the_family(self) -> None:
        payload = pairwise_model_comparisons(
            _four_model_rows(drop_tokens_of=("d", "w3")), bootstrap_samples=500, seed=0
        )
        comparisons = payload["comparisons"]
        self.assertEqual(len(comparisons), 6)

        evaluable = [item for item in comparisons if item["model_a"] != "d" and item["model_b"] != "d"]
        missing = [item for item in comparisons if "d" in (item["model_a"], item["model_b"])]
        self.assertEqual(len(missing), 3)
        alphas = sorted(item["token_weighted_holm_alpha"] for item in evaluable)
        # Seis en la familia: el primer paso usa α/6, no α/3.
        self.assertAlmostEqual(alphas[0], 0.05 / 6)
        self.assertEqual(alphas, sorted([0.05 / 6, 0.05 / 5, 0.05 / 4]))
        for item in missing:
            with self.subTest(pair=(item["model_a"], item["model_b"])):
                self.assertEqual(item["token_weighted_holm_status"], "not_evaluable")
                self.assertIsNone(item["token_weighted_holm_rejects_zero"])
                self.assertIsNone(item["token_weighted_holm_ci"])
        for item in evaluable:
            self.assertEqual(item["token_weighted_holm_status"], "ok")

    def test_output_records_the_family_size_and_its_tests(self) -> None:
        payload = pairwise_model_comparisons(
            _four_model_rows(drop_tokens_of=("d", "w3")), bootstrap_samples=200, seed=0
        )
        family = payload["holm_family"]
        self.assertEqual(family["family_alpha"], 0.05)
        expected_pairs = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"], ["c", "d"]]
        self.assertEqual(family["tests"], expected_pairs)
        token = family["token_weighted_ci"]
        self.assertEqual(token["family_size"], 6)
        self.assertEqual(token["n_evaluable"], 3)
        self.assertEqual(token["not_evaluable"], [["a", "d"], ["b", "d"], ["c", "d"]])
        wilcoxon = family["wilcoxon"]
        self.assertEqual(wilcoxon["family_size"], 6)
        self.assertEqual(wilcoxon["n_evaluable"], 6)
        self.assertEqual(wilcoxon["not_evaluable"], [])

    def test_complete_data_keeps_the_full_family(self) -> None:
        payload = pairwise_model_comparisons(_four_model_rows(), bootstrap_samples=200, seed=0)
        alphas = sorted(item["token_weighted_holm_alpha"] for item in payload["comparisons"])
        self.assertEqual(alphas, sorted(0.05 / m for m in range(1, 7)))
        self.assertEqual(payload["holm_family"]["token_weighted_ci"]["n_evaluable"], 6)


class WilcoxonHolmFamilyTests(unittest.TestCase):
    def test_an_untestable_pair_still_counts_in_the_wilcoxon_family(self) -> None:
        # alpha-beta: una sola diferencia no nula -> Wilcoxon no evaluable.  Diez obras
        # en común, para que los otros dos pares entren a los pasos (Enmienda 1).
        rows = [
            {"model": model, "canonical_work_id": f"w{index}", "frac": 1.0, "test_nll": nll(index)}
            for index in range(1, 11)
            for model, nll in (
                ("alpha", lambda i: 1.0),
                ("beta", lambda i: 2.0 if i == 10 else 1.0),
                ("gamma", lambda i: 2.0 + 2.0 * i),
            )
        ]
        scripted = [
            SimpleNamespace(statistic=1.0, pvalue=0.01),
            SimpleNamespace(statistic=3.0, pvalue=0.02),
        ]
        with patch("Comparacion.statistics.wilcoxon", side_effect=scripted):
            payload = pairwise_model_comparisons(rows, bootstrap_samples=8, seed=0)
        by_pair = {(item["model_a"], item["model_b"]): item for item in payload["comparisons"]}

        self.assertIsNone(by_pair[("alpha", "beta")]["p_value_holm"])
        self.assertEqual(by_pair[("alpha", "beta")]["wilcoxon_holm_status"], "not_evaluable")
        # m = 3 (la familia completa), no 2: 3 × 0.01 y luego max(0.03, 2 × 0.02).
        self.assertAlmostEqual(by_pair[("alpha", "gamma")]["p_value_holm"], 0.03)
        self.assertAlmostEqual(by_pair[("beta", "gamma")]["p_value_holm"], 0.04)
        wilcoxon = payload["holm_family"]["wilcoxon"]
        self.assertEqual(wilcoxon["family_size"], 3)
        self.assertEqual(wilcoxon["n_evaluable"], 2)
        self.assertEqual(wilcoxon["not_evaluable"], [["alpha", "beta"]])


class ExpectedModelsFamilyTests(unittest.TestCase):
    """La familia la fijan los modelos del diseño, no los que dejaron filas."""

    def test_a_model_without_rows_still_counts_in_the_family(self) -> None:
        rows = [row for row in _four_model_rows() if row["model"] != "d"]
        payload = pairwise_model_comparisons(
            rows, bootstrap_samples=200, seed=0, expected_models=["a", "b", "c", "d"]
        )
        by_pair = {(item["model_a"], item["model_b"]): item for item in payload["comparisons"]}
        self.assertEqual(len(by_pair), 6)
        for pair in [("a", "d"), ("b", "d"), ("c", "d")]:
            with self.subTest(pair=pair):
                self.assertEqual(by_pair[pair]["status"], "unavailable")
                self.assertEqual(by_pair[pair]["token_weighted_holm_status"], "not_evaluable")
                self.assertIsNone(by_pair[pair]["p_value_holm"])
        alphas = sorted(
            item["token_weighted_holm_alpha"]
            for item in payload["comparisons"]
            if item["token_weighted_holm_alpha"] is not None
        )
        self.assertEqual(alphas, sorted([0.05 / 6, 0.05 / 5, 0.05 / 4]))
        family = payload["holm_family"]
        self.assertEqual(family["expected_models"], ["a", "b", "c", "d"])
        self.assertEqual(family["wilcoxon"]["family_size"], 6)
        self.assertEqual(family["wilcoxon"]["not_evaluable"], [["a", "d"], ["b", "d"], ["c", "d"]])

    def test_a_model_outside_the_design_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "familia de Holm"):
            pairwise_model_comparisons(
                _four_model_rows(), bootstrap_samples=8, seed=0, expected_models=["a", "b", "c"]
            )

    def test_each_split_keeps_the_full_family(self) -> None:
        rows = []
        for split_seed in (7, 17):
            for row in _four_model_rows():
                if split_seed == 17 and row["model"] == "d":
                    continue  # d no dejó filas en la segunda partición
                rows.append({**row, "split_seed": split_seed})
        payload = pairwise_model_comparisons(rows, bootstrap_samples=100, seed=0)
        nested = payload["comparisons_by_split_seed"]["17"]
        self.assertEqual(len(nested), 6)
        family = payload["holm_family_by_split_seed"]["17"]
        self.assertEqual(family["token_weighted_ci"]["family_size"], 6)
        self.assertEqual(family["token_weighted_ci"]["n_evaluable"], 3)
        self.assertAlmostEqual(
            min(item["token_weighted_holm_alpha"] for item in nested if item["token_weighted_holm_alpha"]),
            0.05 / 6,
        )


class DegeneratePairHolmTests(unittest.TestCase):
    """Un par con pocas obras en común no afloja el α de las demás (ronda 2)."""

    @staticmethod
    def _rows(*, vomm_works: int) -> list[dict]:
        rng = np.random.default_rng(3)
        rows = []
        for index in range(40):
            base = rng.normal(2, 0.3)
            for model, effect in {"finite_hmm": 0.0, "hdp_hmm": 0.02, "transformer": -0.02}.items():
                rows.append(_row(model, f"w{index}", 50, base + effect + rng.normal(0, 0.05)))
        for index in range(vomm_works):
            rows.append(_row("vomm", f"w{index}", 50, 9.0))
        return rows

    def _by_pair(self, vomm_works: int) -> dict:
        payload = pairwise_model_comparisons(
            self._rows(vomm_works=vomm_works),
            bootstrap_samples=4000,
            seed=0,
            expected_models=["finite_hmm", "hdp_hmm", "transformer", "vomm"],
        )
        return payload, {(item["model_a"], item["model_b"]): item for item in payload["comparisons"]}

    def test_a_pair_with_one_common_work_does_not_reject_nor_shift_the_others(self) -> None:
        _, without = self._by_pair(vomm_works=0)
        payload, with_one = self._by_pair(vomm_works=1)
        for prefix in ("work_mean", "token_weighted"):
            for pair in [("finite_hmm", "vomm"), ("hdp_hmm", "vomm"), ("transformer", "vomm")]:
                with self.subTest(prefix=prefix, pair=pair):
                    self.assertEqual(with_one[pair]["n_pairs"], 1)
                    self.assertEqual(with_one[pair][f"{prefix}_holm_status"], "too_few_paired_works")
                    self.assertIsNone(with_one[pair][f"{prefix}_holm_rejects_zero"])
                    self.assertIsNone(with_one[pair][f"{prefix}_holm_alpha"])
            for pair, item in without.items():
                if "vomm" in pair:
                    continue
                with self.subTest(prefix=prefix, pair=pair):
                    self.assertEqual(with_one[pair][f"{prefix}_holm_alpha"], item[f"{prefix}_holm_alpha"])
                    self.assertEqual(
                        with_one[pair][f"{prefix}_holm_rejects_zero"], item[f"{prefix}_holm_rejects_zero"]
                    )
        block = payload["holm_family"]["work_mean_ci"]
        self.assertEqual(block["family_size"], 6)
        self.assertEqual(block["n_evaluable"], 3)
        self.assertEqual(block["min_paired_works"], 10)
        self.assertEqual(
            block["not_evaluable"],
            [["finite_hmm", "vomm"], ["hdp_hmm", "vomm"], ["transformer", "vomm"]],
        )

    def test_too_few_works_do_not_enter_the_wilcoxon_holm_steps(self) -> None:
        # Ronda 3 (r3sw_holm.py): con 9 obras el p exacto bilateral es 2/512 < 0.05/6 y,
        # si entrara a los pasos, aflojaría el p de Holm de las demás.
        _, without = self._by_pair(vomm_works=0)
        payload, with_nine = self._by_pair(vomm_works=9)
        for pair in [("finite_hmm", "vomm"), ("hdp_hmm", "vomm"), ("transformer", "vomm")]:
            with self.subTest(pair=pair):
                item = with_nine[pair]
                self.assertEqual(item["n_pairs"], 9)
                self.assertAlmostEqual(item["p_value"], 2 / 512)
                self.assertIsNone(item["p_value_holm"])
                self.assertEqual(item["wilcoxon_holm_status"], "too_few_paired_works")
        for pair, item in without.items():
            if "vomm" in pair:
                continue
            with self.subTest(pair=pair):
                self.assertEqual(with_nine[pair]["p_value_holm"], item["p_value_holm"])
                self.assertEqual(with_nine[pair]["wilcoxon_holm_status"], "ok")
        block = payload["holm_family"]["wilcoxon"]
        self.assertEqual(block["family_size"], 6)
        self.assertEqual(block["n_evaluable"], 3)
        self.assertEqual(block["min_paired_works"], 10)
        self.assertEqual(
            block["not_evaluable"],
            [["finite_hmm", "vomm"], ["hdp_hmm", "vomm"], ["transformer", "vomm"]],
        )

    def test_ten_common_works_are_enough_to_enter_the_holm_steps(self) -> None:
        _, by_pair = self._by_pair(vomm_works=10)
        self.assertEqual(by_pair[("hdp_hmm", "vomm")]["n_pairs"], 10)
        self.assertEqual(by_pair[("hdp_hmm", "vomm")]["work_mean_holm_status"], "ok")


class RunnerFamilyModelsTests(unittest.TestCase):
    def test_runner_family_is_the_configured_model_set(self) -> None:
        from Comparacion.runner import _comparison_family_models

        self.assertEqual(
            _comparison_family_models(SimpleNamespace(include_vomm_control=True)),
            ["finite_hmm", "hdp_hmm", "transformer", "vomm"],
        )
        # Aclaración 2: sin VOMM la familia no se encoge; la corrida es exploratoria.
        self.assertEqual(
            _comparison_family_models(SimpleNamespace(include_vomm_control=False)),
            ["finite_hmm", "hdp_hmm", "transformer", "vomm"],
        )


if __name__ == "__main__":
    unittest.main()
