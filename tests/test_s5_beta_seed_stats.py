"""Estadística del criterio (b) en scripts/auditoria_2026_10_26/hdp/s5_beta_seed_v3.py."""

from __future__ import annotations

import importlib.util
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

HDP_DIR = Path(__file__).resolve().parents[1] / "scripts" / "auditoria_2026_10_26" / "hdp"
SCRIPT = HDP_DIR / "s5_beta_seed_v3.py"
OUTPUT = HDP_DIR / "s5_beta_seed_v3_salida_20261026.txt"


def _load_script():
    spec = importlib.util.spec_from_file_location("s5_beta_seed_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CorrelationEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_script()

    def test_fisher_interval_matches_the_textbook_formula(self) -> None:
        low, high = self.script.fisher_interval(-0.156, 50)
        self.assertAlmostEqual(low, -0.416, places=3)
        self.assertAlmostEqual(high, 0.128, places=3)

    def test_equivalence_test_needs_both_one_sided_tests(self) -> None:
        # TOST para |ρ| < 0.3 con la z de Fisher: p = máximo de las dos colas.
        self.assertAlmostEqual(self.script.equivalence_pvalue(-0.156, 50, bound=0.3), 0.148, places=3)
        self.assertAlmostEqual(self.script.equivalence_pvalue(0.071, 50, bound=0.3), 0.051, places=3)
        self.assertLess(self.script.equivalence_pvalue(0.0, 400, bound=0.3), 0.001)

    def test_too_few_seeds_give_nan_without_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(np.isnan(self.script.fisher_interval(0.5, 3)).all())
            self.assertTrue(np.isnan(self.script.equivalence_pvalue(0.5, 3)))

    def test_verdict_does_not_call_a_small_point_estimate_independence(self) -> None:
        verdict = self.script.verdict
        self.assertEqual(verdict(p_value=0.28, equivalence_p=0.148), "sin evidencia de dependencia ni de |r| < 0.3")
        self.assertEqual(verdict(p_value=0.60, equivalence_p=0.01), "|r| < 0.3 demostrado (equivalencia)")
        self.assertEqual(verdict(p_value=0.01, equivalence_p=0.40), "dependencia")

    def test_saved_rows_reproduce_the_reported_correlations(self) -> None:
        sections = self.script.parse_rows(OUTPUT.read_text(encoding="utf8"))

        self.assertEqual(len(sections), 2)
        for (seeds, n_large, mean_k), expected_r in zip(sections, (-0.156, 0.071)):
            self.assertEqual(seeds, list(range(7, 57)))
            self.assertAlmostEqual(stats.pearsonr(n_large, mean_k).statistic, expected_r, places=3)

    def test_summary_reports_interval_equivalence_and_verdict(self) -> None:
        seeds, n_large, mean_k = self.script.parse_rows(OUTPUT.read_text(encoding="utf8"))[0]
        text = "\n".join(self.script.summary(np.asarray(n_large, float), np.asarray(mean_k, float)))

        self.assertIn("IC 95 % de Fisher para r = [-0.416, 0.128]", text)
        self.assertIn("TOST |r| < 0.3: p = 0.148", text)
        self.assertIn("veredicto: sin evidencia de dependencia ni de |r| < 0.3", text)

    def test_script_does_not_silence_every_warning(self) -> None:
        self.assertNotIn('warnings.filterwarnings("ignore")', SCRIPT.read_text(encoding="utf8"))


if __name__ == "__main__":
    unittest.main()
