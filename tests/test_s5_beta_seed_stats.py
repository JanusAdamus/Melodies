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


class ChainTests(unittest.TestCase):
    """El criterio (b) se evalúa con la cadena del preregistro (400 iteraciones, burn-in 200)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_script()

    def _run_main(self, argv, cache=None):
        import contextlib
        import io
        import json
        import sys
        import tempfile

        calls = []

        class Recorder:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                self.seed = kwargs["seed"]

            def fit(self, observation):
                initial_beta = np.where(np.arange(20) <= self.seed % 5, 0.05, 0.0)
                diagnostics = type("D", (), {"initial_beta": initial_beta})()
                return type("R", (), {"diagnostics": diagnostics, "effective_states": 3 + self.seed % 3})()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            path.write_text("\n".join(json.dumps({"tokens": [0, 1, 2, 3] * 5}) for _ in range(2)), encoding="utf8")
            original, self.script.TruncatedHDPHMM = self.script.TruncatedHDPHMM, Recorder
            old_argv, out = sys.argv, io.StringIO()
            try:
                sys.argv = ["s5_beta_seed_v3.py", str(path), "--n-seeds", "4", *argv]
                with contextlib.redirect_stdout(out), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.script.main()
            finally:
                self.script.TruncatedHDPHMM, sys.argv = original, old_argv
        return calls, out.getvalue()

    def test_default_chain_is_the_preregistered_one(self) -> None:
        calls, text = self._run_main([])
        self.assertTrue(calls)
        for kwargs in calls:
            self.assertEqual((kwargs["n_iters"], kwargs["burn_in"]), (400, 200), kwargs)
        self.assertIn("cadena: K=20, 400 iteraciones, burn-in 200", text)

    def test_chain_can_be_changed_and_is_reported(self) -> None:
        calls, text = self._run_main(["--n-iters", "80", "--burn-in", "40"])
        self.assertEqual({(kwargs["n_iters"], kwargs["burn_in"]) for kwargs in calls}, {(80, 40)})
        self.assertIn("cadena: K=20, 80 iteraciones, burn-in 40", text)
        self.assertIn("no es la cadena del preregistro", text)

    def test_saved_short_chain_output_is_not_presented_as_criterion_b(self) -> None:
        # La salida guardada se corrió con 80/40; el resumen debe decir que no evalúa (b).
        import contextlib
        import io
        import sys

        old_argv, out = sys.argv, io.StringIO()
        try:
            sys.argv = ["s5_beta_seed_v3.py", "--resumen", str(OUTPUT)]
            with contextlib.redirect_stdout(out):
                self.script.main()
        finally:
            sys.argv = old_argv
        self.assertIn("cadena: K=20, 80 iteraciones, burn-in 40", out.getvalue())
        self.assertIn("no es la cadena del preregistro", out.getvalue())


if __name__ == "__main__":
    unittest.main()
