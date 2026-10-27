"""Regla de decisión del criterio (b) por bloques disjuntos de semillas (s8_criterio_b_bloques.py)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "auditoria_2026_10_26" / "hdp" / "s8_criterio_b_bloques.py"


def _script():
    spec = importlib.util.spec_from_file_location("s8_criterio_b_bloques", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CriterionBDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decide = _script().criterion_b_decision

    def test_all_blocks_equivalent_is_met_robustly(self) -> None:
        # Con n = 200, r = 0.02 da IC 90 % ≈ [-0.10, 0.14]: equivalencia en cada bloque.
        self.assertEqual(self.decide([(0.02, 200), (-0.03, 200), (0.05, 200)]), "met_robustly")

    def test_one_block_without_equivalence_is_inconclusive(self) -> None:
        # r = 0.15 con n = 50: IC 95 % de Fisher ≈ [-0.13, 0.41], ni equivalencia ni violación clara.
        self.assertEqual(self.decide([(0.02, 200), (0.15, 50), (0.0, 200)]), "inconclusive")

    def test_small_r_without_power_is_not_met_robustly(self) -> None:
        # |r| < 0.3 en todos los bloques no basta si la equivalencia no es significativa.
        self.assertEqual(self.decide([(0.1, 50), (0.1, 50), (0.1, 50)]), "inconclusive")

    def test_one_block_clearly_beyond_the_bound_is_not_met(self) -> None:
        # r = 0.7 con n = 50: IC 95 % ≈ [0.52, 0.82], entero por encima de 0.3.
        self.assertEqual(self.decide([(0.02, 200), (0.7, 50), (0.0, 200)]), "not_met")
        self.assertEqual(self.decide([(-0.7, 50), (0.02, 200), (0.0, 200)]), "not_met")

    def test_needs_at_least_three_blocks_of_fifty(self) -> None:
        with self.assertRaises(ValueError):
            self.decide([(0.0, 200), (0.0, 200)])
        with self.assertRaises(ValueError):
            self.decide([(0.0, 49), (0.0, 200), (0.0, 200)])


if __name__ == "__main__":
    unittest.main()
