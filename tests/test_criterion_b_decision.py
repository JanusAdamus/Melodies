"""Decisión del criterio (b) con todas las semillas juntas (s8_criterio_b_bloques.py).

Criterio escrito (PROMPT_CLAUDE.md, tarea 1.1 (b)): |r| < 0.3.  Se decide con la estimación
puntual de las semillas agrupadas y la equivalencia TOST de |ρ| < 0.3 sobre esas mismas semillas.
Los bloques disjuntos sólo se reportan como descriptivo (estabilidad), no deciden.
"""

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
        self.script = _script()
        self.decide = self.script.criterion_b_decision

    def test_small_r_with_equivalence_is_met(self) -> None:
        # n = 200, r = 0.07: TOST p ≈ 5e-4 < 0.05.
        self.assertEqual(self.decide(0.07, 200), "met")
        self.assertEqual(self.decide(-0.10, 200), "met")

    def test_small_r_without_power_is_inconclusive(self) -> None:
        # n = 20, r = 0.1: |r| < 0.3 pero la equivalencia no es significativa.
        self.assertEqual(self.decide(0.1, 20), "inconclusive")

    def test_point_estimate_at_or_beyond_the_bound_is_not_met(self) -> None:
        self.assertEqual(self.decide(0.3, 200), "not_met")
        self.assertEqual(self.decide(-0.45, 200), "not_met")
        self.assertEqual(self.decide(0.959, 10), "not_met")

    def test_needs_at_least_ten_seeds(self) -> None:
        # PROMPT_CLAUDE.md pide al menos 10 semillas.
        with self.assertRaises(ValueError):
            self.decide(0.0, 9)

    def test_pooled_report_joins_blocks_and_only_describes_them(self) -> None:
        sections = [
            ([0, 1, 2, 3, 4, 5], [3, 4, 5, 6, 7, 8], [3.0, 3.1, 3.0, 3.2, 3.1, 3.0]),
            ([6, 7, 8, 9, 10, 11], [8, 7, 6, 5, 4, 3], [3.1, 3.0, 3.2, 3.0, 3.1, 3.0]),
        ]
        lines = self.script.pooled_report(sections)
        text = "\n".join(lines)
        self.assertIn("n semillas = 12", text)
        self.assertIn("semillas 0..11", text)
        self.assertIn("descriptivo", text)
        self.assertTrue(lines[-1].startswith("decisión del criterio (b): "))

    def test_pooled_report_rejects_repeated_seeds(self) -> None:
        sections = [([0, 1], [3, 4], [3.0, 3.1]), ([1, 2], [5, 6], [3.0, 3.2])]
        with self.assertRaises(ValueError):
            self.script.pooled_report(sections)


if __name__ == "__main__":
    unittest.main()
