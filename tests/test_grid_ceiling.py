"""C2 (auditoría 2026-10-26): rejillas sin techo para K del HMM y orden de la VOMM.

En agosto el HMM eligió K=48 (el máximo) en 30/30 celdas y la VOMM el orden
máximo. Mientras la selección toque el máximo, la rejilla se amplía; si llega al
límite declarado sin dejar de tocarlo, el ajuste falla con un mensaje claro.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Comparacion.classical_models import FiniteGlobalHMM
from Comparacion.grid import next_grid_candidates
from Comparacion.runner import build_finite_hmm_grid_audit, build_vomm_order_grid_audit
from Comparacion.vomm import VariableOrderMarkovModel, select_vomm_by_validation


def _scripted_candidate(optimum: int | None):
    """Validation NLL with a minimum at ``optimum``; ``None`` keeps improving with K."""

    def fit_candidate(self, train_pieces, validation_pieces, *, n_states, bos_token_id, max_context_length, rng):
        nll = 1.0 / n_states if optimum is None else 1.0 + (n_states - optimum) ** 2
        params = (np.ones(n_states) / n_states, np.eye(n_states), np.ones((n_states, 13)) / 13)
        return (*params, nll, [{"iteration": 1}], {"fit_wall_clock_s": 0.0, "validation_wall_clock_s": 0.0})

    return fit_candidate


def _piece(tokens: list[int]) -> SimpleNamespace:
    return SimpleNamespace(piece_id="v", title="", composer="", tokens=tokens)


def _scripted_vomm_evaluate(optimum: int | None):
    def evaluate(self, pieces, max_context_length):
        order = self.max_order
        nll = 1.0 / (order + 1) if optimum is None else 1.0 + (order - optimum) ** 2
        return {"summary": {"test_nll_per_token": nll, "evaluation_wall_clock_s": 0.0}, "piece_metrics": []}

    return evaluate


class NextGridCandidatesTests(unittest.TestCase):
    def test_steps_follow_the_preregistered_escalation(self) -> None:
        self.assertEqual(next_grid_candidates(48), (72, 96))
        self.assertEqual(next_grid_candidates(96), (144, 192))
        self.assertEqual(next_grid_candidates(1), (2,))
        self.assertEqual(next_grid_candidates(0), (1,))


class FiniteHmmGridExtensionTests(unittest.TestCase):
    def test_grid_grows_until_the_selection_leaves_the_maximum(self) -> None:
        model = FiniteGlobalHMM(candidate_num_states=(2, 4), max_iterations=1, tolerance=1e-4, seed=1, max_states=64)
        with patch.object(FiniteGlobalHMM, "_fit_candidate", _scripted_candidate(optimum=6)):
            result = model.fit([], [], bos_token_id=12)

        self.assertEqual(result.selected_states, 6)
        self.assertEqual(result.candidate_grid, (2, 4, 6, 8))
        self.assertEqual(result.grid_extensions, 1)
        self.assertLess(result.selected_states, max(result.candidate_grid))

    def test_reaching_the_limit_while_at_the_maximum_fails_clearly(self) -> None:
        model = FiniteGlobalHMM(candidate_num_states=(2, 4), max_iterations=1, tolerance=1e-4, seed=1, max_states=8)
        with patch.object(FiniteGlobalHMM, "_fit_candidate", _scripted_candidate(optimum=None)):
            with self.assertRaisesRegex(ValueError, "max_states=8"):
                model.fit([], [], bos_token_id=12)

    def test_without_a_limit_the_grid_is_fixed(self) -> None:
        model = FiniteGlobalHMM(candidate_num_states=(2, 4), max_iterations=1, tolerance=1e-4, seed=1)
        with patch.object(FiniteGlobalHMM, "_fit_candidate", _scripted_candidate(optimum=None)):
            result = model.fit([], [], bos_token_id=12)

        self.assertEqual(result.candidate_grid, (2, 4))
        self.assertEqual(result.grid_extensions, 0)


class VommOrderExtensionTests(unittest.TestCase):
    def test_order_grid_grows_until_the_selection_leaves_the_maximum(self) -> None:
        with patch.object(VariableOrderMarkovModel, "evaluate", _scripted_vomm_evaluate(optimum=6)):
            model = select_vomm_by_validation(
                [[0, 1]], [_piece([0, 1])], candidate_orders=(1, 2, 4), vocabulary_size=13,
                bos_token_id=12, max_order=32,
            )

        self.assertEqual(model.selected_order, 6)
        self.assertEqual(model.candidate_grid, (1, 2, 4, 6, 8))
        self.assertEqual(model.grid_extensions, 1)

    def test_reaching_the_order_limit_while_at_the_maximum_fails_clearly(self) -> None:
        with patch.object(VariableOrderMarkovModel, "evaluate", _scripted_vomm_evaluate(optimum=None)):
            with self.assertRaisesRegex(ValueError, "max_order=8"):
                select_vomm_by_validation(
                    [[0, 1]], [_piece([0, 1])], candidate_orders=(1, 2, 4), vocabulary_size=13,
                    bos_token_id=12, max_order=8,
                )


def _rows(model: str, field: str, selections: list[tuple[int, list[int]]]) -> list[dict[str, object]]:
    return [
        {"model": model, field: selected, "candidate_grid": json.dumps(grid)}
        for selected, grid in selections
    ]


class GridAuditTests(unittest.TestCase):
    def test_each_row_is_judged_against_its_own_extended_grid(self) -> None:
        rows = _rows("finite_hmm", "selected_states", [(72, [12, 24, 48, 72, 96]), (48, [12, 24, 48, 72, 96])])

        audit = build_finite_hmm_grid_audit(rows, grid=(12, 24, 48))

        self.assertEqual(audit["n_at_grid_maximum"], 0)
        self.assertEqual(audit["verdict"], "grid_informative")
        self.assertEqual(audit["status"], "passed")

    def test_a_single_selection_at_the_maximum_is_not_informative(self) -> None:
        rows = _rows("finite_hmm", "selected_states", [(48, [12, 24, 48]), (24, [12, 24, 48]), (12, [12, 24, 48])])

        audit = build_finite_hmm_grid_audit(rows, grid=(12, 24, 48))

        self.assertEqual(audit["verdict"], "grid_too_small")
        self.assertEqual(audit["status"], "failed")

    def test_vomm_order_audit_reads_the_selected_order(self) -> None:
        rows = _rows("vomm", "selected_order", [(8, [1, 2, 4, 8])]) + _rows(
            "finite_hmm", "selected_states", [(2, [2, 4])]
        )

        audit = build_vomm_order_grid_audit(rows, grid=(1, 2, 4, 8))

        self.assertEqual(audit["model"], "vomm")
        self.assertEqual(audit["n_selections"], 1)
        self.assertEqual(audit["verdict"], "grid_too_small")
        self.assertFalse(audit["plateau_claimed"])


if __name__ == "__main__":
    unittest.main()
