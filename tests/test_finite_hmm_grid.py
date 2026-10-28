from __future__ import annotations

from dataclasses import replace
import unittest

from Comparacion.cli import build_parser
from Comparacion.config import build_default_learning_curve_config
from Comparacion.runner import build_finite_hmm_grid_audit


def _rows(selected: list[int]) -> list[dict[str, object]]:
    return [
        {
            "model": "finite_hmm",
            "data_seed": index,
            "model_seed": 1,
            "frac": 1.0,
            "selected_states": value,
        }
        for index, value in enumerate(selected)
    ]


class FiniteHmmGridConfigTests(unittest.TestCase):
    def test_grid_must_be_sorted_unique_and_at_least_two(self) -> None:
        config = build_default_learning_curve_config()
        for invalid in ((), (12, 12), (24, 12), (1, 12), (12.5, 24)):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    replace(config, finite_hmm_states=invalid)

    def test_grid_limits_must_cover_the_initial_grid(self) -> None:
        config = build_default_learning_curve_config()
        with self.assertRaises(ValueError):
            replace(config, finite_hmm_states=(12, 24, 48), finite_hmm_max_states=24)
        with self.assertRaises(ValueError):
            replace(config, vomm_candidate_orders=(1, 2, 4, 8), vomm_max_order=4)

    def test_cli_parses_the_grid_limits(self) -> None:
        args = build_parser().parse_args(["--finite-hmm-max-states", "192", "--vomm-max-order", "16"])

        self.assertEqual(args.finite_hmm_max_states, 192)
        self.assertEqual(args.vomm_max_order, 16)

    def test_cli_parses_the_grid(self) -> None:
        args = build_parser().parse_args(["--finite-hmm-states", "48,72,96"])

        self.assertEqual(args.finite_hmm_states, "48,72,96")


class FiniteHmmGridAuditTests(unittest.TestCase):
    def test_selection_at_the_maximum_is_not_a_plateau(self) -> None:
        audit = build_finite_hmm_grid_audit(_rows([96, 96, 72]), grid=(48, 72, 96))

        self.assertEqual(audit["grid_maximum"], 96)
        self.assertEqual(audit["n_selections"], 3)
        self.assertEqual(audit["n_at_grid_maximum"], 2)
        self.assertEqual(audit["verdict"], "grid_too_small")
        self.assertFalse(audit["informative"])
        self.assertFalse(audit["plateau_claimed"])

    def test_one_selection_at_the_maximum_already_makes_the_grid_uninformative(self) -> None:
        # Preregistro 2026-10-26, sección B: ninguna fracción puede tocar el máximo.
        audit = build_finite_hmm_grid_audit(_rows([72, 48, 96]), grid=(48, 72, 96))

        self.assertEqual(audit["n_at_grid_maximum"], 1)
        self.assertEqual(audit["verdict"], "grid_too_small")
        self.assertFalse(audit["informative"])

    def test_no_selection_at_the_maximum_is_informative(self) -> None:
        audit = build_finite_hmm_grid_audit(_rows([72, 48, 72]), grid=(48, 72, 96))

        self.assertEqual(audit["verdict"], "grid_informative")
        self.assertTrue(audit["informative"])

    def test_a_documented_resource_limit_is_recorded_but_not_called_a_plateau(self) -> None:
        audit = build_finite_hmm_grid_audit(
            _rows([96, 96]),
            grid=(48, 72, 96),
            resource_limit_reason="96 estados llenan 15 GiB de RAM",
        )

        self.assertEqual(audit["verdict"], "grid_limited_by_resources")
        self.assertEqual(audit["resource_limit_reason"], "96 estados llenan 15 GiB de RAM")
        self.assertFalse(audit["informative"])
        self.assertFalse(audit["plateau_claimed"])

    def test_no_selections_is_inconclusive(self) -> None:
        audit = build_finite_hmm_grid_audit([], grid=(48, 72, 96))

        self.assertEqual(audit["verdict"], "no_selections")
        self.assertFalse(audit["informative"])


if __name__ == "__main__":
    unittest.main()
