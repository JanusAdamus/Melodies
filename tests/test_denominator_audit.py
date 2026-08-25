from __future__ import annotations

import math
import unittest

from Comparacion.denominator_audit import build_denominator_audit


def _row(
    piece_id: str,
    canonical_work_id: str,
    model: str,
    nll: float,
    *,
    frac: float = 1.0,
) -> dict[str, object]:
    return {
        "piece_id": piece_id,
        "canonical_work_id": canonical_work_id,
        "model": model,
        "frac": frac,
        "data_seed": 1,
        "model_seed": 1,
        "nll_per_token": nll,
    }


def _rows() -> list[dict[str, object]]:
    """Cuatro archivos, tres obras canónicas (dos archivos comparten obra)."""

    rows: list[dict[str, object]] = []
    for model in ("vomm", "finite_hmm"):
        rows.append(_row("file-a", "work-1", model, 2.0))
        rows.append(_row("file-b", "work-1", model, 2.1))
        rows.append(_row("file-c", "work-2", model, 2.2))
        rows.append(_row("file-d", "work-3", model, 2.3))
    return rows


class DenominatorAuditTests(unittest.TestCase):
    def test_distinguishes_files_works_and_grouped_variants(self) -> None:
        audit = build_denominator_audit(_rows())

        self.assertEqual(audit["n_scored_files"], 4)
        self.assertEqual(audit["n_canonical_works"], 3)
        self.assertEqual(audit["n_works_with_multiple_files"], 1)
        self.assertEqual(audit["n_files_absorbed_by_grouping"], 1)
        self.assertEqual(audit["explanation"], "canonicalization_only")

    def test_non_finite_nll_is_discarded_with_a_reason(self) -> None:
        rows = _rows()
        rows.append(_row("file-e", "work-4", "vomm", math.nan))
        audit = build_denominator_audit(rows)

        self.assertEqual(audit["n_scored_files"], 4)
        discards = audit["discards"]
        self.assertEqual(len(discards), 1)
        self.assertEqual(discards[0]["piece_id"], "file-e")
        self.assertEqual(discards[0]["reason"], "non_finite_nll")
        self.assertEqual(discards[0]["model"], "vomm")
        self.assertEqual(audit["explanation"], "canonicalization_and_discards")

    def test_missing_canonical_work_id_is_discarded_with_a_reason(self) -> None:
        rows = _rows()
        rows.append(_row("file-f", "   ", "vomm", 2.5))
        audit = build_denominator_audit(rows)

        reasons = {discard["reason"] for discard in audit["discards"]}
        self.assertEqual(reasons, {"missing_canonical_work_id"})

    def test_common_works_per_model_pair(self) -> None:
        rows = _rows()
        rows.append(_row("file-g", "work-9", "vomm", 2.4))
        audit = build_denominator_audit(rows)

        per_model = audit["per_model"]
        self.assertEqual(per_model["vomm"]["n_canonical_works"], 4)
        self.assertEqual(per_model["finite_hmm"]["n_canonical_works"], 3)
        pairs = {(pair["model_a"], pair["model_b"]): pair for pair in audit["pairs"]}
        pair = pairs[("finite_hmm", "vomm")]
        self.assertEqual(pair["n_common_works"], 3)
        self.assertEqual(pair["only_model_b"], ["work-9"])
        self.assertEqual(pair["only_model_a"], [])

    def test_only_full_fraction_rows_count_towards_denominators(self) -> None:
        rows = _rows()
        rows.append(_row("file-h", "work-5", "vomm", 2.6, frac=0.5))
        audit = build_denominator_audit(rows)

        self.assertEqual(audit["n_scored_files"], 4)
        self.assertEqual(audit["n_canonical_works"], 3)
        self.assertEqual(audit["discards"], [])

    def test_split_pieces_without_rows_are_reported(self) -> None:
        audit = build_denominator_audit(
            _rows(), test_piece_ids=["file-a", "file-b", "file-c", "file-d", "file-z"]
        )

        self.assertEqual(audit["n_test_split_files"], 5)
        self.assertEqual(audit["unscored_test_files"], ["file-z"])
        self.assertEqual(audit["explanation"], "canonicalization_and_discards")

    def test_no_grouping_and_no_discards_is_consistent(self) -> None:
        rows = [
            _row("file-a", "work-1", "vomm", 2.0),
            _row("file-b", "work-2", "vomm", 2.1),
        ]
        audit = build_denominator_audit(rows)

        self.assertEqual(audit["n_scored_files"], 2)
        self.assertEqual(audit["n_canonical_works"], 2)
        self.assertEqual(audit["explanation"], "files_equal_works")


if __name__ == "__main__":
    unittest.main()
