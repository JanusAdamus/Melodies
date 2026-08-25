from __future__ import annotations

import unittest

from Comparacion.canonicalization_audit import (
    build_canonicalization_audit,
    melodic_fingerprint,
)
from next_token_experiment.experiment.splits import (
    canonicalization_details,
    canonicalize_work_label,
)


def _piece(
    piece_id: str,
    title: str,
    composer: str,
    *,
    tokens: list[int] | None = None,
) -> dict[str, object]:
    return {
        "piece_id": piece_id,
        "title": title,
        "composer": composer,
        "canonical_work_id": canonicalize_work_label(f"{composer} {title}"),
        "tokens": tokens if tokens is not None else [0, 2, 4, 5, 7],
    }


class CanonicalizationDetailsTests(unittest.TestCase):
    def test_label_normalization_is_unchanged(self) -> None:
        # Comportamiento vigente de la corrida original: Path().stem borra el
        # último sufijo con punto, así que "No.2" pierde el número.
        self.assertEqual(
            canonicalize_work_label("Chopin Nocturne_Op.9 No.2"), "chopin nocturne op 9 no"
        )

    def test_details_flag_the_stripped_trailing_suffix(self) -> None:
        details = canonicalization_details("Beethoven Sonata No.1")

        self.assertTrue(details["dropped_suffix"])
        self.assertEqual(details["dropped_suffix_text"], ".1")

    def test_details_do_not_flag_labels_without_a_trailing_suffix(self) -> None:
        details = canonicalization_details("Beethoven Sonata No 1")

        self.assertFalse(details["dropped_suffix"])

    def test_details_flag_generic_and_short_labels(self) -> None:
        details = canonicalization_details("untitled")

        self.assertEqual(details["canonical_work_id"], "untitled")
        self.assertTrue(details["is_generic"])
        self.assertTrue(details["is_short"])

    def test_details_flag_empty_labels(self) -> None:
        details = canonicalization_details("   ")

        self.assertEqual(details["canonical_work_id"], "")
        self.assertTrue(details["is_empty"])

    def test_details_expose_an_aggressive_key_for_near_misses(self) -> None:
        first = canonicalization_details("Chopin Nocturne, Op 9 No 2")
        second = canonicalization_details("Chopin  nocturne op9  no2!")

        self.assertNotEqual(first["canonical_work_id"], second["canonical_work_id"])
        self.assertEqual(first["aggressive_key"], second["aggressive_key"])

    def test_details_fold_diacritics_for_transliterations(self) -> None:
        first = canonicalization_details("Dvořák Humoresque")
        second = canonicalization_details("Dvorak Humoresque")

        self.assertEqual(first["aggressive_key"], second["aggressive_key"])


class MelodicFingerprintTests(unittest.TestCase):
    def test_transposition_shares_a_fingerprint(self) -> None:
        self.assertEqual(
            melodic_fingerprint([60, 62, 64, 65]),
            melodic_fingerprint([65, 67, 69, 70]),
        )

    def test_different_openings_differ(self) -> None:
        self.assertNotEqual(melodic_fingerprint([60, 62, 64]), melodic_fingerprint([60, 61, 63]))

    def test_missing_tokens_have_no_fingerprint(self) -> None:
        self.assertIsNone(melodic_fingerprint([]))


class CanonicalizationAuditTests(unittest.TestCase):
    def test_multi_file_groups_are_listed(self) -> None:
        pieces = [
            _piece("a", "Nocturne Op 9 No 2", "Chopin"),
            _piece("b", "Nocturne Op 9 No 2 arrangement", "Chopin"),
            _piece("c", "Humoresque", "Dvorak"),
        ]
        audit = build_canonicalization_audit(pieces)

        self.assertEqual(audit["n_files"], 3)
        self.assertEqual(audit["n_canonical_works"], 2)
        groups = {group["canonical_work_id"]: group for group in audit["multi_file_groups"]}
        self.assertEqual(len(groups), 1)
        group = next(iter(groups.values()))
        self.assertEqual(sorted(group["piece_ids"]), ["a", "b"])

    def test_empty_metadata_is_reported(self) -> None:
        pieces = [_piece("a", "", ""), _piece("b", "Humoresque", "Dvorak")]
        audit = build_canonicalization_audit(pieces)

        self.assertEqual([item["piece_id"] for item in audit["empty_metadata"]], ["a"])

    def test_generic_titles_are_reported(self) -> None:
        pieces = [_piece("a", "Untitled", ""), _piece("b", "Untitled", "")]
        audit = build_canonicalization_audit(pieces)

        generic = {item["canonical_work_id"] for item in audit["generic_labels"]}
        self.assertEqual(generic, {"untitled"})
        self.assertIn("untitled", audit["review_required"])

    def test_close_titles_kept_apart_are_reported_as_near_misses(self) -> None:
        pieces = [
            _piece("a", "Nocturne, Op 9 No 2", "Chopin"),
            _piece("b", "Nocturne Op 9  no2!", "Chopin"),
        ]
        audit = build_canonicalization_audit(pieces)

        self.assertEqual(audit["n_canonical_works"], 2)
        near = audit["near_miss_groups"]
        self.assertEqual(len(near), 1)
        self.assertEqual(len(near[0]["canonical_work_ids"]), 2)

    def test_movement_numbers_written_without_a_dot_stay_apart(self) -> None:
        pieces = [
            _piece("a", "Sonata No 1", "Beethoven"),
            _piece("b", "Sonata No 2", "Beethoven"),
        ]
        audit = build_canonicalization_audit(pieces)

        self.assertEqual(audit["n_canonical_works"], 2)
        self.assertEqual(audit["near_miss_groups"], [])

    def test_movement_numbers_lost_to_the_stem_are_reported(self) -> None:
        pieces = [
            _piece("a", "Sonata No.1", "Beethoven", tokens=[60, 62, 64]),
            _piece("b", "Sonata No.2", "Beethoven", tokens=[71, 60, 55]),
        ]
        audit = build_canonicalization_audit(pieces)

        # Las dos sonatas caen en el mismo identificador: es exactamente el
        # riesgo que la revisión humana debe resolver, no un acierto.
        self.assertEqual(audit["n_canonical_works"], 1)
        reasons = {item["reason"] for item in audit["suspicious_collisions"]}
        self.assertIn("dropped_suffix", reasons)
        self.assertEqual([item["piece_id"] for item in audit["dropped_suffixes"]], ["a", "b"])

    def test_group_with_disagreeing_fingerprints_is_suspicious(self) -> None:
        pieces = [
            _piece("a", "Untitled", "Anon", tokens=[60, 62, 64, 65, 67]),
            _piece("b", "Untitled", "Anon", tokens=[60, 55, 71, 48, 50]),
        ]
        audit = build_canonicalization_audit(pieces)

        collisions = audit["suspicious_collisions"]
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["reason"], "fingerprints_disagree")
        self.assertIn(collisions[0]["canonical_work_id"], audit["review_required"])

    def test_fingerprint_agreement_is_diagnostic_not_proof(self) -> None:
        pieces = [
            _piece("a", "Humoresque", "Dvorak", tokens=[60, 62, 64]),
            _piece("b", "Humoresque", "Dvorak", tokens=[60, 62, 64]),
        ]
        audit = build_canonicalization_audit(pieces)

        group = audit["multi_file_groups"][0]
        self.assertTrue(group["fingerprints_agree"])
        self.assertEqual(audit["suspicious_collisions"], [])
        self.assertEqual(
            audit["fingerprint_policy"],
            "diagnostic_only_never_automatic_identity_proof",
        )


if __name__ == "__main__":
    unittest.main()
