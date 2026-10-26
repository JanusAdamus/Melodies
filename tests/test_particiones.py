"""Particiones y semillas de la comparación PDMX (tarea 1.5, auditoría C3, C4, C6, T3)."""

from __future__ import annotations

import random
import unittest

from Comparacion.splits import (
    assign_work_groups,
    build_grouped_fixed_splits,
    build_nested_training_subsets,
    split_manifest,
)
from next_token_experiment.experiment.splits import canonicalize_work_label
from next_token_experiment.schemas import PreparedPiece


def _melody(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(12) for _ in range(length)]


def _piece(piece_id: str, title: str, tokens: list[int], composer: str = "Unknown") -> PreparedPiece:
    return PreparedPiece(
        piece_id=piece_id,
        source_path=f"C:\\Datasets\\PDMX\\mxl\\{piece_id}",
        title=title,
        composer=composer,
        canonical_work_id=canonicalize_work_label(f"{composer} {title}"),
        representation="pitch_class",
        vocabulary=[str(index) for index in range(12)],
        tokens=tokens,
        n_events=len(tokens),
        metadata={},
    )


def _corpus(n_pieces: int = 40) -> list[PreparedPiece]:
    return [
        _piece(f"{index % 7}\\{index}\\Qm{index:04d}.mxl", f"Tune {index}", _melody(index, 60 + 5 * index))
        for index in range(n_pieces)
    ]


def _with_posix_ids(pieces: list[PreparedPiece]) -> list[PreparedPiece]:
    from dataclasses import replace

    return [
        replace(
            piece,
            piece_id=piece.piece_id.replace("\\", "/"),
            source_path=piece.source_path.replace("\\", "/").replace("C:", "/mnt/c"),
        )
        for piece in pieces
    ]


class CanonicalLabelTests(unittest.TestCase):
    def test_dotted_titles_keep_the_text_after_the_dot(self) -> None:
        self.assertEqual(
            canonicalize_work_label("anon. Scatter the Mud"), "anon scatter the mud"
        )
        self.assertNotEqual(
            canonicalize_work_label("anon. Farewell to Erin"),
            canonicalize_work_label("anon. Wedding dance"),
        )

    def test_movement_numbers_after_a_dot_survive(self) -> None:
        self.assertEqual(
            canonicalize_work_label("Chopin Nocturne_Op.9 No.2"), "chopin nocturne op 9 no 2"
        )

    def test_score_file_extensions_are_still_stripped(self) -> None:
        self.assertEqual(canonicalize_work_label("Canon_in_D_easy_piano.mxl"), "canon in d")


class ContentGroupingTests(unittest.TestCase):
    def test_versions_of_one_work_share_a_group_despite_different_titles(self) -> None:
        theme = _melody(1, 200)
        # Versión transpuesta y recortada: mismos intervalos, otra tonalidad.
        variant = [(token + 5) % 12 for token in theme[20:180]]
        pieces = [
            _piece("a.mxl", "Für Elise", theme, composer="Beethoven"),
            _piece("b.mxl", "Fur Elise (easy)", variant, composer="L. v. B."),
            _piece("c.mxl", "Something else", _melody(2, 200)),
        ]
        grouping = assign_work_groups(pieces)

        self.assertEqual(grouping.group_of["a.mxl"], grouping.group_of["b.mxl"])
        self.assertNotEqual(grouping.group_of["a.mxl"], grouping.group_of["c.mxl"])
        self.assertEqual(len(grouping.content_merges), 1)

    def test_generic_labels_do_not_merge_distinct_works(self) -> None:
        pieces = [
            _piece(f"{index}.mxl", "", _melody(10 + index, 120), composer="anon.")
            for index in range(4)
        ] + [
            _piece(f"t{index}.mxl", "Title", _melody(20 + index, 120), composer="Composer")
            for index in range(2)
        ]
        grouping = assign_work_groups(pieces)

        self.assertEqual(len(set(grouping.group_of.values())), len(pieces))

    def test_a_single_shared_repeated_note_run_does_not_merge(self) -> None:
        drone = [0] * 20
        pieces = [
            _piece("a.mxl", "Tune A", drone),
            _piece("b.mxl", "Tune B", drone + _melody(3, 300)),
        ]
        grouping = assign_work_groups(pieces)

        self.assertNotEqual(grouping.group_of["a.mxl"], grouping.group_of["b.mxl"])

    def test_grouped_splits_keep_content_groups_together(self) -> None:
        theme = _melody(99, 150)
        pieces = _corpus() + [
            _piece(f"v\\{index}.mxl", f"Variant title {index}", [(token + index) % 12 for token in theme])
            for index in range(4)
        ]
        splits = build_grouped_fixed_splits(pieces, test_fraction=0.15, validation_fraction=0.10, seed=7)
        homes = {
            name
            for name, part in (
                ("test", splits.test_pieces),
                ("validation", splits.validation_pieces),
                ("train", splits.train_pool_pieces),
            )
            for piece in part
            if piece.piece_id.startswith("v\\")
        }
        self.assertEqual(len(homes), 1)


class OsIndependentSplitTests(unittest.TestCase):
    def test_windows_and_posix_ids_give_the_same_partition(self) -> None:
        windows = _corpus()
        posix = _with_posix_ids(windows)

        def normalized(splits) -> list[set[str]]:
            return [
                {piece.piece_id.replace("\\", "/") for piece in part}
                for part in (splits.test_pieces, splits.validation_pieces, splits.train_pool_pieces)
            ]

        first = build_grouped_fixed_splits(windows, test_fraction=0.15, validation_fraction=0.10, seed=7)
        second = build_grouped_fixed_splits(posix, test_fraction=0.15, validation_fraction=0.10, seed=7)

        self.assertEqual(normalized(first), normalized(second))
        self.assertEqual(split_manifest(first)["sha256"], split_manifest(second)["sha256"])


class DataSeedTests(unittest.TestCase):
    def test_full_fraction_differs_by_data_seed_and_test_does_not(self) -> None:
        pieces = _corpus(60)
        splits = build_grouped_fixed_splits(pieces, test_fraction=0.15, validation_fraction=0.10, seed=7)
        full_sets = []
        for data_seed in (1, 2, 3):
            again = build_grouped_fixed_splits(pieces, test_fraction=0.15, validation_fraction=0.10, seed=7)
            self.assertEqual(
                [piece.piece_id for piece in again.test_pieces],
                [piece.piece_id for piece in splits.test_pieces],
            )
            nested = dict(
                build_nested_training_subsets(
                    splits.train_pool_pieces, fractions=(0.5, 1.0), data_seed=data_seed
                )
            )
            full = {piece.piece_id for piece in nested[1.0]}
            half = {piece.piece_id for piece in nested[0.5]}
            self.assertTrue(half <= full)
            self.assertEqual(len(full), round(0.9 * len(splits.train_pool_pieces)))
            full_sets.append(frozenset(full))

        self.assertEqual(len(set(full_sets)), 3)


class ManifestTests(unittest.TestCase):
    def test_manifest_reports_hash_and_changes_against_a_reference(self) -> None:
        pieces = _corpus()
        splits = build_grouped_fixed_splits(pieces, test_fraction=0.15, validation_fraction=0.10, seed=7)
        test_ids = [piece.piece_id for piece in splits.test_pieces]
        reference = {
            "test": test_ids[1:] + [splits.train_pool_pieces[0].piece_id.replace("\\", "/")],
            "validation": [piece.piece_id for piece in splits.validation_pieces],
        }

        manifest = split_manifest(splits, reference=reference)

        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertEqual(manifest["splits"]["test"]["n_pieces"], len(test_ids))
        change = manifest["comparison"]["test"]
        self.assertEqual(change["n_shared"], len(test_ids) - 1)
        self.assertEqual(change["n_only_new"], 1)
        self.assertEqual(change["n_only_reference"], 1)


if __name__ == "__main__":
    unittest.main()
