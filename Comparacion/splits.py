from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

from next_token_experiment.schemas import PreparedPiece


@dataclass(frozen=True)
class FixedSplits:
    test_pieces: list[PreparedPiece]
    validation_pieces: list[PreparedPiece]
    train_pool_pieces: list[PreparedPiece]


def _infer_source_label(piece: PreparedPiece) -> str:
    source_path = Path(piece.source_path)
    path_parts = [part.lower() for part in source_path.parts]
    for label in ("musetrainer", "symbtr", "asap", "pdmx"):
        if label in path_parts:
            return label
    if "external" in path_parts:
        index = path_parts.index("external")
        if index + 1 < len(path_parts):
            return path_parts[index + 1]
    return "unknown"


def _length_bucket(piece: PreparedPiece, short_cut: int, medium_cut: int) -> str:
    length = len(piece.tokens)
    if length <= short_cut:
        return "short"
    if length <= medium_cut:
        return "medium"
    return "long"


def _allocate_group_counts(n_total: int, target_fraction: float) -> int:
    if n_total <= 0:
        return 0
    raw = n_total * target_fraction
    return max(0, min(n_total, int(round(raw))))


def _sorted_shuffle(pieces: list[PreparedPiece], seed: int) -> list[PreparedPiece]:
    shuffled = sorted(pieces, key=lambda piece: piece.piece_id)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_fixed_splits(
    pieces: list[PreparedPiece],
    *,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> FixedSplits:
    if not pieces:
        raise ValueError("At least one prepared piece is required.")
    if test_fraction <= 0.0 or validation_fraction <= 0.0 or test_fraction + validation_fraction >= 1.0:
        raise ValueError("Invalid split fractions.")

    ordered = sorted(pieces, key=lambda piece: piece.piece_id)
    lengths = sorted(len(piece.tokens) for piece in ordered)
    short_cut = lengths[min(len(lengths) - 1, max(0, math.floor(len(lengths) / 3) - 1))]
    medium_cut = lengths[min(len(lengths) - 1, max(0, math.floor(2 * len(lengths) / 3) - 1))]

    strata: dict[tuple[str, str], list[PreparedPiece]] = {}
    for piece in ordered:
        key = (_infer_source_label(piece), _length_bucket(piece, short_cut, medium_cut))
        strata.setdefault(key, []).append(piece)

    test_pieces: list[PreparedPiece] = []
    validation_pieces: list[PreparedPiece] = []
    train_pool_pieces: list[PreparedPiece] = []

    for index, key in enumerate(sorted(strata)):
        stratum = _sorted_shuffle(strata[key], seed + index)
        n_total = len(stratum)
        n_test = _allocate_group_counts(n_total, test_fraction)
        remaining_after_test = n_total - n_test
        adjusted_validation_fraction = validation_fraction / (1.0 - test_fraction)
        n_validation = min(
            remaining_after_test,
            _allocate_group_counts(remaining_after_test, adjusted_validation_fraction),
        )
        if remaining_after_test >= 3 and n_validation == 0:
            n_validation = 1
        if n_total >= 4 and n_test == 0:
            n_test = 1
            if n_validation >= remaining_after_test:
                n_validation = max(0, remaining_after_test - 1)

        test_pieces.extend(stratum[:n_test])
        validation_pieces.extend(stratum[n_test : n_test + n_validation])
        train_pool_pieces.extend(stratum[n_test + n_validation :])

    if not test_pieces or not validation_pieces or not train_pool_pieces:
        raise ValueError("Stratified split produced an empty partition.")

    return FixedSplits(
        test_pieces=sorted(test_pieces, key=lambda piece: piece.piece_id),
        validation_pieces=sorted(validation_pieces, key=lambda piece: piece.piece_id),
        train_pool_pieces=sorted(train_pool_pieces, key=lambda piece: piece.piece_id),
    )


def build_nested_training_subsets(
    train_pool_pieces: list[PreparedPiece],
    *,
    fractions: tuple[float, ...],
    data_seed: int,
) -> list[tuple[float, list[PreparedPiece]]]:
    if not train_pool_pieces:
        raise ValueError("Training pool cannot be empty.")

    ordered_pool = _sorted_shuffle(train_pool_pieces, data_seed)
    nested: list[tuple[float, list[PreparedPiece]]] = []
    for fraction in fractions:
        n_items = max(1, min(len(ordered_pool), int(round(fraction * len(ordered_pool)))))
        subset = sorted(ordered_pool[:n_items], key=lambda piece: piece.piece_id)
        nested.append((fraction, subset))
    return nested
