"""Pure metrics for comparing structural boundaries and partitions.

Boundary precision and recall use the following empty-set convention: an
empty denominator receives a score of one.  Consequently two empty boundary
sets have F1 equal to one, while exactly one empty set has F1 equal to zero.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
import math
import operator
from typing import Any


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a nonnegative integer") from exc
    if integer < 0:
        raise ValueError(f"{name} must be nonnegative")
    return integer


def _validated_boundaries(values: Iterable[object], *, name: str) -> list[int]:
    try:
        boundaries = [
            _nonnegative_integer(value, name=f"{name} boundary")
            for value in values
        ]
    except TypeError as exc:
        if values is None:
            raise TypeError(f"{name} boundaries must be iterable") from exc
        raise
    boundaries.sort()
    if any(left == right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"{name} boundaries must not contain duplicate coordinates")
    return boundaries


def boundary_f1(
    reference: Iterable[object],
    predicted: Iterable[object],
    tolerance: int,
) -> dict[str, float | int]:
    """Return tolerant boundary precision, recall and F1.

    Boundaries and ``tolerance`` must be nonnegative integers.  Matches are a
    maximum-cardinality one-to-one matching under absolute distance at most
    ``tolerance``. Duplicate coordinates are rejected as malformed annotations.
    """

    validated_tolerance = _nonnegative_integer(tolerance, name="tolerance")
    reference_boundaries = _validated_boundaries(reference, name="reference")
    predicted_boundaries = _validated_boundaries(predicted, name="predicted")

    reference_index = 0
    predicted_index = 0
    n_matches = 0
    while (
        reference_index < len(reference_boundaries)
        and predicted_index < len(predicted_boundaries)
    ):
        reference_boundary = reference_boundaries[reference_index]
        predicted_boundary = predicted_boundaries[predicted_index]
        if predicted_boundary < reference_boundary - validated_tolerance:
            predicted_index += 1
        elif reference_boundary < predicted_boundary - validated_tolerance:
            reference_index += 1
        else:
            n_matches += 1
            reference_index += 1
            predicted_index += 1

    precision = (
        n_matches / len(predicted_boundaries)
        if predicted_boundaries
        else 1.0
    )
    recall = (
        n_matches / len(reference_boundaries)
        if reference_boundaries
        else 1.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_matches": n_matches,
        "n_reference": len(reference_boundaries),
        "n_predicted": len(predicted_boundaries),
        "tolerance": validated_tolerance,
    }


def _validated_labels(
    reference: Sequence[Hashable] | Iterable[Hashable],
    predicted: Sequence[Hashable] | Iterable[Hashable],
) -> tuple[tuple[Hashable, ...], tuple[Hashable, ...]]:
    reference_labels = tuple(reference)
    predicted_labels = tuple(predicted)
    if not reference_labels:
        raise ValueError("partition label sequences must be nonempty")
    if len(reference_labels) != len(predicted_labels):
        raise ValueError("partition label sequences must have equal lengths")
    for name, labels in (("reference", reference_labels), ("predicted", predicted_labels)):
        for label in labels:
            try:
                hash(label)
            except TypeError as exc:
                raise TypeError(f"{name} labels must be hashable") from exc
    return reference_labels, predicted_labels


def _contingency_counts(
    reference: Sequence[Hashable] | Iterable[Hashable],
    predicted: Sequence[Hashable] | Iterable[Hashable],
) -> tuple[int, dict[tuple[int, int], int], list[int], list[int], bool]:
    reference_labels, predicted_labels = _validated_labels(reference, predicted)
    reference_ids: dict[Any, int] = {}
    predicted_ids: dict[Any, int] = {}
    cell_counts: dict[tuple[int, int], int] = {}

    for reference_label, predicted_label in zip(reference_labels, predicted_labels):
        reference_id = reference_ids.setdefault(reference_label, len(reference_ids))
        predicted_id = predicted_ids.setdefault(predicted_label, len(predicted_ids))
        cell = (reference_id, predicted_id)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1

    row_counts = [0] * len(reference_ids)
    column_counts = [0] * len(predicted_ids)
    for (reference_id, predicted_id), count in cell_counts.items():
        row_counts[reference_id] += count
        column_counts[predicted_id] += count

    occupied_rows = {reference_id for reference_id, _ in cell_counts}
    occupied_columns = {predicted_id for _, predicted_id in cell_counts}
    equivalent_partitions = (
        len(cell_counts) == len(occupied_rows) == len(occupied_columns)
        and len(occupied_rows) == len(reference_ids)
        and len(occupied_columns) == len(predicted_ids)
    )
    return (
        len(reference_labels),
        cell_counts,
        row_counts,
        column_counts,
        equivalent_partitions,
    )


def normalized_mutual_information(
    reference: Sequence[Hashable] | Iterable[Hashable],
    predicted: Sequence[Hashable] | Iterable[Hashable],
) -> float:
    """Return NMI using the arithmetic mean of the two partition entropies."""

    n_items, cells, rows, columns, _ = _contingency_counts(reference, predicted)
    n_items_float = float(n_items)
    mutual_information = 0.0
    for (reference_id, predicted_id), count in cells.items():
        mutual_information += (count / n_items_float) * math.log(
            (count * n_items_float)
            / (rows[reference_id] * columns[predicted_id])
        )

    reference_entropy = -sum(
        (count / n_items_float) * math.log(count / n_items_float)
        for count in rows
    )
    predicted_entropy = -sum(
        (count / n_items_float) * math.log(count / n_items_float)
        for count in columns
    )
    denominator = 0.5 * (reference_entropy + predicted_entropy)
    if denominator == 0.0:
        return 1.0
    return min(1.0, max(0.0, mutual_information / denominator))


def _pairs(count: int) -> int:
    return count * (count - 1) // 2


def adjusted_rand_index(
    reference: Sequence[Hashable] | Iterable[Hashable],
    predicted: Sequence[Hashable] | Iterable[Hashable],
) -> float:
    """Return the chance-adjusted Rand index from a contingency table."""

    n_items, cells, rows, columns, equivalent_partitions = _contingency_counts(
        reference,
        predicted,
    )
    total_pairs = _pairs(n_items)
    if total_pairs == 0:
        return 1.0

    cell_pairs = sum(_pairs(count) for count in cells.values())
    row_pairs = sum(_pairs(count) for count in rows)
    column_pairs = sum(_pairs(count) for count in columns)
    expected_index = row_pairs * column_pairs / total_pairs
    maximum_index = 0.5 * (row_pairs + column_pairs)
    denominator = maximum_index - expected_index
    if math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return 1.0 if equivalent_partitions else 0.0
    score = (cell_pairs - expected_index) / denominator
    return min(1.0, max(-1.0, score))
