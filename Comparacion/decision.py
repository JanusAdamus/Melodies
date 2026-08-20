"""Decision helpers for multidimensional model trade-offs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


def _finite_axis_values(
    row: Mapping[str, object],
    axes: tuple[str, ...],
) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for axis in axes:
        if axis not in row or isinstance(row[axis], bool):
            return None
        try:
            value = float(row[axis])
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        values[axis] = value
    return values


def _dominates(
    candidate: Mapping[str, object],
    target: Mapping[str, object],
    *,
    minimize: tuple[str, ...],
    maximize: tuple[str, ...],
) -> bool:
    axes = (*minimize, *maximize)
    candidate_values = _finite_axis_values(candidate, axes)
    target_values = _finite_axis_values(target, axes)
    if candidate_values is None or target_values is None:
        return False

    no_worse = all(
        candidate_values[axis] <= target_values[axis]
        for axis in minimize
    ) and all(
        candidate_values[axis] >= target_values[axis]
        for axis in maximize
    )
    strictly_better = any(
        candidate_values[axis] < target_values[axis]
        for axis in minimize
    ) or any(
        candidate_values[axis] > target_values[axis]
        for axis in maximize
    )
    return no_worse and strictly_better


def pareto_front(
    rows: Iterable[Mapping[str, object]],
    minimize: tuple[str, ...],
    maximize: tuple[str, ...],
) -> list[dict[str, object]]:
    """Return nondominated row copies in input order.

    Every declared axis is required for each pairwise dominance statement.  If
    either row is missing an axis or has a non-finite value, that pair is
    incomparable.  Dominance requires a candidate to be no worse on all axes
    and strictly better on at least one.
    """

    minimized_axes = tuple(minimize)
    maximized_axes = tuple(maximize)
    conflicting_axes = set(minimized_axes) & set(maximized_axes)
    if conflicting_axes:
        names = ", ".join(sorted(conflicting_axes))
        raise ValueError(f"axes cannot be both minimized and maximized: {names}")

    row_copies = [dict(row) for row in rows]
    nondominated: list[dict[str, object]] = []
    for target_index, target in enumerate(row_copies):
        dominated = any(
            candidate_index != target_index
            and _dominates(
                candidate,
                target,
                minimize=minimized_axes,
                maximize=maximized_axes,
            )
            for candidate_index, candidate in enumerate(row_copies)
        )
        if not dominated:
            nondominated.append(target)
    return nondominated
