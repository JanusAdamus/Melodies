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


def _axis_interval(row: Mapping[str, object], axis: str, value: float) -> tuple[float, float]:
    """``row[axis + "_interval"]`` when given (CI or tolerance band), else the point."""

    interval = row.get(f"{axis}_interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        try:
            lower, upper = float(interval[0]), float(interval[1])
        except (TypeError, ValueError):
            return value, value
        if math.isfinite(lower) and math.isfinite(upper) and lower <= upper:
            return lower, upper
    return value, value


PairedIntervals = Mapping[str, Mapping[tuple[str, str], tuple[float, float]]]


def _paired_difference_interval(
    candidate: Mapping[str, object],
    target: Mapping[str, object],
    axis: str,
    paired_intervals: PairedIntervals | None,
) -> tuple[float, float] | None:
    """CI of ``candidate − target`` on ``axis`` from the paired comparison, if any."""

    by_pair = (paired_intervals or {}).get(axis)
    if not by_pair:
        return None
    candidate_name, target_name = candidate.get("model"), target.get("model")
    interval = by_pair.get((candidate_name, target_name))
    if interval is None:
        reverse = by_pair.get((target_name, candidate_name))
        if reverse is None:
            return None
        interval = (-float(reverse[1]), -float(reverse[0]))
    try:
        lower, upper = float(interval[0]), float(interval[1])
    except (TypeError, ValueError, IndexError):
        return None
    if math.isfinite(lower) and math.isfinite(upper) and lower <= upper:
        return lower, upper
    return None


def _dominates(
    candidate: Mapping[str, object],
    target: Mapping[str, object],
    *,
    minimize: tuple[str, ...],
    maximize: tuple[str, ...],
    paired_intervals: PairedIntervals | None = None,
) -> bool:
    """Interval dominance (C5).

    When ``paired_intervals`` gives the CI of ``candidate − target`` on an axis
    (paired over the same works, the preregistered decision rule), that CI
    decides the axis: worse when it lies entirely on the worse side of 0,
    strictly better when it lies entirely on the better side, a tie when it
    crosses 0.  Otherwise, no worse: the marginal intervals overlap or the
    candidate is better; strictly better: the candidate's interval lies
    entirely on the better side.  Without intervals this is the usual point
    definition.
    """

    axes = (*minimize, *maximize)
    candidate_values = _finite_axis_values(candidate, axes)
    target_values = _finite_axis_values(target, axes)
    if candidate_values is None or target_values is None:
        return False

    no_worse = True
    strictly_better = False
    for axis in axes:
        difference = _paired_difference_interval(candidate, target, axis, paired_intervals)
        if difference is not None:
            low, high = difference
            if axis in minimize:
                no_worse = no_worse and low <= 0.0
                strictly_better = strictly_better or high < 0.0
            else:
                no_worse = no_worse and high >= 0.0
                strictly_better = strictly_better or low > 0.0
            continue
        candidate_low, candidate_high = _axis_interval(candidate, axis, candidate_values[axis])
        target_low, target_high = _axis_interval(target, axis, target_values[axis])
        if axis in minimize:
            no_worse = no_worse and candidate_low <= target_high
            strictly_better = strictly_better or candidate_high < target_low
        else:
            no_worse = no_worse and candidate_high >= target_low
            strictly_better = strictly_better or candidate_low > target_high
    return no_worse and strictly_better


def _strict_json_axis_copy(
    row: Mapping[str, object],
    axes: tuple[str, ...],
) -> dict[str, object]:
    result = dict(row)
    for axis in axes:
        if axis not in result or isinstance(result[axis], bool):
            continue
        try:
            value = float(result[axis])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(value):
            result[axis] = None
    return result


def pareto_front(
    rows: Iterable[Mapping[str, object]],
    minimize: tuple[str, ...],
    maximize: tuple[str, ...],
    *,
    paired_intervals: PairedIntervals | None = None,
) -> list[dict[str, object]]:
    """Return nondominated row copies in input order.

    Every declared axis is required for each pairwise dominance statement.  If
    either row is missing an axis or has a non-finite value, that pair is
    incomparable.  Dominance requires a candidate to be no worse on all axes
    and strictly better on at least one. Non-finite declared-axis values are
    replaced with ``None`` in returned copies so JSON can reject NaN safely.
    ``paired_intervals`` maps an axis to ``{(model_a, model_b): CI of a − b}``;
    for those pairs the paired CI replaces the marginal intervals on that axis.
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
                paired_intervals=paired_intervals,
            )
            for candidate_index, candidate in enumerate(row_copies)
        )
        if not dominated:
            nondominated.append(
                _strict_json_axis_copy(target, (*minimized_axes, *maximized_axes))
            )
    return nondominated
