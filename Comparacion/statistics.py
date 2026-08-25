"""Paired work-level inference for multidimensional model comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
import math
import operator

import numpy as np
from scipy.stats import wilcoxon


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


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


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _full_fraction(row: Mapping[str, object]) -> bool:
    fraction = _finite_float(row.get("frac"))
    return fraction == 1.0


def _row_test_nll(row: Mapping[str, object]) -> float | None:
    if "test_nll" in row:
        return _finite_float(row["test_nll"])
    # Existing per-piece evaluator rows use this equivalent field name.
    return _finite_float(row.get("nll_per_token"))


def _bootstrap_mean_interval(
    differences: np.ndarray,
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> list[float]:
    n_pairs = differences.size
    bootstrap_means = np.empty(bootstrap_samples, dtype=float)
    for sample_index in range(bootstrap_samples):
        sampled_indices = generator.integers(0, n_pairs, size=n_pairs)
        bootstrap_means[sample_index] = float(np.mean(differences[sampled_indices]))
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return [float(lower), float(upper)]


def _empty_comparison(model_a: str, model_b: str) -> dict[str, object]:
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": "test_nll",
        "difference_orientation": "model_a_minus_model_b",
        "interpretation": "negative_favors_model_a",
        "status": "unavailable",
        "reason": "no_paired_finite_works",
        "n_pairs": 0,
        "canonical_work_ids": [],
        "mean_difference": None,
        "median_difference": None,
        "bootstrap_95_ci": None,
        "wilcoxon_status": "unavailable",
        "wilcoxon_reason": "no_paired_finite_works",
        "wilcoxon_statistic": None,
        "p_value": None,
        "p_value_holm": None,
    }


def _apply_holm_adjustment(comparisons: list[dict[str, object]]) -> int:
    valid_indices = [
        index
        for index, comparison in enumerate(comparisons)
        if comparison["wilcoxon_status"] == "ok"
        and comparison["p_value"] is not None
        and math.isfinite(float(comparison["p_value"]))
    ]
    valid_indices.sort(
        key=lambda index: (
            float(comparisons[index]["p_value"]),
            str(comparisons[index]["model_a"]),
            str(comparisons[index]["model_b"]),
        )
    )

    n_tests = len(valid_indices)
    running_adjusted = 0.0
    for rank, comparison_index in enumerate(valid_indices):
        raw_p_value = float(comparisons[comparison_index]["p_value"])
        candidate = min(1.0, (n_tests - rank) * raw_p_value)
        running_adjusted = min(1.0, max(running_adjusted, candidate))
        comparisons[comparison_index]["p_value_holm"] = running_adjusted
    return n_tests


def pairwise_model_comparisons(
    rows: Iterable[Mapping[str, object]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Compare every sorted model pair using canonical works at ``frac == 1.0``.

    Repeated data/model seeds are first averaged within each
    ``(model, canonical_work_id)``.  File-level ``piece_id`` is never used as
    a fallback. Pairwise differences are then computed as
    ``model_a test_nll - model_b test_nll``; negative values favor ``model_a``.
    Rows with missing or non-finite identifiers, fractions, or NLL values are
    excluded rather than imputed.  ``nll_per_token`` is accepted as the
    existing per-piece alias for ``test_nll``.
    """

    validated_bootstrap_samples = _positive_integer(
        bootstrap_samples,
        name="bootstrap_samples",
    )
    validated_seed = _nonnegative_integer(seed, name="seed")

    model_names: set[str] = set()
    repeated_values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        model = row.get("model")
        if not isinstance(model, str) or not model:
            continue
        model_names.add(model)
        if not _full_fraction(row):
            continue
        canonical_work_id = row.get("canonical_work_id")
        if not isinstance(canonical_work_id, str) or not canonical_work_id.strip():
            continue
        test_nll = _row_test_nll(row)
        if test_nll is None:
            continue
        repeated_values.setdefault((model, canonical_work_id), []).append(test_nll)

    work_means = {
        key: float(np.mean(values))
        for key, values in repeated_values.items()
        if values
    }
    works_by_model: dict[str, set[str]] = {model: set() for model in model_names}
    for model, canonical_work_id in work_means:
        works_by_model[model].add(canonical_work_id)

    generator = np.random.default_rng(validated_seed)
    comparisons_payload: list[dict[str, object]] = []
    for model_a, model_b in combinations(sorted(model_names), 2):
        canonical_work_ids = sorted(works_by_model[model_a] & works_by_model[model_b])
        if not canonical_work_ids:
            comparisons_payload.append(_empty_comparison(model_a, model_b))
            continue

        differences = np.asarray(
            [
                work_means[(model_a, canonical_work_id)]
                - work_means[(model_b, canonical_work_id)]
                for canonical_work_id in canonical_work_ids
            ],
            dtype=float,
        )
        comparison: dict[str, object] = {
            "model_a": model_a,
            "model_b": model_b,
            "metric": "test_nll",
            "difference_orientation": "model_a_minus_model_b",
            "interpretation": "negative_favors_model_a",
            "status": "ok",
            "reason": None,
            "n_pairs": len(canonical_work_ids),
            "canonical_work_ids": canonical_work_ids,
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "bootstrap_95_ci": _bootstrap_mean_interval(
                differences,
                bootstrap_samples=validated_bootstrap_samples,
                generator=generator,
            ),
            "wilcoxon_status": "unavailable",
            "wilcoxon_reason": "fewer_than_two_nonzero_differences",
            "wilcoxon_statistic": None,
            "p_value": None,
            "p_value_holm": None,
        }
        if int(np.count_nonzero(differences)) >= 2:
            try:
                result = wilcoxon(differences)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                comparison["wilcoxon_reason"] = "wilcoxon_failed"
            else:
                if math.isfinite(statistic) and math.isfinite(p_value):
                    comparison["wilcoxon_status"] = "ok"
                    comparison["wilcoxon_reason"] = None
                    comparison["wilcoxon_statistic"] = statistic
                    comparison["p_value"] = p_value
                else:
                    comparison["wilcoxon_reason"] = "nonfinite_wilcoxon_result"
        comparisons_payload.append(comparison)

    n_valid_tests = _apply_holm_adjustment(comparisons_payload)
    # El denominador de cada comparación es una obra canónica, no un archivo:
    # dejarlo explícito evita citar 424 archivos donde hay 414 obras.
    denominators = {
        "unit": "canonical_work",
        "n_works_by_model": {
            model: len(works_by_model.get(model, set())) for model in sorted(model_names)
        },
        "n_works_any_model": len({work for works in works_by_model.values() for work in works}),
    }
    return {
        "denominators": denominators,
        "metric": "test_nll",
        "difference_orientation": "model_a_minus_model_b",
        "interpretation": "negative_favors_model_a",
        "bootstrap_samples": validated_bootstrap_samples,
        "seed": validated_seed,
        "models": sorted(model_names),
        "n_comparisons": len(comparisons_payload),
        "n_valid_tests": n_valid_tests,
        "comparisons": comparisons_payload,
    }
