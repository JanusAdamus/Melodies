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
    return _bootstrap_mean_draws(
        differences, bootstrap_samples=bootstrap_samples, generator=generator
    )[0]


def _bootstrap_mean_draws(
    differences: np.ndarray,
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> tuple[list[float], np.ndarray]:
    """IC percentil del 95 % de la media por obra y las réplicas que lo producen."""

    n_pairs = differences.size
    bootstrap_means = np.empty(bootstrap_samples, dtype=float)
    for sample_index in range(bootstrap_samples):
        sampled_indices = generator.integers(0, n_pairs, size=n_pairs)
        bootstrap_means[sample_index] = float(np.mean(differences[sampled_indices]))
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return [float(lower), float(upper)], bootstrap_means


def _row_log_likelihood_and_tokens(row: Mapping[str, object]) -> tuple[float, int] | None:
    n_tokens = _finite_float(row.get("n_tokens"))
    if n_tokens is None or n_tokens <= 0:
        return None
    log_likelihood = _finite_float(row.get("log_likelihood"))
    if log_likelihood is None:
        nll = _row_test_nll(row)
        if nll is None:
            return None
        log_likelihood = -nll * n_tokens
    return log_likelihood, int(n_tokens)


def _nominal_run_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(key)) for key in ("split_seed", "data_seed", "model_seed"))


def _distinct_run_key(row: Mapping[str, object]) -> tuple[str, ...]:
    """Corrida real: conjunto de entrenamiento × model_seed (auditoría C4).

    Dos data_seed que producen el mismo conjunto de entrenamiento no son corridas
    distintas. Sin ``train_subset_sha256`` se usa data_seed como antes.
    """

    subset = row.get("train_subset_sha256")
    training = str(subset) if isinstance(subset, str) and subset else f"data_seed={row.get('data_seed')}"
    return (str(row.get("split_seed")), training, str(row.get("model_seed")))


def run_counts_by_model_and_fraction(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, float], dict[str, int]]:
    """Por (modelo, frac): corridas nominales (semillas) y distintas (datos × modelo)."""

    nominal: dict[tuple[str, float], set[tuple[str, ...]]] = {}
    distinct: dict[tuple[str, float], set[tuple[str, ...]]] = {}
    for row in rows:
        model = row.get("model")
        fraction = _finite_float(row.get("frac"))
        if not isinstance(model, str) or fraction is None:
            continue
        nominal.setdefault((model, fraction), set()).add(_nominal_run_key(row))
        distinct.setdefault((model, fraction), set()).add(_distinct_run_key(row))
    return {
        key: {"n_distinct_runs": len(distinct[key]), "n_nominal_runs": len(nominal[key])}
        for key in nominal
    }


def _work_totals(
    rows: Iterable[Mapping[str, object]],
) -> tuple[dict[tuple[str, float], dict[str, tuple[float, float]]], dict[tuple[str, float], int]]:
    """Per (model, frac): work -> (Σll, Σn) averaged over runs; and the run count.

    A run is one (split_seed, data_seed, model_seed). Repeated runs of the same
    work are averaged, never counted as extra works.
    """

    by_run: dict[tuple[str, float], dict[tuple[object, ...], dict[str, list[float]]]] = {}
    for row in rows:
        model = row.get("model")
        work = row.get("canonical_work_id")
        fraction = _finite_float(row.get("frac"))
        totals = _row_log_likelihood_and_tokens(row)
        if not isinstance(model, str) or not isinstance(work, str) or not work.strip():
            continue
        if fraction is None or totals is None:
            continue
        run = tuple(str(row.get(key)) for key in ("split_seed", "data_seed", "model_seed"))
        sums = by_run.setdefault((model, fraction), {}).setdefault(run, {}).setdefault(work, [0.0, 0.0])
        sums[0] += totals[0]
        sums[1] += totals[1]

    averaged: dict[tuple[str, float], dict[str, tuple[float, float]]] = {}
    run_counts: dict[tuple[str, float], int] = {}
    for key, runs in by_run.items():
        per_work: dict[str, list[list[float]]] = {}
        for works in runs.values():
            for work, sums in works.items():
                per_work.setdefault(work, []).append(sums)
        averaged[key] = {
            work: (float(np.mean([item[0] for item in values])), float(np.mean([item[1] for item in values])))
            for work, values in per_work.items()
        }
        run_counts[key] = len(runs)
    return averaged, run_counts


def _work_nll_means(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, float], dict[str, float]]:
    """Per (model, frac): work -> its per-note NLL, averaged over runs.

    Within one run the NLL of a work is Σll/Σn over its files (Aclaración 2: the
    per-note NLL of the work), never the mean of its files' NLL.  A work with a
    single file and no token count keeps that file's NLL; a multi-file work with
    a file lacking its token count has no per-note NLL in that run and is left
    out rather than imputed.  Runs (split_seed, data_seed, model_seed) are then
    averaged with equal weight.
    """

    by_run: dict[tuple[str, float], dict[str, dict[tuple[str, ...], list[Mapping[str, object]]]]] = {}
    for row in rows:
        model = row.get("model")
        work = row.get("canonical_work_id")
        fraction = _finite_float(row.get("frac"))
        if not isinstance(model, str) or not model or fraction is None:
            continue
        if not isinstance(work, str) or not work.strip() or _row_test_nll(row) is None:
            continue
        by_run.setdefault((model, fraction), {}).setdefault(work, {}).setdefault(
            _nominal_run_key(row), []
        ).append(row)

    result: dict[tuple[str, float], dict[str, float]] = {}
    for key, works in by_run.items():
        per_work: dict[str, float] = {}
        for work, runs in works.items():
            run_values: list[float] = []
            for files in runs.values():
                totals = [_row_log_likelihood_and_tokens(row) for row in files]
                if all(item is not None for item in totals):
                    run_values.append(
                        -sum(item[0] for item in totals) / sum(item[1] for item in totals)
                    )
                elif len(files) == 1:
                    run_values.append(float(_row_test_nll(files[0])))
            if run_values:
                per_work[work] = float(np.mean(run_values))
        if per_work:
            result[key] = per_work
    return result


def _bootstrap_indices(n_works: int, samples: int, generator: np.random.Generator) -> np.ndarray:
    return generator.integers(0, n_works, size=(samples, n_works))


def _percentile_interval(values: np.ndarray) -> list[float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return [float(lower), float(upper)]


def headline_estimates(
    rows: Iterable[Mapping[str, object]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    """Headline NLL figures, each with a bootstrap-by-work CI of its own estimand (C7).

    * ``token_weighted_nll``: Σll / Σn over test works; CI by resampling works and
      recomputing the ratio.
    * ``token_weighted_perplexity``: exp of the above, CI = exp of its bounds.
    * ``work_mean_nll``: mean over works of each work's Σll/Σn; CI of that mean.
    """

    samples = _positive_integer(bootstrap_samples, name="bootstrap_samples")
    generator = np.random.default_rng(_nonnegative_integer(seed, name="seed"))
    rows = list(rows)
    totals, run_counts = _work_totals(rows)
    per_note = _work_nll_means(rows)
    run_accounting = run_counts_by_model_and_fraction(rows)
    estimates: list[dict[str, object]] = []
    for (model, fraction), works in sorted(totals.items()):
        log_likelihoods = np.asarray([value[0] for value in works.values()], dtype=float)
        n_tokens = np.asarray([value[1] for value in works.values()], dtype=float)
        indices = _bootstrap_indices(len(works), samples, generator)
        token_draws = -log_likelihoods[indices].sum(axis=1) / n_tokens[indices].sum(axis=1)
        # La misma NLL por obra que usa la comparación pareada (Σll/Σn por corrida,
        # luego media de corridas): el marginal de Pareto y el pareado coinciden.
        work_values = per_note.get((model, fraction), {})
        n_work_mean = len(work_values)
        if set(work_values) == set(works):
            work_nll = np.asarray([work_values[work] for work in works], dtype=float)
            work_draws = work_nll[indices].mean(axis=1)
        else:
            work_nll = np.asarray(list(work_values.values()), dtype=float)
            work_draws = work_nll[_bootstrap_indices(n_work_mean, samples, generator)].mean(axis=1)
        token_point = float(-log_likelihoods.sum() / n_tokens.sum())
        token_interval = _percentile_interval(token_draws)
        common = {
            "model": model,
            "frac": fraction,
            "n_works": len(works),
            # n_runs: corridas distintas (conjunto de entrenamiento × model_seed);
            # n_nominal_runs: combinaciones de semillas, que pueden repetir datos.
            "n_runs": run_accounting[(model, fraction)]["n_distinct_runs"],
            "n_nominal_runs": run_counts[(model, fraction)],
            "n_tokens": float(n_tokens.sum()),
        }
        estimates.extend(
            [
                {
                    **common,
                    "estimand": "token_weighted_nll",
                    "role": "secondary",
                    "point_estimate": token_point,
                    "bootstrap_95_ci": token_interval,
                    "ci_method": "work_cluster_bootstrap_of_ratio_sum_ll_over_sum_n",
                },
                {
                    **common,
                    "estimand": "token_weighted_perplexity",
                    "role": "secondary",
                    "point_estimate": math.exp(token_point),
                    "bootstrap_95_ci": [math.exp(bound) for bound in token_interval],
                    "ci_method": "exp_of_token_weighted_nll_interval",
                },
                {
                    **common,
                    "n_works": n_work_mean,
                    "estimand": "work_mean_nll",
                    "role": "decides",
                    "point_estimate": float(work_nll.mean()),
                    "bootstrap_95_ci": _percentile_interval(work_draws),
                    "ci_method": "work_bootstrap_of_mean_work_nll",
                },
            ]
        )
    return {
        "unit": "canonical_work",
        # Aclaración 2: cada obra pesa lo mismo; lo ponderado por token es secundario.
        "decision_estimand": "work_mean_nll",
        "runs_averaged_within_work": True,
        "bootstrap_samples": samples,
        "seed": seed,
        "estimates": estimates,
    }


def _token_weighted_difference(
    totals_a: Mapping[str, tuple[float, float]],
    totals_b: Mapping[str, tuple[float, float]],
    works: list[str],
    *,
    bootstrap_samples: int,
    generator: np.random.Generator,
) -> tuple[float, list[float], np.ndarray] | None:
    if not works or any(work not in totals_a or work not in totals_b for work in works):
        return None
    ll_a = np.asarray([totals_a[work][0] for work in works], dtype=float)
    n_a = np.asarray([totals_a[work][1] for work in works], dtype=float)
    ll_b = np.asarray([totals_b[work][0] for work in works], dtype=float)
    n_b = np.asarray([totals_b[work][1] for work in works], dtype=float)
    point = float(-ll_a.sum() / n_a.sum() + ll_b.sum() / n_b.sum())
    indices = _bootstrap_indices(len(works), bootstrap_samples, generator)
    draws = -ll_a[indices].sum(axis=1) / n_a[indices].sum(axis=1) + ll_b[indices].sum(axis=1) / n_b[
        indices
    ].sum(axis=1)
    return point, _percentile_interval(draws), draws


#: Nivel familiar de la regla preregistrada (IC del 95 % «tras Holm»).
HOLM_FAMILY_ALPHA = 0.05


def _bootstrap_two_sided_p_value(draws: np.ndarray) -> float:
    """p bilateral del bootstrap: el menor nivel al que el IC percentil excluye 0."""

    below = float(np.mean(draws <= 0.0))
    above = float(np.mean(draws >= 0.0))
    return min(1.0, 2.0 * min(below, above))


def _empty_holm(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_bootstrap_p_value": None,
        f"{prefix}_holm_alpha": None,
        f"{prefix}_holm_ci": None,
        f"{prefix}_holm_decision_ci": None,
        f"{prefix}_holm_rejects_zero": None,
        f"{prefix}_holm_status": "not_evaluable",
    }


def _apply_bootstrap_holm(
    comparisons: list[dict[str, object]],
    draws_by_index: Mapping[int, np.ndarray],
    *,
    prefix: str,
    family_alpha: float = HOLM_FAMILY_ALPHA,
) -> None:
    """IC de Holm por pasos sobre réplicas bootstrap por obra (preregistro A y B).

    ``prefix`` es ``work_mean`` (la media por obra, que decide según la
    Aclaración 2) o ``token_weighted`` (secundaria).  Se ordenan las
    comparaciones por su p bilateral del bootstrap; la de rango r (desde 0) lleva
    el IC percentil al nivel 1 − α/(m − r).  Se rechaza mientras el IC excluya 0;
    tras el primer no rechazo, ninguna posterior se rechaza.

    ``*_holm_ci`` es ese IC tal cual: tras el primer no rechazo puede excluir 0
    sin que la prueba se rechace.  ``*_holm_decision_ci`` es el que decide: igual
    al anterior si Holm rechaza, ensanchado hasta incluir 0 si no.

    m es la familia completa (todas las comparaciones), no las que tienen datos:
    una prueba sin datos cuenta en la familia como si su p fuera 1 (va al final y
    no se rechaza) y queda «not_evaluable».  Así, faltar el n_tokens de una obra
    no encoge la familia ni afloja el α de las demás.
    """

    for index, comparison in enumerate(comparisons):
        if index not in draws_by_index:
            comparison.update(_empty_holm(prefix))
    ordered = sorted(
        draws_by_index,
        key=lambda index: (
            _bootstrap_two_sided_p_value(draws_by_index[index]),
            str(comparisons[index]["model_a"]),
            str(comparisons[index]["model_b"]),
        ),
    )
    n_tests = len(comparisons)
    still_rejecting = True
    for rank, index in enumerate(ordered):
        draws = draws_by_index[index]
        alpha = family_alpha / (n_tests - rank)
        low, high = np.percentile(draws, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        excludes_zero = bool(low > 0.0 or high < 0.0)
        still_rejecting = still_rejecting and excludes_zero
        interval = [float(low), float(high)]
        comparisons[index][f"{prefix}_bootstrap_p_value"] = _bootstrap_two_sided_p_value(draws)
        comparisons[index][f"{prefix}_holm_alpha"] = alpha
        comparisons[index][f"{prefix}_holm_ci"] = interval
        comparisons[index][f"{prefix}_holm_decision_ci"] = (
            interval if still_rejecting else [min(interval[0], 0.0), max(interval[1], 0.0)]
        )
        comparisons[index][f"{prefix}_holm_rejects_zero"] = still_rejecting
        comparisons[index][f"{prefix}_holm_status"] = "ok"


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
        "estimand": "mean_over_works_of_work_nll_difference",
        "token_weighted_estimand": "token_weighted_nll_difference",
        "token_weighted_difference": None,
        "token_weighted_bootstrap_95_ci": None,
        **_empty_holm("work_mean"),
        **_empty_holm("token_weighted"),
    }


#: Aclaración 2 al preregistro: qué estimando decide y qué campos leer.
DECISION_RULE: dict[str, object] = {
    "estimand": "mean_over_works_of_work_nll_difference",
    "point": "mean_difference",
    "interval": "work_mean_holm_decision_ci",
    "rejects_zero": "work_mean_holm_rejects_zero",
    "ci_method": "work_bootstrap_holm_step_down_over_the_preregistered_family",
    "secondary_estimand": "token_weighted_nll_difference",
    "secondary_interval": "token_weighted_holm_decision_ci",
    "source": "docs/preregistro-2026-10-26.md, Aclaración 2",
}

_CI_USAGE = (
    "*_holm_decision_ci decides: equal to *_holm_ci when Holm rejects, widened to include 0 "
    "when it does not. *_holm_ci is the percentile CI at alpha/(m - rank) and, after the first "
    "non-rejection, may exclude 0 although the test is not rejected; do not read it alone."
)


def _holm_family_record(comparisons: list[dict[str, object]]) -> dict[str, object]:
    """Tamaño y composición de la familia de Holm, y qué pruebas no se evaluaron."""

    tests = [[str(item["model_a"]), str(item["model_b"])] for item in comparisons]

    def _block(is_evaluable) -> dict[str, object]:
        missing = [pair for pair, item in zip(tests, comparisons) if not is_evaluable(item)]
        return {
            "family_size": len(tests),
            "n_evaluable": len(tests) - len(missing),
            "not_evaluable": missing,
        }

    return {
        "family_alpha": HOLM_FAMILY_ALPHA,
        "definition": "all_model_pairs_never_shrunk_not_evaluable_count_as_p_equal_1",
        "tests": tests,
        "ci_usage": _CI_USAGE,
        "wilcoxon": _block(lambda item: item["p_value_holm"] is not None),
        "work_mean_ci": _block(lambda item: item["work_mean_holm_status"] == "ok"),
        "token_weighted_ci": _block(lambda item: item["token_weighted_holm_status"] == "ok"),
    }


def _apply_holm_adjustment(comparisons: list[dict[str, object]]) -> int:
    """Holm sobre los p de Wilcoxon; devuelve cuántas pruebas fueron evaluables.

    El factor usa la familia completa (todas las comparaciones): una prueba no
    evaluable cuenta como p = 1 y no reduce el factor de las demás.
    """

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

    n_tests = len(comparisons)
    running_adjusted = 0.0
    for rank, comparison_index in enumerate(valid_indices):
        raw_p_value = float(comparisons[comparison_index]["p_value"])
        candidate = min(1.0, (n_tests - rank) * raw_p_value)
        running_adjusted = min(1.0, max(running_adjusted, candidate))
        comparisons[comparison_index]["p_value_holm"] = running_adjusted
    return len(valid_indices)


def _split_seed_of(row: Mapping[str, object]) -> int | None:
    value = row.get("split_seed")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_split_variation(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Separa la variación entre obras de la variación entre particiones.

    Una misma obra evaluada bajo varias semillas de partición no aporta
    observaciones independientes: la unidad sigue siendo la obra dentro de una
    partición.
    """

    per_model_split: dict[str, dict[int, dict[str, list[float]]]] = {}
    works: set[str] = set()
    work_split_pairs: set[tuple[str, int]] = set()
    split_seeds: set[int] = set()

    for row in rows:
        if not isinstance(row, Mapping) or not _full_fraction(row):
            continue
        model = row.get("model")
        canonical_work_id = row.get("canonical_work_id")
        split_seed = _split_seed_of(row)
        value = _row_test_nll(row)
        if not isinstance(model, str) or not isinstance(canonical_work_id, str):
            continue
        if not canonical_work_id.strip() or split_seed is None or value is None:
            continue
        split_seeds.add(split_seed)
        works.add(canonical_work_id)
        work_split_pairs.add((canonical_work_id, split_seed))
        by_split = per_model_split.setdefault(model, {})
        by_split.setdefault(split_seed, {}).setdefault(canonical_work_id, []).append(value)

    models_payload: dict[str, object] = {}
    for model, by_split in sorted(per_model_split.items()):
        # Las repeticiones de una misma obra dentro de una partición se promedian
        # antes de medir dispersión: no son observaciones nuevas.
        work_means = {
            split_seed: {work: float(np.mean(values)) for work, values in works_by_id.items()}
            for split_seed, works_by_id in by_split.items()
        }
        per_split_mean = {
            str(split_seed): float(np.mean(list(values.values())))
            for split_seed, values in sorted(work_means.items())
        }
        within_split_stds = [
            float(np.std(list(values.values()), ddof=0))
            for values in work_means.values()
            if len(values) > 1
        ]
        split_means = list(per_split_mean.values())
        models_payload[model] = {
            "per_split_mean_nll": per_split_mean,
            "between_split_std": float(np.std(split_means, ddof=0)) if len(split_means) > 1 else None,
            "between_work_std": float(np.mean(within_split_stds)) if within_split_stds else 0.0,
            "n_works_per_split": {
                str(split_seed): len(values) for split_seed, values in sorted(work_means.items())
            },
        }

    return {
        "status": "single_split_seed" if len(split_seeds) < 2 else "multiple_split_seeds",
        "split_seeds": sorted(split_seeds),
        "n_split_seeds": len(split_seeds),
        "n_distinct_works": len(works),
        "n_work_split_observations": len(work_split_pairs),
        "independence_unit": "canonical_work_within_split_seed",
        "models": models_payload,
    }


def pairwise_model_comparisons(
    rows: Iterable[Mapping[str, object]],
    bootstrap_samples: int,
    seed: int,
    expected_models: Iterable[str] | None = None,
) -> dict[str, object]:
    """Compare every sorted model pair using canonical works at ``frac == 1.0``.

    ``expected_models`` fija la familia de Holm del diseño: todos sus pares
    cuentan aunque un modelo no haya dejado filas (sus pares quedan no
    evaluables), y un modelo fuera de ese conjunto es un error.  Sin él, la
    familia son los modelos presentes en ``rows``; las particiones anidadas
    heredan siempre la familia de la llamada exterior.

    The NLL of a work in one run is its per-note NLL (Σll/Σn over its files);
    repeated data/model seeds are then averaged within each
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

    rows_list = [row for row in rows if isinstance(row, Mapping)]
    model_names: set[str] = set()
    for row in rows_list:
        model = row.get("model")
        if isinstance(model, str) and model:
            model_names.add(model)

    if expected_models is not None:
        expected = {str(model) for model in expected_models}
        unexpected = sorted(model_names - expected)
        if unexpected:
            raise ValueError(
                "modelos fuera de la familia de Holm preregistrada: " + ", ".join(unexpected)
            )
        model_names = expected
    # Aclaración 2: la NLL de cada obra es su NLL por nota (Σll/Σn de sus
    # archivos en cada corrida, luego media de corridas), la misma que usa el
    # marginal work_mean_nll de headline_estimates y la frontera de Pareto.
    work_means = {
        (model, canonical_work_id): value
        for (model, _fraction), works in _work_nll_means(
            row for row in rows_list if _full_fraction(row)
        ).items()
        for canonical_work_id, value in works.items()
    }
    works_by_model: dict[str, set[str]] = {model: set() for model in model_names}
    for model, canonical_work_id in work_means:
        works_by_model[model].add(canonical_work_id)

    full_fraction_totals = {
        model: works
        for (model, fraction), works in _work_totals(
            row for row in rows_list if _full_fraction(row)
        )[0].items()
    }
    generator = np.random.default_rng(validated_seed)
    comparisons_payload: list[dict[str, object]] = []
    token_weighted_draws: dict[int, np.ndarray] = {}
    work_mean_draws: dict[int, np.ndarray] = {}
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
        work_mean_interval, work_mean_draws[len(comparisons_payload)] = _bootstrap_mean_draws(
            differences,
            bootstrap_samples=validated_bootstrap_samples,
            generator=generator,
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
            # Aclaración 2: esta media por obra es el estimando que decide.
            "bootstrap_95_ci": work_mean_interval,
            "wilcoxon_status": "unavailable",
            "wilcoxon_reason": "fewer_than_two_nonzero_differences",
            "wilcoxon_statistic": None,
            "p_value": None,
            "p_value_holm": None,
            # C7: el titular ponderado por token lleva el IC de su propio cociente.
            "estimand": "mean_over_works_of_work_nll_difference",
            "token_weighted_estimand": "token_weighted_nll_difference",
            "token_weighted_difference": None,
            "token_weighted_bootstrap_95_ci": None,
            **_empty_holm("work_mean"),
            **_empty_holm("token_weighted"),
        }
        token_weighted = _token_weighted_difference(
            full_fraction_totals.get(model_a, {}),
            full_fraction_totals.get(model_b, {}),
            canonical_work_ids,
            bootstrap_samples=validated_bootstrap_samples,
            generator=generator,
        )
        if token_weighted is not None:
            comparison["token_weighted_difference"] = token_weighted[0]
            comparison["token_weighted_bootstrap_95_ci"] = token_weighted[1]
            token_weighted_draws[len(comparisons_payload)] = token_weighted[2]
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
    _apply_bootstrap_holm(comparisons_payload, work_mean_draws, prefix="work_mean")
    _apply_bootstrap_holm(comparisons_payload, token_weighted_draws, prefix="token_weighted")
    holm_family = {
        "expected_models": sorted(model_names),
        **_holm_family_record(comparisons_payload),
    }
    # Con varias semillas de partición, las obras se repiten entre particiones y
    # no pueden agruparse como observaciones nuevas: se compara dentro de cada
    # partición y se reporta la variación aparte.
    materialized_rows = list(rows_list)
    split_seeds = sorted(
        {seed for seed in (_split_seed_of(row) for row in materialized_rows) if seed is not None}
    )
    comparisons_by_split_seed: dict[str, list[dict[str, object]]] = {}
    holm_family_by_split_seed: dict[str, dict[str, object]] = {}
    if len(split_seeds) > 1:
        for split_seed in split_seeds:
            subset = [
                row for row in materialized_rows if _split_seed_of(row) == split_seed
            ]
            nested = pairwise_model_comparisons(
                subset,
                bootstrap_samples=validated_bootstrap_samples,
                seed=validated_seed,
                expected_models=sorted(model_names),
            )
            comparisons_by_split_seed[str(split_seed)] = nested["comparisons"]
            holm_family_by_split_seed[str(split_seed)] = nested["holm_family"]
    # El denominador de cada comparación es una obra canónica, no un archivo:
    # dejarlo explícito evita citar 424 archivos donde hay 414 obras.
    denominators = {
        "unit": "canonical_work",
        "n_works_by_model": {
            model: len(works_by_model.get(model, set())) for model in sorted(model_names)
        },
        "n_works_any_model": len({work for works in works_by_model.values() for work in works}),
    }
    full_fraction_runs = {
        model: counts
        for (model, fraction), counts in sorted(
            run_counts_by_model_and_fraction(
                row for row in materialized_rows if _full_fraction(row)
            ).items()
        )
    }
    return {
        "denominators": denominators,
        # C4/(e): n real de corridas en frac = 1.0, aparte del número de semillas.
        "full_fraction_runs": full_fraction_runs,
        "split_seeds": split_seeds,
        "pooling": (
            "per_split_seed_never_pooled_across_splits"
            if len(split_seeds) > 1
            else "single_split_seed"
        ),
        "comparisons_by_split_seed": comparisons_by_split_seed,
        "holm_family_by_split_seed": holm_family_by_split_seed,
        "split_variation": summarize_split_variation(materialized_rows),
        "metric": "test_nll",
        "decision": dict(DECISION_RULE),
        "difference_orientation": "model_a_minus_model_b",
        "interpretation": "negative_favors_model_a",
        "bootstrap_samples": validated_bootstrap_samples,
        "seed": validated_seed,
        "models": sorted(model_names),
        "n_comparisons": len(comparisons_payload),
        "n_valid_tests": n_valid_tests,
        "holm_family": holm_family,
        "comparisons": comparisons_payload,
    }
