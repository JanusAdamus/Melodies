"""Lectura prudente de estabilidad para las cadenas del HDP-HMM.

Una traza plana no es convergencia. Este módulo compara ventanas temprana y
tardía, mide autocorrelación de retardo 1 y contrasta cadenas entre semillas;
cuando la evidencia no alcanza, lo dice con ``diagnostics_inconclusive`` en vez
de declarar convergencia.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import math
from pathlib import Path

POLICY = (
    "reports window stability and cross-chain agreement; never claims convergence "
    "from a flat trace alone"
)

#: Tolerancias de acuerdo entre cadenas.
_LOG_LIKELIHOOD_TOLERANCE_RATIO = 0.02
_ACTIVE_STATES_TOLERANCE = 1.0
#: Deriva relativa admitida entre la ventana temprana y la tardía.
_DRIFT_TOLERANCE_RATIO = 0.01


def _finite(values) -> list[float]:
    finite: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return finite


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def lag_one_autocorrelation(values: Sequence[float]) -> float | None:
    """Autocorrelación de retardo 1. ``None`` si la serie es corta o constante."""

    series = _finite(values)
    if len(series) < 3:
        return None
    mean = sum(series) / len(series)
    centered = [value - mean for value in series]
    denominator = sum(value * value for value in centered)
    if denominator <= 0.0:
        return None
    numerator = sum(first * second for first, second in zip(centered, centered[1:]))
    return numerator / denominator


def _reject_mixed_chains(rows: Sequence[Mapping[str, object]]) -> None:
    iterations = [row.get("iteration") for row in rows if row.get("iteration") is not None]
    if len(iterations) != len(set(iterations)):
        raise ValueError(
            "a chain repeats iteration numbers: rows from distinct chains or cells were "
            "concatenated; group traces by (split_seed, data_seed, frac, model_seed) first"
        )


def _chain_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _reject_mixed_chains(rows)
    log_likelihoods = [row.get("log_likelihood") for row in rows]
    active_states = [row.get("active_states") for row in rows]
    finite_log_likelihoods = _finite(log_likelihoods)
    n_non_finite = len(log_likelihoods) - len(finite_log_likelihoods)

    half = max(1, len(finite_log_likelihoods) // 2)
    early = finite_log_likelihoods[:half]
    late = finite_log_likelihoods[half:]
    early_mean = _mean(early)
    late_mean = _mean(late)

    drift = None
    window_stability = None
    if early_mean is not None and late_mean is not None:
        drift = late_mean - early_mean
        scale = max(abs(early_mean), abs(late_mean), 1.0)
        window_stability = abs(drift) <= _DRIFT_TOLERANCE_RATIO * scale

    return {
        "n_iterations": len(rows),
        "n_non_finite": n_non_finite,
        "early_window_mean_log_likelihood": early_mean,
        "late_window_mean_log_likelihood": late_mean,
        "log_likelihood_drift": drift,
        "window_stability": window_stability,
        "lag_one_autocorrelation": lag_one_autocorrelation(finite_log_likelihoods),
        "mean_active_states": _mean(_finite(active_states)),
        "late_mean_active_states": _mean(_finite(active_states)[half:]),
    }


def build_chain_diagnostics(
    traces_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    min_iterations: int = 50,
) -> dict[str, object]:
    """Resume estabilidad por cadena y acuerdo entre cadenas."""

    chains = {
        str(seed): _chain_diagnostics(rows) for seed, rows in sorted(traces_by_seed.items())
    }
    n_chains = len(chains)
    too_short = [
        name for name, chain in chains.items() if int(chain["n_iterations"]) < min_iterations
    ]

    late_means = [
        chain["late_window_mean_log_likelihood"]
        for chain in chains.values()
        if chain["late_window_mean_log_likelihood"] is not None
    ]
    late_states = [
        chain["late_mean_active_states"]
        for chain in chains.values()
        if chain["late_mean_active_states"] is not None
    ]
    log_likelihood_agree: bool | None = None
    active_states_agree: bool | None = None
    if len(late_means) > 1:
        spread = max(late_means) - min(late_means)
        scale = max(abs(value) for value in late_means) or 1.0
        log_likelihood_agree = spread <= _LOG_LIKELIHOOD_TOLERANCE_RATIO * scale
    if len(late_states) > 1:
        active_states_agree = (max(late_states) - min(late_states)) <= _ACTIVE_STATES_TOLERANCE

    stability_flags = [
        chain["window_stability"] for chain in chains.values() if chain["window_stability"] is not None
    ]
    if not chains:
        status, reason, verdict = "diagnostics_inconclusive", "no_chains", "no_evidence"
    elif too_short:
        status = "diagnostics_inconclusive"
        reason = "chains_shorter_than_min_iterations"
        verdict = "chains_too_short"
    elif stability_flags and not all(stability_flags):
        status, reason, verdict = "ok", None, "drift_detected"
    elif log_likelihood_agree is False or active_states_agree is False:
        status, reason, verdict = "ok", None, "chains_disagree"
    elif n_chains < 2:
        status, reason, verdict = "ok", None, "single_chain_no_cross_chain_evidence"
    else:
        status, reason, verdict = "ok", None, "stable_within_the_observed_window"

    return {
        "status": status,
        "reason": reason,
        "verdict": verdict,
        "n_chains": n_chains,
        "min_iterations": int(min_iterations),
        "chains": chains,
        "cross_chain": {
            "log_likelihood_agree": log_likelihood_agree,
            "active_states_agree": active_states_agree,
            "late_window_means": late_means,
            "late_mean_active_states": late_states,
        },
        # La estabilidad observada nunca se reporta como convergencia.
        "convergence_claimed": False,
        "policy": POLICY,
    }


#: Del veredicto más preocupante al menos preocupante; el global es el peor por celda.
_VERDICT_SEVERITY = (
    "no_evidence",
    "chains_too_short",
    "drift_detected",
    # Semillas de una misma celda que eligieron hiperparámetros distintos y cuyas
    # cadenas discrepan: posible multimodalidad de la selección.  No se suaviza a
    # «una sola cadena» por haberlas separado por posterior.
    "selected_hyperparameters_disagree",
    "chains_disagree",
    "single_chain_no_cross_chain_evidence",
    "stable_within_the_observed_window",
)

_CELL_KEYS = ("split_seed", "data_seed", "frac")


def _in_burn_in(row: Mapping[str, object], burn_in: int) -> bool:
    return int(row["iteration"]) <= burn_in


#: Hiperparámetros que definen la posterior; cada model_seed los elige en validación.
_POSTERIOR_KEYS = ("alpha", "alpha0", "gamma", "kappa", "eta", "n_states_truncation")


def build_cell_chain_diagnostics(
    trace_rows: Sequence[Mapping[str, object]],
    *,
    burn_in: int,
    min_iterations: int,
) -> dict[str, object]:
    """Diagnóstico por celda (split_seed, data_seed, frac) y posterior, sin burn-in.

    Cada ``model_seed`` elige sus hiperparámetros en validación; dos semillas que
    eligen hiperparámetros distintos muestrean posteriores distintas.  Por eso una
    cadena es una ``model_seed`` dentro de (celda, hiperparámetros seleccionados) y
    sólo se comparan cadenas de la misma posterior.  Celdas distintas nunca se
    concatenan (H7).
    """

    groups: dict[
        tuple[tuple[object, ...], tuple[object, ...]], dict[int, list[Mapping[str, object]]]
    ] = {}
    chains_by_cell: dict[tuple[object, ...], dict[int, list[Mapping[str, object]]]] = {}
    for row in trace_rows:
        if _in_burn_in(row, burn_in):
            continue
        cell_key = tuple(row.get(name) for name in _CELL_KEYS)
        posterior_key = tuple(row.get(name) for name in _POSTERIOR_KEYS)
        groups.setdefault((cell_key, posterior_key), {}).setdefault(row["model_seed"], []).append(row)
        chains_by_cell.setdefault(cell_key, {}).setdefault(row["model_seed"], []).append(row)

    posteriors_by_cell: dict[tuple[object, ...], set[tuple[object, ...]]] = {}
    for cell_key, posterior_key in groups:
        posteriors_by_cell.setdefault(cell_key, set()).add(posterior_key)
    mixed_cells = [
        dict(zip(_CELL_KEYS, cell_key))
        for cell_key, posteriors in sorted(posteriors_by_cell.items(), key=lambda item: tuple(map(str, item[0])))
        if len(posteriors) > 1
    ]

    cell_reports = [
        {
            "cell": dict(zip(_CELL_KEYS, cell_key)),
            "posterior": {
                name: value
                for name, value in zip(_POSTERIOR_KEYS, posterior_key)
                if value is not None
            },
            "diagnostics": build_chain_diagnostics(chains, min_iterations=min_iterations),
        }
        for (cell_key, posterior_key), chains in sorted(
            groups.items(), key=lambda item: (tuple(map(str, item[0][0])), tuple(map(str, item[0][1])))
        )
    ]
    # En una celda mixta las cadenas no estiman la misma posterior, pero que las
    # semillas discrepen entre sí es evidencia que el veredicto global no puede
    # perder: se contrastan todas las semillas de la celda y, si discrepan, la celda
    # cuenta como «selected_hyperparameters_disagree».
    mixed_reports = []
    for cell in mixed_cells:
        cell_key = tuple(cell[name] for name in _CELL_KEYS)
        diagnostics = build_chain_diagnostics(chains_by_cell[cell_key], min_iterations=min_iterations)
        cross = diagnostics["cross_chain"]
        disagree = cross["log_likelihood_agree"] is False or cross["active_states_agree"] is False
        mixed_reports.append(
            {
                "cell": cell,
                "verdict": (
                    "selected_hyperparameters_disagree" if disagree else "selected_hyperparameters_agree"
                ),
                "diagnostics": diagnostics,
            }
        )
    verdict_counts: dict[str, int] = {}
    for report in cell_reports:
        verdict = str(report["diagnostics"]["verdict"])
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    for report in mixed_reports:
        if report["verdict"] == "selected_hyperparameters_disagree":
            verdict_counts[report["verdict"]] = verdict_counts.get(report["verdict"], 0) + 1
    present = [verdict for verdict in _VERDICT_SEVERITY if verdict in verdict_counts]
    return {
        "status": "ok" if cell_reports else "diagnostics_inconclusive",
        "verdict": present[0] if present else "no_evidence",
        "verdict_counts": verdict_counts,
        "unit": "chain = model_seed within (split_seed, data_seed, frac) and selected hyperparameters",
        "posterior_keys": list(_POSTERIOR_KEYS),
        # Celdas donde las semillas eligieron hiperparámetros distintos: sus cadenas
        # no se comparan entre sí, y hay menos evidencia entre cadenas por posterior.
        "cells_with_mixed_selected_hyperparameters": mixed_cells,
        "mixed_cell_cross_posterior_diagnostics": mixed_reports,
        "burn_in_excluded": int(burn_in),
        "min_iterations": int(min_iterations),
        # n_cells: celdas (split_seed, data_seed, frac) distintas; cada celda puede
        # tener varios grupos (celda, posterior) en «cells».
        "n_cells": len(chains_by_cell),
        "n_cell_posterior_groups": len(cell_reports),
        "cells": cell_reports,
        "convergence_claimed": False,
        "policy": POLICY,
    }


def write_cell_traces(
    trace_rows: Sequence[Mapping[str, object]], output_root: str | Path, *, burn_in: int
) -> str | None:
    """Escribe hdp_traces.csv en formato largo, con la marca ``in_burn_in``."""

    if not trace_rows:
        return None
    rows = [{**row, "in_burn_in": _in_burn_in(row, burn_in)} for row in trace_rows]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "hdp_traces.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
