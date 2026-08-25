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


def _chain_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
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


def write_chain_traces(
    traces_by_seed: Mapping[int, Sequence[Mapping[str, object]]], output_root: str | Path
) -> list[str]:
    """Escribe hdp_trace_SEED.csv, una traza por semilla de modelo."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for seed, rows in sorted(traces_by_seed.items()):
        if not rows:
            continue
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        path = root / f"hdp_trace_{seed}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        written.append(str(path))
    return written
