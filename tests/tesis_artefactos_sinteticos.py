"""Artefactos sintéticos con el formato de las corridas reales, para probar el generador
de tablas y el verificador de cifras de la tesis sin correr modelos.

* Por obra: el de ``src/cli/porobra_heldout.py`` (rama ``feat/porobra-heldout``):
  ``per_work.csv``, ``summary.json`` y ``manifest.json``.
* PDMX: el de ``Comparacion/runner.py``: ``results_summary.csv``,
  ``headline_estimates.json``, ``pairwise_comparisons.json``, ``pareto_summary.json``,
  ``engineering_costs.csv`` y ``hdp_chain_diagnostics.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

POROBRA_MODELS = ("hdp", "finite", "bigram", "unigram")


def _holm(status: str, decision_ci, rejects) -> dict:
    return {
        "work_mean_holm_status": status,
        "work_mean_holm_decision_ci": decision_ci,
        "work_mean_holm_rejects_zero": rejects,
        "work_mean_holm_ci": decision_ci,
    }


def porobra_rows() -> list[dict]:
    """Doce obras; la última no converge.  Las NLL están elegidas a mano."""

    rows = []
    for index in range(12):
        source = "MuseTrainer" if index < 8 else "SymbTr"
        rows.append(
            {
                "work_id": f"{source}:obra_{index:02d}.mxl",
                "source": source,
                "relative_path": f"obra_{index:02d}.mxl",
                "makam": "" if source == "MuseTrainer" else "hicaz",
                "content_fingerprint": f"huella{index}",
                "n_events": 100 + index,
                "n_fit": 80,
                "n_test": 20 + index,
                "hdp_test_nll": 2.0 + 0.01 * index,
                "finite_test_nll": 2.1 + 0.01 * index,
                "bigram_test_nll": 2.0 + 0.01 * index + (0.02 if index % 2 else -0.02),
                "unigram_test_nll": 2.4 + 0.01 * index,
                "hdp_converged": index != 11,
                "hdp_rhat": 1.01 if index != 11 else 1.4,
                "hdp_eta": 3.0,
                "hdp_states_mean": 4.0 + 0.1 * index,
                "hdp_states_sd": 0.5,
                "finite_k": 8,
                "finite_k_hit_limit": False,
                "finite_states_mean": 5.0,
                "finite_states_sd": 0.25,
                "unigram_alpha": 0.5,
                "bigram_alpha": 1.0,
            }
        )
    return rows


def _mean_difference(rows, other) -> float:
    kept = [row for row in rows if row["hdp_converged"]]
    return sum(row["hdp_test_nll"] - row[f"{other}_test_nll"] for row in kept) / len(kept)


def porobra_summary(rows: list[dict]) -> dict:
    converged = [row for row in rows if row["hdp_converged"]]
    comparisons = []
    holm = {
        "finite": _holm("ok", [-0.15, -0.05], True),
        "bigram": _holm("ok", [-0.04, 0.03], False),
        "unigram": _holm("ok", [-0.5, -0.3], True),
    }
    verdicts = {"finite": "mejora", "bigram": "no_hay_evidencia", "unigram": "mejora"}
    for other in ("finite", "bigram", "unigram"):
        comparisons.append(
            {
                "model_a": "hdp",
                "model_b": other,
                "status": "ok",
                "n_pairs": len(converged),
                "mean_difference": _mean_difference(rows, other),
                "bootstrap_95_ci": [-0.2, 0.1],
                "token_weighted_difference": -0.09,
                "token_weighted_bootstrap_95_ci": [-0.12, -0.06],
                "p_value": 0.01,
                "p_value_holm": 0.03,
                "wilcoxon_holm_status": "ok",
                **holm[other],
                "decision_estimand": "mean_over_works_of_work_nll_difference",
                "verdict": verdicts[other],
            }
        )
    return {
        "preregistro": "docs/preregistro-2026-10-26.md, sección A, Aclaración 2, Enmienda 3",
        "corpus": {
            "extraction": "melody",
            "n_candidates": 15,
            "n_by_source_candidates": {"MuseTrainer": 10, "SymbTr": 5},
            "n_failures": 1,
            "failures": [],
            "n_duplicates": 1,
            "duplicates": [],
            "min_events": 20,
            "n_excluded_short": 1,
            "excluded_short_works": [],
            "n_analyzed": len(rows),
            "n_by_source_analyzed": {"MuseTrainer": 8, "SymbTr": 4},
        },
        "convergence": {
            "rhat_threshold": 1.1,
            "n_works": len(rows),
            "n_not_converged": 1,
            "fraction_not_converged": 1 / len(rows),
            "not_converged_work_ids": [rows[-1]["work_id"]],
            "exceeds_10_percent": False,
            "action_required": None,
        },
        "comparisons": {
            "decision_rule": "media por obra",
            "bootstrap_samples": 10000,
            "seed": 0,
            "n_works_converged": len(converged),
            "comparisons": comparisons,
            "holm_family": {},
        },
        "selected_hyperparameters": {
            "hdp_eta": {"3.0": len(rows)},
            "finite_k": {"8": len(rows)},
            "finite_k_hit_limit": [],
            "unigram_alpha": {"0.5": len(rows)},
            "bigram_alpha": {"1.0": len(rows)},
        },
        "wall_clock_s": 12.5,
    }


def write_porobra_run(root: Path, *, name: str = "porobra_heldout_20261101_abcdef1", rows=None, summary=None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    rows = porobra_rows() if rows is None else rows
    summary = porobra_summary(rows) if summary is None else summary
    with (run / "per_work.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"commit": "abcdef1234567", "config": {"truncation": 20}}), encoding="utf-8")
    return run


PDMX_MODELS = ("finite_hmm", "hdp_hmm", "transformer", "vomm")
PDMX_NLL = {"finite_hmm": 2.15, "hdp_hmm": 2.22, "transformer": 2.05, "vomm": 2.20}


def _comparison(a: str, b: str, *, mean: float, decision_ci, status: str = "ok") -> dict:
    rejects = None if status != "ok" else (decision_ci[1] < 0 or decision_ci[0] > 0)
    return {
        "model_a": a,
        "model_b": b,
        "status": "ok",
        "n_pairs": 30,
        "mean_difference": mean,
        "bootstrap_95_ci": [mean - 0.05, mean + 0.05],
        "token_weighted_difference": mean / 2,
        "token_weighted_bootstrap_95_ci": [mean / 2 - 0.02, mean / 2 + 0.02],
        "p_value": 0.002,
        "p_value_holm": 0.01 if status == "ok" else None,
        "wilcoxon_holm_status": status,
        "work_mean_holm_status": status,
        "work_mean_holm_decision_ci": decision_ci if status == "ok" else None,
        "work_mean_holm_rejects_zero": rejects,
        "work_mean_holm_ci": decision_ci if status == "ok" else None,
    }


def pdmx_comparisons() -> list[dict]:
    return [
        _comparison("finite_hmm", "hdp_hmm", mean=-0.07, decision_ci=[-0.2, 0.05]),
        # El transformador es model_b: el generador debe orientar «transformador − finito».
        _comparison("finite_hmm", "transformer", mean=0.10, decision_ci=[0.05, 0.2]),
        _comparison("finite_hmm", "vomm", mean=-0.05, decision_ci=[-0.08, -0.01]),
        _comparison("hdp_hmm", "transformer", mean=0.17, decision_ci=[0.1, 0.25]),
        _comparison("hdp_hmm", "vomm", mean=0.02, decision_ci=[-0.03, 0.06]),
        _comparison("transformer", "vomm", mean=-0.15, decision_ci=None, status="too_few_paired_works"),
    ]


def write_pdmx_run(root: Path, *, name: str = "tesis_corregida_20261101_abcdef1", comparisons=None, status="preregistered") -> Path:
    run = root / name
    run.mkdir(parents=True)
    analysis_status = {"status": status, "evaluates_preregistered_rule_B": status == "preregistered", "preregistration_gaps": []}
    fieldnames = ["model", "frac", "mean_n_train_tokens", "mean_test_nll", "runs", "n_distinct_runs",
                  "test_nll_estimand", "test_nll_ci95_low", "test_nll_ci95_high"]
    with (run / "results_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in PDMX_MODELS:
            for frac, tokens in (("0.5", 1837), ("1.0", 3361)):
                writer.writerow({"model": model, "frac": frac, "mean_n_train_tokens": tokens,
                                 "mean_test_nll": PDMX_NLL[model] + 0.3 + (0.05 if frac == "0.5" else 0.0), "runs": 6, "n_distinct_runs": 6,
                                 "test_nll_estimand": "token_weighted_nll", "test_nll_ci95_low": PDMX_NLL[model] + 0.1,
                                 "test_nll_ci95_high": PDMX_NLL[model] + 0.5})
    estimates = []
    for model in PDMX_MODELS:
        for frac in (0.5, 1.0):
            for estimand, shift, role in (("token_weighted_nll", 0.3, "secondary"), ("work_mean_nll", 0.0, "decides")):
                value = PDMX_NLL[model] + shift + (0.05 if frac == 0.5 else 0.0)
                estimates.append({"model": model, "frac": frac, "n_works": 30, "n_runs": 6, "estimand": estimand,
                                  "role": role, "point_estimate": value, "bootstrap_95_ci": [value - 0.1, value + 0.1]})
    (run / "headline_estimates.json").write_text(json.dumps({"analysis_status": analysis_status,
                                                             "decision_estimand": "work_mean_nll", "estimates": estimates}), encoding="utf-8")
    (run / "pairwise_comparisons.json").write_text(json.dumps({
        "analysis_status": analysis_status,
        "difference_orientation": "model_a_minus_model_b",
        "comparisons": pdmx_comparisons() if comparisons is None else comparisons,
    }), encoding="utf-8")
    candidates = [{"model": model, "work_mean_test_nll": PDMX_NLL[model],
                   "work_mean_test_nll_interval": [PDMX_NLL[model] - 0.1, PDMX_NLL[model] + 0.1],
                   "mean_selected_fit_wall_clock_s": seconds, "mean_selected_fit_wall_clock_s_interval": [seconds * 0.9, seconds * 1.1],
                   "mean_evaluation_wall_clock_s": 0.25, "devices": [device], "runs": 6}
                  for model, seconds, device in (("finite_hmm", 5.5, "cpu"), ("hdp_hmm", 7.25, "cpu"),
                                                 ("transformer", 0.44, "cuda"), ("vomm", 0.0143, "cpu"))]
    (run / "pareto_summary.json").write_text(json.dumps({
        "analysis_status": analysis_status,
        "predictive_cost_partial_frontier": {"status": "ok", "candidates": candidates,
                                             "frontier": [c for c in candidates if c["model"] in ("transformer", "vomm")],
                                             "devices_differ": True, "nll_paired_pairs_skipped": []},
    }), encoding="utf-8")
    cost_fields = ["model", "data_seed", "model_seed", "frac", "selected_fit_wall_clock_s", "full_search_wall_clock_s",
                   "evaluation_wall_clock_s", "device", "selection_peak_process_tree_rss_bytes", "selection_peak_cuda_allocated_bytes"]
    with (run / "engineering_costs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cost_fields)
        writer.writeheader()
        for model, device in (("vomm", "cpu"), ("finite_hmm", "cpu"), ("hdp_hmm", "cpu"), ("transformer", "cuda")):
            for frac in ("0.5", "1.0"):
                for seed in (1, 2):
                    writer.writerow({"model": model, "data_seed": 1, "model_seed": seed, "frac": frac,
                                     "selected_fit_wall_clock_s": 2.0 * seed, "full_search_wall_clock_s": 10.0 * seed,
                                     "evaluation_wall_clock_s": 0.5, "device": device,
                                     "selection_peak_process_tree_rss_bytes": 1048576 * 100 * seed,
                                     "selection_peak_cuda_allocated_bytes": 1048576 * 50 if device == "cuda" else ""})
    (run / "hdp_chain_diagnostics.json").write_text(json.dumps({
        "status": "ok", "verdict": "drift_detected", "convergence_claimed": False,
        "cells": [{"cell": {"split_seed": 7, "data_seed": 1, "frac": 1.0},
                   "diagnostics": {"verdict": "drift_detected", "n_chains": 2, "convergence_claimed": False,
                                   "chains": {"1": {"late_mean_active_states": 10.0, "lag_one_autocorrelation": 0.9},
                                              "2": {"late_mean_active_states": 8.0, "lag_one_autocorrelation": 0.8}}}}],
    }), encoding="utf-8")
    (run / "learning_curve.png").write_bytes(b"\x89PNG\r\n\x1a\nfalso")
    return run
