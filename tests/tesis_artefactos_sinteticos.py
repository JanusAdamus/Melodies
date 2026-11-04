"""Artefactos sintéticos con el formato de las corridas reales, para probar el generador
de tablas y el verificador de cifras de la tesis sin correr modelos.

* Por obra: el formato final de ``src/cli/porobra_heldout.py`` (``per_work.csv``,
  ``summary.json``, ``manifest.json``, ``porobra_audit.json``).  Las filas por obra son
  sintéticas (NLL elegidas a mano, nada de obras reales); los bloques del resumen los
  calcula el código real (``porobra.summary_blocks``) desde esas filas, y el CSV se escribe
  como lo escribe el CLI (listas y diccionarios en JSON).  Con 12 obras: la última no
  converge, las dos primeras llevan ids de obras expuestas (Aclaraciones 8 y 9), tres
  eligen el tope K = 512 (Enmienda 11; sin ellas quedan 9 obras: «no_evaluable») y una
  termina el reajuste del finito por el tope de iteraciones de EM.  El manifiesto lleva los
  hashes de ``outputs`` y la auditoría la ruta de la corrida (``run_dir``), como el CLI.
* PDMX: el de ``Comparacion/runner.py``: ``results_summary.csv``,
  ``headline_estimates.json``, ``pairwise_comparisons.json``, ``pareto_summary.json``,
  ``engineering_costs.csv`` y ``hdp_chain_diagnostics.json``, más ``config.json``,
  ``preprocessing_report.json`` y las auditorías (``protocol_audit.json``,
  ``training_exposure_audit.json``, ``denominator_audit.json``).  Por omisión, con el
  diseño de la corrida principal de la sección B.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.analysis import porobra

POROBRA_RUN_NAME = "porobra_heldout_20261101_abcdef1"
EXPOSED_SYNTHETIC = ("MuseTrainer:Bella_Ciao.mxl", "MuseTrainer:Passacaglia.mxl")
K_CEILING_INDICES = (2, 3, 4)
EM_CAP_INDEX = 5
CHAIN_OFFSETS = (-0.02, 0.0, 0.01, 0.03)


def porobra_rows(n: int = 12) -> list[dict]:
    """``n`` obras sintéticas, ordenadas por ``work_id`` como las deja el CLI."""

    rows = []
    for index in range(n):
        source = "MuseTrainer" if index < (2 * n) // 3 else "SymbTr"
        work_id = EXPOSED_SYNTHETIC[index] if index < len(EXPOSED_SYNTHETIC) else f"{source}:obra_{index:02d}.mxl"
        base = 2.0 + 0.01 * index
        by_chain = [base + offset for offset in CHAIN_OFFSETS]
        ceiling = index in K_CEILING_INDICES
        grid = [2, 4, 8, 16, 32, 64, 128, 256, 512] if ceiling else [2, 4, 8, 16, 32]
        converged = index != n - 1
        states = [4.0 + 0.1 * index + d for d in (0.0, 0.2, -0.1, 0.1)]
        rows.append({
            "work_id": work_id,
            "source": source,
            "relative_path": work_id.split(":", 1)[1],
            "makam": "" if source == "MuseTrainer" else "hicaz",
            "content_fingerprint": f"huella{index}",
            "n_events": 125 + 5 * index,
            "n_fit": 100 + 4 * index,
            "n_test": 25 + index,
            "n_inner_train": 80 + 3 * index,
            "n_validation": 20 + index,
            "unigram_alpha": 0.5,
            "unigram_alpha_validation_nll": {"0.1": 2.5, "0.5": 2.4, "1.0": 2.45},
            "unigram_test_nll": base + 0.4 + 0.01 * (index % 2),
            "bigram_alpha": 1.0,
            "bigram_alpha_validation_nll": {"0.1": 2.3, "0.5": 2.2, "1.0": 2.1},
            "bigram_test_nll": base + (0.02 if index % 2 else -0.02),
            "finite_k": 512 if ceiling else 8,
            "finite_alpha": 0.1 if index < 9 else 1.0,
            "finite_restarts": 5,
            "finite_validation_nll": {"0.1": {"8": 2.3}},
            "finite_selected_validation_nll": 2.3,
            "finite_k_grid": grid,
            "finite_grid_extensions": len(grid) - 5,
            "finite_k_hit_limit": ceiling,
            "finite_em_hit_max_iterations": index == EM_CAP_INDEX,
            "finite_test_nll": base + 0.1 + 0.005 * (index % 3),
            "finite_states_mean": 5.0 + 0.05 * index,
            "finite_states_sd": 0.25,
            "finite_insample_ll_per_note": -2.0,
            "hdp_eta": 1.0 if index == 8 else 3.0,
            "hdp_eta_validation_nll": {"1.0": 2.2, "3.0": 2.1, "6.0": 2.15},
            "hdp_test_nll": base + 0.005,
            "hdp_rhat": 1.01 if converged else 1.4,
            "hdp_converged": converged,
            "hdp_test_nll_by_chain": by_chain,
            "hdp_test_nll_chain_range": max(by_chain) - min(by_chain),
            "hdp_predictive_rhat": 1.02 if converged else 1.3,
            "hdp_states_mean_by_chain": states,
            "hdp_states_sd_by_chain": [0.5, 0.4, 0.6, 0.5],
            "hdp_states_pooled_chains_mean": sum(states) / len(states),
            "hdp_states_pooled_chains_sd": 0.55,
            "hdp_insample_ll_per_note": -1.9,
            "hdp_posterior_mean_log_likelihood_per_note": -1.95,
            "hdp_wall_clock_s": 10.0,
            "wall_clock_s": 12.0,
        })
    return sorted(rows, key=lambda row: row["work_id"])


def _distribution(values) -> dict:
    series = pd.Series(list(values))
    return {str(key): int(count) for key, count in series.value_counts().sort_index().items()}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def porobra_summary(rows: list[dict], *, bootstrap_samples: int = 1000, run_kind: str = "principal",
                    truncation: int = 20, analysis_failures: list[dict] | None = None) -> dict:
    """``summary.json`` con los bloques que calcula el código real desde las filas.

    Sin ``analysis_failures`` por omisión: la auditoría marca «failed» una corrida que las tiene.
    """

    analysis_failures = list(analysis_failures or [])
    records = [{"work_id": r["work_id"], "n_test": int(r["n_test"]), "hdp_converged": bool(r["hdp_converged"]),
                "finite_k_hit_limit": bool(r["finite_k_hit_limit"]),
                **{f"{m}_test_nll": float(r[f"{m}_test_nll"]) for m in porobra.MODELS}} for r in rows]
    blocks = porobra.summary_blocks(records, bootstrap_samples=bootstrap_samples, seed=0,
                                    exposed_work_ids=porobra.TEST_EXPOSED_WORK_IDS)
    analyzed = _distribution(r["source"] for r in rows)
    return _jsonable({
        "preregistro": "docs/preregistro-2026-10-26.md, sección A, Aclaración 2, Enmiendas 3, 5, 6 y 11, Aclaraciones 7, 8, 9 y 10",
        "run": {"truncation": truncation, "run_kind": run_kind, "note": "sintética", "deviations": [],
                "decides": "Sus veredictos deciden sólo si porobra_audit.json dice confirmatory: true.",
                "extraction_failures": ["MuseTrainer:obra_rota.mxl"]},
        "corpus": {
            "extraction": "melody",
            "n_candidates": len(rows) + 4,
            "n_by_source_candidates": {k: v + (3 if k == "MuseTrainer" else 1) for k, v in analyzed.items()},
            "n_failures": 1,
            "failures": [{"source": "MuseTrainer", "relative_path": "obra_rota.mxl", "path": "x", "error_stage": "parse",
                          "error": "MusicXMLImportException"}],
            "n_duplicates": 1,
            "duplicates": [{"source": "MuseTrainer", "relative_path": "obra_dup.mxl", "path": "x", "duplicate_of": "y"}],
            "min_events": 20,
            "n_excluded_short": 1,
            "excluded_short_works": [{"source": "SymbTr", "relative_path": "corta.xml", "path": "x", "n_events": 12}],
            "n_analysis_failures": len(analysis_failures),
            "n_analyzed": len(rows),
            "n_by_source_analyzed": analyzed,
        },
        "analysis_failures": analysis_failures,
        "convergence": blocks["convergence"],
        "comparisons": blocks["comparisons"],
        "sensitivity_converged_only": blocks["sensitivity_converged_only"],
        "test_exposure": {
            "exposed_work_ids": list(porobra.TEST_EXPOSED_WORK_IDS),
            "exposed_works": dict(porobra.TEST_EXPOSED_WORKS),
            "exposed_in_run": sorted(set(porobra.TEST_EXPOSED_WORK_IDS) & {r["work_id"] for r in rows}),
            "note": porobra.TEST_EXPOSURE_NOTE,
        },
        "sensitivity_without_exposed": blocks["sensitivity_without_exposed"],
        "sensitivity_without_k_ceiling": blocks["sensitivity_without_k_ceiling"],
        "hdp_states_note": porobra.HDP_STATES_NOTE,
        "selected_hyperparameters": {
            "hdp_eta": _distribution(r["hdp_eta"] for r in rows),
            "finite_k": _distribution(r["finite_k"] for r in rows),
            "finite_alpha": _distribution(r["finite_alpha"] for r in rows),
            "finite_k_hit_limit": [r["work_id"] for r in rows if r["finite_k_hit_limit"]],
            "finite_em_hit_max_iterations": [r["work_id"] for r in rows if r["finite_em_hit_max_iterations"]],
            "unigram_alpha": _distribution(r["unigram_alpha"] for r in rows),
            "bigram_alpha": _distribution(r["bigram_alpha"] for r in rows),
        },
        "wall_clock_s": 12.5,
    })


def porobra_audit(*, status: str = "passed", confirmatory: bool = True, run_kind: str = "principal",
                  deviations: dict | None = None) -> dict:
    return {
        "status": status,
        "preregistration": {"status": "not_preregistered" if deviations else "preregistered", "run_kind": run_kind,
                            "run_kind_note": "sintética", "deviations": deviations or {}},
        "confirmatory": confirmatory,
        "run_dir": "sintética",
        "checks": [],
    }


def write_porobra_run(root: Path, *, name: str = POROBRA_RUN_NAME, rows=None, summary=None, audit=None,
                      config: dict | None = None) -> Path:
    """Escribe la corrida como el CLI: ``per_work.csv`` con pandas, listas y diccionarios en JSON."""

    run = root / name
    run.mkdir(parents=True)
    rows = porobra_rows() if rows is None else rows
    summary = porobra_summary(rows) if summary is None else summary
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True))
    frame.to_csv(run / "per_work.csv", index=False)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_config = {"truncation": 20, "max_states": 512, "n_chains": 4, "rhat_threshold": 1.1, "min_events": 20,
                       **(config or {})}
    (run / "README.txt").write_text("sintética\n", encoding="utf-8")
    outputs = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in ("per_work.csv", "summary.json", "README.txt")}
    (run / "manifest.json").write_text(json.dumps({"commit": "abcdef1234567", "config": manifest_config, "outputs": outputs}),
                                       encoding="utf-8")
    audit = porobra_audit() if audit is None else audit
    if audit.get("run_dir") == "sintética":
        audit = {**audit, "run_dir": run.as_posix()}  # como el CLI: la ruta de la corrida auditada
    (run / "porobra_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
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


#: Diseño de la corrida principal de B en ``config.json`` (``asdict(LearningCurveConfig)``): los valores por omisión
#: de ``Comparacion/config.py`` y el comando de la tarea 2.2 del handoff.
PDMX_MAIN_CONFIG = {
    "split_seed": 7, "data_seeds": [1, 2, 3], "model_seeds": [1, 2], "train_fractions": [0.1, 0.25, 0.5, 0.75, 1.0],
    "test_fraction": 0.15, "validation_fraction": 0.1, "include_vomm_control": True,
    "experiment": {"windows": {"max_context_length": 128, "min_window_length": 32, "train_stride": 64, "eval_stride": 128},
                   "transformer": {"max_epochs": 25, "n_layers": 3}},
}


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def write_pdmx_run(root: Path, *, name: str = "tesis_corregida_20261101_abcdef1", comparisons=None, status="preregistered",
                   gaps=(), config: dict | None = None, n_files: int = 3000) -> Path:
    """``config`` cambia campos de ``config.json``; ``n_files``, las partituras consideradas (preparadas + excluidas)."""

    run = root / name
    run.mkdir(parents=True)
    run_config = _merge({**PDMX_MAIN_CONFIG, "include_vomm_control": status == "preregistered"}, config or {})
    (run / "config.json").write_text(json.dumps(run_config), encoding="utf-8")
    (run / "preprocessing_report.json").write_text(json.dumps({"n_prepared_pieces": n_files - 3, "n_exclusions": 3,
                                                               "n_total_tokens": 99, "n_total_events": 99}), encoding="utf-8")
    (run / "protocol_audit.json").write_text(json.dumps({
        "status": "passed", "policy": "every musical test event must be exposed by model evaluation exactly once",
        "unexpected_piece_ids": [], "evidence": []}), encoding="utf-8")
    (run / "training_exposure_audit.json").write_text(json.dumps({
        "max_context_length": 128, "train_stride": 64, "eval_stride": 128, "splits": {},
        "evaluation_is_non_overlapping": True}), encoding="utf-8")
    (run / "denominator_audit.json").write_text(json.dumps({
        "n_scored_files": 30, "n_canonical_works": 30, "unscored_test_files": [], "discards": [], "pairs": [],
        "explanation": "files_equal_works"}), encoding="utf-8")
    analysis_status = {"status": status, "evaluates_preregistered_rule_B": status == "preregistered",
                       "preregistration_gaps": list(gaps)}
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
