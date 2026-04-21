from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.finite_hmm import CHORD_STATES, FiniteHMMResult
from src.models.hdp_hmm import HDPHMMResult
from src.models.utils import contiguous_segments, normalize


def trajectory_stability(state_samples: list[np.ndarray]) -> float | None:
    """Mide estabilidad como frecuencia modal promedio por posicion temporal."""

    if not state_samples:
        return None
    stacked = np.vstack(state_samples)
    scores = []
    for column in stacked.T:
        values, counts = np.unique(column, return_counts=True)
        _ = values
        scores.append(float(np.max(counts) / len(column)))
    return float(np.mean(scores))


def segmentation_statistics(states: np.ndarray) -> dict[str, float]:
    segments = contiguous_segments(states)
    lengths = [end - start for _, start, end in segments]
    return {
        "n_segments": float(len(segments)),
        "mean_segment_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_segment_length": float(np.median(lengths)) if lengths else 0.0,
    }


def sequence_dataframe(
    states: np.ndarray,
    state_labels: list[str],
    observation_labels: list[str],
    offsets: list[float],
) -> pd.DataFrame:
    rows = []
    for index, (state, observation, offset) in enumerate(zip(states, observation_labels, offsets)):
        rows.append(
            {
                "t": index,
                "offset": offset,
                "state": int(state),
                "state_label": state_labels[int(state)],
                "observation": observation,
            }
        )
    return pd.DataFrame(rows)


def matrix_to_dataframe(matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=row_labels, columns=col_labels)


def active_submatrix(matrix: np.ndarray, active_states: list[int]) -> np.ndarray:
    return matrix[np.ix_(active_states, active_states)]


def summarize_finite_result(result: FiniteHMMResult) -> pd.DataFrame:
    segmentation = segmentation_statistics(result.latent_states)
    return pd.DataFrame(
        {
            "metric": [
                "log_likelihood",
                "viterbi_score",
                "n_active_states",
                "n_segments",
                "mean_segment_length",
            ],
            "value": [
                result.log_likelihood,
                result.viterbi_score,
                len(result.active_states),
                segmentation["n_segments"],
                segmentation["mean_segment_length"],
            ],
        }
    )


def summarize_hdp_result(result: HDPHMMResult) -> pd.DataFrame:
    segmentation = segmentation_statistics(result.latent_states)
    stability = trajectory_stability(result.diagnostics.state_samples)
    return pd.DataFrame(
        {
            "metric": [
                "log_likelihood",
                "n_active_states",
                "n_segments",
                "mean_segment_length",
                "trajectory_stability",
                "best_iteration",
            ],
            "value": [
                result.log_likelihood,
                result.effective_states,
                segmentation["n_segments"],
                segmentation["mean_segment_length"],
                stability,
                result.diagnostics.best_iteration,
            ],
        }
    )


def compare_models(
    finite_result: FiniteHMMResult | None,
    hdp_result: HDPHMMResult | None,
) -> pd.DataFrame:
    rows = []
    if finite_result is not None:
        finite_seg = segmentation_statistics(finite_result.latent_states)
        rows.append(
            {
                "model": "finite_hmm",
                "log_likelihood": finite_result.log_likelihood,
                "effective_states": len(finite_result.active_states),
                "n_segments": finite_seg["n_segments"],
                "mean_segment_length": finite_seg["mean_segment_length"],
                "trajectory_stability": 1.0,
            }
        )
    if hdp_result is not None:
        hdp_seg = segmentation_statistics(hdp_result.latent_states)
        rows.append(
            {
                "model": "hdp_hmm",
                "log_likelihood": hdp_result.log_likelihood,
                "effective_states": hdp_result.effective_states,
                "n_segments": hdp_seg["n_segments"],
                "mean_segment_length": hdp_seg["mean_segment_length"],
                "trajectory_stability": trajectory_stability(hdp_result.diagnostics.state_samples),
            }
        )
    return pd.DataFrame(rows)


def export_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    excel_name: str = "resultados.xlsx",
) -> dict[str, Path]:
    """Exporta tablas en CSV y, si es posible, en un workbook Excel."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    for name, dataframe in tables.items():
        csv_path = output_path / f"{name}.csv"
        dataframe.to_csv(csv_path, index=True)
        exported[name] = csv_path

    try:
        excel_path = output_path / excel_name
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for name, dataframe in tables.items():
                safe_name = name[:31]
                dataframe.to_excel(writer, sheet_name=safe_name)
        exported["excel"] = excel_path
    except Exception:
        pass

    return exported


def build_finite_tables(
    observations_df: pd.DataFrame,
    result: FiniteHMMResult,
) -> dict[str, pd.DataFrame]:
    offsets = [float(event.offset) for event in result.observations.events]
    sequence_df = pd.DataFrame(
        {
            "t": np.arange(result.observations.size),
            "offset": offsets,
            "state": result.latent_states,
            "state_label": result.latent_labels,
            "modal_label": result.modal_labels,
            "observation": result.observations.decoded,
        }
    )
    active_labels = [result.state_space[state].label for state in result.active_states]
    return {
        "observations": observations_df,
        "latent_sequence": sequence_df,
        "summary": summarize_finite_result(result),
        "transition_matrix_full": matrix_to_dataframe(
            result.empirical_transition_matrix,
            [state.label for state in result.state_space],
            [state.label for state in result.state_space],
        ),
        "transition_matrix_active": matrix_to_dataframe(
            active_submatrix(result.empirical_transition_matrix, result.active_states),
            active_labels,
            active_labels,
        ),
        "emission_matrix_full": matrix_to_dataframe(
            result.emission_matrix,
            [state.label for state in result.state_space],
            result.observations.vocabulary,
        ),
        "harmonic_vocabulary": result.vocabulary_dataframe(),
    }


def build_hdp_tables(
    observations_df: pd.DataFrame,
    result: HDPHMMResult,
    interpretation_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    offsets = [float(event.offset) for event in result.observations.events]
    sequence_df = pd.DataFrame(
        {
            "t": np.arange(result.observations.size),
            "offset": offsets,
            "state": result.latent_states,
            "state_label": [f"z{state}" for state in result.latent_states],
            "observation": result.observations.decoded,
        }
    )
    state_labels = [f"z{index}" for index in range(result.posterior_transition_mean.shape[0])]
    active_labels = [f"z{state}" for state in result.active_states]
    tables = {
        "observations": observations_df,
        "latent_sequence": sequence_df,
        "summary": summarize_hdp_result(result),
        "transition_matrix_full": matrix_to_dataframe(
            result.posterior_transition_mean,
            state_labels,
            state_labels,
        ),
        "transition_matrix_active": matrix_to_dataframe(
            active_submatrix(result.posterior_transition_mean, result.active_states),
            active_labels,
            active_labels,
        ),
        "emission_matrix_full": matrix_to_dataframe(
            result.posterior_emission_mean,
            state_labels,
            result.observations.vocabulary,
        ),
        "diagnostics": result.diagnostics.to_dataframe(),
    }
    if interpretation_df is not None:
        tables["interpretation"] = interpretation_df
    return tables
