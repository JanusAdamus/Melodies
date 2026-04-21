from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.observations import PITCH_CLASS_NAMES
from src.models.harmony import (
    build_harmonic_state_space,
    harmonic_profile_from_observations,
    infer_mode_candidates,
    score_state_against_profile,
)
from src.models.hdp_hmm import HDPHMMResult, TruncatedHDPHMM

HARMONIC_STATE_SPACE = build_harmonic_state_space()


def _top_items(probabilities: np.ndarray, labels: list[str], n_items: int) -> str:
    indices = np.argsort(probabilities)[::-1][:n_items]
    return ", ".join(f"{labels[index]} ({probabilities[index]:.3f})" for index in indices)


def _best_harmonic_candidates(profile: np.ndarray, top_n: int = 3) -> list[tuple[str, float]]:
    scored = sorted(
        ((state.label, score_state_against_profile(profile, state)) for state in HARMONIC_STATE_SPACE),
        key=lambda item: item[1],
        reverse=True,
    )
    return scored[:top_n]


def _root_from_label(label: str) -> int | None:
    root_name = label.split(":", 1)[0]
    return PITCH_CLASS_NAMES.index(root_name) if root_name in PITCH_CLASS_NAMES else None


def _interpret_label(result: HDPHMMResult, emission_profile: np.ndarray | None) -> tuple[str, str, str]:
    if emission_profile is None:
        return "", "", "contexto latente no tonal directamente interpretable"

    harmonic_candidates = _best_harmonic_candidates(emission_profile, top_n=3)
    best_chord = harmonic_candidates[0][0]
    root = _root_from_label(best_chord)
    dominant_pitch_classes = list(np.argsort(emission_profile)[::-1][:7])
    modal_candidates = infer_mode_candidates(dominant_pitch_classes, tonic=root, top_n=3) if root is not None else []
    best_mode = modal_candidates[0][0] if modal_candidates else ""
    if best_mode:
        label = f"{best_chord} en contexto {best_mode}"
    else:
        label = f"contexto centrado en {best_chord}"
    return (
        ", ".join(f"{name} ({score:.3f})" for name, score in harmonic_candidates),
        ", ".join(f"{name} ({score:.3f})" for name, score in modal_candidates),
        label,
    )


def interpret_hdp_states(
    result: HDPHMMResult,
    max_examples: int = 5,
    top_n_emissions: int = 5,
    top_n_successors: int = 3,
) -> pd.DataFrame:
    """Genera un resumen musical interpretable para estados activos."""

    posterior_transition = result.posterior_transition_mean
    posterior_emission = result.posterior_emission_mean
    dwell = TruncatedHDPHMM.dwell_times(result.latent_states)

    rows = []
    for state in result.active_states:
        positions = np.where(result.latent_states == state)[0]
        emission_probs = posterior_emission[state]
        successor_probs = posterior_transition[state]
        successor_indices = list(np.argsort(successor_probs)[::-1][:top_n_successors])
        segment_lengths = dwell.get(state, [])
        example_positions = positions[:max_examples].tolist()
        example_offsets = [float(result.observations.events[pos].offset) for pos in example_positions]
        emission_profile = harmonic_profile_from_observations(
            observations=emission_probs,
            vocabulary=result.observations.vocabulary,
            observation_type=result.observations.observation_type,
        )
        harmonic_candidates, modal_candidates, tentative_label = _interpret_label(result, emission_profile)

        rows.append(
            {
                "state": state,
                "label": f"z{state}",
                "occupation": int(len(positions)),
                "mean_dwell": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
                "median_dwell": float(np.median(segment_lengths)) if segment_lengths else 0.0,
                "top_emissions": _top_items(emission_probs, result.observations.vocabulary, top_n_emissions),
                "harmonic_candidates": harmonic_candidates,
                "modal_candidates": modal_candidates,
                "top_successors": ", ".join(
                    f"z{successor} ({successor_probs[successor]:.3f})" for successor in successor_indices
                ),
                "example_positions": str(example_positions),
                "example_offsets": str(example_offsets),
                "tentative_label": tentative_label,
            }
        )
    interpretation_df = pd.DataFrame(rows).sort_values(["occupation", "state"], ascending=[False, True])
    return interpretation_df.reset_index(drop=True)
