from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.observations import ObservationSequence

from .inference import ffbs_sample_batch
from .utils import (
    EPSILON,
    contiguous_segments,
    count_emissions,
    count_transitions,
    dirichlet_logpdf,
    normalize,
    sample_truncated_stick_breaking,
    set_random_seed,
    stick_breaking_from_v,
)

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - cubierto por fallback cuando scipy no esta.
    minimize = None
    SCIPY_AVAILABLE = False


def _count_segmented_transitions(
    state_sequences: list[np.ndarray],
    n_states: int,
) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=int)
    for states in state_sequences:
        counts += count_transitions(states, n_states)
    return counts


def _concatenate_observations(
    observations: list[ObservationSequence],
) -> ObservationSequence:
    if len(observations) == 1:
        return observations[0]
    first = observations[0]
    return ObservationSequence(
        observation_type=first.observation_type,
        tokens=np.concatenate([item.tokens for item in observations]),
        vocabulary=list(first.vocabulary),
        decoded=[decoded for item in observations for decoded in item.decoded],
        events=[event for item in observations for event in item.events],
        extra={"sequence_lengths": [item.size for item in observations]},
    )


@dataclass
class HDPHMMDiagnostics:
    """Historias de inferencia utiles para diagnostico."""

    log_likelihood_history: list[float]
    active_state_history: list[int]
    beta_entropy_history: list[float]
    state_samples: list[np.ndarray]
    beta_update_mode: str
    best_iteration: int

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for iteration, (log_likelihood, active_states, entropy) in enumerate(
            zip(
                self.log_likelihood_history,
                self.active_state_history,
                self.beta_entropy_history,
            ),
            start=1,
        ):
            rows.append(
                {
                    "iteration": iteration,
                    "log_likelihood": log_likelihood,
                    "active_states": active_states,
                    "beta_entropy": entropy,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class HDPHMMResult:
    """Resultado resumido del HDP-HMM truncado."""

    latent_states: np.ndarray
    active_states: list[int]
    beta: np.ndarray
    initial_probs: np.ndarray
    transition_matrix: np.ndarray
    emission_matrix: np.ndarray
    state_usage: np.ndarray
    observations: ObservationSequence
    diagnostics: HDPHMMDiagnostics
    log_likelihood: float
    effective_states: int
    active_state_labels: list[str]
    posterior_transition_mean: np.ndarray
    posterior_emission_mean: np.ndarray
    posterior_beta_mean: np.ndarray
    posterior_initial_mean: np.ndarray

    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": [
                    "log_likelihood",
                    "n_active_states",
                    "K_truncation",
                    "n_observations",
                    "beta_update_mode",
                ],
                "value": [
                    self.log_likelihood,
                    self.effective_states,
                    self.transition_matrix.shape[0],
                    self.observations.size,
                    self.diagnostics.beta_update_mode,
                ],
            }
        )


class TruncatedHDPHMM:
    """Aproximacion weak-limit del iHMM via HDP-HMM truncado."""

    def __init__(
        self,
        n_states: int = 30,
        alpha: float = 8.0,
        alpha0: float = 4.0,
        gamma: float = 2.0,
        eta: float = 1.0,
        kappa: float = 0.0,
        n_iters: int = 200,
        burn_in: int = 100,
        seed: int | None = 7,
        init_active_states: int = 4,
        store_every: int = 1,
    ) -> None:
        self.n_states = int(n_states)
        self.alpha = float(alpha)
        self.alpha0 = float(alpha0)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.n_iters = int(n_iters)
        self.burn_in = int(burn_in)
        self.seed = seed
        self.init_active_states = int(init_active_states)
        self.store_every = int(store_every)
        self.rng = set_random_seed(seed)

    def _initialize_states(self, observations: np.ndarray, vocab_size: int) -> np.ndarray:
        active = min(self.init_active_states, self.n_states, max(2, min(vocab_size, self.n_states)))
        if active <= 0:
            active = min(2, self.n_states)
        state_map = {token: index % active for index, token in enumerate(sorted(set(int(token) for token in observations)))}
        states = np.array([state_map[int(token)] for token in observations], dtype=int)
        noise_mask = self.rng.random(len(states)) < 0.1
        if np.any(noise_mask):
            states[noise_mask] = self.rng.integers(0, active, size=int(np.sum(noise_mask)))
        return states

    def _sample_emissions(self, states: np.ndarray, observations: np.ndarray, vocab_size: int) -> np.ndarray:
        counts = count_emissions(states, observations, self.n_states, vocab_size)
        prior = np.full(vocab_size, self.eta / vocab_size, dtype=float)
        matrix = np.zeros((self.n_states, vocab_size), dtype=float)
        for state in range(self.n_states):
            matrix[state] = self.rng.dirichlet(prior + counts[state])
        return matrix

    def _sample_initial(self, state_sequences: list[np.ndarray], beta: np.ndarray) -> np.ndarray:
        initial_counts = np.bincount(
            [int(states[0]) for states in state_sequences],
            minlength=self.n_states,
        )
        prior = self.alpha0 * beta + EPSILON
        return self.rng.dirichlet(prior + initial_counts)

    def _sample_transitions(self, state_sequences: list[np.ndarray], beta: np.ndarray) -> np.ndarray:
        counts = _count_segmented_transitions(state_sequences, self.n_states)
        matrix = np.zeros((self.n_states, self.n_states), dtype=float)
        for state in range(self.n_states):
            prior = self.alpha * beta + EPSILON
            if self.kappa > 0.0:
                prior[state] += self.kappa
            matrix[state] = self.rng.dirichlet(prior + counts[state])
        return matrix

    def _beta_to_v(self, beta: np.ndarray) -> np.ndarray:
        if beta.size == 1:
            return np.array([], dtype=float)
        remaining = 1.0
        values = []
        for weight in beta[:-1]:
            if remaining <= EPSILON:
                values.append(0.5)
            else:
                values.append(np.clip(weight / remaining, EPSILON, 1.0 - EPSILON))
            remaining -= weight
        return np.array(values, dtype=float)

    def _negative_beta_log_posterior(
        self,
        logits: np.ndarray,
        transition_matrix: np.ndarray,
        initial_probs: np.ndarray,
    ) -> float:
        v = 1.0 / (1.0 + np.exp(-logits))
        beta = stick_breaking_from_v(v)
        total = 0.0
        total -= float(np.sum((self.gamma - 1.0) * np.log(1.0 - v + EPSILON)))
        total -= dirichlet_logpdf(initial_probs, self.alpha0 * beta + EPSILON)
        for state in range(self.n_states):
            prior = self.alpha * beta + EPSILON
            if self.kappa > 0.0:
                prior[state] += self.kappa
            total -= dirichlet_logpdf(transition_matrix[state], prior)
        if not np.isfinite(total):
            return 1e12
        return float(total)

    def _update_beta(
        self,
        beta: np.ndarray,
        transition_matrix: np.ndarray,
        initial_probs: np.ndarray,
        states: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        if self.n_states == 1:
            return np.array([1.0], dtype=float), "degenerate"

        if SCIPY_AVAILABLE:
            initial_v = np.clip(self._beta_to_v(beta), 1e-4, 1.0 - 1e-4)
            initial_logits = np.log(initial_v) - np.log(1.0 - initial_v)
            result = minimize(
                fun=self._negative_beta_log_posterior,
                x0=initial_logits,
                args=(transition_matrix, initial_probs),
                method="L-BFGS-B",
                bounds=[(-6.0, 6.0)] * len(initial_logits),
            )
            if result.success and np.isfinite(result.fun):
                optimized_v = 1.0 / (1.0 + np.exp(-result.x))
                return stick_breaking_from_v(optimized_v), "map_stick_breaking"

        usage = np.bincount(states, minlength=self.n_states).astype(float)
        pseudo = normalize(usage + self.gamma * beta + EPSILON)
        updated = self.rng.dirichlet(pseudo * self.n_states + EPSILON)
        return normalize(updated), "approx_dirichlet_fallback"

    def fit(self, observations: ObservationSequence) -> HDPHMMResult:
        """Fit one observation sequence, preserving the original public API."""

        return self.fit_sequences([observations])

    def fit_sequences(self, observations: list[ObservationSequence]) -> HDPHMMResult:
        """Ejecuta blocked Gibbs sampling sobre el HDP-HMM truncado."""

        if not observations:
            raise ValueError("Se requiere al menos una secuencia de observaciones.")
        if any(item.size == 0 for item in observations):
            raise ValueError("Las secuencias de observaciones no pueden estar vacias.")
        if sum(item.size for item in observations) < 2:
            raise ValueError("Se requieren al menos dos observaciones para ajustar el HDP-HMM.")
        first = observations[0]
        if any(item.vocabulary != first.vocabulary for item in observations[1:]):
            raise ValueError("Todas las secuencias deben compartir el mismo vocabulario.")

        sequences = [item.tokens.astype(int) for item in observations]
        sequence = np.concatenate(sequences)
        combined_observations = _concatenate_observations(observations)
        vocab_size = first.vocab_size
        _, beta = sample_truncated_stick_breaking(self.gamma, self.n_states, self.rng)
        state_sequences = [self._initialize_states(item, vocab_size) for item in sequences]
        states = np.concatenate(state_sequences)
        transition_matrix = self._sample_transitions(state_sequences, beta)
        emission_matrix = self._sample_emissions(states, sequence, vocab_size)
        initial_probs = self._sample_initial(state_sequences, beta)

        log_likelihood_history: list[float] = []
        active_state_history: list[int] = []
        beta_entropy_history: list[float] = []
        state_samples: list[np.ndarray] = []
        beta_update_mode = "prior_only"

        posterior_transition_sum = np.zeros_like(transition_matrix)
        posterior_emission_sum = np.zeros_like(emission_matrix)
        posterior_beta_sum = np.zeros_like(beta)
        posterior_initial_sum = np.zeros_like(initial_probs)
        posterior_samples = 0

        best_log_likelihood = -np.inf
        best_states = states.copy()
        best_transition = transition_matrix.copy()
        best_emission = emission_matrix.copy()
        best_initial = initial_probs.copy()
        best_beta = beta.copy()
        best_iteration = 0

        for iteration in range(self.n_iters):
            states = np.concatenate(state_sequences)
            emission_matrix = self._sample_emissions(states, sequence, vocab_size)
            transition_matrix = self._sample_transitions(state_sequences, beta)
            initial_probs = self._sample_initial(state_sequences, beta)
            beta, beta_update_mode = self._update_beta(beta, transition_matrix, initial_probs, states)

            state_sequences, sequence_log_likelihoods = ffbs_sample_batch(
                initial_probs=initial_probs,
                transition_matrix=transition_matrix,
                emission_matrix=emission_matrix,
                sequences=sequences,
                rng=self.rng,
            )
            states = np.concatenate(state_sequences)
            log_likelihood = float(sequence_log_likelihoods.sum())

            active_states = int(np.unique(states).size)
            beta_entropy = -float(np.sum(beta * np.log(beta + EPSILON)))

            log_likelihood_history.append(log_likelihood)
            active_state_history.append(active_states)
            beta_entropy_history.append(beta_entropy)

            if iteration >= self.burn_in and ((iteration - self.burn_in) % self.store_every == 0):
                posterior_transition_sum += transition_matrix
                posterior_emission_sum += emission_matrix
                posterior_beta_sum += beta
                posterior_initial_sum += initial_probs
                posterior_samples += 1
                state_samples.append(states.copy())

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_states = states.copy()
                best_transition = transition_matrix.copy()
                best_emission = emission_matrix.copy()
                best_initial = initial_probs.copy()
                best_beta = beta.copy()
                best_iteration = iteration + 1

        if posterior_samples == 0:
            posterior_transition_mean = best_transition.copy()
            posterior_emission_mean = best_emission.copy()
            posterior_beta_mean = best_beta.copy()
            posterior_initial_mean = best_initial.copy()
        else:
            posterior_transition_mean = posterior_transition_sum / posterior_samples
            posterior_emission_mean = posterior_emission_sum / posterior_samples
            posterior_beta_mean = posterior_beta_sum / posterior_samples
            posterior_initial_mean = posterior_initial_sum / posterior_samples

        state_usage = np.bincount(best_states, minlength=self.n_states)
        active_state_indices = sorted(int(state) for state in np.unique(best_states))
        diagnostics = HDPHMMDiagnostics(
            log_likelihood_history=log_likelihood_history,
            active_state_history=active_state_history,
            beta_entropy_history=beta_entropy_history,
            state_samples=state_samples,
            beta_update_mode=beta_update_mode,
            best_iteration=best_iteration,
        )
        return HDPHMMResult(
            latent_states=best_states,
            active_states=active_state_indices,
            beta=best_beta,
            initial_probs=best_initial,
            transition_matrix=best_transition,
            emission_matrix=best_emission,
            state_usage=state_usage,
            observations=combined_observations,
            diagnostics=diagnostics,
            log_likelihood=best_log_likelihood,
            effective_states=len(active_state_indices),
            active_state_labels=[f"z{state}" for state in active_state_indices],
            posterior_transition_mean=posterior_transition_mean,
            posterior_emission_mean=posterior_emission_mean,
            posterior_beta_mean=posterior_beta_mean,
            posterior_initial_mean=posterior_initial_mean,
        )

    @staticmethod
    def dwell_times(states: np.ndarray) -> dict[int, list[int]]:
        dwell: dict[int, list[int]] = {}
        for state, start, end in contiguous_segments(states):
            dwell.setdefault(state, []).append(end - start)
        return dwell
