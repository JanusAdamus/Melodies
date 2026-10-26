from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.observations import ObservationSequence

from .inference import ffbs_sample_batch
from .utils import (
    EPSILON,
    contiguous_segments,
    count_emissions,
    count_transitions,
    sample_dirichlet,
    set_random_seed,
)

#: Valor de `beta_update_mode`: β | m ~ Dir(γ/K + Σ_j m̄_jk) con mesas m por CRT (Fox et al., 2011).
BETA_UPDATE_MODE = "crt_auxiliary_gibbs"


def _sample_crt(counts: np.ndarray, concentration: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mesas de un restaurante chino: m = Σ_{i<n} Bernoulli(c / (c + i)), celda a celda."""

    counts = np.asarray(counts, dtype=int)
    concentration = np.broadcast_to(np.asarray(concentration, dtype=float), counts.shape)
    flat_counts = counts.ravel()
    occupied = np.flatnonzero(flat_counts)
    tables = np.zeros(flat_counts.size, dtype=int)
    if occupied.size == 0:
        return tables.reshape(counts.shape)
    customers = flat_counts[occupied]
    cell = np.repeat(np.arange(occupied.size), customers)
    position = np.arange(cell.size) - np.repeat(np.cumsum(customers) - customers, customers)
    rate = concentration.ravel()[occupied][cell]
    # El primer cliente siempre abre mesa; escribirlo aparte evita 0/0 si c subdesborda.
    probability = np.where(position == 0, 1.0, rate / (rate + np.maximum(position, 1)))
    opened = rng.random(cell.size) < probability
    tables[occupied] = np.bincount(cell, weights=opened, minlength=occupied.size).astype(int)
    return tables.reshape(counts.shape)


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
    #: Hiperparámetros de la cadena; van con la traza para que sea auditable.
    hyperparameters: dict[str, float] = field(default_factory=dict)
    #: β del primer sorteo (prior); sirve para auditar que el resultado no lo hereda.
    initial_beta: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def trace_rows(self) -> list[dict[str, float | int | str]]:
        """Traza por iteración, con los hiperparámetros que la produjeron."""

        rows: list[dict[str, float | int | str]] = []
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
                    "log_likelihood": float(log_likelihood),
                    "active_states": int(active_states),
                    "beta_entropy": float(entropy),
                    "beta_update_mode": self.beta_update_mode,
                    **{str(key): value for key, value in self.hyperparameters.items()},
                }
            )
        return rows

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
        if min(self.alpha, self.alpha0, self.gamma, self.eta) <= 0.0 or self.kappa < 0.0:
            raise ValueError("alpha, alpha0, gamma y eta deben ser > 0; kappa >= 0.")
        if self.n_states < 1 or self.store_every < 1 or not 0 <= self.burn_in < self.n_iters:
            raise ValueError("Se requiere n_states >= 1, store_every >= 1 y 0 <= burn_in < n_iters.")
        # El generador se crea en cada ajuste a partir de `seed` (hallazgo H8).
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
        return sample_dirichlet(self.eta / vocab_size + counts, self.rng)

    def _sample_initial(self, state_sequences: list[np.ndarray], beta: np.ndarray) -> np.ndarray:
        return sample_dirichlet(self.alpha0 * beta + self._initial_counts(state_sequences), self.rng)

    def _sample_transitions(self, state_sequences: list[np.ndarray], beta: np.ndarray) -> np.ndarray:
        counts = _count_segmented_transitions(state_sequences, self.n_states)
        return sample_dirichlet(self._transition_concentration(beta) + counts, self.rng)

    def _initial_counts(self, state_sequences: list[np.ndarray]) -> np.ndarray:
        return np.bincount([int(states[0]) for states in state_sequences], minlength=self.n_states)

    def _transition_concentration(self, beta: np.ndarray) -> np.ndarray:
        """Concentración de la fila j: alpha*beta + kappa*e_j."""

        return self.alpha * beta[None, :] + self.kappa * np.eye(self.n_states)

    def _update_beta(self, beta: np.ndarray, state_sequences: list[np.ndarray]) -> tuple[np.ndarray, str]:
        """β | z por variables auxiliares (Fox et al., 2011, weak limit), con π integrado.

        m_jk ~ CRT(n_jk, alpha*beta_k + kappa*[j=k]); si kappa > 0, w_j ~ Bin(m_jj,
        rho / (rho + beta_j (1 - rho))) con rho = kappa / (alpha + kappa) y m̄_jj = m_jj - w_j.
        La distribución inicial aporta sus propias mesas con concentración alpha0*beta.
        Finalmente beta ~ Dir(gamma/K + Σ_j m̄_jk).
        """

        if self.n_states == 1:
            return np.array([1.0], dtype=float), "degenerate"
        counts = _count_segmented_transitions(state_sequences, self.n_states)
        tables = _sample_crt(counts, self._transition_concentration(beta), self.rng)
        if self.kappa > 0.0:
            rho = self.kappa / (self.alpha + self.kappa)
            diagonal = np.diag(tables)
            override = self.rng.binomial(diagonal, rho / (rho + beta * (1.0 - rho)))
            tables = tables - np.diag(override)
        initial_tables = _sample_crt(self._initial_counts(state_sequences), self.alpha0 * beta, self.rng)
        concentration = self.gamma / self.n_states + tables.sum(axis=0) + initial_tables
        return sample_dirichlet(concentration, self.rng), BETA_UPDATE_MODE

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
        self.rng = set_random_seed(self.seed)
        beta = sample_dirichlet(np.full(self.n_states, self.gamma / self.n_states), self.rng)
        initial_beta = beta.copy()
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
            beta, beta_update_mode = self._update_beta(beta, state_sequences)
            emission_matrix = self._sample_emissions(states, sequence, vocab_size)
            transition_matrix = self._sample_transitions(state_sequences, beta)
            initial_probs = self._sample_initial(state_sequences, beta)

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
            hyperparameters={
                "alpha": float(self.alpha),
                "alpha0": float(self.alpha0),
                "gamma": float(self.gamma),
                "kappa": float(self.kappa),
                "eta": float(self.eta),
                "n_states_truncation": int(self.n_states),
                "burn_in": int(self.burn_in),
            },
            initial_beta=initial_beta,
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
