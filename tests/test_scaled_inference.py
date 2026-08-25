"""Regression guard for the rescaled, batched HMM inner loops.

Both fast paths must reproduce the log-domain formulation they replaced. The
reference implementations below are the code that used to run in production; keep
them here so the test owns its own oracle.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from Comparacion.classical_models import _expectation_batch, _pad_sequences
from src.models.inference import (
    ffbs_sample_batch,
    forward_log_likelihood,
    scaled_forward_log_likelihood,
)
from src.models.utils import EPSILON, count_emissions, count_transitions

# Ragged on purpose: padding and masking is where a batched rewrite breaks.
LENGTHS = (1, 2, 3, 17, 263, 60, 300, 9)

_MINIMAL_SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1"><measure number="1">
    <attributes><divisions>1</divisions>
      <key><fifths>0</fifths></key>
      <time><beats>4</beats><beat-type>4</beat-type></time>
      <clef><sign>G</sign><line>2</line></clef>
    </attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
  </measure></part>
</score-partwise>
"""


def _parameters(n_states: int, vocab_size: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.dirichlet(np.ones(n_states)),
        rng.dirichlet(np.ones(n_states), size=n_states),
        rng.dirichlet(np.ones(vocab_size), size=n_states),
    )


def _sequences(vocab_size: int, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab_size, size=n) for n in LENGTHS]


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - max_values), axis=axis, keepdims=True)
    return np.squeeze(max_values + np.log(np.maximum(summed, EPSILON)), axis=axis)


def _reference_e_step(initial, transition, emission, sequences, n_states, vocab_size):
    """The per-sequence log-domain E-step that `_expectation_batch` replaced."""

    initial_counts = np.zeros(n_states)
    transition_counts = np.zeros((n_states, n_states))
    emission_counts = np.zeros((n_states, vocab_size))
    total = 0.0
    log_transition = np.log(transition + EPSILON)
    log_emission = np.log(emission + EPSILON)
    log_initial = np.log(initial + EPSILON)

    for observations in sequences:
        emission_log_probs = log_emission[:, observations]
        alpha = np.full((len(observations), n_states), -np.inf)
        alpha[0] = log_initial + emission_log_probs[:, 0]
        for step in range(1, len(observations)):
            alpha[step] = emission_log_probs[:, step] + _logsumexp(
                alpha[step - 1][:, None] + log_transition, axis=0
            )
        log_likelihood = float(_logsumexp(alpha[-1], axis=0))

        beta = np.zeros_like(alpha)
        for step in range(len(observations) - 2, -1, -1):
            beta[step] = _logsumexp(
                log_transition
                + emission_log_probs[:, step + 1][None, :]
                + beta[step + 1][None, :],
                axis=1,
            )

        gamma = np.exp(alpha + beta - log_likelihood)
        total += log_likelihood
        initial_counts += gamma[0]
        for step, token in enumerate(observations):
            emission_counts[:, int(token)] += gamma[step]
        for step in range(len(observations) - 1):
            transition_counts += np.exp(
                alpha[step][:, None]
                + log_transition
                + emission_log_probs[:, step + 1][None, :]
                + beta[step + 1][None, :]
                - log_likelihood
            )
    return initial_counts, transition_counts, emission_counts, total


class ScaledInferenceRegressionTests(unittest.TestCase):
    def test_batched_e_step_matches_log_domain(self) -> None:
        n_states, vocab_size = 24, 13
        initial, transition, emission = _parameters(n_states, vocab_size)
        sequences = _sequences(vocab_size)

        expected = _reference_e_step(
            initial, transition, emission, sequences, n_states, vocab_size
        )

        actual = [
            np.zeros(n_states),
            np.zeros((n_states, n_states)),
            np.zeros((n_states, vocab_size)),
            0.0,
        ]
        for padded, lengths in _pad_sequences(sequences):
            parts = _expectation_batch(initial, transition, emission, padded, lengths)
            for index in range(3):
                actual[index] = actual[index] + parts[index]
            actual[3] += parts[3]

        for name, want, got in zip(
            ("initial", "transition", "emission", "log_likelihood"), expected, actual
        ):
            want, got = np.asarray(want, dtype=float), np.asarray(got, dtype=float)
            scale = max(float(np.max(np.abs(want))), 1.0)
            self.assertLess(
                np.max(np.abs(want - got)) / scale,
                1e-8,
                f"{name} drifted",
            )

    def test_batched_ffbs_log_likelihood_matches_forward(self) -> None:
        n_states, vocab_size = 40, 13
        initial, transition, emission = _parameters(n_states, vocab_size)
        sequences = _sequences(vocab_size)

        expected = np.array(
            [
                forward_log_likelihood(initial, transition, emission, item)[0]
                for item in sequences
            ]
        )
        states, actual = ffbs_sample_batch(
            initial, transition, emission, sequences, np.random.default_rng(7)
        )

        self.assertTrue(np.allclose(expected, actual, atol=1e-6))
        self.assertEqual(
            [item.size for item in states],
            [item.size for item in sequences],
        )
        self.assertTrue(
            all(item.min() >= 0 and item.max() < n_states for item in states)
        )

    def test_scaled_forward_matches_log_domain(self) -> None:
        n_states, vocab_size = 40, 13
        initial, transition, emission = _parameters(n_states, vocab_size)
        for observations in _sequences(vocab_size):
            expected, _ = forward_log_likelihood(
                initial, transition, emission, observations
            )
            actual = scaled_forward_log_likelihood(
                initial, transition, emission, observations
            )
            self.assertLess(abs(expected - actual), 1e-6)

    def test_batched_ffbs_sampler_follows_the_emissions(self) -> None:
        """A near-deterministic emission matrix pins each state to its own token."""

        n_states = 3
        initial = np.full(n_states, 1 / n_states)
        transition = np.full((n_states, n_states), 1 / n_states)
        emission = np.full((n_states, n_states), 1e-9)
        np.fill_diagonal(emission, 1.0)
        emission /= emission.sum(axis=1, keepdims=True)

        observations = [np.array([0, 1, 2, 1, 0, 2, 2, 0])]
        states, _ = ffbs_sample_batch(
            initial, transition, emission, observations, np.random.default_rng(3)
        )
        self.assertTrue(np.array_equal(states[0], observations[0]))

    def test_counting_helpers_match_the_naive_loops(self) -> None:
        for n_states, vocab_size in ((4, 5), (12, 13)):
            with self.subTest(n_states=n_states, vocab_size=vocab_size):
                rng = np.random.default_rng(2)
                states = rng.integers(0, n_states, size=500)
                observations = rng.integers(0, vocab_size, size=500)

                expected_transitions = np.zeros((n_states, n_states), dtype=int)
                for left, right in zip(states[:-1], states[1:]):
                    expected_transitions[left, right] += 1
                self.assertTrue(
                    np.array_equal(
                        count_transitions(states, n_states), expected_transitions
                    )
                )

                expected_emissions = np.zeros((n_states, vocab_size), dtype=int)
                for state, observation in zip(states, observations):
                    expected_emissions[state, observation] += 1
                self.assertTrue(
                    np.array_equal(
                        count_emissions(
                            states, observations, n_states, vocab_size
                        ),
                        expected_emissions,
                    )
                )

    def test_corpus_cache_respects_the_current_selection(self) -> None:
        """A cache holding more scores than `max_files` must not drag the extras in."""

        from dataclasses import replace

        from next_token_experiment.data.preprocess import prepare_corpus
        from next_token_experiment.protocol import build_default_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            corpus = tmp_path / "scores"
            corpus.mkdir()
            for index in range(6):
                (corpus / f"{index}.musicxml").write_text(
                    _MINIMAL_SCORE, encoding="utf-8"
                )

            config = build_default_experiment_config()
            config = replace(
                config,
                corpus=replace(
                    config.corpus,
                    root_dir=str(corpus),
                    min_events_per_piece=1,
                ),
            )
            cache = tmp_path / "cache.jsonl"

            everything = prepare_corpus(
                config, n_workers=1, cache_path=cache, progress_every=0
            )
            self.assertEqual(
                len(everything.pieces) + len(everything.exclusions), 6
            )

            # Same cache, smaller selection: the cached extras must stay out.
            subset = prepare_corpus(
                config,
                max_files=2,
                n_workers=1,
                cache_path=cache,
                progress_every=0,
            )
            self.assertEqual(len(subset.pieces) + len(subset.exclusions), 2)

    def test_count_transitions_handles_short_input(self) -> None:
        self.assertTrue(
            np.array_equal(
                count_transitions(np.array([2]), 4),
                np.zeros((4, 4), dtype=int),
            )
        )


if __name__ == "__main__":
    unittest.main()
