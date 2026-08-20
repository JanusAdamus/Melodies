from __future__ import annotations

import unittest

import numpy as np

from src.models import hdp_hmm as hdp_hmm_module
from src.data.observations import ObservationSequence
from src.data.parsing import MusicalEvent
from src.models.hdp_hmm import TruncatedHDPHMM


def dummy_events(length: int) -> list[MusicalEvent]:
    events = []
    for index in range(length):
        events.append(
            MusicalEvent(
                index=index,
                offset=float(index),
                duration=1.0,
                kind="note",
                label=f"N{index}",
                pitch_classes=(index % 12,),
                midi_pitches=(60 + index % 12,),
                representative_pitch_class=index % 12,
                representative_midi=60 + index % 12,
                measure=1,
                beat=1.0,
            )
        )
    return events


def _observations(tokens: list[int]) -> ObservationSequence:
    vocabulary = ["A", "B", "C"]
    return ObservationSequence(
        observation_type="symbolic_test",
        tokens=np.array(tokens, dtype=int),
        vocabulary=vocabulary,
        decoded=[vocabulary[token] for token in tokens],
        events=dummy_events(len(tokens)),
        extra={},
    )


class HDPHMMTests(unittest.TestCase):
    def test_segmented_transition_counts_exclude_sequence_boundaries(self) -> None:
        self.assertTrue(
            hasattr(hdp_hmm_module, "_count_segmented_transitions"),
            "segmented transition counting helper must exist",
        )
        counts = hdp_hmm_module._count_segmented_transitions(
            [np.array([0, 1]), np.array([2, 0])],
            n_states=3,
        )

        np.testing.assert_array_equal(
            counts,
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                ]
            ),
        )
        self.assertEqual(counts[1, 2], 0)

    def test_fit_sequences_counts_each_sequence_start(self) -> None:
        model = TruncatedHDPHMM(
            n_states=3,
            n_iters=1,
            burn_in=0,
            seed=11,
            init_active_states=3,
        )
        self.assertTrue(
            hasattr(model, "fit_sequences"),
            "fit_sequences must fit independent observation sequences",
        )
        initial_contributions_seen: list[np.ndarray] = []
        base_rng = model.rng

        class RecordingRNG:
            record_dirichlet = False
            last_concentration: np.ndarray | None = None

            def __getattr__(self, name):
                return getattr(base_rng, name)

            def dirichlet(self, concentration, *args, **kwargs):
                if self.record_dirichlet:
                    self.last_concentration = np.array(concentration, copy=True)
                return base_rng.dirichlet(concentration, *args, **kwargs)

        recording_rng = RecordingRNG()
        model.rng = recording_rng
        original_sample_initial = model._sample_initial

        def recording_sample_initial(state_sequences, beta):
            recording_rng.record_dirichlet = True
            try:
                sampled = original_sample_initial(state_sequences, beta)
            finally:
                recording_rng.record_dirichlet = False
            initial_contributions_seen.append(
                recording_rng.last_concentration - model.alpha0 * beta - hdp_hmm_module.EPSILON
            )
            return sampled

        model._sample_initial = recording_sample_initial
        result = model.fit_sequences([_observations([0, 1]), _observations([2, 0])])

        self.assertEqual(int(np.count_nonzero(initial_contributions_seen[0] > 0.5)), 2)
        self.assertTrue(
            all(np.isclose(contributions.sum(), 2.0) for contributions in initial_contributions_seen)
        )
        self.assertEqual(result.observations.size, 4)
        self.assertEqual(len(result.latent_states), 4)

    def test_truncated_hdp_hmm_uses_no_more_than_K_states(self) -> None:
        tokens = np.array([0, 0, 1, 1, 2, 2, 0, 0, 3, 3, 4, 4], dtype=int)
        vocabulary = ["A", "B", "C", "D", "E"]
        observations = ObservationSequence(
            observation_type="symbolic_test",
            tokens=tokens,
            vocabulary=vocabulary,
            decoded=[vocabulary[token] for token in tokens],
            events=dummy_events(len(tokens)),
            extra={},
        )

        model = TruncatedHDPHMM(n_states=8, n_iters=20, burn_in=10, seed=11)
        result = model.fit(observations)

        self.assertLessEqual(result.effective_states, 8)
        self.assertLessEqual(max(result.active_states), 7)
        self.assertTrue(np.allclose(result.transition_matrix.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(result.emission_matrix.sum(axis=1), 1.0))
        self.assertEqual(len(result.diagnostics.log_likelihood_history), 20)


if __name__ == "__main__":
    unittest.main()
