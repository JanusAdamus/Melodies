from __future__ import annotations

import unittest

import numpy as np

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


class HDPHMMTests(unittest.TestCase):
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
