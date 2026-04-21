from __future__ import annotations

import unittest

from next_token_experiment.config import RepresentationConfig
from next_token_experiment.data.preprocess import build_representation_tokens, build_representation_vocabulary
from next_token_experiment.data.validation import validate_representation_config
from src.data.parsing import MusicalEvent


def make_event(
    *,
    pitch_class: int | None,
    duration: float,
    beat: float | None,
    kind: str = "note",
    label: str = "event",
) -> MusicalEvent:
    return MusicalEvent(
        index=0,
        offset=0.0,
        duration=duration,
        kind=kind,
        label=label,
        pitch_classes=tuple() if pitch_class is None else (pitch_class,),
        midi_pitches=tuple(),
        representative_pitch_class=pitch_class,
        representative_midi=None,
        measure=1,
        beat=beat,
    )


class RepresentationTests(unittest.TestCase):
    def test_event_pitch_duration_metrical_vocabulary_size_matches_cartesian_product(self) -> None:
        config = RepresentationConfig()
        vocabulary = build_representation_vocabulary(config, "event_pitch_duration_metrical")
        expected_size = (12 + 1) * len(config.duration_bins) * len(config.metrical_levels)
        self.assertEqual(len(vocabulary), expected_size)

    def test_event_pitch_duration_metrical_encodes_pitch_and_rest_events(self) -> None:
        config = RepresentationConfig()
        events = [
            make_event(pitch_class=0, duration=1.0, beat=1.0, kind="note", label="C4"),
            make_event(pitch_class=None, duration=0.5, beat=1.5, kind="rest", label="Rest"),
            make_event(pitch_class=7, duration=0.25, beat=1.25, kind="note", label="G4"),
        ]

        tokens, vocabulary = build_representation_tokens(events, config, "event_pitch_duration_metrical")
        decoded = [vocabulary[token] for token in tokens]

        self.assertEqual(len(tokens), 3)
        self.assertIn("C|dur=1|metric=downbeat", decoded)
        self.assertIn("REST|dur=0.5|metric=offbeat", decoded)
        self.assertIn("G|dur=0.25|metric=subbeat", decoded)

    def test_representation_config_rejects_duplicate_metrical_levels(self) -> None:
        config = RepresentationConfig(metrical_levels=("downbeat", "beat", "beat"))
        issues = validate_representation_config(config)
        self.assertIn("Metrical levels must be unique.", issues)


if __name__ == "__main__":
    unittest.main()
