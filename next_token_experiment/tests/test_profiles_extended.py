from __future__ import annotations

import unittest

from next_token_experiment.profiles import build_profile_config


class ProfileExtensionTests(unittest.TestCase):
    def test_research_richer_events_profile_switches_to_research_track(self) -> None:
        config = build_profile_config("research_richer_events")

        self.assertEqual(config.representation.primary, "event_pitch_duration_metrical")
        self.assertTrue(config.preprocessing.include_rests)
        self.assertEqual(config.windows.max_context_length, 256)
        self.assertEqual(config.transformer.n_layers, 6)
        self.assertEqual(config.transformer.d_model, 256)
        self.assertTrue(config.transformer.use_relative_position_bias)
        self.assertIn("track:research", config.notes)


if __name__ == "__main__":
    unittest.main()
