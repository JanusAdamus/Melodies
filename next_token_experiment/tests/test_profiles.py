from __future__ import annotations

import unittest

from next_token_experiment.profiles import build_profile_config, profile_requires_scope_validation
from next_token_experiment.protocol import validate_experiment_scope


class ProfileTests(unittest.TestCase):
    def test_cpu_baseline_profile_stays_within_bounded_scope(self) -> None:
        config = build_profile_config("cpu_baseline")
        self.assertTrue(profile_requires_scope_validation("cpu_baseline"))
        self.assertEqual(validate_experiment_scope(config), [])
        self.assertEqual(config.hardware.target_device, "cpu")
        self.assertFalse(config.transformer.use_relative_position_bias)
        self.assertEqual(config.storage.results_root, "artifacts/next_token_experiment/results")

    def test_gpu_profile_requests_cuda_and_is_marked_as_extension(self) -> None:
        config = build_profile_config("gpu_extended")
        self.assertFalse(profile_requires_scope_validation("gpu_extended"))
        self.assertEqual(config.hardware.target_device, "cuda")
        self.assertTrue(config.hardware.gpu_required)
        self.assertEqual(config.hardware.precision, "bf16")
        self.assertEqual(config.transformer.attention_implementation, "eager")


if __name__ == "__main__":
    unittest.main()
