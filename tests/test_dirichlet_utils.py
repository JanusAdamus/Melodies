"""Dirichlet sin ceros silenciosos (hallazgo H1 de la auditoría)."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.utils import dirichlet_logpdf, sample_dirichlet


class DirichletTests(unittest.TestCase):
    def test_logpdf_rejects_zeros_instead_of_returning_minus_infinity(self) -> None:
        with self.assertRaises(ValueError):
            dirichlet_logpdf(np.array([0.0, 1.0]), np.array([0.5, 0.5]))

    def test_logpdf_matches_scipy_inside_the_simplex(self) -> None:
        from scipy.stats import dirichlet

        x = np.array([0.2, 0.3, 0.5])
        alpha = np.array([0.4, 2.0, 3.5])
        self.assertAlmostEqual(dirichlet_logpdf(x, alpha), dirichlet.logpdf(x, alpha), places=10)

    def test_sampler_has_no_exact_zeros_with_tiny_concentrations(self) -> None:
        rng = np.random.default_rng(0)
        alpha = np.full((200, 20), 1e-3)
        samples = sample_dirichlet(alpha, rng)
        self.assertTrue(np.all(samples > 0.0))
        np.testing.assert_allclose(samples.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
        # Numpy produce ceros exactos con estas concentraciones; ése era el origen de H1.
        self.assertTrue(np.any(rng.dirichlet(alpha[0], size=200) == 0.0))

    def test_sampler_has_the_dirichlet_mean_and_variance(self) -> None:
        rng = np.random.default_rng(1)
        alpha = np.array([0.3, 1.0, 4.0])
        samples = sample_dirichlet(np.tile(alpha, (40_000, 1)), rng)
        total = alpha.sum()
        mean = alpha / total
        variance = mean * (1 - mean) / (total + 1)
        standard_error = np.sqrt(variance / samples.shape[0])
        self.assertTrue(np.all(np.abs(samples.mean(axis=0) - mean) < 4 * standard_error))
        np.testing.assert_allclose(samples.var(axis=0), variance, rtol=0.05)

    def test_sampler_rejects_non_positive_concentrations(self) -> None:
        with self.assertRaises(ValueError):
            sample_dirichlet(np.array([1.0, 0.0]), np.random.default_rng(0))



if __name__ == "__main__":
    unittest.main()
