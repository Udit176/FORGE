"""
Tests for the synthetic teacher module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from humans.synthetic_teacher import (
    SyntheticTeacher, DemonstratorParams, DemonstratorPopulation,
    DemonstratorPopulationConfig, compute_dtw_lag
)


class TestDemonstratorParams(unittest.TestCase):
    """Tests for DemonstratorParams class."""

    def test_default_params(self):
        params = DemonstratorParams()
        self.assertEqual(params.tau, 0.0)
        self.assertEqual(params.g, 1.0)
        self.assertEqual(params.sigma, 0.0)

    def test_to_array(self):
        params = DemonstratorParams(tau=1.0, g=0.8, sigma=0.1, delta=0.05, sat=0.9, dropout=0.02)
        arr = params.to_array()
        self.assertEqual(arr.shape, (6,))
        self.assertAlmostEqual(arr[0], 1.0)
        self.assertAlmostEqual(arr[1], 0.8)

    def test_to_dict(self):
        params = DemonstratorParams(tau=2.0)
        d = params.to_dict()
        self.assertEqual(d['tau'], 2.0)


class TestSyntheticTeacher(unittest.TestCase):
    """Tests for SyntheticTeacher class."""

    def test_identity_transform(self):
        """Test that neutral params return (near) oracle labels."""
        params = DemonstratorParams(tau=0, g=1.0, sigma=0, delta=0, sat=1.0, dropout=0)
        teacher = SyntheticTeacher(params, u_max=1.0, seed=42)

        oracle_actions = np.array([0.5, 0.3, -0.2, 0.1, 0.0])
        human_actions = teacher.generate_episode_labels(oracle_actions)

        np.testing.assert_array_almost_equal(human_actions, oracle_actions, decimal=5)

    def test_gain_scaling(self):
        """Test that gain scales actions correctly."""
        params = DemonstratorParams(tau=0, g=0.5, sigma=0, delta=0, sat=1.0, dropout=0)
        teacher = SyntheticTeacher(params, u_max=1.0, seed=42)

        oracle_actions = np.array([1.0, 0.5, -0.5])
        human_actions = teacher.generate_episode_labels(oracle_actions)

        # Actions should be scaled by gain
        expected = np.array([0.5, 0.25, -0.25])
        np.testing.assert_array_almost_equal(human_actions, expected, decimal=5)

    def test_deadzone(self):
        """Test that deadzone suppresses small actions."""
        params = DemonstratorParams(tau=0, g=1.0, sigma=0, delta=0.2, sat=1.0, dropout=0)
        teacher = SyntheticTeacher(params, u_max=1.0, seed=42)

        oracle_actions = np.array([0.5, 0.1, -0.05, 0.3])
        human_actions = teacher.generate_episode_labels(oracle_actions)

        # Small actions (< 0.2) should be zero
        self.assertEqual(human_actions[1], 0.0)
        self.assertEqual(human_actions[2], 0.0)
        # Larger actions should pass through
        self.assertNotEqual(human_actions[0], 0.0)
        self.assertNotEqual(human_actions[3], 0.0)

    def test_saturation(self):
        """Test saturation clipping."""
        params = DemonstratorParams(tau=0, g=1.0, sigma=0, delta=0, sat=0.5, dropout=0)
        teacher = SyntheticTeacher(params, u_max=1.0, seed=42)

        oracle_actions = np.array([0.8, -0.9, 0.3])
        human_actions = teacher.generate_episode_labels(oracle_actions)

        # Actions should be clipped to [-0.5, 0.5]
        self.assertLessEqual(max(human_actions), 0.5)
        self.assertGreaterEqual(min(human_actions), -0.5)

    def test_deterministic_with_seed(self):
        """Test reproducibility with same seed."""
        params = DemonstratorParams(tau=0, g=1.0, sigma=0.1, delta=0, sat=1.0, dropout=0)

        teacher1 = SyntheticTeacher(params, u_max=1.0, seed=42)
        teacher2 = SyntheticTeacher(params, u_max=1.0, seed=42)

        oracle_actions = np.array([0.5, 0.3, -0.2, 0.1, 0.4])

        human1 = teacher1.generate_episode_labels(oracle_actions)
        human2 = teacher2.generate_episode_labels(oracle_actions)

        np.testing.assert_array_equal(human1, human2)

    def test_noise_adds_variability(self):
        """Test that noise adds variability."""
        params = DemonstratorParams(tau=0, g=1.0, sigma=0.2, delta=0, sat=1.0, dropout=0)
        teacher = SyntheticTeacher(params, u_max=1.0, seed=42)

        oracle_actions = np.array([0.5] * 20)
        human_actions = teacher.generate_episode_labels(oracle_actions)

        # With noise, actions should vary
        std = np.std(human_actions)
        self.assertGreater(std, 0.05)


class TestDemonstratorPopulation(unittest.TestCase):
    """Tests for DemonstratorPopulation class."""

    def test_population_size(self):
        config = DemonstratorPopulationConfig(n_demonstrators=10)
        pop = DemonstratorPopulation(config, u_max=1.0, seed=42)

        self.assertEqual(len(pop), 10)
        self.assertEqual(len(pop.demonstrators), 10)

    def test_parameter_bounds(self):
        config = DemonstratorPopulationConfig(
            n_demonstrators=50,
            tau_range=(-3.0, 3.0),
            g_range=(0.7, 1.3)
        )
        pop = DemonstratorPopulation(config, u_max=1.0, seed=42)

        params_array = pop.get_all_params_array()

        # Check tau bounds
        self.assertGreaterEqual(params_array[:, 0].min(), -3.0)
        self.assertLessEqual(params_array[:, 0].max(), 3.0)

        # Check g bounds
        self.assertGreaterEqual(params_array[:, 1].min(), 0.7)
        self.assertLessEqual(params_array[:, 1].max(), 1.3)

    def test_deterministic_population(self):
        config = DemonstratorPopulationConfig(n_demonstrators=5)

        pop1 = DemonstratorPopulation(config, u_max=1.0, seed=42)
        pop2 = DemonstratorPopulation(config, u_max=1.0, seed=42)

        params1 = pop1.get_all_params_array()
        params2 = pop2.get_all_params_array()

        np.testing.assert_array_equal(params1, params2)


class TestDTWLag(unittest.TestCase):
    """Tests for DTW lag computation."""

    def test_zero_lag(self):
        """Test that identical sequences have zero lag."""
        seq = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
        lag = compute_dtw_lag(seq, seq)
        self.assertAlmostEqual(lag, 0.0, places=1)

    def test_positive_lag(self):
        """Test detection of positive lag (human behind)."""
        oracle = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
        human = np.array([0.0, 0.0, 0.0, 0.5, 0.5])  # Shifted right by 1

        lag = compute_dtw_lag(human, oracle, window=5)
        # Human is behind, so lag should be positive
        self.assertGreater(lag, 0)


if __name__ == '__main__':
    unittest.main()
