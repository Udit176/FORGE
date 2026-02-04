"""
Tests for the driving environment module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.driving_env import DrivingEnv, EnvConfig, MapConfig, MapGenerator, Obstacle


class TestObstacle(unittest.TestCase):
    """Tests for Obstacle class."""

    def test_contains_point_inside(self):
        obs = Obstacle(0, 0, 2, 2)
        self.assertTrue(obs.contains_point(1, 1))
        self.assertTrue(obs.contains_point(0.5, 0.5))

    def test_contains_point_outside(self):
        obs = Obstacle(0, 0, 2, 2)
        self.assertFalse(obs.contains_point(3, 3))
        self.assertFalse(obs.contains_point(-1, 1))

    def test_contains_point_with_radius(self):
        obs = Obstacle(0, 0, 2, 2)
        # Point just outside but within radius
        self.assertTrue(obs.contains_point(2.1, 1, radius=0.2))
        # Point outside radius
        self.assertFalse(obs.contains_point(3, 1, radius=0.2))

    def test_center_and_size(self):
        obs = Obstacle(0, 0, 4, 2)
        self.assertEqual(obs.center, (2, 1))
        self.assertEqual(obs.size, (4, 2))


class TestMapGenerator(unittest.TestCase):
    """Tests for MapGenerator class."""

    def test_deterministic_with_seed(self):
        config = MapConfig()
        gen1 = MapGenerator(config, seed=42)
        gen2 = MapGenerator(config, seed=42)

        obs1, start1, goal1, grid1 = gen1.generate()
        obs2, start2, goal2, grid2 = gen2.generate()

        self.assertEqual(len(obs1), len(obs2))
        np.testing.assert_array_equal(start1, start2)
        np.testing.assert_array_equal(goal1, goal2)

    def test_generates_valid_obstacles(self):
        config = MapConfig(n_obstacles_range=(3, 5))
        gen = MapGenerator(config, seed=42)
        obstacles, start, goal, grid = gen.generate()

        self.assertGreaterEqual(len(obstacles), 3)
        self.assertLessEqual(len(obstacles), 5)

        for obs in obstacles:
            self.assertLess(obs.x_min, obs.x_max)
            self.assertLess(obs.y_min, obs.y_max)

    def test_start_goal_distance(self):
        config = MapConfig(start_goal_min_dist=5.0, start_goal_max_dist=10.0)
        gen = MapGenerator(config, seed=42)

        for _ in range(5):
            _, start, goal, _ = gen.generate()
            dist = np.linalg.norm(start[:2] - goal)
            self.assertGreaterEqual(dist, 5.0 - 0.5)  # Allow small tolerance
            self.assertLessEqual(dist, 10.0 + 0.5)


class TestDrivingEnv(unittest.TestCase):
    """Tests for DrivingEnv class."""

    def setUp(self):
        self.env_config = EnvConfig(dt=0.1, v=1.0, T_max=100)
        self.map_config = MapConfig(n_obstacles_range=(2, 4))
        self.env = DrivingEnv(self.env_config, self.map_config)

    def test_reset_returns_valid_state(self):
        self.env.set_map_seed(42)
        state = self.env.reset()

        self.assertEqual(state.shape, (4,))
        self.assertEqual(state.dtype, np.float32)

        # Check velocity
        self.assertAlmostEqual(state[3], self.env_config.v)

    def test_step_updates_state(self):
        self.env.set_map_seed(42)
        state = self.env.reset()
        initial_pos = state[:2].copy()

        next_state, cost, done, info = self.env.step(0.0)

        # Position should change (moving forward)
        self.assertFalse(np.allclose(next_state[:2], initial_pos))

    def test_step_returns_bounded_action(self):
        self.env.set_map_seed(42)
        self.env.reset()

        # Even with extreme action, should be clipped
        _, _, _, _ = self.env.step(100.0)
        # Check that action was clipped in internal storage
        self.assertLessEqual(abs(self.env.actions[-1]), self.env_config.u_max)

    def test_collision_detection(self):
        # Create environment with known obstacle
        self.env.set_map_seed(42)
        state = self.env.reset()

        # Run until done
        for _ in range(self.env_config.T_max):
            _, _, done, info = self.env.step(0.0)
            if done:
                break

        # Should eventually either succeed, collide, or timeout
        self.assertTrue(self.env.done)

    def test_get_features_shape(self):
        self.env.set_map_seed(42)
        state = self.env.reset()
        goal = self.env.get_goal()

        features = self.env.get_features(state, goal)

        self.assertEqual(features.shape, (12,))
        self.assertEqual(features.dtype, np.float32)

    def test_deterministic_dynamics(self):
        """Test that dynamics are deterministic."""
        self.env.set_map_seed(42)
        state1 = self.env.reset()
        s1, c1, d1, i1 = self.env.step(0.5)

        self.env.set_map_seed(42)
        state2 = self.env.reset()
        s2, c2, d2, i2 = self.env.step(0.5)

        np.testing.assert_array_almost_equal(s1, s2)

    def test_episode_data(self):
        self.env.set_map_seed(42)
        self.env.reset()

        for _ in range(10):
            self.env.step(0.1)

        data = self.env.get_episode_data()

        self.assertIn('trajectory', data)
        self.assertIn('actions', data)
        self.assertIn('goal', data)
        self.assertEqual(len(data['actions']), len(data['trajectory']) - 1)


class TestDynamics(unittest.TestCase):
    """Tests for environment dynamics correctness."""

    def test_heading_update(self):
        """Test heading updates correctly with steering."""
        config = EnvConfig(dt=0.1, v=0.0)  # Zero velocity for isolated heading test
        env = DrivingEnv(config, MapConfig())
        env.set_map_seed(42)

        # Override with known initial state
        obstacles = [Obstacle(100, 100, 101, 101)]  # Far away
        start = np.array([10.0, 10.0, 0.0], dtype=np.float32)
        goal = np.array([15.0, 10.0], dtype=np.float32)

        state = env.reset(obstacles, start, goal, np.zeros((80, 80), dtype=np.uint8))

        # Apply steering for one step
        omega = 0.5
        next_state, _, _, _ = env.step(omega)

        expected_heading = 0.0 + 0.1 * 0.5  # psi + dt * omega
        self.assertAlmostEqual(next_state[2], expected_heading, places=5)

    def test_position_update(self):
        """Test position updates correctly with velocity."""
        config = EnvConfig(dt=0.1, v=1.0)
        env = DrivingEnv(config, MapConfig())
        env.set_map_seed(42)

        obstacles = [Obstacle(100, 100, 101, 101)]
        start = np.array([10.0, 10.0, 0.0], dtype=np.float32)  # Heading = 0 (east)
        goal = np.array([15.0, 10.0], dtype=np.float32)

        state = env.reset(obstacles, start, goal, np.zeros((80, 80), dtype=np.uint8))

        next_state, _, _, _ = env.step(0.0)

        # x should increase, y should stay same (heading = 0)
        expected_x = 10.0 + 0.1 * 1.0 * np.cos(0)
        expected_y = 10.0 + 0.1 * 1.0 * np.sin(0)

        self.assertAlmostEqual(next_state[0], expected_x, places=5)
        self.assertAlmostEqual(next_state[1], expected_y, places=5)


if __name__ == '__main__':
    unittest.main()
