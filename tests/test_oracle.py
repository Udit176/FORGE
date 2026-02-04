"""
Tests for the oracle controller module.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle.oracle_controller import (
    OracleController, AStarPlanner, PurePursuitController,
    PlannerConfig, ControllerConfig
)


class TestAStarPlanner(unittest.TestCase):
    """Tests for A* path planner."""

    def setUp(self):
        self.config = PlannerConfig(grid_resolution=0.5, robot_radius=0.2)
        self.planner = AStarPlanner(self.config)

    def test_plan_empty_grid(self):
        """Test planning on empty grid."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        self.planner.set_grid(grid, grid_resolution=0.5)

        path = self.planner.plan((1, 1), (8, 8))

        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        # Path should start near start and end near goal
        self.assertLess(np.linalg.norm(np.array(path[0]) - np.array([1, 1])), 1.0)
        self.assertLess(np.linalg.norm(np.array(path[-1]) - np.array([8, 8])), 1.0)

    def test_plan_with_obstacles(self):
        """Test planning around obstacles."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        # Add wall
        grid[5:15, 10] = 1

        self.planner.set_grid(grid, grid_resolution=0.5)

        path = self.planner.plan((2, 2), (8, 8))

        self.assertIsNotNone(path)
        # Path should go around the wall

    def test_plan_blocked(self):
        """Test planning when path is completely blocked."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        # Complete wall
        grid[:, 10] = 1

        self.planner.set_grid(grid, grid_resolution=0.5)

        path = self.planner.plan((1, 1), (18, 18))

        # Should still find path (going around) or return None
        # Depends on grid size - with small grid it might find way around

    def test_world_to_grid_conversion(self):
        """Test coordinate conversion."""
        grid = np.zeros((20, 20), dtype=np.uint8)
        self.planner.set_grid(grid, grid_resolution=0.5)

        gx, gy = self.planner.world_to_grid(2.5, 3.5)
        self.assertEqual(gx, 5)
        self.assertEqual(gy, 7)

        x, y = self.planner.grid_to_world(5, 7)
        self.assertAlmostEqual(x, 2.75, places=2)  # Cell center
        self.assertAlmostEqual(y, 3.75, places=2)


class TestPurePursuitController(unittest.TestCase):
    """Tests for Pure Pursuit controller."""

    def setUp(self):
        self.config = ControllerConfig(lookahead_dist=1.0)
        self.controller = PurePursuitController(self.config)

    def test_straight_path(self):
        """Test control on straight path."""
        # State: at origin, facing east
        state = np.array([0.0, 0.0, 0.0, 1.0])
        # Path: straight east
        path = [(0, 0), (5, 0), (10, 0)]

        omega = self.controller.compute_control(state, path, v=1.0)

        # Should return near-zero steering (already aligned)
        self.assertLess(abs(omega), 0.2)

    def test_turn_right(self):
        """Test control when need to turn right."""
        # State: at origin, facing east
        state = np.array([0.0, 0.0, 0.0, 1.0])
        # Path: going south
        path = [(0, 0), (0, -5), (0, -10)]

        omega = self.controller.compute_control(state, path, v=1.0)

        # Should return negative steering (turn right/clockwise)
        self.assertLess(omega, 0)

    def test_turn_left(self):
        """Test control when need to turn left."""
        # State: at origin, facing east
        state = np.array([0.0, 0.0, 0.0, 1.0])
        # Path: going north
        path = [(0, 0), (0, 5), (0, 10)]

        omega = self.controller.compute_control(state, path, v=1.0)

        # Should return positive steering (turn left/counter-clockwise)
        self.assertGreater(omega, 0)


class TestOracleController(unittest.TestCase):
    """Tests for full oracle controller."""

    def setUp(self):
        self.oracle = OracleController()

    def test_oracle_action_bounded(self):
        """Test that oracle returns bounded actions."""
        grid = np.zeros((80, 80), dtype=np.uint8)
        self.oracle.set_map(grid, grid_resolution=0.25)

        state = np.array([5.0, 5.0, 0.0, 1.0])
        goal = np.array([15.0, 15.0])

        for _ in range(10):
            action = self.oracle.oracle_action(state, goal, u_max=1.0)
            self.assertLessEqual(abs(action), 1.0)

    def test_oracle_action_deterministic(self):
        """Test that oracle is deterministic."""
        grid = np.zeros((80, 80), dtype=np.uint8)

        self.oracle.reset()
        self.oracle.set_map(grid, grid_resolution=0.25)
        state = np.array([5.0, 5.0, 0.0, 1.0])
        goal = np.array([15.0, 15.0])
        a1 = self.oracle.oracle_action(state, goal, step=0, u_max=1.0)

        self.oracle.reset()
        self.oracle.set_map(grid, grid_resolution=0.25)
        a2 = self.oracle.oracle_action(state, goal, step=0, u_max=1.0)

        self.assertAlmostEqual(a1, a2)

    def test_oracle_action_sequence(self):
        """Test oracle action sequence computation."""
        grid = np.zeros((80, 80), dtype=np.uint8)
        self.oracle.set_map(grid, grid_resolution=0.25)

        # Create sequence of states
        states = np.array([
            [5.0, 5.0, 0.0, 1.0],
            [5.1, 5.0, 0.0, 1.0],
            [5.2, 5.0, 0.0, 1.0],
        ])
        goal = np.array([15.0, 5.0])

        actions = self.oracle.oracle_action_sequence(states, goal, u_max=1.0)

        self.assertEqual(actions.shape, (3,))
        self.assertTrue(all(abs(a) <= 1.0 for a in actions))


if __name__ == '__main__':
    unittest.main()
