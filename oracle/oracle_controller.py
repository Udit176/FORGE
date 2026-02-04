"""
Oracle Controller for optimal action generation.

Two-stage oracle:
1. Path planner (A* on occupancy grid) to produce waypoints
2. Local tracking controller (Pure Pursuit or Stanley) to output steering rate
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import heapq


@dataclass
class PlannerConfig:
    """Configuration for A* path planner."""
    grid_resolution: float = 0.25  # Meters per grid cell
    robot_radius: float = 0.3      # Inflation radius for obstacles
    heuristic_weight: float = 1.0  # A* heuristic weight (1.0 = optimal)
    max_iterations: int = 10000    # Max A* iterations
    replan_interval: int = 10      # Replan every K steps (0 = plan once)


@dataclass
class ControllerConfig:
    """Configuration for Pure Pursuit controller."""
    lookahead_dist: float = 1.0    # Lookahead distance
    min_lookahead: float = 0.5     # Minimum lookahead
    max_lookahead: float = 2.0     # Maximum lookahead
    k_stanley: float = 0.5         # Stanley controller gain
    use_stanley: bool = False      # Use Stanley instead of Pure Pursuit


class AStarPlanner:
    """A* path planner on occupancy grid."""

    def __init__(self, config: PlannerConfig):
        self.config = config
        self._cached_grid = None
        self._inflated_grid = None
        self._grid_origin = (0.0, 0.0)

    def set_grid(self, occupancy_grid: np.ndarray,
                 grid_resolution: float,
                 origin: Tuple[float, float] = (0.0, 0.0)):
        """
        Set occupancy grid for planning.

        Args:
            occupancy_grid: Binary grid (1=occupied, 0=free)
            grid_resolution: Meters per cell
            origin: World coordinates of grid origin (bottom-left)
        """
        self._cached_grid = occupancy_grid
        self.config.grid_resolution = grid_resolution
        self._grid_origin = origin

        # Inflate obstacles by robot radius
        inflate_cells = int(np.ceil(self.config.robot_radius / grid_resolution))
        self._inflated_grid = self._inflate_obstacles(occupancy_grid, inflate_cells)

    def _inflate_obstacles(self, grid: np.ndarray, radius_cells: int) -> np.ndarray:
        """Inflate obstacles by given radius in cells."""
        from scipy.ndimage import binary_dilation

        # Create circular structuring element
        y, x = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
        structure = x**2 + y**2 <= radius_cells**2

        inflated = binary_dilation(grid, structure=structure)
        return inflated.astype(np.uint8)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((x - self._grid_origin[0]) / self.config.grid_resolution)
        gy = int((y - self._grid_origin[1]) / self.config.grid_resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        x = gx * self.config.grid_resolution + self._grid_origin[0] + self.config.grid_resolution / 2
        y = gy * self.config.grid_resolution + self._grid_origin[1] + self.config.grid_resolution / 2
        return x, y

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using A*.

        Args:
            start: (x, y) start position in world coordinates
            goal: (x, y) goal position in world coordinates

        Returns:
            List of (x, y) waypoints in world coordinates, or None if no path found
        """
        if self._inflated_grid is None:
            return None

        # Convert to grid coordinates
        start_g = self.world_to_grid(start[0], start[1])
        goal_g = self.world_to_grid(goal[0], goal[1])

        # Check bounds
        h, w = self._inflated_grid.shape
        if not (0 <= start_g[0] < w and 0 <= start_g[1] < h):
            return None
        if not (0 <= goal_g[0] < w and 0 <= goal_g[1] < h):
            return None

        # Check if start or goal in obstacle
        if self._inflated_grid[start_g[1], start_g[0]] == 1:
            # Try to find nearby free cell
            start_g = self._find_nearest_free(start_g)
            if start_g is None:
                return None

        if self._inflated_grid[goal_g[1], goal_g[0]] == 1:
            goal_g = self._find_nearest_free(goal_g)
            if goal_g is None:
                return None

        # A* algorithm
        path_grid = self._astar(start_g, goal_g)

        if path_grid is None:
            return None

        # Convert to world coordinates
        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]

        # Simplify path
        path_simplified = self._simplify_path(path_world)

        return path_simplified

    def _find_nearest_free(self, cell: Tuple[int, int],
                           max_search: int = 10) -> Optional[Tuple[int, int]]:
        """Find nearest free cell to given cell."""
        h, w = self._inflated_grid.shape
        gx, gy = cell

        for r in range(1, max_search):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if self._inflated_grid[ny, nx] == 0:
                                return (nx, ny)
        return None

    def _astar(self, start: Tuple[int, int],
               goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* search on grid."""
        h, w = self._inflated_grid.shape

        def heuristic(a, b):
            return self.config.heuristic_weight * np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        # 8-connected neighbors with costs
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
        ]

        open_set = [(0 + heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}

        iterations = 0
        while open_set and iterations < self.config.max_iterations:
            iterations += 1
            _, current_g, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current_g > g_score.get(current, float('inf')):
                continue

            for dx, dy, cost in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < w and 0 <= neighbor[1] < h):
                    continue

                # Check obstacle
                if self._inflated_grid[neighbor[1], neighbor[0]] == 1:
                    continue

                tentative_g = g_score[current] + cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None  # No path found

    def _simplify_path(self, path: List[Tuple[float, float]],
                       tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """Simplify path using Douglas-Peucker algorithm."""
        if len(path) <= 2:
            return path

        # Find point with maximum distance from line start-end
        start = np.array(path[0])
        end = np.array(path[-1])

        max_dist = 0
        max_idx = 0

        for i in range(1, len(path) - 1):
            p = np.array(path[i])
            # Distance from point to line segment
            line_vec = end - start
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                dist = np.linalg.norm(p - start)
            else:
                line_unit = line_vec / line_len
                proj_len = np.dot(p - start, line_unit)
                proj_len = np.clip(proj_len, 0, line_len)
                proj = start + proj_len * line_unit
                dist = np.linalg.norm(p - proj)

            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > tolerance:
            left = self._simplify_path(path[:max_idx+1], tolerance)
            right = self._simplify_path(path[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]


class PurePursuitController:
    """Pure Pursuit path tracking controller."""

    def __init__(self, config: ControllerConfig):
        self.config = config

    def compute_control(self, state: np.ndarray,
                        path: List[Tuple[float, float]],
                        v: float) -> float:
        """
        Compute steering rate using Pure Pursuit.

        Args:
            state: [x, y, psi, v] current state
            path: List of (x, y) waypoints
            v: Current velocity

        Returns:
            omega: Steering rate
        """
        if len(path) < 2:
            return 0.0

        x, y, psi = state[0], state[1], state[2]

        # Find lookahead point
        lookahead = self._adaptive_lookahead(v)
        target = self._find_lookahead_point(x, y, path, lookahead)

        if target is None:
            # Head toward last waypoint
            target = path[-1]

        # Pure Pursuit control law
        dx = target[0] - x
        dy = target[1] - y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 0.01:
            return 0.0

        # Transform to body frame
        dx_body = dx * np.cos(psi) + dy * np.sin(psi)
        dy_body = -dx * np.sin(psi) + dy * np.cos(psi)

        # Curvature
        if abs(dx_body) < 0.01:
            curvature = 0.0
        else:
            curvature = 2 * dy_body / (dist**2)

        # Steering rate (omega = v * curvature for kinematic model)
        omega = v * curvature

        return omega

    def _adaptive_lookahead(self, v: float) -> float:
        """Compute adaptive lookahead distance based on velocity."""
        lookahead = self.config.lookahead_dist * (1 + 0.5 * abs(v))
        return np.clip(lookahead, self.config.min_lookahead, self.config.max_lookahead)

    def _find_lookahead_point(self, x: float, y: float,
                              path: List[Tuple[float, float]],
                              lookahead: float) -> Optional[Tuple[float, float]]:
        """Find point on path at lookahead distance."""
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0

        for i, (px, py) in enumerate(path):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Search forward from closest point
        cumulative_dist = 0.0
        for i in range(closest_idx, len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            seg_len = np.linalg.norm(p2 - p1)

            if cumulative_dist + seg_len >= lookahead:
                # Interpolate
                remaining = lookahead - cumulative_dist
                if seg_len > 0:
                    t = remaining / seg_len
                    target = p1 + t * (p2 - p1)
                    return (target[0], target[1])
                else:
                    return (p2[0], p2[1])

            cumulative_dist += seg_len

        # Return last point if lookahead exceeds path
        return path[-1] if path else None


class StanleyController:
    """Stanley path tracking controller."""

    def __init__(self, config: ControllerConfig):
        self.config = config

    def compute_control(self, state: np.ndarray,
                        path: List[Tuple[float, float]],
                        v: float) -> float:
        """
        Compute steering rate using Stanley controller.

        Args:
            state: [x, y, psi, v] current state
            path: List of (x, y) waypoints
            v: Current velocity

        Returns:
            omega: Steering rate
        """
        if len(path) < 2:
            return 0.0

        x, y, psi = state[0], state[1], state[2]

        # Find closest point on path and path tangent
        closest_idx = 0
        min_dist = float('inf')

        for i, (px, py) in enumerate(path):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Path tangent
        if closest_idx < len(path) - 1:
            dx_path = path[closest_idx + 1][0] - path[closest_idx][0]
            dy_path = path[closest_idx + 1][1] - path[closest_idx][1]
        else:
            dx_path = path[closest_idx][0] - path[closest_idx - 1][0]
            dy_path = path[closest_idx][1] - path[closest_idx - 1][1]

        path_heading = np.arctan2(dy_path, dx_path)

        # Heading error
        heading_error = path_heading - psi
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Cross-track error (signed)
        px, py = path[closest_idx]
        dx = x - px
        dy = y - py
        cross_track = -dx * np.sin(path_heading) + dy * np.cos(path_heading)

        # Stanley control law
        k = self.config.k_stanley
        omega = heading_error + np.arctan2(k * cross_track, max(v, 0.1))

        return omega


class OracleController:
    """
    Oracle controller combining path planning and local tracking.

    Provides optimal action labels for any state.
    """

    def __init__(self, planner_config: Optional[PlannerConfig] = None,
                 controller_config: Optional[ControllerConfig] = None):
        self.planner_config = planner_config or PlannerConfig()
        self.controller_config = controller_config or ControllerConfig()

        self.planner = AStarPlanner(self.planner_config)

        if self.controller_config.use_stanley:
            self.controller = StanleyController(self.controller_config)
        else:
            self.controller = PurePursuitController(self.controller_config)

        self._current_path: Optional[List[Tuple[float, float]]] = None
        self._last_replan_step: int = -1
        self._goal: Optional[Tuple[float, float]] = None

    def set_map(self, occupancy_grid: np.ndarray,
                grid_resolution: float = 0.25,
                origin: Tuple[float, float] = (0.0, 0.0)):
        """Set occupancy grid for planning."""
        self.planner.set_grid(occupancy_grid, grid_resolution, origin)
        self._current_path = None

    def reset(self):
        """Reset controller state for new episode."""
        self._current_path = None
        self._last_replan_step = -1
        self._goal = None

    def oracle_action(self, state: np.ndarray,
                      goal: np.ndarray,
                      step: int = 0,
                      u_max: float = 1.0,
                      force_replan: bool = False) -> float:
        """
        Compute oracle (optimal) action for given state.

        Args:
            state: [x, y, psi, v] current state
            goal: [x, y] goal position
            step: Current timestep (for replanning)
            u_max: Max steering rate for clipping
            force_replan: Force path replanning

        Returns:
            omega: Optimal steering rate (clipped to [-u_max, u_max])
        """
        x, y, psi, v = state

        # Check if we need to replan
        goal_changed = self._goal is None or not np.allclose(goal, self._goal)
        interval_passed = (step - self._last_replan_step >= self.planner_config.replan_interval
                          if self.planner_config.replan_interval > 0 else False)

        if force_replan or goal_changed or interval_passed or self._current_path is None:
            self._goal = goal.copy()
            self._current_path = self.planner.plan((x, y), (goal[0], goal[1]))
            self._last_replan_step = step

        if self._current_path is None or len(self._current_path) < 2:
            # No path found - head directly toward goal
            dx = goal[0] - x
            dy = goal[1] - y
            goal_heading = np.arctan2(dy, dx)
            heading_error = goal_heading - psi
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            omega = 2.0 * heading_error  # Proportional control
        else:
            # Use path tracking controller
            omega = self.controller.compute_control(state, self._current_path, v)

        # Clip to bounds
        omega = np.clip(omega, -u_max, u_max)

        return float(omega)

    def get_path(self) -> Optional[List[Tuple[float, float]]]:
        """Get current planned path."""
        return self._current_path.copy() if self._current_path else None

    def oracle_action_sequence(self, states: np.ndarray,
                               goal: np.ndarray,
                               u_max: float = 1.0) -> np.ndarray:
        """
        Compute oracle actions for sequence of states.

        Useful for computing time-shifted oracle labels.

        Args:
            states: (T, 4) array of states
            goal: (2,) goal position
            u_max: Max steering rate

        Returns:
            actions: (T,) array of oracle actions
        """
        actions = np.zeros(len(states), dtype=np.float32)

        for t, state in enumerate(states):
            actions[t] = self.oracle_action(state, goal, step=t, u_max=u_max)

        return actions

    def compute_path_curvature(self, state: np.ndarray,
                               goal: np.ndarray) -> float:
        """
        Compute curvature of planned path at current position.

        Useful as diagnostic feature.
        """
        if self._current_path is None or len(self._current_path) < 3:
            return 0.0

        x, y = state[0], state[1]

        # Find closest point on path
        closest_idx = 0
        min_dist = float('inf')
        for i, (px, py) in enumerate(self._current_path):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Need 3 points for curvature
        if closest_idx == 0:
            idx = 1
        elif closest_idx >= len(self._current_path) - 1:
            idx = len(self._current_path) - 2
        else:
            idx = closest_idx

        p1 = np.array(self._current_path[idx - 1])
        p2 = np.array(self._current_path[idx])
        p3 = np.array(self._current_path[idx + 1])

        # Menger curvature: 4*area / (|a|*|b|*|c|)
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        if a * b * c < 1e-6:
            return 0.0

        # Signed area using cross product
        area = 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                           (p3[0] - p1[0]) * (p2[1] - p1[1]))

        curvature = 4 * area / (a * b * c)
        return curvature


def create_oracle_from_config(config: Dict[str, Any]) -> OracleController:
    """Create oracle controller from configuration dict."""
    planner_config = PlannerConfig(
        grid_resolution=config.get('grid_resolution', 0.25),
        robot_radius=config.get('robot_radius', 0.3),
        heuristic_weight=config.get('heuristic_weight', 1.0),
        max_iterations=config.get('max_iterations', 10000),
        replan_interval=config.get('replan_interval', 10)
    )

    controller_config = ControllerConfig(
        lookahead_dist=config.get('lookahead_dist', 1.0),
        min_lookahead=config.get('min_lookahead', 0.5),
        max_lookahead=config.get('max_lookahead', 2.0),
        k_stanley=config.get('k_stanley', 0.5),
        use_stanley=config.get('use_stanley', False)
    )

    return OracleController(planner_config, controller_config)
