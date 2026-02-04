"""
2D Kinematic Driving Environment.

Implements a car-like point with heading for imitation learning experiments.
State: s_t = [x, y, psi, v] (position, heading, velocity)
Control: u_t = omega (steering rate), constrained to [-u_max, u_max]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import hashlib
import json


@dataclass
class Obstacle:
    """Axis-aligned rectangular obstacle."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains_point(self, x: float, y: float, radius: float = 0.0) -> bool:
        """Check if point (with optional radius) collides with obstacle."""
        return (self.x_min - radius <= x <= self.x_max + radius and
                self.y_min - radius <= y <= self.y_max + radius)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def size(self) -> Tuple[float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min)


@dataclass
class MapConfig:
    """Configuration for map generation."""
    world_size: Tuple[float, float] = (20.0, 20.0)
    n_obstacles_range: Tuple[int, int] = (3, 8)
    obstacle_size_range: Tuple[float, float] = (0.5, 2.0)
    start_goal_min_dist: float = 8.0
    start_goal_max_dist: float = 15.0
    margin: float = 1.0  # Margin from world boundaries

    def to_dict(self) -> Dict[str, Any]:
        return {
            'world_size': self.world_size,
            'n_obstacles_range': self.n_obstacles_range,
            'obstacle_size_range': self.obstacle_size_range,
            'start_goal_min_dist': self.start_goal_min_dist,
            'start_goal_max_dist': self.start_goal_max_dist,
            'margin': self.margin
        }


class MapGenerator:
    """Generator for environment maps with obstacles."""

    def __init__(self, config: MapConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def generate(self) -> Tuple[List[Obstacle], np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a map with obstacles, start position, and goal.

        Returns:
            obstacles: List of Obstacle objects
            start: [x, y, psi] initial state (position + heading)
            goal: [x, y] goal position
            occupancy_grid: Binary occupancy grid for planning
        """
        cfg = self.config
        w, h = cfg.world_size

        # Generate obstacles
        n_obs = self.rng.randint(cfg.n_obstacles_range[0], cfg.n_obstacles_range[1] + 1)
        obstacles = []

        for _ in range(n_obs * 3):  # Try multiple times to place obstacles
            if len(obstacles) >= n_obs:
                break

            # Random obstacle size
            obs_w = self.rng.uniform(cfg.obstacle_size_range[0], cfg.obstacle_size_range[1])
            obs_h = self.rng.uniform(cfg.obstacle_size_range[0], cfg.obstacle_size_range[1])

            # Random position (within bounds)
            x = self.rng.uniform(cfg.margin + obs_w/2, w - cfg.margin - obs_w/2)
            y = self.rng.uniform(cfg.margin + obs_h/2, h - cfg.margin - obs_h/2)

            new_obs = Obstacle(x - obs_w/2, y - obs_h/2, x + obs_w/2, y + obs_h/2)

            # Check for overlap with existing obstacles (with some spacing)
            overlap = False
            for obs in obstacles:
                if (new_obs.x_min < obs.x_max + 0.5 and new_obs.x_max > obs.x_min - 0.5 and
                    new_obs.y_min < obs.y_max + 0.5 and new_obs.y_max > obs.y_min - 0.5):
                    overlap = True
                    break

            if not overlap:
                obstacles.append(new_obs)

        # Generate start and goal positions
        for _ in range(100):  # Try to find valid start/goal
            start_x = self.rng.uniform(cfg.margin, w - cfg.margin)
            start_y = self.rng.uniform(cfg.margin, h - cfg.margin)
            start_psi = self.rng.uniform(-np.pi, np.pi)

            goal_x = self.rng.uniform(cfg.margin, w - cfg.margin)
            goal_y = self.rng.uniform(cfg.margin, h - cfg.margin)

            dist = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

            if cfg.start_goal_min_dist <= dist <= cfg.start_goal_max_dist:
                # Check no collision at start or goal
                start_ok = not any(obs.contains_point(start_x, start_y, 0.3) for obs in obstacles)
                goal_ok = not any(obs.contains_point(goal_x, goal_y, 0.3) for obs in obstacles)

                if start_ok and goal_ok:
                    break

        start = np.array([start_x, start_y, start_psi], dtype=np.float32)
        goal = np.array([goal_x, goal_y], dtype=np.float32)

        # Generate occupancy grid for planning
        grid_resolution = 0.25
        grid_w = int(w / grid_resolution)
        grid_h = int(h / grid_resolution)
        occupancy = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for obs in obstacles:
            x_min_idx = max(0, int(obs.x_min / grid_resolution))
            x_max_idx = min(grid_w, int(np.ceil(obs.x_max / grid_resolution)))
            y_min_idx = max(0, int(obs.y_min / grid_resolution))
            y_max_idx = min(grid_h, int(np.ceil(obs.y_max / grid_resolution)))
            occupancy[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = 1

        return obstacles, start, goal, occupancy


@dataclass
class EnvConfig:
    """Environment configuration."""
    dt: float = 0.1  # Time step
    v: float = 1.0   # Fixed velocity (or initial velocity if dynamic)
    v_dynamics: bool = False  # Whether velocity has dynamics
    v_tau: float = 0.5  # Time constant for velocity dynamics
    u_max: float = 1.0  # Max steering rate
    r_car: float = 0.2  # Car collision radius
    r_goal: float = 0.5  # Goal reaching radius
    T_max: int = 200    # Max episode length

    # Cost weights for evaluation
    w_dist: float = 1.0
    w_ctrl: float = 0.1
    collision_penalty: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dt': self.dt, 'v': self.v, 'v_dynamics': self.v_dynamics,
            'v_tau': self.v_tau, 'u_max': self.u_max, 'r_car': self.r_car,
            'r_goal': self.r_goal, 'T_max': self.T_max, 'w_dist': self.w_dist,
            'w_ctrl': self.w_ctrl, 'collision_penalty': self.collision_penalty
        }


class DrivingEnv:
    """
    2D Kinematic Driving Environment.

    State: [x, y, psi, v]
    Action: omega (steering rate)
    """

    def __init__(self, config: EnvConfig, map_config: Optional[MapConfig] = None):
        self.config = config
        self.map_config = map_config or MapConfig()
        self.map_generator = MapGenerator(self.map_config)

        # Episode state
        self.state: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.obstacles: List[Obstacle] = []
        self.occupancy_grid: Optional[np.ndarray] = None
        self.t: int = 0
        self.done: bool = False
        self.collision: bool = False
        self.success: bool = False

        # Episode info
        self.trajectory: List[np.ndarray] = []
        self.actions: List[float] = []
        self.costs: List[float] = []

    def set_map_seed(self, seed: int):
        """Set seed for map generation."""
        self.map_generator.set_seed(seed)

    def reset(self, obstacles: Optional[List[Obstacle]] = None,
              start: Optional[np.ndarray] = None,
              goal: Optional[np.ndarray] = None,
              occupancy_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environment to initial state.

        If obstacles/start/goal provided, use them. Otherwise generate new map.

        Returns:
            state: Initial state [x, y, psi, v]
        """
        if obstacles is None:
            self.obstacles, start_pos, self.goal, self.occupancy_grid = self.map_generator.generate()
            self.state = np.array([start_pos[0], start_pos[1], start_pos[2], self.config.v],
                                  dtype=np.float32)
        else:
            self.obstacles = obstacles
            self.goal = goal
            self.occupancy_grid = occupancy_grid
            if start is not None:
                if len(start) == 3:
                    self.state = np.array([start[0], start[1], start[2], self.config.v],
                                          dtype=np.float32)
                else:
                    self.state = start.copy()
            else:
                self.state = np.array([1.0, 1.0, 0.0, self.config.v], dtype=np.float32)

        self.t = 0
        self.done = False
        self.collision = False
        self.success = False

        self.trajectory = [self.state.copy()]
        self.actions = []
        self.costs = []

        return self.state.copy()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: omega (steering rate)

        Returns:
            next_state: [x, y, psi, v]
            cost: Step cost
            done: Whether episode is done
            info: Additional info dict
        """
        if self.done:
            return self.state.copy(), 0.0, True, {'collision': self.collision, 'success': self.success}

        # Clip action to bounds
        omega = np.clip(action, -self.config.u_max, self.config.u_max)

        # Current state
        x, y, psi, v = self.state

        # Dynamics update
        # psi_{t+1} = psi_t + dt * omega
        # x_{t+1} = x_t + dt * v * cos(psi_t)
        # y_{t+1} = y_t + dt * v * sin(psi_t)
        dt = self.config.dt

        psi_new = psi + dt * omega
        # Wrap angle to [-pi, pi]
        psi_new = np.arctan2(np.sin(psi_new), np.cos(psi_new))

        x_new = x + dt * v * np.cos(psi)
        y_new = y + dt * v * np.sin(psi)

        # Velocity dynamics (optional)
        if self.config.v_dynamics:
            v_new = v + dt * (self.config.v - v) / self.config.v_tau
        else:
            v_new = v

        self.state = np.array([x_new, y_new, psi_new, v_new], dtype=np.float32)
        self.t += 1

        # Check collision
        self.collision = any(obs.contains_point(x_new, y_new, self.config.r_car)
                            for obs in self.obstacles)

        # Check goal reached
        dist_to_goal = np.sqrt((x_new - self.goal[0])**2 + (y_new - self.goal[1])**2)
        self.success = dist_to_goal < self.config.r_goal

        # Check termination
        self.done = self.collision or self.success or self.t >= self.config.T_max

        # Compute cost
        cost = (self.config.w_dist * dist_to_goal +
                self.config.w_ctrl * omega**2)
        if self.collision:
            cost += self.config.collision_penalty

        # Store trajectory
        self.trajectory.append(self.state.copy())
        self.actions.append(omega)
        self.costs.append(cost)

        info = {
            'collision': self.collision,
            'success': self.success,
            'dist_to_goal': dist_to_goal,
            't': self.t
        }

        return self.state.copy(), cost, self.done, info

    def get_features(self, state: Optional[np.ndarray] = None,
                     goal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute feature vector phi(s, goal) for policy input.

        Features:
        - Position (x, y) normalized
        - Heading (sin(psi), cos(psi))
        - Velocity (normalized)
        - Vector to goal in body frame (dx_body, dy_body)
        - Distance to goal (normalized)
        - Heading error to goal (sin, cos)
        - Nearest obstacle distance (normalized)

        Returns:
            features: Fixed-length feature vector (dim=12)
        """
        if state is None:
            state = self.state
        if goal is None:
            goal = self.goal

        x, y, psi, v = state
        gx, gy = goal

        # Normalize by world size
        world_w, world_h = self.map_config.world_size

        # Position features (normalized)
        x_norm = x / world_w
        y_norm = y / world_h

        # Heading features
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)

        # Velocity (normalized by typical max)
        v_norm = v / 2.0

        # Vector to goal in world frame
        dx_world = gx - x
        dy_world = gy - y

        # Transform to body frame (rotate by -psi)
        dx_body = dx_world * cos_psi + dy_world * sin_psi
        dy_body = -dx_world * sin_psi + dy_world * cos_psi

        # Normalize by max distance
        max_dist = np.sqrt(world_w**2 + world_h**2)
        dx_body_norm = dx_body / max_dist
        dy_body_norm = dy_body / max_dist

        # Distance to goal
        dist = np.sqrt(dx_world**2 + dy_world**2)
        dist_norm = dist / max_dist

        # Heading error to goal
        goal_heading = np.arctan2(dy_world, dx_world)
        heading_error = goal_heading - psi
        # Wrap to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        sin_err = np.sin(heading_error)
        cos_err = np.cos(heading_error)

        # Nearest obstacle distance
        min_obs_dist = max_dist
        for obs in self.obstacles:
            # Distance to obstacle center
            cx, cy = obs.center
            d = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Approximate distance to obstacle edge
            half_w, half_h = obs.size[0]/2, obs.size[1]/2
            d_edge = max(0, d - np.sqrt(half_w**2 + half_h**2))
            min_obs_dist = min(min_obs_dist, d_edge)

        min_obs_dist_norm = min_obs_dist / max_dist

        features = np.array([
            x_norm, y_norm,
            sin_psi, cos_psi,
            v_norm,
            dx_body_norm, dy_body_norm,
            dist_norm,
            sin_err, cos_err,
            min_obs_dist_norm,
            heading_error / np.pi  # Normalized heading error
        ], dtype=np.float32)

        return features

    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()

    def get_goal(self) -> np.ndarray:
        """Get current goal."""
        return self.goal.copy()

    def get_obstacles(self) -> List[Obstacle]:
        """Get current obstacles."""
        return self.obstacles.copy()

    def get_occupancy_grid(self) -> np.ndarray:
        """Get occupancy grid."""
        return self.occupancy_grid.copy() if self.occupancy_grid is not None else None

    def get_episode_data(self) -> Dict[str, Any]:
        """Get data from current episode."""
        return {
            'trajectory': np.array(self.trajectory),
            'actions': np.array(self.actions),
            'costs': np.array(self.costs),
            'goal': self.goal.copy(),
            'obstacles': [(obs.x_min, obs.y_min, obs.x_max, obs.y_max) for obs in self.obstacles],
            'success': self.success,
            'collision': self.collision,
            't_final': self.t
        }

    def render_to_array(self, resolution: int = 200) -> np.ndarray:
        """
        Render current state to RGB array.

        Returns:
            img: (H, W, 3) uint8 array
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle, FancyArrow

        fig, ax = plt.subplots(figsize=(6, 6))

        world_w, world_h = self.map_config.world_size
        ax.set_xlim(0, world_w)
        ax.set_ylim(0, world_h)
        ax.set_aspect('equal')

        # Draw obstacles
        for obs in self.obstacles:
            rect = Rectangle((obs.x_min, obs.y_min),
                            obs.x_max - obs.x_min,
                            obs.y_max - obs.y_min,
                            facecolor='gray', edgecolor='black')
            ax.add_patch(rect)

        # Draw goal
        goal_circle = Circle((self.goal[0], self.goal[1]),
                            self.config.r_goal,
                            facecolor='green', alpha=0.5)
        ax.add_patch(goal_circle)

        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.5)

        # Draw car
        if self.state is not None:
            x, y, psi, v = self.state
            car_circle = Circle((x, y), self.config.r_car,
                               facecolor='blue' if not self.collision else 'red')
            ax.add_patch(car_circle)

            # Heading arrow
            arrow_len = 0.5
            dx = arrow_len * np.cos(psi)
            dy = arrow_len * np.sin(psi)
            ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.1, fc='blue', ec='blue')

        ax.set_title(f't={self.t}, success={self.success}, collision={self.collision}')

        # Convert to array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img


def create_shift_configs(base_map_config: MapConfig, shift_type: str) -> MapConfig:
    """
    Create shifted map configuration for distribution shift testing.

    Args:
        base_map_config: Base configuration
        shift_type: One of 'A' (new layouts), 'B' (more obstacles), 'C' (dynamics)

    Returns:
        Shifted MapConfig
    """
    shifted = MapConfig(
        world_size=base_map_config.world_size,
        n_obstacles_range=base_map_config.n_obstacles_range,
        obstacle_size_range=base_map_config.obstacle_size_range,
        start_goal_min_dist=base_map_config.start_goal_min_dist,
        start_goal_max_dist=base_map_config.start_goal_max_dist,
        margin=base_map_config.margin
    )

    if shift_type == 'A':
        # Same distribution, just different random seeds
        pass
    elif shift_type == 'B':
        # Increased obstacle density
        shifted.n_obstacles_range = (
            base_map_config.n_obstacles_range[1],
            base_map_config.n_obstacles_range[1] + 5
        )

    return shifted


def create_env_shift_config(base_env_config: EnvConfig, shift_type: str) -> EnvConfig:
    """
    Create shifted environment configuration for dynamics shift testing.

    Args:
        base_env_config: Base configuration
        shift_type: 'C' for dynamics shift

    Returns:
        Shifted EnvConfig
    """
    shifted = EnvConfig(
        dt=base_env_config.dt,
        v=base_env_config.v,
        v_dynamics=base_env_config.v_dynamics,
        v_tau=base_env_config.v_tau,
        u_max=base_env_config.u_max,
        r_car=base_env_config.r_car,
        r_goal=base_env_config.r_goal,
        T_max=base_env_config.T_max,
        w_dist=base_env_config.w_dist,
        w_ctrl=base_env_config.w_ctrl,
        collision_penalty=base_env_config.collision_penalty
    )

    if shift_type == 'C':
        # Dynamics shift: change dt or v
        shifted.dt = base_env_config.dt * 1.5
        shifted.v = base_env_config.v * 0.8

    return shifted


def compute_config_hash(env_config: EnvConfig, map_config: MapConfig, seed: int) -> str:
    """Compute hash of configuration for caching."""
    config_dict = {
        'env': env_config.to_dict(),
        'map': map_config.to_dict(),
        'seed': seed
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]
