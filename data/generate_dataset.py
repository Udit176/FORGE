"""
Dataset Generation Module.

Generates calibration, training, and test datasets for imitation learning.
Includes caching with configuration hashing.
"""

import numpy as np
import os
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pickle

from envs.driving_env import DrivingEnv, EnvConfig, MapConfig, Obstacle
from oracle.oracle_controller import OracleController, PlannerConfig, ControllerConfig
from humans.synthetic_teacher import (
    SyntheticTeacher, DemonstratorParams, DemonstratorPopulation,
    DemonstratorPopulationConfig
)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    # Episode counts per demonstrator
    k_calib: int = 8       # Calibration episodes
    k_train: int = 16      # Training episodes
    k_test_id: int = 8     # In-distribution test episodes
    k_test_shift_a: int = 8  # Shift A test episodes
    k_test_shift_b: int = 8  # Shift B test episodes
    k_test_shift_c: int = 8  # Shift C test episodes

    # Data storage options
    store_trajectory: bool = True
    store_features: bool = True
    store_path_curvature: bool = False

    # Cache settings
    cache_dir: str = 'data/cache'
    use_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'k_calib': self.k_calib,
            'k_train': self.k_train,
            'k_test_id': self.k_test_id,
            'k_test_shift_a': self.k_test_shift_a,
            'k_test_shift_b': self.k_test_shift_b,
            'k_test_shift_c': self.k_test_shift_c,
            'store_trajectory': self.store_trajectory,
            'store_features': self.store_features,
            'store_path_curvature': self.store_path_curvature
        }


@dataclass
class EpisodeData:
    """Data from a single episode."""
    # Identifiers
    demonstrator_id: int
    episode_id: int
    split: str  # 'calib', 'train', 'test_id', 'test_shift_a', etc.

    # Environment setup
    map_seed: int
    obstacles: List[Tuple[float, float, float, float]]  # (x_min, y_min, x_max, y_max)
    start_state: np.ndarray  # [x, y, psi]
    goal: np.ndarray  # [x, y]

    # Trajectory data
    states: np.ndarray        # (T, 4) states
    features: np.ndarray      # (T, D) feature vectors
    oracle_actions: np.ndarray  # (T,) oracle labels
    human_actions: np.ndarray   # (T,) human labels

    # Episode outcomes
    success: bool
    collision: bool
    t_final: int

    # Optional diagnostic data
    path_curvatures: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'demonstrator_id': self.demonstrator_id,
            'episode_id': self.episode_id,
            'split': self.split,
            'map_seed': self.map_seed,
            'obstacles': self.obstacles,
            'start_state': self.start_state,
            'goal': self.goal,
            'states': self.states,
            'features': self.features,
            'oracle_actions': self.oracle_actions,
            'human_actions': self.human_actions,
            'success': self.success,
            'collision': self.collision,
            't_final': self.t_final,
            'path_curvatures': self.path_curvatures
        }


class DatasetCache:
    """Cache manager for generated datasets."""

    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Compute hash of configuration for cache lookup."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def get_cache_path(self, config_hash: str, split: str) -> Path:
        """Get cache file path for given config and split."""
        return self.cache_dir / f'dataset_{config_hash}_{split}.npz'

    def exists(self, config_hash: str, split: str) -> bool:
        """Check if cached dataset exists."""
        return self.get_cache_path(config_hash, split).exists()

    def save(self, episodes: List[EpisodeData], config_hash: str, split: str):
        """Save dataset to cache."""
        cache_path = self.get_cache_path(config_hash, split)

        # Convert to arrays for efficient storage
        data = {
            'demonstrator_ids': np.array([e.demonstrator_id for e in episodes]),
            'episode_ids': np.array([e.episode_id for e in episodes]),
            'splits': np.array([e.split for e in episodes]),
            'map_seeds': np.array([e.map_seed for e in episodes]),
            'successes': np.array([e.success for e in episodes]),
            'collisions': np.array([e.collision for e in episodes]),
            't_finals': np.array([e.t_final for e in episodes]),
        }

        # Store variable-length arrays using object arrays
        data['obstacles'] = np.array([e.obstacles for e in episodes], dtype=object)
        data['start_states'] = np.array([e.start_state for e in episodes])
        data['goals'] = np.array([e.goal for e in episodes])
        data['states'] = np.array([e.states for e in episodes], dtype=object)
        data['features'] = np.array([e.features for e in episodes], dtype=object)
        data['oracle_actions'] = np.array([e.oracle_actions for e in episodes], dtype=object)
        data['human_actions'] = np.array([e.human_actions for e in episodes], dtype=object)

        if episodes[0].path_curvatures is not None:
            data['path_curvatures'] = np.array([e.path_curvatures for e in episodes], dtype=object)

        np.savez_compressed(cache_path, **data)

    def load(self, config_hash: str, split: str) -> List[EpisodeData]:
        """Load dataset from cache."""
        cache_path = self.get_cache_path(config_hash, split)
        data = np.load(cache_path, allow_pickle=True)

        episodes = []
        n = len(data['demonstrator_ids'])

        for i in range(n):
            path_curv = None
            if 'path_curvatures' in data:
                path_curv = data['path_curvatures'][i]

            episode = EpisodeData(
                demonstrator_id=int(data['demonstrator_ids'][i]),
                episode_id=int(data['episode_ids'][i]),
                split=str(data['splits'][i]),
                map_seed=int(data['map_seeds'][i]),
                obstacles=list(data['obstacles'][i]),
                start_state=data['start_states'][i],
                goal=data['goals'][i],
                states=data['states'][i],
                features=data['features'][i],
                oracle_actions=data['oracle_actions'][i],
                human_actions=data['human_actions'][i],
                success=bool(data['successes'][i]),
                collision=bool(data['collisions'][i]),
                t_final=int(data['t_finals'][i]),
                path_curvatures=path_curv
            )
            episodes.append(episode)

        return episodes


class DatasetGenerator:
    """
    Generates datasets for imitation learning experiments.

    Creates calibration, training, and test episodes for each demonstrator.
    """

    def __init__(self, env_config: EnvConfig,
                 map_config: MapConfig,
                 dataset_config: DatasetConfig,
                 population: DemonstratorPopulation,
                 seed: int = 42):
        self.env_config = env_config
        self.map_config = map_config
        self.dataset_config = dataset_config
        self.population = population
        self.seed = seed

        # Create environment and oracle
        self.env = DrivingEnv(env_config, map_config)
        self.oracle = OracleController()

        # Create cache
        self.cache = DatasetCache(dataset_config.cache_dir)

        # Compute configuration hash for caching
        self.config_hash = self._compute_full_config_hash()

    def _compute_full_config_hash(self) -> str:
        """Compute hash of full configuration."""
        config = {
            'env': self.env_config.to_dict(),
            'map': self.map_config.to_dict(),
            'dataset': self.dataset_config.to_dict(),
            'population': self.population.config.to_dict(),
            'seed': self.seed
        }
        return self.cache.compute_config_hash(config)

    def generate_all(self, verbose: bool = True) -> Dict[str, List[EpisodeData]]:
        """
        Generate all datasets (calibration, training, test).

        Returns:
            Dict mapping split names to episode lists
        """
        from tqdm import tqdm

        datasets = {}

        # Define splits and their configs
        splits = [
            ('calib', self.dataset_config.k_calib, 'train', None, None),
            ('train', self.dataset_config.k_train, 'train', None, None),
            ('test_id', self.dataset_config.k_test_id, 'train', None, None),
            ('test_shift_a', self.dataset_config.k_test_shift_a, 'shift_a', None, None),
            ('test_shift_b', self.dataset_config.k_test_shift_b, 'shift_b', None, None),
            ('test_shift_c', self.dataset_config.k_test_shift_c, 'shift_c', None, None),
        ]

        for split_name, k_episodes, map_type, map_cfg_override, env_cfg_override in splits:
            if verbose:
                print(f"Generating {split_name} dataset...")

            # Check cache
            if self.dataset_config.use_cache and self.cache.exists(self.config_hash, split_name):
                if verbose:
                    print(f"  Loading from cache...")
                datasets[split_name] = self.cache.load(self.config_hash, split_name)
                continue

            # Generate dataset
            episodes = self._generate_split(
                split_name, k_episodes, map_type,
                map_cfg_override, env_cfg_override, verbose
            )
            datasets[split_name] = episodes

            # Cache dataset
            if self.dataset_config.use_cache:
                self.cache.save(episodes, self.config_hash, split_name)

        return datasets

    def _generate_split(self, split_name: str, k_episodes: int,
                        map_type: str,
                        map_cfg_override: Optional[MapConfig],
                        env_cfg_override: Optional[EnvConfig],
                        verbose: bool) -> List[EpisodeData]:
        """Generate episodes for a single split."""
        from tqdm import tqdm

        episodes = []

        # Setup map config for this split
        if map_type == 'train':
            map_cfg = self.map_config
        elif map_type == 'shift_a':
            # New obstacle layouts (same distribution, different seeds)
            map_cfg = self.map_config
        elif map_type == 'shift_b':
            # Increased obstacle density
            map_cfg = MapConfig(
                world_size=self.map_config.world_size,
                n_obstacles_range=(
                    self.map_config.n_obstacles_range[1],
                    self.map_config.n_obstacles_range[1] + 5
                ),
                obstacle_size_range=self.map_config.obstacle_size_range,
                start_goal_min_dist=self.map_config.start_goal_min_dist,
                start_goal_max_dist=self.map_config.start_goal_max_dist,
                margin=self.map_config.margin
            )
        elif map_type == 'shift_c':
            map_cfg = self.map_config
        else:
            map_cfg = map_cfg_override or self.map_config

        # Setup env config for this split
        if map_type == 'shift_c':
            env_cfg = EnvConfig(
                dt=self.env_config.dt * 1.5,
                v=self.env_config.v * 0.8,
                v_dynamics=self.env_config.v_dynamics,
                v_tau=self.env_config.v_tau,
                u_max=self.env_config.u_max,
                r_car=self.env_config.r_car,
                r_goal=self.env_config.r_goal,
                T_max=int(self.env_config.T_max * 1.5),
                w_dist=self.env_config.w_dist,
                w_ctrl=self.env_config.w_ctrl,
                collision_penalty=self.env_config.collision_penalty
            )
        else:
            env_cfg = env_cfg_override or self.env_config

        # Create environment with appropriate configs
        env = DrivingEnv(env_cfg, map_cfg)

        n_demos = len(self.population)
        total_episodes = n_demos * k_episodes

        pbar = tqdm(total=total_episodes, disable=not verbose, desc=f"  {split_name}")

        for demo_idx in range(n_demos):
            teacher = self.population.get_demonstrator(demo_idx)

            for ep_idx in range(k_episodes):
                # Deterministic seed for this episode
                ep_seed = self.seed * 100000 + hash(split_name) % 10000 + demo_idx * 1000 + ep_idx

                episode = self._generate_episode(
                    env, teacher, demo_idx, ep_idx, split_name, ep_seed
                )
                episodes.append(episode)
                pbar.update(1)

        pbar.close()
        return episodes

    def _generate_episode(self, env: DrivingEnv, teacher: SyntheticTeacher,
                          demo_idx: int, ep_idx: int, split_name: str,
                          seed: int) -> EpisodeData:
        """Generate a single episode."""
        # Set seeds
        env.set_map_seed(seed)
        teacher.set_seed(seed + 1)

        # Reset environment
        state = env.reset()
        goal = env.get_goal()
        obstacles = [(obs.x_min, obs.y_min, obs.x_max, obs.y_max)
                    for obs in env.get_obstacles()]
        start_state = state[:3].copy()

        # Setup oracle
        self.oracle.reset()
        occ_grid = env.get_occupancy_grid()
        if occ_grid is not None:
            self.oracle.set_map(occ_grid, grid_resolution=0.25)

        # Collect trajectory
        states = [state.copy()]
        features = [env.get_features(state, goal)]
        oracle_actions = []
        path_curvatures = []

        t = 0
        while not env.done:
            # Get oracle action
            oracle_act = self.oracle.oracle_action(
                state, goal, step=t, u_max=env.config.u_max
            )
            oracle_actions.append(oracle_act)

            # Store path curvature if requested
            if self.dataset_config.store_path_curvature:
                curv = self.oracle.compute_path_curvature(state, goal)
                path_curvatures.append(curv)

            # Take action (using oracle for data collection)
            state, cost, done, info = env.step(oracle_act)

            if not done:
                states.append(state.copy())
                features.append(env.get_features(state, goal))

            t += 1

        # Convert to arrays
        states = np.array(states, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        oracle_actions = np.array(oracle_actions, dtype=np.float32)

        # Generate human labels
        teacher.reset()
        human_actions = teacher.generate_episode_labels(oracle_actions)

        # Path curvatures
        path_curv = np.array(path_curvatures) if path_curvatures else None

        return EpisodeData(
            demonstrator_id=demo_idx,
            episode_id=ep_idx,
            split=split_name,
            map_seed=seed,
            obstacles=obstacles,
            start_state=start_state,
            goal=goal,
            states=states,
            features=features,
            oracle_actions=oracle_actions,
            human_actions=human_actions,
            success=env.success,
            collision=env.collision,
            t_final=env.t,
            path_curvatures=path_curv
        )

    def generate_online_episode(self, policy, demo_idx: int, ep_idx: int,
                                seed: int, teacher: Optional[SyntheticTeacher] = None,
                                map_type: str = 'train') -> EpisodeData:
        """
        Generate episode with policy rollout (for DAgger).

        Policy controls the agent, but we collect human labels at visited states.
        """
        # Setup environment
        if map_type == 'shift_b':
            map_cfg = MapConfig(
                world_size=self.map_config.world_size,
                n_obstacles_range=(
                    self.map_config.n_obstacles_range[1],
                    self.map_config.n_obstacles_range[1] + 5
                ),
                obstacle_size_range=self.map_config.obstacle_size_range,
                start_goal_min_dist=self.map_config.start_goal_min_dist,
                start_goal_max_dist=self.map_config.start_goal_max_dist,
                margin=self.map_config.margin
            )
            env = DrivingEnv(self.env_config, map_cfg)
        else:
            env = self.env

        env.set_map_seed(seed)

        if teacher is None:
            teacher = self.population.get_demonstrator(demo_idx)
        teacher.set_seed(seed + 1)

        # Reset
        state = env.reset()
        goal = env.get_goal()
        obstacles = [(obs.x_min, obs.y_min, obs.x_max, obs.y_max)
                    for obs in env.get_obstacles()]
        start_state = state[:3].copy()

        # Setup oracle
        self.oracle.reset()
        occ_grid = env.get_occupancy_grid()
        if occ_grid is not None:
            self.oracle.set_map(occ_grid, grid_resolution=0.25)

        # Collect trajectory
        states = [state.copy()]
        features_list = [env.get_features(state, goal)]
        oracle_actions = []
        policy_actions = []

        t = 0
        while not env.done:
            # Get oracle action (for labeling)
            oracle_act = self.oracle.oracle_action(
                state, goal, step=t, u_max=env.config.u_max
            )
            oracle_actions.append(oracle_act)

            # Get policy action
            feat = env.get_features(state, goal)
            policy_act = policy.predict(feat)
            policy_actions.append(policy_act)

            # Execute policy action
            state, cost, done, info = env.step(policy_act)

            if not done:
                states.append(state.copy())
                features_list.append(env.get_features(state, goal))

            t += 1

        # Convert to arrays
        states = np.array(states, dtype=np.float32)
        features = np.array(features_list, dtype=np.float32)
        oracle_actions = np.array(oracle_actions, dtype=np.float32)

        # Generate human labels
        teacher.reset()
        human_actions = teacher.generate_episode_labels(oracle_actions)

        return EpisodeData(
            demonstrator_id=demo_idx,
            episode_id=ep_idx,
            split='online',
            map_seed=seed,
            obstacles=obstacles,
            start_state=start_state,
            goal=goal,
            states=states,
            features=features,
            oracle_actions=oracle_actions,
            human_actions=human_actions,
            success=env.success,
            collision=env.collision,
            t_final=env.t,
            path_curvatures=None
        )


def create_dataset_generator_from_config(config: Dict[str, Any],
                                         seed: int = 42) -> DatasetGenerator:
    """Create dataset generator from configuration dict."""
    # Environment config
    env_config = EnvConfig(
        dt=config.get('dt', 0.1),
        v=config.get('v', 1.0),
        v_dynamics=config.get('v_dynamics', False),
        v_tau=config.get('v_tau', 0.5),
        u_max=config.get('u_max', 1.0),
        r_car=config.get('r_car', 0.2),
        r_goal=config.get('r_goal', 0.5),
        T_max=config.get('T_max', 200),
        w_dist=config.get('w_dist', 1.0),
        w_ctrl=config.get('w_ctrl', 0.1),
        collision_penalty=config.get('collision_penalty', 100.0)
    )

    # Map config
    map_config = MapConfig(
        world_size=tuple(config.get('world_size', [20.0, 20.0])),
        n_obstacles_range=tuple(config.get('n_obstacles_range', [3, 8])),
        obstacle_size_range=tuple(config.get('obstacle_size_range', [0.5, 2.0])),
        start_goal_min_dist=config.get('start_goal_min_dist', 8.0),
        start_goal_max_dist=config.get('start_goal_max_dist', 15.0),
        margin=config.get('margin', 1.0)
    )

    # Dataset config
    dataset_config = DatasetConfig(
        k_calib=config.get('k_calib', 8),
        k_train=config.get('k_train', 16),
        k_test_id=config.get('k_test_id', 8),
        k_test_shift_a=config.get('k_test_shift_a', 8),
        k_test_shift_b=config.get('k_test_shift_b', 8),
        k_test_shift_c=config.get('k_test_shift_c', 8),
        store_trajectory=config.get('store_trajectory', True),
        store_features=config.get('store_features', True),
        store_path_curvature=config.get('store_path_curvature', False),
        cache_dir=config.get('cache_dir', 'data/cache'),
        use_cache=config.get('use_cache', True)
    )

    # Population config
    pop_config = DemonstratorPopulationConfig(
        n_demonstrators=config.get('n_demonstrators', 50),
        tau_range=tuple(config.get('tau_range', [-5.0, 5.0])),
        g_range=tuple(config.get('g_range', [0.6, 1.4])),
        sigma_range=tuple(config.get('sigma_range', [0.0, 0.15])),
        delta_range=tuple(config.get('delta_range', [0.0, 0.1])),
        sat_range=tuple(config.get('sat_range', [0.8, 1.0])),
        dropout_range=tuple(config.get('dropout_range', [0.0, 0.1])),
        tau_g_correlation=config.get('tau_g_correlation', 0.0)
    )

    population = DemonstratorPopulation(pop_config, env_config.u_max, seed)

    return DatasetGenerator(
        env_config=env_config,
        map_config=map_config,
        dataset_config=dataset_config,
        population=population,
        seed=seed
    )


def flatten_episodes_to_arrays(episodes: List[EpisodeData]) -> Dict[str, np.ndarray]:
    """
    Flatten list of episodes into arrays for training.

    Returns:
        features: (N, D) feature array
        human_labels: (N,) human action labels
        oracle_labels: (N,) oracle action labels
        demo_ids: (N,) demonstrator IDs
    """
    all_features = []
    all_human = []
    all_oracle = []
    all_demo_ids = []

    for ep in episodes:
        T = len(ep.oracle_actions)
        all_features.append(ep.features[:T])
        all_human.append(ep.human_actions)
        all_oracle.append(ep.oracle_actions)
        all_demo_ids.append(np.full(T, ep.demonstrator_id, dtype=np.int32))

    return {
        'features': np.concatenate(all_features, axis=0),
        'human_labels': np.concatenate(all_human, axis=0),
        'oracle_labels': np.concatenate(all_oracle, axis=0),
        'demo_ids': np.concatenate(all_demo_ids, axis=0)
    }


def split_by_demonstrator(episodes: List[EpisodeData]) -> Dict[int, List[EpisodeData]]:
    """Split episodes by demonstrator ID."""
    by_demo = {}
    for ep in episodes:
        if ep.demonstrator_id not in by_demo:
            by_demo[ep.demonstrator_id] = []
        by_demo[ep.demonstrator_id].append(ep)
    return by_demo
