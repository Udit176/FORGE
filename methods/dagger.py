"""
DAgger (Dataset Aggregation) Implementation.

Implements classic DAgger loop for imitation learning with various
label sources (human, corrected).
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import copy

from methods.policy import MLPPolicy, PolicyConfig, create_policy_from_config
from data.generate_dataset import EpisodeData, flatten_episodes_to_arrays
from envs.driving_env import DrivingEnv, EnvConfig, MapConfig
from oracle.oracle_controller import OracleController
from humans.synthetic_teacher import SyntheticTeacher


@dataclass
class DAggerConfig:
    """Configuration for DAgger algorithm."""
    n_iterations: int = 5           # Number of DAgger iterations
    m_rollouts_per_iter: int = 10   # Rollouts per iteration per demonstrator
    k_init: int = 4                 # Initial episodes for BC (from train data)
    beta_schedule: str = 'constant' # 'constant', 'linear_decay', 'exponential_decay'
    beta_init: float = 1.0          # Initial mixing parameter (1=all expert)
    beta_final: float = 0.0         # Final mixing parameter

    # Aggregation options
    aggregate_mode: str = 'all'     # 'all', 'recent', 'weighted'
    recent_window: int = 2          # Window for 'recent' mode
    weight_decay: float = 0.9       # Weight decay for 'weighted' mode

    # Label source
    label_source: str = 'human'     # 'human', 'oracle', 'corrected'

    # Training options
    warm_start: bool = True         # Warm start policy from previous iteration
    retrain_from_scratch: bool = False  # Retrain policy from scratch each iteration

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_iterations': self.n_iterations,
            'm_rollouts_per_iter': self.m_rollouts_per_iter,
            'k_init': self.k_init,
            'beta_schedule': self.beta_schedule,
            'beta_init': self.beta_init,
            'beta_final': self.beta_final,
            'aggregate_mode': self.aggregate_mode,
            'recent_window': self.recent_window,
            'weight_decay': self.weight_decay,
            'label_source': self.label_source,
            'warm_start': self.warm_start,
            'retrain_from_scratch': self.retrain_from_scratch
        }


class DAgger:
    """
    DAgger algorithm for imitation learning.

    Supports different label sources:
    - 'human': Use synthetic teacher labels (for Simple DAgger baseline)
    - 'oracle': Use oracle labels (upper bound)
    - 'corrected': Use externally provided corrector function
    """

    def __init__(self, config: DAggerConfig,
                 policy_config: PolicyConfig,
                 env_config: EnvConfig,
                 map_config: MapConfig,
                 seed: int = 42):
        self.config = config
        self.policy_config = policy_config
        self.env_config = env_config
        self.map_config = map_config
        self.seed = seed

        # Create environment and oracle
        self.env = DrivingEnv(env_config, map_config)
        self.oracle = OracleController()

        # Policy
        self.policy = create_policy_from_config(policy_config.to_dict(), seed)

        # Dataset
        self.aggregated_features: List[np.ndarray] = []
        self.aggregated_labels: List[np.ndarray] = []
        self.iteration_data: List[List[EpisodeData]] = []

        # Label corrector (set externally for 'corrected' mode)
        self.label_corrector: Optional[Callable] = None

        # Training history
        self.iteration_history: List[Dict[str, Any]] = []

    def set_label_corrector(self, corrector: Callable):
        """
        Set label corrector function.

        Corrector signature: corrector(human_labels, features, demo_id, context) -> corrected_labels
        """
        self.label_corrector = corrector

    def initialize_from_data(self, initial_episodes: List[EpisodeData],
                             teachers: Dict[int, SyntheticTeacher],
                             verbose: bool = False):
        """
        Initialize DAgger with behavior cloning on initial data.

        Args:
            initial_episodes: Initial training episodes
            teachers: Dict mapping demo_id to SyntheticTeacher
            verbose: Print progress
        """
        # Select k_init episodes per demonstrator
        by_demo = {}
        for ep in initial_episodes:
            if ep.demonstrator_id not in by_demo:
                by_demo[ep.demonstrator_id] = []
            by_demo[ep.demonstrator_id].append(ep)

        selected_episodes = []
        for demo_id, eps in by_demo.items():
            selected = eps[:self.config.k_init]
            selected_episodes.extend(selected)

        # Extract features and labels
        features, labels = self._extract_labels_from_episodes(
            selected_episodes, teachers
        )

        self.aggregated_features = [features]
        self.aggregated_labels = [labels]

        # Train initial policy
        if verbose:
            print("Training initial policy via behavior cloning...")

        all_features = np.concatenate(self.aggregated_features, axis=0)
        all_labels = np.concatenate(self.aggregated_labels, axis=0)

        history = self.policy.train(all_features, all_labels, verbose=verbose)

        self.iteration_history.append({
            'iteration': 0,
            'type': 'init',
            'n_samples': len(all_labels),
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None
        })

    def run_iteration(self, iteration: int,
                      teachers: Dict[int, SyntheticTeacher],
                      demo_ids: Optional[List[int]] = None,
                      verbose: bool = False) -> Dict[str, Any]:
        """
        Run a single DAgger iteration.

        Args:
            iteration: Current iteration number (1-indexed)
            teachers: Dict mapping demo_id to SyntheticTeacher
            demo_ids: List of demonstrator IDs to collect from (None = all)
            verbose: Print progress

        Returns:
            Iteration statistics
        """
        if demo_ids is None:
            demo_ids = list(teachers.keys())

        # Compute beta for this iteration
        beta = self._compute_beta(iteration)

        # Collect rollouts
        new_episodes = []
        ep_seed = self.seed * 10000 + iteration * 1000

        for demo_id in demo_ids:
            teacher = teachers[demo_id]

            for m in range(self.config.m_rollouts_per_iter):
                episode = self._collect_rollout(
                    demo_id, teacher, beta, ep_seed + demo_id * 100 + m
                )
                new_episodes.append(episode)

        # Extract labels
        features, labels = self._extract_labels_from_episodes(new_episodes, teachers)

        # Aggregate
        self.aggregated_features.append(features)
        self.aggregated_labels.append(labels)
        self.iteration_data.append(new_episodes)

        # Get aggregated dataset
        all_features, all_labels = self._get_aggregated_data()

        # Train policy
        if self.config.retrain_from_scratch:
            self.policy = create_policy_from_config(
                self.policy_config.to_dict(), self.seed + iteration
            )

        if verbose:
            print(f"DAgger iteration {iteration}: training on {len(all_labels)} samples...")

        history = self.policy.train(all_features, all_labels, verbose=verbose)

        # Compute statistics
        stats = {
            'iteration': iteration,
            'beta': beta,
            'n_new_samples': len(labels),
            'n_total_samples': len(all_labels),
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'success_rate': np.mean([ep.success for ep in new_episodes]),
            'collision_rate': np.mean([ep.collision for ep in new_episodes])
        }

        self.iteration_history.append(stats)
        return stats

    def run(self, initial_episodes: List[EpisodeData],
            teachers: Dict[int, SyntheticTeacher],
            verbose: bool = True) -> MLPPolicy:
        """
        Run full DAgger training loop.

        Args:
            initial_episodes: Initial training episodes for BC
            teachers: Dict mapping demo_id to SyntheticTeacher
            verbose: Print progress

        Returns:
            Trained policy
        """
        # Initialize
        self.initialize_from_data(initial_episodes, teachers, verbose)

        # Run iterations
        for i in range(1, self.config.n_iterations + 1):
            if verbose:
                print(f"\n=== DAgger Iteration {i}/{self.config.n_iterations} ===")

            stats = self.run_iteration(i, teachers, verbose=verbose)

            if verbose:
                print(f"  Success rate: {stats['success_rate']:.3f}")
                print(f"  Collision rate: {stats['collision_rate']:.3f}")

        return self.policy

    def _compute_beta(self, iteration: int) -> float:
        """Compute mixing parameter beta for given iteration."""
        cfg = self.config

        if cfg.beta_schedule == 'constant':
            return cfg.beta_init

        progress = iteration / cfg.n_iterations

        if cfg.beta_schedule == 'linear_decay':
            return cfg.beta_init + (cfg.beta_final - cfg.beta_init) * progress

        elif cfg.beta_schedule == 'exponential_decay':
            decay = np.log(cfg.beta_final / (cfg.beta_init + 1e-8))
            return cfg.beta_init * np.exp(decay * progress)

        return cfg.beta_init

    def _collect_rollout(self, demo_id: int, teacher: SyntheticTeacher,
                         beta: float, seed: int) -> EpisodeData:
        """Collect a single rollout with mixed policy."""
        self.env.set_map_seed(seed)
        teacher.set_seed(seed + 1)

        # Reset environment
        state = self.env.reset()
        goal = self.env.get_goal()
        obstacles = [(obs.x_min, obs.y_min, obs.x_max, obs.y_max)
                    for obs in self.env.get_obstacles()]
        start_state = state[:3].copy()

        # Setup oracle
        self.oracle.reset()
        occ_grid = self.env.get_occupancy_grid()
        if occ_grid is not None:
            self.oracle.set_map(occ_grid, grid_resolution=0.25)

        # Collect trajectory
        states = [state.copy()]
        features_list = [self.env.get_features(state, goal)]
        oracle_actions = []
        executed_actions = []

        rng = np.random.RandomState(seed + 2)
        t = 0

        while not self.env.done:
            # Get oracle action
            oracle_act = self.oracle.oracle_action(
                state, goal, step=t, u_max=self.env.config.u_max
            )
            oracle_actions.append(oracle_act)

            # Get policy action
            feat = self.env.get_features(state, goal)
            policy_act = self.policy.predict(feat)

            # Mix actions according to beta
            if rng.random() < beta:
                action = oracle_act  # Expert action
            else:
                action = policy_act  # Policy action

            executed_actions.append(action)

            # Step environment
            state, cost, done, info = self.env.step(action)

            if not done:
                states.append(state.copy())
                features_list.append(self.env.get_features(state, goal))

            t += 1

        # Convert to arrays
        states = np.array(states, dtype=np.float32)
        features = np.array(features_list, dtype=np.float32)
        oracle_actions = np.array(oracle_actions, dtype=np.float32)

        # Generate human labels
        teacher.reset()
        human_actions = teacher.generate_episode_labels(oracle_actions)

        return EpisodeData(
            demonstrator_id=demo_id,
            episode_id=seed,
            split='dagger',
            map_seed=seed,
            obstacles=obstacles,
            start_state=start_state,
            goal=goal,
            states=states,
            features=features,
            oracle_actions=oracle_actions,
            human_actions=human_actions,
            success=self.env.success,
            collision=self.env.collision,
            t_final=self.env.t
        )

    def _extract_labels_from_episodes(self, episodes: List[EpisodeData],
                                       teachers: Dict[int, SyntheticTeacher]
                                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from episodes based on label source.

        Returns:
            features: (N, D) array
            labels: (N,) array
        """
        all_features = []
        all_labels = []

        for ep in episodes:
            T = len(ep.oracle_actions)
            features = ep.features[:T]
            all_features.append(features)

            if self.config.label_source == 'oracle':
                labels = ep.oracle_actions

            elif self.config.label_source == 'human':
                labels = ep.human_actions

            elif self.config.label_source == 'corrected':
                if self.label_corrector is None:
                    raise ValueError("Label corrector not set for 'corrected' mode")

                # Apply correction
                labels = self.label_corrector(
                    human_labels=ep.human_actions,
                    features=features,
                    demo_id=ep.demonstrator_id,
                    oracle_labels=ep.oracle_actions,  # For context
                    states=ep.states[:T]
                )
            else:
                raise ValueError(f"Unknown label source: {self.config.label_source}")

            all_labels.append(labels)

        return (np.concatenate(all_features, axis=0),
                np.concatenate(all_labels, axis=0))

    def _get_aggregated_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get aggregated dataset based on aggregation mode."""
        cfg = self.config

        if cfg.aggregate_mode == 'all':
            features = np.concatenate(self.aggregated_features, axis=0)
            labels = np.concatenate(self.aggregated_labels, axis=0)

        elif cfg.aggregate_mode == 'recent':
            start_idx = max(0, len(self.aggregated_features) - cfg.recent_window)
            features = np.concatenate(self.aggregated_features[start_idx:], axis=0)
            labels = np.concatenate(self.aggregated_labels[start_idx:], axis=0)

        elif cfg.aggregate_mode == 'weighted':
            # Weight samples by iteration recency
            n_iters = len(self.aggregated_features)
            all_features = []
            all_labels = []
            all_weights = []

            for i, (feat, lab) in enumerate(zip(self.aggregated_features,
                                                 self.aggregated_labels)):
                weight = cfg.weight_decay ** (n_iters - 1 - i)
                all_features.append(feat)
                all_labels.append(lab)
                all_weights.append(np.full(len(lab), weight))

            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            # Note: weighted sampling could be implemented in training

        else:
            raise ValueError(f"Unknown aggregate mode: {cfg.aggregate_mode}")

        return features, labels

    def get_policy(self) -> MLPPolicy:
        """Get trained policy."""
        return self.policy

    def get_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.iteration_history


class SimpleDAgger(DAgger):
    """
    Simple DAgger with human labels.

    Convenience class that pre-configures DAgger with human labels.
    """

    def __init__(self, policy_config: PolicyConfig,
                 env_config: EnvConfig,
                 map_config: MapConfig,
                 n_iterations: int = 5,
                 m_rollouts: int = 10,
                 k_init: int = 4,
                 seed: int = 42):
        config = DAggerConfig(
            n_iterations=n_iterations,
            m_rollouts_per_iter=m_rollouts,
            k_init=k_init,
            label_source='human',
            beta_schedule='linear_decay',
            beta_init=1.0,
            beta_final=0.0
        )
        super().__init__(config, policy_config, env_config, map_config, seed)


class CorrectedDAgger(DAgger):
    """
    DAgger with corrected labels.

    Used for MIND MELD and Cognitive Model pipelines.
    """

    def __init__(self, policy_config: PolicyConfig,
                 env_config: EnvConfig,
                 map_config: MapConfig,
                 corrector: Callable,
                 n_iterations: int = 5,
                 m_rollouts: int = 10,
                 k_init: int = 4,
                 seed: int = 42):
        config = DAggerConfig(
            n_iterations=n_iterations,
            m_rollouts_per_iter=m_rollouts,
            k_init=k_init,
            label_source='corrected',
            beta_schedule='linear_decay',
            beta_init=1.0,
            beta_final=0.0
        )
        super().__init__(config, policy_config, env_config, map_config, seed)
        self.set_label_corrector(corrector)


def create_dagger_from_config(config: Dict[str, Any],
                              policy_config: PolicyConfig,
                              env_config: EnvConfig,
                              map_config: MapConfig,
                              seed: int = 42) -> DAgger:
    """Create DAgger from configuration dict."""
    dagger_config = DAggerConfig(
        n_iterations=config.get('n_iterations', 5),
        m_rollouts_per_iter=config.get('m_rollouts_per_iter', 10),
        k_init=config.get('k_init', 4),
        beta_schedule=config.get('beta_schedule', 'constant'),
        beta_init=config.get('beta_init', 1.0),
        beta_final=config.get('beta_final', 0.0),
        aggregate_mode=config.get('aggregate_mode', 'all'),
        recent_window=config.get('recent_window', 2),
        weight_decay=config.get('weight_decay', 0.9),
        label_source=config.get('label_source', 'human'),
        warm_start=config.get('warm_start', True),
        retrain_from_scratch=config.get('retrain_from_scratch', False)
    )
    return DAgger(dagger_config, policy_config, env_config, map_config, seed)
