"""
Rollout Evaluation Module.

Evaluates policies by rolling out in the environment.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from envs.driving_env import DrivingEnv, EnvConfig, MapConfig, Obstacle
from oracle.oracle_controller import OracleController
from eval.metrics import MetricsSummary, compute_metrics, compute_episode_metrics


@dataclass
class EvalConfig:
    """Configuration for rollout evaluation."""
    n_episodes: int = 20
    seed_offset: int = 10000  # Offset for eval seeds (separate from training)
    verbose: bool = False
    save_trajectories: bool = True


class RolloutEvaluator:
    """
    Evaluates policies by rolling out in the environment.

    Supports evaluation on different test distributions.
    """

    def __init__(self, env_config: EnvConfig,
                 map_config: MapConfig,
                 eval_config: Optional[EvalConfig] = None):
        self.env_config = env_config
        self.map_config = map_config
        self.eval_config = eval_config or EvalConfig()

        # Create environment
        self.env = DrivingEnv(env_config, map_config)

        # Oracle for comparison
        self.oracle = OracleController()

    def evaluate_policy(self, policy,
                        n_episodes: Optional[int] = None,
                        seed: int = 42,
                        verbose: Optional[bool] = None) -> Tuple[MetricsSummary, List[Dict[str, Any]]]:
        """
        Evaluate policy on test episodes.

        Args:
            policy: Policy with predict(features) method
            n_episodes: Number of episodes (default from config)
            seed: Random seed for reproducibility
            verbose: Print progress

        Returns:
            summary: MetricsSummary object
            episodes: List of episode data dicts
        """
        if n_episodes is None:
            n_episodes = self.eval_config.n_episodes
        if verbose is None:
            verbose = self.eval_config.verbose

        episodes = []
        base_seed = seed + self.eval_config.seed_offset

        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="Evaluating")

        for i in iterator:
            ep_seed = base_seed + i
            ep_data = self._run_episode(policy, ep_seed)
            episodes.append(ep_data)

        summary = compute_metrics(episodes)
        return summary, episodes

    def evaluate_oracle(self, n_episodes: Optional[int] = None,
                        seed: int = 42,
                        verbose: Optional[bool] = None) -> Tuple[MetricsSummary, List[Dict[str, Any]]]:
        """
        Evaluate oracle controller (upper bound).

        Returns:
            summary: MetricsSummary object
            episodes: List of episode data dicts
        """
        if n_episodes is None:
            n_episodes = self.eval_config.n_episodes
        if verbose is None:
            verbose = self.eval_config.verbose

        episodes = []
        base_seed = seed + self.eval_config.seed_offset

        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="Oracle eval")

        for i in iterator:
            ep_seed = base_seed + i
            ep_data = self._run_oracle_episode(ep_seed)
            episodes.append(ep_data)

        summary = compute_metrics(episodes)
        return summary, episodes

    def _run_episode(self, policy, seed: int) -> Dict[str, Any]:
        """Run a single evaluation episode with policy."""
        self.env.set_map_seed(seed)
        state = self.env.reset()
        goal = self.env.get_goal()

        # Setup oracle for potential feature computation
        self.oracle.reset()
        occ_grid = self.env.get_occupancy_grid()
        if occ_grid is not None:
            self.oracle.set_map(occ_grid)

        trajectory = [state.copy()]
        actions = []
        costs = []

        while not self.env.done:
            # Get features
            features = self.env.get_features(state, goal)

            # Get policy action
            action = policy.predict(features)

            # Step
            state, cost, done, info = self.env.step(action)

            trajectory.append(state.copy())
            actions.append(action)
            costs.append(cost)

        return {
            'trajectory': np.array(trajectory),
            'actions': np.array(actions),
            'costs': np.array(costs),
            'goal': goal,
            'success': self.env.success,
            'collision': self.env.collision,
            'r_goal': self.env.config.r_goal,
            'T_max': self.env.config.T_max
        }

    def _run_oracle_episode(self, seed: int) -> Dict[str, Any]:
        """Run a single evaluation episode with oracle."""
        self.env.set_map_seed(seed)
        state = self.env.reset()
        goal = self.env.get_goal()

        # Setup oracle
        self.oracle.reset()
        occ_grid = self.env.get_occupancy_grid()
        if occ_grid is not None:
            self.oracle.set_map(occ_grid)

        trajectory = [state.copy()]
        actions = []
        costs = []
        t = 0

        while not self.env.done:
            # Get oracle action
            action = self.oracle.oracle_action(state, goal, step=t,
                                               u_max=self.env.config.u_max)

            # Step
            state, cost, done, info = self.env.step(action)

            trajectory.append(state.copy())
            actions.append(action)
            costs.append(cost)
            t += 1

        return {
            'trajectory': np.array(trajectory),
            'actions': np.array(actions),
            'costs': np.array(costs),
            'goal': goal,
            'success': self.env.success,
            'collision': self.env.collision,
            'r_goal': self.env.config.r_goal,
            'T_max': self.env.config.T_max
        }

    def evaluate_on_shift(self, policy, shift_type: str,
                          n_episodes: Optional[int] = None,
                          seed: int = 42,
                          verbose: Optional[bool] = None) -> Tuple[MetricsSummary, List[Dict[str, Any]]]:
        """
        Evaluate policy on distribution shift.

        Args:
            policy: Policy to evaluate
            shift_type: 'A' (new layouts), 'B' (more obstacles), 'C' (dynamics)
            n_episodes: Number of episodes
            seed: Random seed
            verbose: Print progress

        Returns:
            summary: MetricsSummary
            episodes: List of episode data
        """
        # Create shifted configs
        if shift_type == 'A':
            # Same distribution, different seeds
            shifted_map = self.map_config
            shifted_env = self.env_config
            seed_offset = 50000  # Different seed range

        elif shift_type == 'B':
            # More obstacles
            shifted_map = MapConfig(
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
            shifted_env = self.env_config
            seed_offset = self.eval_config.seed_offset

        elif shift_type == 'C':
            # Dynamics shift
            shifted_map = self.map_config
            shifted_env = EnvConfig(
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
            seed_offset = self.eval_config.seed_offset
        else:
            raise ValueError(f"Unknown shift type: {shift_type}")

        # Create shifted environment
        shifted_evaluator = RolloutEvaluator(shifted_env, shifted_map, self.eval_config)

        # Evaluate
        return shifted_evaluator.evaluate_policy(
            policy, n_episodes, seed + (seed_offset if shift_type == 'A' else 0), verbose
        )


class MultiConditionEvaluator:
    """
    Evaluates policies across multiple test conditions.
    """

    def __init__(self, env_config: EnvConfig,
                 map_config: MapConfig,
                 eval_config: Optional[EvalConfig] = None):
        self.base_evaluator = RolloutEvaluator(env_config, map_config, eval_config)
        self.env_config = env_config
        self.map_config = map_config
        self.eval_config = eval_config or EvalConfig()

    def evaluate_all(self, policy,
                     seed: int = 42,
                     verbose: bool = False) -> Dict[str, Tuple[MetricsSummary, List[Dict]]]:
        """
        Evaluate policy on all test conditions.

        Returns:
            Dict mapping condition name to (summary, episodes)
        """
        results = {}

        if verbose:
            print("Evaluating in-distribution...")
        results['in_dist'] = self.base_evaluator.evaluate_policy(
            policy, seed=seed, verbose=verbose
        )

        if verbose:
            print("Evaluating shift A (new layouts)...")
        results['shift_a'] = self.base_evaluator.evaluate_on_shift(
            policy, 'A', seed=seed, verbose=verbose
        )

        if verbose:
            print("Evaluating shift B (more obstacles)...")
        results['shift_b'] = self.base_evaluator.evaluate_on_shift(
            policy, 'B', seed=seed, verbose=verbose
        )

        if verbose:
            print("Evaluating shift C (dynamics)...")
        results['shift_c'] = self.base_evaluator.evaluate_on_shift(
            policy, 'C', seed=seed, verbose=verbose
        )

        return results

    def evaluate_all_methods(self, methods: Dict[str, Any],
                             seed: int = 42,
                             verbose: bool = False) -> Dict[str, Dict[str, MetricsSummary]]:
        """
        Evaluate multiple methods across all conditions.

        Args:
            methods: Dict mapping method name to policy (or 'oracle' for oracle)
            seed: Random seed
            verbose: Print progress

        Returns:
            Dict[method_name][condition_name] = MetricsSummary
        """
        results = {}

        for method_name, policy in methods.items():
            if verbose:
                print(f"\n=== Evaluating {method_name} ===")

            if method_name == 'oracle' or policy == 'oracle':
                # Special handling for oracle
                results[method_name] = {}
                summary, _ = self.base_evaluator.evaluate_oracle(seed=seed, verbose=verbose)
                results[method_name]['in_dist'] = summary
                # Oracle on shifts
                for shift in ['A', 'B', 'C']:
                    # Create shifted evaluator
                    if shift == 'A':
                        shifted_eval = self.base_evaluator
                        shift_seed = seed + 50000
                    elif shift == 'B':
                        shifted_map = MapConfig(
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
                        shifted_eval = RolloutEvaluator(self.env_config, shifted_map, self.eval_config)
                        shift_seed = seed
                    else:  # C
                        shifted_env = EnvConfig(
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
                        shifted_eval = RolloutEvaluator(shifted_env, self.map_config, self.eval_config)
                        shift_seed = seed

                    summary, _ = shifted_eval.evaluate_oracle(seed=shift_seed, verbose=False)
                    results[method_name][f'shift_{shift.lower()}'] = summary
            else:
                all_results = self.evaluate_all(policy, seed, verbose)
                results[method_name] = {k: v[0] for k, v in all_results.items()}

        return results


def create_evaluator_from_config(config: Dict[str, Any]) -> RolloutEvaluator:
    """Create evaluator from configuration dict."""
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

    map_config = MapConfig(
        world_size=tuple(config.get('world_size', [20.0, 20.0])),
        n_obstacles_range=tuple(config.get('n_obstacles_range', [3, 8])),
        obstacle_size_range=tuple(config.get('obstacle_size_range', [0.5, 2.0])),
        start_goal_min_dist=config.get('start_goal_min_dist', 8.0),
        start_goal_max_dist=config.get('start_goal_max_dist', 15.0),
        margin=config.get('margin', 1.0)
    )

    eval_config = EvalConfig(
        n_episodes=config.get('n_eval_episodes', 20),
        seed_offset=config.get('eval_seed_offset', 10000),
        verbose=config.get('eval_verbose', False),
        save_trajectories=config.get('save_trajectories', True)
    )

    return RolloutEvaluator(env_config, map_config, eval_config)


def summarize_results_table(results: Dict[str, Dict[str, MetricsSummary]]) -> str:
    """
    Create text table of results.

    Args:
        results: Dict[method_name][condition_name] = MetricsSummary

    Returns:
        Formatted table string
    """
    # Get all conditions
    conditions = set()
    for method_results in results.values():
        conditions.update(method_results.keys())
    conditions = sorted(list(conditions))

    # Header
    header = "Method".ljust(15) + "".join(c.ljust(15) for c in conditions)
    lines = [header, "-" * len(header)]

    # Rows
    for method_name, method_results in sorted(results.items()):
        row = method_name.ljust(15)
        for condition in conditions:
            if condition in method_results:
                summary = method_results[condition]
                row += f"{summary.success_rate:.3f}".ljust(15)
            else:
                row += "N/A".ljust(15)
        lines.append(row)

    return "\n".join(lines)
