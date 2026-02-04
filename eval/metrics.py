"""
Evaluation Metrics Module.

Computes metrics for evaluating imitation learning policies.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class MetricsSummary:
    """Summary of evaluation metrics for a set of episodes."""
    # Episode outcomes
    success_rate: float = 0.0
    collision_rate: float = 0.0
    timeout_rate: float = 0.0

    # Performance metrics
    mean_time_to_goal: float = 0.0
    std_time_to_goal: float = 0.0
    mean_path_length: float = 0.0
    std_path_length: float = 0.0

    # Cost metrics
    mean_integrated_cost: float = 0.0
    std_integrated_cost: float = 0.0
    mean_final_dist_to_goal: float = 0.0

    # Control metrics
    mean_control_effort: float = 0.0  # Mean |omega|
    control_smoothness: float = 0.0   # Mean |delta omega|
    mean_abs_heading_error: float = 0.0

    # Sample count
    n_episodes: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'success_rate': self.success_rate,
            'collision_rate': self.collision_rate,
            'timeout_rate': self.timeout_rate,
            'mean_time_to_goal': self.mean_time_to_goal,
            'std_time_to_goal': self.std_time_to_goal,
            'mean_path_length': self.mean_path_length,
            'std_path_length': self.std_path_length,
            'mean_integrated_cost': self.mean_integrated_cost,
            'std_integrated_cost': self.std_integrated_cost,
            'mean_final_dist_to_goal': self.mean_final_dist_to_goal,
            'mean_control_effort': self.mean_control_effort,
            'control_smoothness': self.control_smoothness,
            'mean_abs_heading_error': self.mean_abs_heading_error,
            'n_episodes': self.n_episodes
        }

    def __repr__(self):
        return (f"MetricsSummary(success={self.success_rate:.3f}, "
                f"collision={self.collision_rate:.3f}, "
                f"time={self.mean_time_to_goal:.1f}±{self.std_time_to_goal:.1f})")


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    success: bool = False
    collision: bool = False
    timeout: bool = False
    time_to_goal: int = 0
    path_length: float = 0.0
    integrated_cost: float = 0.0
    final_dist_to_goal: float = 0.0
    control_effort: float = 0.0
    control_smoothness: float = 0.0
    mean_heading_error: float = 0.0


def compute_episode_metrics(trajectory: np.ndarray,
                            actions: np.ndarray,
                            goal: np.ndarray,
                            success: bool,
                            collision: bool,
                            costs: Optional[np.ndarray] = None,
                            r_goal: float = 0.5,
                            T_max: int = 200) -> EpisodeMetrics:
    """
    Compute metrics for a single episode.

    Args:
        trajectory: (T+1, 4) array of states [x, y, psi, v]
        actions: (T,) array of actions
        goal: (2,) goal position
        success: Whether goal was reached
        collision: Whether collision occurred
        costs: (T,) array of step costs (optional)
        r_goal: Goal radius
        T_max: Max episode length

    Returns:
        EpisodeMetrics object
    """
    metrics = EpisodeMetrics()

    metrics.success = success
    metrics.collision = collision
    metrics.timeout = not success and not collision and len(actions) >= T_max
    metrics.time_to_goal = len(actions)

    # Path length
    if len(trajectory) > 1:
        diffs = np.diff(trajectory[:, :2], axis=0)
        metrics.path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        metrics.path_length = 0.0

    # Integrated cost
    if costs is not None:
        metrics.integrated_cost = float(np.sum(costs))
    else:
        metrics.integrated_cost = 0.0

    # Final distance to goal
    final_pos = trajectory[-1, :2]
    metrics.final_dist_to_goal = float(np.linalg.norm(final_pos - goal))

    # Control metrics
    if len(actions) > 0:
        metrics.control_effort = float(np.mean(np.abs(actions)))

        if len(actions) > 1:
            action_diffs = np.diff(actions)
            metrics.control_smoothness = float(np.mean(np.abs(action_diffs)))
        else:
            metrics.control_smoothness = 0.0

    # Heading error
    if len(trajectory) > 1:
        heading_errors = []
        for i in range(len(trajectory) - 1):
            x, y, psi = trajectory[i, :3]
            dx = goal[0] - x
            dy = goal[1] - y
            goal_heading = np.arctan2(dy, dx)
            error = goal_heading - psi
            error = np.arctan2(np.sin(error), np.cos(error))  # Wrap to [-pi, pi]
            heading_errors.append(abs(error))
        metrics.mean_heading_error = float(np.mean(heading_errors))

    return metrics


def compute_metrics(episode_list: List[Dict[str, Any]]) -> MetricsSummary:
    """
    Compute aggregate metrics over a list of episodes.

    Args:
        episode_list: List of episode dicts with keys:
            - trajectory, actions, goal, success, collision, costs (optional)

    Returns:
        MetricsSummary object
    """
    if not episode_list:
        return MetricsSummary()

    all_metrics = []

    for ep in episode_list:
        metrics = compute_episode_metrics(
            trajectory=ep.get('trajectory', np.zeros((2, 4))),
            actions=ep.get('actions', np.array([])),
            goal=ep.get('goal', np.array([0, 0])),
            success=ep.get('success', False),
            collision=ep.get('collision', False),
            costs=ep.get('costs'),
            r_goal=ep.get('r_goal', 0.5),
            T_max=ep.get('T_max', 200)
        )
        all_metrics.append(metrics)

    # Aggregate
    summary = MetricsSummary()
    summary.n_episodes = len(all_metrics)

    summary.success_rate = np.mean([m.success for m in all_metrics])
    summary.collision_rate = np.mean([m.collision for m in all_metrics])
    summary.timeout_rate = np.mean([m.timeout for m in all_metrics])

    times = [m.time_to_goal for m in all_metrics if m.success]
    if times:
        summary.mean_time_to_goal = np.mean(times)
        summary.std_time_to_goal = np.std(times)

    lengths = [m.path_length for m in all_metrics if m.success]
    if lengths:
        summary.mean_path_length = np.mean(lengths)
        summary.std_path_length = np.std(lengths)

    costs = [m.integrated_cost for m in all_metrics]
    summary.mean_integrated_cost = np.mean(costs)
    summary.std_integrated_cost = np.std(costs)

    summary.mean_final_dist_to_goal = np.mean([m.final_dist_to_goal for m in all_metrics])
    summary.mean_control_effort = np.mean([m.control_effort for m in all_metrics])
    summary.control_smoothness = np.mean([m.control_smoothness for m in all_metrics])
    summary.mean_abs_heading_error = np.mean([m.mean_heading_error for m in all_metrics])

    return summary


def compute_label_metrics(human_labels: np.ndarray,
                          oracle_labels: np.ndarray,
                          corrected_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute offline label metrics.

    Args:
        human_labels: (N,) raw human labels
        oracle_labels: (N,) oracle labels
        corrected_labels: (N,) corrected labels (optional)

    Returns:
        Dict with MSE, correlation, etc.
    """
    metrics = {}

    # Raw metrics
    metrics['mse_raw'] = float(np.mean((human_labels - oracle_labels)**2))
    metrics['mae_raw'] = float(np.mean(np.abs(human_labels - oracle_labels)))

    if len(human_labels) > 1:
        corr = np.corrcoef(human_labels, oracle_labels)[0, 1]
        metrics['correlation_raw'] = float(corr) if not np.isnan(corr) else 0.0
    else:
        metrics['correlation_raw'] = 0.0

    # Corrected metrics
    if corrected_labels is not None:
        metrics['mse_corrected'] = float(np.mean((corrected_labels - oracle_labels)**2))
        metrics['mae_corrected'] = float(np.mean(np.abs(corrected_labels - oracle_labels)))

        if len(corrected_labels) > 1:
            corr = np.corrcoef(corrected_labels, oracle_labels)[0, 1]
            metrics['correlation_corrected'] = float(corr) if not np.isnan(corr) else 0.0
        else:
            metrics['correlation_corrected'] = 0.0

        # Improvement
        metrics['mse_improvement'] = ((metrics['mse_raw'] - metrics['mse_corrected']) /
                                      (metrics['mse_raw'] + 1e-8))

    # Amplitude bias
    oracle_mag = np.mean(np.abs(oracle_labels)) + 1e-8
    human_signed = np.mean(human_labels * np.sign(oracle_labels + 1e-8))
    metrics['amplitude_bias_raw'] = float(human_signed / oracle_mag - 1.0)

    if corrected_labels is not None:
        corrected_signed = np.mean(corrected_labels * np.sign(oracle_labels + 1e-8))
        metrics['amplitude_bias_corrected'] = float(corrected_signed / oracle_mag - 1.0)

    return metrics


def compute_timing_metrics(human_labels: np.ndarray,
                           oracle_labels: np.ndarray,
                           corrected_labels: Optional[np.ndarray] = None,
                           max_lag: int = 20) -> Dict[str, float]:
    """
    Compute timing (lag) metrics.

    Args:
        human_labels: (N,) raw human labels
        oracle_labels: (N,) oracle labels
        corrected_labels: (N,) corrected labels (optional)
        max_lag: Maximum lag to search

    Returns:
        Dict with timing metrics
    """
    from scipy.signal import correlate

    metrics = {}

    def estimate_lag(seq1, seq2):
        if len(seq1) < 5 or len(seq2) < 5:
            return 0.0

        # Normalize
        s1 = seq1 - np.mean(seq1)
        s2 = seq2 - np.mean(seq2)
        s1 = s1 / (np.std(s1) + 1e-8)
        s2 = s2 / (np.std(s2) + 1e-8)

        # Cross-correlation
        corr = correlate(s1, s2, mode='full')
        lags = np.arange(-len(s2) + 1, len(s1))

        # Find peak
        valid = np.abs(lags) <= max_lag
        if not np.any(valid):
            return 0.0

        peak_idx = np.argmax(corr[valid])
        lag = lags[valid][peak_idx]
        return float(lag)

    metrics['timing_offset_raw'] = estimate_lag(human_labels, oracle_labels)

    if corrected_labels is not None:
        metrics['timing_offset_corrected'] = estimate_lag(corrected_labels, oracle_labels)
        metrics['timing_improvement'] = (abs(metrics['timing_offset_raw']) -
                                         abs(metrics['timing_offset_corrected']))

    return metrics


def compare_methods(method_results: Dict[str, MetricsSummary],
                    base_method: str = 'oracle') -> Dict[str, Dict[str, float]]:
    """
    Compare multiple methods against a baseline.

    Args:
        method_results: Dict mapping method name to MetricsSummary
        base_method: Name of baseline method

    Returns:
        Dict with relative performance for each method
    """
    comparison = {}

    if base_method not in method_results:
        base = MetricsSummary(success_rate=1.0)
    else:
        base = method_results[base_method]

    for method_name, metrics in method_results.items():
        if method_name == base_method:
            continue

        comparison[method_name] = {
            'success_rate_diff': metrics.success_rate - base.success_rate,
            'success_rate_relative': metrics.success_rate / (base.success_rate + 1e-8),
            'collision_rate_diff': metrics.collision_rate - base.collision_rate,
            'time_diff': metrics.mean_time_to_goal - base.mean_time_to_goal,
            'cost_diff': metrics.mean_integrated_cost - base.mean_integrated_cost
        }

    return comparison


def compute_statistical_comparison(metrics1: List[float],
                                   metrics2: List[float],
                                   method: str = 'ttest') -> Dict[str, float]:
    """
    Compute statistical comparison between two sets of metrics.

    Args:
        metrics1: List of metric values for method 1
        metrics2: List of metric values for method 2
        method: Statistical test ('ttest', 'wilcoxon', 'bootstrap')

    Returns:
        Dict with p-value, effect size, etc.
    """
    from scipy import stats

    result = {}

    if method == 'ttest':
        stat, pvalue = stats.ttest_rel(metrics1, metrics2)
        result['statistic'] = float(stat)
        result['p_value'] = float(pvalue)

    elif method == 'wilcoxon':
        stat, pvalue = stats.wilcoxon(metrics1, metrics2)
        result['statistic'] = float(stat)
        result['p_value'] = float(pvalue)

    elif method == 'bootstrap':
        # Bootstrap confidence interval for difference
        n_bootstrap = 1000
        diffs = []
        n = len(metrics1)
        rng = np.random.RandomState(42)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            diff = np.mean(np.array(metrics1)[idx]) - np.mean(np.array(metrics2)[idx])
            diffs.append(diff)

        diffs = np.array(diffs)
        result['mean_diff'] = float(np.mean(diffs))
        result['ci_lower'] = float(np.percentile(diffs, 2.5))
        result['ci_upper'] = float(np.percentile(diffs, 97.5))
        result['p_value'] = float(np.mean(diffs > 0)) if np.mean(diffs) < 0 else float(np.mean(diffs < 0))

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(metrics1)**2 + np.std(metrics2)**2) / 2)
    result['effect_size'] = float((np.mean(metrics1) - np.mean(metrics2)) / (pooled_std + 1e-8))

    return result
