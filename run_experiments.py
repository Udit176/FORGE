#!/usr/bin/env python3
"""
Forward-Model Guided Demonstration Learning - Main Experiment Runner

Usage:
    python run_experiments.py --config configs/base.yaml
    python run_experiments.py --config configs/quick.yaml
    python run_experiments.py --config configs/base.yaml --mode quick
    python run_experiments.py --config configs/ablation_mindmeld.yaml --ablation A_MM_1

For a list of available options:
    python run_experiments.py --help
"""

import argparse
import yaml
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from envs.driving_env import DrivingEnv, EnvConfig, MapConfig
from oracle.oracle_controller import OracleController, PlannerConfig, ControllerConfig
from humans.synthetic_teacher import (
    DemonstratorPopulation, DemonstratorPopulationConfig, SyntheticTeacher
)
from data.generate_dataset import (
    DatasetGenerator, DatasetConfig, EpisodeData, split_by_demonstrator,
    flatten_episodes_to_arrays
)
from methods.policy import MLPPolicy, PolicyConfig, OraclePolicy
from methods.dagger import DAgger, DAggerConfig, SimpleDAgger, CorrectedDAgger
from methods.mindmeld import MindMeld, MindMeldConfig, MindMeldNonPersonalized
from methods.cognitive_model import CognitiveModel, CognitiveModelConfig
from eval.metrics import MetricsSummary, compute_metrics, compute_label_metrics
from eval.rollout_eval import RolloutEvaluator, MultiConditionEvaluator, EvalConfig
from experiments.ablations import AblationRunner, HypothesisTester


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str, mode: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply quick mode overrides if requested
    if mode == 'quick' or config.get('experiment', {}).get('mode') == 'quick':
        quick = config.get('quick_mode', {})
        if quick:
            # Apply quick mode settings
            if 'n_seeds' in quick:
                config['experiment']['n_seeds'] = quick['n_seeds']
            if 'n_demonstrators' in quick:
                config['population']['n_demonstrators'] = quick['n_demonstrators']
            if 'k_calib' in quick:
                config['dataset']['k_calib'] = quick['k_calib']
            if 'k_train' in quick:
                config['dataset']['k_train'] = quick['k_train']
            if 'k_test_id' in quick:
                config['dataset']['k_test_id'] = quick['k_test_id']
            if 'k_test_shift_a' in quick:
                config['dataset']['k_test_shift_a'] = quick['k_test_shift_a']
            if 'k_test_shift_b' in quick:
                config['dataset']['k_test_shift_b'] = quick['k_test_shift_b']
            if 'k_test_shift_c' in quick:
                config['dataset']['k_test_shift_c'] = quick['k_test_shift_c']
            if 'dagger_n_iterations' in quick:
                config['dagger']['n_iterations'] = quick['dagger_n_iterations']
            if 'dagger_m_rollouts' in quick:
                config['dagger']['m_rollouts_per_iter'] = quick['dagger_m_rollouts']
            if 'policy_epochs' in quick:
                config['policy']['epochs'] = quick['policy_epochs']
            if 'mindmeld_epochs' in quick:
                config['mindmeld']['epochs'] = quick['mindmeld_epochs']
            if 'eval_n_episodes' in quick:
                config['evaluation']['n_episodes'] = quick['eval_n_episodes']

    return config


def flatten_config(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested config dict for easy access."""
    flat = {}
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, new_key))
        else:
            flat[new_key] = value
    return flat


def create_results_dir(config: Dict[str, Any]) -> Path:
    """Create results directory with timestamp."""
    base_dir = config.get('experiment', {}).get('save_dir', 'results')
    exp_name = config.get('experiment', {}).get('name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (results_dir / 'logs').mkdir(exist_ok=True)
    (results_dir / 'plots').mkdir(exist_ok=True)
    (results_dir / 'models').mkdir(exist_ok=True)
    (results_dir / 'data').mkdir(exist_ok=True)

    return results_dir


def run_single_experiment(config: Dict[str, Any],
                          seed: int,
                          results_dir: Path,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Run a single experiment with given seed.

    Returns dict with results for all methods.
    """
    set_seed(seed)
    results = {'seed': seed}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed={seed}")
        print(f"{'='*60}")

    # Create configs from flat config
    env_cfg = config.get('environment', {})
    map_cfg = config.get('map', {})
    pop_cfg = config.get('population', {})
    data_cfg = config.get('dataset', {})
    pol_cfg = config.get('policy', {})
    dag_cfg = config.get('dagger', {})
    mm_cfg = config.get('mindmeld', {})
    cog_cfg = config.get('cognitive', {})
    eval_cfg = config.get('evaluation', {})

    # Create environment config
    env_config = EnvConfig(
        dt=env_cfg.get('dt', 0.1),
        v=env_cfg.get('v', 1.0),
        v_dynamics=env_cfg.get('v_dynamics', False),
        v_tau=env_cfg.get('v_tau', 0.5),
        u_max=env_cfg.get('u_max', 1.0),
        r_car=env_cfg.get('r_car', 0.2),
        r_goal=env_cfg.get('r_goal', 0.5),
        T_max=env_cfg.get('T_max', 200),
        w_dist=env_cfg.get('w_dist', 1.0),
        w_ctrl=env_cfg.get('w_ctrl', 0.1),
        collision_penalty=env_cfg.get('collision_penalty', 100.0)
    )

    map_config = MapConfig(
        world_size=tuple(map_cfg.get('world_size', [20.0, 20.0])),
        n_obstacles_range=tuple(map_cfg.get('n_obstacles_range', [3, 8])),
        obstacle_size_range=tuple(map_cfg.get('obstacle_size_range', [0.5, 2.0])),
        start_goal_min_dist=map_cfg.get('start_goal_min_dist', 8.0),
        start_goal_max_dist=map_cfg.get('start_goal_max_dist', 15.0),
        margin=map_cfg.get('margin', 1.0)
    )

    # Create population
    pop_config = DemonstratorPopulationConfig(
        n_demonstrators=pop_cfg.get('n_demonstrators', 50),
        tau_range=tuple(pop_cfg.get('tau_range', [-5.0, 5.0])),
        g_range=tuple(pop_cfg.get('g_range', [0.6, 1.4])),
        sigma_range=tuple(pop_cfg.get('sigma_range', [0.0, 0.15])),
        delta_range=tuple(pop_cfg.get('delta_range', [0.0, 0.1])),
        sat_range=tuple(pop_cfg.get('sat_range', [0.8, 1.0])),
        dropout_range=tuple(pop_cfg.get('dropout_range', [0.0, 0.1]))
    )
    population = DemonstratorPopulation(pop_config, env_config.u_max, seed)

    if verbose:
        print(f"\nGenerated {len(population)} demonstrators")

    # Create dataset config
    dataset_config = DatasetConfig(
        k_calib=data_cfg.get('k_calib', 8),
        k_train=data_cfg.get('k_train', 16),
        k_test_id=data_cfg.get('k_test_id', 8),
        k_test_shift_a=data_cfg.get('k_test_shift_a', 8),
        k_test_shift_b=data_cfg.get('k_test_shift_b', 8),
        k_test_shift_c=data_cfg.get('k_test_shift_c', 8),
        cache_dir=data_cfg.get('cache_dir', 'data/cache'),
        use_cache=data_cfg.get('use_cache', True)
    )

    # Generate datasets
    if verbose:
        print("\nGenerating datasets...")

    generator = DatasetGenerator(env_config, map_config, dataset_config, population, seed)
    datasets = generator.generate_all(verbose=verbose)

    # Split calibration data by demonstrator
    calib_by_demo = split_by_demonstrator(datasets['calib'])
    train_episodes = datasets['train']

    # Create teachers dict
    teachers = {i: population.get_demonstrator(i) for i in range(len(population))}

    # Create policy config
    policy_config = PolicyConfig(
        input_dim=pol_cfg.get('input_dim', 12),
        hidden_dims=pol_cfg.get('hidden_dims', [64, 64]),
        output_dim=pol_cfg.get('output_dim', 1),
        activation=pol_cfg.get('activation', 'relu'),
        dropout=pol_cfg.get('dropout', 0.0),
        lr=pol_cfg.get('lr', 1e-3),
        weight_decay=pol_cfg.get('weight_decay', 1e-4),
        batch_size=pol_cfg.get('batch_size', 64),
        epochs=pol_cfg.get('epochs', 30),
        early_stopping_patience=pol_cfg.get('early_stopping_patience', 5),
        val_fraction=pol_cfg.get('val_fraction', 0.1),
        u_max=env_config.u_max
    )

    # Create evaluator
    eval_config_obj = EvalConfig(
        n_episodes=eval_cfg.get('n_episodes', 20),
        seed_offset=eval_cfg.get('seed_offset', 10000),
        verbose=verbose
    )
    evaluator = MultiConditionEvaluator(env_config, map_config, eval_config_obj)

    # Methods to run
    methods_to_run = config.get('methods', ['oracle', 'dagger_human', 'dagger_mm', 'dagger_cog'])

    # Results storage
    method_results = {}
    trained_policies = {}

    # ========== C0: Oracle Policy ==========
    if 'oracle' in methods_to_run:
        if verbose:
            print("\n--- Evaluating Oracle Policy (C0) ---")

        oracle_results = {}
        for condition in ['in_dist', 'shift_a', 'shift_b', 'shift_c']:
            if condition == 'in_dist':
                summary, _ = evaluator.base_evaluator.evaluate_oracle(seed=seed, verbose=False)
            elif condition == 'shift_a':
                # Same distribution, different seeds
                summary, _ = evaluator.base_evaluator.evaluate_oracle(seed=seed + 50000, verbose=False)
            elif condition == 'shift_b':
                # More obstacles - create shifted evaluator
                shifted_map = MapConfig(
                    world_size=map_config.world_size,
                    n_obstacles_range=(
                        map_config.n_obstacles_range[1],
                        map_config.n_obstacles_range[1] + 5
                    ),
                    obstacle_size_range=map_config.obstacle_size_range,
                    start_goal_min_dist=map_config.start_goal_min_dist,
                    start_goal_max_dist=map_config.start_goal_max_dist,
                    margin=map_config.margin
                )
                shifted_eval = RolloutEvaluator(env_config, shifted_map, eval_config_obj)
                summary, _ = shifted_eval.evaluate_oracle(seed=seed, verbose=False)
            else:  # shift_c
                # Dynamics shift - create shifted evaluator
                shifted_env = EnvConfig(
                    dt=env_config.dt * 1.5,
                    v=env_config.v * 0.8,
                    v_dynamics=env_config.v_dynamics,
                    v_tau=env_config.v_tau,
                    u_max=env_config.u_max,
                    r_car=env_config.r_car,
                    r_goal=env_config.r_goal,
                    T_max=int(env_config.T_max * 1.5),
                    w_dist=env_config.w_dist,
                    w_ctrl=env_config.w_ctrl,
                    collision_penalty=env_config.collision_penalty
                )
                shifted_eval = RolloutEvaluator(shifted_env, map_config, eval_config_obj)
                summary, _ = shifted_eval.evaluate_oracle(seed=seed, verbose=False)
            oracle_results[condition] = summary

        method_results['oracle'] = oracle_results
        results['oracle'] = {k: v.to_dict() for k, v in oracle_results.items()}

        if verbose:
            print(f"  In-dist success: {oracle_results['in_dist'].success_rate:.3f}")

    # ========== C1: Simple DAgger (Human Labels) ==========
    if 'dagger_human' in methods_to_run:
        if verbose:
            print("\n--- Training Simple DAgger (C1) ---")

        dagger_config = DAggerConfig(
            n_iterations=dag_cfg.get('n_iterations', 5),
            m_rollouts_per_iter=dag_cfg.get('m_rollouts_per_iter', 10),
            k_init=dag_cfg.get('k_init', 4),
            beta_schedule=dag_cfg.get('beta_schedule', 'linear_decay'),
            beta_init=dag_cfg.get('beta_init', 1.0),
            beta_final=dag_cfg.get('beta_final', 0.0),
            label_source='human'
        )

        dagger = DAgger(dagger_config, policy_config, env_config, map_config, seed)
        policy_human = dagger.run(train_episodes, teachers, verbose=verbose)

        trained_policies['dagger_human'] = policy_human

        # Evaluate
        if verbose:
            print("  Evaluating...")
        human_results = evaluator.evaluate_all(policy_human, seed=seed, verbose=False)
        method_results['dagger_human'] = {k: v[0] for k, v in human_results.items()}
        results['dagger_human'] = {k: v[0].to_dict() for k, v in human_results.items()}

        if verbose:
            print(f"  In-dist success: {method_results['dagger_human']['in_dist'].success_rate:.3f}")

    # ========== Train correction models ==========
    # MIND MELD
    mindmeld = None
    if 'dagger_mm' in methods_to_run:
        if verbose:
            print("\n--- Training MIND MELD ---")

        mm_config = MindMeldConfig(
            embedding_dim=mm_cfg.get('embedding_dim', 8),
            context_window=mm_cfg.get('context_window', 5),
            lstm_hidden=mm_cfg.get('lstm_hidden', 32),
            use_bidirectional=mm_cfg.get('use_bidirectional', True),
            use_state_features=mm_cfg.get('use_state_features', True),
            mode=mm_cfg.get('mode', 'learned_embedding'),
            mi_weight=mm_cfg.get('mi_weight', 0.1),
            lr=mm_cfg.get('lr', 1e-3),
            epochs=mm_cfg.get('epochs', 50),
            early_stopping_patience=mm_cfg.get('early_stopping_patience', 10),
            n_demonstrators=len(population)
        )
        mindmeld = MindMeld(mm_config, seed)
        mindmeld.fit(calib_by_demo, verbose=verbose)

    # Cognitive Model
    cognitive = None
    if 'dagger_cog' in methods_to_run:
        if verbose:
            print("\n--- Fitting Cognitive Model ---")

        cog_config = CognitiveModelConfig(
            infer_tau=cog_cfg.get('infer_tau', True),
            infer_g=cog_cfg.get('infer_g', True),
            infer_delta=cog_cfg.get('infer_delta', True),
            infer_sigma=cog_cfg.get('infer_sigma', True),
            correction_mode=cog_cfg.get('correction_mode', 'full'),
            smooth_correction=cog_cfg.get('smooth_correction', True)
        )
        cognitive = CognitiveModel(cog_config)
        cognitive.fit(calib_by_demo, verbose=verbose)

    # ========== C2: DAgger + MIND MELD ==========
    if 'dagger_mm' in methods_to_run and mindmeld is not None:
        if verbose:
            print("\n--- Training DAgger + MIND MELD (C2) ---")

        dagger_mm_config = DAggerConfig(
            n_iterations=dag_cfg.get('n_iterations', 5),
            m_rollouts_per_iter=dag_cfg.get('m_rollouts_per_iter', 10),
            k_init=dag_cfg.get('k_init', 4),
            beta_schedule=dag_cfg.get('beta_schedule', 'linear_decay'),
            beta_init=dag_cfg.get('beta_init', 1.0),
            beta_final=dag_cfg.get('beta_final', 0.0),
            label_source='corrected'
        )

        dagger_mm = DAgger(dagger_mm_config, policy_config, env_config, map_config, seed + 100)
        dagger_mm.set_label_corrector(mindmeld.create_corrector(env_config.u_max))
        policy_mm = dagger_mm.run(train_episodes, teachers, verbose=verbose)

        trained_policies['dagger_mm'] = policy_mm

        # Evaluate
        if verbose:
            print("  Evaluating...")
        mm_results = evaluator.evaluate_all(policy_mm, seed=seed, verbose=False)
        method_results['dagger_mm'] = {k: v[0] for k, v in mm_results.items()}
        results['dagger_mm'] = {k: v[0].to_dict() for k, v in mm_results.items()}

        if verbose:
            print(f"  In-dist success: {method_results['dagger_mm']['in_dist'].success_rate:.3f}")

    # ========== C3: DAgger + Cognitive ==========
    if 'dagger_cog' in methods_to_run and cognitive is not None:
        if verbose:
            print("\n--- Training DAgger + Cognitive (C3) ---")

        dagger_cog_config = DAggerConfig(
            n_iterations=dag_cfg.get('n_iterations', 5),
            m_rollouts_per_iter=dag_cfg.get('m_rollouts_per_iter', 10),
            k_init=dag_cfg.get('k_init', 4),
            beta_schedule=dag_cfg.get('beta_schedule', 'linear_decay'),
            beta_init=dag_cfg.get('beta_init', 1.0),
            beta_final=dag_cfg.get('beta_final', 0.0),
            label_source='corrected'
        )

        dagger_cog = DAgger(dagger_cog_config, policy_config, env_config, map_config, seed + 200)
        dagger_cog.set_label_corrector(cognitive.create_corrector(env_config.u_max))
        policy_cog = dagger_cog.run(train_episodes, teachers, verbose=verbose)

        trained_policies['dagger_cog'] = policy_cog

        # Evaluate
        if verbose:
            print("  Evaluating...")
        cog_results = evaluator.evaluate_all(policy_cog, seed=seed, verbose=False)
        method_results['dagger_cog'] = {k: v[0] for k, v in cog_results.items()}
        results['dagger_cog'] = {k: v[0].to_dict() for k, v in cog_results.items()}

        if verbose:
            print(f"  In-dist success: {method_results['dagger_cog']['in_dist'].success_rate:.3f}")

    # ========== Compute label metrics ==========
    if verbose:
        print("\n--- Computing Label Metrics ---")

    label_metrics = compute_label_correction_metrics(
        datasets, calib_by_demo, mindmeld, cognitive, teachers
    )
    results['label_metrics'] = label_metrics

    # ========== Save models ==========
    if results_dir:
        for name, policy in trained_policies.items():
            policy.save(str(results_dir / 'models' / f'{name}_seed{seed}.pt'))

        if mindmeld is not None:
            mindmeld.save(str(results_dir / 'models' / f'mindmeld_seed{seed}.pt'))

        if cognitive is not None:
            cognitive.save(str(results_dir / 'models' / f'cognitive_seed{seed}.pkl'))

    return results


def compute_label_correction_metrics(datasets: Dict[str, List[EpisodeData]],
                                      calib_by_demo: Dict[int, List[EpisodeData]],
                                      mindmeld: Optional[MindMeld],
                                      cognitive: Optional[CognitiveModel],
                                      teachers: Dict[int, SyntheticTeacher]) -> Dict[str, float]:
    """Compute offline label correction metrics."""
    metrics = {}

    all_human = []
    all_oracle = []
    all_mm_corrected = []
    all_cog_corrected = []

    for demo_id, episodes in calib_by_demo.items():
        for ep in episodes:
            all_human.extend(ep.human_actions)
            all_oracle.extend(ep.oracle_actions)

            if mindmeld is not None:
                corrected = mindmeld.correct(demo_id, ep.human_actions, ep.features)
                all_mm_corrected.extend(corrected)

            if cognitive is not None:
                corrected = cognitive.correct(demo_id, ep.human_actions, ep.features)
                all_cog_corrected.extend(corrected)

    # Convert to arrays
    all_human = np.array(all_human)
    all_oracle = np.array(all_oracle)

    metrics['mse_raw'] = float(np.mean((all_human - all_oracle)**2))
    metrics['mae_raw'] = float(np.mean(np.abs(all_human - all_oracle)))

    if all_mm_corrected:
        all_mm_corrected = np.array(all_mm_corrected)
        metrics['mse_mm'] = float(np.mean((all_mm_corrected - all_oracle)**2))
        metrics['mae_mm'] = float(np.mean(np.abs(all_mm_corrected - all_oracle)))
        metrics['mse_improvement_mm'] = (metrics['mse_raw'] - metrics['mse_mm']) / (metrics['mse_raw'] + 1e-8)

    if all_cog_corrected:
        all_cog_corrected = np.array(all_cog_corrected)
        metrics['mse_cog'] = float(np.mean((all_cog_corrected - all_oracle)**2))
        metrics['mae_cog'] = float(np.mean(np.abs(all_cog_corrected - all_oracle)))
        metrics['mse_improvement_cog'] = (metrics['mse_raw'] - metrics['mse_cog']) / (metrics['mse_raw'] + 1e-8)

    return metrics


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across seeds."""
    aggregated = {}

    # Get methods and conditions
    methods = set()
    conditions = set()
    for result in all_results:
        for method in result.keys():
            if method not in ['seed', 'label_metrics']:
                methods.add(method)
                if isinstance(result[method], dict):
                    conditions.update(result[method].keys())

    # Aggregate per method and condition
    for method in methods:
        aggregated[method] = {}
        for condition in conditions:
            values = {}
            for result in all_results:
                if method in result and isinstance(result[method], dict) and condition in result[method]:
                    for metric, value in result[method][condition].items():
                        if isinstance(value, (int, float)):
                            if metric not in values:
                                values[metric] = []
                            values[metric].append(value)

            # Compute mean and std
            aggregated[method][condition] = {
                f'{m}_mean': float(np.mean(v)) for m, v in values.items()
            }
            aggregated[method][condition].update({
                f'{m}_std': float(np.std(v)) for m, v in values.items()
            })

    # Aggregate label metrics
    label_metrics = {}
    for result in all_results:
        if 'label_metrics' in result:
            for metric, value in result['label_metrics'].items():
                if metric not in label_metrics:
                    label_metrics[metric] = []
                label_metrics[metric].append(value)

    aggregated['label_metrics'] = {
        f'{m}_mean': float(np.mean(v)) for m, v in label_metrics.items()
    }
    aggregated['label_metrics'].update({
        f'{m}_std': float(np.std(v)) for m, v in label_metrics.items()
    })

    return aggregated


def generate_plots(results_dir: Path, all_results: List[Dict[str, Any]],
                   aggregated: Dict[str, Any]):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plots_dir = results_dir / 'plots'

    # 1. Success rate comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['oracle', 'dagger_human', 'dagger_mm', 'dagger_cog']
    conditions = ['in_dist', 'shift_a', 'shift_b', 'shift_c']
    x = np.arange(len(conditions))
    width = 0.2

    for i, method in enumerate(methods):
        if method in aggregated:
            means = []
            stds = []
            for cond in conditions:
                if cond in aggregated[method]:
                    means.append(aggregated[method][cond].get('success_rate_mean', 0))
                    stds.append(aggregated[method][cond].get('success_rate_std', 0))
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i * width, means, width, yerr=stds, label=method, capsize=3)

    ax.set_xlabel('Test Condition')
    ax.set_ylabel('Success Rate')
    ax.set_title('Policy Performance by Test Condition')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(['In-Dist', 'Shift A', 'Shift B', 'Shift C'])
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(plots_dir / 'success_rate_comparison.png', dpi=150)
    plt.close()

    # 2. Label correction metrics
    if 'label_metrics' in aggregated:
        fig, ax = plt.subplots(figsize=(8, 5))

        metrics = ['mse_raw', 'mse_mm', 'mse_cog']
        labels = ['Raw Human', 'MIND MELD', 'Cognitive']
        values = []
        errors = []

        for m in metrics:
            values.append(aggregated['label_metrics'].get(f'{m}_mean', 0))
            errors.append(aggregated['label_metrics'].get(f'{m}_std', 0))

        ax.bar(labels, values, yerr=errors, capsize=5, color=['red', 'blue', 'green'])
        ax.set_ylabel('MSE to Oracle')
        ax.set_title('Label Correction Quality')

        plt.tight_layout()
        plt.savefig(plots_dir / 'label_correction_mse.png', dpi=150)
        plt.close()

    print(f"Plots saved to {plots_dir}")


def save_results_csv(results_dir: Path, all_results: List[Dict[str, Any]],
                     aggregated: Dict[str, Any]):
    """Save results as CSV files."""
    # Detailed results per seed
    rows = []
    for result in all_results:
        seed = result['seed']
        for method in ['oracle', 'dagger_human', 'dagger_mm', 'dagger_cog']:
            if method in result:
                for condition, metrics in result[method].items():
                    row = {'seed': seed, 'method': method, 'condition': condition}
                    row.update(metrics)
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / 'detailed_results.csv', index=False)

    # Summary table
    summary_rows = []
    for method in ['oracle', 'dagger_human', 'dagger_mm', 'dagger_cog']:
        if method in aggregated:
            for condition in ['in_dist', 'shift_a', 'shift_b', 'shift_c']:
                if condition in aggregated[method]:
                    row = {
                        'method': method,
                        'condition': condition,
                        'success_rate': f"{aggregated[method][condition].get('success_rate_mean', 0):.3f} ± {aggregated[method][condition].get('success_rate_std', 0):.3f}",
                        'collision_rate': f"{aggregated[method][condition].get('collision_rate_mean', 0):.3f} ± {aggregated[method][condition].get('collision_rate_std', 0):.3f}"
                    }
                    summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / 'summary_table.csv', index=False)

    print(f"Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Forward-Model Guided Demonstration Learning Experiments'
    )
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--mode', type=str, choices=['quick', 'full'], default=None,
                        help='Override experiment mode')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed to run (overrides n_seeds)')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Run specific ablation study')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config, args.mode)

    # Create results directory
    results_dir = create_results_dir(config)
    print(f"Results will be saved to {results_dir}")

    # Save configuration
    with open(results_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    else:
        base_seed = config.get('experiment', {}).get('seed', 42)
        n_seeds = config.get('experiment', {}).get('n_seeds', 3)
        seeds = [base_seed + i * 1000 for i in range(n_seeds)]

    print(f"Running with seeds: {seeds}")

    # Run experiments
    start_time = time.time()
    all_results = []

    for seed in seeds:
        result = run_single_experiment(config, seed, results_dir, verbose=args.verbose)
        all_results.append(result)

        # Save intermediate results
        with open(results_dir / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    # Aggregate results
    aggregated = aggregate_results(all_results)

    with open(results_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Generate plots and CSV
    if not args.no_plots:
        generate_plots(results_dir, all_results, aggregated)

    save_results_csv(results_dir, all_results, aggregated)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")

    print("\nSummary (In-Distribution Success Rates):")
    for method in ['oracle', 'dagger_human', 'dagger_mm', 'dagger_cog']:
        if method in aggregated and 'in_dist' in aggregated[method]:
            mean = aggregated[method]['in_dist'].get('success_rate_mean', 0)
            std = aggregated[method]['in_dist'].get('success_rate_std', 0)
            print(f"  {method:15s}: {mean:.3f} ± {std:.3f}")


if __name__ == '__main__':
    main()
