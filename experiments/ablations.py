"""
Ablation Studies Module.

Implements ablation runners and hypothesis testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from scipy import stats
import itertools

from methods.mindmeld import MindMeld, MindMeldConfig, MindMeldNonPersonalized
from methods.cognitive_model import CognitiveModel, CognitiveModelConfig, CognitiveModelAblation
from eval.metrics import MetricsSummary, compute_statistical_comparison


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    parameter: str
    values: List[Any]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'parameter': self.parameter,
            'values': self.values,
            'description': self.description
        }


class AblationRunner:
    """
    Runs ablation studies across different configurations.
    """

    def __init__(self, base_config: Dict[str, Any],
                 results_dir: str = 'results/ablations'):
        self.base_config = base_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, pd.DataFrame] = {}

    def run_ablation(self, ablation: AblationConfig,
                     run_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
                     seeds: List[int] = [42],
                     verbose: bool = True) -> pd.DataFrame:
        """
        Run a single ablation study.

        Args:
            ablation: AblationConfig defining what to vary
            run_fn: Function that takes config and returns results dict
            seeds: List of random seeds for repetitions
            verbose: Print progress

        Returns:
            DataFrame with results for each configuration
        """
        results_list = []

        total = len(ablation.values) * len(seeds)
        completed = 0

        for value in ablation.values:
            for seed in seeds:
                # Create modified config
                config = self.base_config.copy()
                config[ablation.parameter] = value
                config['seed'] = seed

                if verbose:
                    print(f"[{completed + 1}/{total}] Running {ablation.name}: "
                          f"{ablation.parameter}={value}, seed={seed}")

                # Run experiment
                try:
                    result = run_fn(config)
                    result['ablation_value'] = value
                    result['seed'] = seed
                    result['ablation_name'] = ablation.name
                    results_list.append(result)
                except Exception as e:
                    print(f"  Error: {e}")
                    results_list.append({
                        'ablation_value': value,
                        'seed': seed,
                        'ablation_name': ablation.name,
                        'error': str(e)
                    })

                completed += 1

        df = pd.DataFrame(results_list)
        self.results[ablation.name] = df

        # Save to CSV
        df.to_csv(self.results_dir / f'{ablation.name}.csv', index=False)

        return df

    def run_mindmeld_ablations(self, run_fn: Callable,
                               seeds: List[int] = [42, 123, 456],
                               verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run all MIND MELD ablation studies.

        A_MM_1: No personalization
        A_MM_2: No sequence context
        A_MM_3: No state conditioning
        A_MM_4: Embedding dimension
        A_MM_5: Remove MI term
        """
        ablations = [
            AblationConfig(
                name='A_MM_1_personalization',
                parameter='mm_personalized',
                values=[True, False],
                description='Effect of personalized embeddings'
            ),
            AblationConfig(
                name='A_MM_2_context',
                parameter='mm_context_window',
                values=[0, 3, 5, 10],
                description='Effect of sequence context window'
            ),
            AblationConfig(
                name='A_MM_3_state',
                parameter='mm_use_state_features',
                values=[True, False],
                description='Effect of state conditioning'
            ),
            AblationConfig(
                name='A_MM_4_embedding_dim',
                parameter='mm_embedding_dim',
                values=[2, 4, 8, 16],
                description='Effect of embedding dimension'
            ),
            AblationConfig(
                name='A_MM_5_mi_term',
                parameter='mm_mi_weight',
                values=[0.0, 0.01, 0.1, 0.5],
                description='Effect of MI regularization'
            )
        ]

        results = {}
        for ablation in ablations:
            if verbose:
                print(f"\n=== Ablation: {ablation.name} ===")
            df = self.run_ablation(ablation, run_fn, seeds, verbose)
            results[ablation.name] = df

        return results

    def run_cognitive_ablations(self, run_fn: Callable,
                                seeds: List[int] = [42, 123, 456],
                                verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run all Cognitive Model ablation studies.

        A_Cog_1: Remove de-lag
        A_Cog_2: Remove gain correction
        A_Cog_3: Remove deadzone modeling
        A_Cog_4: Noise estimation
        A_Cog_5: Correction mode
        """
        ablations = [
            AblationConfig(
                name='A_Cog_1_delag',
                parameter='cog_infer_tau',
                values=[True, False],
                description='Effect of timing correction'
            ),
            AblationConfig(
                name='A_Cog_2_gain',
                parameter='cog_infer_g',
                values=[True, False],
                description='Effect of gain correction'
            ),
            AblationConfig(
                name='A_Cog_3_deadzone',
                parameter='cog_infer_delta',
                values=[True, False],
                description='Effect of deadzone modeling'
            ),
            AblationConfig(
                name='A_Cog_4_noise',
                parameter='cog_infer_sigma',
                values=[True, False],
                description='Effect of noise estimation'
            ),
            AblationConfig(
                name='A_Cog_5_correction_mode',
                parameter='cog_correction_mode',
                values=['full', 'rescale_only', 'delag_only'],
                description='Comparison of correction modes'
            )
        ]

        results = {}
        for ablation in ablations:
            if verbose:
                print(f"\n=== Ablation: {ablation.name} ===")
            df = self.run_ablation(ablation, run_fn, seeds, verbose)
            results[ablation.name] = df

        return results

    def run_protocol_ablations(self, run_fn: Callable,
                               seeds: List[int] = [42, 123, 456],
                               verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run protocol ablation studies.

        A_P_1: Calibration size
        A_P_2: Teacher noise
        A_P_3: Timing bias
        A_P_4: Dynamics shift
        A_P_5: Data budget
        """
        ablations = [
            AblationConfig(
                name='A_P_1_calib_size',
                parameter='k_calib',
                values=[1, 2, 4, 8, 16],
                description='Effect of calibration data size'
            ),
            AblationConfig(
                name='A_P_2_noise',
                parameter='sigma_max',
                values=[0.05, 0.1, 0.2, 0.3],
                description='Effect of teacher noise level'
            ),
            AblationConfig(
                name='A_P_3_timing',
                parameter='tau_max',
                values=[0, 2, 5, 10],
                description='Effect of timing bias range'
            ),
            AblationConfig(
                name='A_P_4_dynamics_shift',
                parameter='dynamics_shift_level',
                values=[0.0, 0.25, 0.5, 0.75],
                description='Effect of dynamics shift severity'
            ),
            AblationConfig(
                name='A_P_5_data_budget',
                parameter='m_rollouts_per_iter',
                values=[5, 10, 20, 40],
                description='Effect of data collection budget'
            )
        ]

        results = {}
        for ablation in ablations:
            if verbose:
                print(f"\n=== Ablation: {ablation.name} ===")
            df = self.run_ablation(ablation, run_fn, seeds, verbose)
            results[ablation.name] = df

        return results

    def aggregate_results(self, df: pd.DataFrame,
                          group_by: str = 'ablation_value',
                          metrics: List[str] = None) -> pd.DataFrame:
        """
        Aggregate results across seeds.

        Returns DataFrame with mean ± std for each configuration.
        """
        if metrics is None:
            # Auto-detect numeric columns
            metrics = [col for col in df.columns
                      if df[col].dtype in ['float64', 'float32', 'int64', 'int32']
                      and col not in ['seed', 'ablation_value']]

        agg_dict = {}
        for metric in metrics:
            agg_dict[f'{metric}_mean'] = (metric, 'mean')
            agg_dict[f'{metric}_std'] = (metric, 'std')

        aggregated = df.groupby(group_by).agg(**agg_dict).reset_index()
        return aggregated

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table of all ablation results."""
        rows = []

        for ablation_name, df in self.results.items():
            if 'success_rate' in df.columns:
                agg = self.aggregate_results(df)

                for _, row in agg.iterrows():
                    rows.append({
                        'ablation': ablation_name,
                        'value': row['ablation_value'],
                        'success_rate': f"{row.get('success_rate_mean', 0):.3f} ± {row.get('success_rate_std', 0):.3f}",
                        'collision_rate': f"{row.get('collision_rate_mean', 0):.3f} ± {row.get('collision_rate_std', 0):.3f}"
                    })

        return pd.DataFrame(rows)


class HypothesisTester:
    """
    Tests pre-defined hypotheses about experimental results.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results: Dict[str, Dict[str, Any]] = {}

    def test_h1_correction_reduces_mse(self, raw_mse: List[float],
                                       corrected_mse: List[float]) -> Dict[str, Any]:
        """
        H1: Corrected labels reduce MSE vs raw human labels.
        """
        result = compute_statistical_comparison(raw_mse, corrected_mse, 'ttest')
        result['hypothesis'] = 'H1: Corrected labels reduce MSE'
        result['supported'] = (result['p_value'] < self.alpha and
                              np.mean(raw_mse) > np.mean(corrected_mse))
        result['raw_mean'] = float(np.mean(raw_mse))
        result['corrected_mean'] = float(np.mean(corrected_mse))

        self.results['H1'] = result
        return result

    def test_h2_cognitive_vs_mindmeld(self, cog_success: List[float],
                                       mm_success: List[float],
                                       condition: str = 'shift') -> Dict[str, Any]:
        """
        H2: Cognitive model success rate > MIND MELD under shifts.
        """
        result = compute_statistical_comparison(cog_success, mm_success, 'ttest')
        result['hypothesis'] = f'H2: Cognitive > MIND MELD on {condition}'
        result['supported'] = (result['p_value'] < self.alpha and
                              np.mean(cog_success) > np.mean(mm_success))
        result['cog_mean'] = float(np.mean(cog_success))
        result['mm_mean'] = float(np.mean(mm_success))

        self.results['H2'] = result
        return result

    def test_h3_calibration_size_effect(self, results_by_size: Dict[int, List[float]]) -> Dict[str, Any]:
        """
        H3: Performance gap increases as calibration size decreases.
        """
        sizes = sorted(results_by_size.keys())
        means = [np.mean(results_by_size[s]) for s in sizes]

        # Test for monotonic increase (smaller size = worse performance)
        # Spearman correlation between size and performance
        rho, p_value = stats.spearmanr(sizes, means)

        result = {
            'hypothesis': 'H3: Performance gap increases with smaller calibration',
            'correlation': float(rho),
            'p_value': float(p_value),
            'supported': p_value < self.alpha and rho > 0,  # Positive = more data = better
            'sizes': sizes,
            'means': means
        }

        self.results['H3'] = result
        return result

    def test_h4_embedding_correlation(self, cognitive_params: np.ndarray,
                                       mm_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        H4: Cognitive latent estimates correlate with MIND MELD embedding axes.
        """
        from sklearn.decomposition import PCA

        # PCA on embeddings
        pca = PCA(n_components=min(2, mm_embeddings.shape[1]))
        embedding_pca = pca.fit_transform(mm_embeddings)

        correlations = []

        # Correlate each cognitive param with first 2 PCA axes
        param_names = ['tau', 'g', 'delta', 'sigma']

        for i, param_name in enumerate(param_names[:cognitive_params.shape[1]]):
            param_vals = cognitive_params[:, i]

            for j in range(embedding_pca.shape[1]):
                pca_vals = embedding_pca[:, j]

                r, p = stats.pearsonr(param_vals, pca_vals)
                rho, p_spearman = stats.spearmanr(param_vals, pca_vals)

                correlations.append({
                    'param': param_name,
                    'pca_axis': j,
                    'pearson_r': float(r),
                    'pearson_p': float(p),
                    'spearman_rho': float(rho),
                    'spearman_p': float(p_spearman)
                })

        # Find best correlation for each param
        best_correlations = {}
        for param_name in param_names[:cognitive_params.shape[1]]:
            param_corrs = [c for c in correlations if c['param'] == param_name]
            best = max(param_corrs, key=lambda x: abs(x['pearson_r']))
            best_correlations[param_name] = best

        result = {
            'hypothesis': 'H4: Cognitive params correlate with MM embedding axes',
            'correlations': correlations,
            'best_correlations': best_correlations,
            'supported': any(abs(c['pearson_r']) > 0.3 and c['pearson_p'] < self.alpha
                            for c in correlations)
        }

        self.results['H4'] = result
        return result

    def generate_summary(self) -> str:
        """Generate text summary of hypothesis tests."""
        lines = ["=" * 60, "HYPOTHESIS TEST SUMMARY", "=" * 60, ""]

        for h_name, result in self.results.items():
            status = "✓ SUPPORTED" if result['supported'] else "✗ NOT SUPPORTED"
            lines.append(f"{h_name}: {result['hypothesis']}")
            lines.append(f"  Status: {status}")
            lines.append(f"  p-value: {result.get('p_value', 'N/A')}")
            if 'effect_size' in result:
                lines.append(f"  Effect size: {result['effect_size']:.3f}")
            lines.append("")

        return "\n".join(lines)

    def save_results(self, path: str):
        """Save hypothesis test results to JSON."""
        # Convert numpy values to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        serializable = {}
        for key, value in self.results.items():
            serializable[key] = {k: convert(v) for k, v in value.items()}

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)


def create_ablation_configs() -> Dict[str, List[AblationConfig]]:
    """Create all predefined ablation configurations."""
    return {
        'mindmeld': [
            AblationConfig('A_MM_1', 'mm_personalized', [True, False]),
            AblationConfig('A_MM_2', 'mm_context_window', [0, 3, 5, 10]),
            AblationConfig('A_MM_3', 'mm_use_state_features', [True, False]),
            AblationConfig('A_MM_4', 'mm_embedding_dim', [2, 4, 8, 16]),
            AblationConfig('A_MM_5', 'mm_mi_weight', [0.0, 0.01, 0.1, 0.5])
        ],
        'cognitive': [
            AblationConfig('A_Cog_1', 'cog_infer_tau', [True, False]),
            AblationConfig('A_Cog_2', 'cog_infer_g', [True, False]),
            AblationConfig('A_Cog_3', 'cog_infer_delta', [True, False]),
            AblationConfig('A_Cog_4', 'cog_infer_sigma', [True, False]),
            AblationConfig('A_Cog_5', 'cog_correction_mode', ['full', 'rescale_only', 'delag_only'])
        ],
        'protocol': [
            AblationConfig('A_P_1', 'k_calib', [1, 2, 4, 8, 16]),
            AblationConfig('A_P_2', 'sigma_max', [0.05, 0.1, 0.2, 0.3]),
            AblationConfig('A_P_3', 'tau_max', [0, 2, 5, 10]),
            AblationConfig('A_P_4', 'dynamics_shift_level', [0.0, 0.25, 0.5, 0.75]),
            AblationConfig('A_P_5', 'm_rollouts_per_iter', [5, 10, 20, 40])
        ]
    }


def run_grid_search(base_config: Dict[str, Any],
                    param_grid: Dict[str, List[Any]],
                    run_fn: Callable,
                    seeds: List[int] = [42],
                    verbose: bool = True) -> pd.DataFrame:
    """
    Run grid search over parameter combinations.

    Args:
        base_config: Base configuration
        param_grid: Dict of parameter names to lists of values
        run_fn: Function that takes config and returns results
        seeds: Random seeds
        verbose: Print progress

    Returns:
        DataFrame with results for all combinations
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    results_list = []
    total = len(combinations) * len(seeds)
    completed = 0

    for combo in combinations:
        for seed in seeds:
            config = base_config.copy()
            for name, value in zip(param_names, combo):
                config[name] = value
            config['seed'] = seed

            if verbose:
                params_str = ", ".join(f"{n}={v}" for n, v in zip(param_names, combo))
                print(f"[{completed + 1}/{total}] {params_str}, seed={seed}")

            try:
                result = run_fn(config)
                for name, value in zip(param_names, combo):
                    result[name] = value
                result['seed'] = seed
                results_list.append(result)
            except Exception as e:
                print(f"  Error: {e}")

            completed += 1

    return pd.DataFrame(results_list)
