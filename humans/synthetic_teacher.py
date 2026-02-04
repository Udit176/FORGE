"""
Synthetic Teacher Model.

Implements cognitive-state-based label corruption to simulate human demonstrators.
Each demonstrator has latent cognitive parameters that transform oracle labels
into realistic human labels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from scipy.interpolate import interp1d
from scipy.signal import correlate


@dataclass
class DemonstratorParams:
    """
    Latent cognitive parameters for a demonstrator.

    tau: Temporal offset in steps (negative=anticipatory, positive=delayed)
    g: Gain multiplier (over/under correction)
    sigma: Motor noise standard deviation
    delta: Deadzone threshold (intervention threshold)
    sat: Saturation limit (tighter than env limits)
    dropout: Probability of missed correction
    """
    tau: float = 0.0      # Temporal offset in steps
    g: float = 1.0        # Gain multiplier
    sigma: float = 0.0    # Motor noise std
    delta: float = 0.0    # Deadzone threshold
    sat: float = 1.0      # Saturation limit
    dropout: float = 0.0  # Dropout probability

    def to_dict(self) -> Dict[str, float]:
        return {
            'tau': self.tau, 'g': self.g, 'sigma': self.sigma,
            'delta': self.delta, 'sat': self.sat, 'dropout': self.dropout
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'DemonstratorParams':
        return cls(**d)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for comparison/embedding."""
        return np.array([self.tau, self.g, self.sigma, self.delta, self.sat, self.dropout],
                       dtype=np.float32)


@dataclass
class DemonstratorPopulationConfig:
    """Configuration for generating a population of demonstrators."""
    n_demonstrators: int = 50
    tau_range: Tuple[float, float] = (-5.0, 5.0)  # Anticipatory to delayed
    g_range: Tuple[float, float] = (0.6, 1.4)     # Under to over correction
    sigma_range: Tuple[float, float] = (0.0, 0.15)
    delta_range: Tuple[float, float] = (0.0, 0.1)
    sat_range: Tuple[float, float] = (0.8, 1.0)   # As fraction of u_max
    dropout_range: Tuple[float, float] = (0.0, 0.1)

    # Correlation structure (optional)
    tau_g_correlation: float = 0.0  # Can induce correlation between timing and gain

    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_demonstrators': self.n_demonstrators,
            'tau_range': self.tau_range,
            'g_range': self.g_range,
            'sigma_range': self.sigma_range,
            'delta_range': self.delta_range,
            'sat_range': self.sat_range,
            'dropout_range': self.dropout_range,
            'tau_g_correlation': self.tau_g_correlation
        }


class SyntheticTeacher:
    """
    Synthetic teacher that transforms oracle labels into human-like labels.

    Implements cognitive biases: temporal offset, gain, deadzone, saturation,
    motor noise, and attention dropout.
    """

    def __init__(self, params: DemonstratorParams, u_max: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize synthetic teacher.

        Args:
            params: Cognitive parameters for this demonstrator
            u_max: Environment's max control bound
            seed: Random seed for reproducibility
        """
        self.params = params
        self.u_max = u_max
        self.rng = np.random.RandomState(seed)

        # Internal state for dropout
        self._last_action = 0.0
        self._step_count = 0

        # Style metrics (computed on calibration)
        self.measured_timing_offset: Optional[float] = None
        self.amplitude_bias: Optional[float] = None

    def set_seed(self, seed: int):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)

    def reset(self):
        """Reset internal state for new episode."""
        self._last_action = 0.0
        self._step_count = 0

    def generate_label(self, oracle_actions: np.ndarray, t: int,
                       return_intermediate: bool = False) -> float:
        """
        Generate human label for timestep t given oracle action sequence.

        Args:
            oracle_actions: Array of oracle actions (should extend to t+H for anticipatory)
            t: Current timestep
            return_intermediate: If True, return dict with intermediate values

        Returns:
            Human action label (or dict if return_intermediate)
        """
        p = self.params

        # 1. Time warp: interpolate oracle at shifted time
        shifted_t = t - p.tau  # Negative tau = anticipatory (look ahead)

        # Create interpolation function
        T = len(oracle_actions)
        times = np.arange(T)

        if T < 2:
            o_shifted = oracle_actions[0] if T > 0 else 0.0
        else:
            # Extrapolate at boundaries
            interp_func = interp1d(times, oracle_actions, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
            o_shifted = float(interp_func(shifted_t))

        # 2. Gain modulation
        u = p.g * o_shifted

        # 3. Deadzone: if |u| < delta, output 0
        if abs(u) < p.delta:
            u = 0.0

        # 4. Saturation: clip to teacher's personal limit
        teacher_sat = p.sat * self.u_max
        u = np.clip(u, -teacher_sat, teacher_sat)

        # 5. Motor noise
        if p.sigma > 0:
            u = u + self.rng.normal(0, p.sigma)

        # 6. Clip to environment bounds
        u = np.clip(u, -self.u_max, self.u_max)

        # 7. Dropout: with probability dropout, return 0 or last action
        if p.dropout > 0 and self.rng.random() < p.dropout:
            u = self._last_action  # Repeat last action instead of 0

        self._last_action = u
        self._step_count += 1

        if return_intermediate:
            return {
                'human_action': float(u),
                'oracle_at_t': oracle_actions[t] if t < len(oracle_actions) else 0.0,
                'shifted_oracle': o_shifted,
                'after_gain': p.g * o_shifted,
                'shift_amount': p.tau
            }

        return float(u)

    def generate_episode_labels(self, oracle_actions: np.ndarray,
                                horizon_padding: int = 10) -> np.ndarray:
        """
        Generate human labels for entire episode.

        Args:
            oracle_actions: (T,) array of oracle actions
            horizon_padding: Extra padding for boundary handling

        Returns:
            human_actions: (T,) array of human labels
        """
        self.reset()
        T = len(oracle_actions)

        # Pad oracle actions for boundary handling
        padded = np.concatenate([
            np.zeros(horizon_padding) + oracle_actions[0],
            oracle_actions,
            np.zeros(horizon_padding) + oracle_actions[-1]
        ])

        human_actions = np.zeros(T, dtype=np.float32)

        for t in range(T):
            # Shift index to account for padding
            human_actions[t] = self.generate_label(padded, t + horizon_padding)

        return human_actions

    def compute_style_metrics(self, human_actions: np.ndarray,
                              oracle_actions: np.ndarray) -> Dict[str, float]:
        """
        Compute style metrics comparing human and oracle actions.

        These metrics can be used for correlation analysis.

        Args:
            human_actions: (T,) array of human actions
            oracle_actions: (T,) array of oracle actions

        Returns:
            Dict with timing_offset, amplitude_bias, etc.
        """
        # Timing offset via cross-correlation
        timing_offset = self._estimate_timing_offset(human_actions, oracle_actions)

        # Amplitude bias: mean(human) / mean(|oracle|) - 1
        oracle_mag = np.mean(np.abs(oracle_actions)) + 1e-6
        human_signed_scale = np.mean(human_actions * np.sign(oracle_actions + 1e-8))
        amplitude_bias = human_signed_scale / oracle_mag - 1.0

        # Action magnitude ratio
        human_mag = np.mean(np.abs(human_actions)) + 1e-6
        magnitude_ratio = human_mag / oracle_mag

        # Correlation
        if len(human_actions) > 1:
            corr = np.corrcoef(human_actions, oracle_actions)[0, 1]
        else:
            corr = 1.0

        # MSE
        mse = np.mean((human_actions - oracle_actions)**2)

        self.measured_timing_offset = timing_offset
        self.amplitude_bias = amplitude_bias

        return {
            'timing_offset': timing_offset,
            'amplitude_bias': amplitude_bias,
            'magnitude_ratio': magnitude_ratio,
            'correlation': corr,
            'mse': mse
        }

    def _estimate_timing_offset(self, human_actions: np.ndarray,
                                oracle_actions: np.ndarray,
                                max_lag: int = 20) -> float:
        """
        Estimate timing offset using cross-correlation.

        Positive offset = human lags behind oracle.
        Negative offset = human anticipates oracle.
        """
        if len(human_actions) < 3 or len(oracle_actions) < 3:
            return 0.0

        # Normalize
        h = human_actions - np.mean(human_actions)
        o = oracle_actions - np.mean(oracle_actions)

        h_std = np.std(h) + 1e-8
        o_std = np.std(o) + 1e-8

        h = h / h_std
        o = o / o_std

        # Cross-correlation
        correlation = correlate(h, o, mode='full')
        lags = np.arange(-len(o) + 1, len(h))

        # Find peak within max_lag
        valid_mask = np.abs(lags) <= max_lag
        if not np.any(valid_mask):
            return 0.0

        valid_corr = correlation[valid_mask]
        valid_lags = lags[valid_mask]

        peak_idx = np.argmax(valid_corr)
        offset = valid_lags[peak_idx]

        return float(offset)


class DemonstratorPopulation:
    """Generate and manage a population of synthetic demonstrators."""

    def __init__(self, config: DemonstratorPopulationConfig,
                 u_max: float = 1.0,
                 seed: Optional[int] = None):
        self.config = config
        self.u_max = u_max
        self.rng = np.random.RandomState(seed)
        self.seed = seed

        self.demonstrators: List[SyntheticTeacher] = []
        self.params_list: List[DemonstratorParams] = []

        self._generate_population()

    def _generate_population(self):
        """Generate population of demonstrators with sampled parameters."""
        cfg = self.config
        n = cfg.n_demonstrators

        # Sample parameters
        taus = self.rng.uniform(cfg.tau_range[0], cfg.tau_range[1], n)
        gs = self.rng.uniform(cfg.g_range[0], cfg.g_range[1], n)
        sigmas = self.rng.uniform(cfg.sigma_range[0], cfg.sigma_range[1], n)
        deltas = self.rng.uniform(cfg.delta_range[0], cfg.delta_range[1], n)
        sats = self.rng.uniform(cfg.sat_range[0], cfg.sat_range[1], n)
        dropouts = self.rng.uniform(cfg.dropout_range[0], cfg.dropout_range[1], n)

        # Optional: induce correlation between tau and g
        if abs(cfg.tau_g_correlation) > 0:
            # Adjust g based on tau
            tau_normalized = (taus - np.mean(taus)) / (np.std(taus) + 1e-8)
            g_mean = np.mean(gs)
            g_std = np.std(gs)
            gs = g_mean + cfg.tau_g_correlation * tau_normalized * g_std + \
                 np.sqrt(1 - cfg.tau_g_correlation**2) * (gs - g_mean)
            gs = np.clip(gs, cfg.g_range[0], cfg.g_range[1])

        # Create demonstrators
        self.demonstrators = []
        self.params_list = []

        for i in range(n):
            params = DemonstratorParams(
                tau=taus[i],
                g=gs[i],
                sigma=sigmas[i],
                delta=deltas[i],
                sat=sats[i],
                dropout=dropouts[i]
            )
            self.params_list.append(params)

            # Each demonstrator gets deterministic seed based on index
            demo_seed = self.seed * 1000 + i if self.seed is not None else None
            teacher = SyntheticTeacher(params, self.u_max, seed=demo_seed)
            self.demonstrators.append(teacher)

    def get_demonstrator(self, idx: int) -> SyntheticTeacher:
        """Get demonstrator by index."""
        return self.demonstrators[idx]

    def get_params(self, idx: int) -> DemonstratorParams:
        """Get parameters for demonstrator."""
        return self.params_list[idx]

    def get_all_params_array(self) -> np.ndarray:
        """Get all parameters as (n_demo, 6) array."""
        return np.array([p.to_array() for p in self.params_list])

    def get_param_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics of population parameters."""
        params_array = self.get_all_params_array()
        param_names = ['tau', 'g', 'sigma', 'delta', 'sat', 'dropout']

        stats = {}
        for i, name in enumerate(param_names):
            stats[name] = {
                'mean': float(np.mean(params_array[:, i])),
                'std': float(np.std(params_array[:, i])),
                'min': float(np.min(params_array[:, i])),
                'max': float(np.max(params_array[:, i]))
            }
        return stats

    def __len__(self):
        return len(self.demonstrators)

    def __iter__(self):
        return iter(self.demonstrators)


def compute_dtw_alignment(seq1: np.ndarray, seq2: np.ndarray,
                          window: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Compute Dynamic Time Warping alignment between two sequences.

    Args:
        seq1: First sequence (T1,)
        seq2: Second sequence (T2,)
        window: Sakoe-Chiba band width (None for full matrix)

    Returns:
        distance: DTW distance
        path: (N, 2) alignment path
    """
    n, m = len(seq1), len(seq2)

    if window is None:
        window = max(n, m)

    # Initialize cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    # Fill matrix
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m + 1, i + window + 1)
        for j in range(j_start, j_end):
            cost = (seq1[i-1] - seq2[j-1])**2
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

    distance = np.sqrt(dtw[n, m])

    # Backtrack to find path
    path = [(n-1, m-1)]
    i, j = n, m
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            argmin = np.argmin([dtw[i-1, j-1], dtw[i-1, j], dtw[i, j-1]])
            if argmin == 0:
                i, j = i-1, j-1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
        path.append((i-1, j-1))

    path.reverse()
    return distance, np.array(path)


def compute_dtw_lag(human_actions: np.ndarray, oracle_actions: np.ndarray,
                    window: int = 20) -> float:
    """
    Compute average lag from DTW alignment.

    Positive lag = human behind oracle.
    """
    _, path = compute_dtw_alignment(human_actions, oracle_actions, window)

    # Average difference in indices
    lags = path[:, 0] - path[:, 1]
    avg_lag = np.mean(lags)

    return float(avg_lag)


def create_teacher_from_config(config: Dict[str, Any],
                               u_max: float = 1.0,
                               seed: Optional[int] = None) -> SyntheticTeacher:
    """Create synthetic teacher from configuration dict."""
    params = DemonstratorParams(
        tau=config.get('tau', 0.0),
        g=config.get('g', 1.0),
        sigma=config.get('sigma', 0.0),
        delta=config.get('delta', 0.0),
        sat=config.get('sat', 1.0),
        dropout=config.get('dropout', 0.0)
    )
    return SyntheticTeacher(params, u_max, seed)


def create_population_from_config(config: Dict[str, Any],
                                  u_max: float = 1.0,
                                  seed: Optional[int] = None) -> DemonstratorPopulation:
    """Create demonstrator population from configuration dict."""
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
    return DemonstratorPopulation(pop_config, u_max, seed)
