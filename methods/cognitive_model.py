"""
Cognitive Model Implementation.

Implements explicit cognitive parameter inference and label correction.
Models human biases (timing offset, gain, deadzone, noise) and inverts them.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate
from scipy.interpolate import interp1d
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import copy

from data.generate_dataset import EpisodeData
from humans.synthetic_teacher import compute_dtw_lag


@dataclass
class CognitiveModelConfig:
    """Configuration for cognitive model."""
    # Which parameters to infer
    infer_tau: bool = True       # Temporal offset
    infer_g: bool = True         # Gain
    infer_delta: bool = True     # Deadzone
    infer_sigma: bool = True     # Noise

    # Parameter bounds
    tau_range: Tuple[float, float] = (-10.0, 10.0)
    g_range: Tuple[float, float] = (0.3, 2.0)
    delta_range: Tuple[float, float] = (0.0, 0.3)
    sigma_range: Tuple[float, float] = (0.0, 0.5)

    # Inference settings
    max_lag_search: int = 15     # Max lag for cross-correlation
    use_dtw: bool = False        # Use DTW for lag estimation (slower but more robust)
    robust_regression: bool = True  # Use robust regression for gain

    # Correction settings
    correction_mode: str = 'full'  # 'full', 'rescale_only', 'delag_only'
    smooth_correction: bool = True  # Apply smoothing to corrected labels
    smooth_window: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'infer_tau': self.infer_tau,
            'infer_g': self.infer_g,
            'infer_delta': self.infer_delta,
            'infer_sigma': self.infer_sigma,
            'tau_range': self.tau_range,
            'g_range': self.g_range,
            'delta_range': self.delta_range,
            'sigma_range': self.sigma_range,
            'max_lag_search': self.max_lag_search,
            'use_dtw': self.use_dtw,
            'robust_regression': self.robust_regression,
            'correction_mode': self.correction_mode,
            'smooth_correction': self.smooth_correction,
            'smooth_window': self.smooth_window
        }


@dataclass
class InferredParams:
    """Inferred cognitive parameters for a demonstrator."""
    tau_hat: float = 0.0      # Estimated temporal offset
    g_hat: float = 1.0        # Estimated gain
    delta_hat: float = 0.0    # Estimated deadzone
    sigma_hat: float = 0.0    # Estimated noise

    # Confidence/quality metrics
    tau_confidence: float = 1.0
    g_r_squared: float = 1.0
    fit_mse: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            'tau_hat': self.tau_hat,
            'g_hat': self.g_hat,
            'delta_hat': self.delta_hat,
            'sigma_hat': self.sigma_hat,
            'tau_confidence': self.tau_confidence,
            'g_r_squared': self.g_r_squared,
            'fit_mse': self.fit_mse
        }

    def to_array(self) -> np.ndarray:
        return np.array([self.tau_hat, self.g_hat, self.delta_hat, self.sigma_hat],
                       dtype=np.float32)


class CognitiveModel:
    """
    Cognitive model for inferring human biases and correcting labels.

    Infers per-demonstrator parameters: tau (timing), g (gain), delta (deadzone), sigma (noise).
    Then inverts these transformations to correct human labels toward oracle labels.
    """

    def __init__(self, config: CognitiveModelConfig):
        self.config = config
        self.inferred_params: Dict[int, InferredParams] = {}

    def fit(self, calibration_data: Dict[int, List[EpisodeData]],
            verbose: bool = False) -> Dict[int, InferredParams]:
        """
        Fit cognitive model on calibration data.

        Args:
            calibration_data: Dict mapping demo_id to list of calibration episodes
            verbose: Print progress

        Returns:
            Dict mapping demo_id to InferredParams
        """
        self.inferred_params = {}

        for demo_id, episodes in calibration_data.items():
            if verbose:
                print(f"Inferring parameters for demonstrator {demo_id}...")

            params = self._infer_single_demonstrator(episodes)
            self.inferred_params[demo_id] = params

            if verbose:
                print(f"  tau={params.tau_hat:.2f}, g={params.g_hat:.2f}, "
                      f"delta={params.delta_hat:.3f}, sigma={params.sigma_hat:.3f}")

        return self.inferred_params

    def _infer_single_demonstrator(self, episodes: List[EpisodeData]) -> InferredParams:
        """Infer parameters for a single demonstrator from their episodes."""
        # Concatenate all episode data
        all_human = []
        all_oracle = []

        for ep in episodes:
            all_human.append(ep.human_actions)
            all_oracle.append(ep.oracle_actions)

        human_concat = np.concatenate(all_human)
        oracle_concat = np.concatenate(all_oracle)

        # 1. Estimate tau (temporal offset)
        tau_hat, tau_conf = self._estimate_tau(human_concat, oracle_concat)

        # 2. Time-shift oracle for gain estimation
        oracle_shifted = self._shift_sequence(oracle_concat, tau_hat)

        # 3. Estimate delta (deadzone) first
        delta_hat = self._estimate_delta(human_concat, oracle_shifted)

        # 4. Estimate g (gain) using only non-deadzone samples
        g_hat, g_r2 = self._estimate_gain(human_concat, oracle_shifted, delta_hat)

        # 5. Estimate sigma (noise) from residuals
        sigma_hat, fit_mse = self._estimate_sigma(human_concat, oracle_shifted, g_hat, delta_hat)

        return InferredParams(
            tau_hat=tau_hat,
            g_hat=g_hat,
            delta_hat=delta_hat,
            sigma_hat=sigma_hat,
            tau_confidence=tau_conf,
            g_r_squared=g_r2,
            fit_mse=fit_mse
        )

    def _estimate_tau(self, human: np.ndarray, oracle: np.ndarray) -> Tuple[float, float]:
        """
        Estimate temporal offset using cross-correlation.

        Returns:
            tau_hat: Estimated offset (positive = human lags)
            confidence: Confidence in estimate (peak correlation value)
        """
        if not self.config.infer_tau:
            return 0.0, 1.0

        if self.config.use_dtw:
            tau_hat = compute_dtw_lag(human, oracle, window=self.config.max_lag_search)
            return tau_hat, 1.0

        # Cross-correlation method
        if len(human) < 5 or len(oracle) < 5:
            return 0.0, 0.0

        # Normalize
        h = human - np.mean(human)
        o = oracle - np.mean(oracle)
        h_std = np.std(h) + 1e-8
        o_std = np.std(o) + 1e-8
        h = h / h_std
        o = o / o_std

        # Cross-correlation
        correlation = correlate(h, o, mode='full')
        lags = np.arange(-len(o) + 1, len(h))

        # Find peak within max_lag
        max_lag = self.config.max_lag_search
        valid_mask = np.abs(lags) <= max_lag
        if not np.any(valid_mask):
            return 0.0, 0.0

        valid_corr = correlation[valid_mask]
        valid_lags = lags[valid_mask]

        # Normalize correlation
        valid_corr = valid_corr / len(human)

        peak_idx = np.argmax(valid_corr)
        tau_hat = float(valid_lags[peak_idx])
        confidence = float(valid_corr[peak_idx])

        # Clip to range
        tau_hat = np.clip(tau_hat, self.config.tau_range[0], self.config.tau_range[1])

        return tau_hat, confidence

    def _shift_sequence(self, seq: np.ndarray, tau: float) -> np.ndarray:
        """Shift sequence by tau (interpolated for fractional tau)."""
        if abs(tau) < 0.01:
            return seq.copy()

        T = len(seq)
        times = np.arange(T)
        shifted_times = times + tau  # Shift forward in time

        interp_func = interp1d(times, seq, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        shifted = interp_func(shifted_times)

        return shifted.astype(np.float32)

    def _estimate_delta(self, human: np.ndarray, oracle_shifted: np.ndarray) -> float:
        """
        Estimate deadzone threshold.

        Actions below this threshold are suppressed to zero by the human.
        """
        if not self.config.infer_delta:
            return 0.0

        # Find where oracle is non-zero but human is near zero
        oracle_nonzero = np.abs(oracle_shifted) > 0.01
        human_near_zero = np.abs(human) < 0.05

        # Threshold is the max oracle magnitude where human still outputs zero
        suppressions = np.abs(oracle_shifted[oracle_nonzero & human_near_zero])

        if len(suppressions) == 0:
            return 0.0

        # Use percentile to be robust
        delta_hat = float(np.percentile(suppressions, 90))
        delta_hat = np.clip(delta_hat, self.config.delta_range[0], self.config.delta_range[1])

        return delta_hat

    def _estimate_gain(self, human: np.ndarray, oracle_shifted: np.ndarray,
                       delta: float) -> Tuple[float, float]:
        """
        Estimate gain using regression.

        Returns:
            g_hat: Estimated gain
            r_squared: R-squared of fit
        """
        if not self.config.infer_g:
            return 1.0, 1.0

        # Filter out deadzone samples
        mask = np.abs(oracle_shifted) > delta + 0.01
        if np.sum(mask) < 5:
            return 1.0, 0.0

        h = human[mask]
        o = oracle_shifted[mask]

        if self.config.robust_regression:
            # Robust regression using iteratively reweighted least squares
            g_hat = self._robust_regression(o, h)
        else:
            # Simple OLS: h = g * o
            g_hat = float(np.sum(h * o) / (np.sum(o * o) + 1e-8))

        # Clip to range
        g_hat = np.clip(g_hat, self.config.g_range[0], self.config.g_range[1])

        # Compute R-squared
        pred = g_hat * o
        ss_res = np.sum((h - pred)**2)
        ss_tot = np.sum((h - np.mean(h))**2) + 1e-8
        r_squared = max(0, 1 - ss_res / ss_tot)

        return g_hat, r_squared

    def _robust_regression(self, x: np.ndarray, y: np.ndarray,
                          n_iter: int = 5) -> float:
        """Robust regression using Huber weights."""
        # Initial estimate
        g = float(np.sum(y * x) / (np.sum(x * x) + 1e-8))

        for _ in range(n_iter):
            # Compute residuals
            residuals = y - g * x
            mad = np.median(np.abs(residuals)) + 1e-8

            # Huber weights
            k = 1.345 * mad
            weights = np.where(np.abs(residuals) <= k, 1.0,
                              k / (np.abs(residuals) + 1e-8))

            # Weighted regression
            g = float(np.sum(weights * y * x) / (np.sum(weights * x * x) + 1e-8))

        return g

    def _estimate_sigma(self, human: np.ndarray, oracle_shifted: np.ndarray,
                        g: float, delta: float) -> Tuple[float, float]:
        """
        Estimate noise standard deviation from residuals.

        Returns:
            sigma_hat: Estimated noise std
            mse: Mean squared error of prediction
        """
        if not self.config.infer_sigma:
            return 0.0, 0.0

        # Apply model: predicted_human = g * oracle (with deadzone)
        predicted = g * oracle_shifted
        predicted[np.abs(oracle_shifted) < delta] = 0.0

        residuals = human - predicted
        mse = float(np.mean(residuals**2))
        sigma_hat = float(np.std(residuals))

        sigma_hat = np.clip(sigma_hat, self.config.sigma_range[0], self.config.sigma_range[1])

        return sigma_hat, mse

    def correct(self, demo_id: int,
                human_labels: np.ndarray,
                features: Optional[np.ndarray] = None,
                u_max: float = 1.0) -> np.ndarray:
        """
        Correct human labels using inferred parameters.

        Inverts the cognitive model transformation.

        Args:
            demo_id: Demonstrator ID
            human_labels: (T,) human action labels
            features: (T, D) state features (unused in basic model)
            u_max: Max action for clipping

        Returns:
            corrected: (T,) corrected labels
        """
        if demo_id not in self.inferred_params:
            # No calibration for this demonstrator - return unchanged
            return human_labels.copy()

        params = self.inferred_params[demo_id]
        mode = self.config.correction_mode

        # Start with human labels
        corrected = human_labels.copy()

        if mode in ['full', 'rescale_only']:
            # 1. Inverse gain
            if abs(params.g_hat) > 0.1:
                corrected = corrected / params.g_hat

            # 2. Inverse deadzone (approximate)
            # If action is near zero, it might have been in deadzone
            # We can't perfectly recover, but we can leave small actions unchanged
            # More sophisticated: use context to interpolate

        if mode in ['full', 'delag_only']:
            # 3. Inverse time shift (de-lag)
            corrected = self._shift_sequence(corrected, -params.tau_hat)

        # Smooth if configured
        if self.config.smooth_correction:
            corrected = self._smooth(corrected, self.config.smooth_window)

        # Clip to bounds
        corrected = np.clip(corrected, -u_max, u_max)

        return corrected

    def correct_with_uncertainty(self, demo_id: int,
                                 human_labels: np.ndarray,
                                 features: Optional[np.ndarray] = None,
                                 u_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct human labels and estimate uncertainty.

        Returns:
            corrected: (T,) corrected labels
            uncertainty: (T,) uncertainty estimates
        """
        corrected = self.correct(demo_id, human_labels, features, u_max)

        if demo_id in self.inferred_params:
            params = self.inferred_params[demo_id]
            # Uncertainty from estimated noise and fit quality
            base_uncertainty = params.sigma_hat / max(params.g_hat, 0.1)
            uncertainty = np.full_like(corrected, base_uncertainty)

            # Higher uncertainty near deadzone
            near_zero = np.abs(human_labels) < params.delta_hat + 0.02
            uncertainty[near_zero] *= 2.0
        else:
            uncertainty = np.full_like(corrected, 0.1)

        return corrected, uncertainty

    def _smooth(self, seq: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window <= 1:
            return seq
        kernel = np.ones(window) / window
        smoothed = np.convolve(seq, kernel, mode='same')
        return smoothed.astype(np.float32)

    def create_corrector(self, u_max: float = 1.0):
        """
        Create a corrector function for use with DAgger.

        Returns:
            Corrector function
        """
        def corrector(human_labels, features, demo_id, **kwargs):
            return self.correct(demo_id, human_labels, features, u_max)
        return corrector

    def get_params(self, demo_id: int) -> InferredParams:
        """Get inferred parameters for a demonstrator."""
        return self.inferred_params.get(demo_id, InferredParams())

    def get_all_params_array(self) -> np.ndarray:
        """Get all inferred parameters as array."""
        if not self.inferred_params:
            return np.array([])

        max_id = max(self.inferred_params.keys())
        params_array = np.zeros((max_id + 1, 4), dtype=np.float32)

        for demo_id, params in self.inferred_params.items():
            params_array[demo_id] = params.to_array()

        return params_array

    def compute_correction_metrics(self, episodes: List[EpisodeData]) -> Dict[str, float]:
        """
        Compute metrics on label correction quality.

        Args:
            episodes: Episodes to evaluate

        Returns:
            Dict with MSE improvements and other metrics
        """
        mse_before = []
        mse_after = []
        timing_before = []
        timing_after = []

        for ep in episodes:
            demo_id = ep.demonstrator_id
            human = ep.human_actions
            oracle = ep.oracle_actions

            # Before correction
            mse_before.append(np.mean((human - oracle)**2))

            # After correction
            corrected = self.correct(demo_id, human)
            mse_after.append(np.mean((corrected - oracle)**2))

            # Timing metrics
            tau_before = self._estimate_tau(human, oracle)[0]
            tau_after = self._estimate_tau(corrected, oracle)[0]
            timing_before.append(abs(tau_before))
            timing_after.append(abs(tau_after))

        return {
            'mse_before': np.mean(mse_before),
            'mse_after': np.mean(mse_after),
            'mse_improvement': (np.mean(mse_before) - np.mean(mse_after)) / (np.mean(mse_before) + 1e-8),
            'timing_before': np.mean(timing_before),
            'timing_after': np.mean(timing_after),
            'timing_improvement': (np.mean(timing_before) - np.mean(timing_after)) / (np.mean(timing_before) + 1e-8)
        }

    def save(self, path: str):
        """Save model to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config.to_dict(),
                'params': {k: v.to_dict() for k, v in self.inferred_params.items()}
            }, f)

    def load(self, path: str):
        """Load model from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        for key, value in data['config'].items():
            setattr(self.config, key, value)

        self.inferred_params = {}
        for demo_id, params_dict in data['params'].items():
            self.inferred_params[int(demo_id)] = InferredParams(**params_dict)


class CognitiveModelAblation(CognitiveModel):
    """
    Cognitive model variants for ablation studies.
    """

    def __init__(self, config: CognitiveModelConfig, ablation: str = 'none'):
        """
        Args:
            config: Base configuration
            ablation: Ablation type:
                - 'none': Full model
                - 'no_delag': tau fixed to 0
                - 'no_gain': g fixed to 1
                - 'no_deadzone': delta fixed to 0
                - 'no_noise': sigma fixed to 0
        """
        super().__init__(config)
        self.ablation = ablation

        if ablation == 'no_delag':
            self.config.infer_tau = False
        elif ablation == 'no_gain':
            self.config.infer_g = False
        elif ablation == 'no_deadzone':
            self.config.infer_delta = False
        elif ablation == 'no_noise':
            self.config.infer_sigma = False


def create_cognitive_model_from_config(config: Dict[str, Any]) -> CognitiveModel:
    """Create cognitive model from configuration dict."""
    cog_config = CognitiveModelConfig(
        infer_tau=config.get('infer_tau', True),
        infer_g=config.get('infer_g', True),
        infer_delta=config.get('infer_delta', True),
        infer_sigma=config.get('infer_sigma', True),
        tau_range=tuple(config.get('tau_range', [-10.0, 10.0])),
        g_range=tuple(config.get('g_range', [0.3, 2.0])),
        delta_range=tuple(config.get('delta_range', [0.0, 0.3])),
        sigma_range=tuple(config.get('sigma_range', [0.0, 0.5])),
        max_lag_search=config.get('max_lag_search', 15),
        use_dtw=config.get('use_dtw', False),
        robust_regression=config.get('robust_regression', True),
        correction_mode=config.get('correction_mode', 'full'),
        smooth_correction=config.get('smooth_correction', True),
        smooth_window=config.get('smooth_window', 3)
    )
    return CognitiveModel(cog_config)
