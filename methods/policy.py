"""
Policy Module.

Implements MLP regression policy for imitation learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import copy


@dataclass
class PolicyConfig:
    """Configuration for MLP policy."""
    input_dim: int = 12         # Feature dimension
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_dim: int = 1         # Output dimension (steering rate)
    activation: str = 'relu'    # Activation function
    dropout: float = 0.0        # Dropout rate
    batch_norm: bool = False    # Use batch normalization

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 30
    early_stopping_patience: int = 5
    val_fraction: float = 0.1

    # Action bounds
    u_max: float = 1.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'dropout': self.dropout,
            'batch_norm': self.batch_norm,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'val_fraction': self.val_fraction,
            'u_max': self.u_max
        }


class MLPNetwork(nn.Module):
    """Multi-layer perceptron network."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.input_dim

        # Get activation function
        if config.activation == 'relu':
            act_fn = nn.ReLU
        elif config.activation == 'tanh':
            act_fn = nn.Tanh
        elif config.activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU

        # Build hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPPolicy:
    """
    MLP regression policy for imitation learning.

    Supports training with MSE loss and early stopping.
    """

    def __init__(self, config: PolicyConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device('cpu')  # CPU only
        self.model = MLPNetwork(config).to(self.device)
        self.optimizer = None
        self.best_model_state = None
        self.training_history = []

    def train(self, features: np.ndarray, labels: np.ndarray,
              val_features: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None,
              verbose: bool = False) -> Dict[str, List[float]]:
        """
        Train policy on provided data.

        Args:
            features: (N, D) feature array
            labels: (N,) or (N, 1) action labels
            val_features: Optional validation features
            val_labels: Optional validation labels
            verbose: Print training progress

        Returns:
            History dict with train/val losses per epoch
        """
        cfg = self.config

        # Ensure labels are 2D
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        # Create validation set if not provided
        if val_features is None and cfg.val_fraction > 0:
            n = len(features)
            n_val = int(n * cfg.val_fraction)
            indices = np.random.permutation(n)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            val_features = features[val_idx]
            val_labels = labels[val_idx]
            features = features[train_idx]
            labels = labels[train_idx]
        elif val_labels is not None and val_labels.ndim == 1:
            val_labels = val_labels.reshape(-1, 1)

        # Convert to tensors
        X_train = torch.FloatTensor(features).to(self.device)
        y_train = torch.FloatTensor(labels).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

        if val_features is not None:
            X_val = torch.FloatTensor(val_features).to(self.device)
            y_val = torch.FloatTensor(val_labels).to(self.device)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
        else:
            val_loader = None

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(cfg.epochs):
            # Training
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = nn.MSELoss()(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        pred = self.model(batch_x)
                        loss = nn.MSELoss()(pred, batch_y)
                        val_losses.append(loss.item())
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

                if verbose:
                    print(f"Epoch {epoch + 1}/{cfg.epochs}: "
                          f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{cfg.epochs}: train_loss={avg_train_loss:.6f}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.training_history = history
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict action(s) for given features.

        Args:
            features: (D,) or (N, D) feature array

        Returns:
            actions: Scalar or (N,) action array, clipped to bounds
        """
        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            pred = self.model(X).cpu().numpy()

        # Clip to bounds
        pred = np.clip(pred, -self.config.u_max, self.config.u_max)

        if single:
            return float(pred[0, 0])
        return pred.flatten()

    def predict_with_uncertainty(self, features: np.ndarray,
                                 n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with dropout-based uncertainty estimation.

        Only works if dropout > 0 in config.
        """
        if self.config.dropout == 0:
            pred = self.predict(features)
            if isinstance(pred, float):
                return pred, 0.0
            return pred, np.zeros_like(pred)

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        # Enable dropout during inference
        self.model.train()

        predictions = []
        X = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X).cpu().numpy()
                predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Clip mean
        mean_pred = np.clip(mean_pred, -self.config.u_max, self.config.u_max)

        self.model.eval()

        if single:
            return float(mean_pred[0, 0]), float(std_pred[0, 0])
        return mean_pred.flatten(), std_pred.flatten()

    def save(self, path: str):
        """Save policy to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }, path)

    def load(self, path: str):
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_history = checkpoint.get('training_history', [])

    def clone(self) -> 'MLPPolicy':
        """Create a copy of this policy."""
        new_policy = MLPPolicy(self.config, self.seed)
        new_policy.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        new_policy.training_history = copy.deepcopy(self.training_history)
        return new_policy

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights as numpy arrays."""
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights from numpy arrays."""
        state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)


class OraclePolicy:
    """
    Policy wrapper that uses oracle controller directly.

    Used as upper bound baseline.
    """

    def __init__(self, oracle_controller, env, u_max: float = 1.0):
        self.oracle = oracle_controller
        self.env = env
        self.u_max = u_max
        self._step = 0

    def reset(self, occupancy_grid: Optional[np.ndarray] = None):
        """Reset oracle controller for new episode."""
        self.oracle.reset()
        self._step = 0
        if occupancy_grid is not None:
            self.oracle.set_map(occupancy_grid)

    def set_map(self, occupancy_grid: np.ndarray, grid_resolution: float = 0.25):
        """Set occupancy grid for planning."""
        self.oracle.set_map(occupancy_grid, grid_resolution)

    def predict(self, features: np.ndarray, state: Optional[np.ndarray] = None,
                goal: Optional[np.ndarray] = None) -> float:
        """
        Predict action using oracle controller.

        Note: This requires full state and goal, not just features.
        If state/goal not provided, uses current env state.
        """
        if state is None:
            state = self.env.get_state()
        if goal is None:
            goal = self.env.get_goal()

        action = self.oracle.oracle_action(state, goal, step=self._step, u_max=self.u_max)
        self._step += 1
        return action


def create_policy_from_config(config: Dict[str, Any],
                              seed: Optional[int] = None) -> MLPPolicy:
    """Create MLP policy from configuration dict."""
    policy_config = PolicyConfig(
        input_dim=config.get('input_dim', 12),
        hidden_dims=config.get('hidden_dims', [64, 64]),
        output_dim=config.get('output_dim', 1),
        activation=config.get('activation', 'relu'),
        dropout=config.get('dropout', 0.0),
        batch_norm=config.get('batch_norm', False),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        batch_size=config.get('batch_size', 64),
        epochs=config.get('epochs', 30),
        early_stopping_patience=config.get('early_stopping_patience', 5),
        val_fraction=config.get('val_fraction', 0.1),
        u_max=config.get('u_max', 1.0)
    )
    return MLPPolicy(policy_config, seed)
