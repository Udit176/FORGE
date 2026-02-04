"""
MIND MELD Implementation.

Learns per-demonstrator embeddings and maps human labels to corrected labels.
Supports both simplified (learned embedding) and full (variational MI) modes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import copy

from data.generate_dataset import EpisodeData, split_by_demonstrator


@dataclass
class MindMeldConfig:
    """Configuration for MIND MELD model."""
    # Architecture
    embedding_dim: int = 8           # Demonstrator embedding dimension
    context_window: int = 5          # Past/future context window L
    lstm_hidden: int = 32            # LSTM hidden dimension
    use_bidirectional: bool = True   # Use bidirectional LSTM
    use_state_features: bool = True  # Include state features in input

    # Mode
    mode: str = 'learned_embedding'  # 'learned_embedding' or 'variational_mi'

    # Variational MI settings (if mode='variational_mi')
    mi_weight: float = 0.1           # Weight for MI term
    encoder_hidden: int = 32         # Encoder hidden dimension

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10

    # Input dimensions (set during training)
    state_feature_dim: int = 12
    n_demonstrators: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding_dim': self.embedding_dim,
            'context_window': self.context_window,
            'lstm_hidden': self.lstm_hidden,
            'use_bidirectional': self.use_bidirectional,
            'use_state_features': self.use_state_features,
            'mode': self.mode,
            'mi_weight': self.mi_weight,
            'encoder_hidden': self.encoder_hidden,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'state_feature_dim': self.state_feature_dim,
            'n_demonstrators': self.n_demonstrators
        }


class SequenceEncoder(nn.Module):
    """Encodes sequence of human actions using LSTM."""

    def __init__(self, config: MindMeldConfig):
        super().__init__()
        self.config = config

        # Input: human action at each timestep (1-dim)
        input_dim = 1
        if config.use_state_features:
            input_dim += config.state_feature_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=config.use_bidirectional
        )

        output_dim = config.lstm_hidden * (2 if config.use_bidirectional else 1)
        self.output_dim = output_dim

    def forward(self, actions: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sequence of actions.

        Args:
            actions: (batch, seq_len, 1) human actions
            features: (batch, seq_len, D) state features (optional)

        Returns:
            encoded: (batch, seq_len, hidden) encoded representations
        """
        if features is not None and self.config.use_state_features:
            x = torch.cat([actions, features], dim=-1)
        else:
            x = actions

        encoded, _ = self.lstm(x)
        return encoded


class CorrectionNetwork(nn.Module):
    """Predicts correction from encoded sequence and embedding."""

    def __init__(self, config: MindMeldConfig):
        super().__init__()
        self.config = config

        encoder_output_dim = config.lstm_hidden * (2 if config.use_bidirectional else 1)
        input_dim = encoder_output_dim + config.embedding_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict oracle action directly
        )

    def forward(self, encoded: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict corrected action.

        Args:
            encoded: (batch, hidden) encoded sequence at current timestep
            embedding: (batch, embed_dim) demonstrator embedding

        Returns:
            corrected: (batch, 1) predicted oracle action
        """
        x = torch.cat([encoded, embedding], dim=-1)
        return self.network(x)


class EmbeddingEncoder(nn.Module):
    """Encodes demonstrator data into embedding (for variational mode)."""

    def __init__(self, config: MindMeldConfig):
        super().__init__()
        self.config = config

        input_dim = config.lstm_hidden * (2 if config.use_bidirectional else 1)

        # Encoder that aggregates sequence to embedding
        self.aggregator = nn.Sequential(
            nn.Linear(input_dim, config.encoder_hidden),
            nn.ReLU()
        )

        # Mean and log-variance for variational embedding
        self.mean_head = nn.Linear(config.encoder_hidden, config.embedding_dim)
        self.logvar_head = nn.Linear(config.encoder_hidden, config.embedding_dim)

    def forward(self, encoded_seq: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence to embedding distribution.

        Args:
            encoded_seq: (batch, seq_len, hidden) encoded sequence
            mask: (batch, seq_len) valid timestep mask

        Returns:
            mean: (batch, embed_dim) embedding mean
            logvar: (batch, embed_dim) embedding log-variance
        """
        # Mean pooling over sequence
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            pooled = (encoded_seq * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            pooled = encoded_seq.mean(dim=1)

        h = self.aggregator(pooled)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        return mean, logvar


class MindMeldModel(nn.Module):
    """Full MIND MELD model."""

    def __init__(self, config: MindMeldConfig):
        super().__init__()
        self.config = config

        # Sequence encoder
        self.seq_encoder = SequenceEncoder(config)

        # Demonstrator embeddings
        if config.mode == 'learned_embedding':
            # Learned lookup table
            self.embeddings = nn.Embedding(config.n_demonstrators, config.embedding_dim)
            nn.init.normal_(self.embeddings.weight, std=0.1)
        else:
            # Variational encoder
            self.embedding_encoder = EmbeddingEncoder(config)

        # Correction network
        self.correction_net = CorrectionNetwork(config)

    def get_embedding(self, demo_ids: torch.Tensor,
                      encoded_seqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get embedding for demonstrators."""
        if self.config.mode == 'learned_embedding':
            return self.embeddings(demo_ids)
        else:
            if encoded_seqs is None:
                raise ValueError("Need encoded sequences for variational mode")
            mean, logvar = self.embedding_encoder(encoded_seqs)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

    def forward(self, actions: torch.Tensor,
                demo_ids: torch.Tensor,
                features: Optional[torch.Tensor] = None,
                timestep_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to predict corrected actions.

        Args:
            actions: (batch, seq_len, 1) human actions in context window
            demo_ids: (batch,) demonstrator IDs
            features: (batch, seq_len, D) state features (optional)
            timestep_idx: (batch,) index of target timestep in window

        Returns:
            corrected: (batch, 1) predicted oracle actions
        """
        # Encode sequence
        encoded = self.seq_encoder(actions, features)

        # Get embedding
        if self.config.mode == 'learned_embedding':
            embedding = self.embeddings(demo_ids)
        else:
            embedding = self.get_embedding(demo_ids, encoded)

        # Get encoded state at target timestep
        if timestep_idx is None:
            # Use middle of window
            mid = encoded.size(1) // 2
            encoded_t = encoded[:, mid, :]
        else:
            batch_idx = torch.arange(encoded.size(0), device=encoded.device)
            encoded_t = encoded[batch_idx, timestep_idx, :]

        # Predict correction
        corrected = self.correction_net(encoded_t, embedding)

        return corrected

    def compute_mi_loss(self, actions: torch.Tensor,
                        demo_ids: torch.Tensor,
                        features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute MI regularization loss (for variational mode)."""
        if self.config.mode != 'variational_mi':
            return torch.tensor(0.0, device=actions.device)

        encoded = self.seq_encoder(actions, features)
        mean, logvar = self.embedding_encoder(encoded)

        # KL divergence from standard normal
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()


class MindMeld:
    """
    MIND MELD training and inference wrapper.

    Learns to correct human labels to oracle labels using per-demonstrator embeddings.
    """

    def __init__(self, config: MindMeldConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device('cpu')
        self.model: Optional[MindMeldModel] = None
        self.training_history = []

        # Cached embeddings after training
        self._embeddings: Optional[np.ndarray] = None

    def fit(self, calibration_data: Dict[int, List[EpisodeData]],
            verbose: bool = False) -> Dict[str, List[float]]:
        """
        Train MIND MELD on calibration data.

        Args:
            calibration_data: Dict mapping demo_id to list of calibration episodes
            verbose: Print training progress

        Returns:
            Training history
        """
        # Prepare training data
        windows, labels, demo_ids, features = self._prepare_training_data(calibration_data)

        if verbose:
            print(f"Training MIND MELD on {len(labels)} samples...")
            print(f"  Demonstrators: {len(calibration_data)}")
            print(f"  Context window: {self.config.context_window}")

        # Update config
        self.config.n_demonstrators = max(max(calibration_data.keys()) + 1,
                                           self.config.n_demonstrators)
        if features is not None:
            self.config.state_feature_dim = features.shape[-1]

        # Create model
        self.model = MindMeldModel(self.config).to(self.device)

        # Create data loaders
        tensors = [
            torch.FloatTensor(windows),
            torch.FloatTensor(labels),
            torch.LongTensor(demo_ids)
        ]
        if features is not None:
            tensors.append(torch.FloatTensor(features))

        dataset = TensorDataset(*tensors)

        # Split into train/val
        n = len(dataset)
        n_val = max(1, int(0.1 * n))
        n_train = n - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'mi_loss': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            self.model.train()
            train_losses = []
            mi_losses = []

            for batch in train_loader:
                if features is not None:
                    batch_windows, batch_labels, batch_demo_ids, batch_features = batch
                    batch_features = batch_features.to(self.device)
                else:
                    batch_windows, batch_labels, batch_demo_ids = batch
                    batch_features = None

                batch_windows = batch_windows.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_demo_ids = batch_demo_ids.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pred = self.model(batch_windows.unsqueeze(-1), batch_demo_ids,
                                 batch_features)
                pred_loss = nn.MSELoss()(pred.squeeze(-1), batch_labels)

                # MI loss (variational mode)
                mi_loss = self.model.compute_mi_loss(batch_windows.unsqueeze(-1),
                                                     batch_demo_ids, batch_features)

                loss = pred_loss + self.config.mi_weight * mi_loss

                loss.backward()
                optimizer.step()

                train_losses.append(pred_loss.item())
                mi_losses.append(mi_loss.item())

            # Validate
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    if features is not None:
                        batch_windows, batch_labels, batch_demo_ids, batch_features = batch
                        batch_features = batch_features.to(self.device)
                    else:
                        batch_windows, batch_labels, batch_demo_ids = batch
                        batch_features = None

                    batch_windows = batch_windows.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_demo_ids = batch_demo_ids.to(self.device)

                    pred = self.model(batch_windows.unsqueeze(-1), batch_demo_ids,
                                     batch_features)
                    loss = nn.MSELoss()(pred.squeeze(-1), batch_labels)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            avg_mi = np.mean(mi_losses)

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['mi_loss'].append(avg_mi)

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}: train={avg_train:.5f}, val={avg_val:.5f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Cache embeddings
        self._cache_embeddings()

        self.training_history = history
        return history

    def _prepare_training_data(self, calibration_data: Dict[int, List[EpisodeData]]
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare training data from calibration episodes.

        Creates sliding windows of human actions with oracle label targets.
        """
        L = self.config.context_window
        window_size = 2 * L + 1

        all_windows = []
        all_labels = []
        all_demo_ids = []
        all_features = []

        for demo_id, episodes in calibration_data.items():
            for ep in episodes:
                T = len(ep.oracle_actions)
                human = ep.human_actions
                oracle = ep.oracle_actions
                features = ep.features[:T] if self.config.use_state_features else None

                # Pad sequences
                human_padded = np.concatenate([
                    np.zeros(L) + human[0],
                    human,
                    np.zeros(L) + human[-1]
                ])
                if features is not None:
                    features_padded = np.concatenate([
                        np.tile(features[0:1], (L, 1)),
                        features,
                        np.tile(features[-1:], (L, 1))
                    ], axis=0)

                # Create windows
                for t in range(T):
                    window = human_padded[t:t + window_size]
                    all_windows.append(window)
                    all_labels.append(oracle[t])
                    all_demo_ids.append(demo_id)

                    if features is not None:
                        feat_window = features_padded[t:t + window_size]
                        all_features.append(feat_window)

        windows = np.array(all_windows, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.float32)
        demo_ids = np.array(all_demo_ids, dtype=np.int32)

        if all_features:
            features = np.array(all_features, dtype=np.float32)
        else:
            features = None

        return windows, labels, demo_ids, features

    def _cache_embeddings(self):
        """Cache learned embeddings for each demonstrator."""
        if self.model is None:
            return

        self.model.eval()
        with torch.no_grad():
            if self.config.mode == 'learned_embedding':
                self._embeddings = self.model.embeddings.weight.cpu().numpy()
            else:
                # For variational mode, we'd need to encode calibration data
                # For now, just store the embedding means
                self._embeddings = None

    def correct(self, demo_id: int,
                human_labels: np.ndarray,
                features: Optional[np.ndarray] = None,
                u_max: float = 1.0) -> np.ndarray:
        """
        Correct human labels to oracle labels.

        Args:
            demo_id: Demonstrator ID
            human_labels: (T,) human action labels
            features: (T, D) state features (optional)
            u_max: Max action for clipping

        Returns:
            corrected: (T,) corrected labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()
        L = self.config.context_window
        window_size = 2 * L + 1
        T = len(human_labels)

        # Pad sequences
        human_padded = np.concatenate([
            np.zeros(L) + human_labels[0],
            human_labels,
            np.zeros(L) + human_labels[-1]
        ])

        if features is not None and self.config.use_state_features:
            features_padded = np.concatenate([
                np.tile(features[0:1], (L, 1)),
                features,
                np.tile(features[-1:], (L, 1))
            ], axis=0)
        else:
            features_padded = None

        corrected = np.zeros(T, dtype=np.float32)

        with torch.no_grad():
            for t in range(T):
                window = human_padded[t:t + window_size]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1).to(self.device)
                demo_tensor = torch.LongTensor([demo_id]).to(self.device)

                if features_padded is not None:
                    feat_window = features_padded[t:t + window_size]
                    feat_tensor = torch.FloatTensor(feat_window).unsqueeze(0).to(self.device)
                else:
                    feat_tensor = None

                pred = self.model(window_tensor, demo_tensor, feat_tensor)
                corrected[t] = pred.item()

        # Clip to bounds
        corrected = np.clip(corrected, -u_max, u_max)
        return corrected

    def get_embedding(self, demo_id: int) -> np.ndarray:
        """Get learned embedding for a demonstrator."""
        if self._embeddings is None:
            if self.model is not None:
                self._cache_embeddings()
            else:
                raise ValueError("Model not trained")

        if self._embeddings is None:
            raise ValueError("Embeddings not available (variational mode)")

        return self._embeddings[demo_id].copy()

    def get_all_embeddings(self) -> np.ndarray:
        """Get all learned embeddings."""
        if self._embeddings is None:
            if self.model is not None:
                self._cache_embeddings()
            else:
                raise ValueError("Model not trained")

        return self._embeddings.copy() if self._embeddings is not None else None

    def create_corrector(self, u_max: float = 1.0):
        """
        Create a corrector function for use with DAgger.

        Returns:
            Corrector function with signature (human_labels, features, demo_id, ...) -> corrected
        """
        def corrector(human_labels, features, demo_id, **kwargs):
            return self.correct(demo_id, human_labels, features, u_max)

        return corrector

    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config.to_dict(),
            'embeddings': self._embeddings,
            'training_history': self.training_history
        }, path)

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)

        # Recreate config
        for key, value in checkpoint['config'].items():
            setattr(self.config, key, value)

        # Recreate model
        self.model = MindMeldModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        self._embeddings = checkpoint.get('embeddings')
        self.training_history = checkpoint.get('training_history', [])


class MindMeldNonPersonalized(MindMeld):
    """
    MIND MELD variant without personalization.

    Uses single shared embedding for all demonstrators (ablation baseline).
    """

    def __init__(self, config: MindMeldConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        # Force single embedding
        self.config.n_demonstrators = 1

    def correct(self, demo_id: int,
                human_labels: np.ndarray,
                features: Optional[np.ndarray] = None,
                u_max: float = 1.0) -> np.ndarray:
        # Always use demo_id=0 (shared embedding)
        return super().correct(0, human_labels, features, u_max)


def create_mindmeld_from_config(config: Dict[str, Any],
                                seed: Optional[int] = None) -> MindMeld:
    """Create MIND MELD from configuration dict."""
    mm_config = MindMeldConfig(
        embedding_dim=config.get('embedding_dim', 8),
        context_window=config.get('context_window', 5),
        lstm_hidden=config.get('lstm_hidden', 32),
        use_bidirectional=config.get('use_bidirectional', True),
        use_state_features=config.get('use_state_features', True),
        mode=config.get('mode', 'learned_embedding'),
        mi_weight=config.get('mi_weight', 0.1),
        encoder_hidden=config.get('encoder_hidden', 32),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        batch_size=config.get('batch_size', 64),
        epochs=config.get('epochs', 50),
        early_stopping_patience=config.get('early_stopping_patience', 10),
        state_feature_dim=config.get('state_feature_dim', 12),
        n_demonstrators=config.get('n_demonstrators', 50)
    )
    return MindMeld(mm_config, seed)
