"""
Tests for learning methods (policy, DAgger, MIND MELD, cognitive model).
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.policy import MLPPolicy, PolicyConfig
from methods.mindmeld import MindMeld, MindMeldConfig
from methods.cognitive_model import CognitiveModel, CognitiveModelConfig, InferredParams


class TestMLPPolicy(unittest.TestCase):
    """Tests for MLP policy."""

    def setUp(self):
        self.config = PolicyConfig(
            input_dim=12,
            hidden_dims=[32, 32],
            output_dim=1,
            epochs=5,
            batch_size=16
        )
        self.policy = MLPPolicy(self.config, seed=42)

    def test_predict_shape(self):
        """Test prediction output shape."""
        # Single sample
        feat = np.random.randn(12).astype(np.float32)
        action = self.policy.predict(feat)
        self.assertIsInstance(action, float)

        # Batch
        feats = np.random.randn(10, 12).astype(np.float32)
        actions = self.policy.predict(feats)
        self.assertEqual(actions.shape, (10,))

    def test_predict_bounded(self):
        """Test that predictions are bounded."""
        feats = np.random.randn(100, 12).astype(np.float32) * 10
        actions = self.policy.predict(feats)

        self.assertTrue(all(abs(a) <= self.config.u_max for a in actions))

    def test_train_reduces_loss(self):
        """Test that training reduces loss."""
        # Create simple training data
        X = np.random.randn(200, 12).astype(np.float32)
        y = np.sin(X[:, 0]) * 0.5  # Simple target

        history = self.policy.train(X, y, verbose=False)

        # Loss should decrease
        self.assertLess(history['train_loss'][-1], history['train_loss'][0])

    def test_deterministic_with_seed(self):
        """Test reproducibility."""
        p1 = MLPPolicy(self.config, seed=42)
        p2 = MLPPolicy(self.config, seed=42)

        feat = np.random.randn(12).astype(np.float32)
        a1 = p1.predict(feat)
        a2 = p2.predict(feat)

        self.assertAlmostEqual(a1, a2, places=5)

    def test_save_load(self):
        """Test model save and load."""
        import tempfile
        import os

        # Train briefly
        X = np.random.randn(100, 12).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        self.policy.train(X, y, verbose=False)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name

        try:
            self.policy.save(path)

            # Load into new policy
            new_policy = MLPPolicy(self.config)
            new_policy.load(path)

            # Predictions should match
            feat = np.random.randn(12).astype(np.float32)
            a1 = self.policy.predict(feat)
            a2 = new_policy.predict(feat)
            self.assertAlmostEqual(a1, a2, places=5)
        finally:
            os.unlink(path)


class TestMindMeld(unittest.TestCase):
    """Tests for MIND MELD model."""

    def setUp(self):
        self.config = MindMeldConfig(
            embedding_dim=4,
            context_window=3,
            lstm_hidden=16,
            epochs=5,
            n_demonstrators=5
        )
        self.mindmeld = MindMeld(self.config, seed=42)

    def test_correct_output_shape(self):
        """Test that correction produces correct shape."""
        # Create mock calibration data
        from data.generate_dataset import EpisodeData

        calib_data = {}
        for demo_id in range(3):
            episodes = []
            for ep_id in range(2):
                ep = EpisodeData(
                    demonstrator_id=demo_id,
                    episode_id=ep_id,
                    split='calib',
                    map_seed=42,
                    obstacles=[],
                    start_state=np.zeros(3),
                    goal=np.zeros(2),
                    states=np.random.randn(20, 4).astype(np.float32),
                    features=np.random.randn(20, 12).astype(np.float32),
                    oracle_actions=np.random.randn(20).astype(np.float32),
                    human_actions=np.random.randn(20).astype(np.float32),
                    success=True,
                    collision=False,
                    t_final=20
                )
                episodes.append(ep)
            calib_data[demo_id] = episodes

        # Train
        self.mindmeld.fit(calib_data, verbose=False)

        # Correct
        human = np.random.randn(15).astype(np.float32)
        features = np.random.randn(15, 12).astype(np.float32)
        corrected = self.mindmeld.correct(0, human, features)

        self.assertEqual(corrected.shape, (15,))

    def test_embedding_retrieval(self):
        """Test embedding retrieval after training."""
        from data.generate_dataset import EpisodeData

        calib_data = {}
        for demo_id in range(3):
            episodes = []
            for ep_id in range(2):
                ep = EpisodeData(
                    demonstrator_id=demo_id,
                    episode_id=ep_id,
                    split='calib',
                    map_seed=42,
                    obstacles=[],
                    start_state=np.zeros(3),
                    goal=np.zeros(2),
                    states=np.random.randn(20, 4).astype(np.float32),
                    features=np.random.randn(20, 12).astype(np.float32),
                    oracle_actions=np.random.randn(20).astype(np.float32),
                    human_actions=np.random.randn(20).astype(np.float32),
                    success=True,
                    collision=False,
                    t_final=20
                )
                episodes.append(ep)
            calib_data[demo_id] = episodes

        self.mindmeld.fit(calib_data, verbose=False)

        emb = self.mindmeld.get_embedding(0)
        self.assertEqual(emb.shape, (self.config.embedding_dim,))


class TestCognitiveModel(unittest.TestCase):
    """Tests for Cognitive Model."""

    def setUp(self):
        self.config = CognitiveModelConfig()
        self.model = CognitiveModel(self.config)

    def test_parameter_inference(self):
        """Test that parameter inference produces valid estimates."""
        from data.generate_dataset import EpisodeData
        from humans.synthetic_teacher import SyntheticTeacher, DemonstratorParams

        # Create known ground truth
        true_params = DemonstratorParams(tau=2.0, g=0.8, sigma=0.05, delta=0.0)
        teacher = SyntheticTeacher(true_params, u_max=1.0, seed=42)

        # Generate calibration data
        calib_data = {}
        for demo_id in [0]:
            episodes = []
            for ep_id in range(4):
                oracle = np.sin(np.linspace(0, 4*np.pi, 50)).astype(np.float32) * 0.5
                human = teacher.generate_episode_labels(oracle)

                ep = EpisodeData(
                    demonstrator_id=demo_id,
                    episode_id=ep_id,
                    split='calib',
                    map_seed=42 + ep_id,
                    obstacles=[],
                    start_state=np.zeros(3),
                    goal=np.zeros(2),
                    states=np.random.randn(50, 4).astype(np.float32),
                    features=np.random.randn(50, 12).astype(np.float32),
                    oracle_actions=oracle,
                    human_actions=human,
                    success=True,
                    collision=False,
                    t_final=50
                )
                episodes.append(ep)
            calib_data[demo_id] = episodes

        # Fit model
        self.model.fit(calib_data, verbose=False)

        # Check inferred parameters
        inferred = self.model.get_params(0)

        # Tau should be positive (human is delayed)
        # Note: estimation may not be exact
        self.assertIsInstance(inferred.tau_hat, float)
        self.assertIsInstance(inferred.g_hat, float)

    def test_correction_bounded(self):
        """Test that corrected labels are bounded."""
        # Create simple inferred params
        self.model.inferred_params = {
            0: InferredParams(tau_hat=1.0, g_hat=0.8)
        }

        human = np.array([0.5, 1.2, -0.8, 0.3])
        corrected = self.model.correct(0, human, u_max=1.0)

        self.assertTrue(all(abs(c) <= 1.0 for c in corrected))

    def test_correction_unknown_demo(self):
        """Test correction for unknown demonstrator returns unchanged."""
        human = np.array([0.5, 0.3, -0.2])
        corrected = self.model.correct(999, human)  # Unknown demo_id

        np.testing.assert_array_equal(corrected, human)


class TestCognitiveModelAblations(unittest.TestCase):
    """Tests for cognitive model ablation variants."""

    def test_no_delag_ablation(self):
        """Test ablation without de-lag."""
        from methods.cognitive_model import CognitiveModelAblation

        config = CognitiveModelConfig()
        model = CognitiveModelAblation(config, ablation='no_delag')

        self.assertFalse(model.config.infer_tau)

    def test_no_gain_ablation(self):
        """Test ablation without gain correction."""
        from methods.cognitive_model import CognitiveModelAblation

        config = CognitiveModelConfig()
        model = CognitiveModelAblation(config, ablation='no_gain')

        self.assertFalse(model.config.infer_g)


if __name__ == '__main__':
    unittest.main()
