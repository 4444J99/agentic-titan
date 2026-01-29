"""
Tests for Reward Model Trainer (Phase 18A)
"""

import pytest
import os
import tempfile
from pathlib import Path

from titan.learning.preference_pairs import PreferencePair, PreferencePairDataset
from titan.learning.reward_model import (
    RewardModelConfig,
    RewardMetrics,
    RewardModel,
    RewardModelTrainer,
    TrainingRun,
    get_reward_model_trainer,
)


class TestRewardModelConfig:
    """Tests for RewardModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RewardModelConfig()

        assert config.base_model == "distilbert-base-uncased"
        assert config.learning_rate == 2e-5
        assert config.batch_size == 8
        assert config.epochs == 3
        assert config.eval_split == 0.1
        assert config.device == "auto"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RewardModelConfig(
            base_model="bert-base-uncased",
            learning_rate=1e-5,
            batch_size=16,
            epochs=5,
        )

        assert config.base_model == "bert-base-uncased"
        assert config.learning_rate == 1e-5
        assert config.batch_size == 16
        assert config.epochs == 5

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = RewardModelConfig(base_model="test-model")
        data = config.to_dict()

        assert data["base_model"] == "test-model"
        assert "learning_rate" in data
        assert "batch_size" in data

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "base_model": "custom-model",
            "learning_rate": 3e-5,
            "batch_size": 4,
        }

        config = RewardModelConfig.from_dict(data)
        assert config.base_model == "custom-model"
        assert config.learning_rate == 3e-5
        assert config.batch_size == 4


class TestRewardMetrics:
    """Tests for RewardMetrics."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = RewardMetrics()

        assert metrics.accuracy == 0.0
        assert metrics.loss == 0.0
        assert metrics.samples_evaluated == 0

    def test_metrics_with_values(self):
        """Test metrics with values."""
        metrics = RewardMetrics(
            accuracy=0.85,
            loss=0.35,
            mean_reward=0.5,
            std_reward=0.2,
            samples_evaluated=100,
        )

        assert metrics.accuracy == 0.85
        assert metrics.loss == 0.35
        assert metrics.mean_reward == 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = RewardMetrics(accuracy=0.9, loss=0.1)
        data = metrics.to_dict()

        assert data["accuracy"] == 0.9
        assert data["loss"] == 0.1
        assert "timestamp" in data


class TestRewardModel:
    """Tests for RewardModel."""

    def test_create_empty_model(self):
        """Test creating model without weights."""
        model = RewardModel()

        assert model._model is None
        assert model._tokenizer is None

    def test_predict_without_model(self):
        """Test prediction without loaded model."""
        model = RewardModel()

        score = model.predict("What is AI?", "AI is artificial intelligence.")
        assert score == 0.0  # Default when no model

    def test_compare(self):
        """Test comparing two responses."""
        model = RewardModel()

        # Without model, should return 0 (tie)
        result = model.compare("Q", "Response A", "Response B")
        assert result == 0

    def test_predict_batch(self):
        """Test batch prediction."""
        model = RewardModel()

        pairs = [
            ("Q1", "A1"),
            ("Q2", "A2"),
        ]

        scores = model.predict_batch(pairs)
        assert len(scores) == 2


class TestRewardModelTrainer:
    """Tests for RewardModelTrainer."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        dataset = PreferencePairDataset(name="test")
        for i in range(10):
            dataset.add(PreferencePair(
                prompt=f"Question {i}",
                chosen=f"Good answer {i}",
                rejected=f"Bad answer {i}",
            ))
        return dataset

    def test_create_trainer(self):
        """Test creating trainer."""
        trainer = RewardModelTrainer()
        assert trainer._config is not None

    def test_create_trainer_with_config(self):
        """Test creating trainer with custom config."""
        config = RewardModelConfig(batch_size=4)
        trainer = RewardModelTrainer(config)

        assert trainer._config.batch_size == 4

    def test_mock_train(self, sample_dataset):
        """Test training (mock mode without transformers)."""
        trainer = RewardModelTrainer()

        # This will use mock training if transformers not installed
        model = trainer.train(sample_dataset)

        assert model is not None
        assert isinstance(model, RewardModel)

    def test_training_runs_tracked(self, sample_dataset):
        """Test that training runs are tracked."""
        trainer = RewardModelTrainer()
        trainer.train(sample_dataset)

        runs = trainer.get_training_runs()
        assert len(runs) >= 1

        run = runs[-1]
        assert run.status in ("completed", "failed")

    def test_evaluate(self, sample_dataset):
        """Test model evaluation."""
        trainer = RewardModelTrainer()
        model = trainer.train(sample_dataset)

        metrics = trainer.evaluate(model, sample_dataset)

        assert isinstance(metrics, RewardMetrics)
        assert metrics.samples_evaluated == len(sample_dataset)

    def test_save_load(self, sample_dataset):
        """Test saving and loading model."""
        trainer = RewardModelTrainer()
        model = trainer.train(sample_dataset)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"

            # Save
            trainer.save(model, str(path))
            assert (path / "config.json").exists()

            # Load
            loaded = trainer.load(str(path))
            assert loaded._config is not None


class TestTrainingRun:
    """Tests for TrainingRun."""

    def test_create_run(self):
        """Test creating a training run."""
        run = TrainingRun()

        assert run.run_id is not None
        assert run.status == "pending"
        assert run.training_samples == 0

    def test_run_with_metrics(self):
        """Test run with final metrics."""
        metrics = RewardMetrics(accuracy=0.9)
        run = TrainingRun(
            status="completed",
            training_samples=100,
            final_metrics=metrics,
        )

        assert run.status == "completed"
        assert run.final_metrics.accuracy == 0.9


class TestGetRewardModelTrainer:
    """Tests for factory function."""

    def test_get_default_trainer(self):
        """Test getting default trainer."""
        trainer = get_reward_model_trainer()
        assert trainer is not None
        assert isinstance(trainer, RewardModelTrainer)

    def test_get_with_config(self):
        """Test getting trainer with config."""
        config = RewardModelConfig(epochs=10)
        trainer = get_reward_model_trainer(config)

        assert trainer._config.epochs == 10
