"""
Tests for DPO Trainer (Phase 18A)
"""

import pytest
import os
import tempfile
from pathlib import Path

from titan.learning.preference_pairs import PreferencePair, PreferencePairDataset
from titan.learning.dpo_trainer import (
    DPOConfig,
    DPOMetrics,
    DPOTrainer,
    TrainingResult,
    get_dpo_trainer,
)


class TestDPOConfig:
    """Tests for DPOConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DPOConfig()

        assert config.base_model == "gpt2"
        assert config.beta == 0.1
        assert config.learning_rate == 1e-6
        assert config.batch_size == 4
        assert config.max_steps == 1000
        assert config.use_peft is True
        assert config.lora_r == 16

    def test_custom_config(self):
        """Test custom configuration."""
        config = DPOConfig(
            base_model="llama-2-7b",
            beta=0.2,
            learning_rate=5e-7,
            use_peft=False,
        )

        assert config.base_model == "llama-2-7b"
        assert config.beta == 0.2
        assert config.use_peft is False

    def test_lora_config(self):
        """Test LoRA configuration."""
        config = DPOConfig(
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1,
        )

        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = DPOConfig(beta=0.15)
        data = config.to_dict()

        assert data["beta"] == 0.15
        assert "base_model" in data
        assert "lora_r" in data
        assert "use_peft" in data

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "base_model": "custom-model",
            "beta": 0.2,
            "max_steps": 500,
        }

        config = DPOConfig.from_dict(data)
        assert config.base_model == "custom-model"
        assert config.beta == 0.2
        assert config.max_steps == 500


class TestDPOMetrics:
    """Tests for DPOMetrics."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = DPOMetrics()

        assert metrics.loss == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.rewards_chosen == 0.0
        assert metrics.rewards_rejected == 0.0

    def test_metrics_with_values(self):
        """Test metrics with values."""
        metrics = DPOMetrics(
            loss=0.5,
            accuracy=0.75,
            rewards_chosen=0.3,
            rewards_rejected=-0.2,
            reward_margin=0.5,
        )

        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.75
        assert metrics.reward_margin == 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = DPOMetrics(loss=0.4, accuracy=0.8)
        data = metrics.to_dict()

        assert data["loss"] == 0.4
        assert data["accuracy"] == 0.8
        assert "timestamp" in data


class TestTrainingResult:
    """Tests for TrainingResult."""

    def test_create_result(self):
        """Test creating a training result."""
        result = TrainingResult()

        assert result.run_id is not None
        assert result.status == "pending"
        assert result.training_samples == 0

    def test_result_with_metrics(self):
        """Test result with final metrics."""
        metrics = DPOMetrics(accuracy=0.85)
        result = TrainingResult(
            status="completed",
            training_samples=1000,
            final_metrics=metrics,
        )

        assert result.status == "completed"
        assert result.final_metrics.accuracy == 0.85


class TestDPOTrainer:
    """Tests for DPOTrainer."""

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
        trainer = DPOTrainer()
        assert trainer._config is not None

    def test_create_trainer_with_config(self):
        """Test creating trainer with custom config."""
        config = DPOConfig(beta=0.2)
        trainer = DPOTrainer(config)

        assert trainer._config.beta == 0.2

    def test_mock_train(self, sample_dataset):
        """Test training (mock mode without TRL)."""
        trainer = DPOTrainer()

        # This will use mock training if TRL not installed
        result = trainer.train(sample_dataset)

        assert result is not None
        assert isinstance(result, TrainingResult)
        assert result.status in ("completed", "failed")

    def test_training_with_eval_dataset(self, sample_dataset):
        """Test training with evaluation dataset."""
        trainer = DPOTrainer()

        # Split dataset
        train_data = PreferencePairDataset()
        eval_data = PreferencePairDataset()

        for i, pair in enumerate(sample_dataset.pairs):
            if i < 8:
                train_data.add(pair)
            else:
                eval_data.add(pair)

        result = trainer.train(train_data, eval_data)

        assert result.training_samples == len(train_data)
        assert result.eval_samples == len(eval_data)

    def test_training_results_tracked(self, sample_dataset):
        """Test that training results are tracked."""
        trainer = DPOTrainer()
        trainer.train(sample_dataset)

        results = trainer.get_training_results()
        assert len(results) >= 1

    def test_evaluate(self, sample_dataset):
        """Test model evaluation."""
        trainer = DPOTrainer()

        metrics = trainer.evaluate(sample_dataset)

        assert isinstance(metrics, DPOMetrics)

    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        trainer = DPOTrainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save_checkpoint(100, tmpdir)

            assert os.path.exists(path)
            assert os.path.exists(os.path.join(path, "dpo_config.json"))

    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        trainer = DPOTrainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            trainer.save_checkpoint(100, tmpdir)

            # Load
            step = trainer.load_checkpoint(tmpdir)
            assert step == 100


class TestGetDPOTrainer:
    """Tests for factory function."""

    def test_get_default_trainer(self):
        """Test getting default trainer."""
        trainer = get_dpo_trainer()
        assert trainer is not None
        assert isinstance(trainer, DPOTrainer)

    def test_get_with_config(self):
        """Test getting trainer with config."""
        config = DPOConfig(max_steps=2000)
        trainer = get_dpo_trainer(config)

        assert trainer._config.max_steps == 2000
