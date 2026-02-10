"""
Titan Learning - Reward Model Training

Wrapper for reward model training using transformer-based models.
Supports training, evaluation, and model persistence.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from titan.learning.preference_pairs import PreferencePairDataset

logger = logging.getLogger("titan.learning.reward_model")


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""

    # Model settings
    base_model: str = "distilbert-base-uncased"
    num_labels: int = 1  # Scalar reward output

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512

    # Evaluation
    eval_split: float = 0.1
    eval_steps: int = 100
    save_steps: int = 500

    # Output
    output_dir: str = "./reward_model_output"
    save_total_limit: int = 3

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_model": self.base_model,
            "num_labels": self.num_labels,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "eval_split": self.eval_split,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
            "save_total_limit": self.save_total_limit,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardModelConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RewardMetrics:
    """Metrics from reward model training/evaluation."""

    accuracy: float = 0.0
    loss: float = 0.0
    eval_loss: float = 0.0
    correlation: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    samples_evaluated: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "loss": self.loss,
            "eval_loss": self.eval_loss,
            "correlation": self.correlation,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "samples_evaluated": self.samples_evaluated,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrainingRun:
    """Record of a training run."""

    run_id: str = field(default_factory=lambda: str(uuid4())[:8])
    config: RewardModelConfig = field(default_factory=RewardModelConfig)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    final_metrics: RewardMetrics | None = None
    training_samples: int = 0
    eval_samples: int = 0
    status: str = "pending"  # pending, running, completed, failed
    error: str | None = None
    model_path: str | None = None


class RewardModel:
    """
    Trained reward model wrapper.

    Provides inference interface for scoring responses.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        config: RewardModelConfig | None = None,
        model_path: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._config = config or RewardModelConfig()
        self._model_path = model_path
        self._device = "cpu"

    def predict(self, prompt: str, response: str) -> float:
        """
        Predict reward score for a prompt-response pair.

        Args:
            prompt: The input prompt
            response: The model response

        Returns:
            Reward score (higher = better)
        """
        if self._model is None:
            logger.warning("No model loaded, returning default score")
            return 0.0

        try:
            import torch

            # Tokenize
            text = f"{prompt}\n{response}"
            inputs = self._tokenizer(
                text,
                truncation=True,
                max_length=self._config.max_length,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self._model(**inputs)
                reward = outputs.logits.squeeze().item()

            return float(reward)

        except Exception as e:
            logger.error(f"Error predicting reward: {e}")
            return 0.0

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Predict rewards for multiple prompt-response pairs.

        Args:
            pairs: List of (prompt, response) tuples

        Returns:
            List of reward scores
        """
        return [self.predict(prompt, response) for prompt, response in pairs]

    def compare(self, prompt: str, response_a: str, response_b: str) -> int:
        """
        Compare two responses for the same prompt.

        Args:
            prompt: The input prompt
            response_a: First response
            response_b: Second response

        Returns:
            1 if response_a is preferred, -1 if response_b, 0 if tie
        """
        score_a = self.predict(prompt, response_a)
        score_b = self.predict(prompt, response_b)

        if abs(score_a - score_b) < 0.01:  # Tie threshold
            return 0
        return 1 if score_a > score_b else -1

    def to(self, device: str) -> RewardModel:
        """Move model to device."""
        if self._model is not None:
            self._model = self._model.to(device)
            self._device = device
        return self


class RewardModelTrainer:
    """
    Trainer for reward models.

    Uses HuggingFace transformers for training.
    """

    def __init__(self, config: RewardModelConfig | None = None) -> None:
        self._config = config or RewardModelConfig()
        self._training_runs: list[TrainingRun] = []

    def train(
        self,
        dataset: PreferencePairDataset,
        eval_dataset: PreferencePairDataset | None = None,
    ) -> RewardModel:
        """
        Train a reward model on preference pair data.

        Args:
            dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Trained RewardModel
        """
        run = TrainingRun(
            config=self._config,
            training_samples=len(dataset),
            status="running",
        )
        self._training_runs.append(run)

        try:
            # Try to use transformers
            model, tokenizer = self._train_transformers(dataset, eval_dataset, run)

            run.status = "completed"
            run.completed_at = datetime.now(UTC)
            run.model_path = self._config.output_dir

            return RewardModel(
                model=model,
                tokenizer=tokenizer,
                config=self._config,
                model_path=self._config.output_dir,
            )

        except ImportError:
            # Fall back to mock training
            logger.warning("transformers not installed, using mock training")
            return self._mock_train(dataset, run)

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.completed_at = datetime.now(UTC)
            logger.error(f"Training failed: {e}")
            raise

    def _train_transformers(
        self,
        dataset: PreferencePairDataset,
        eval_dataset: PreferencePairDataset | None,
        run: TrainingRun,
    ) -> tuple[Any, Any]:
        """Train using HuggingFace transformers."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._config.base_model,
            num_labels=self._config.num_labels,
        )

        # Prepare data
        train_data = self._prepare_reward_data(dataset, tokenizer)

        eval_data = None
        if eval_dataset:
            eval_data = self._prepare_reward_data(eval_dataset, tokenizer)
            run.eval_samples = len(eval_dataset)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self._config.output_dir,
            learning_rate=self._config.learning_rate,
            per_device_train_batch_size=self._config.batch_size,
            per_device_eval_batch_size=self._config.batch_size,
            num_train_epochs=self._config.epochs,
            weight_decay=self._config.weight_decay,
            warmup_ratio=self._config.warmup_ratio,
            evaluation_strategy="steps" if eval_data else "no",
            eval_steps=self._config.eval_steps if eval_data else None,
            save_steps=self._config.save_steps,
            save_total_limit=self._config.save_total_limit,
            logging_steps=50,
            load_best_model_at_end=True if eval_data else False,
            report_to="none",  # Disable wandb by default
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
        )

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(self._config.output_dir)
        tokenizer.save_pretrained(self._config.output_dir)

        # Record metrics
        if eval_data:
            eval_results = trainer.evaluate()
            run.final_metrics = RewardMetrics(
                eval_loss=eval_results.get("eval_loss", 0.0),
                samples_evaluated=len(eval_data),
            )

        return model, tokenizer

    def _prepare_reward_data(
        self,
        dataset: PreferencePairDataset,
        tokenizer: Any,
    ) -> Any:
        """Prepare preference pair data for reward model training."""
        from datasets import Dataset

        # For reward model training, we create examples where:
        # - Input: prompt + response
        # - Label: 1 for chosen, 0 for rejected
        examples = []

        for pair in dataset.pairs:
            # Chosen example (positive)
            examples.append(
                {
                    "text": f"{pair.prompt}\n{pair.chosen}",
                    "label": 1.0,
                }
            )
            # Rejected example (negative)
            examples.append(
                {
                    "text": f"{pair.prompt}\n{pair.rejected}",
                    "label": 0.0,
                }
            )

        hf_dataset = Dataset.from_list(examples)

        # Tokenize
        def tokenize_function(batch: dict[str, list[str]]) -> dict[str, Any]:
            return cast(
                dict[str, Any],
                tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=self._config.max_length,
                    padding="max_length",
                ),
            )

        tokenized = hf_dataset.map(tokenize_function, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return tokenized

    def _mock_train(
        self,
        dataset: PreferencePairDataset,
        run: TrainingRun,
    ) -> RewardModel:
        """Mock training for when transformers is not installed."""
        logger.info(f"Mock training on {len(dataset)} preference pairs")

        run.status = "completed"
        run.completed_at = datetime.now(UTC)
        run.final_metrics = RewardMetrics(
            accuracy=0.5,
            loss=1.0,
            samples_evaluated=len(dataset),
        )

        return RewardModel(config=self._config)

    def evaluate(
        self,
        model: RewardModel,
        dataset: PreferencePairDataset,
    ) -> RewardMetrics:
        """
        Evaluate a reward model on a dataset.

        Args:
            model: Trained reward model
            dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0
        rewards = []

        for pair in dataset.pairs:
            chosen_reward = model.predict(pair.prompt, pair.chosen)
            rejected_reward = model.predict(pair.prompt, pair.rejected)

            rewards.extend([chosen_reward, rejected_reward])

            # Correct if chosen has higher reward
            if chosen_reward > rejected_reward:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        import statistics

        mean_reward = statistics.mean(rewards) if rewards else 0.0
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

        return RewardMetrics(
            accuracy=accuracy,
            mean_reward=mean_reward,
            std_reward=std_reward,
            samples_evaluated=total,
        )

    def save(self, model: RewardModel, path: str) -> None:
        """
        Save a reward model to disk.

        Args:
            model: Model to save
            path: Output path
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config_path = Path(path) / "config.json"
        with open(config_path, "w") as f:
            json.dump(model._config.to_dict(), f, indent=2)

        # Save model if available
        if model._model is not None:
            try:
                model._model.save_pretrained(path)
                if model._tokenizer is not None:
                    model._tokenizer.save_pretrained(path)
            except Exception as e:
                logger.warning(f"Could not save model weights: {e}")

        logger.info(f"Saved reward model to {path}")

    def load(self, path: str) -> RewardModel:
        """
        Load a reward model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded RewardModel
        """
        # Load config
        config_path = Path(path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = RewardModelConfig.from_dict(json.load(f))
        else:
            config = RewardModelConfig()

        # Try to load model
        model = None
        tokenizer = None

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            model = AutoModelForSequenceClassification.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)

        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")

        return RewardModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            model_path=path,
        )

    def get_training_runs(self) -> list[TrainingRun]:
        """Get history of training runs."""
        return self._training_runs.copy()


# Factory function
_trainer: RewardModelTrainer | None = None


def get_reward_model_trainer(
    config: RewardModelConfig | None = None,
) -> RewardModelTrainer:
    """Get the default reward model trainer."""
    global _trainer
    if _trainer is None or config is not None:
        _trainer = RewardModelTrainer(config)
    return _trainer
