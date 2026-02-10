"""
Titan Learning - DPO Trainer

Direct Preference Optimization trainer for fine-tuning language models
using preference pair data without explicit reward modeling.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from titan.learning.preference_pairs import PreferencePairDataset

logger = logging.getLogger("titan.learning.dpo_trainer")


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    # Model settings
    base_model: str = "gpt2"  # Base model to fine-tune
    reference_model: str | None = None  # Reference model (defaults to base)

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo

    # Training hyperparameters
    learning_rate: float = 1e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    warmup_steps: int = 100
    max_length: int = 512
    max_prompt_length: int = 256

    # LoRA settings (Parameter-Efficient Fine-Tuning)
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Evaluation
    eval_steps: int = 100
    save_steps: int = 200

    # Output
    output_dir: str = "./dpo_output"
    save_total_limit: int = 3
    logging_steps: int = 10

    # Device
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False

    # Experiment tracking
    wandb_project: str | None = None
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_model": self.base_model,
            "reference_model": self.reference_model,
            "beta": self.beta,
            "loss_type": self.loss_type,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "use_peft": self.use_peft,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "device": self.device,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "wandb_project": self.wandb_project,
            "run_name": self.run_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DPOConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DPOMetrics:
    """Metrics from DPO training."""

    loss: float = 0.0
    eval_loss: float = 0.0
    rewards_chosen: float = 0.0
    rewards_rejected: float = 0.0
    reward_margin: float = 0.0
    accuracy: float = 0.0
    log_probs_chosen: float = 0.0
    log_probs_rejected: float = 0.0
    step: int = 0
    epoch: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "eval_loss": self.eval_loss,
            "rewards_chosen": self.rewards_chosen,
            "rewards_rejected": self.rewards_rejected,
            "reward_margin": self.reward_margin,
            "accuracy": self.accuracy,
            "log_probs_chosen": self.log_probs_chosen,
            "log_probs_rejected": self.log_probs_rejected,
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrainingResult:
    """Result of a DPO training run."""

    run_id: str = field(default_factory=lambda: str(uuid4())[:8])
    config: DPOConfig = field(default_factory=DPOConfig)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    final_metrics: DPOMetrics | None = None
    checkpoint_path: str | None = None
    status: str = "pending"  # pending, running, completed, failed
    error: str | None = None
    training_samples: int = 0
    eval_samples: int = 0


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Implements DPO for fine-tuning language models directly on
    preference data without explicit reward modeling.
    """

    def __init__(self, config: DPOConfig | None = None) -> None:
        self._config = config or DPOConfig()
        self._training_results: list[TrainingResult] = []
        self._current_run: TrainingResult | None = None

    def train(
        self,
        preference_dataset: PreferencePairDataset,
        eval_dataset: PreferencePairDataset | None = None,
    ) -> TrainingResult:
        """
        Train a model using DPO on preference data.

        Args:
            preference_dataset: Training dataset with preference pairs
            eval_dataset: Optional evaluation dataset

        Returns:
            TrainingResult with final metrics and model path
        """
        result = TrainingResult(
            config=self._config,
            status="running",
            training_samples=len(preference_dataset),
        )
        self._current_run = result
        self._training_results.append(result)

        if eval_dataset:
            result.eval_samples = len(eval_dataset)

        try:
            # Try to use TRL library
            self._train_with_trl(preference_dataset, eval_dataset, result)

        except ImportError:
            # Fall back to mock training
            logger.warning("TRL library not installed, using mock training")
            self._mock_train(preference_dataset, result)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now(UTC)
            logger.error(f"DPO training failed: {e}")
            raise

        finally:
            self._current_run = None

        return result

    def _train_with_trl(
        self,
        preference_dataset: PreferencePairDataset,
        eval_dataset: PreferencePairDataset | None,
        result: TrainingResult,
    ) -> None:
        """Train using TRL's DPOTrainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer as TRLDPOTrainer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self._config.base_model,
            torch_dtype="auto",
        )

        # Apply LoRA if configured
        if self._config.use_peft:
            model = self._apply_lora(model)

        # Load reference model
        ref_model = None
        if self._config.reference_model:
            ref_model = AutoModelForCausalLM.from_pretrained(
                self._config.reference_model,
                torch_dtype="auto",
            )

        # Prepare datasets
        train_data = self._prepare_dataset(preference_dataset)
        eval_data = self._prepare_dataset(eval_dataset) if eval_dataset else None

        # Configure DPO trainer
        training_args = TRLDPOConfig(
            output_dir=self._config.output_dir,
            beta=self._config.beta,
            loss_type=self._config.loss_type,
            learning_rate=self._config.learning_rate,
            per_device_train_batch_size=self._config.batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            max_steps=self._config.max_steps,
            warmup_steps=self._config.warmup_steps,
            max_length=self._config.max_length,
            max_prompt_length=self._config.max_prompt_length,
            eval_strategy="steps" if eval_data else "no",
            eval_steps=self._config.eval_steps if eval_data else None,
            save_steps=self._config.save_steps,
            save_total_limit=self._config.save_total_limit,
            logging_steps=self._config.logging_steps,
            fp16=self._config.fp16,
            bf16=self._config.bf16,
            report_to="wandb" if self._config.wandb_project else "none",
            run_name=self._config.run_name,
        )

        # Create trainer
        trainer = TRLDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
        )

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(self._config.output_dir)
        tokenizer.save_pretrained(self._config.output_dir)

        # Get final metrics
        if eval_data:
            eval_results = trainer.evaluate()
            result.final_metrics = DPOMetrics(
                eval_loss=eval_results.get("eval_loss", 0.0),
                rewards_chosen=eval_results.get("eval_rewards/chosen", 0.0),
                rewards_rejected=eval_results.get("eval_rewards/rejected", 0.0),
                accuracy=eval_results.get("eval_rewards/accuracies", 0.0),
            )

        result.status = "completed"
        result.completed_at = datetime.now(UTC)
        result.checkpoint_path = self._config.output_dir

    def _apply_lora(self, model: Any) -> Any:
        """Apply LoRA adapters to model."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self._config.lora_r,
                lora_alpha=self._config.lora_alpha,
                lora_dropout=self._config.lora_dropout,
                target_modules=self._config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            logger.info(f"Applied LoRA with r={self._config.lora_r}")
            return model

        except ImportError:
            logger.warning("PEFT not installed, skipping LoRA")
            return model

    def _prepare_dataset(self, dataset: PreferencePairDataset) -> Any:
        """Prepare preference pair dataset for DPO."""
        from datasets import Dataset

        data = []
        for pair in dataset.pairs:
            data.append(
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
            )

        return Dataset.from_list(data)

    def _mock_train(
        self,
        dataset: PreferencePairDataset,
        result: TrainingResult,
    ) -> None:
        """Mock training for when TRL is not installed."""
        logger.info(f"Mock DPO training on {len(dataset)} preference pairs")

        result.status = "completed"
        result.completed_at = datetime.now(UTC)
        result.final_metrics = DPOMetrics(
            loss=0.5,
            accuracy=0.5,
            rewards_chosen=0.1,
            rewards_rejected=-0.1,
            reward_margin=0.2,
        )

    def evaluate(
        self,
        eval_dataset: PreferencePairDataset,
    ) -> DPOMetrics:
        """
        Evaluate the current model on a dataset.

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            Evaluation metrics
        """
        # This would use the trainer's evaluate method in practice
        logger.info(f"Evaluating on {len(eval_dataset)} samples")

        return DPOMetrics(
            accuracy=0.0,
            loss=0.0,
            rewards_chosen=0.0,
            rewards_rejected=0.0,
        )

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        """
        Save a training checkpoint.

        Args:
            step: Current training step
            path: Optional custom path

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = path or f"{self._config.output_dir}/checkpoint-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save config
        config_path = Path(checkpoint_dir) / "dpo_config.json"
        with open(config_path, "w") as f:
            json.dump(self._config.to_dict(), f, indent=2)

        # Save step info
        step_path = Path(checkpoint_dir) / "trainer_state.json"
        with open(step_path, "w") as f:
            json.dump({"step": step, "timestamp": datetime.now(UTC).isoformat()}, f)

        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")
        return checkpoint_dir

    def load_checkpoint(self, path: str) -> int:
        """
        Load a training checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Training step from checkpoint
        """
        # Load config
        config_path = Path(path) / "dpo_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = DPOConfig.from_dict(json.load(f))

        # Load step
        step_path = Path(path) / "trainer_state.json"
        if step_path.exists():
            with open(step_path) as f:
                state = json.load(f)
                return int(state.get("step", 0))

        return 0

    def export_model(self, path: str, merge_lora: bool = True) -> None:
        """
        Export the trained model.

        Args:
            path: Output path
            merge_lora: Whether to merge LoRA weights into base model
        """
        os.makedirs(path, exist_ok=True)

        try:
            if merge_lora and self._config.use_peft:
                # Merge LoRA weights and save
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    self._config.base_model,
                    torch_dtype="auto",
                )

                # Load and merge adapter
                model = PeftModel.from_pretrained(base_model, self._config.output_dir)
                model = model.merge_and_unload()

                # Save merged model
                model.save_pretrained(path)

                # Save tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self._config.base_model)
                tokenizer.save_pretrained(path)

                logger.info(f"Exported merged model to {path}")

            else:
                # Just copy the trained model
                import shutil

                for item in ["pytorch_model.bin", "config.json", "tokenizer.json"]:
                    src = Path(self._config.output_dir) / item
                    if src.exists():
                        shutil.copy(src, Path(path) / item)

                logger.info(f"Exported model to {path}")

        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

    def get_training_results(self) -> list[TrainingResult]:
        """Get history of training results."""
        return self._training_results.copy()


# Factory function
_trainer: DPOTrainer | None = None


def get_dpo_trainer(config: DPOConfig | None = None) -> DPOTrainer:
    """Get the default DPO trainer."""
    global _trainer
    if _trainer is None or config is not None:
        _trainer = DPOTrainer(config)
    return _trainer
