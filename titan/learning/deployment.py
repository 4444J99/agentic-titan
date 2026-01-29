"""
Titan Learning - RLHF Deployment

A/B testing and safe rollout for RLHF-trained models.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger("titan.learning.deployment")


@dataclass
class DeploymentConfig:
    """Configuration for A/B testing deployment."""

    # Traffic split
    canary_percentage: float = 0.1  # Start with 10% traffic to new model

    # Promotion criteria
    min_samples_before_promote: int = 100
    quality_threshold: float = 0.05  # Must be this much better than baseline
    confidence_threshold: float = 0.95  # Statistical confidence required

    # Rollback criteria
    rollback_threshold: float = -0.1  # Auto-rollback if worse by this amount
    max_errors_per_minute: int = 10  # Error rate threshold
    latency_threshold_ms: float = 5000.0  # Max latency threshold

    # Gradual rollout
    enable_gradual_rollout: bool = True
    rollout_increments: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0]
    )
    hours_between_increments: float = 24.0

    # Storage
    state_dir: str = "./deployment_state"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canary_percentage": self.canary_percentage,
            "min_samples_before_promote": self.min_samples_before_promote,
            "quality_threshold": self.quality_threshold,
            "confidence_threshold": self.confidence_threshold,
            "rollback_threshold": self.rollback_threshold,
            "max_errors_per_minute": self.max_errors_per_minute,
            "latency_threshold_ms": self.latency_threshold_ms,
            "enable_gradual_rollout": self.enable_gradual_rollout,
            "rollout_increments": self.rollout_increments,
            "hours_between_increments": self.hours_between_increments,
            "state_dir": self.state_dir,
        }


@dataclass
class ModelMetrics:
    """Metrics collected for a model in A/B test."""

    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    total_reward: float = 0.0
    positive_feedback: int = 0
    negative_feedback: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.requests if self.requests > 0 else 0.0

    @property
    def feedback_score(self) -> float:
        total = self.positive_feedback + self.negative_feedback
        return self.positive_feedback / total if total > 0 else 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_reward": self.avg_reward,
            "feedback_score": self.feedback_score,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
        }


@dataclass
class ABTestStats:
    """Statistics for an A/B test."""

    test_id: str
    new_model: str
    baseline: str
    status: str  # running, promoting, rolled_back, completed
    started_at: datetime
    updated_at: datetime

    # Traffic
    current_percentage: float
    target_percentage: float

    # Metrics
    new_model_metrics: ModelMetrics
    baseline_metrics: ModelMetrics

    # Results
    relative_improvement: float = 0.0
    confidence: float = 0.0
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "new_model": self.new_model,
            "baseline": self.baseline,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_percentage": self.current_percentage,
            "target_percentage": self.target_percentage,
            "new_model_metrics": self.new_model_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "relative_improvement": self.relative_improvement,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
        }


@dataclass
class ABTest:
    """An active A/B test."""

    test_id: str = field(default_factory=lambda: str(uuid4())[:8])
    new_model: str = ""
    baseline: str = ""
    config: DeploymentConfig = field(default_factory=DeploymentConfig)

    status: str = "running"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    current_percentage: float = 0.1
    rollout_stage: int = 0

    new_model_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    baseline_metrics: ModelMetrics = field(default_factory=ModelMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "new_model": self.new_model,
            "baseline": self.baseline,
            "config": self.config.to_dict(),
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_percentage": self.current_percentage,
            "rollout_stage": self.rollout_stage,
            "new_model_metrics": self.new_model_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ABTest:
        test = cls(
            test_id=data.get("test_id", str(uuid4())[:8]),
            new_model=data.get("new_model", ""),
            baseline=data.get("baseline", ""),
            status=data.get("status", "running"),
            current_percentage=data.get("current_percentage", 0.1),
            rollout_stage=data.get("rollout_stage", 0),
        )
        if data.get("started_at"):
            test.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("updated_at"):
            test.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("completed_at"):
            test.completed_at = datetime.fromisoformat(data["completed_at"])
        return test


class RLHFDeployment:
    """
    A/B testing and safe rollout for RLHF models.

    Features:
    - Traffic splitting between new and baseline models
    - Real-time metrics collection
    - Automatic rollback on quality degradation
    - Gradual rollout with configurable increments
    - Statistical significance testing
    """

    def __init__(self, config: DeploymentConfig | None = None) -> None:
        self._config = config or DeploymentConfig()
        self._active_tests: dict[str, ABTest] = {}
        self._completed_tests: list[ABTest] = []

        # Ensure state directory exists
        os.makedirs(self._config.state_dir, exist_ok=True)

        # Load existing tests
        self._load_state()

    def start_ab_test(
        self,
        new_model: str,
        baseline: str,
        config: DeploymentConfig | None = None,
    ) -> str:
        """
        Start a new A/B test.

        Args:
            new_model: Path or identifier for new model
            baseline: Path or identifier for baseline model
            config: Optional custom deployment config

        Returns:
            Test ID
        """
        test_config = config or self._config

        test = ABTest(
            new_model=new_model,
            baseline=baseline,
            config=test_config,
            current_percentage=test_config.canary_percentage,
        )

        self._active_tests[test.test_id] = test
        self._save_state()

        logger.info(
            f"Started A/B test {test.test_id}: {new_model} vs {baseline} "
            f"({test.current_percentage:.0%} traffic to new model)"
        )

        return test.test_id

    def get_ab_stats(self, test_id: str) -> ABTestStats | None:
        """
        Get statistics for an A/B test.

        Args:
            test_id: Test ID

        Returns:
            ABTestStats or None if not found
        """
        test = self._active_tests.get(test_id)
        if not test:
            # Check completed tests
            for completed in self._completed_tests:
                if completed.test_id == test_id:
                    test = completed
                    break

        if not test:
            return None

        # Calculate relative improvement
        baseline_score = test.baseline_metrics.avg_reward
        new_score = test.new_model_metrics.avg_reward

        if baseline_score != 0:
            relative_improvement = (new_score - baseline_score) / abs(baseline_score)
        else:
            relative_improvement = new_score

        # Calculate statistical confidence
        confidence = self._calculate_confidence(test)

        # Generate recommendation
        recommendation = self._generate_recommendation(test, relative_improvement, confidence)

        return ABTestStats(
            test_id=test.test_id,
            new_model=test.new_model,
            baseline=test.baseline,
            status=test.status,
            started_at=test.started_at,
            updated_at=test.updated_at,
            current_percentage=test.current_percentage,
            target_percentage=1.0 if test.status == "promoting" else test.current_percentage,
            new_model_metrics=test.new_model_metrics,
            baseline_metrics=test.baseline_metrics,
            relative_improvement=relative_improvement,
            confidence=confidence,
            recommendation=recommendation,
        )

    def route_request(self, test_id: str) -> str:
        """
        Route a request to either new or baseline model.

        Args:
            test_id: Test ID

        Returns:
            Model identifier to use
        """
        test = self._active_tests.get(test_id)
        if not test or test.status != "running":
            # Fall back to baseline
            return test.baseline if test else ""

        # Random routing based on percentage
        if random.random() < test.current_percentage:
            return test.new_model
        return test.baseline

    def record_result(
        self,
        test_id: str,
        model: str,
        latency_ms: float,
        reward: float = 0.0,
        error: bool = False,
        feedback: int | None = None,  # 1 for positive, -1 for negative
    ) -> None:
        """
        Record a request result for A/B test.

        Args:
            test_id: Test ID
            model: Model that served the request
            latency_ms: Request latency in milliseconds
            reward: Reward score for the response
            error: Whether an error occurred
            feedback: User feedback (1/-1)
        """
        test = self._active_tests.get(test_id)
        if not test:
            return

        # Update appropriate metrics
        if model == test.new_model:
            metrics = test.new_model_metrics
        elif model == test.baseline:
            metrics = test.baseline_metrics
        else:
            return

        metrics.requests += 1
        metrics.total_latency_ms += latency_ms
        metrics.total_reward += reward

        if error:
            metrics.errors += 1

        if feedback == 1:
            metrics.positive_feedback += 1
        elif feedback == -1:
            metrics.negative_feedback += 1

        test.updated_at = datetime.now(timezone.utc)

        # Check for automatic actions
        self._check_auto_actions(test)
        self._save_state()

    def promote_winner(self, test_id: str) -> bool:
        """
        Promote the new model as winner.

        Args:
            test_id: Test ID

        Returns:
            True if promotion successful
        """
        test = self._active_tests.get(test_id)
        if not test:
            logger.warning(f"Test {test_id} not found")
            return False

        if test.status != "running":
            logger.warning(f"Test {test_id} is not running (status: {test.status})")
            return False

        # Start gradual promotion
        if test.config.enable_gradual_rollout:
            test.status = "promoting"
            test.rollout_stage += 1
            if test.rollout_stage < len(test.config.rollout_increments):
                test.current_percentage = test.config.rollout_increments[test.rollout_stage]
            else:
                test.current_percentage = 1.0
                test.status = "completed"
                test.completed_at = datetime.now(timezone.utc)
                self._complete_test(test)
        else:
            # Immediate promotion
            test.current_percentage = 1.0
            test.status = "completed"
            test.completed_at = datetime.now(timezone.utc)
            self._complete_test(test)

        self._save_state()
        logger.info(
            f"Promoted test {test_id}: new model at {test.current_percentage:.0%} traffic"
        )
        return True

    def rollback(self, test_id: str) -> bool:
        """
        Rollback to baseline model.

        Args:
            test_id: Test ID

        Returns:
            True if rollback successful
        """
        test = self._active_tests.get(test_id)
        if not test:
            logger.warning(f"Test {test_id} not found")
            return False

        test.status = "rolled_back"
        test.current_percentage = 0.0
        test.completed_at = datetime.now(timezone.utc)

        self._complete_test(test)
        self._save_state()

        logger.warning(f"Rolled back test {test_id} to baseline")
        return True

    def advance_rollout(self, test_id: str) -> bool:
        """
        Advance to next rollout stage.

        Args:
            test_id: Test ID

        Returns:
            True if advanced, False if already at max
        """
        test = self._active_tests.get(test_id)
        if not test or test.status not in ("running", "promoting"):
            return False

        test.rollout_stage += 1
        if test.rollout_stage < len(test.config.rollout_increments):
            test.current_percentage = test.config.rollout_increments[test.rollout_stage]
            test.updated_at = datetime.now(timezone.utc)
            self._save_state()
            logger.info(
                f"Advanced test {test_id} to stage {test.rollout_stage}: "
                f"{test.current_percentage:.0%} traffic"
            )
            return True
        else:
            # Reached full rollout
            return self.promote_winner(test_id)

    def list_active_tests(self) -> list[ABTestStats]:
        """List all active A/B tests."""
        return [
            self.get_ab_stats(test_id)
            for test_id in self._active_tests
            if self.get_ab_stats(test_id) is not None
        ]

    def _check_auto_actions(self, test: ABTest) -> None:
        """Check if automatic rollback or promotion is needed."""
        if test.status != "running":
            return

        config = test.config

        # Check for rollback conditions
        new_metrics = test.new_model_metrics
        baseline_metrics = test.baseline_metrics

        # Error rate check
        if (
            new_metrics.requests >= 10
            and new_metrics.error_rate > baseline_metrics.error_rate * 2
        ):
            logger.warning(f"Test {test.test_id}: High error rate, rolling back")
            self.rollback(test.test_id)
            return

        # Latency check
        if (
            new_metrics.requests >= 10
            and new_metrics.avg_latency_ms > config.latency_threshold_ms
        ):
            logger.warning(f"Test {test.test_id}: High latency, rolling back")
            self.rollback(test.test_id)
            return

        # Quality check
        total_samples = new_metrics.requests + baseline_metrics.requests
        if total_samples >= config.min_samples_before_promote:
            relative_improvement = self._calculate_relative_improvement(test)

            if relative_improvement < config.rollback_threshold:
                logger.warning(
                    f"Test {test.test_id}: Quality below threshold "
                    f"({relative_improvement:.2%}), rolling back"
                )
                self.rollback(test.test_id)
                return

            # Check for auto-promotion
            confidence = self._calculate_confidence(test)
            if (
                relative_improvement >= config.quality_threshold
                and confidence >= config.confidence_threshold
            ):
                logger.info(
                    f"Test {test.test_id}: Quality threshold met "
                    f"({relative_improvement:.2%} improvement, {confidence:.2%} confidence)"
                )
                # Don't auto-promote, but log

    def _calculate_relative_improvement(self, test: ABTest) -> float:
        """Calculate relative improvement of new model over baseline."""
        baseline_score = test.baseline_metrics.avg_reward
        new_score = test.new_model_metrics.avg_reward

        if baseline_score != 0:
            return (new_score - baseline_score) / abs(baseline_score)
        return new_score

    def _calculate_confidence(self, test: ABTest) -> float:
        """Calculate statistical confidence using approximate z-test."""
        n1 = test.new_model_metrics.requests
        n2 = test.baseline_metrics.requests

        if n1 < 10 or n2 < 10:
            return 0.0

        # Use feedback scores as proportion
        p1 = test.new_model_metrics.feedback_score
        p2 = test.baseline_metrics.feedback_score

        # Pooled proportion
        p = (
            test.new_model_metrics.positive_feedback + test.baseline_metrics.positive_feedback
        ) / (
            test.new_model_metrics.positive_feedback
            + test.new_model_metrics.negative_feedback
            + test.baseline_metrics.positive_feedback
            + test.baseline_metrics.negative_feedback
            + 1  # Avoid division by zero
        )

        # Standard error
        import math

        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) if p > 0 and p < 1 else 0.5

        # Z-score
        z = abs(p1 - p2) / se if se > 0 else 0

        # Approximate confidence from z-score
        # Using normal CDF approximation
        if z > 3.0:
            return 0.999
        elif z > 2.576:
            return 0.99
        elif z > 1.96:
            return 0.95
        elif z > 1.645:
            return 0.90
        else:
            return 0.5 + (z / 3.0) * 0.4

    def _generate_recommendation(
        self, test: ABTest, relative_improvement: float, confidence: float
    ) -> str:
        """Generate recommendation based on test results."""
        config = test.config

        if test.status == "rolled_back":
            return "ROLLED_BACK: New model showed degraded performance"

        if test.status == "completed":
            return "COMPLETED: New model fully deployed"

        total_samples = test.new_model_metrics.requests + test.baseline_metrics.requests

        if total_samples < config.min_samples_before_promote:
            return f"COLLECTING_DATA: Need {config.min_samples_before_promote - total_samples} more samples"

        if relative_improvement < config.rollback_threshold:
            return "ROLLBACK_RECOMMENDED: New model significantly underperforming"

        if relative_improvement >= config.quality_threshold and confidence >= config.confidence_threshold:
            return "PROMOTE_RECOMMENDED: New model shows significant improvement"

        if relative_improvement >= 0:
            return "CONTINUE_TESTING: Positive trend, but not yet statistically significant"

        return "MONITOR: Mixed results, continue collecting data"

    def _complete_test(self, test: ABTest) -> None:
        """Move test from active to completed."""
        if test.test_id in self._active_tests:
            del self._active_tests[test.test_id]
            self._completed_tests.append(test)

    def _save_state(self) -> None:
        """Save deployment state to disk."""
        state_file = Path(self._config.state_dir) / "deployment_state.json"
        state = {
            "active_tests": {k: v.to_dict() for k, v in self._active_tests.items()},
            "completed_tests": [t.to_dict() for t in self._completed_tests[-100:]],  # Keep last 100
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load deployment state from disk."""
        state_file = Path(self._config.state_dir) / "deployment_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            for test_id, test_data in state.get("active_tests", {}).items():
                self._active_tests[test_id] = ABTest.from_dict(test_data)

            for test_data in state.get("completed_tests", []):
                self._completed_tests.append(ABTest.from_dict(test_data))

        except Exception as e:
            logger.warning(f"Error loading deployment state: {e}")


# Factory function
_deployment: RLHFDeployment | None = None


def get_rlhf_deployment(config: DeploymentConfig | None = None) -> RLHFDeployment:
    """Get the default RLHF deployment manager."""
    global _deployment
    if _deployment is None or config is not None:
        _deployment = RLHFDeployment(config)
    return _deployment
