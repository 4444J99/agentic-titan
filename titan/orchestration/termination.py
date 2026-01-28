"""
Titan Orchestration - Termination Conditions

Defines formal termination conditions for workflows and agent execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("titan.orchestration.termination")


class TerminationReason(str, Enum):
    """Reasons for workflow termination."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    MAX_ITERATIONS = "max_iterations"
    MAX_COST = "max_cost"
    FAILURE_THRESHOLD = "failure_threshold"
    MANUAL_STOP = "manual_stop"
    CIRCUIT_BREAKER = "circuit_breaker"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONDITION_MET = "condition_met"


@dataclass
class TerminationResult:
    """Result of termination check."""

    should_terminate: bool
    reason: TerminationReason | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "should_terminate": self.should_terminate,
            "reason": self.reason.value if self.reason else None,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowState:
    """State of a workflow for termination checking."""

    workflow_id: str
    start_time: datetime
    iteration_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_cost_usd: float = 0.0
    memory_usage_mb: float = 0.0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get workflow duration in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def failure_rate(self) -> float:
        """Get failure rate."""
        total = self.success_count + self.failure_count
        return self.failure_count / total if total > 0 else 0.0


class TerminationCondition(ABC):
    """Abstract base class for termination conditions."""

    name: str = "base_condition"

    @abstractmethod
    def check(self, state: WorkflowState) -> TerminationResult:
        """
        Check if termination condition is met.

        Args:
            state: Current workflow state

        Returns:
            TerminationResult indicating whether to terminate
        """
        pass


class TimeoutCondition(TerminationCondition):
    """Terminate after a maximum duration."""

    name = "timeout"

    def __init__(self, max_duration_seconds: float) -> None:
        self.max_duration_seconds = max_duration_seconds

    def check(self, state: WorkflowState) -> TerminationResult:
        if state.duration_seconds >= self.max_duration_seconds:
            return TerminationResult(
                should_terminate=True,
                reason=TerminationReason.TIMEOUT,
                message=f"Timeout after {state.duration_seconds:.1f}s "
                f"(max: {self.max_duration_seconds}s)",
            )
        return TerminationResult(should_terminate=False)


class MaxIterationsCondition(TerminationCondition):
    """Terminate after maximum iterations."""

    name = "max_iterations"

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations

    def check(self, state: WorkflowState) -> TerminationResult:
        if state.iteration_count >= self.max_iterations:
            return TerminationResult(
                should_terminate=True,
                reason=TerminationReason.MAX_ITERATIONS,
                message=f"Max iterations reached: {state.iteration_count}",
            )
        return TerminationResult(should_terminate=False)


class MaxCostCondition(TerminationCondition):
    """Terminate when cost limit is exceeded."""

    name = "max_cost"

    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = max_cost_usd

    def check(self, state: WorkflowState) -> TerminationResult:
        if state.total_cost_usd >= self.max_cost_usd:
            return TerminationResult(
                should_terminate=True,
                reason=TerminationReason.MAX_COST,
                message=f"Cost limit exceeded: ${state.total_cost_usd:.4f} "
                f"(max: ${self.max_cost_usd:.4f})",
            )
        return TerminationResult(should_terminate=False)


class FailureThresholdCondition(TerminationCondition):
    """Terminate when failure rate exceeds threshold."""

    name = "failure_threshold"

    def __init__(
        self,
        max_failure_rate: float = 0.5,
        min_attempts: int = 3,
    ) -> None:
        self.max_failure_rate = max_failure_rate
        self.min_attempts = min_attempts

    def check(self, state: WorkflowState) -> TerminationResult:
        total = state.success_count + state.failure_count
        if total >= self.min_attempts and state.failure_rate > self.max_failure_rate:
            return TerminationResult(
                should_terminate=True,
                reason=TerminationReason.FAILURE_THRESHOLD,
                message=f"Failure rate {state.failure_rate:.1%} exceeds "
                f"threshold {self.max_failure_rate:.1%}",
            )
        return TerminationResult(should_terminate=False)


class SuccessCondition(TerminationCondition):
    """Terminate when success criteria is met."""

    name = "success"

    def __init__(
        self,
        success_criteria: Callable[[WorkflowState], bool],
        description: str = "Custom success criteria",
    ) -> None:
        self.success_criteria = success_criteria
        self.description = description

    def check(self, state: WorkflowState) -> TerminationResult:
        if self.success_criteria(state):
            return TerminationResult(
                should_terminate=True,
                reason=TerminationReason.SUCCESS,
                message=self.description,
            )
        return TerminationResult(should_terminate=False)


class ResourceExhaustionCondition(TerminationCondition):
    """Terminate when resources are exhausted."""

    name = "resource_exhaustion"

    def __init__(self, max_memory_mb: float = 1024) -> None:
        self.max_memory_mb = max_memory_mb

    def check(self, state: WorkflowState) -> TerminationResult:
        if state.memory_usage_mb >= self.max_memory_mb:
            return TerminationResult(
                should_terminate=True,
                reason=TerminationReason.RESOURCE_EXHAUSTION,
                message=f"Memory limit exceeded: {state.memory_usage_mb:.1f}MB "
                f"(max: {self.max_memory_mb:.1f}MB)",
            )
        return TerminationResult(should_terminate=False)


class CompositeTerminationCondition(TerminationCondition):
    """Combines multiple termination conditions."""

    name = "composite"

    def __init__(
        self,
        conditions: list[TerminationCondition],
        require_all: bool = False,
    ) -> None:
        """
        Args:
            conditions: List of conditions to check
            require_all: If True, all must be met; if False, any one triggers termination
        """
        self.conditions = conditions
        self.require_all = require_all

    def check(self, state: WorkflowState) -> TerminationResult:
        results = [c.check(state) for c in self.conditions]
        termination_results = [r for r in results if r.should_terminate]

        if self.require_all:
            if len(termination_results) == len(self.conditions):
                reasons = [r.reason.value for r in termination_results if r.reason]
                return TerminationResult(
                    should_terminate=True,
                    reason=TerminationReason.CONDITION_MET,
                    message=f"All conditions met: {', '.join(reasons)}",
                )
        else:
            if termination_results:
                first = termination_results[0]
                return TerminationResult(
                    should_terminate=True,
                    reason=first.reason,
                    message=first.message,
                )

        return TerminationResult(should_terminate=False)


class DefaultTerminationConditions:
    """Factory for common termination condition sets."""

    @staticmethod
    def for_agent(
        max_turns: int = 20,
        timeout_seconds: float = 300,
        max_cost_usd: float = 2.0,
    ) -> CompositeTerminationCondition:
        """Default conditions for agent execution."""
        return CompositeTerminationCondition([
            MaxIterationsCondition(max_turns),
            TimeoutCondition(timeout_seconds),
            MaxCostCondition(max_cost_usd),
            FailureThresholdCondition(max_failure_rate=0.8, min_attempts=3),
        ])

    @staticmethod
    def for_workflow(
        max_iterations: int = 100,
        timeout_seconds: float = 3600,
        max_cost_usd: float = 50.0,
    ) -> CompositeTerminationCondition:
        """Default conditions for workflow execution."""
        return CompositeTerminationCondition([
            MaxIterationsCondition(max_iterations),
            TimeoutCondition(timeout_seconds),
            MaxCostCondition(max_cost_usd),
            FailureThresholdCondition(max_failure_rate=0.5, min_attempts=5),
            ResourceExhaustionCondition(max_memory_mb=2048),
        ])

    @staticmethod
    def strict(
        max_iterations: int = 10,
        timeout_seconds: float = 60,
    ) -> CompositeTerminationCondition:
        """Strict conditions for testing or limited operations."""
        return CompositeTerminationCondition([
            MaxIterationsCondition(max_iterations),
            TimeoutCondition(timeout_seconds),
            FailureThresholdCondition(max_failure_rate=0.3, min_attempts=2),
        ])
