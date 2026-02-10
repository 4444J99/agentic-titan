"""
Titan Orchestration - Execution Watchdog

Monitors execution and enforces termination conditions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from titan.orchestration.termination import (
    CompositeTerminationCondition,
    DefaultTerminationConditions,
    TerminationCondition,
    TerminationReason,
    TerminationResult,
    WorkflowState,
)

if TYPE_CHECKING:
    from agents.framework.resilience import CircuitBreaker

logger = logging.getLogger("titan.orchestration.watchdog")


class AlertLevel(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class WatchdogAlert:
    """Alert raised by the watchdog."""

    id: UUID = field(default_factory=uuid4)
    level: AlertLevel = AlertLevel.INFO
    workflow_id: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "level": self.level.value,
            "workflow_id": self.workflow_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class WatchdogConfig:
    """Configuration for the execution watchdog."""

    # Check interval
    check_interval_seconds: float = 5.0

    # Default termination conditions
    default_timeout_seconds: float = 300.0
    default_max_iterations: int = 100
    default_max_cost_usd: float = 10.0

    # Alert thresholds
    warn_at_iteration_percent: float = 80.0
    warn_at_time_percent: float = 80.0
    warn_at_cost_percent: float = 80.0

    # Memory monitoring
    enable_memory_monitoring: bool = True
    memory_check_interval_seconds: float = 30.0
    memory_warning_threshold_mb: float = 512.0

    # Circuit breaker integration
    connect_circuit_breakers: bool = True


# Type for alert callbacks
AlertCallback = Callable[[WatchdogAlert], Coroutine[Any, Any, None]]


class ExecutionWatchdog:
    """
    Monitors workflow execution and enforces termination.

    Features:
    - Periodic health checks
    - Termination condition evaluation
    - Circuit breaker integration
    - Alert notifications
    - Forced termination capability
    """

    def __init__(self, config: WatchdogConfig | None = None) -> None:
        self.config = config or WatchdogConfig()
        self._workflows: dict[str, WorkflowState] = {}
        self._conditions: dict[str, TerminationCondition] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._alert_callbacks: list[AlertCallback] = []
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the watchdog."""
        self._running = True
        logger.info("Execution watchdog started")

    async def stop(self) -> None:
        """Stop the watchdog and all monitoring tasks."""
        self._running = False

        # Cancel all monitoring tasks
        for task in self._monitor_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks.values(), return_exceptions=True)

        self._monitor_tasks.clear()
        logger.info("Execution watchdog stopped")

    def add_alert_callback(self, callback: AlertCallback) -> None:
        """Add callback for alert notifications."""
        self._alert_callbacks.append(callback)

    async def register_workflow(
        self,
        workflow_id: str,
        conditions: TerminationCondition | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """
        Register a workflow for monitoring.

        Args:
            workflow_id: Unique workflow identifier
            conditions: Custom termination conditions
            circuit_breaker: Circuit breaker to connect
        """
        async with self._lock:
            # Create initial state
            state = WorkflowState(
                workflow_id=workflow_id,
                start_time=datetime.now(UTC),
            )
            self._workflows[workflow_id] = state

            # Set termination conditions
            if conditions:
                self._conditions[workflow_id] = conditions
            else:
                self._conditions[workflow_id] = DefaultTerminationConditions.for_workflow(
                    timeout_seconds=self.config.default_timeout_seconds,
                    max_iterations=self.config.default_max_iterations,
                    max_cost_usd=self.config.default_max_cost_usd,
                )

            # Connect circuit breaker
            if circuit_breaker and self.config.connect_circuit_breakers:
                self._circuit_breakers[workflow_id] = circuit_breaker

            # Start monitoring task
            task = asyncio.create_task(self._monitor_workflow(workflow_id))
            self._monitor_tasks[workflow_id] = task

            logger.info(f"Registered workflow for monitoring: {workflow_id}")

    async def unregister_workflow(self, workflow_id: str) -> None:
        """Unregister a workflow from monitoring."""
        async with self._lock:
            # Cancel monitoring task
            if workflow_id in self._monitor_tasks:
                self._monitor_tasks[workflow_id].cancel()
                del self._monitor_tasks[workflow_id]

            # Clean up state
            self._workflows.pop(workflow_id, None)
            self._conditions.pop(workflow_id, None)
            self._circuit_breakers.pop(workflow_id, None)

            logger.info(f"Unregistered workflow: {workflow_id}")

    async def update_state(
        self,
        workflow_id: str,
        iteration_count: int | None = None,
        success_count: int | None = None,
        failure_count: int | None = None,
        total_cost_usd: float | None = None,
        custom_metrics: dict[str, Any] | None = None,
    ) -> TerminationResult:
        """
        Update workflow state and check termination.

        Args:
            workflow_id: Workflow identifier
            iteration_count: New iteration count
            success_count: New success count
            failure_count: New failure count
            total_cost_usd: New total cost
            custom_metrics: Additional metrics

        Returns:
            TerminationResult indicating if workflow should stop
        """
        async with self._lock:
            state = self._workflows.get(workflow_id)
            if not state:
                return TerminationResult(
                    should_terminate=True,
                    reason=TerminationReason.MANUAL_STOP,
                    message="Workflow not registered",
                )

            # Update state
            if iteration_count is not None:
                state.iteration_count = iteration_count
            if success_count is not None:
                state.success_count = success_count
            if failure_count is not None:
                state.failure_count = failure_count
            if total_cost_usd is not None:
                state.total_cost_usd = total_cost_usd
            if custom_metrics:
                state.custom_metrics.update(custom_metrics)

            # Check termination
            return await self._check_termination(workflow_id)

    async def _check_termination(self, workflow_id: str) -> TerminationResult:
        """Check if workflow should terminate."""
        state = self._workflows.get(workflow_id)
        condition = self._conditions.get(workflow_id)

        if not state or not condition:
            return TerminationResult(should_terminate=False)

        # Check condition
        result = condition.check(state)

        # Also check circuit breaker if connected
        circuit_breaker = self._circuit_breakers.get(workflow_id)
        if circuit_breaker and not result.should_terminate:
            if circuit_breaker.state.value == "open":
                result = TerminationResult(
                    should_terminate=True,
                    reason=TerminationReason.CIRCUIT_BREAKER,
                    message="Circuit breaker is open",
                )

        # Trigger alerts based on thresholds
        await self._check_alert_thresholds(workflow_id, state)

        return result

    async def _check_alert_thresholds(
        self,
        workflow_id: str,
        state: WorkflowState,
    ) -> None:
        """Check if any alert thresholds are crossed."""
        condition = self._conditions.get(workflow_id)
        if not condition:
            return

        alerts = []

        # Check iteration threshold
        if isinstance(condition, CompositeTerminationCondition):
            for sub in condition.conditions:
                if hasattr(sub, "max_iterations"):
                    percent = (state.iteration_count / sub.max_iterations) * 100
                    if percent >= self.config.warn_at_iteration_percent:
                        alerts.append(
                            WatchdogAlert(
                                level=AlertLevel.WARNING,
                                workflow_id=workflow_id,
                                message=f"Iteration count at {percent:.0f}% of limit",
                                metadata={
                                    "current": state.iteration_count,
                                    "max": sub.max_iterations,
                                },
                            )
                        )
                        break

                if hasattr(sub, "max_duration_seconds"):
                    percent = (state.duration_seconds / sub.max_duration_seconds) * 100
                    if percent >= self.config.warn_at_time_percent:
                        alerts.append(
                            WatchdogAlert(
                                level=AlertLevel.WARNING,
                                workflow_id=workflow_id,
                                message=f"Duration at {percent:.0f}% of timeout",
                                metadata={
                                    "current": state.duration_seconds,
                                    "max": sub.max_duration_seconds,
                                },
                            )
                        )
                        break

                if hasattr(sub, "max_cost_usd"):
                    percent = (state.total_cost_usd / sub.max_cost_usd) * 100
                    if percent >= self.config.warn_at_cost_percent:
                        alerts.append(
                            WatchdogAlert(
                                level=AlertLevel.WARNING,
                                workflow_id=workflow_id,
                                message=f"Cost at {percent:.0f}% of budget",
                                metadata={"current": state.total_cost_usd, "max": sub.max_cost_usd},
                            )
                        )
                        break

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

    async def _monitor_workflow(self, workflow_id: str) -> None:
        """Background task to monitor a workflow."""
        logger.debug(f"Started monitoring workflow: {workflow_id}")

        while self._running and workflow_id in self._workflows:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)

                # Check memory if enabled
                if self.config.enable_memory_monitoring:
                    await self._check_memory(workflow_id)

                # Check termination conditions
                result = await self._check_termination(workflow_id)
                if result.should_terminate:
                    logger.warning(f"Workflow {workflow_id} should terminate: {result.message}")
                    await self._send_alert(
                        WatchdogAlert(
                            level=AlertLevel.CRITICAL,
                            workflow_id=workflow_id,
                            message=f"Termination triggered: {result.message}",
                            metadata=result.to_dict(),
                        )
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error for {workflow_id}: {e}")

        logger.debug(f"Stopped monitoring workflow: {workflow_id}")

    async def _check_memory(self, workflow_id: str) -> None:
        """Check memory usage for a workflow."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)

            state = self._workflows.get(workflow_id)
            if state:
                state.memory_usage_mb = memory_mb

                if memory_mb > self.config.memory_warning_threshold_mb:
                    await self._send_alert(
                        WatchdogAlert(
                            level=AlertLevel.WARNING,
                            workflow_id=workflow_id,
                            message=f"High memory usage: {memory_mb:.1f}MB",
                            metadata={"memory_mb": memory_mb},
                        )
                    )
        except ImportError:
            pass  # psutil not available
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

    async def _send_alert(self, alert: WatchdogAlert) -> None:
        """Send alert to all registered callbacks."""
        logger.log(
            logging.WARNING if alert.level == AlertLevel.CRITICAL else logging.INFO,
            f"Watchdog alert [{alert.level.value}]: {alert.message}",
        )

        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def force_terminate(
        self,
        workflow_id: str,
        reason: str = "Manual termination",
    ) -> bool:
        """
        Force terminate a workflow.

        Args:
            workflow_id: Workflow to terminate
            reason: Reason for termination

        Returns:
            True if workflow was terminated
        """
        async with self._lock:
            if workflow_id not in self._workflows:
                return False

            await self._send_alert(
                WatchdogAlert(
                    level=AlertLevel.CRITICAL,
                    workflow_id=workflow_id,
                    message=f"Force terminated: {reason}",
                )
            )

            # Unregister the workflow
            await self.unregister_workflow(workflow_id)

            logger.warning(f"Force terminated workflow: {workflow_id} - {reason}")
            return True

    def get_workflow_state(self, workflow_id: str) -> WorkflowState | None:
        """Get current state of a workflow."""
        return self._workflows.get(workflow_id)

    def get_all_workflows(self) -> dict[str, WorkflowState]:
        """Get all monitored workflows."""
        return self._workflows.copy()


# Singleton instance
_default_watchdog: ExecutionWatchdog | None = None


def get_watchdog(config: WatchdogConfig | None = None) -> ExecutionWatchdog:
    """Get the default execution watchdog."""
    global _default_watchdog
    if _default_watchdog is None:
        _default_watchdog = ExecutionWatchdog(config)
    return _default_watchdog


async def init_watchdog(config: WatchdogConfig | None = None) -> ExecutionWatchdog:
    """Initialize and start the watchdog."""
    watchdog = get_watchdog(config)
    await watchdog.start()
    return watchdog
