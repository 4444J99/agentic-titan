"""
Titan Batch - Recovery Utilities

Batch recovery strategies for handling stalled and failed batches.
Provides configurable recovery mechanisms for different failure scenarios.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from titan.batch.models import BatchJob, QueuedSession
    from titan.batch.orchestrator import BatchOrchestrator

logger = logging.getLogger("titan.batch.recovery")


class RecoveryStrategy(StrEnum):
    """Recovery strategies for stalled or failed batches."""

    RETRY = "retry"  # Retry failed sessions
    SKIP = "skip"  # Skip failed, continue with rest
    FAIL = "fail"  # Mark whole batch as failed
    MANUAL = "manual"  # Pause for manual intervention


class RecoveryResult:
    """Result of a recovery operation."""

    def __init__(
        self,
        success: bool,
        batch_id: UUID,
        strategy: RecoveryStrategy,
        sessions_recovered: int = 0,
        sessions_failed: int = 0,
        message: str = "",
    ) -> None:
        self.success = success
        self.batch_id = batch_id
        self.strategy = strategy
        self.sessions_recovered = sessions_recovered
        self.sessions_failed = sessions_failed
        self.message = message
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "batch_id": str(self.batch_id),
            "strategy": self.strategy.value,
            "sessions_recovered": self.sessions_recovered,
            "sessions_failed": self.sessions_failed,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


async def recover_batch(
    orchestrator: BatchOrchestrator,
    batch_id: UUID | str,
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    max_retry_count: int = 3,
) -> RecoveryResult:
    """
    Recover a stalled or failed batch using the specified strategy.

    Args:
        orchestrator: Batch orchestrator instance
        batch_id: ID of batch to recover
        strategy: Recovery strategy to use
        max_retry_count: Maximum retry attempts for RETRY strategy

    Returns:
        RecoveryResult with operation outcome
    """
    target = UUID(batch_id) if isinstance(batch_id, str) else batch_id
    batch = orchestrator.get_batch(target)

    if not batch:
        return RecoveryResult(
            success=False,
            batch_id=target,
            strategy=strategy,
            message=f"Batch not found: {target}",
        )

    logger.info(f"Recovering batch {target} with strategy {strategy.value}")

    if strategy == RecoveryStrategy.RETRY:
        return await _recover_retry(orchestrator, batch, max_retry_count)
    elif strategy == RecoveryStrategy.SKIP:
        return await _recover_skip(orchestrator, batch)
    elif strategy == RecoveryStrategy.FAIL:
        return await _recover_fail(orchestrator, batch)
    elif strategy == RecoveryStrategy.MANUAL:
        return await _recover_manual(orchestrator, batch)
    else:
        return RecoveryResult(
            success=False,
            batch_id=target,
            strategy=strategy,
            message=f"Unknown strategy: {strategy}",
        )


async def _recover_retry(
    orchestrator: BatchOrchestrator,
    batch: BatchJob,
    max_retry_count: int,
) -> RecoveryResult:
    """
    Retry strategy: Reset stalled sessions and re-queue them.

    Sessions that have exceeded max_retry_count are marked as failed.
    """
    from titan.batch.models import SessionQueueStatus

    recovered = 0
    failed = 0

    for session in batch.sessions:
        if session.status == SessionQueueStatus.RUNNING:
            # Stalled running session
            if session.retry_count < max_retry_count:
                session.status = SessionQueueStatus.PENDING
                session.retry_count += 1
                session.error = "Recovered from stalled state"
                session.worker_id = None
                session.celery_task_id = None
                recovered += 1
                logger.debug(f"Reset stalled session {session.id} for retry")
            else:
                session.status = SessionQueueStatus.FAILED
                session.error = f"Exceeded max retries ({max_retry_count})"
                session.completed_at = datetime.now()
                failed += 1
                logger.warning(f"Session {session.id} exceeded max retries")

    # Re-dispatch pending sessions
    if recovered > 0:
        await orchestrator._queue_pending_sessions(batch)
        await orchestrator._persist_batch(batch)

    return RecoveryResult(
        success=recovered > 0 or failed > 0,
        batch_id=batch.id,
        strategy=RecoveryStrategy.RETRY,
        sessions_recovered=recovered,
        sessions_failed=failed,
        message=f"Recovered {recovered} sessions, {failed} exceeded retries",
    )


async def _recover_skip(
    orchestrator: BatchOrchestrator,
    batch: BatchJob,
) -> RecoveryResult:
    """
    Skip strategy: Mark stalled sessions as failed and continue.

    Allows the batch to complete with partial results.
    """
    from titan.batch.models import SessionQueueStatus

    skipped = 0

    for session in batch.sessions:
        if session.status == SessionQueueStatus.RUNNING:
            session.status = SessionQueueStatus.FAILED
            session.error = "Skipped - stalled without progress"
            session.completed_at = datetime.now()
            skipped += 1
            logger.info(f"Skipped stalled session {session.id}")

    # Check if batch should complete
    await orchestrator._check_batch_completion(batch)
    await orchestrator._persist_batch(batch)

    return RecoveryResult(
        success=True,
        batch_id=batch.id,
        strategy=RecoveryStrategy.SKIP,
        sessions_failed=skipped,
        message=f"Skipped {skipped} stalled sessions",
    )


async def _recover_fail(
    orchestrator: BatchOrchestrator,
    batch: BatchJob,
) -> RecoveryResult:
    """
    Fail strategy: Mark the entire batch as failed.

    Used when recovery is not possible or not desired.
    """
    from titan.batch.models import BatchStatus, SessionQueueStatus

    failed_sessions = 0

    # Cancel all non-terminal sessions
    for session in batch.sessions:
        if not session.is_terminal:
            session.status = SessionQueueStatus.CANCELLED
            session.error = "Batch recovery failed"
            session.completed_at = datetime.now()
            failed_sessions += 1

    # Mark batch as failed
    batch.status = BatchStatus.FAILED
    batch.error = "Recovery failed - batch marked as failed"
    batch.completed_at = datetime.now()

    # Stop monitoring
    orchestrator._stop_monitoring(batch.id)
    await orchestrator._persist_batch(batch)

    return RecoveryResult(
        success=True,
        batch_id=batch.id,
        strategy=RecoveryStrategy.FAIL,
        sessions_failed=failed_sessions,
        message=f"Batch marked as failed, {failed_sessions} sessions cancelled",
    )


async def _recover_manual(
    orchestrator: BatchOrchestrator,
    batch: BatchJob,
) -> RecoveryResult:
    """
    Manual strategy: Pause batch for manual intervention.

    Preserves current state and notifies for human review.
    """
    from titan.batch.models import BatchStatus

    # Pause the batch
    batch.status = BatchStatus.PAUSED
    batch.metadata["recovery_paused_at"] = datetime.now().isoformat()
    batch.metadata["recovery_reason"] = "Stalled - awaiting manual intervention"

    # Stop automated processing
    orchestrator._stop_monitoring(batch.id)
    await orchestrator._persist_batch(batch)

    # Count stalled sessions
    from titan.batch.models import SessionQueueStatus

    stalled = sum(1 for s in batch.sessions if s.status == SessionQueueStatus.RUNNING)

    return RecoveryResult(
        success=True,
        batch_id=batch.id,
        strategy=RecoveryStrategy.MANUAL,
        sessions_recovered=0,
        sessions_failed=0,
        message=f"Batch paused for manual review, {stalled} sessions stalled",
    )


async def detect_stalled_sessions(
    batch: BatchJob,
    threshold_minutes: int = 30,
) -> list[QueuedSession]:
    """
    Detect sessions that appear stalled (running too long without progress).

    Args:
        batch: Batch to check
        threshold_minutes: Minutes without progress to consider stalled

    Returns:
        List of stalled sessions
    """
    from titan.batch.models import SessionQueueStatus

    stalled = []
    cutoff = datetime.now() - timedelta(minutes=threshold_minutes)

    for session in batch.sessions:
        if session.status == SessionQueueStatus.RUNNING:
            if session.started_at and session.started_at < cutoff:
                stalled.append(session)

    return stalled


async def get_recovery_recommendation(
    batch: BatchJob,
) -> dict[str, Any]:
    """
    Analyze batch state and recommend a recovery strategy.

    Args:
        batch: Batch to analyze

    Returns:
        Dictionary with recommended strategy and reasoning
    """
    stalled = await detect_stalled_sessions(batch)
    progress = batch.progress

    # Calculate health metrics
    total_sessions = progress.total
    completed_sessions = progress.completed
    failed_sessions = progress.failed
    stalled_count = len(stalled)

    completion_rate = completed_sessions / total_sessions if total_sessions > 0 else 0
    failure_rate = failed_sessions / total_sessions if total_sessions > 0 else 0
    stall_rate = stalled_count / total_sessions if total_sessions > 0 else 0

    # Determine recommendation
    if stall_rate > 0.5:
        # More than half stalled - likely systemic issue
        strategy = RecoveryStrategy.MANUAL
        reason = "High stall rate suggests systemic issue"
    elif completion_rate > 0.7:
        # Most completed - skip remaining issues
        strategy = RecoveryStrategy.SKIP
        reason = "High completion rate - skip remaining failures"
    elif failure_rate > 0.5:
        # High failure rate - consider failing batch
        strategy = RecoveryStrategy.FAIL
        reason = "High failure rate - batch unlikely to succeed"
    else:
        # Default to retry
        strategy = RecoveryStrategy.RETRY
        reason = "Retry stalled sessions"

    return {
        "recommended_strategy": strategy.value,
        "reason": reason,
        "metrics": {
            "total_sessions": total_sessions,
            "completed": completed_sessions,
            "failed": failed_sessions,
            "stalled": stalled_count,
            "completion_rate": round(completion_rate * 100, 1),
            "failure_rate": round(failure_rate * 100, 1),
            "stall_rate": round(stall_rate * 100, 1),
        },
        "stalled_session_ids": [str(s.id) for s in stalled],
    }
