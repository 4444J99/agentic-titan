"""
Tests for Batch Recovery Utilities.

Covers stalled batch detection and recovery strategies.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from titan.batch.models import (
    BatchJob,
    BatchStatus,
    SessionQueueStatus,
)
from titan.batch.recovery import (
    RecoveryResult,
    RecoveryStrategy,
    detect_stalled_sessions,
    get_recovery_recommendation,
    recover_batch,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock batch orchestrator."""
    orchestrator = MagicMock()
    orchestrator._max_retries = 3
    orchestrator._persist_batch = AsyncMock()
    orchestrator._queue_pending_sessions = AsyncMock()
    orchestrator._check_batch_completion = AsyncMock()
    orchestrator._stop_monitoring = MagicMock()
    return orchestrator


@pytest.fixture
def stalled_batch():
    """Create a batch with stalled sessions."""
    batch = BatchJob(
        topics=["topic1", "topic2", "topic3"],
        status=BatchStatus.PROCESSING,
    )

    # Mark first session as stalled (running for a long time)
    batch.sessions[0].status = SessionQueueStatus.RUNNING
    batch.sessions[0].started_at = datetime.now() - timedelta(hours=1)
    batch.sessions[0].worker_id = "worker-1"

    # Second session completed
    batch.sessions[1].status = SessionQueueStatus.COMPLETED
    batch.sessions[1].completed_at = datetime.now()

    # Third session pending
    batch.sessions[2].status = SessionQueueStatus.PENDING

    return batch


@pytest.fixture
def partially_completed_batch():
    """Create a batch with some completed and some failed sessions."""
    batch = BatchJob(
        topics=["t1", "t2", "t3", "t4", "t5"],
        status=BatchStatus.PROCESSING,
    )

    # 3 completed
    for i in range(3):
        batch.sessions[i].status = SessionQueueStatus.COMPLETED
        batch.sessions[i].completed_at = datetime.now()

    # 2 stalled
    for i in range(3, 5):
        batch.sessions[i].status = SessionQueueStatus.RUNNING
        batch.sessions[i].started_at = datetime.now() - timedelta(hours=1)

    return batch


# =============================================================================
# RecoveryResult Tests
# =============================================================================


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_create_recovery_result(self):
        """Test creating a recovery result."""
        batch_id = uuid4()
        result = RecoveryResult(
            success=True,
            batch_id=batch_id,
            strategy=RecoveryStrategy.RETRY,
            sessions_recovered=2,
            message="Test message",
        )

        assert result.success
        assert result.batch_id == batch_id
        assert result.strategy == RecoveryStrategy.RETRY
        assert result.sessions_recovered == 2

    def test_recovery_result_to_dict(self):
        """Test converting recovery result to dictionary."""
        batch_id = uuid4()
        result = RecoveryResult(
            success=True,
            batch_id=batch_id,
            strategy=RecoveryStrategy.SKIP,
            sessions_recovered=1,
            sessions_failed=2,
            message="Test",
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["batch_id"] == str(batch_id)
        assert result_dict["strategy"] == "skip"
        assert result_dict["sessions_recovered"] == 1
        assert result_dict["sessions_failed"] == 2


# =============================================================================
# Detect Stalled Sessions Tests
# =============================================================================


class TestDetectStalledSessions:
    """Tests for stalled session detection."""

    @pytest.mark.asyncio
    async def test_detect_stalled_sessions(self, stalled_batch):
        """Test detecting stalled sessions."""
        stalled = await detect_stalled_sessions(
            stalled_batch,
            threshold_minutes=30,
        )

        # Only first session is stalled (running for 1 hour)
        assert len(stalled) == 1
        assert stalled[0].id == stalled_batch.sessions[0].id

    @pytest.mark.asyncio
    async def test_no_stalled_sessions(self):
        """Test when no sessions are stalled."""
        batch = BatchJob(topics=["t1", "t2"])

        # Sessions running for only 5 minutes
        for session in batch.sessions:
            session.status = SessionQueueStatus.RUNNING
            session.started_at = datetime.now() - timedelta(minutes=5)

        stalled = await detect_stalled_sessions(
            batch,
            threshold_minutes=30,
        )

        assert len(stalled) == 0

    @pytest.mark.asyncio
    async def test_custom_threshold(self, stalled_batch):
        """Test with custom threshold."""
        # With very high threshold, nothing is stalled
        stalled = await detect_stalled_sessions(
            stalled_batch,
            threshold_minutes=120,  # 2 hours
        )

        assert len(stalled) == 0


# =============================================================================
# Recovery Strategy Tests
# =============================================================================


class TestRecoverBatch:
    """Tests for batch recovery."""

    @pytest.mark.asyncio
    async def test_recover_retry_strategy(self, mock_orchestrator, stalled_batch):
        """Test retry recovery strategy."""
        mock_orchestrator.get_batch.return_value = stalled_batch

        result = await recover_batch(
            mock_orchestrator,
            stalled_batch.id,
            strategy=RecoveryStrategy.RETRY,
        )

        assert result.success
        assert result.strategy == RecoveryStrategy.RETRY
        assert result.sessions_recovered >= 0

        # Stalled session should be reset to pending
        assert stalled_batch.sessions[0].status == SessionQueueStatus.PENDING
        assert stalled_batch.sessions[0].retry_count == 1

        # Should have persisted changes
        mock_orchestrator._persist_batch.assert_called()
        mock_orchestrator._queue_pending_sessions.assert_called()

    @pytest.mark.asyncio
    async def test_recover_skip_strategy(self, mock_orchestrator, stalled_batch):
        """Test skip recovery strategy."""
        mock_orchestrator.get_batch.return_value = stalled_batch

        result = await recover_batch(
            mock_orchestrator,
            stalled_batch.id,
            strategy=RecoveryStrategy.SKIP,
        )

        assert result.success
        assert result.strategy == RecoveryStrategy.SKIP

        # Stalled session should be marked failed
        assert stalled_batch.sessions[0].status == SessionQueueStatus.FAILED
        assert stalled_batch.sessions[0].completed_at is not None

    @pytest.mark.asyncio
    async def test_recover_fail_strategy(self, mock_orchestrator, stalled_batch):
        """Test fail recovery strategy."""
        mock_orchestrator.get_batch.return_value = stalled_batch

        result = await recover_batch(
            mock_orchestrator,
            stalled_batch.id,
            strategy=RecoveryStrategy.FAIL,
        )

        assert result.success
        assert result.strategy == RecoveryStrategy.FAIL

        # Batch should be marked failed
        assert stalled_batch.status == BatchStatus.FAILED
        assert stalled_batch.completed_at is not None

        # All non-terminal sessions cancelled
        for session in stalled_batch.sessions:
            if session.status != SessionQueueStatus.COMPLETED:
                assert session.status == SessionQueueStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_recover_manual_strategy(self, mock_orchestrator, stalled_batch):
        """Test manual recovery strategy."""
        mock_orchestrator.get_batch.return_value = stalled_batch

        result = await recover_batch(
            mock_orchestrator,
            stalled_batch.id,
            strategy=RecoveryStrategy.MANUAL,
        )

        assert result.success
        assert result.strategy == RecoveryStrategy.MANUAL

        # Batch should be paused
        assert stalled_batch.status == BatchStatus.PAUSED
        assert "recovery_paused_at" in stalled_batch.metadata

    @pytest.mark.asyncio
    async def test_recover_batch_not_found(self, mock_orchestrator):
        """Test recovery when batch not found."""
        mock_orchestrator.get_batch.return_value = None

        result = await recover_batch(
            mock_orchestrator,
            uuid4(),
            strategy=RecoveryStrategy.RETRY,
        )

        assert not result.success
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_retry_with_max_retries_exceeded(self, mock_orchestrator, stalled_batch):
        """Test retry when max retries exceeded."""
        # Set retry count to max
        stalled_batch.sessions[0].retry_count = 3
        mock_orchestrator.get_batch.return_value = stalled_batch

        result = await recover_batch(
            mock_orchestrator,
            stalled_batch.id,
            strategy=RecoveryStrategy.RETRY,
            max_retry_count=3,
        )

        assert result.success
        # Session should be failed, not retried
        assert stalled_batch.sessions[0].status == SessionQueueStatus.FAILED


# =============================================================================
# Recovery Recommendation Tests
# =============================================================================


class TestGetRecoveryRecommendation:
    """Tests for recovery recommendation."""

    @pytest.mark.asyncio
    async def test_recommend_retry_for_low_stall_rate(self, stalled_batch):
        """Test recommends retry when stall rate is low."""
        rec = await get_recovery_recommendation(stalled_batch)

        assert "recommended_strategy" in rec
        assert "metrics" in rec
        assert "stalled_session_ids" in rec

        # With 1 stalled out of 3, should likely recommend retry
        assert rec["metrics"]["stall_rate"] < 50

    @pytest.mark.asyncio
    async def test_recommend_skip_for_high_completion(self, partially_completed_batch):
        """Test recommends skip when completion rate is high."""
        rec = await get_recovery_recommendation(partially_completed_batch)

        # 3/5 completed = 60%, should recommend skip
        assert rec["metrics"]["completion_rate"] >= 60.0

    @pytest.mark.asyncio
    async def test_recommend_fail_for_high_failure(self):
        """Test recommends fail when failure rate is high."""
        batch = BatchJob(topics=["t1", "t2", "t3", "t4"])

        # Mark most as failed
        for i, session in enumerate(batch.sessions):
            if i < 3:  # 3 failed
                session.status = SessionQueueStatus.FAILED
                session.completed_at = datetime.now()
            else:  # 1 stalled
                session.status = SessionQueueStatus.RUNNING
                session.started_at = datetime.now() - timedelta(hours=1)

        rec = await get_recovery_recommendation(batch)

        # With 75% failure rate, should recommend fail
        assert rec["metrics"]["failure_rate"] > 50

    @pytest.mark.asyncio
    async def test_recommend_manual_for_high_stall_rate(self):
        """Test recommends manual when stall rate is very high."""
        batch = BatchJob(topics=["t1", "t2", "t3", "t4"])

        # All stalled
        for session in batch.sessions:
            session.status = SessionQueueStatus.RUNNING
            session.started_at = datetime.now() - timedelta(hours=1)

        rec = await get_recovery_recommendation(batch)

        # With 100% stall rate, should recommend manual
        assert rec["recommended_strategy"] == "manual"
        assert rec["metrics"]["stall_rate"] == 100.0
