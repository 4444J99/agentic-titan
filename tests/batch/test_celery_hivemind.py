"""
Tests for Celery-HiveMind Integration.

Covers worker registration and deregistration with HiveMind.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if celery is not installed
pytest.importorskip("celery")


# =============================================================================
# Worker Signal Tests
# =============================================================================


class TestWorkerSignals:
    """Tests for Celery worker signal handlers."""

    @pytest.mark.asyncio
    async def test_worker_ready_registers_with_hivemind(self):
        """Test that worker_ready registers worker with HiveMind."""
        mock_hive = AsyncMock()
        mock_sender = MagicMock()
        mock_sender.hostname = "worker-1"

        with patch.dict(
            "os.environ",
            {
                "CELERY_WORKER_ID": "test-worker-1",
                "WORKER_RUNTIME_TYPE": "local",
                "CELERY_CONCURRENCY": "4",
            },
        ):
            with patch("hive.memory.HiveMind", return_value=mock_hive):
                mock_hive.initialize = AsyncMock()
                mock_hive.register_agent = AsyncMock()

                from titan.batch.celery_app import on_worker_ready

                # Call the handler
                on_worker_ready(mock_sender)

        # Verify registration was attempted
        mock_hive.initialize.assert_called()
        mock_hive.register_agent.assert_called()

    @pytest.mark.asyncio
    async def test_worker_shutdown_deregisters_from_hivemind(self):
        """Test that worker_shutdown deregisters worker from HiveMind."""
        mock_hive = AsyncMock()
        mock_hive.delete = AsyncMock()
        mock_hive.shutdown = AsyncMock()

        mock_sender = MagicMock()
        mock_sender.hostname = "worker-1"
        mock_sender._titan_hive = mock_hive

        with patch.dict("os.environ", {"CELERY_WORKER_ID": "test-worker-1"}):
            from titan.batch.celery_app import on_worker_shutdown

            # Call the handler
            on_worker_shutdown(mock_sender)

        # Verify deregistration was attempted
        mock_hive.delete.assert_called()
        mock_hive.shutdown.assert_called()

    @pytest.mark.asyncio
    async def test_worker_ready_handles_hivemind_failure(self):
        """Test that worker_ready gracefully handles HiveMind failure."""
        mock_sender = MagicMock()
        mock_sender.hostname = "worker-1"

        with patch.dict("os.environ", {"CELERY_WORKER_ID": "test-worker-1"}):
            with patch("hive.memory.HiveMind", side_effect=Exception("Connection failed")):
                from titan.batch.celery_app import on_worker_ready

                # Should not raise
                on_worker_ready(mock_sender)


# =============================================================================
# Worker Metrics Tests
# =============================================================================


class TestWorkerMetrics:
    """Tests for worker metrics broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_worker_metrics(self):
        """Test broadcasting worker metrics to HiveMind."""
        from titan.batch.worker import broadcast_worker_metrics

        mock_hive = AsyncMock()
        mock_hive.initialize = AsyncMock()
        mock_hive.set = AsyncMock()

        with patch("titan.batch.worker.get_hive_mind", return_value=mock_hive):
            await broadcast_worker_metrics(
                "worker-1",
                {
                    "active_tasks": 2,
                    "cpu_usage": 50.0,
                    "memory_usage": 60.0,
                },
            )

        mock_hive.set.assert_called_once()
        call_args = mock_hive.set.call_args

        # Check key includes worker_id
        assert "worker_metrics:worker-1" in call_args[0][0]

        # Check metrics are included
        metrics = call_args[0][1]
        assert metrics["active_tasks"] == 2
        assert metrics["cpu_usage"] == 50.0

    @pytest.mark.asyncio
    async def test_broadcast_handles_hivemind_failure(self):
        """Test that broadcast gracefully handles HiveMind failure."""
        from titan.batch.worker import broadcast_worker_metrics

        mock_hive = AsyncMock()
        mock_hive.initialize = AsyncMock(side_effect=Exception("Connection failed"))

        with patch("titan.batch.worker.get_hive_mind", return_value=mock_hive):
            # Should not raise
            await broadcast_worker_metrics("worker-1", {"test": True})


# =============================================================================
# Async Helper Tests
# =============================================================================


class TestRunAsync:
    """Tests for async helper function."""

    def test_run_async_executes_coroutine(self):
        """Test that run_async executes coroutines."""
        from titan.batch.worker import run_async

        async def sample_coro():
            return "result"

        result = run_async(sample_coro())
        assert result == "result"

    def test_run_async_handles_exception(self):
        """Test that run_async propagates exceptions."""
        from titan.batch.worker import run_async

        async def failing_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())


# =============================================================================
# Integration Tests (require full stack)
# =============================================================================


@pytest.mark.skip(reason="Requires running infrastructure")
class TestCeleryHiveMindIntegration:
    """Integration tests requiring running Celery and HiveMind."""

    @pytest.mark.asyncio
    async def test_worker_lifecycle(self):
        """Test full worker registration lifecycle."""
        pass
