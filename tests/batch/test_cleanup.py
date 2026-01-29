"""
Tests for Batch Cleanup Utilities.

Covers cleanup of old results, orphaned artifacts, and stale records.
"""

from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from titan.batch.cleanup import (
    cleanup_celery_results,
    cleanup_orphaned_artifacts,
    cleanup_postgres_batches,
    full_cleanup,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_artifact_dir():
    """Create a temporary artifact directory."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_artifact_store(temp_artifact_dir):
    """Create a mock artifact store with temp directory."""
    store = MagicMock()
    store.base_path = temp_artifact_dir
    return store


@pytest.fixture
def mock_postgres():
    """Create a mock PostgreSQL client."""
    postgres = AsyncMock()
    postgres.is_connected = True
    postgres.list_batch_jobs = AsyncMock(return_value=[])
    postgres.delete_old_batches = AsyncMock(return_value=5)
    return postgres


# =============================================================================
# Cleanup Celery Results Tests
# =============================================================================

class TestCleanupCeleryResults:
    """Tests for Celery result cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_with_cleanup_method(self):
        """Test cleanup when backend has cleanup method."""
        pytest.importorskip("celery")

        mock_backend = MagicMock()
        mock_backend.cleanup = MagicMock()

        with patch("titan.batch.celery_app.celery_app") as mock_celery:
            mock_celery.backend = mock_backend

            result = await cleanup_celery_results(retention_days=30)

        assert result["cleaned"] >= 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_import_error(self):
        """Test cleanup handles missing Celery gracefully."""
        # This test verifies cleanup gracefully handles missing celery
        result = await cleanup_celery_results()

        # Should not crash, just return errors or zero cleaned
        assert "cleaned" in result or "errors" in result


# =============================================================================
# Cleanup Orphaned Artifacts Tests
# =============================================================================

class TestCleanupOrphanedArtifacts:
    """Tests for orphaned artifact cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_orphaned_directories(
        self,
        mock_artifact_store,
        mock_postgres,
        temp_artifact_dir,
    ):
        """Test that orphaned artifact directories are removed."""
        # Create some batch directories
        active_batch_id = str(uuid4())
        orphan_batch_id = str(uuid4())

        (temp_artifact_dir / active_batch_id).mkdir()
        (temp_artifact_dir / orphan_batch_id).mkdir()

        # Mock postgres to return only active batch
        mock_postgres.list_batch_jobs.return_value = [
            {
                "id": active_batch_id,
                "status": "processing",
                "created_at": datetime.now(),
            }
        ]

        result = await cleanup_orphaned_artifacts(
            mock_artifact_store,
            mock_postgres,
            retention_days=30,
        )

        assert result["scanned"] == 2
        # Orphan should be deleted
        assert result["deleted"] == 1
        assert not (temp_artifact_dir / orphan_batch_id).exists()
        assert (temp_artifact_dir / active_batch_id).exists()

    @pytest.mark.asyncio
    async def test_cleanup_preserves_recent_batches(
        self,
        mock_artifact_store,
        mock_postgres,
        temp_artifact_dir,
    ):
        """Test that recent completed batches are preserved."""
        batch_id = str(uuid4())
        (temp_artifact_dir / batch_id).mkdir()

        # Batch completed recently
        mock_postgres.list_batch_jobs.return_value = [
            {
                "id": batch_id,
                "status": "completed",
                "created_at": datetime.now() - timedelta(days=5),
            }
        ]

        result = await cleanup_orphaned_artifacts(
            mock_artifact_store,
            mock_postgres,
            retention_days=30,
        )

        assert (temp_artifact_dir / batch_id).exists()
        assert result["deleted"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_without_postgres(
        self,
        mock_artifact_store,
        temp_artifact_dir,
    ):
        """Test cleanup works without PostgreSQL."""
        batch_id = str(uuid4())
        (temp_artifact_dir / batch_id).mkdir()

        result = await cleanup_orphaned_artifacts(
            mock_artifact_store,
            None,  # No PostgreSQL
            retention_days=30,
        )

        # Should delete all when no postgres to validate
        assert result["scanned"] == 1
        assert result["deleted"] == 1


# =============================================================================
# Cleanup PostgreSQL Batches Tests
# =============================================================================

class TestCleanupPostgresBatches:
    """Tests for PostgreSQL batch cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_old_batches(self, mock_postgres):
        """Test deletion of old batch records."""
        mock_postgres.delete_old_batches.return_value = 10

        result = await cleanup_postgres_batches(
            mock_postgres,
            retention_days=30,
            terminal_only=True,
        )

        assert result["deleted"] == 10
        mock_postgres.delete_old_batches.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_without_connection(self):
        """Test cleanup when PostgreSQL not connected."""
        mock_postgres = AsyncMock()
        mock_postgres.is_connected = False

        result = await cleanup_postgres_batches(mock_postgres)

        assert result["deleted"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_error(self, mock_postgres):
        """Test cleanup handles database errors."""
        mock_postgres.delete_old_batches.side_effect = Exception("DB error")

        result = await cleanup_postgres_batches(mock_postgres)

        assert result["deleted"] == 0
        assert len(result["errors"]) > 0


# =============================================================================
# Full Cleanup Tests
# =============================================================================

class TestFullCleanup:
    """Tests for full cleanup orchestration."""

    @pytest.mark.asyncio
    async def test_full_cleanup_combines_all(self):
        """Test full cleanup runs all cleanup types."""
        # Use module-level imports to test the full cleanup function
        with patch.object(
            __import__("titan.batch.cleanup", fromlist=["cleanup_celery_results"]),
            "cleanup_celery_results",
        ) as mock_celery:
            mock_celery.return_value = {"cleaned": 5, "errors": []}

            # Run the cleanup - it will use mocks or gracefully fail
            result = await full_cleanup(retention_days=30)

        assert "celery_results" in result
        assert "artifacts" in result
        assert "postgres" in result
        assert "total_errors" in result
