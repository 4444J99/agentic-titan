"""
Titan Batch - Cleanup Utilities

Centralized cleanup utilities for batch processing system.
Handles cleanup of old results, orphaned artifacts, and stale records.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.batch.artifact_store import ArtifactStore
    from titan.persistence.postgres import PostgresClient

logger = logging.getLogger("titan.batch.cleanup")


async def cleanup_celery_results(
    retention_days: int = 30,
) -> dict[str, Any]:
    """
    Clean old Celery task results from the result backend.

    Args:
        retention_days: Number of days to retain results

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {"cleaned": 0, "errors": []}

    try:
        from titan.batch.celery_app import celery_app

        backend = celery_app.backend

        # Try cleanup method if available (Redis backend has this)
        if hasattr(backend, "cleanup"):
            backend.cleanup()
            stats["cleaned"] = 1
            logger.info("Cleaned Celery result backend")

        # For Redis backend, we can also delete old keys
        if hasattr(backend, "client"):
            redis_client = backend.client
            cutoff_timestamp = (
                datetime.now() - timedelta(days=retention_days)
            ).timestamp()

            # Redis backend stores results with celery-task-meta- prefix
            pattern = "celery-task-meta-*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    try:
                        # Check key TTL - if no TTL set and older than retention
                        ttl = redis_client.ttl(key)
                        if ttl == -1:  # No expiry set
                            # Delete keys without TTL (old results)
                            redis_client.delete(key)
                            deleted += 1
                    except Exception:
                        pass
                if cursor == 0:
                    break

            stats["cleaned"] = deleted
            logger.info(f"Deleted {deleted} old Celery result keys")

    except Exception as e:
        stats["errors"].append(str(e))
        logger.warning(f"Celery result cleanup failed: {e}")

    return stats


async def cleanup_orphaned_artifacts(
    artifact_store: ArtifactStore,
    postgres: PostgresClient | None,
    retention_days: int = 30,
) -> dict[str, Any]:
    """
    Remove artifacts for deleted or old batches.

    Args:
        artifact_store: Artifact store instance
        postgres: PostgreSQL client for batch lookups
        retention_days: Number of days to retain artifacts

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {"scanned": 0, "deleted": 0, "errors": []}
    cutoff = datetime.now() - timedelta(days=retention_days)

    try:
        # Get all batch IDs that should be retained
        active_batch_ids = set()
        if postgres and postgres.is_connected:
            # Get batches created within retention period or still active
            batches = await postgres.list_batch_jobs(limit=10000)
            for batch in batches:
                created_at = batch.get("created_at")
                status = batch.get("status", "")

                # Keep if still active or within retention
                if status in ("pending", "queued", "processing", "paused"):
                    active_batch_ids.add(str(batch["id"]))
                elif created_at:
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at)
                    if created_at > cutoff:
                        active_batch_ids.add(str(batch["id"]))

        # Scan artifact directories and delete orphaned ones
        from pathlib import Path

        if hasattr(artifact_store, "base_path"):
            base_path = Path(artifact_store.base_path)
            for batch_dir in base_path.iterdir():
                if batch_dir.is_dir():
                    stats["scanned"] += 1
                    batch_id = batch_dir.name

                    # Delete if not in active batches
                    if batch_id not in active_batch_ids:
                        try:
                            # Delete all files in directory
                            for file in batch_dir.iterdir():
                                file.unlink()
                            batch_dir.rmdir()
                            stats["deleted"] += 1
                            logger.debug(f"Deleted orphaned artifacts for batch {batch_id}")
                        except Exception as e:
                            stats["errors"].append(f"Failed to delete {batch_id}: {e}")

    except Exception as e:
        stats["errors"].append(str(e))
        logger.warning(f"Artifact cleanup failed: {e}")

    return stats


async def cleanup_postgres_batches(
    postgres: PostgresClient,
    retention_days: int = 30,
    terminal_only: bool = True,
) -> dict[str, Any]:
    """
    Archive and delete old batch records from PostgreSQL.

    Args:
        postgres: PostgreSQL client
        retention_days: Number of days to retain records
        terminal_only: Only delete batches in terminal states

    Returns:
        Dictionary with cleanup statistics
    """
    stats = {"deleted": 0, "errors": []}

    if not postgres or not postgres.is_connected:
        return stats

    try:
        deleted = await postgres.delete_old_batches(
            before=datetime.now() - timedelta(days=retention_days),
            terminal_only=terminal_only,
        )
        stats["deleted"] = deleted
        logger.info(f"Deleted {deleted} old batch records from PostgreSQL")

    except Exception as e:
        stats["errors"].append(str(e))
        logger.warning(f"PostgreSQL batch cleanup failed: {e}")

    return stats


async def full_cleanup(
    retention_days: int = 30,
) -> dict[str, Any]:
    """
    Perform full cleanup of all batch processing artifacts.

    Args:
        retention_days: Number of days to retain data

    Returns:
        Dictionary with combined cleanup statistics
    """
    logger.info(f"Starting full cleanup (retention: {retention_days} days)")

    stats = {
        "celery_results": {},
        "artifacts": {},
        "postgres": {},
        "total_errors": 0,
    }

    # 1. Cleanup Celery results
    stats["celery_results"] = await cleanup_celery_results(retention_days)

    # 2. Cleanup artifacts
    try:
        from titan.batch.artifact_store import get_artifact_store
        from titan.persistence.postgres import get_postgres_client

        artifact_store = get_artifact_store()
        postgres = get_postgres_client()

        if not postgres.is_connected:
            await postgres.connect()

        stats["artifacts"] = await cleanup_orphaned_artifacts(
            artifact_store,
            postgres,
            retention_days,
        )

        # 3. Cleanup PostgreSQL records
        stats["postgres"] = await cleanup_postgres_batches(
            postgres,
            retention_days,
        )

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        stats["total_errors"] += 1

    # Count total errors
    for section in ["celery_results", "artifacts", "postgres"]:
        if section in stats and "errors" in stats[section]:
            stats["total_errors"] += len(stats[section]["errors"])

    logger.info(
        f"Full cleanup completed: "
        f"results={stats['celery_results'].get('cleaned', 0)}, "
        f"artifacts={stats['artifacts'].get('deleted', 0)}, "
        f"postgres={stats['postgres'].get('deleted', 0)}, "
        f"errors={stats['total_errors']}"
    )

    return stats
