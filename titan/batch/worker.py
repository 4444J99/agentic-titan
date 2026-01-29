"""
Titan Batch - Celery Worker Tasks

Celery task definitions for distributed batch processing.
Wraps InquiryEngine for execution on remote workers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

logger = logging.getLogger("titan.batch.worker")

# Worker identification
WORKER_ID = os.getenv("CELERY_WORKER_ID", f"worker-{os.getpid()}")
RUNTIME_TYPE = os.getenv("WORKER_RUNTIME_TYPE", "local")


# =============================================================================
# Helper Functions
# =============================================================================

def run_async(coro):
    """Run async function in synchronous context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_inquiry_engine():
    """Get or create InquiryEngine for this worker."""
    from titan.workflows.inquiry_engine import InquiryEngine

    # Create engine with worker-specific configuration
    return InquiryEngine()


def get_orchestrator():
    """Get the batch orchestrator for callbacks."""
    from titan.batch.orchestrator import get_batch_orchestrator

    return get_batch_orchestrator()


def get_artifact_store():
    """Get the artifact store for persisting results."""
    from titan.batch.artifact_store import get_artifact_store

    return get_artifact_store()


def get_hive_mind():
    """Get or create HiveMind instance for cross-worker coordination."""
    from hive.memory import HiveMind, MemoryConfig

    hive = HiveMind(MemoryConfig())
    return hive


async def broadcast_worker_metrics(worker_id: str, metrics: dict[str, Any]) -> None:
    """Broadcast worker metrics to HiveMind for coordination."""
    try:
        hive = get_hive_mind()
        await hive.initialize()
        await hive.set(
            f"worker_metrics:{worker_id}",
            {
                "worker_id": worker_id,
                "timestamp": datetime.now().isoformat(),
                **metrics,
            },
            ttl=300,  # 5 minute TTL
        )
    except Exception as e:
        logger.debug(f"Failed to broadcast worker metrics: {e}")


# =============================================================================
# Main Inquiry Task
# =============================================================================

@shared_task(
    bind=True,
    name="titan.batch.worker.run_inquiry_session_task",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=600,
    acks_late=True,
    reject_on_worker_lost=True,
    time_limit=1800,
    soft_time_limit=1500,
)
def run_inquiry_session_task(
    self,
    session_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute an inquiry session for a batch.

    This is the main task that runs on Celery workers.
    It executes the full inquiry workflow for a single topic.

    Args:
        session_data: Dictionary containing:
            - session_id: QueuedSession ID
            - batch_id: Parent BatchJob ID
            - topic: Topic to explore
            - workflow_name: Workflow to use
            - budget_remaining: Optional budget limit
            - metadata: Additional metadata

    Returns:
        Dictionary with execution results
    """
    session_id = session_data["session_id"]
    batch_id = session_data["batch_id"]
    topic = session_data["topic"]
    workflow_name = session_data.get("workflow_name", "expansive")

    logger.info(
        f"Worker {WORKER_ID} starting session {session_id} "
        f"for topic: {topic[:50]}..."
    )

    start_time = time.time()

    try:
        # Notify orchestrator of start
        run_async(_notify_session_started(batch_id, session_id, WORKER_ID))

        # Get workflow
        from titan.workflows.inquiry_config import get_workflow

        workflow = get_workflow(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        # Run inquiry
        engine = get_inquiry_engine()
        inquiry_session = run_async(
            engine.start_inquiry(
                topic=topic,
                workflow=workflow,
                metadata={
                    "batch_id": batch_id,
                    "queued_session_id": session_id,
                    "worker_id": WORKER_ID,
                    "runtime_type": RUNTIME_TYPE,
                    **session_data.get("metadata", {}),
                },
            )
        )

        # Execute full workflow
        completed_session = run_async(engine.run_full_workflow(inquiry_session))

        # Calculate metrics
        total_tokens = sum(r.tokens_used for r in completed_session.results)
        duration_ms = int((time.time() - start_time) * 1000)

        # Estimate cost
        cost_usd = _estimate_cost(total_tokens, completed_session.results)

        # Store artifact
        artifact_uri = run_async(
            _store_session_artifact(
                batch_id=batch_id,
                session_id=session_id,
                inquiry_session=completed_session,
            )
        )

        # Notify orchestrator of completion
        run_async(
            _notify_session_completed(
                batch_id=batch_id,
                session_id=session_id,
                artifact_uri=artifact_uri,
                tokens_used=total_tokens,
                cost_usd=cost_usd,
                inquiry_session_id=completed_session.id,
            )
        )

        result = {
            "status": "completed",
            "session_id": session_id,
            "inquiry_session_id": completed_session.id,
            "artifact_uri": artifact_uri,
            "tokens_used": total_tokens,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "stages_completed": len(completed_session.results),
            "worker_id": WORKER_ID,
        }

        logger.info(
            f"Session {session_id} completed: "
            f"{total_tokens} tokens, ${cost_usd:.4f}, {duration_ms}ms"
        )

        return result

    except SoftTimeLimitExceeded:
        error = "Task exceeded soft time limit (25 minutes)"
        logger.warning(f"Session {session_id} hit soft time limit")
        run_async(_notify_session_failed(batch_id, session_id, error))
        raise

    except MaxRetriesExceededError:
        error = f"Max retries ({self.max_retries}) exceeded"
        logger.error(f"Session {session_id} exceeded max retries")
        run_async(_notify_session_failed(batch_id, session_id, error))
        raise

    except Exception as e:
        error = str(e)
        logger.error(f"Session {session_id} failed: {error}", exc_info=True)

        # Notify orchestrator of failure
        run_async(_notify_session_failed(batch_id, session_id, error))

        # Retry with exponential backoff
        try:
            raise self.retry(exc=e)
        except MaxRetriesExceededError:
            return {
                "status": "failed",
                "session_id": session_id,
                "error": error,
                "worker_id": WORKER_ID,
            }


# =============================================================================
# Synthesis Task
# =============================================================================

@shared_task(
    bind=True,
    name="titan.batch.worker.synthesize_batch_task",
    max_retries=2,
    acks_late=True,
    time_limit=600,
)
def synthesize_batch_task(
    self,
    batch_id: str,
) -> dict[str, Any]:
    """
    Generate cross-session synthesis for a completed batch.

    Args:
        batch_id: Batch job ID

    Returns:
        Dictionary with synthesis results
    """
    logger.info(f"Synthesizing batch {batch_id}")

    try:
        from titan.batch.synthesizer import BatchSynthesizer

        synthesizer = BatchSynthesizer()
        result = run_async(synthesizer.synthesize_batch(batch_id))

        return {
            "status": "completed",
            "batch_id": batch_id,
            "synthesis_uri": result.get("artifact_uri"),
            "summary": result.get("summary", "")[:500],  # Truncate for result
        }

    except Exception as e:
        logger.error(f"Synthesis failed for batch {batch_id}: {e}")
        return {
            "status": "failed",
            "batch_id": batch_id,
            "error": str(e),
        }


# =============================================================================
# Artifact Storage Task
# =============================================================================

@shared_task(
    name="titan.batch.worker.store_artifact_task",
    max_retries=3,
    acks_late=True,
)
def store_artifact_task(
    batch_id: str,
    session_id: str,
    content: str,
    format: str = "markdown",
) -> dict[str, Any]:
    """
    Store a session artifact.

    Args:
        batch_id: Batch job ID
        session_id: Session ID
        content: Artifact content
        format: Content format

    Returns:
        Dictionary with storage result
    """
    try:
        artifact_store = get_artifact_store()
        uri = run_async(
            artifact_store.save_artifact(
                batch_id=batch_id,
                session_id=session_id,
                content=content.encode("utf-8"),
                format=format,
            )
        )

        return {
            "status": "stored",
            "artifact_uri": uri,
            "size_bytes": len(content.encode("utf-8")),
        }

    except Exception as e:
        logger.error(f"Failed to store artifact: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# Maintenance Tasks
# =============================================================================

@shared_task(name="titan.batch.worker.cleanup_old_results_task")
def cleanup_old_results_task(
    retention_days: int = 30,
) -> dict[str, Any]:
    """
    Clean up old task results and artifacts.

    Runs periodically via Celery Beat to prevent resource leaks.

    Args:
        retention_days: Number of days to retain results (default 30)

    Returns:
        Dictionary with cleanup statistics
    """
    logger.info(f"Running cleanup task (retention: {retention_days} days)")

    stats: dict[str, Any] = {
        "results_cleaned": 0,
        "artifacts_cleaned": 0,
        "batches_cleaned": 0,
        "errors": [],
    }
    cutoff = datetime.now() - timedelta(days=retention_days)

    try:
        # 1. Clean old Celery results from backend
        from titan.batch.celery_app import celery_app

        backend = celery_app.backend
        if hasattr(backend, "cleanup"):
            try:
                backend.cleanup()
                stats["results_cleaned"] = 1
                logger.info("Cleaned Celery backend results")
            except Exception as e:
                stats["errors"].append(f"Backend cleanup failed: {e}")
                logger.warning(f"Backend cleanup failed: {e}")

        # 2. Clean orphaned artifacts (completed batches > retention_days)
        async def cleanup_artifacts() -> int:
            artifact_store = get_artifact_store()
            cleaned = 0

            # Try to find orphaned artifacts
            if hasattr(artifact_store, "find_orphaned_artifacts"):
                orphaned = await artifact_store.find_orphaned_artifacts(before=cutoff)
                for artifact_uri in orphaned:
                    try:
                        await artifact_store.delete_artifact(artifact_uri)
                        cleaned += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete artifact {artifact_uri}: {e}")
            return cleaned

        stats["artifacts_cleaned"] = run_async(cleanup_artifacts())

        # 3. Clean completed batch records from PostgreSQL
        async def cleanup_postgres() -> int:
            try:
                from titan.persistence.postgres import get_postgres_client

                postgres = get_postgres_client()
                if postgres.is_connected:
                    return await postgres.delete_old_batches(
                        before=cutoff,
                        terminal_only=True,
                    )
            except Exception as e:
                logger.warning(f"PostgreSQL cleanup failed: {e}")
            return 0

        stats["batches_cleaned"] = run_async(cleanup_postgres())

        # 4. Log cleanup to HiveMind for tracking
        async def log_cleanup() -> None:
            try:
                hive = get_hive_mind()
                await hive.initialize()
                await hive.set(
                    f"cleanup_log:{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "retention_days": retention_days,
                        **stats,
                    },
                    ttl=86400 * 7,  # Keep for 7 days
                )
            except Exception as e:
                logger.debug(f"Failed to log cleanup: {e}")

        run_async(log_cleanup())

    except Exception as e:
        stats["errors"].append(str(e))
        logger.error(f"Cleanup error: {e}")

    logger.info(
        f"Cleanup completed: {stats['results_cleaned']} results, "
        f"{stats['artifacts_cleaned']} artifacts, {stats['batches_cleaned']} batches"
    )

    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        **stats,
    }


@shared_task(name="titan.batch.worker.check_stalled_batches_task")
def check_stalled_batches_task(
    stall_threshold_minutes: int = 30,
) -> dict[str, Any]:
    """
    Check for and handle stalled batches.

    Runs periodically via Celery Beat to detect batches that have
    stopped making progress and attempt recovery.

    Args:
        stall_threshold_minutes: Minutes without progress to consider stalled

    Returns:
        Dictionary with check statistics
    """
    logger.info(f"Checking for stalled batches (threshold: {stall_threshold_minutes}m)")

    stats: dict[str, Any] = {
        "checked": 0,
        "stalled": 0,
        "recovered": 0,
        "failed": 0,
        "errors": [],
    }

    try:
        from titan.batch.orchestrator import get_batch_orchestrator

        orchestrator = get_batch_orchestrator()

        # Get stalled batches
        async def check_and_recover() -> dict[str, int]:
            stalled_batches = await orchestrator.get_stalled_batches(
                threshold_minutes=stall_threshold_minutes
            )
            result = {
                "checked": len(orchestrator._batches),
                "stalled": len(stalled_batches),
                "recovered": 0,
                "failed": 0,
            }

            for batch_id in stalled_batches:
                try:
                    recovered = await orchestrator.recover_stalled_batch(batch_id)
                    if recovered:
                        result["recovered"] += 1
                        logger.info(f"Recovered stalled batch {batch_id}")
                    else:
                        result["failed"] += 1
                        logger.warning(f"Failed to recover batch {batch_id}")
                except Exception as e:
                    result["failed"] += 1
                    logger.error(f"Error recovering batch {batch_id}: {e}")

            return result

        check_result = run_async(check_and_recover())
        stats.update(check_result)

        # Log to HiveMind for monitoring
        async def log_check() -> None:
            try:
                hive = get_hive_mind()
                await hive.initialize()
                await hive.set(
                    f"stalled_check:{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "threshold_minutes": stall_threshold_minutes,
                        **stats,
                    },
                    ttl=86400,  # Keep for 1 day
                )
            except Exception as e:
                logger.debug(f"Failed to log stalled check: {e}")

        run_async(log_check())

    except Exception as e:
        stats["errors"].append(str(e))
        logger.error(f"Stalled batch check error: {e}")

    logger.info(
        f"Stalled check completed: {stats['checked']} checked, "
        f"{stats['stalled']} stalled, {stats['recovered']} recovered"
    )

    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        **stats,
    }


# =============================================================================
# Internal Helpers
# =============================================================================

async def _notify_session_started(
    batch_id: str,
    session_id: str,
    worker_id: str,
) -> None:
    """Notify orchestrator that session has started."""
    try:
        orchestrator = get_orchestrator()
        await orchestrator.handle_session_started(batch_id, session_id, worker_id)
    except Exception as e:
        logger.warning(f"Failed to notify session start: {e}")


async def _notify_session_completed(
    batch_id: str,
    session_id: str,
    artifact_uri: str,
    tokens_used: int,
    cost_usd: float,
    inquiry_session_id: str,
) -> None:
    """Notify orchestrator that session has completed."""
    try:
        orchestrator = get_orchestrator()
        await orchestrator.handle_session_completed(
            batch_id=batch_id,
            session_id=session_id,
            artifact_uri=artifact_uri,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            inquiry_session_id=inquiry_session_id,
        )
    except Exception as e:
        logger.warning(f"Failed to notify session completion: {e}")


async def _notify_session_failed(
    batch_id: str,
    session_id: str,
    error: str,
) -> None:
    """Notify orchestrator that session has failed."""
    try:
        orchestrator = get_orchestrator()
        await orchestrator.handle_session_failed(batch_id, session_id, error)
    except Exception as e:
        logger.warning(f"Failed to notify session failure: {e}")


async def _store_session_artifact(
    batch_id: str,
    session_id: str,
    inquiry_session,
) -> str:
    """Store completed session as artifact."""
    from titan.workflows.inquiry_export import export_session_to_markdown

    # Export to markdown
    content = export_session_to_markdown(inquiry_session)

    # Store artifact
    artifact_store = get_artifact_store()
    return await artifact_store.save_artifact(
        batch_id=batch_id,
        session_id=session_id,
        content=content.encode("utf-8"),
        format="markdown",
        metadata={
            "topic": inquiry_session.topic,
            "workflow": inquiry_session.workflow.name,
            "stages": len(inquiry_session.results),
            "inquiry_session_id": inquiry_session.id,
        },
    )


def _estimate_cost(total_tokens: int, results: list) -> float:
    """
    Estimate cost based on tokens and models used.

    Uses rough pricing for common models.
    """
    # Rough pricing per 1M tokens (input + output averaged)
    model_pricing = {
        "claude-3-5-sonnet-20241022": 7.50,
        "claude-3-opus-20240229": 37.50,
        "claude-3-haiku-20240307": 0.625,
        "gpt-4-turbo": 15.00,
        "gpt-4o": 7.50,
        "gpt-4o-mini": 0.30,
        "gemini-1.5-pro": 3.50,
        "gemini-1.5-flash": 0.35,
    }

    default_price = 7.50  # Default to Sonnet pricing

    total_cost = 0.0
    for result in results:
        model = result.model_used if hasattr(result, "model_used") else "unknown"
        tokens = result.tokens_used if hasattr(result, "tokens_used") else 0
        price_per_million = model_pricing.get(model, default_price)
        total_cost += (tokens / 1_000_000) * price_per_million

    return round(total_cost, 6)
