"""
Titan Persistence - PostgreSQL Client

Async PostgreSQL client with connection pooling and graceful degradation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

logger = logging.getLogger("titan.persistence.postgres")


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "titan"
    user: str = "titan"
    password: str = ""  # allow-secret
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 10.0
    command_timeout: float = 30.0

    # SSL settings
    ssl: bool = False
    ssl_ca_file: str | None = None

    @classmethod
    def from_env(cls) -> PostgresConfig:
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "titan"),
            user=os.getenv("POSTGRES_USER", "titan"),
            password=os.getenv("POSTGRES_PASSWORD", ""),  # allow-secret
            min_connections=int(os.getenv("POSTGRES_MIN_CONN", "2")),
            max_connections=int(os.getenv("POSTGRES_MAX_CONN", "10")),
            ssl=os.getenv("POSTGRES_SSL", "").lower() == "true",
        )

    @property
    def dsn(self) -> str:
        """Get connection DSN."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class PostgresClient:
    """
    Async PostgreSQL client with connection pooling.

    Features:
    - Connection pool management
    - Graceful degradation (falls back to in-memory if unavailable)
    - Transaction support
    - Query timeout handling
    """

    def __init__(self, config: PostgresConfig | None = None) -> None:
        self.config = config or PostgresConfig.from_env()
        self._pool: Any | None = None
        self._connected = False
        self._fallback_store: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Check if connected to PostgreSQL."""
        return self._connected and self._pool is not None

    async def connect(self) -> bool:
        """
        Establish connection pool.

        Returns:
            True if connected, False if fell back to in-memory.
        """
        async with self._lock:
            if self._connected:
                return True

            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,  # allow-secret
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    timeout=self.config.connection_timeout,
                    command_timeout=self.config.command_timeout,
                )
                self._connected = True
                logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")

                # Initialize tables
                await self._init_tables()

                return True

            except ImportError:
                logger.warning("asyncpg not installed, using in-memory fallback")
                self._connected = False
                return False

            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}, using in-memory fallback")
                self._connected = False
                return False

    async def _init_tables(self) -> None:
        """Initialize database tables if they don't exist."""
        from titan.persistence.models import (
            AUDIT_EVENTS_TABLE_SQL,
            AGENT_DECISIONS_TABLE_SQL,
            BATCH_CLEANUP_LOG_TABLE_SQL,
            BATCH_JOBS_STALLED_INDEX_SQL,
            BATCH_JOBS_CLEANUP_INDEX_SQL,
        )
        from titan.batch.models import (
            BATCH_JOBS_TABLE_SQL,
            QUEUED_SESSIONS_TABLE_SQL,
            SESSION_ARTIFACTS_TABLE_SQL,
        )

        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(AUDIT_EVENTS_TABLE_SQL)
            await conn.execute(AGENT_DECISIONS_TABLE_SQL)
            # Batch pipeline tables
            await conn.execute(BATCH_JOBS_TABLE_SQL)
            await conn.execute(QUEUED_SESSIONS_TABLE_SQL)
            await conn.execute(SESSION_ARTIFACTS_TABLE_SQL)
            # Cleanup and monitoring tables
            await conn.execute(BATCH_CLEANUP_LOG_TABLE_SQL)
            await conn.execute(BATCH_JOBS_STALLED_INDEX_SQL)
            await conn.execute(BATCH_JOBS_CLEANUP_INDEX_SQL)
            logger.info("Database tables initialized")

    async def disconnect(self) -> None:
        """Close connection pool."""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
            self._connected = False
            logger.info("Disconnected from PostgreSQL")

    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> str:
        """
        Execute a query.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            Status string from execution
        """
        if not self._connected or not self._pool:
            logger.debug("PostgreSQL not connected, skipping execute")
            return "NOT_CONNECTED"

        async with self._pool.acquire() as conn:
            return await conn.execute(
                query,
                *args,
                timeout=timeout or self.config.command_timeout,
            )

    async def fetch(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch multiple rows.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            List of row dictionaries
        """
        if not self._connected or not self._pool:
            logger.debug("PostgreSQL not connected, returning empty")
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                *args,
                timeout=timeout or self.config.command_timeout,
            )
            return [dict(row) for row in rows]

    async def fetchrow(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch a single row.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            Row dictionary or None
        """
        if not self._connected or not self._pool:
            logger.debug("PostgreSQL not connected, returning None")
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                *args,
                timeout=timeout or self.config.command_timeout,
            )
            return dict(row) if row else None

    async def fetchval(
        self,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> Any:
        """
        Fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            Single value or None
        """
        if not self._connected or not self._pool:
            return None

        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                query,
                *args,
                timeout=timeout or self.config.command_timeout,
            )

    async def insert_audit_event(
        self,
        event_id: UUID,
        timestamp: datetime,
        event_type: str,
        action: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        checksum: str = "",
    ) -> bool:
        """
        Insert an audit event.

        Returns:
            True if inserted, False if fell back to in-memory.
        """
        if not self._connected or not self._pool:
            # Fallback to in-memory
            self._fallback_store.append({
                "id": str(event_id),
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "action": action,
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "input_data": input_data,
                "output_data": output_data,
                "metadata": metadata or {},
                "checksum": checksum,
            })
            return False

        try:
            await self.execute(
                """
                INSERT INTO audit_events
                (id, timestamp, event_type, agent_id, session_id, user_id,
                 action, input_data, output_data, metadata, checksum)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                event_id,
                timestamp,
                event_type,
                agent_id,
                session_id,
                user_id,
                action,
                json.dumps(input_data) if input_data else None,
                json.dumps(output_data) if output_data else None,
                json.dumps(metadata or {}),
                checksum,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert audit event: {e}")
            # Fallback
            self._fallback_store.append({
                "id": str(event_id),
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "action": action,
                "agent_id": agent_id,
                "session_id": session_id,
                "user_id": user_id,
                "input_data": input_data,
                "output_data": output_data,
                "metadata": metadata or {},
                "checksum": checksum,
            })
            return False

    async def insert_agent_decision(
        self,
        decision_id: UUID,
        audit_event_id: UUID,
        decision_type: str,
        rationale: str,
        selected_option: str,
        confidence: float,
        alternatives: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Insert an agent decision record.

        Returns:
            True if inserted, False if fell back.
        """
        if not self._connected or not self._pool:
            return False

        try:
            await self.execute(
                """
                INSERT INTO agent_decisions
                (id, audit_event_id, decision_type, rationale, alternatives,
                 selected_option, confidence, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                decision_id,
                audit_event_id,
                decision_type,
                rationale,
                json.dumps(alternatives or []),
                selected_option,
                confidence,
                json.dumps(metadata or {}),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert agent decision: {e}")
            return False

    async def get_audit_events(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Query audit events with filters.

        Args:
            agent_id: Filter by agent
            session_id: Filter by session
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of audit events
        """
        if not self._connected or not self._pool:
            # Return from fallback store with basic filtering
            results = self._fallback_store
            if agent_id:
                results = [r for r in results if r.get("agent_id") == agent_id]
            if session_id:
                results = [r for r in results if r.get("session_id") == session_id]
            if event_type:
                results = [r for r in results if r.get("event_type") == event_type]
            return results[offset : offset + limit]

        # Build dynamic query
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if agent_id:
            conditions.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1

        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        if start_time:
            conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_time)
            param_idx += 1

        if end_time:
            conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_time)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT * FROM audit_events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        return await self.fetch(query, *params)

    async def verify_audit_integrity(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Verify integrity of audit events by checking checksums.

        Returns:
            Verification results with counts and any invalid events.
        """
        from titan.persistence.models import AuditEvent

        events = await self.get_audit_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000,  # Reasonable batch size
        )

        total = len(events)
        valid = 0
        invalid = []

        for event_data in events:
            # Reconstruct event and verify checksum
            try:
                event = AuditEvent.from_dict(event_data)
                if event.verify_checksum():
                    valid += 1
                else:
                    invalid.append({
                        "id": str(event.id),
                        "timestamp": event.timestamp.isoformat(),
                        "stored_checksum": event_data.get("checksum"),
                        "computed_checksum": event._compute_checksum(),
                    })
            except Exception as e:
                invalid.append({
                    "id": event_data.get("id"),
                    "error": str(e),
                })

        return {
            "total_events": total,
            "valid_events": valid,
            "invalid_events": len(invalid),
            "integrity_percentage": (valid / total * 100) if total > 0 else 100,
            "invalid_details": invalid[:10],  # Limit details
        }

    def get_fallback_events(self) -> list[dict[str, Any]]:
        """Get events stored in fallback (in-memory) store."""
        return self._fallback_store.copy()

    # =========================================================================
    # Batch Job Persistence
    # =========================================================================

    async def insert_batch_job(
        self,
        batch_id: UUID,
        topics: list[str],
        workflow_name: str,
        max_concurrent: int,
        status: str,
        budget_limit_usd: float | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Insert a new batch job.

        Returns:
            True if inserted, False if fell back to in-memory.
        """
        if not self._connected or not self._pool:
            return False

        try:
            await self.execute(
                """
                INSERT INTO batch_jobs
                (id, topics, workflow_name, max_concurrent, budget_limit_usd,
                 status, user_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO NOTHING
                """,
                batch_id,
                json.dumps(topics),
                workflow_name,
                max_concurrent,
                budget_limit_usd,
                status,
                user_id,
                json.dumps(metadata or {}),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert batch job: {e}")
            return False

    async def update_batch_job(
        self,
        batch_id: UUID,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a batch job.

        Args:
            batch_id: Batch job ID
            updates: Dictionary of field updates

        Returns:
            True if updated successfully.
        """
        if not self._connected or not self._pool:
            return False

        if not updates:
            return True

        # Build dynamic update query
        set_clauses = []
        params: list[Any] = []
        param_idx = 1

        for field_name, value in updates.items():
            if field_name in ("topics", "metadata"):
                value = json.dumps(value)
            set_clauses.append(f"{field_name} = ${param_idx}")
            params.append(value)
            param_idx += 1

        params.append(batch_id)
        query = f"""
            UPDATE batch_jobs
            SET {", ".join(set_clauses)}
            WHERE id = ${param_idx}
        """

        try:
            await self.execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"Failed to update batch job {batch_id}: {e}")
            return False

    async def get_batch_job(self, batch_id: UUID | str) -> dict[str, Any] | None:
        """
        Get a batch job by ID.

        Returns:
            Batch job data or None if not found.
        """
        if not self._connected or not self._pool:
            return None

        target = UUID(batch_id) if isinstance(batch_id, str) else batch_id
        row = await self.fetchrow(
            "SELECT * FROM batch_jobs WHERE id = $1",
            target,
        )

        if not row:
            return None

        # Parse JSONB fields
        result = dict(row)
        if "topics" in result and isinstance(result["topics"], str):
            result["topics"] = json.loads(result["topics"])
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def list_batch_jobs(
        self,
        status: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List batch jobs with optional filtering.

        Args:
            status: Filter by status
            user_id: Filter by user ID
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of batch jobs.
        """
        if not self._connected or not self._pool:
            return []

        conditions = []
        params: list[Any] = []
        param_idx = 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT * FROM batch_jobs
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await self.fetch(query, *params)

        # Parse JSONB fields
        results = []
        for row in rows:
            result = dict(row)
            if "topics" in result and isinstance(result["topics"], str):
                result["topics"] = json.loads(result["topics"])
            if "metadata" in result and isinstance(result["metadata"], str):
                result["metadata"] = json.loads(result["metadata"])
            results.append(result)

        return results

    async def delete_batch_job(self, batch_id: UUID | str) -> bool:
        """Delete a batch job and its sessions (cascade)."""
        if not self._connected or not self._pool:
            return False

        target = UUID(batch_id) if isinstance(batch_id, str) else batch_id
        try:
            await self.execute("DELETE FROM batch_jobs WHERE id = $1", target)
            return True
        except Exception as e:
            logger.error(f"Failed to delete batch job {batch_id}: {e}")
            return False

    # =========================================================================
    # Queued Session Persistence
    # =========================================================================

    async def insert_queued_session(
        self,
        session_id: UUID,
        batch_id: UUID,
        topic: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Insert a new queued session.

        Returns:
            True if inserted successfully.
        """
        if not self._connected or not self._pool:
            return False

        try:
            await self.execute(
                """
                INSERT INTO queued_sessions
                (id, batch_id, topic, status, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO NOTHING
                """,
                session_id,
                batch_id,
                topic,
                status,
                json.dumps(metadata or {}),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to insert queued session: {e}")
            return False

    async def update_queued_session(
        self,
        session_id: UUID | str,
        updates: dict[str, Any],
    ) -> bool:
        """
        Update a queued session.

        Args:
            session_id: Session ID
            updates: Dictionary of field updates

        Returns:
            True if updated successfully.
        """
        if not self._connected or not self._pool:
            return False

        if not updates:
            return True

        target = UUID(session_id) if isinstance(session_id, str) else session_id

        # Build dynamic update query
        set_clauses = []
        params: list[Any] = []
        param_idx = 1

        for field_name, value in updates.items():
            if field_name == "metadata":
                value = json.dumps(value)
            set_clauses.append(f"{field_name} = ${param_idx}")
            params.append(value)
            param_idx += 1

        params.append(target)
        query = f"""
            UPDATE queued_sessions
            SET {", ".join(set_clauses)}
            WHERE id = ${param_idx}
        """

        try:
            await self.execute(query, *params)
            return True
        except Exception as e:
            logger.error(f"Failed to update queued session {session_id}: {e}")
            return False

    async def get_queued_session(
        self,
        session_id: UUID | str,
    ) -> dict[str, Any] | None:
        """Get a queued session by ID."""
        if not self._connected or not self._pool:
            return None

        target = UUID(session_id) if isinstance(session_id, str) else session_id
        row = await self.fetchrow(
            "SELECT * FROM queued_sessions WHERE id = $1",
            target,
        )

        if not row:
            return None

        result = dict(row)
        if "metadata" in result and isinstance(result["metadata"], str):
            result["metadata"] = json.loads(result["metadata"])
        return result

    async def get_sessions_for_batch(
        self,
        batch_id: UUID | str,
    ) -> list[dict[str, Any]]:
        """Get all sessions for a batch."""
        if not self._connected or not self._pool:
            return []

        target = UUID(batch_id) if isinstance(batch_id, str) else batch_id
        rows = await self.fetch(
            "SELECT * FROM queued_sessions WHERE batch_id = $1 ORDER BY created_at",
            target,
        )

        results = []
        for row in rows:
            result = dict(row)
            if "metadata" in result and isinstance(result["metadata"], str):
                result["metadata"] = json.loads(result["metadata"])
            results.append(result)

        return results

    async def delete_old_batches(
        self,
        before: datetime,
        terminal_only: bool = True,
    ) -> int:
        """
        Delete old batch jobs and their sessions.

        Args:
            before: Delete batches completed before this time
            terminal_only: Only delete batches in terminal states

        Returns:
            Number of batches deleted
        """
        if not self._connected or not self._pool:
            return 0

        try:
            terminal_states = (
                "'completed'", "'failed'", "'cancelled'", "'partially_completed'"
            )

            if terminal_only:
                query = f"""
                    DELETE FROM batch_jobs
                    WHERE completed_at < $1
                    AND status IN ({", ".join(terminal_states)})
                """
            else:
                query = """
                    DELETE FROM batch_jobs
                    WHERE completed_at < $1
                """

            result = await self.execute(query, before)
            # Parse "DELETE N" result
            deleted = 0
            if result and isinstance(result, str):
                parts = result.split()
                if len(parts) >= 2 and parts[0] == "DELETE":
                    try:
                        deleted = int(parts[1])
                    except ValueError:
                        pass

            logger.info(f"Deleted {deleted} old batch jobs from PostgreSQL")
            return deleted

        except Exception as e:
            logger.error(f"Failed to delete old batches: {e}")
            return 0

    async def log_cleanup(
        self,
        cleanup_type: str,
        items_processed: int = 0,
        items_deleted: int = 0,
        errors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Log a cleanup operation for audit purposes.

        Args:
            cleanup_type: Type of cleanup (results, artifacts, batches)
            items_processed: Number of items processed
            items_deleted: Number of items deleted
            errors: List of error messages
            metadata: Additional metadata

        Returns:
            True if logged successfully
        """
        if not self._connected or not self._pool:
            return False

        try:
            await self.execute(
                """
                INSERT INTO batch_cleanup_log
                (cleanup_type, completed_at, items_processed, items_deleted, errors, metadata)
                VALUES ($1, NOW(), $2, $3, $4, $5)
                """,
                cleanup_type,
                items_processed,
                items_deleted,
                json.dumps(errors or []),
                json.dumps(metadata or {}),
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to log cleanup: {e}")
            return False

    async def flush_fallback_to_postgres(self) -> int:
        """
        Flush fallback store to PostgreSQL when connection is restored.

        Returns:
            Number of events flushed.
        """
        if not self._connected or not self._pool or not self._fallback_store:
            return 0

        flushed = 0
        to_flush = self._fallback_store.copy()
        self._fallback_store.clear()

        for event in to_flush:
            try:
                await self.insert_audit_event(
                    event_id=UUID(event["id"]),
                    timestamp=datetime.fromisoformat(event["timestamp"]),
                    event_type=event["event_type"],
                    action=event["action"],
                    agent_id=event.get("agent_id"),
                    session_id=event.get("session_id"),
                    user_id=event.get("user_id"),
                    input_data=event.get("input_data"),
                    output_data=event.get("output_data"),
                    metadata=event.get("metadata"),
                    checksum=event.get("checksum", ""),
                )
                flushed += 1
            except Exception as e:
                logger.error(f"Failed to flush event {event['id']}: {e}")
                # Put back in fallback
                self._fallback_store.append(event)

        logger.info(f"Flushed {flushed} events from fallback to PostgreSQL")
        return flushed

    async def health_check(self) -> dict[str, Any]:
        """
        Check PostgreSQL health.

        Returns:
            Health status with connection info.
        """
        if not self._connected or not self._pool:
            return {
                "healthy": False,
                "connected": False,
                "fallback_events": len(self._fallback_store),
            }

        try:
            result = await self.fetchval("SELECT 1")
            pool_size = self._pool.get_size() if hasattr(self._pool, "get_size") else 0

            return {
                "healthy": result == 1,
                "connected": True,
                "pool_size": pool_size,
                "fallback_events": len(self._fallback_store),
                "host": self.config.host,
                "database": self.config.database,
            }
        except Exception as e:
            return {
                "healthy": False,
                "connected": True,
                "error": str(e),
                "fallback_events": len(self._fallback_store),
            }

    async def __aenter__(self) -> PostgresClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()


# Singleton instance
_default_client: PostgresClient | None = None


def get_postgres_client(config: PostgresConfig | None = None) -> PostgresClient:
    """Get or create the default PostgreSQL client."""
    global _default_client
    if _default_client is None:
        _default_client = PostgresClient(config)
    return _default_client


async def init_postgres(config: PostgresConfig | None = None) -> PostgresClient:
    """Initialize and connect the PostgreSQL client."""
    client = get_postgres_client(config)
    await client.connect()
    return client
