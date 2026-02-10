"""
Titan Persistence - Audit Logger

Provides high-level audit logging functionality with automatic
event capture and decision tracking.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from titan.persistence.models import (
    AgentDecision,
    AuditEvent,
    AuditEventType,
    DecisionType,
)
from titan.persistence.postgres import PostgresClient, get_postgres_client

logger = logging.getLogger("titan.persistence.audit")


@dataclass
class AuditContext:
    """Context for tracking audit events within a session."""

    session_id: str
    agent_id: str | None = None
    user_id: str | None = None
    parent_event_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """
    High-level audit logging interface.

    Features:
    - Automatic event creation with checksums
    - Decision tracking linked to events
    - Batch logging for performance
    - Context management for session tracking
    """

    def __init__(
        self,
        client: PostgresClient | None = None,
        batch_size: int = 10,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self._client = client or get_postgres_client()
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._pending_events: list[AuditEvent] = []
        self._pending_decisions: list[AgentDecision] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the audit logger with background flushing."""
        if self._running:
            return

        await self._client.connect()
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Audit logger started")

    async def stop(self) -> None:
        """Stop the audit logger and flush pending events."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush()
        logger.info("Audit logger stopped")

    async def _flush_loop(self) -> None:
        """Background task to periodically flush pending events."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")

    async def _flush(self) -> None:
        """Flush pending events and decisions to storage."""
        async with self._lock:
            if not self._pending_events and not self._pending_decisions:
                return

            # Flush events
            for event in self._pending_events:
                await self._client.insert_audit_event(
                    event_id=event.id,
                    timestamp=event.timestamp,
                    event_type=event.event_type.value,
                    action=event.action,
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    user_id=event.user_id,
                    input_data=event.input_data,
                    output_data=event.output_data,
                    metadata=event.metadata,
                    checksum=event.checksum,
                )

            events_flushed = len(self._pending_events)
            self._pending_events.clear()

            # Flush decisions
            for decision in self._pending_decisions:
                await self._client.insert_agent_decision(
                    decision_id=decision.id,
                    audit_event_id=decision.audit_event_id,
                    decision_type=decision.decision_type.value,
                    rationale=decision.rationale,
                    selected_option=decision.selected_option,
                    confidence=decision.confidence,
                    alternatives=decision.alternatives,
                    metadata=decision.metadata,
                )

            decisions_flushed = len(self._pending_decisions)
            self._pending_decisions.clear()

            if events_flushed or decisions_flushed:
                logger.debug(f"Flushed {events_flushed} events, {decisions_flushed} decisions")

    async def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        flush: bool = False,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            action: Description of the action
            agent_id: Agent that performed the action
            session_id: Session identifier
            user_id: User identifier
            input_data: Input to the action
            output_data: Output from the action
            metadata: Additional metadata
            flush: Force immediate flush

        Returns:
            The created audit event
        """
        event = AuditEvent(
            event_type=event_type,
            action=action,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
        )

        async with self._lock:
            self._pending_events.append(event)

            if flush or len(self._pending_events) >= self._batch_size:
                await self._flush()

        logger.debug(f"Logged event: {event_type.value} - {action[:50]}")
        return event

    async def log_decision(
        self,
        audit_event_id: UUID,
        decision_type: DecisionType,
        rationale: str,
        selected_option: str,
        confidence: float,
        alternatives: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentDecision:
        """
        Log an agent decision.

        Args:
            audit_event_id: Parent audit event ID
            decision_type: Type of decision
            rationale: Why this decision was made
            selected_option: The chosen option
            confidence: Confidence score (0-1)
            alternatives: Other options considered
            metadata: Additional metadata

        Returns:
            The created decision record
        """
        decision = AgentDecision(
            audit_event_id=audit_event_id,
            decision_type=decision_type,
            rationale=rationale,
            selected_option=selected_option,
            confidence=confidence,
            alternatives=alternatives or [],
            metadata=metadata or {},
        )

        async with self._lock:
            self._pending_decisions.append(decision)

        logger.debug(f"Logged decision: {decision_type.value} -> {selected_option}")
        return decision

    # Convenience methods for common events

    async def log_agent_started(
        self,
        agent_id: str,
        session_id: str,
        agent_name: str,
        capabilities: list[str],
        config: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Log agent start event."""
        return await self.log_event(
            event_type=AuditEventType.AGENT_STARTED,
            action=f"Agent '{agent_name}' started",
            agent_id=agent_id,
            session_id=session_id,
            input_data={
                "name": agent_name,
                "capabilities": capabilities,
                "config": config,
            },
        )

    async def log_agent_completed(
        self,
        agent_id: str,
        session_id: str,
        result: Any,
        turns_taken: int,
        execution_time_ms: int,
    ) -> AuditEvent:
        """Log agent completion event."""
        return await self.log_event(
            event_type=AuditEventType.AGENT_COMPLETED,
            action="Agent completed successfully",
            agent_id=agent_id,
            session_id=session_id,
            output_data={
                "result": str(result)[:1000] if result else None,  # Truncate large results
                "turns_taken": turns_taken,
                "execution_time_ms": execution_time_ms,
            },
        )

    async def log_agent_failed(
        self,
        agent_id: str,
        session_id: str,
        error: str,
        turns_taken: int,
        execution_time_ms: int,
    ) -> AuditEvent:
        """Log agent failure event."""
        return await self.log_event(
            event_type=AuditEventType.AGENT_FAILED,
            action=f"Agent failed: {error[:100]}",
            agent_id=agent_id,
            session_id=session_id,
            output_data={
                "error": error,
                "turns_taken": turns_taken,
                "execution_time_ms": execution_time_ms,
            },
        )

    async def log_tool_called(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> AuditEvent:
        """Log tool call event."""
        return await self.log_event(
            event_type=AuditEventType.TOOL_CALLED,
            action=f"Tool '{tool_name}' called",
            agent_id=agent_id,
            session_id=session_id,
            input_data={
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )

    async def log_tool_completed(
        self,
        agent_id: str,
        session_id: str,
        tool_name: str,
        success: bool,
        output: Any,
        execution_time_ms: int,
        error: str | None = None,
    ) -> AuditEvent:
        """Log tool completion event."""
        event_type = AuditEventType.TOOL_COMPLETED if success else AuditEventType.TOOL_FAILED
        return await self.log_event(
            event_type=event_type,
            action=f"Tool '{tool_name}' {'completed' if success else 'failed'}",
            agent_id=agent_id,
            session_id=session_id,
            output_data={
                "tool_name": tool_name,
                "success": success,
                "output": str(output)[:1000] if output else None,
                "execution_time_ms": execution_time_ms,
                "error": error,
            },
        )

    async def log_approval_requested(
        self,
        agent_id: str,
        session_id: str,
        action: str,
        risk_level: str,
        context: dict[str, Any],
    ) -> AuditEvent:
        """Log HITL approval request."""
        return await self.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED,
            action=f"Approval requested for: {action[:100]}",
            agent_id=agent_id,
            session_id=session_id,
            input_data={
                "action": action,
                "risk_level": risk_level,
                "context": context,
            },
        )

    async def log_approval_response(
        self,
        agent_id: str,
        session_id: str,
        request_id: str,
        approved: bool,
        responder: str | None = None,
        reason: str | None = None,
    ) -> AuditEvent:
        """Log HITL approval response."""
        event_type = AuditEventType.APPROVAL_GRANTED if approved else AuditEventType.APPROVAL_DENIED
        return await self.log_event(
            event_type=event_type,
            action=f"Approval {'granted' if approved else 'denied'}",
            agent_id=agent_id,
            session_id=session_id,
            user_id=responder,
            output_data={
                "request_id": request_id,
                "approved": approved,
                "responder": responder,
                "reason": reason,
            },
        )

    async def log_content_filtered(
        self,
        agent_id: str,
        session_id: str,
        filter_type: str,
        original_content: str,
        filtered_content: str | None,
        reason: str,
    ) -> AuditEvent:
        """Log content filtering event."""
        return await self.log_event(
            event_type=AuditEventType.CONTENT_FILTERED,
            action=f"Content filtered by {filter_type}",
            agent_id=agent_id,
            session_id=session_id,
            input_data={
                "original_content_preview": original_content[:200] if original_content else None,
            },
            output_data={
                "filter_type": filter_type,
                "reason": reason,
                "filtered": filtered_content is not None,
            },
        )

    async def log_llm_request(
        self,
        agent_id: str,
        session_id: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        messages_count: int,
    ) -> AuditEvent:
        """Log LLM request event."""
        return await self.log_event(
            event_type=AuditEventType.LLM_REQUEST,
            action=f"LLM request to {provider}/{model}",
            agent_id=agent_id,
            session_id=session_id,
            input_data={
                "provider": provider,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "messages_count": messages_count,
            },
        )

    async def log_llm_response(
        self,
        agent_id: str,
        session_id: str,
        provider: str,
        model: str,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float | None = None,
        cached_tokens: int = 0,
    ) -> AuditEvent:
        """Log LLM response event."""
        return await self.log_event(
            event_type=AuditEventType.LLM_RESPONSE,
            action=f"LLM response from {provider}/{model}",
            agent_id=agent_id,
            session_id=session_id,
            output_data={
                "provider": provider,
                "model": model,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "cached_tokens": cached_tokens,
            },
        )

    @asynccontextmanager
    async def audit_context(
        self,
        session_id: str,
        agent_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncIterator[AuditContext]:
        """
        Context manager for tracking audit events within a session.

        Usage:
            async with audit_logger.audit_context("session-123", "agent-456") as ctx:
                # Events logged here will have session/agent IDs set automatically
                pass
        """
        ctx = AuditContext(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
        )
        try:
            yield ctx
        finally:
            # Ensure pending events are flushed
            await self._flush()

    async def get_events(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """
        Query audit events.

        Args:
            agent_id: Filter by agent
            session_id: Filter by session
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results

        Returns:
            List of audit events
        """
        events_data = await self._client.get_audit_events(
            agent_id=agent_id,
            session_id=session_id,
            event_type=event_type.value if event_type else None,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        return [AuditEvent.from_dict(e) for e in events_data]

    async def verify_integrity(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Verify integrity of audit log."""
        return await self._client.verify_audit_integrity(start_time, end_time)


# Singleton instance
_default_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get or create the default audit logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger()
    return _default_logger


async def init_audit_logger() -> AuditLogger:
    """Initialize and start the audit logger."""
    audit_logger = get_audit_logger()
    await audit_logger.start()
    return audit_logger
