"""
Titan Persistence - SQLAlchemy Models

Defines the data models for audit logging and decision tracking.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Agent lifecycle
    AGENT_CREATED = "agent.created"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_CANCELLED = "agent.cancelled"

    # Tool execution
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # HITL events
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_DENIED = "approval.denied"
    APPROVAL_TIMEOUT = "approval.timeout"

    # Security events
    CONTENT_FILTERED = "security.content_filtered"
    PERMISSION_DENIED = "security.permission_denied"
    RATE_LIMIT_HIT = "security.rate_limit"

    # System events
    TOPOLOGY_CHANGED = "system.topology_changed"
    CONFIG_CHANGED = "system.config_changed"
    BUDGET_EXCEEDED = "system.budget_exceeded"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_ERROR = "llm.error"


class AuditEvent(BaseModel):
    """
    Immutable audit event record.

    Each event has a SHA256 checksum computed from its contents
    to ensure immutability verification.
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: AuditEventType
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    action: str
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    checksum: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Compute checksum after initialization."""
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of event contents."""
        content = {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "action": self.action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify that the checksum matches the content."""
        expected = self._compute_checksum()
        return self.checksum == expected

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "action": self.action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            timestamp=data["timestamp"] if isinstance(data["timestamp"], datetime)
            else datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            action=data["action"],
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )


class DecisionType(str, Enum):
    """Types of agent decisions."""

    TOOL_SELECTION = "tool_selection"
    MODEL_SELECTION = "model_selection"
    TOPOLOGY_SELECTION = "topology_selection"
    TASK_DELEGATION = "task_delegation"
    ERROR_RECOVERY = "error_recovery"
    BUDGET_ALLOCATION = "budget_allocation"


class AgentDecision(BaseModel):
    """
    Record of an agent decision for auditability.

    Links to the parent audit event and captures decision rationale.
    """

    id: UUID = Field(default_factory=uuid4)
    audit_event_id: UUID
    decision_type: DecisionType
    rationale: str
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    selected_option: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": str(self.id),
            "audit_event_id": str(self.audit_event_id),
            "decision_type": self.decision_type.value,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "selected_option": self.selected_option,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDecision:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            audit_event_id=UUID(data["audit_event_id"])
            if isinstance(data["audit_event_id"], str)
            else data["audit_event_id"],
            decision_type=DecisionType(data["decision_type"]),
            rationale=data["rationale"],
            alternatives=data.get("alternatives", []),
            selected_option=data["selected_option"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {}),
        )


# SQLAlchemy table definitions (for use with alembic migrations)
AUDIT_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    agent_id VARCHAR(100),
    session_id VARCHAR(100),
    user_id VARCHAR(100),
    action VARCHAR(255) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    metadata JSONB DEFAULT '{}',
    checksum VARCHAR(64) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_agent_id ON audit_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_session_id ON audit_events(session_id);
"""

AGENT_DECISIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_decisions (
    id UUID PRIMARY KEY,
    audit_event_id UUID REFERENCES audit_events(id),
    decision_type VARCHAR(50) NOT NULL,
    rationale TEXT,
    alternatives JSONB DEFAULT '[]',
    selected_option VARCHAR(255) NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_decisions_audit_event_id ON agent_decisions(audit_event_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_decision_type ON agent_decisions(decision_type);
"""
