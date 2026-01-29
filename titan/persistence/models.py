"""
Titan Persistence - SQLAlchemy Models

Defines the data models for audit logging and decision tracking.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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

# ============================================================================
# Assembly Theory Tables
# ============================================================================

ASSEMBLY_PATHS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS assembly_paths (
    id UUID PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    path_type VARCHAR(50) NOT NULL,
    assembly_index INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_assembly_paths_session_id ON assembly_paths(session_id);
CREATE INDEX IF NOT EXISTS idx_assembly_paths_path_type ON assembly_paths(path_type);
CREATE INDEX IF NOT EXISTS idx_assembly_paths_created_at ON assembly_paths(created_at);
"""

ASSEMBLY_STEPS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS assembly_steps (
    id UUID PRIMARY KEY,
    path_id UUID REFERENCES assembly_paths(id) ON DELETE CASCADE,
    step_number INT NOT NULL,
    step_type VARCHAR(50) NOT NULL,
    input_state JSONB,
    output_state JSONB,
    transformation TEXT,
    duration_ms FLOAT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_assembly_steps_path_id ON assembly_steps(path_id);
CREATE INDEX IF NOT EXISTS idx_assembly_steps_step_type ON assembly_steps(step_type);
CREATE INDEX IF NOT EXISTS idx_assembly_steps_created_at ON assembly_steps(created_at);
"""

ASSEMBLY_METRICS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS assembly_metrics (
    id UUID PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_assembly FLOAT NOT NULL,
    selection_signal VARCHAR(20) NOT NULL,
    unique_paths INT NOT NULL DEFAULT 0,
    total_objects INT NOT NULL DEFAULT 0,
    max_assembly_index INT NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_assembly_metrics_session_id ON assembly_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_assembly_metrics_timestamp ON assembly_metrics(timestamp);
"""

# ============================================================================
# Stigmergy Tables
# ============================================================================

PHEROMONE_TRACES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pheromone_traces (
    id UUID PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    trace_type VARCHAR(50) NOT NULL,
    location VARCHAR(255) NOT NULL,
    intensity FLOAT NOT NULL CHECK (intensity >= 0),
    decay_rate FLOAT NOT NULL DEFAULT 0.1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    payload JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_pheromone_traces_location ON pheromone_traces(location);
CREATE INDEX IF NOT EXISTS idx_pheromone_traces_trace_type ON pheromone_traces(trace_type);
CREATE INDEX IF NOT EXISTS idx_pheromone_traces_expires_at ON pheromone_traces(expires_at);
"""

# ============================================================================
# Territory Tables
# ============================================================================

TERRITORIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS territories (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    owner_agent_id VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dissolved_at TIMESTAMPTZ,
    boundary_config JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_territories_owner_agent_id ON territories(owner_agent_id);
CREATE INDEX IF NOT EXISTS idx_territories_created_at ON territories(created_at);
"""

TERRITORY_AGENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS territory_agents (
    id UUID PRIMARY KEY,
    territory_id UUID REFERENCES territories(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    role VARCHAR(50) DEFAULT 'member',
    is_boundary_agent BOOLEAN DEFAULT FALSE,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    left_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_territory_agents_territory_id ON territory_agents(territory_id);
CREATE INDEX IF NOT EXISTS idx_territory_agents_agent_id ON territory_agents(agent_id);
"""

# ============================================================================
# Neighborhood Tables
# ============================================================================

NEIGHBOR_INTERACTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS neighbor_interactions (
    id UUID PRIMARY KEY,
    agent_a_id VARCHAR(100) NOT NULL,
    agent_b_id VARCHAR(100) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    quality FLOAT CHECK (quality >= 0 AND quality <= 1),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_neighbor_interactions_agent_a ON neighbor_interactions(agent_a_id);
CREATE INDEX IF NOT EXISTS idx_neighbor_interactions_agent_b ON neighbor_interactions(agent_b_id);
CREATE INDEX IF NOT EXISTS idx_neighbor_interactions_timestamp ON neighbor_interactions(timestamp);
"""

# Combined SQL for all assembly-related tables
ALL_ASSEMBLY_TABLES_SQL = (
    ASSEMBLY_PATHS_TABLE_SQL
    + ASSEMBLY_STEPS_TABLE_SQL
    + ASSEMBLY_METRICS_TABLE_SQL
    + PHEROMONE_TRACES_TABLE_SQL
    + TERRITORIES_TABLE_SQL
    + TERRITORY_AGENTS_TABLE_SQL
    + NEIGHBOR_INTERACTIONS_TABLE_SQL
)

# ============================================================================
# Batch Cleanup Tables
# ============================================================================

BATCH_CLEANUP_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS batch_cleanup_log (
    id SERIAL PRIMARY KEY,
    cleanup_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    items_processed INT DEFAULT 0,
    items_deleted INT DEFAULT 0,
    errors JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_batch_cleanup_log_started_at
ON batch_cleanup_log(started_at);

CREATE INDEX IF NOT EXISTS idx_batch_cleanup_log_cleanup_type
ON batch_cleanup_log(cleanup_type);
"""

# Index for efficient stalled batch queries
BATCH_JOBS_STALLED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_batch_jobs_stalled
ON batch_jobs (status, created_at)
WHERE status IN ('processing', 'paused');
"""

# Index for cleanup of old batches (terminal states)
BATCH_JOBS_CLEANUP_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_batch_jobs_cleanup
ON batch_jobs (completed_at, status)
WHERE status IN ('completed', 'failed', 'cancelled', 'partially_completed');
"""
