"""
Titan Persistence Layer

Provides:
- PostgreSQL-based durable audit logging
- Conversation checkpointing
- Agent state persistence
- Session recovery
"""

from titan.persistence.audit import (
    AuditContext,
    AuditLogger,
    get_audit_logger,
    init_audit_logger,
)
from titan.persistence.checkpoint import (
    Checkpoint,
    CheckpointManager,
    create_checkpoint,
    restore_checkpoint,
)
from titan.persistence.models import (
    AgentDecision,
    AuditEvent,
    AuditEventType,
    DecisionType,
)
from titan.persistence.postgres import (
    PostgresClient,
    PostgresConfig,
    get_postgres_client,
    init_postgres,
)

__all__ = [
    # Checkpoint
    "Checkpoint",
    "CheckpointManager",
    "create_checkpoint",
    "restore_checkpoint",
    # PostgreSQL
    "PostgresClient",
    "PostgresConfig",
    "get_postgres_client",
    "init_postgres",
    # Audit
    "AuditEvent",
    "AgentDecision",
    "AuditEventType",
    "DecisionType",
    "AuditLogger",
    "AuditContext",
    "get_audit_logger",
    "init_audit_logger",
]
