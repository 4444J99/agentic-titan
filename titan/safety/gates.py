"""
Titan Safety - Approval Gates

Defines approval gates that pause execution pending human approval.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from titan.safety.policies import ActionPolicy, RiskLevel

logger = logging.getLogger("titan.safety.gates")


class ApprovalStatus(StrEnum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """
    Request for human approval of an action.

    Contains all context needed for a human to make an informed decision.
    """

    id: UUID = field(default_factory=uuid4)
    action: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    agent_id: str = ""
    session_id: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    timeout_seconds: int = 300
    fallback_action: str = "deny"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "action": self.action,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "fallback_action": self.fallback_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRequest:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            action=data["action"],
            description=data.get("description", ""),
            risk_level=RiskLevel(data["risk_level"]),
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            tool_name=data.get("tool_name"),
            arguments=data.get("arguments", {}),
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"],
            timeout_seconds=data.get("timeout_seconds", 300),
            fallback_action=data.get("fallback_action", "deny"),
        )


@dataclass
class ApprovalResult:
    """
    Result of an approval request.

    Contains the decision and any associated metadata.
    """

    request_id: UUID
    status: ApprovalStatus
    approved: bool = False
    responder: str | None = None
    reason: str | None = None
    responded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": str(self.request_id),
            "status": self.status.value,
            "approved": self.approved,
            "responder": self.responder,
            "reason": self.reason,
            "responded_at": self.responded_at.isoformat(),
            "metadata": self.metadata,
        }


class ApprovalGate:
    """
    Gate that pauses execution pending human approval.

    Manages the approval workflow for a specific action type.
    """

    def __init__(
        self,
        name: str,
        policy: ActionPolicy,
        description: str = "",
    ) -> None:
        self.name = name
        self.policy = policy
        self.description = description
        self._pending_requests: dict[UUID, asyncio.Event] = {}
        self._results: dict[UUID, ApprovalResult] = {}

    async def request_approval(
        self,
        action: str,
        agent_id: str,
        session_id: str,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """
        Create an approval request.

        Args:
            action: Description of the action
            agent_id: Agent requesting approval
            session_id: Session identifier
            tool_name: Name of tool being called
            arguments: Tool arguments
            context: Additional context

        Returns:
            ApprovalRequest with unique ID
        """
        request = ApprovalRequest(
            action=action,
            description=self.description or self.policy.description,
            risk_level=self.policy.risk_level,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments or {},
            context=context or {},
            timeout_seconds=self.policy.timeout_seconds,
            fallback_action=self.policy.fallback_action,
        )

        # Create event for waiting
        self._pending_requests[request.id] = asyncio.Event()

        logger.info(
            f"Approval request {request.id} created: {action[:50]}... "
            f"(risk={self.policy.risk_level.value})"
        )

        return request

    async def wait_for_approval(
        self,
        request_id: UUID,
        timeout: int | None = None,
    ) -> ApprovalResult:
        """
        Wait for an approval decision.

        Args:
            request_id: ID of the approval request
            timeout: Optional timeout override

        Returns:
            ApprovalResult with the decision
        """
        if request_id not in self._pending_requests:
            return ApprovalResult(
                request_id=request_id,
                status=ApprovalStatus.CANCELLED,
                approved=False,
                reason="Request not found",
            )

        event = self._pending_requests[request_id]
        wait_timeout = timeout or self.policy.timeout_seconds

        try:
            await asyncio.wait_for(event.wait(), timeout=wait_timeout)

            # Return the result
            result = self._results.get(request_id)
            if result:
                return result

            # This shouldn't happen, but handle it
            return ApprovalResult(
                request_id=request_id,
                status=ApprovalStatus.CANCELLED,
                approved=False,
                reason="Result not found after approval",
            )

        except TimeoutError:
            logger.warning(f"Approval request {request_id} timed out after {wait_timeout}s")

            # Handle timeout based on fallback action
            if self.policy.fallback_action == "allow":
                result = ApprovalResult(
                    request_id=request_id,
                    status=ApprovalStatus.TIMEOUT,
                    approved=True,
                    reason=f"Auto-approved after {wait_timeout}s timeout (fallback: allow)",
                )
            elif self.policy.fallback_action == "escalate":
                result = ApprovalResult(
                    request_id=request_id,
                    status=ApprovalStatus.ESCALATED,
                    approved=False,
                    reason=f"Escalated after {wait_timeout}s timeout",
                )
            else:
                result = ApprovalResult(
                    request_id=request_id,
                    status=ApprovalStatus.TIMEOUT,
                    approved=False,
                    reason=f"Denied after {wait_timeout}s timeout (fallback: deny)",
                )

            self._results[request_id] = result
            return result

        finally:
            # Clean up
            self._pending_requests.pop(request_id, None)

    def respond(
        self,
        request_id: UUID,
        approved: bool,
        responder: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """
        Respond to an approval request.

        Args:
            request_id: ID of the request
            approved: Whether the action is approved
            responder: Who made the decision
            reason: Reason for the decision

        Returns:
            True if response was recorded, False if request not found
        """
        if request_id not in self._pending_requests:
            logger.warning(f"Cannot respond to unknown request {request_id}")
            return False

        result = ApprovalResult(
            request_id=request_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED,
            approved=approved,
            responder=responder,
            reason=reason,
        )

        self._results[request_id] = result
        self._pending_requests[request_id].set()

        logger.info(
            f"Approval request {request_id} "
            f"{'approved' if approved else 'denied'} by {responder or 'unknown'}"
        )

        return True

    def get_pending_requests(self) -> list[UUID]:
        """Get all pending request IDs."""
        return list(self._pending_requests.keys())


class GateRegistry:
    """
    Registry of approval gates.

    Manages gates for different action types and risk levels.
    """

    def __init__(self) -> None:
        self._gates: dict[str, ApprovalGate] = {}
        self._risk_level_gates: dict[RiskLevel, ApprovalGate] = {}
        self._setup_default_gates()

    def _setup_default_gates(self) -> None:
        """Set up default gates for each risk level."""
        # Critical actions - always require approval with long timeout
        self._risk_level_gates[RiskLevel.CRITICAL] = ApprovalGate(
            name="critical",
            policy=ActionPolicy(
                risk_level=RiskLevel.CRITICAL,
                requires_approval=True,
                timeout_seconds=600,
                fallback_action="deny",
                description="Critical action requiring approval",
            ),
        )

        # High risk - require approval
        self._risk_level_gates[RiskLevel.HIGH] = ApprovalGate(
            name="high",
            policy=ActionPolicy(
                risk_level=RiskLevel.HIGH,
                requires_approval=True,
                timeout_seconds=300,
                fallback_action="deny",
                description="High-risk action requiring approval",
            ),
        )

        # Medium risk - optional approval
        self._risk_level_gates[RiskLevel.MEDIUM] = ApprovalGate(
            name="medium",
            policy=ActionPolicy(
                risk_level=RiskLevel.MEDIUM,
                requires_approval=False,
                timeout_seconds=120,
                fallback_action="deny",
                description="Medium-risk action",
            ),
        )

        # Low risk - auto-approve
        self._risk_level_gates[RiskLevel.LOW] = ApprovalGate(
            name="low",
            policy=ActionPolicy(
                risk_level=RiskLevel.LOW,
                requires_approval=False,
                timeout_seconds=60,
                fallback_action="allow",
                description="Low-risk action",
            ),
        )

    def register(self, name: str, gate: ApprovalGate) -> None:
        """Register a named gate."""
        self._gates[name] = gate
        logger.info(f"Registered gate '{name}' (risk={gate.policy.risk_level.value})")

    def get(self, name: str) -> ApprovalGate | None:
        """Get a gate by name."""
        return self._gates.get(name)

    def get_for_risk_level(self, risk_level: RiskLevel) -> ApprovalGate:
        """Get the gate for a specific risk level."""
        return self._risk_level_gates[risk_level]

    def get_for_policy(self, policy: ActionPolicy) -> ApprovalGate:
        """Get the appropriate gate for an action policy."""
        # Check for custom gate first
        gate_name = f"{policy.risk_level.value}_{hash(policy.description) % 1000}"
        if gate_name in self._gates:
            return self._gates[gate_name]

        # Return the risk-level gate
        return self._risk_level_gates[policy.risk_level]

    def get_all_pending(self) -> dict[str, list[UUID]]:
        """Get all pending requests across all gates."""
        pending = {}
        for name, gate in self._gates.items():
            reqs = gate.get_pending_requests()
            if reqs:
                pending[name] = reqs
        for level, gate in self._risk_level_gates.items():
            reqs = gate.get_pending_requests()
            if reqs:
                pending[f"risk_{level.value}"] = reqs
        return pending


# Singleton instance
_default_registry: GateRegistry | None = None


def get_gate_registry() -> GateRegistry:
    """Get the default gate registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = GateRegistry()
    return _default_registry
