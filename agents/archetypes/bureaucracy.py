"""
Bureaucracy Agent - Rule-bound administrative machine.

Implements bureaucratic organization patterns:
- Strict adherence to written rules
- Specialized roles within hierarchy
- Institutional "immortality" (survives individual failures)
- Formal procedures and documentation

Based on Weber's ideal-type bureaucracy and its treatment
in the assembly research as a "machine-body."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentState

logger = logging.getLogger("titan.agents.bureaucracy")


class RuleType(str, Enum):
    """Types of bureaucratic rules."""

    PROCEDURE = "procedure"      # How to do something
    CONSTRAINT = "constraint"    # What cannot be done
    REQUIREMENT = "requirement"  # What must be done
    AUTHORIZATION = "authorization"  # Who can do what


class RequestStatus(str, Enum):
    """Status of a bureaucratic request."""

    SUBMITTED = "submitted"
    PENDING_REVIEW = "pending_review"
    REQUIRES_INFO = "requires_info"
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"
    COMPLETED = "completed"


class BureaucraticRole(str, Enum):
    """Roles within the bureaucracy."""

    CLERK = "clerk"          # Handles routine processing
    REVIEWER = "reviewer"    # Reviews and validates
    SUPERVISOR = "supervisor"  # Approves and escalates
    ADMINISTRATOR = "administrator"  # Sets policies
    ARCHIVIST = "archivist"  # Maintains records


@dataclass
class Rule:
    """A bureaucratic rule."""

    rule_id: str
    rule_type: RuleType
    description: str
    applies_to: list[str] = field(default_factory=list)  # Roles
    conditions: dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def applies(self, role: BureaucraticRole, context: dict[str, Any]) -> bool:
        """Check if this rule applies in context.

        Args:
            role: The role being checked.
            context: Contextual information.

        Returns:
            True if rule applies.
        """
        if not self.active:
            return False

        if self.applies_to and role.value not in self.applies_to:
            return False

        # Check conditions
        for key, expected in self.conditions.items():
            if key in context and context[key] != expected:
                return False

        return True


@dataclass
class Request:
    """A bureaucratic request being processed."""

    request_id: str
    request_type: str
    description: str
    submitted_by: str
    status: RequestStatus = RequestStatus.SUBMITTED
    assigned_to: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def add_history(self, action: str, by: str, notes: str = "") -> None:
        """Add an entry to request history."""
        self.history.append({
            "action": action,
            "by": by,
            "notes": notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


@dataclass
class BureaucracyConfig:
    """Configuration for bureaucracy behavior."""

    require_documentation: bool = True
    escalation_threshold: int = 3  # Max processing attempts before escalate
    approval_chain_length: int = 2  # Approvals needed
    auto_archive_days: int = 30
    strict_rule_enforcement: bool = True


class BureaucracyAgent(BaseAgent):
    """
    Agent implementing bureaucratic organization patterns.

    The bureaucracy operates as a "machine-body" with:
    - Written rules that constrain individual discretion
    - Specialized roles with defined responsibilities
    - Formal procedures for all actions
    - Institutional memory via documentation
    - Survival beyond individual agent failures

    Capabilities:
    - Process requests according to rules
    - Route requests through approval chains
    - Maintain comprehensive records
    - Enforce constraints consistently
    """

    def __init__(
        self,
        role: BureaucraticRole = BureaucraticRole.CLERK,
        config: BureaucracyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize bureaucracy agent.

        Args:
            role: Role within the bureaucracy.
            config: Bureaucracy configuration.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", f"bureaucracy_{role.value}")
        kwargs.setdefault("capabilities", [
            "request_processing",
            "rule_enforcement",
            "documentation",
            "escalation",
        ])
        super().__init__(**kwargs)

        self._role = role
        self._config = config or BureaucracyConfig()
        self._rules: list[Rule] = []
        self._pending_requests: list[Request] = []
        self._completed_requests: list[Request] = []
        self._peers: dict[BureaucraticRole, list[str]] = {}

    @property
    def role(self) -> BureaucraticRole:
        """Get agent's bureaucratic role."""
        return self._role

    async def initialize(self) -> None:
        """Initialize bureaucracy agent."""
        self._state = AgentState.READY
        self._load_default_rules()
        logger.info(f"Bureaucracy agent initialized (role={self._role.value})")

    def _load_default_rules(self) -> None:
        """Load default bureaucratic rules."""
        self._rules = [
            Rule(
                rule_id="R001",
                rule_type=RuleType.PROCEDURE,
                description="All requests must be documented with reason and requester",
                applies_to=["clerk", "reviewer"],
            ),
            Rule(
                rule_id="R002",
                rule_type=RuleType.AUTHORIZATION,
                description="Only supervisors can approve requests over threshold",
                applies_to=["supervisor"],
                conditions={"requires_supervisor": True},
            ),
            Rule(
                rule_id="R003",
                rule_type=RuleType.CONSTRAINT,
                description="Requests cannot be self-approved",
                applies_to=["clerk", "reviewer", "supervisor"],
            ),
            Rule(
                rule_id="R004",
                rule_type=RuleType.REQUIREMENT,
                description="All approvals must include justification",
                applies_to=["reviewer", "supervisor"],
            ),
        ]

    async def work(self) -> dict[str, Any]:
        """Process pending requests according to role."""
        result = {
            "role": self._role.value,
            "processed": [],
            "escalated": [],
            "completed": [],
        }

        # Process based on role
        for request in self._pending_requests[:10]:
            try:
                action_result = await self._process_request(request)
                result["processed"].append({
                    "request_id": request.request_id,
                    "action": action_result["action"],
                    "new_status": request.status.value,
                })

                if action_result["action"] == "escalated":
                    result["escalated"].append(request.request_id)
                elif request.status == RequestStatus.COMPLETED:
                    result["completed"].append(request.request_id)

            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {e}")
                request.add_history("error", self.agent_id, str(e))

        return result

    async def shutdown(self) -> None:
        """Shutdown bureaucracy agent."""
        # Archive any pending work
        logger.info(
            f"Bureaucracy agent shutdown "
            f"(pending={len(self._pending_requests)}, "
            f"completed={len(self._completed_requests)})"
        )

    async def _process_request(self, request: Request) -> dict[str, Any]:
        """Process a single request according to rules.

        Args:
            request: Request to process.

        Returns:
            Processing result.
        """
        result = {"action": "processed", "violations": []}

        # Check applicable rules
        context = {
            "request_type": request.request_type,
            "status": request.status.value,
            **request.data,
        }

        for rule in self._rules:
            if rule.applies(self._role, context):
                violation = self._check_rule_compliance(rule, request)
                if violation:
                    result["violations"].append(violation)

        # Handle violations in strict mode
        if result["violations"] and self._config.strict_rule_enforcement:
            request.status = RequestStatus.REQUIRES_INFO
            request.add_history(
                "requires_info",
                self.agent_id,
                f"Rule violations: {result['violations']}",
            )
            return result

        # Process based on role
        if self._role == BureaucraticRole.CLERK:
            result = await self._clerk_process(request)
        elif self._role == BureaucraticRole.REVIEWER:
            result = await self._reviewer_process(request)
        elif self._role == BureaucraticRole.SUPERVISOR:
            result = await self._supervisor_process(request)
        elif self._role == BureaucraticRole.ADMINISTRATOR:
            result = await self._administrator_process(request)
        elif self._role == BureaucraticRole.ARCHIVIST:
            result = await self._archivist_process(request)

        return result

    def _check_rule_compliance(
        self,
        rule: Rule,
        request: Request,
    ) -> str | None:
        """Check if a request complies with a rule.

        Args:
            rule: Rule to check.
            request: Request to validate.

        Returns:
            Violation description if any, None if compliant.
        """
        if rule.rule_type == RuleType.REQUIREMENT:
            # Check documentation requirement
            if "documentation" in rule.description.lower():
                if not request.description or not request.submitted_by:
                    return f"Missing required documentation ({rule.rule_id})"

            # Check justification requirement
            if "justification" in rule.description.lower():
                if "justification" not in request.data:
                    return f"Missing justification ({rule.rule_id})"

        elif rule.rule_type == RuleType.CONSTRAINT:
            # Check self-approval constraint
            if "self-approved" in rule.description.lower():
                if request.submitted_by == request.assigned_to:
                    return f"Self-approval not allowed ({rule.rule_id})"

        return None

    async def _clerk_process(self, request: Request) -> dict[str, Any]:
        """Clerk processing: initial validation and routing."""
        result = {"action": "routed"}

        # Validate request
        if not request.description:
            request.status = RequestStatus.REQUIRES_INFO
            request.add_history("validation_failed", self.agent_id, "Missing description")
            result["action"] = "validation_failed"
            return result

        # Route to reviewer
        request.status = RequestStatus.PENDING_REVIEW
        request.add_history("routed", self.agent_id, "Sent to reviewer")

        # Find reviewer peer
        if BureaucraticRole.REVIEWER in self._peers:
            reviewer_id = self._peers[BureaucraticRole.REVIEWER][0]
            request.assigned_to = reviewer_id

        return result

    async def _reviewer_process(self, request: Request) -> dict[str, Any]:
        """Reviewer processing: validate and approve/escalate."""
        result = {"action": "reviewed"}

        # Check if needs supervisor approval
        needs_supervisor = request.data.get("requires_supervisor", False)
        processing_attempts = len([
            h for h in request.history if h["action"] == "reviewed"
        ])

        if needs_supervisor or processing_attempts >= self._config.escalation_threshold:
            # Escalate to supervisor
            request.status = RequestStatus.ESCALATED
            request.add_history("escalated", self.agent_id, "Requires supervisor approval")
            result["action"] = "escalated"

            if BureaucraticRole.SUPERVISOR in self._peers:
                request.assigned_to = self._peers[BureaucraticRole.SUPERVISOR][0]
        else:
            # Approve
            request.status = RequestStatus.APPROVED
            request.add_history("approved", self.agent_id, "Meets requirements")
            result["action"] = "approved"

        return result

    async def _supervisor_process(self, request: Request) -> dict[str, Any]:
        """Supervisor processing: final approval authority."""
        result = {"action": "supervisor_reviewed"}

        # Count approvals in chain
        approvals = len([
            h for h in request.history
            if h["action"] in ["approved", "supervisor_approved"]
        ])

        if approvals >= self._config.approval_chain_length - 1:
            # Final approval
            request.status = RequestStatus.APPROVED
            request.add_history("supervisor_approved", self.agent_id, "Final approval")
            result["action"] = "approved"
        else:
            # Needs more review
            request.add_history("supervisor_reviewed", self.agent_id, "Additional review needed")

        return result

    async def _administrator_process(self, request: Request) -> dict[str, Any]:
        """Administrator processing: policy-level decisions."""
        result = {"action": "administered"}

        # Administrators handle policy requests
        if request.request_type == "policy_change":
            request.add_history("policy_reviewed", self.agent_id, "Policy reviewed")
            result["action"] = "policy_reviewed"

        return result

    async def _archivist_process(self, request: Request) -> dict[str, Any]:
        """Archivist processing: record keeping and archival."""
        result = {"action": "archived"}

        # Move completed requests to archive
        if request.status == RequestStatus.APPROVED:
            request.status = RequestStatus.COMPLETED
            request.completed_at = datetime.now(timezone.utc)
            request.add_history("archived", self.agent_id, "Request completed and archived")

            if request in self._pending_requests:
                self._pending_requests.remove(request)
            self._completed_requests.append(request)

        return result

    # =========================================================================
    # Request Management
    # =========================================================================

    def submit_request(
        self,
        request_type: str,
        description: str,
        submitted_by: str,
        data: dict[str, Any] | None = None,
    ) -> Request:
        """Submit a new request to the bureaucracy.

        Args:
            request_type: Type of request.
            description: Request description.
            submitted_by: Agent submitting the request.
            data: Additional request data.

        Returns:
            The created Request.
        """
        import uuid
        request = Request(
            request_id=f"REQ-{uuid.uuid4().hex[:8]}",
            request_type=request_type,
            description=description,
            submitted_by=submitted_by,
            data=data or {},
        )

        request.add_history("submitted", submitted_by, "Request created")
        self._pending_requests.append(request)

        logger.info(f"Request submitted: {request.request_id} by {submitted_by}")
        return request

    def get_request_status(self, request_id: str) -> Request | None:
        """Get status of a request.

        Args:
            request_id: Request ID to look up.

        Returns:
            Request if found, None otherwise.
        """
        for req in self._pending_requests + self._completed_requests:
            if req.request_id == request_id:
                return req
        return None

    # =========================================================================
    # Rule Management
    # =========================================================================

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the bureaucracy.

        Args:
            rule: Rule to add.
        """
        self._rules.append(rule)
        logger.info(f"Rule added: {rule.rule_id}")

    def get_applicable_rules(
        self,
        role: BureaucraticRole,
        context: dict[str, Any],
    ) -> list[Rule]:
        """Get rules that apply in a context.

        Args:
            role: Role to check for.
            context: Contextual information.

        Returns:
            List of applicable rules.
        """
        return [r for r in self._rules if r.applies(role, context)]

    # =========================================================================
    # Peer Management
    # =========================================================================

    def register_peer(self, role: BureaucraticRole, agent_id: str) -> None:
        """Register a peer agent by role.

        Args:
            role: Role of the peer.
            agent_id: Agent ID.
        """
        if role not in self._peers:
            self._peers[role] = []
        if agent_id not in self._peers[role]:
            self._peers[role].append(agent_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get bureaucracy statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "role": self._role.value,
            "pending_requests": len(self._pending_requests),
            "completed_requests": len(self._completed_requests),
            "active_rules": len([r for r in self._rules if r.active]),
            "peers": {
                role.value: len(agents)
                for role, agents in self._peers.items()
            },
        }
