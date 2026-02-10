"""
Government Branch Agents - Three branches of governance.

Implements separation of powers with:
- ExecutiveAgent: Implements decisions, leads execution
- LegislativeAgent: Proposes and debates policies
- JudicialAgent: Reviews for compliance, resolves disputes

These agents can check each other's work and coordinate
on governance decisions.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agents.framework.base_agent import AgentState, BaseAgent

logger = logging.getLogger("titan.agents.government")


class PolicyStatus(StrEnum):
    """Status of a policy proposal."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    DEBATING = "debating"
    PASSED = "passed"
    REJECTED = "rejected"
    VETOED = "vetoed"
    ENACTED = "enacted"
    INVALIDATED = "invalidated"


class DisputeStatus(StrEnum):
    """Status of a dispute."""

    FILED = "filed"
    REVIEWING = "reviewing"
    HEARING = "hearing"
    DECIDED = "decided"
    APPEALED = "appealed"


@dataclass
class Policy:
    """A policy proposal or enacted policy."""

    policy_id: str
    title: str
    description: str
    proposed_by: str
    status: PolicyStatus = PolicyStatus.DRAFT
    votes_for: int = 0
    votes_against: int = 0
    enacted_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "title": self.title,
            "description": self.description,
            "proposed_by": self.proposed_by,
            "status": self.status.value,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "enacted_at": self.enacted_at.isoformat() if self.enacted_at else None,
            "metadata": self.metadata,
        }


@dataclass
class ExecutiveOrder:
    """An executive order or directive."""

    order_id: str
    title: str
    directive: str
    issued_by: str
    issued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    implements_policy: str | None = None  # Policy ID
    active: bool = True


@dataclass
class Dispute:
    """A dispute requiring judicial review."""

    dispute_id: str
    title: str
    description: str
    filed_by: str
    against: str
    status: DisputeStatus = DisputeStatus.FILED
    ruling: str = ""
    ruled_at: datetime | None = None


class GovernmentAgent(BaseAgent):
    """Base class for government branch agents."""

    branch: str = "unknown"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize government agent."""
        kwargs.setdefault(
            "capabilities",
            [
                "governance",
                "coordination",
                "policy_management",
            ],
        )
        super().__init__(**kwargs)
        self._peer_branches: dict[str, str] = {}  # branch -> agent_id

    def register_peer_branch(self, branch: str, agent_id: str) -> None:
        """Register a peer branch for inter-branch coordination.

        Args:
            branch: The branch name (executive, legislative, judicial).
            agent_id: Agent ID for that branch.
        """
        self._peer_branches[branch] = agent_id

    @abstractmethod
    async def check_peer(self, branch: str, action: dict[str, Any]) -> bool:
        """Check a peer branch's action for validity.

        Args:
            branch: The branch being checked.
            action: The action to check.

        Returns:
            True if action is valid.
        """
        pass


class ExecutiveAgent(GovernmentAgent):
    """
    Executive branch agent - implements decisions and leads execution.

    Responsibilities:
    - Execute enacted policies
    - Issue executive orders
    - Coordinate implementation
    - Report on execution status
    - Subject to judicial review
    """

    branch = "executive"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize executive agent."""
        kwargs.setdefault("name", "executive")
        kwargs["capabilities"] = kwargs.get("capabilities", []) + [
            "policy_execution",
            "order_issuance",
            "coordination",
        ]
        super().__init__(**kwargs)

        self._active_orders: list[ExecutiveOrder] = []
        self._execution_queue: list[Policy] = []
        self._execution_status: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Initialize executive agent."""
        self._state = AgentState.READY
        logger.info("Executive agent initialized")

    async def work(self) -> dict[str, Any]:
        """Execute pending policies and manage orders."""
        result: dict[str, Any] = {
            "branch": "executive",
            "policies_executed": [],
            "orders_issued": [],
            "status_updates": [],
        }

        # Process execution queue
        for policy in self._execution_queue[:5]:  # Process up to 5
            execution_result = await self._execute_policy(policy)
            result["policies_executed"].append(
                {
                    "policy_id": policy.policy_id,
                    "success": execution_result["success"],
                    "details": execution_result.get("details", ""),
                }
            )

        # Update execution status
        for policy_id, status in self._execution_status.items():
            result["status_updates"].append(
                {
                    "policy_id": policy_id,
                    "progress": status.get("progress", 0),
                    "issues": status.get("issues", []),
                }
            )

        return result

    async def shutdown(self) -> None:
        """Shutdown executive agent."""
        logger.info("Executive agent shutdown")

    async def enqueue_policy(self, policy: Policy) -> bool:
        """Add a policy to the execution queue.

        Args:
            policy: Enacted policy to execute.

        Returns:
            True if added to queue.
        """
        if policy.status != PolicyStatus.ENACTED:
            logger.warning(f"Cannot execute non-enacted policy: {policy.policy_id}")
            return False

        self._execution_queue.append(policy)
        return True

    async def _execute_policy(self, policy: Policy) -> dict[str, Any]:
        """Execute a policy.

        Args:
            policy: Policy to execute.

        Returns:
            Execution result.
        """
        self._execution_status[policy.policy_id] = {
            "status": "executing",
            "started_at": datetime.now(UTC).isoformat(),
            "progress": 0,
            "issues": [],
        }

        # Simulate execution
        result = {
            "success": True,
            "details": f"Policy {policy.policy_id} execution initiated",
        }

        self._execution_status[policy.policy_id]["progress"] = 100
        self._execution_status[policy.policy_id]["status"] = "completed"

        # Remove from queue
        if policy in self._execution_queue:
            self._execution_queue.remove(policy)

        return result

    async def issue_order(
        self,
        title: str,
        directive: str,
        implements_policy: str | None = None,
    ) -> ExecutiveOrder:
        """Issue an executive order.

        Args:
            title: Order title.
            directive: The directive content.
            implements_policy: Optional policy this implements.

        Returns:
            The issued ExecutiveOrder.
        """
        import uuid

        order = ExecutiveOrder(
            order_id=f"EO-{uuid.uuid4().hex[:8]}",
            title=title,
            directive=directive,
            issued_by=self.agent_id,
            implements_policy=implements_policy,
        )

        self._active_orders.append(order)
        logger.info(f"Executive order issued: {order.order_id} - {title}")

        return order

    async def veto_policy(self, policy: Policy, reason: str) -> bool:
        """Veto a passed policy.

        Args:
            policy: Policy to veto.
            reason: Reason for veto.

        Returns:
            True if vetoed successfully.
        """
        if policy.status != PolicyStatus.PASSED:
            return False

        policy.status = PolicyStatus.VETOED
        policy.metadata["veto_reason"] = reason
        policy.metadata["vetoed_by"] = self.agent_id

        logger.info(f"Policy vetoed: {policy.policy_id} - {reason}")
        return True

    async def check_peer(self, branch: str, action: dict[str, Any]) -> bool:
        """Check peer branch action (executive checks legislative)."""
        if branch == "legislative":
            # Executive can veto
            return True
        return True

    def get_active_orders(self) -> list[ExecutiveOrder]:
        """Get all active executive orders."""
        return [o for o in self._active_orders if o.active]


class LegislativeAgent(GovernmentAgent):
    """
    Legislative branch agent - proposes and debates policies.

    Responsibilities:
    - Draft policy proposals
    - Debate and discuss policies
    - Vote on policies
    - Override executive vetoes
    - Subject to judicial review
    """

    branch = "legislative"

    def __init__(
        self,
        override_threshold: float = 0.67,
        **kwargs: Any,
    ) -> None:
        """Initialize legislative agent.

        Args:
            override_threshold: Threshold for veto override.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", "legislative")
        kwargs["capabilities"] = kwargs.get("capabilities", []) + [
            "policy_drafting",
            "debate",
            "voting",
        ]
        super().__init__(**kwargs)

        self._override_threshold = override_threshold
        self._proposals: list[Policy] = []
        self._enacted_policies: list[Policy] = []

    async def initialize(self) -> None:
        """Initialize legislative agent."""
        self._state = AgentState.READY
        logger.info("Legislative agent initialized")

    async def work(self) -> dict[str, Any]:
        """Process policy proposals through debate and voting."""
        result: dict[str, Any] = {
            "branch": "legislative",
            "proposals_reviewed": [],
            "policies_passed": [],
            "policies_rejected": [],
        }

        # Process proposals in debate
        for proposal in [p for p in self._proposals if p.status == PolicyStatus.DEBATING]:
            # Simulate voting
            vote_result = await self._vote_on_policy(proposal)
            result["proposals_reviewed"].append(
                {
                    "policy_id": proposal.policy_id,
                    "result": vote_result,
                }
            )

            if vote_result == "passed":
                result["policies_passed"].append(proposal.policy_id)
            else:
                result["policies_rejected"].append(proposal.policy_id)

        return result

    async def shutdown(self) -> None:
        """Shutdown legislative agent."""
        logger.info("Legislative agent shutdown")

    async def propose_policy(
        self,
        title: str,
        description: str,
    ) -> Policy:
        """Draft a new policy proposal.

        Args:
            title: Policy title.
            description: Policy description.

        Returns:
            The created Policy.
        """
        import uuid

        policy = Policy(
            policy_id=f"POL-{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            proposed_by=self.agent_id,
            status=PolicyStatus.PROPOSED,
        )

        self._proposals.append(policy)
        logger.info(f"Policy proposed: {policy.policy_id} - {title}")

        return policy

    async def begin_debate(self, policy: Policy) -> bool:
        """Begin debate on a policy.

        Args:
            policy: Policy to debate.

        Returns:
            True if debate started.
        """
        if policy.status != PolicyStatus.PROPOSED:
            return False

        policy.status = PolicyStatus.DEBATING
        return True

    async def _vote_on_policy(self, policy: Policy) -> str:
        """Conduct voting on a policy.

        Args:
            policy: Policy to vote on.

        Returns:
            "passed" or "rejected".
        """
        # Simulate votes based on policy content
        # In practice, this would involve peer agents
        import random

        votes_for = random.randint(3, 7)
        votes_against = random.randint(1, 5)

        policy.votes_for = votes_for
        policy.votes_against = votes_against

        if votes_for > votes_against:
            policy.status = PolicyStatus.PASSED
            return "passed"
        else:
            policy.status = PolicyStatus.REJECTED
            return "rejected"

    async def override_veto(self, policy: Policy) -> bool:
        """Attempt to override an executive veto.

        Args:
            policy: Vetoed policy.

        Returns:
            True if override successful.
        """
        if policy.status != PolicyStatus.VETOED:
            return False

        # Check if enough votes for override
        total_votes = policy.votes_for + policy.votes_against
        if total_votes == 0:
            return False

        override_support = policy.votes_for / total_votes

        if override_support >= self._override_threshold:
            policy.status = PolicyStatus.ENACTED
            policy.enacted_at = datetime.now(UTC)
            self._enacted_policies.append(policy)
            logger.info(f"Veto override successful: {policy.policy_id}")
            return True

        return False

    async def enact_policy(self, policy: Policy) -> bool:
        """Enact a passed policy (if not vetoed).

        Args:
            policy: Policy to enact.

        Returns:
            True if enacted.
        """
        if policy.status != PolicyStatus.PASSED:
            return False

        policy.status = PolicyStatus.ENACTED
        policy.enacted_at = datetime.now(UTC)
        self._enacted_policies.append(policy)

        logger.info(f"Policy enacted: {policy.policy_id}")
        return True

    async def check_peer(self, branch: str, action: dict[str, Any]) -> bool:
        """Check peer branch action."""
        if branch == "executive":
            # Can override vetoes
            return True
        return True

    def get_pending_proposals(self) -> list[Policy]:
        """Get proposals pending debate or voting."""
        return [
            p for p in self._proposals if p.status in [PolicyStatus.PROPOSED, PolicyStatus.DEBATING]
        ]


class JudicialAgent(GovernmentAgent):
    """
    Judicial branch agent - reviews for compliance and resolves disputes.

    Responsibilities:
    - Review policies for compliance
    - Review executive orders
    - Resolve disputes between agents
    - Interpret rules and policies
    - Can invalidate non-compliant actions
    """

    branch = "judicial"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize judicial agent."""
        kwargs.setdefault("name", "judicial")
        kwargs["capabilities"] = kwargs.get("capabilities", []) + [
            "compliance_review",
            "dispute_resolution",
            "rule_interpretation",
        ]
        super().__init__(**kwargs)

        self._pending_reviews: list[dict[str, Any]] = []
        self._disputes: list[Dispute] = []
        self._rulings: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize judicial agent."""
        self._state = AgentState.READY
        logger.info("Judicial agent initialized")

    async def work(self) -> dict[str, Any]:
        """Process reviews and disputes."""
        result: dict[str, Any] = {
            "branch": "judicial",
            "reviews_completed": [],
            "disputes_resolved": [],
            "rulings": [],
        }

        # Process reviews
        for review in self._pending_reviews[:5]:
            review_result = await self._conduct_review(review)
            result["reviews_completed"].append(review_result)

        # Process disputes
        for dispute in [d for d in self._disputes if d.status == DisputeStatus.HEARING]:
            ruling = await self._rule_on_dispute(dispute)
            result["disputes_resolved"].append(
                {
                    "dispute_id": dispute.dispute_id,
                    "ruling": ruling,
                }
            )
            result["rulings"].append(ruling)

        return result

    async def shutdown(self) -> None:
        """Shutdown judicial agent."""
        logger.info("Judicial agent shutdown")

    async def submit_for_review(
        self,
        item_type: str,
        item: Any,
        requester: str,
    ) -> str:
        """Submit an item for judicial review.

        Args:
            item_type: Type of item (policy, order, action).
            item: The item to review.
            requester: Who requested the review.

        Returns:
            Review ID.
        """
        import uuid

        review_id = f"REV-{uuid.uuid4().hex[:8]}"

        self._pending_reviews.append(
            {
                "review_id": review_id,
                "item_type": item_type,
                "item": item,
                "requester": requester,
                "submitted_at": datetime.now(UTC).isoformat(),
            }
        )

        return review_id

    async def _conduct_review(self, review: dict[str, Any]) -> dict[str, Any]:
        """Conduct a compliance review.

        Args:
            review: Review request.

        Returns:
            Review result.
        """
        item = review["item"]
        item_type = review["item_type"]

        result = {
            "review_id": review["review_id"],
            "item_type": item_type,
            "compliant": True,
            "issues": [],
            "ruling": "",
        }

        # Check based on type
        if item_type == "policy" and isinstance(item, Policy):
            # Check policy compliance
            if not item.title or not item.description:
                result["compliant"] = False
                result["issues"].append("Policy must have title and description")

        elif item_type == "order" and isinstance(item, ExecutiveOrder):
            # Check order compliance
            if not item.directive:
                result["compliant"] = False
                result["issues"].append("Order must have directive")

        if not result["compliant"]:
            result["ruling"] = "INVALIDATED - compliance issues found"
            if item_type == "policy" and isinstance(item, Policy):
                item.status = PolicyStatus.INVALIDATED

        else:
            result["ruling"] = "UPHELD - compliant"

        # Remove from pending
        self._pending_reviews.remove(review)
        self._rulings.append(result)

        return result

    async def file_dispute(
        self,
        title: str,
        description: str,
        filed_by: str,
        against: str,
    ) -> Dispute:
        """File a dispute for resolution.

        Args:
            title: Dispute title.
            description: Description of the dispute.
            filed_by: Agent filing the dispute.
            against: Agent or action being disputed.

        Returns:
            The created Dispute.
        """
        import uuid

        dispute = Dispute(
            dispute_id=f"DIS-{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            filed_by=filed_by,
            against=against,
        )

        self._disputes.append(dispute)
        logger.info(f"Dispute filed: {dispute.dispute_id} - {title}")

        return dispute

    async def begin_hearing(self, dispute: Dispute) -> bool:
        """Begin hearing on a dispute.

        Args:
            dispute: Dispute to hear.

        Returns:
            True if hearing started.
        """
        if dispute.status != DisputeStatus.FILED:
            return False

        dispute.status = DisputeStatus.HEARING
        return True

    async def _rule_on_dispute(self, dispute: Dispute) -> str:
        """Rule on a dispute.

        Args:
            dispute: Dispute to rule on.

        Returns:
            The ruling.
        """
        # In practice, this would involve analysis
        # Simplified: rule based on description content
        ruling = f"After review of '{dispute.title}', "

        if "violation" in dispute.description.lower():
            ruling += "the court finds in favor of the complainant."
        else:
            ruling += "the court finds insufficient grounds for relief."

        dispute.ruling = ruling
        dispute.ruled_at = datetime.now(UTC)
        dispute.status = DisputeStatus.DECIDED

        return ruling

    async def check_peer(self, branch: str, action: dict[str, Any]) -> bool:
        """Check peer branch action (judicial reviews all)."""
        # Judicial can review any action
        return True

    def get_pending_disputes(self) -> list[Dispute]:
        """Get disputes awaiting hearing."""
        return [
            d for d in self._disputes if d.status in [DisputeStatus.FILED, DisputeStatus.REVIEWING]
        ]

    def get_rulings(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent rulings."""
        return self._rulings[-limit:]
