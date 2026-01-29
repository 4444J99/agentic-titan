"""
DAO Agent - Decentralized Autonomous Organization patterns.

Implements DAO governance mechanisms:
- Proposal/vote mechanics
- Smart contract-like rule execution
- Token-weighted governance
- Iron Law of Oligarchy detection

DAOs replace traditional hierarchy with code-based rules,
but must guard against informal power concentration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentState

logger = logging.getLogger("titan.agents.dao")


class ProposalStatus(str, Enum):
    """Status of a DAO proposal."""

    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"
    VETOED = "vetoed"


class VoteChoice(str, Enum):
    """Voting choices."""

    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"


class ProposalType(str, Enum):
    """Types of proposals."""

    GOVERNANCE = "governance"      # Change rules/parameters
    TREASURY = "treasury"          # Spend resources
    MEMBERSHIP = "membership"      # Add/remove members
    EXECUTION = "execution"        # Execute an action
    CONSTITUTIONAL = "constitutional"  # Major rule changes


@dataclass
class Vote:
    """A single vote on a proposal."""

    voter_id: str
    choice: VoteChoice
    weight: float = 1.0  # Token/voting power
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rationale: str = ""


@dataclass
class Proposal:
    """A governance proposal."""

    proposal_id: str
    proposal_type: ProposalType
    title: str
    description: str
    proposer: str
    status: ProposalStatus = ProposalStatus.DRAFT
    votes: list[Vote] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    execution_payload: dict[str, Any] = field(default_factory=dict)
    quorum_threshold: float = 0.5
    approval_threshold: float = 0.5

    @property
    def total_for(self) -> float:
        """Total voting weight for the proposal."""
        return sum(v.weight for v in self.votes if v.choice == VoteChoice.FOR)

    @property
    def total_against(self) -> float:
        """Total voting weight against the proposal."""
        return sum(v.weight for v in self.votes if v.choice == VoteChoice.AGAINST)

    @property
    def total_votes(self) -> float:
        """Total voting weight cast (excluding abstentions)."""
        return self.total_for + self.total_against

    @property
    def approval_ratio(self) -> float:
        """Approval ratio (for votes / total non-abstention votes)."""
        if self.total_votes == 0:
            return 0.0
        return self.total_for / self.total_votes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "title": self.title,
            "description": self.description,
            "proposer": self.proposer,
            "status": self.status.value,
            "votes_for": self.total_for,
            "votes_against": self.total_against,
            "approval_ratio": self.approval_ratio,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Member:
    """A DAO member."""

    member_id: str
    voting_power: float = 1.0
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    proposals_created: int = 0
    votes_cast: int = 0
    delegated_to: str | None = None  # Voting power delegation
    delegations_received: list[str] = field(default_factory=list)
    reputation: float = 1.0

    @property
    def effective_voting_power(self) -> float:
        """Voting power including delegations received."""
        # Base power plus delegations (simplified)
        return self.voting_power * (1 + len(self.delegations_received) * 0.1)


@dataclass
class OligarchyIndicators:
    """Indicators of oligarchy formation."""

    power_concentration: float = 0.0  # Gini coefficient
    proposal_concentration: float = 0.0  # How concentrated proposal creation is
    voting_participation: float = 0.0  # Active voter ratio
    delegation_centralization: float = 0.0  # Delegation concentration
    whale_influence: float = 0.0  # Top holders' voting share

    @property
    def oligarchy_risk(self) -> str:
        """Overall oligarchy risk level."""
        score = (
            self.power_concentration * 0.3 +
            self.proposal_concentration * 0.2 +
            (1 - self.voting_participation) * 0.2 +
            self.delegation_centralization * 0.15 +
            self.whale_influence * 0.15
        )

        if score > 0.7:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class DAOConfig:
    """Configuration for DAO behavior."""

    default_quorum: float = 0.5
    default_approval: float = 0.5
    proposal_duration_hours: int = 168  # 1 week
    min_voting_power_to_propose: float = 0.01
    constitutional_threshold: float = 0.67
    enable_delegation: bool = True
    max_delegation_depth: int = 1


class DAOAgent(BaseAgent):
    """
    Agent implementing DAO governance patterns.

    Implements decentralized governance with:
    - Proposal creation and voting
    - Token-weighted voting power
    - Smart contract-like rule execution
    - Delegation support
    - Oligarchy detection

    Capabilities:
    - Proposal management
    - Voting
    - Rule execution
    - Governance monitoring
    - Oligarchy detection
    """

    def __init__(
        self,
        dao_name: str = "default_dao",
        config: DAOConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DAO agent.

        Args:
            dao_name: Name of the DAO.
            config: DAO configuration.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", f"dao_{dao_name}")
        kwargs.setdefault("capabilities", [
            "proposal_management",
            "voting",
            "governance",
            "rule_execution",
        ])
        super().__init__(**kwargs)

        self._dao_name = dao_name
        self._config = config or DAOConfig()
        self._members: dict[str, Member] = {}
        self._proposals: dict[str, Proposal] = {}
        self._executed_proposals: list[str] = []
        self._treasury: float = 0.0
        self._rules: dict[str, Any] = {}

    @property
    def dao_name(self) -> str:
        """Get DAO name."""
        return self._dao_name

    @property
    def member_count(self) -> int:
        """Get number of members."""
        return len(self._members)

    async def initialize(self) -> None:
        """Initialize DAO agent."""
        self._state = AgentState.READY

        # Register self as founding member
        self._members[self.agent_id] = Member(
            member_id=self.agent_id,
            voting_power=1.0,
        )

        logger.info(f"DAO agent initialized (dao={self._dao_name})")

    async def work(self) -> dict[str, Any]:
        """Perform DAO governance cycle."""
        result = {
            "dao": self._dao_name,
            "members": self.member_count,
            "active_proposals": len(self._get_active_proposals()),
            "actions": [],
        }

        # Process expired proposals
        expired = await self._process_expired_proposals()
        if expired:
            result["actions"].append(f"expired_{len(expired)}_proposals")

        # Execute passed proposals
        executed = await self._execute_passed_proposals()
        if executed:
            result["actions"].append(f"executed_{len(executed)}_proposals")

        # Check for oligarchy
        indicators = self.detect_oligarchy()
        result["oligarchy_risk"] = indicators.oligarchy_risk
        if indicators.oligarchy_risk == "HIGH":
            result["actions"].append("oligarchy_warning")

        return result

    async def shutdown(self) -> None:
        """Shutdown DAO agent."""
        logger.info(
            f"DAO agent shutdown "
            f"(members={self.member_count}, proposals={len(self._proposals)})"
        )

    # =========================================================================
    # Member Management
    # =========================================================================

    def add_member(
        self,
        member_id: str,
        voting_power: float = 1.0,
    ) -> Member:
        """Add a member to the DAO.

        Args:
            member_id: Member identifier.
            voting_power: Initial voting power.

        Returns:
            The created Member.
        """
        member = Member(
            member_id=member_id,
            voting_power=voting_power,
        )
        self._members[member_id] = member

        logger.info(f"Member added to {self._dao_name}: {member_id}")
        return member

    def remove_member(self, member_id: str) -> bool:
        """Remove a member from the DAO.

        Args:
            member_id: Member to remove.

        Returns:
            True if removed.
        """
        if member_id not in self._members:
            return False

        # Handle delegations
        member = self._members[member_id]
        for delegator_id in member.delegations_received:
            if delegator_id in self._members:
                self._members[delegator_id].delegated_to = None

        if member.delegated_to and member.delegated_to in self._members:
            delegate = self._members[member.delegated_to]
            if member_id in delegate.delegations_received:
                delegate.delegations_received.remove(member_id)

        del self._members[member_id]
        return True

    def delegate_vote(self, from_member: str, to_member: str) -> bool:
        """Delegate voting power to another member.

        Args:
            from_member: Member delegating power.
            to_member: Member receiving delegation.

        Returns:
            True if delegation successful.
        """
        if not self._config.enable_delegation:
            return False

        if from_member not in self._members or to_member not in self._members:
            return False

        # Check delegation depth
        delegate = self._members[to_member]
        if delegate.delegated_to:
            # Already delegated - check depth
            depth = 1
            current = delegate.delegated_to
            while current and depth < self._config.max_delegation_depth:
                if current in self._members and self._members[current].delegated_to:
                    current = self._members[current].delegated_to
                    depth += 1
                else:
                    break

            if depth >= self._config.max_delegation_depth:
                return False

        # Perform delegation
        self._members[from_member].delegated_to = to_member
        self._members[to_member].delegations_received.append(from_member)

        return True

    def get_member(self, member_id: str) -> Member | None:
        """Get a member by ID."""
        return self._members.get(member_id)

    def get_total_voting_power(self) -> float:
        """Get total voting power in the DAO."""
        return sum(m.voting_power for m in self._members.values())

    # =========================================================================
    # Proposal Management
    # =========================================================================

    def create_proposal(
        self,
        proposer: str,
        title: str,
        description: str,
        proposal_type: ProposalType = ProposalType.GOVERNANCE,
        execution_payload: dict[str, Any] | None = None,
        custom_quorum: float | None = None,
        custom_approval: float | None = None,
    ) -> Proposal | None:
        """Create a new proposal.

        Args:
            proposer: Member creating the proposal.
            title: Proposal title.
            description: Proposal description.
            proposal_type: Type of proposal.
            execution_payload: Data for execution.
            custom_quorum: Override default quorum.
            custom_approval: Override default approval threshold.

        Returns:
            Created Proposal or None if proposer ineligible.
        """
        member = self._members.get(proposer)
        if not member:
            return None

        # Check minimum voting power
        if member.voting_power < self._config.min_voting_power_to_propose:
            return None

        import uuid
        from datetime import timedelta

        quorum = custom_quorum or (
            self._config.constitutional_threshold
            if proposal_type == ProposalType.CONSTITUTIONAL
            else self._config.default_quorum
        )
        approval = custom_approval or (
            self._config.constitutional_threshold
            if proposal_type == ProposalType.CONSTITUTIONAL
            else self._config.default_approval
        )

        proposal = Proposal(
            proposal_id=f"PROP-{uuid.uuid4().hex[:8]}",
            proposal_type=proposal_type,
            title=title,
            description=description,
            proposer=proposer,
            status=ProposalStatus.ACTIVE,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self._config.proposal_duration_hours),
            execution_payload=execution_payload or {},
            quorum_threshold=quorum,
            approval_threshold=approval,
        )

        self._proposals[proposal.proposal_id] = proposal
        member.proposals_created += 1

        logger.info(f"Proposal created: {proposal.proposal_id} - {title}")
        return proposal

    def vote(
        self,
        voter_id: str,
        proposal_id: str,
        choice: VoteChoice,
        rationale: str = "",
    ) -> Vote | None:
        """Cast a vote on a proposal.

        Args:
            voter_id: Member voting.
            proposal_id: Proposal to vote on.
            choice: Vote choice.
            rationale: Reason for vote.

        Returns:
            Created Vote or None if invalid.
        """
        member = self._members.get(voter_id)
        proposal = self._proposals.get(proposal_id)

        if not member or not proposal:
            return None

        if proposal.status != ProposalStatus.ACTIVE:
            return None

        # Check if already voted
        existing_vote = next(
            (v for v in proposal.votes if v.voter_id == voter_id),
            None
        )
        if existing_vote:
            return None  # Already voted

        # Get effective voting power (including delegations)
        voting_power = member.effective_voting_power

        vote = Vote(
            voter_id=voter_id,
            choice=choice,
            weight=voting_power,
            rationale=rationale,
        )

        proposal.votes.append(vote)
        member.votes_cast += 1

        # Check if proposal can be resolved
        self._check_proposal_resolution(proposal)

        return vote

    def _check_proposal_resolution(self, proposal: Proposal) -> None:
        """Check if proposal can be resolved (passed/rejected).

        Args:
            proposal: Proposal to check.
        """
        total_power = self.get_total_voting_power()

        # Check quorum
        participation = proposal.total_votes / total_power if total_power > 0 else 0
        if participation < proposal.quorum_threshold:
            return  # Not enough participation yet

        # Check approval
        if proposal.approval_ratio >= proposal.approval_threshold:
            proposal.status = ProposalStatus.PASSED
            logger.info(f"Proposal passed: {proposal.proposal_id}")
        elif (1 - proposal.approval_ratio) >= proposal.approval_threshold:
            # Enough votes against
            proposal.status = ProposalStatus.REJECTED
            logger.info(f"Proposal rejected: {proposal.proposal_id}")

    def _get_active_proposals(self) -> list[Proposal]:
        """Get all active proposals."""
        return [
            p for p in self._proposals.values()
            if p.status == ProposalStatus.ACTIVE
        ]

    async def _process_expired_proposals(self) -> list[str]:
        """Process proposals that have expired.

        Returns:
            List of expired proposal IDs.
        """
        now = datetime.now(timezone.utc)
        expired: list[str] = []

        for proposal in self._get_active_proposals():
            if proposal.expires_at and now > proposal.expires_at:
                total_power = self.get_total_voting_power()
                participation = proposal.total_votes / total_power if total_power > 0 else 0

                if participation < proposal.quorum_threshold:
                    proposal.status = ProposalStatus.EXPIRED
                elif proposal.approval_ratio >= proposal.approval_threshold:
                    proposal.status = ProposalStatus.PASSED
                else:
                    proposal.status = ProposalStatus.REJECTED

                expired.append(proposal.proposal_id)

        return expired

    async def _execute_passed_proposals(self) -> list[str]:
        """Execute proposals that have passed.

        Returns:
            List of executed proposal IDs.
        """
        executed: list[str] = []

        for proposal in self._proposals.values():
            if proposal.status == ProposalStatus.PASSED:
                if proposal.proposal_id not in self._executed_proposals:
                    success = await self._execute_proposal(proposal)
                    if success:
                        proposal.status = ProposalStatus.EXECUTED
                        self._executed_proposals.append(proposal.proposal_id)
                        executed.append(proposal.proposal_id)

        return executed

    async def _execute_proposal(self, proposal: Proposal) -> bool:
        """Execute a passed proposal.

        Args:
            proposal: Proposal to execute.

        Returns:
            True if executed successfully.
        """
        payload = proposal.execution_payload

        if proposal.proposal_type == ProposalType.TREASURY:
            # Execute treasury action
            amount = payload.get("amount", 0)
            recipient = payload.get("recipient")
            if recipient and amount <= self._treasury:
                self._treasury -= amount
                logger.info(f"Treasury: transferred {amount} to {recipient}")
                return True

        elif proposal.proposal_type == ProposalType.MEMBERSHIP:
            # Execute membership action
            action = payload.get("action")
            target = payload.get("target")
            if action == "add" and target:
                self.add_member(target, payload.get("voting_power", 1.0))
                return True
            elif action == "remove" and target:
                return self.remove_member(target)

        elif proposal.proposal_type == ProposalType.GOVERNANCE:
            # Update governance parameters
            if "rules" in payload:
                self._rules.update(payload["rules"])
                return True

        # Default: mark as executed
        return True

    # =========================================================================
    # Oligarchy Detection
    # =========================================================================

    def detect_oligarchy(self) -> OligarchyIndicators:
        """Detect signs of oligarchy formation.

        The Iron Law of Oligarchy suggests that even democratic
        organizations tend toward oligarchy. This detects indicators.

        Returns:
            OligarchyIndicators with risk metrics.
        """
        indicators = OligarchyIndicators()

        if not self._members:
            return indicators

        # Power concentration (Gini-like coefficient)
        powers = sorted([m.voting_power for m in self._members.values()])
        n = len(powers)
        if n > 1:
            total = sum(powers)
            if total > 0:
                cumulative = 0
                area = 0
                for i, power in enumerate(powers):
                    cumulative += power
                    area += cumulative
                gini = (n + 1 - 2 * area / total) / n
                indicators.power_concentration = max(0, min(1, gini))

        # Proposal concentration
        if self._proposals:
            proposers = [p.proposer for p in self._proposals.values()]
            unique_proposers = len(set(proposers))
            indicators.proposal_concentration = 1 - (unique_proposers / len(self._members))

        # Voting participation
        total_possible_votes = len(self._members) * len(self._proposals) if self._proposals else 1
        actual_votes = sum(m.votes_cast for m in self._members.values())
        indicators.voting_participation = actual_votes / total_possible_votes if total_possible_votes > 0 else 0

        # Delegation centralization
        delegates_with_delegations = sum(
            1 for m in self._members.values()
            if m.delegations_received
        )
        if len(self._members) > 1:
            indicators.delegation_centralization = delegates_with_delegations / len(self._members)

        # Whale influence (top 10% voting power share)
        if len(powers) >= 10:
            top_10_percent = powers[-len(powers)//10:]
            indicators.whale_influence = sum(top_10_percent) / sum(powers) if sum(powers) > 0 else 0
        else:
            indicators.whale_influence = powers[-1] / sum(powers) if sum(powers) > 0 else 0

        return indicators

    # =========================================================================
    # Treasury Management
    # =========================================================================

    def deposit_to_treasury(self, amount: float) -> float:
        """Deposit funds to the treasury.

        Args:
            amount: Amount to deposit.

        Returns:
            New treasury balance.
        """
        self._treasury += amount
        return self._treasury

    def get_treasury_balance(self) -> float:
        """Get current treasury balance."""
        return self._treasury

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_governance_summary(self) -> dict[str, Any]:
        """Get summary of DAO governance state.

        Returns:
            Dictionary with governance summary.
        """
        active = self._get_active_proposals()
        passed = [p for p in self._proposals.values() if p.status == ProposalStatus.PASSED]
        rejected = [p for p in self._proposals.values() if p.status == ProposalStatus.REJECTED]

        indicators = self.detect_oligarchy()

        return {
            "dao_name": self._dao_name,
            "members": self.member_count,
            "total_voting_power": self.get_total_voting_power(),
            "treasury": self._treasury,
            "proposals": {
                "total": len(self._proposals),
                "active": len(active),
                "passed": len(passed),
                "rejected": len(rejected),
                "executed": len(self._executed_proposals),
            },
            "oligarchy": {
                "risk": indicators.oligarchy_risk,
                "power_concentration": indicators.power_concentration,
                "voting_participation": indicators.voting_participation,
            },
        }
