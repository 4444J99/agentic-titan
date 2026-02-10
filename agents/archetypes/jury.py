"""
Jury Agent - Deliberative adjudication body.

Implements jury-style deliberation for reaching collective decisions
through evidence evaluation and peer discussion.

Capabilities:
- Evidence evaluation
- Deliberation with peers
- Verdict reaching through consensus
- Foreperson coordination
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agents.framework.base_agent import AgentState, BaseAgent

logger = logging.getLogger("titan.agents.jury")


class VerdictType(StrEnum):
    """Types of verdicts a jury can reach."""

    APPROVED = "approved"
    REJECTED = "rejected"
    HUNG = "hung"  # Cannot reach consensus
    MISTRIAL = "mistrial"  # Process failed
    PENDING = "pending"


class DeliberationPhase(StrEnum):
    """Phases of jury deliberation."""

    EVIDENCE_PRESENTATION = "evidence_presentation"
    INITIAL_DISCUSSION = "initial_discussion"
    DETAILED_DELIBERATION = "detailed_deliberation"
    VOTING = "voting"
    VERDICT = "verdict"


@dataclass
class Evidence:
    """A piece of evidence for jury consideration."""

    evidence_id: str
    description: str
    source: str
    weight: float = 1.0  # Importance weight (0-2)
    supporting: bool = True  # True = supports approval
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JurorVote:
    """A single juror's vote."""

    juror_id: str
    vote: VerdictType
    confidence: float = 1.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DeliberationState:
    """Current state of jury deliberation."""

    phase: DeliberationPhase = DeliberationPhase.EVIDENCE_PRESENTATION
    evidence: list[Evidence] = field(default_factory=list)
    discussion_points: list[dict[str, Any]] = field(default_factory=list)
    votes: list[JurorVote] = field(default_factory=list)
    verdict: VerdictType = VerdictType.PENDING
    verdict_reasoning: str = ""
    rounds_completed: int = 0
    max_rounds: int = 3


@dataclass
class JuryConfiguration:
    """Configuration for jury behavior."""

    required_unanimity: bool = False  # True = unanimous required
    quorum_threshold: float = 0.67  # Fraction needed for verdict
    min_deliberation_rounds: int = 1
    max_deliberation_rounds: int = 5
    evidence_weight_threshold: float = 0.5  # Min weight to consider
    allow_abstention: bool = False


class JuryAgent(BaseAgent):
    """
    Agent specialized in deliberative adjudication.

    Implements jury-style decision making where evidence is presented,
    deliberated upon with peers, and a verdict is reached through
    voting and consensus building.

    Can act as:
    - Individual juror (voting member)
    - Foreperson (coordinating deliberation)
    """

    def __init__(
        self,
        is_foreperson: bool = False,
        config: JuryConfiguration | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the jury agent.

        Args:
            is_foreperson: Whether this agent coordinates deliberation.
            config: Jury behavior configuration.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", "jury")
        kwargs.setdefault(
            "capabilities",
            [
                "evidence_evaluation",
                "deliberation",
                "voting",
                "consensus_building",
            ],
        )
        super().__init__(**kwargs)

        self._is_foreperson = is_foreperson
        self._config = config or JuryConfiguration()
        self._deliberation = DeliberationState(max_rounds=self._config.max_deliberation_rounds)
        self._peer_jurors: list[str] = []
        self._case_description: str = ""

    @property
    def is_foreperson(self) -> bool:
        """Whether this agent is the foreperson."""
        return self._is_foreperson

    @property
    def deliberation_state(self) -> DeliberationState:
        """Get current deliberation state."""
        return self._deliberation

    async def initialize(self) -> None:
        """Initialize the jury agent."""
        self._state = AgentState.READY
        logger.info(f"Jury agent initialized (foreperson={self._is_foreperson})")

    async def work(self) -> dict[str, Any]:
        """Main jury work loop."""
        if self._is_foreperson:
            return await self._foreperson_work()
        else:
            return await self._juror_work()

    async def _foreperson_work(self) -> dict[str, Any]:
        """Foreperson coordinates the deliberation process."""
        result: dict[str, Any] = {
            "role": "foreperson",
            "verdict": VerdictType.PENDING.value,
            "phases_completed": [],
        }

        # Phase 1: Present evidence
        await self._present_evidence()
        result["phases_completed"].append("evidence_presentation")

        # Phase 2-3: Deliberation rounds
        for round_num in range(self._config.max_deliberation_rounds):
            self._deliberation.phase = DeliberationPhase.DETAILED_DELIBERATION
            await self._facilitate_discussion()

            # Check if ready for voting
            if await self._ready_for_vote():
                break

            self._deliberation.rounds_completed += 1

        result["phases_completed"].append("deliberation")

        # Phase 4: Voting
        self._deliberation.phase = DeliberationPhase.VOTING
        await self._collect_votes()
        result["phases_completed"].append("voting")

        # Phase 5: Determine verdict
        self._deliberation.phase = DeliberationPhase.VERDICT
        verdict = await self._determine_verdict()
        result["verdict"] = verdict.value
        result["verdict_reasoning"] = self._deliberation.verdict_reasoning
        result["votes"] = [
            {"juror": v.juror_id, "vote": v.vote.value, "reasoning": v.reasoning}
            for v in self._deliberation.votes
        ]

        return result

    async def _juror_work(self) -> dict[str, Any]:
        """Individual juror participates in deliberation."""
        result: dict[str, Any] = {
            "role": "juror",
            "voted": False,
            "contributions": [],
        }

        # Evaluate evidence
        evaluation = await self._evaluate_evidence()
        result["evidence_evaluation"] = evaluation

        # Participate in discussion
        contribution = await self._contribute_to_discussion()
        if contribution:
            result["contributions"].append(contribution)

        # Cast vote when called
        vote = await self._cast_vote()
        if vote:
            result["voted"] = True
            result["vote"] = vote.vote.value
            result["vote_reasoning"] = vote.reasoning

        return result

    async def shutdown(self) -> None:
        """Shutdown the jury agent."""
        logger.info(f"Jury agent shutdown (verdict={self._deliberation.verdict.value})")

    # =========================================================================
    # Evidence Management
    # =========================================================================

    async def submit_evidence(self, evidence: Evidence) -> None:
        """Submit evidence for jury consideration.

        Args:
            evidence: The evidence to submit.
        """
        if evidence.weight >= self._config.evidence_weight_threshold:
            self._deliberation.evidence.append(evidence)
            logger.debug(f"Evidence submitted: {evidence.evidence_id}")

    async def _present_evidence(self) -> None:
        """Present all evidence to the jury."""
        self._deliberation.phase = DeliberationPhase.EVIDENCE_PRESENTATION

        # Sort by weight
        self._deliberation.evidence.sort(key=lambda e: e.weight, reverse=True)

        # Broadcast evidence summary to peer jurors
        if self._hive_mind and self._peer_jurors:
            summary = self._format_evidence_summary()
            await self._hive_mind.broadcast(
                source_agent_id=self.agent_id,
                message={"content": summary, "importance": 0.9},
                topic="jury_evidence",
            )

    def _format_evidence_summary(self) -> str:
        """Format evidence for presentation."""
        lines = [f"CASE: {self._case_description}\n", "EVIDENCE:"]

        for i, ev in enumerate(self._deliberation.evidence, 1):
            stance = "SUPPORTS" if ev.supporting else "OPPOSES"
            lines.append(f"  {i}. [{stance}] {ev.description} (weight: {ev.weight:.1f})")

        return "\n".join(lines)

    async def _evaluate_evidence(self) -> dict[str, Any]:
        """Evaluate evidence and form initial opinion."""
        supporting_weight = sum(e.weight for e in self._deliberation.evidence if e.supporting)
        opposing_weight = sum(e.weight for e in self._deliberation.evidence if not e.supporting)
        total = supporting_weight + opposing_weight

        if total == 0:
            lean = 0.5
        else:
            lean = supporting_weight / total

        return {
            "supporting_weight": supporting_weight,
            "opposing_weight": opposing_weight,
            "initial_lean": "approve" if lean > 0.5 else "reject",
            "confidence": abs(lean - 0.5) * 2,
        }

    # =========================================================================
    # Deliberation
    # =========================================================================

    async def _facilitate_discussion(self) -> None:
        """Facilitate a round of deliberation (foreperson only)."""
        # Request input from all jurors
        if self._hive_mind and self._peer_jurors:
            await self._hive_mind.broadcast(
                source_agent_id=self.agent_id,
                message={
                    "content": (
                        f"Deliberation round {self._deliberation.rounds_completed + 1}: "
                        "Please share your thoughts on the evidence."
                    ),
                    "importance": 0.8,
                },
                topic="jury_discussion",
            )

            # Wait for responses
            await asyncio.sleep(0.1)

    async def _contribute_to_discussion(self) -> dict[str, Any] | None:
        """Contribute to jury discussion."""
        evaluation = await self._evaluate_evidence()

        contribution: dict[str, Any] = {
            "juror_id": self.agent_id,
            "round": self._deliberation.rounds_completed,
            "position": evaluation["initial_lean"],
            "key_points": [],
        }

        # Identify key evidence points
        strong_evidence = [e for e in self._deliberation.evidence if e.weight >= 1.5]
        for ev in strong_evidence[:3]:
            contribution["key_points"].append(ev.description)

        self._deliberation.discussion_points.append(contribution)

        return contribution

    async def _ready_for_vote(self) -> bool:
        """Check if jury is ready to vote."""
        # Minimum deliberation rounds
        if self._deliberation.rounds_completed < self._config.min_deliberation_rounds:
            return False

        # Check if positions are converging
        if len(self._deliberation.discussion_points) >= len(self._peer_jurors) + 1:
            positions = [
                str(d["position"])
                for d in self._deliberation.discussion_points[-len(self._peer_jurors) :]
            ]
            if len(set(positions)) == 1:
                return True  # All agree

        return self._deliberation.rounds_completed >= self._config.max_deliberation_rounds - 1

    # =========================================================================
    # Voting
    # =========================================================================

    async def _collect_votes(self) -> None:
        """Collect votes from all jurors (foreperson only)."""
        # Request votes
        if self._hive_mind and self._peer_jurors:
            await self._hive_mind.broadcast(
                source_agent_id=self.agent_id,
                message={
                    "content": "Please cast your vote: APPROVED or REJECTED",
                    "importance": 0.95,
                },
                topic="jury_vote",
            )

        # Foreperson also votes
        own_vote = await self._cast_vote()
        if own_vote:
            self._deliberation.votes.append(own_vote)

    async def _cast_vote(self) -> JurorVote | None:
        """Cast a vote based on evidence evaluation."""
        evaluation = await self._evaluate_evidence()

        if evaluation["confidence"] < 0.1 and self._config.allow_abstention:
            return None  # Abstain

        vote_type = (
            VerdictType.APPROVED
            if evaluation["initial_lean"] == "approve"
            else VerdictType.REJECTED
        )

        # Build reasoning
        supporting = [e.description for e in self._deliberation.evidence if e.supporting]
        opposing = [e.description for e in self._deliberation.evidence if not e.supporting]

        if vote_type == VerdictType.APPROVED:
            reasoning = (
                f"Supporting evidence ({len(supporting)} items) "
                f"outweighs opposing ({len(opposing)} items)"
            )
        else:
            reasoning = (
                f"Opposing evidence ({len(opposing)} items) "
                f"outweighs supporting ({len(supporting)} items)"
            )

        vote = JurorVote(
            juror_id=self.agent_id,
            vote=vote_type,
            confidence=evaluation["confidence"],
            reasoning=reasoning,
        )

        return vote

    async def _determine_verdict(self) -> VerdictType:
        """Determine final verdict from votes."""
        if not self._deliberation.votes:
            self._deliberation.verdict = VerdictType.MISTRIAL
            self._deliberation.verdict_reasoning = "No votes cast"
            return VerdictType.MISTRIAL

        # Count votes
        approve_count = sum(1 for v in self._deliberation.votes if v.vote == VerdictType.APPROVED)
        reject_count = sum(1 for v in self._deliberation.votes if v.vote == VerdictType.REJECTED)
        total_votes = len(self._deliberation.votes)

        # Check for unanimity if required
        if self._config.required_unanimity:
            if approve_count == total_votes:
                verdict = VerdictType.APPROVED
            elif reject_count == total_votes:
                verdict = VerdictType.REJECTED
            else:
                verdict = VerdictType.HUNG
        else:
            # Quorum-based decision
            approval_fraction = approve_count / total_votes
            if approval_fraction >= self._config.quorum_threshold:
                verdict = VerdictType.APPROVED
            elif (1 - approval_fraction) >= self._config.quorum_threshold:
                verdict = VerdictType.REJECTED
            else:
                verdict = VerdictType.HUNG

        self._deliberation.verdict = verdict
        self._deliberation.verdict_reasoning = (
            f"Votes: {approve_count} approve, {reject_count} reject "
            f"({approval_fraction:.0%} approval)"
        )

        logger.info(
            f"Jury verdict: {verdict.value} "
            f"({approve_count}/{reject_count} = {approval_fraction:.0%})"
        )

        return verdict

    # =========================================================================
    # Jury Management
    # =========================================================================

    def set_case(self, description: str) -> None:
        """Set the case description for deliberation.

        Args:
            description: Description of the matter being decided.
        """
        self._case_description = description

    def add_peer_juror(self, juror_id: str) -> None:
        """Add a peer juror to the jury.

        Args:
            juror_id: ID of the peer juror agent.
        """
        if juror_id not in self._peer_jurors:
            self._peer_jurors.append(juror_id)

    def set_foreperson(self, is_foreperson: bool = True) -> None:
        """Set whether this agent is the foreperson.

        Args:
            is_foreperson: True to make this agent foreperson.
        """
        self._is_foreperson = is_foreperson

    def reset_deliberation(self) -> None:
        """Reset deliberation state for a new case."""
        self._deliberation = DeliberationState(max_rounds=self._config.max_deliberation_rounds)
        self._case_description = ""

    def get_verdict(self) -> tuple[VerdictType, str]:
        """Get the current verdict and reasoning.

        Returns:
            Tuple of (verdict, reasoning).
        """
        return self._deliberation.verdict, self._deliberation.verdict_reasoning
