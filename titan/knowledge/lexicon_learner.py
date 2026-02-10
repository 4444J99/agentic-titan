"""Agent learning interface for Body Lexicon.

Provides methods for agents to learn new assembly patterns
and contribute to the shared lexicon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from titan.knowledge.lexicon import (
    BodyEntry,
    InteractionRule,
    InteractionType,
    LexiconCategory,
)
from titan.knowledge.lexicon_query import LexiconQueryInterface

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.knowledge.lexicon_learner")


@dataclass
class LearningProposal:
    """A proposed new lexicon entry from an agent."""

    agent_id: str
    body_id: str
    category: LexiconCategory
    referent: str
    notes: str = ""
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    interaction_rules: list[InteractionRule] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "pending"  # pending, approved, rejected

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "body_id": self.body_id,
            "category": self.category.value,
            "referent": self.referent,
            "notes": self.notes,
            "aliases": self.aliases,
            "links": self.links,
            "interaction_rules": [r.to_dict() for r in self.interaction_rules],
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
        }


class LexiconLearner:
    """Agent interface for learning and contributing to the Body Lexicon.

    Enables agents to:
    - Propose new assembly patterns they discover
    - Refine existing entries with new observations
    - Find relevant patterns for their current task
    - Build analogies across domains
    """

    # Minimum confidence required for auto-approval
    AUTO_APPROVE_THRESHOLD = 0.85

    # Maximum pending proposals per agent
    MAX_PENDING_PROPOSALS = 10

    def __init__(
        self,
        query_interface: LexiconQueryInterface,
        agent_id: str,
        auto_approve: bool = False,
    ) -> None:
        """Initialize the learner.

        Args:
            query_interface: LexiconQueryInterface instance.
            agent_id: ID of the agent using this learner.
            auto_approve: Whether to auto-approve high-confidence proposals.
        """
        self._query = query_interface
        self._agent_id = agent_id
        self._auto_approve = auto_approve
        self._pending_proposals: list[LearningProposal] = []

    @property
    def query(self) -> LexiconQueryInterface:
        """Access the query interface."""
        return self._query

    def propose_new_body(
        self,
        body_id: str,
        category: LexiconCategory,
        referent: str,
        notes: str = "",
        aliases: list[str] | None = None,
        links: list[str] | None = None,
        interaction_rules: list[InteractionRule] | None = None,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
    ) -> LearningProposal:
        """Propose a new assembly body for the lexicon.

        Args:
            body_id: Unique identifier for the body.
            category: The category this body belongs to.
            referent: Description of what this body refers to.
            notes: Additional notes or observations.
            aliases: Alternative names for this body.
            links: Cross-links to related entries.
            interaction_rules: How units within this body interact.
            confidence: Agent's confidence in this proposal (0-1).
            evidence: Supporting evidence for this proposal.

        Returns:
            The created LearningProposal.
        """
        # Check for existing entry
        existing = self._query.get_by_id(body_id)
        if existing:
            logger.warning(f"Body {body_id} already exists, updating instead")
            return self._propose_update(existing, notes, confidence, evidence or [])

        proposal = LearningProposal(
            agent_id=self._agent_id,
            body_id=body_id,
            category=category,
            referent=referent,
            notes=notes,
            aliases=aliases or [],
            links=links or [],
            interaction_rules=interaction_rules or [],
            confidence=confidence,
            evidence=evidence or [],
        )

        # Auto-approve if confidence is high enough
        if self._auto_approve and confidence >= self.AUTO_APPROVE_THRESHOLD:
            self._approve_proposal(proposal)
        else:
            self._pending_proposals.append(proposal)
            if len(self._pending_proposals) > self.MAX_PENDING_PROPOSALS:
                # Remove oldest pending proposal
                self._pending_proposals.pop(0)

        logger.info(
            f"Agent {self._agent_id} proposed new body: {body_id} "
            f"(confidence: {confidence:.2f}, status: {proposal.status})"
        )

        return proposal

    def _propose_update(
        self,
        existing: BodyEntry,
        new_notes: str,
        confidence: float,
        evidence: list[str],
    ) -> LearningProposal:
        """Create a proposal to update an existing entry."""
        updated_notes = f"{existing.notes}\n\n[Agent {self._agent_id} observation]: {new_notes}"

        proposal = LearningProposal(
            agent_id=self._agent_id,
            body_id=existing.id,
            category=existing.category,
            referent=existing.referent,
            notes=updated_notes,
            aliases=existing.aliases,
            links=existing.links,
            interaction_rules=existing.interaction_rules,
            confidence=confidence,
            evidence=evidence,
            status="pending",
        )

        self._pending_proposals.append(proposal)
        return proposal

    def _approve_proposal(self, proposal: LearningProposal) -> BodyEntry:
        """Approve and add a proposal to the lexicon."""
        entry = self._query.add_learned_entry(
            body_id=proposal.body_id,
            category=proposal.category,
            referent=proposal.referent,
            notes=proposal.notes,
            aliases=proposal.aliases,
            links=proposal.links,
            interaction_rules=proposal.interaction_rules,
        )
        proposal.status = "approved"

        logger.info(f"Auto-approved proposal: {proposal.body_id}")
        return entry

    def find_relevant_patterns(
        self,
        task_description: str,
        limit: int = 5,
    ) -> list[BodyEntry]:
        """Find lexicon entries relevant to a task.

        Args:
            task_description: Description of the current task.
            limit: Maximum number of results.

        Returns:
            List of relevant BodyEntry objects.
        """
        result = self._query.search(task_description, limit=limit)
        return result.entries

    def find_analogous_patterns(
        self,
        current_pattern_id: str,
        target_domain: LexiconCategory | None = None,
        limit: int = 3,
    ) -> list[tuple[BodyEntry, str]]:
        """Find analogous patterns in other domains.

        Args:
            current_pattern_id: ID of the current pattern.
            target_domain: Optional target category to search in.
            limit: Maximum number of analogies.

        Returns:
            List of (BodyEntry, reasoning) tuples.
        """
        analogies = self._query.find_analogies(
            current_pattern_id,
            target_category=target_domain,
            limit=limit,
        )

        return [(a.target_entry, a.reasoning) for a in analogies]

    def suggest_interaction_type(
        self,
        observed_behavior: str,
    ) -> InteractionType | None:
        """Suggest an interaction type based on observed behavior.

        Args:
            observed_behavior: Description of observed agent/unit behavior.

        Returns:
            Suggested InteractionType or None if uncertain.
        """
        # Simple keyword-based matching
        behavior_lower = observed_behavior.lower()

        keywords_map = {
            InteractionType.STIGMERGIC: [
                "environment",
                "trace",
                "pheromone",
                "indirect",
                "modify surroundings",
            ],
            InteractionType.TOPOLOGICAL_NEIGHBOR: [
                "neighbor",
                "nearest",
                "local",
                "adjacent",
                "proximity",
            ],
            InteractionType.HIERARCHICAL: [
                "hierarchy",
                "command",
                "leader",
                "tree",
                "top-down",
                "authority",
            ],
            InteractionType.RHIZOMATIC: [
                "network",
                "decentralized",
                "horizontal",
                "any-to-any",
                "distributed",
            ],
            InteractionType.VOTING: [
                "vote",
                "consensus",
                "majority",
                "deliberate",
                "decide together",
            ],
            InteractionType.SIGNALING: ["signal", "communicate", "message", "broadcast", "inform"],
            InteractionType.TERRITORIAL: ["territory", "boundary", "domain", "region", "zone"],
            InteractionType.PHEROMONE: ["trail", "mark", "scent", "deposit", "evaporate"],
            InteractionType.SELF_ASSEMBLY: [
                "emerge",
                "spontaneous",
                "self-organize",
                "bottom-up",
                "local rules",
            ],
            InteractionType.ALLELOMIMETIC: ["copy", "imitate", "follow", "mimic", "herd"],
        }

        best_type: InteractionType | None = None
        best_count = 0

        for itype, keywords in keywords_map.items():
            count = sum(1 for kw in keywords if kw in behavior_lower)
            if count > best_count:
                best_count = count
                best_type = itype

        return best_type if best_count > 0 else None

    def infer_category(self, description: str) -> LexiconCategory:
        """Infer the most likely category for a new body.

        Args:
            description: Description of the body.

        Returns:
            Most likely LexiconCategory.
        """
        desc_lower = description.lower()

        category_keywords = {
            LexiconCategory.INORGANIC_PHYSICS: [
                "gravity",
                "particle",
                "physics",
                "force",
                "material",
                "granular",
            ],
            LexiconCategory.BIOLOGICAL_SUPERORGANISM: [
                "cell",
                "organism",
                "colony",
                "gene",
                "evolution",
                "biological",
            ],
            LexiconCategory.ANIMAL_ETHOLOGY: [
                "flock",
                "herd",
                "animal",
                "bird",
                "swarm",
                "prey",
                "predator",
            ],
            LexiconCategory.HUMAN_ORGANIZATION: [
                "organization",
                "company",
                "government",
                "team",
                "crowd",
                "society",
            ],
            LexiconCategory.PHILOSOPHICAL_ASSEMBLAGE: [
                "assemblage",
                "network",
                "actor",
                "rhizome",
                "territory",
            ],
            LexiconCategory.DIGITAL_ALGORITHMIC: [
                "algorithm",
                "digital",
                "software",
                "optimization",
                "robot",
                "ai",
            ],
            LexiconCategory.ASSEMBLY_THEORY: [
                "assembly index",
                "complexity",
                "selection",
                "history",
            ],
        }

        best_category = LexiconCategory.HUMAN_ORGANIZATION  # Default
        best_count = 0

        for category, keywords in category_keywords.items():
            count = sum(1 for kw in keywords if kw in desc_lower)
            if count > best_count:
                best_count = count
                best_category = category

        return best_category

    def get_pending_proposals(self) -> list[LearningProposal]:
        """Get all pending proposals from this agent.

        Returns:
            List of pending LearningProposal objects.
        """
        return [p for p in self._pending_proposals if p.status == "pending"]

    def approve_proposal(self, body_id: str) -> BodyEntry | None:
        """Manually approve a pending proposal.

        Args:
            body_id: The body ID of the proposal to approve.

        Returns:
            Created BodyEntry if approved, None if not found.
        """
        for proposal in self._pending_proposals:
            if proposal.body_id == body_id and proposal.status == "pending":
                entry = self._approve_proposal(proposal)
                return entry

        return None

    def reject_proposal(self, body_id: str, reason: str = "") -> bool:
        """Reject a pending proposal.

        Args:
            body_id: The body ID of the proposal to reject.
            reason: Reason for rejection.

        Returns:
            True if rejected, False if not found.
        """
        for proposal in self._pending_proposals:
            if proposal.body_id == body_id and proposal.status == "pending":
                proposal.status = "rejected"
                logger.info(f"Rejected proposal {body_id}: {reason}")
                return True

        return False
