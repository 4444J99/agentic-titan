"""
Actor-Network Agent - ANT-based coordination patterns.

Implements Actor-Network Theory (ANT) patterns from Bruno Latour:
- Treats tools/services as actants with agency
- Enrolls and tracks actant loyalty
- Translation operations to align interests
- Hybrid networks of humans and nonhumans

The network is defined by its relations, not its nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from agents.framework.base_agent import AgentState, BaseAgent

logger = logging.getLogger("titan.agents.actor_network")


class ActantType(StrEnum):
    """Types of actants in the network."""

    HUMAN = "human"  # Human agents
    NONHUMAN = "nonhuman"  # Tools, services, infrastructure
    HYBRID = "hybrid"  # Human-nonhuman combinations
    CONCEPTUAL = "conceptual"  # Ideas, standards, protocols


class LoyaltyLevel(StrEnum):
    """Loyalty level of an actant."""

    ENROLLED = "enrolled"  # Fully committed
    INTERESTED = "interested"  # Showing interest
    NEUTRAL = "neutral"  # Not yet engaged
    RESISTANT = "resistant"  # Actively opposing
    DEFECTED = "defected"  # Formerly enrolled, now left


class TranslationType(StrEnum):
    """Types of translation operations."""

    PROBLEMATIZATION = "problematization"  # Defining the problem
    INTERESSEMENT = "interessement"  # Getting others interested
    ENROLLMENT = "enrollment"  # Committing actants
    MOBILIZATION = "mobilization"  # Speaking for others


@dataclass
class Actant:
    """An actant in the network.

    Actants are anything that makes a difference - they can be
    human or nonhuman, material or conceptual.
    """

    actant_id: str
    actant_type: ActantType
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    loyalty: LoyaltyLevel = LoyaltyLevel.NEUTRAL
    enrolled_at: datetime | None = None
    interests: list[str] = field(default_factory=list)
    translations_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "actant_id": self.actant_id,
            "actant_type": self.actant_type.value,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "loyalty": self.loyalty.value,
            "enrolled_at": self.enrolled_at.isoformat() if self.enrolled_at else None,
            "interests": self.interests,
            "translations_applied": self.translations_applied,
            "metadata": self.metadata,
        }


@dataclass
class Translation:
    """A translation operation in the network.

    Translations transform the interests and identities of actants
    to align them with the network's goals.
    """

    translation_id: str
    translation_type: TranslationType
    source_actant: str
    target_actants: list[str]
    description: str
    success: bool = False
    performed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    result: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "translation_id": self.translation_id,
            "translation_type": self.translation_type.value,
            "source_actant": self.source_actant,
            "target_actants": self.target_actants,
            "description": self.description,
            "success": self.success,
            "performed_at": self.performed_at.isoformat(),
            "result": self.result,
        }


@dataclass
class Inscription:
    """An inscription - materialized knowledge in the network.

    Inscriptions are ways of recording knowledge that can be
    transported, combined, and used to convince others.
    """

    inscription_id: str
    content: str
    inscribed_by: str
    carriers: list[str]  # Actants carrying this inscription
    immutability: float = 1.0  # How unchangeable (0-1)
    combinability: float = 1.0  # How easily combined (0-1)
    mobility: float = 1.0  # How easily moved (0-1)


class ActorNetworkAgent(BaseAgent):
    """
    Agent implementing Actor-Network Theory patterns.

    Builds and maintains networks of human and nonhuman actants,
    using translation operations to align interests and enroll
    participants.

    Key concepts:
    - Actants: Anything that acts - humans, tools, concepts
    - Translation: Transforming interests to align
    - Enrollment: Committing actants to the network
    - Inscriptions: Materialized knowledge

    Capabilities:
    - Actant enrollment and tracking
    - Translation operations
    - Network mapping
    - Inscription management
    """

    def __init__(
        self,
        network_name: str = "default_network",
        **kwargs: Any,
    ) -> None:
        """Initialize actor-network agent.

        Args:
            network_name: Name for this network.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", "actor_network")
        kwargs.setdefault(
            "capabilities",
            [
                "actant_enrollment",
                "translation",
                "network_mapping",
                "inscription_management",
            ],
        )
        super().__init__(**kwargs)

        self._network_name = network_name
        self._actants: dict[str, Actant] = {}
        self._translations: list[Translation] = []
        self._inscriptions: list[Inscription] = []
        self._associations: dict[str, list[str]] = {}  # actant -> connected actants
        self._obligatory_passage_points: list[str] = []  # Key nodes

    @property
    def network_name(self) -> str:
        """Get network name."""
        return self._network_name

    async def initialize(self) -> None:
        """Initialize actor-network agent."""
        self._state = AgentState.READY

        # Register self as an actant
        self._register_actant(
            actant_id=self.agent_id,
            actant_type=ActantType.HYBRID,
            name=f"Network Builder: {self._network_name}",
            description="Agent coordinating the actor-network",
            capabilities=self.capabilities,
        )

        logger.info(f"Actor-network agent initialized (network={self._network_name})")

    async def work(self) -> dict[str, Any]:
        """Perform actor-network work cycle."""
        actions: list[str] = []
        result: dict[str, Any] = {
            "network": self._network_name,
            "actant_count": len(self._actants),
            "enrolled_count": self._count_enrolled(),
            "actions": actions,
        }

        # Check loyalty of enrolled actants
        defections = await self._check_loyalty()
        if defections:
            actions.append(f"detected_{len(defections)}_defections")
            result["defections"] = defections

        # Attempt to enroll interested actants
        new_enrollments = await self._attempt_enrollments()
        if new_enrollments:
            actions.append(f"enrolled_{len(new_enrollments)}_actants")
            result["new_enrollments"] = new_enrollments

        # Strengthen associations
        await self._strengthen_network()
        actions.append("strengthened_associations")

        return result

    async def shutdown(self) -> None:
        """Shutdown actor-network agent."""
        logger.info(
            f"Actor-network agent shutdown "
            f"(actants={len(self._actants)}, enrolled={self._count_enrolled()})"
        )

    # =========================================================================
    # Actant Management
    # =========================================================================

    def _register_actant(
        self,
        actant_id: str,
        actant_type: ActantType,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
        interests: list[str] | None = None,
    ) -> Actant:
        """Register an actant in the network.

        Args:
            actant_id: Unique actant identifier.
            actant_type: Type of actant.
            name: Actant name.
            description: Actant description.
            capabilities: What the actant can do.
            interests: What the actant wants.

        Returns:
            The registered Actant.
        """
        actant = Actant(
            actant_id=actant_id,
            actant_type=actant_type,
            name=name,
            description=description,
            capabilities=capabilities or [],
            interests=interests or [],
        )

        self._actants[actant_id] = actant
        self._associations[actant_id] = []

        return actant

    def register_human_actant(
        self,
        actant_id: str,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
        interests: list[str] | None = None,
    ) -> Actant:
        """Register a human actant.

        Args:
            actant_id: Unique identifier.
            name: Human's name/role.
            description: Description.
            capabilities: Capabilities.
            interests: Interests.

        Returns:
            The registered Actant.
        """
        return self._register_actant(
            actant_id, ActantType.HUMAN, name, description, capabilities, interests
        )

    def register_nonhuman_actant(
        self,
        actant_id: str,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
    ) -> Actant:
        """Register a nonhuman actant (tool, service, etc.).

        Examples: door closer, microchip, API, speed bump, fiber optic cable

        Args:
            actant_id: Unique identifier.
            name: Actant name.
            description: What it is.
            capabilities: What it can do.

        Returns:
            The registered Actant.
        """
        return self._register_actant(
            actant_id, ActantType.NONHUMAN, name, description, capabilities, []
        )

    def register_conceptual_actant(
        self,
        actant_id: str,
        name: str,
        description: str,
    ) -> Actant:
        """Register a conceptual actant (standard, protocol, idea).

        Args:
            actant_id: Unique identifier.
            name: Concept name.
            description: What the concept represents.

        Returns:
            The registered Actant.
        """
        return self._register_actant(actant_id, ActantType.CONCEPTUAL, name, description, [], [])

    def get_actant(self, actant_id: str) -> Actant | None:
        """Get an actant by ID."""
        return self._actants.get(actant_id)

    def get_actants_by_type(self, actant_type: ActantType) -> list[Actant]:
        """Get all actants of a type."""
        return [a for a in self._actants.values() if a.actant_type == actant_type]

    def get_enrolled_actants(self) -> list[Actant]:
        """Get all enrolled actants."""
        return [a for a in self._actants.values() if a.loyalty == LoyaltyLevel.ENROLLED]

    def _count_enrolled(self) -> int:
        """Count enrolled actants."""
        return len(self.get_enrolled_actants())

    # =========================================================================
    # Translation Operations
    # =========================================================================

    async def problematize(
        self,
        problem_statement: str,
        target_actants: list[str],
    ) -> Translation:
        """Perform problematization - define the problem and make self indispensable.

        Args:
            problem_statement: The problem being defined.
            target_actants: Actants to include in problematization.

        Returns:
            The Translation record.
        """
        import uuid

        translation = Translation(
            translation_id=f"TR-{uuid.uuid4().hex[:8]}",
            translation_type=TranslationType.PROBLEMATIZATION,
            source_actant=self.agent_id,
            target_actants=target_actants,
            description=problem_statement,
        )

        # Update target actants
        for actant_id in target_actants:
            if actant_id in self._actants:
                self._actants[actant_id].translations_applied.append(translation.translation_id)
                # Successful problematization creates interest
                if self._actants[actant_id].loyalty == LoyaltyLevel.NEUTRAL:
                    self._actants[actant_id].loyalty = LoyaltyLevel.INTERESTED

        translation.success = True
        translation.result = f"Problematized for {len(target_actants)} actants"

        self._translations.append(translation)

        logger.info(f"Problematization complete: {problem_statement[:50]}...")
        return translation

    async def interesse(
        self,
        proposition: str,
        target_actants: list[str],
    ) -> Translation:
        """Perform interessement - attract and interest actants.

        Args:
            proposition: What is being offered/proposed.
            target_actants: Actants to interest.

        Returns:
            The Translation record.
        """
        import uuid

        translation = Translation(
            translation_id=f"TR-{uuid.uuid4().hex[:8]}",
            translation_type=TranslationType.INTERESSEMENT,
            source_actant=self.agent_id,
            target_actants=target_actants,
            description=proposition,
        )

        successful = 0
        for actant_id in target_actants:
            if actant_id in self._actants:
                actant = self._actants[actant_id]
                actant.translations_applied.append(translation.translation_id)

                # Interessement moves interested actants toward enrollment
                if actant.loyalty in [LoyaltyLevel.NEUTRAL, LoyaltyLevel.INTERESTED]:
                    # Check if proposition aligns with interests
                    if any(interest in proposition.lower() for interest in actant.interests):
                        actant.loyalty = LoyaltyLevel.INTERESTED
                        successful += 1

        translation.success = successful > 0
        translation.result = f"Interested {successful}/{len(target_actants)} actants"

        self._translations.append(translation)
        return translation

    async def enroll(
        self,
        commitment_terms: str,
        target_actants: list[str],
    ) -> Translation:
        """Perform enrollment - commit actants to the network.

        Args:
            commitment_terms: Terms of enrollment.
            target_actants: Actants to enroll.

        Returns:
            The Translation record.
        """
        import uuid

        translation = Translation(
            translation_id=f"TR-{uuid.uuid4().hex[:8]}",
            translation_type=TranslationType.ENROLLMENT,
            source_actant=self.agent_id,
            target_actants=target_actants,
            description=commitment_terms,
        )

        enrolled = 0
        for actant_id in target_actants:
            if actant_id in self._actants:
                actant = self._actants[actant_id]

                # Only interested actants can be enrolled
                if actant.loyalty == LoyaltyLevel.INTERESTED:
                    actant.loyalty = LoyaltyLevel.ENROLLED
                    actant.enrolled_at = datetime.now(UTC)
                    actant.translations_applied.append(translation.translation_id)
                    enrolled += 1

                    # Create association with network builder
                    self.associate(self.agent_id, actant_id)

        translation.success = enrolled > 0
        translation.result = f"Enrolled {enrolled}/{len(target_actants)} actants"

        self._translations.append(translation)

        logger.info(f"Enrollment: {enrolled} actants enrolled")
        return translation

    async def mobilize(
        self,
        spokesperson: str,
        represented_actants: list[str],
    ) -> Translation:
        """Perform mobilization - have enrolled actants speak for others.

        Args:
            spokesperson: Actant speaking for others.
            represented_actants: Actants being represented.

        Returns:
            The Translation record.
        """
        import uuid

        translation = Translation(
            translation_id=f"TR-{uuid.uuid4().hex[:8]}",
            translation_type=TranslationType.MOBILIZATION,
            source_actant=spokesperson,
            target_actants=represented_actants,
            description=f"{spokesperson} speaks for {len(represented_actants)} actants",
        )

        # Verify spokesperson is enrolled
        if spokesperson in self._actants:
            if self._actants[spokesperson].loyalty == LoyaltyLevel.ENROLLED:
                translation.success = True
                translation.result = "Mobilization successful"

                # Mark spokesperson as obligatory passage point
                if spokesperson not in self._obligatory_passage_points:
                    self._obligatory_passage_points.append(spokesperson)

        self._translations.append(translation)
        return translation

    # =========================================================================
    # Network Structure
    # =========================================================================

    def associate(self, actant_a: str, actant_b: str) -> bool:
        """Create an association between two actants.

        Args:
            actant_a: First actant.
            actant_b: Second actant.

        Returns:
            True if association created.
        """
        if actant_a not in self._actants or actant_b not in self._actants:
            return False

        if actant_b not in self._associations.get(actant_a, []):
            self._associations[actant_a].append(actant_b)
        if actant_a not in self._associations.get(actant_b, []):
            self._associations[actant_b].append(actant_a)

        return True

    def get_associations(self, actant_id: str) -> list[str]:
        """Get all actants associated with an actant."""
        return self._associations.get(actant_id, [])

    def set_obligatory_passage_point(self, actant_id: str) -> bool:
        """Set an actant as an obligatory passage point.

        Obligatory passage points are actants through which
        other actants must pass to participate in the network.

        Args:
            actant_id: Actant to set as OPP.

        Returns:
            True if set successfully.
        """
        if actant_id not in self._actants:
            return False

        if actant_id not in self._obligatory_passage_points:
            self._obligatory_passage_points.append(actant_id)

        return True

    def get_obligatory_passage_points(self) -> list[Actant]:
        """Get all obligatory passage points."""
        return [
            self._actants[aid] for aid in self._obligatory_passage_points if aid in self._actants
        ]

    # =========================================================================
    # Inscription Management
    # =========================================================================

    def create_inscription(
        self,
        content: str,
        carriers: list[str],
    ) -> Inscription:
        """Create an inscription - materialized knowledge.

        Inscriptions are "immutable mobiles" - they can be
        transported while maintaining their form.

        Args:
            content: The inscribed content.
            carriers: Actants carrying this inscription.

        Returns:
            The created Inscription.
        """
        import uuid

        inscription = Inscription(
            inscription_id=f"INS-{uuid.uuid4().hex[:8]}",
            content=content,
            inscribed_by=self.agent_id,
            carriers=carriers,
        )

        self._inscriptions.append(inscription)
        return inscription

    def get_inscriptions_carried_by(self, actant_id: str) -> list[Inscription]:
        """Get all inscriptions carried by an actant."""
        return [i for i in self._inscriptions if actant_id in i.carriers]

    # =========================================================================
    # Network Maintenance
    # =========================================================================

    async def _check_loyalty(self) -> list[str]:
        """Check for loyalty changes and defections.

        Returns:
            List of defected actant IDs.
        """
        import random

        defections: list[str] = []

        for actant in self._actants.values():
            if actant.loyalty == LoyaltyLevel.ENROLLED:
                # Small chance of defection
                if random.random() < 0.02:
                    actant.loyalty = LoyaltyLevel.DEFECTED
                    defections.append(actant.actant_id)
                    logger.warning(f"Actant defected: {actant.name}")

        return defections

    async def _attempt_enrollments(self) -> list[str]:
        """Attempt to enroll interested actants.

        Returns:
            List of newly enrolled actant IDs.
        """
        interested = [a for a in self._actants.values() if a.loyalty == LoyaltyLevel.INTERESTED]

        if not interested:
            return []

        # Attempt enrollment
        target_ids = [a.actant_id for a in interested[:3]]
        await self.enroll(
            "Standard network participation terms",
            target_ids,
        )

        return [
            aid
            for aid in target_ids
            if aid in self._actants and self._actants[aid].loyalty == LoyaltyLevel.ENROLLED
        ]

    async def _strengthen_network(self) -> None:
        """Strengthen associations between enrolled actants."""
        enrolled = self.get_enrolled_actants()

        # Create associations between enrolled actants that aren't connected
        for i, actant_a in enumerate(enrolled):
            for actant_b in enrolled[i + 1 :]:
                if actant_b.actant_id not in self._associations.get(actant_a.actant_id, []):
                    # 20% chance of forming new association
                    import random

                    if random.random() < 0.2:
                        self.associate(actant_a.actant_id, actant_b.actant_id)

    def get_network_summary(self) -> dict[str, Any]:
        """Get summary of the actor-network.

        Returns:
            Dictionary with network summary.
        """
        by_type: dict[str, int] = {}
        by_loyalty: dict[str, int] = {}

        for actant in self._actants.values():
            by_type[actant.actant_type.value] = by_type.get(actant.actant_type.value, 0) + 1
            by_loyalty[actant.loyalty.value] = by_loyalty.get(actant.loyalty.value, 0) + 1

        return {
            "network_name": self._network_name,
            "total_actants": len(self._actants),
            "by_type": by_type,
            "by_loyalty": by_loyalty,
            "total_translations": len(self._translations),
            "successful_translations": len([t for t in self._translations if t.success]),
            "obligatory_passage_points": len(self._obligatory_passage_points),
            "inscriptions": len(self._inscriptions),
        }
