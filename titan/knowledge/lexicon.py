"""Body Lexicon data models.

Defines the taxonomy of assembly patterns based on the "Morphodynamics of Assembly"
framework, covering physical, biological, ethological, human, philosophical,
digital, and assembly theory categories.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.knowledge.lexicon")


class LexiconCategory(str, Enum):
    """Categories of assembly bodies in the lexicon."""

    INORGANIC_PHYSICS = "inorganic_physics"
    BIOLOGICAL_SUPERORGANISM = "biological_superorganism"
    ANIMAL_ETHOLOGY = "animal_ethology"
    HUMAN_ORGANIZATION = "human_organization"
    PHILOSOPHICAL_ASSEMBLAGE = "philosophical_assemblage"
    DIGITAL_ALGORITHMIC = "digital_algorithmic"
    ASSEMBLY_THEORY = "assembly_theory"

    @classmethod
    def from_category_id(cls, category_id: str) -> LexiconCategory:
        """Convert $CATEGORY_* ID to enum value."""
        mapping = {
            "$CATEGORY_INORGANIC_PHYSICS": cls.INORGANIC_PHYSICS,
            "$CATEGORY_BIOLOGICAL_SUPERORGANISM": cls.BIOLOGICAL_SUPERORGANISM,
            "$CATEGORY_ANIMAL_ETHOLOGY": cls.ANIMAL_ETHOLOGY,
            "$CATEGORY_HUMAN_ORGANIZATION": cls.HUMAN_ORGANIZATION,
            "$CATEGORY_PHILOSOPHICAL_ASSEMBLAGE": cls.PHILOSOPHICAL_ASSEMBLAGE,
            "$CATEGORY_DIGITAL_ALGORITHMIC": cls.DIGITAL_ALGORITHMIC,
            "$CATEGORY_ASSEMBLY_THEORY": cls.ASSEMBLY_THEORY,
        }
        return mapping.get(category_id, cls.INORGANIC_PHYSICS)


class InteractionType(str, Enum):
    """Types of internal interaction mechanisms."""

    GRAVITATIONAL = "gravitational"
    JAMMING = "jamming"
    SELF_ASSEMBLY = "self_assembly"
    STIGMERGIC = "stigmergic"
    TOPOLOGICAL_NEIGHBOR = "topological_neighbor"
    ALLELOMIMETIC = "allelomimetic"
    HIERARCHICAL = "hierarchical"
    RHIZOMATIC = "rhizomatic"
    TERRITORIAL = "territorial"
    SIGNALING = "signaling"
    VOTING = "voting"
    PHEROMONE = "pheromone"
    CONTRACTUAL = "contractual"


@dataclass
class InteractionRule:
    """Describes how units within an assembly interact.

    Based on the three universal principles:
    1. Constraint - units surrender some autonomy
    2. Communication - information flow determines integrity
    3. Selection - assemblies endure when selection favors them
    """

    interaction_type: InteractionType
    description: str
    constraint_mechanism: str | None = None
    communication_mechanism: str | None = None
    selection_mechanism: str | None = None
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interaction_type": self.interaction_type.value,
            "description": self.description,
            "constraint_mechanism": self.constraint_mechanism,
            "communication_mechanism": self.communication_mechanism,
            "selection_mechanism": self.selection_mechanism,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionRule:
        """Create from dictionary."""
        return cls(
            interaction_type=InteractionType(data["interaction_type"]),
            description=data["description"],
            constraint_mechanism=data.get("constraint_mechanism"),
            communication_mechanism=data.get("communication_mechanism"),
            selection_mechanism=data.get("selection_mechanism"),
            examples=data.get("examples", []),
        )


@dataclass
class BodyEntry:
    """A single entry in the Body Lexicon.

    Represents an assembly body with its category, aliases, referent
    description, notes, and cross-links to related bodies.
    """

    id: str
    category: LexiconCategory
    aliases: list[str] = field(default_factory=list)
    referent: str = ""
    notes: str = ""
    links: list[str] = field(default_factory=list)
    interaction_rules: list[InteractionRule] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "seed"  # seed, learned, user

    def __post_init__(self) -> None:
        """Validate entry after initialization."""
        if not self.id:
            raise ValueError("BodyEntry id cannot be empty")

    @property
    def content_hash(self) -> str:
        """Generate a hash of the entry content for deduplication."""
        content = f"{self.id}:{self.category.value}:{self.referent}:{self.notes}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def searchable_text(self) -> str:
        """Get combined text for embedding/search."""
        parts = [
            self.id.replace("_", " "),
            self.referent,
            self.notes,
        ]
        parts.extend(alias.replace("_", " ") for alias in self.aliases)
        return " ".join(filter(None, parts))

    def matches_alias(self, query: str) -> bool:
        """Check if query matches any alias (case-insensitive, fuzzy)."""
        query_lower = query.lower().replace(" ", "_").replace("-", "_")
        id_lower = self.id.lower()

        # Exact match
        if query_lower == id_lower:
            return True

        # Alias match
        for alias in self.aliases:
            alias_lower = alias.lower()
            if query_lower == alias_lower or query_lower in alias_lower:
                return True

        # Partial match in id
        if query_lower in id_lower:
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "aliases": self.aliases,
            "referent": self.referent,
            "notes": self.notes,
            "links": self.links,
            "interaction_rules": [r.to_dict() for r in self.interaction_rules],
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyEntry:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now(timezone.utc)

        interaction_rules = [
            InteractionRule.from_dict(r)
            for r in data.get("interaction_rules", [])
        ]

        return cls(
            id=data["id"],
            category=LexiconCategory(data["category"]),
            aliases=data.get("aliases", []),
            referent=data.get("referent", ""),
            notes=data.get("notes", ""),
            links=data.get("links", []),
            interaction_rules=interaction_rules,
            embedding=data.get("embedding"),
            created_at=created_at,
            updated_at=updated_at,
            source=data.get("source", "seed"),
        )

    @classmethod
    def from_lexicon_entry(
        cls,
        body_id: str,
        category: LexiconCategory,
        entry_data: dict[str, Any],
    ) -> BodyEntry:
        """Create from raw $BODY_LEXICON format."""
        return cls(
            id=body_id,
            category=category,
            aliases=entry_data.get("aliases", []),
            referent=entry_data.get("referent", ""),
            notes=entry_data.get("notes", ""),
            links=entry_data.get("links", []),
            source="seed",
        )


# Pre-defined interaction rules for each category
CATEGORY_INTERACTION_RULES: dict[LexiconCategory, list[InteractionRule]] = {
    LexiconCategory.INORGANIC_PHYSICS: [
        InteractionRule(
            interaction_type=InteractionType.GRAVITATIONAL,
            description="Units coalesce via gravitational attraction, forming force chains",
            constraint_mechanism="Angular momentum and dissipation cause material to spiral",
            communication_mechanism="Gravitational waves and field interactions",
            selection_mechanism="Self-gravity leads to scale-free clustering",
            examples=["galaxy", "solar_system", "planet"],
        ),
        InteractionRule(
            interaction_type=InteractionType.JAMMING,
            description="Systems exist at threshold between fluid and solid",
            constraint_mechanism="Friction and local contacts constrain motion",
            communication_mechanism="Stress chains propagate through assembly",
            selection_mechanism="Packing fraction determines rigidity transition",
            examples=["granular_matter_pile", "foam", "grain_silo_pack"],
        ),
        InteractionRule(
            interaction_type=InteractionType.SELF_ASSEMBLY,
            description="Particles encode instructions via physical properties",
            constraint_mechanism="Local interactions via encoded properties",
            communication_mechanism="External fields supply energy",
            selection_mechanism="Distributed minimization of energy",
            examples=["magnetic_bead_assembly", "particle_lattice"],
        ),
    ],
    LexiconCategory.BIOLOGICAL_SUPERORGANISM: [
        InteractionRule(
            interaction_type=InteractionType.SIGNALING,
            description="Adhesion molecules and cell-signaling networks coordinate",
            constraint_mechanism="Germ-soma separation aligns interests",
            communication_mechanism="Morphogens and hormones coordinate differentiation",
            selection_mechanism="Apoptosis eliminates uncooperative cells",
            examples=["multicellular_organism", "tissue", "eukaryotic_cell"],
        ),
        InteractionRule(
            interaction_type=InteractionType.STIGMERGIC,
            description="Individuals modify environment, others respond to cues",
            constraint_mechanism="Worker policing enforces queen's reproductive monopoly",
            communication_mechanism="Pheromone trails and soil pellets as signals",
            selection_mechanism="Reproductive bottlenecks align genetic interests",
            examples=["eusocial_colony", "ant", "bee", "termite"],
        ),
    ],
    LexiconCategory.ANIMAL_ETHOLOGY: [
        InteractionRule(
            interaction_type=InteractionType.TOPOLOGICAL_NEIGHBOR,
            description="Each unit couples to ~6-7 nearest neighbors regardless of density",
            constraint_mechanism="Topological rule keeps network intact as flock expands",
            communication_mechanism="Scale-free correlations propagate velocity changes",
            selection_mechanism="System operates at criticality for maximal responsiveness",
            examples=["murmuration", "starling_flock", "flock"],
        ),
        InteractionRule(
            interaction_type=InteractionType.ALLELOMIMETIC,
            description="Animals copy movements of neighbors",
            constraint_mechanism="Selfish herd rule drives individuals toward center",
            communication_mechanism="Visual cues propagate quickly through group",
            selection_mechanism="Minimizes predation risk through dense packing",
            examples=["sheep_herd", "herd"],
        ),
    ],
    LexiconCategory.HUMAN_ORGANIZATION: [
        InteractionRule(
            interaction_type=InteractionType.HIERARCHICAL,
            description="Chains of command, specialized roles, written rules",
            constraint_mechanism="Rules constrain individual discretion",
            communication_mechanism="Formal channels and documentation",
            selection_mechanism="Structure outlives members, organizational immortality",
            examples=["bureaucracy", "government", "executive_branch"],
        ),
        InteractionRule(
            interaction_type=InteractionType.RHIZOMATIC,
            description="Rapid assembly without central leader",
            constraint_mechanism="Implicit scripts guide behavior",
            communication_mechanism="Mobile communication enables coordination",
            selection_mechanism="Group cohesion depends on information flow",
            examples=["flash_mob", "smart_mob", "political_movement"],
        ),
        InteractionRule(
            interaction_type=InteractionType.VOTING,
            description="Deliberative bodies reach decisions through procedure",
            constraint_mechanism="Selection and procedure constitute the body",
            communication_mechanism="Evidence presentation and debate",
            selection_mechanism="Consensus or majority rule",
            examples=["jury", "congress", "legislature"],
        ),
    ],
    LexiconCategory.PHILOSOPHICAL_ASSEMBLAGE: [
        InteractionRule(
            interaction_type=InteractionType.TERRITORIAL,
            description="Heterogeneous collections held by functional connections",
            constraint_mechanism="Territorialization stabilizes the assemblage",
            communication_mechanism="Functional connections across diverse parts",
            selection_mechanism="Deterritorialization can destabilize",
            examples=["assemblage", "territorialized_herd", "knight_assemblage"],
        ),
        InteractionRule(
            interaction_type=InteractionType.RHIZOMATIC,
            description="Any node can connect to any other, horizontal rules",
            constraint_mechanism="No central authority, distributed control",
            communication_mechanism="Lateral, adaptive connections",
            selection_mechanism="Resilience through redundancy",
            examples=["rhizome", "internet", "wikipedia", "underground_resistance_movement"],
        ),
        InteractionRule(
            interaction_type=InteractionType.HIERARCHICAL,
            description="Tree-like structure with top-down rules",
            constraint_mechanism="Limited connectivity enforced by hierarchy",
            communication_mechanism="Command flows downward, reports upward",
            selection_mechanism="State capture and striation",
            examples=["arborescent_structure", "state"],
        ),
    ],
    LexiconCategory.DIGITAL_ALGORITHMIC: [
        InteractionRule(
            interaction_type=InteractionType.PHEROMONE,
            description="Virtual ants deposit pheromone on good paths",
            constraint_mechanism="Pheromone evaporates, only frequent paths persist",
            communication_mechanism="Indirect coordination via environment",
            selection_mechanism="Near-optimal solutions emerge from local rules",
            examples=["ant_colony_optimization", "pheromone_field"],
        ),
        InteractionRule(
            interaction_type=InteractionType.SELF_ASSEMBLY,
            description="Particles explore landscape based on local and global best",
            constraint_mechanism="Momentum prevents premature convergence",
            communication_mechanism="Global best shared across swarm",
            selection_mechanism="Balance of exploration and exploitation",
            examples=["particle_swarm_optimization", "swarm_robotics"],
        ),
        InteractionRule(
            interaction_type=InteractionType.CONTRACTUAL,
            description="Smart contracts replace traditional leaders",
            constraint_mechanism="Code enforces rules automatically",
            communication_mechanism="Proposals and votes on-chain",
            selection_mechanism="Token-weighted governance, oligarchy risks",
            examples=["decentralized_autonomous_organization", "smart_contract", "blockchain_governance"],
        ),
    ],
    LexiconCategory.ASSEMBLY_THEORY: [
        InteractionRule(
            interaction_type=InteractionType.SELF_ASSEMBLY,
            description="Complex objects defined by minimum assembly steps",
            constraint_mechanism="Each step adds complexity constrained by prior state",
            communication_mechanism="Assembly history encoded in structure",
            selection_mechanism="High AI + abundance signals selection process",
            examples=["molecule", "protein", "smartphone", "iphone"],
        ),
    ],
}
