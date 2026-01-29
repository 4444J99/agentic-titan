"""
Assemblage Agent - Heterogeneous assembly coordination.

Implements assemblage theory patterns from Deleuze & Guattari:
- Composed of diverse, heterogeneous parts
- Territorialization/deterritorialization dynamics
- Function-based cohesion (not similarity)
- Can include material, expressive, and conceptual components

An assemblage is what it does, not what it is made of.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentState

logger = logging.getLogger("titan.agents.assemblage")


class ComponentType(str, Enum):
    """Types of components in an assemblage."""

    MATERIAL = "material"      # Physical/computational resources
    EXPRESSIVE = "expressive"  # Symbols, communications, interfaces
    MACHINIC = "machinic"      # Functional processes
    ENUNCIATIVE = "enunciative"  # Language, statements, rules


class TerritorialState(str, Enum):
    """Territorial state of the assemblage."""

    DETERRITORIALIZED = "deterritorialized"  # Fluid, escaping
    RETERRITORIALIZING = "reterritorializing"  # Becoming stable
    TERRITORIALIZED = "territorialized"  # Stable, bounded
    STRATIFIED = "stratified"  # Over-coded, rigid


@dataclass
class AssemblageComponent:
    """A component of the assemblage."""

    component_id: str
    component_type: ComponentType
    description: str
    function: str  # What it does
    connections: list[str] = field(default_factory=list)
    intensity: float = 1.0  # How active/present
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "description": self.description,
            "function": self.function,
            "connections": self.connections,
            "intensity": self.intensity,
            "metadata": self.metadata,
        }


@dataclass
class AssemblageRelation:
    """A relation between components."""

    from_component: str
    to_component: str
    relation_type: str
    strength: float = 1.0
    bidirectional: bool = False


@dataclass
class AssemblageState:
    """Current state of the assemblage."""

    territorial_state: TerritorialState = TerritorialState.DETERRITORIALIZED
    cohesion: float = 0.5  # 0-1, how well-integrated
    stability: float = 0.5  # 0-1, resistance to change
    expressivity: float = 0.5  # 0-1, capacity for expression
    coding_level: float = 0.5  # 0-1, how rule-governed


class AssemblageAgent(BaseAgent):
    """
    Agent implementing assemblage theory patterns.

    An assemblage is a heterogeneous collection held together by
    functional connections rather than similarity. This agent
    tracks its composition and territorial dynamics.

    Key concepts:
    - Heterogeneity: diverse components working together
    - Territorialization: becoming bounded, stable
    - Deterritorialization: becoming fluid, escaping
    - Lines of flight: escape from fixed structures

    Capabilities:
    - Component management
    - Territorial state tracking
    - Relation mapping
    - Dynamic reconfiguration
    """

    # Thresholds for state transitions
    STABILITY_LOW = 0.3
    STABILITY_HIGH = 0.8
    COHESION_CRITICAL = 0.2

    def __init__(
        self,
        initial_state: TerritorialState = TerritorialState.DETERRITORIALIZED,
        **kwargs: Any,
    ) -> None:
        """Initialize assemblage agent.

        Args:
            initial_state: Initial territorial state.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", "assemblage")
        kwargs.setdefault("capabilities", [
            "heterogeneous_composition",
            "territorialization",
            "deterritorialization",
            "functional_cohesion",
        ])
        super().__init__(**kwargs)

        self._state_data = AssemblageState(territorial_state=initial_state)
        self._components: dict[str, AssemblageComponent] = {}
        self._relations: list[AssemblageRelation] = []
        self._lines_of_flight: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []

    @property
    def territorial_state(self) -> TerritorialState:
        """Get current territorial state."""
        return self._state_data.territorial_state

    @property
    def assemblage_state(self) -> AssemblageState:
        """Get full assemblage state."""
        return self._state_data

    async def initialize(self) -> None:
        """Initialize assemblage agent."""
        self._state = AgentState.READY
        logger.info(
            f"Assemblage agent initialized "
            f"(territorial_state={self._state_data.territorial_state.value})"
        )

    async def work(self) -> dict[str, Any]:
        """Perform assemblage work cycle."""
        result = {
            "territorial_state": self._state_data.territorial_state.value,
            "cohesion": self._state_data.cohesion,
            "component_count": len(self._components),
            "actions": [],
        }

        # Update state based on composition
        await self._update_state()

        # Check for state transitions
        transition = await self._check_transitions()
        if transition:
            result["actions"].append(f"transitioned_to_{transition}")

        # Process any lines of flight
        flights = await self._process_lines_of_flight()
        if flights:
            result["actions"].append(f"processed_{len(flights)}_lines_of_flight")
            result["lines_of_flight"] = flights

        # Maintain cohesion
        if self._state_data.cohesion < self.COHESION_CRITICAL:
            await self._reinforce_connections()
            result["actions"].append("reinforced_connections")

        return result

    async def shutdown(self) -> None:
        """Shutdown assemblage agent."""
        logger.info(f"Assemblage agent shutdown (components={len(self._components)})")

    # =========================================================================
    # Component Management
    # =========================================================================

    def add_component(
        self,
        component_id: str,
        component_type: ComponentType,
        description: str,
        function: str,
        metadata: dict[str, Any] | None = None,
    ) -> AssemblageComponent:
        """Add a component to the assemblage.

        Args:
            component_id: Unique component identifier.
            component_type: Type of component.
            description: What the component is.
            function: What the component does.
            metadata: Additional metadata.

        Returns:
            The created component.
        """
        component = AssemblageComponent(
            component_id=component_id,
            component_type=component_type,
            description=description,
            function=function,
            metadata=metadata or {},
        )

        self._components[component_id] = component

        # Adding components can deterritorialize
        if self._state_data.territorial_state == TerritorialState.STRATIFIED:
            self._state_data.territorial_state = TerritorialState.TERRITORIALIZED
            self._state_data.stability -= 0.1

        self._record_event("component_added", {"component_id": component_id})

        return component

    def remove_component(self, component_id: str) -> bool:
        """Remove a component from the assemblage.

        Args:
            component_id: Component to remove.

        Returns:
            True if removed.
        """
        if component_id not in self._components:
            return False

        # Remove relations involving this component
        self._relations = [
            r for r in self._relations
            if r.from_component != component_id and r.to_component != component_id
        ]

        del self._components[component_id]

        # Removing components affects cohesion
        self._state_data.cohesion -= 0.1

        self._record_event("component_removed", {"component_id": component_id})

        return True

    def connect_components(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        strength: float = 1.0,
        bidirectional: bool = False,
    ) -> AssemblageRelation | None:
        """Create a relation between components.

        Args:
            from_id: Source component.
            to_id: Target component.
            relation_type: Type of relation.
            strength: Relation strength.
            bidirectional: Whether relation goes both ways.

        Returns:
            The created relation.
        """
        if from_id not in self._components or to_id not in self._components:
            return None

        relation = AssemblageRelation(
            from_component=from_id,
            to_component=to_id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
        )

        self._relations.append(relation)

        # Update component connections
        self._components[from_id].connections.append(to_id)
        if bidirectional:
            self._components[to_id].connections.append(from_id)

        # Connections increase cohesion
        self._state_data.cohesion = min(1.0, self._state_data.cohesion + 0.05)

        return relation

    def get_components_by_type(
        self,
        component_type: ComponentType,
    ) -> list[AssemblageComponent]:
        """Get all components of a type.

        Args:
            component_type: Type to filter by.

        Returns:
            List of matching components.
        """
        return [
            c for c in self._components.values()
            if c.component_type == component_type
        ]

    def get_component_by_function(self, function: str) -> list[AssemblageComponent]:
        """Get components by their function.

        Args:
            function: Function to match.

        Returns:
            List of components with matching function.
        """
        function_lower = function.lower()
        return [
            c for c in self._components.values()
            if function_lower in c.function.lower()
        ]

    # =========================================================================
    # Territorialization Dynamics
    # =========================================================================

    async def territorialize(self, reason: str = "") -> bool:
        """Move toward more stable, bounded state.

        Args:
            reason: Reason for territorialization.

        Returns:
            True if state changed.
        """
        current = self._state_data.territorial_state

        if current == TerritorialState.STRATIFIED:
            return False  # Already maximally territorialized

        transitions = {
            TerritorialState.DETERRITORIALIZED: TerritorialState.RETERRITORIALIZING,
            TerritorialState.RETERRITORIALIZING: TerritorialState.TERRITORIALIZED,
            TerritorialState.TERRITORIALIZED: TerritorialState.STRATIFIED,
        }

        if current in transitions:
            self._state_data.territorial_state = transitions[current]
            self._state_data.stability += 0.1
            self._state_data.coding_level += 0.1

            self._record_event("territorialized", {
                "from": current.value,
                "to": self._state_data.territorial_state.value,
                "reason": reason,
            })

            return True

        return False

    async def deterritorialize(self, reason: str = "") -> bool:
        """Move toward more fluid, escaping state.

        Args:
            reason: Reason for deterritorialization.

        Returns:
            True if state changed.
        """
        current = self._state_data.territorial_state

        if current == TerritorialState.DETERRITORIALIZED:
            return False  # Already maximally deterritorialized

        transitions = {
            TerritorialState.STRATIFIED: TerritorialState.TERRITORIALIZED,
            TerritorialState.TERRITORIALIZED: TerritorialState.RETERRITORIALIZING,
            TerritorialState.RETERRITORIALIZING: TerritorialState.DETERRITORIALIZED,
        }

        if current in transitions:
            self._state_data.territorial_state = transitions[current]
            self._state_data.stability = max(0.0, self._state_data.stability - 0.1)
            self._state_data.coding_level = max(0.0, self._state_data.coding_level - 0.1)

            self._record_event("deterritorialized", {
                "from": current.value,
                "to": self._state_data.territorial_state.value,
                "reason": reason,
            })

            return True

        return False

    async def initiate_line_of_flight(
        self,
        component_id: str,
        direction: str,
        intensity: float = 0.5,
    ) -> dict[str, Any]:
        """Initiate a line of flight for a component.

        A line of flight is an escape from fixed structures,
        potentially leading to transformation.

        Args:
            component_id: Component escaping.
            direction: Direction/nature of escape.
            intensity: Intensity of escape.

        Returns:
            Line of flight data.
        """
        if component_id not in self._components:
            return {"error": "component_not_found"}

        flight = {
            "component_id": component_id,
            "direction": direction,
            "intensity": intensity,
            "initiated_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }

        self._lines_of_flight.append(flight)

        # Reduce component intensity
        self._components[component_id].intensity -= intensity * 0.5

        # Lines of flight destabilize
        self._state_data.stability = max(0.0, self._state_data.stability - intensity * 0.2)

        self._record_event("line_of_flight", flight)

        return flight

    async def _process_lines_of_flight(self) -> list[dict[str, Any]]:
        """Process active lines of flight.

        Returns:
            List of processed flight results.
        """
        processed: list[dict[str, Any]] = []

        for flight in self._lines_of_flight:
            if flight["status"] != "active":
                continue

            # Lines of flight can lead to transformation or recapture
            import random
            if random.random() < 0.3:  # 30% chance of transformation
                flight["status"] = "transformed"
                flight["result"] = "new_assemblage_potential"
                processed.append(flight)

                # May create new component
                if random.random() < 0.5:
                    self.add_component(
                        component_id=f"emerged_{flight['component_id']}",
                        component_type=ComponentType.MACHINIC,
                        description=f"Emerged from line of flight: {flight['direction']}",
                        function="transformation_product",
                    )

            elif random.random() < 0.5:  # Recapture
                flight["status"] = "recaptured"
                flight["result"] = "reterritorialized"
                processed.append(flight)

                # Restore component intensity
                if flight["component_id"] in self._components:
                    self._components[flight["component_id"]].intensity = 1.0

        return processed

    async def _check_transitions(self) -> str | None:
        """Check if territorial state should transition.

        Returns:
            New state name if transitioned, None otherwise.
        """
        # Auto-territorialize if very stable
        if (
            self._state_data.stability > self.STABILITY_HIGH and
            self._state_data.territorial_state != TerritorialState.STRATIFIED
        ):
            await self.territorialize("high_stability")
            return self._state_data.territorial_state.value

        # Auto-deterritorialize if unstable
        if (
            self._state_data.stability < self.STABILITY_LOW and
            self._state_data.territorial_state != TerritorialState.DETERRITORIALIZED
        ):
            await self.deterritorialize("low_stability")
            return self._state_data.territorial_state.value

        return None

    async def _update_state(self) -> None:
        """Update assemblage state based on composition."""
        if not self._components:
            self._state_data.cohesion = 0.0
            return

        # Cohesion based on connections
        total_possible = len(self._components) * (len(self._components) - 1)
        total_actual = sum(len(c.connections) for c in self._components.values())
        if total_possible > 0:
            connection_density = total_actual / total_possible
            self._state_data.cohesion = min(1.0, connection_density * 2)

        # Expressivity based on expressive components
        expressive = self.get_components_by_type(ComponentType.EXPRESSIVE)
        self._state_data.expressivity = min(1.0, len(expressive) / max(len(self._components), 1))

    async def _reinforce_connections(self) -> None:
        """Reinforce weak connections to maintain cohesion."""
        for relation in self._relations:
            if relation.strength < 0.5:
                relation.strength = min(1.0, relation.strength + 0.1)

    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Record an event in history."""
        self._history.append({
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Keep only recent history
        if len(self._history) > 100:
            self._history = self._history[-50:]

    def get_composition(self) -> dict[str, Any]:
        """Get assemblage composition summary.

        Returns:
            Dictionary with composition data.
        """
        by_type: dict[str, int] = {}
        for c in self._components.values():
            by_type[c.component_type.value] = by_type.get(c.component_type.value, 0) + 1

        return {
            "total_components": len(self._components),
            "by_type": by_type,
            "total_relations": len(self._relations),
            "active_lines_of_flight": len([f for f in self._lines_of_flight if f["status"] == "active"]),
            "state": {
                "territorial": self._state_data.territorial_state.value,
                "cohesion": self._state_data.cohesion,
                "stability": self._state_data.stability,
                "expressivity": self._state_data.expressivity,
                "coding_level": self._state_data.coding_level,
            },
        }
