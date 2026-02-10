"""Machine Dynamics: War Machine vs State Machine.

Implements the tension between two organizational poles from
Deleuze & Guattari's work:

- State Machine: Capture, striate, organize, hierarchize
- War Machine: Nomadic, smooth space, resist capture, distribute

These dynamics enable agents to shift between structured and
emergent coordination patterns.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hive.assembly import AssemblyManager

logger = logging.getLogger("titan.hive.machines")


class MachineType(StrEnum):
    """Types of organizational machines."""

    STATE = "state"  # Hierarchical, capturing, striated
    WAR = "war"  # Nomadic, smooth, distributed
    HYBRID = "hybrid"  # Mixture of both


class OperationType(StrEnum):
    """Types of machine operations."""

    # State machine operations
    CAPTURE = "capture"  # Absorb into structure
    STRIATE = "striate"  # Impose divisions/hierarchy
    OVERCODING = "overcoding"  # Apply meta-rules
    APPROPRIATION = "appropriation"  # Take over resources

    # War machine operations
    SMOOTH = "smooth"  # Remove striations
    NOMADIZE = "nomadize"  # Enable mobility
    LINE_OF_FLIGHT = "line_of_flight"  # Enable escape
    DETERRITORIALIZE = "deterritorialize"  # Break boundaries


@dataclass
class MachineOperation:
    """Record of a machine operation."""

    operation_type: OperationType
    machine_type: MachineType
    target_agents: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_type": self.operation_type.value,
            "machine_type": self.machine_type.value,
            "target_agents": self.target_agents,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass
class MachineState:
    """Current state of the machine dynamics."""

    dominant_machine: MachineType = MachineType.HYBRID
    state_intensity: float = 0.5  # 0-1, how much state capture
    war_intensity: float = 0.5  # 0-1, how much nomadic activity
    capture_rate: float = 0.0  # Recent capture operations
    escape_rate: float = 0.0  # Recent escape operations
    striation_level: float = 0.5  # How structured the space is

    @property
    def balance(self) -> float:
        """Calculate state/war balance (-1 to 1).

        -1 = fully war machine
        0 = balanced/hybrid
        1 = fully state machine
        """
        return self.state_intensity - self.war_intensity


class MachineDynamics:
    """Manages the tension between State and War machine dynamics.

    The State machine seeks to:
    - Capture nomadic elements
    - Create striations (divisions, hierarchy)
    - Impose codes and rules
    - Appropriate resources

    The War machine seeks to:
    - Escape capture
    - Create smooth space
    - Enable lines of flight
    - Distribute power
    """

    # Thresholds for machine dominance
    STATE_DOMINANCE_THRESHOLD = 0.7
    WAR_DOMINANCE_THRESHOLD = -0.7

    def __init__(
        self,
        assembly_manager: AssemblyManager | None = None,
        initial_balance: float = 0.0,
    ) -> None:
        """Initialize machine dynamics.

        Args:
            assembly_manager: Optional assembly manager for coordination.
            initial_balance: Initial state/war balance (-1 to 1).
        """
        self._assembly = assembly_manager
        self._state = MachineState(
            state_intensity=0.5 + initial_balance * 0.5,
            war_intensity=0.5 - initial_balance * 0.5,
        )
        self._operations: list[MachineOperation] = []
        self._registered_topologies: dict[str, Any] = {}

    @property
    def state(self) -> MachineState:
        """Get current machine state."""
        return self._state

    @property
    def dominant_machine(self) -> MachineType:
        """Determine which machine is dominant."""
        balance = self._state.balance

        if balance > self.STATE_DOMINANCE_THRESHOLD:
            return MachineType.STATE
        elif balance < self.WAR_DOMINANCE_THRESHOLD:
            return MachineType.WAR
        else:
            return MachineType.HYBRID

    def register_topology(self, topology_id: str, topology: Any) -> None:
        """Register a topology for operations."""
        self._registered_topologies[topology_id] = topology

    # =========================================================================
    # State Machine Operations
    # =========================================================================

    async def capture_operation(
        self,
        target_agents: list[str],
        into_structure: str | None = None,
    ) -> MachineOperation:
        """Execute a capture operation (State machine).

        Attempt to absorb nomadic/escaped agents into structure.

        Args:
            target_agents: Agents to capture.
            into_structure: Optional structure/territory to capture into.

        Returns:
            MachineOperation record.
        """
        captured: list[str] = []

        for topology in self._registered_topologies.values():
            for agent_id in target_agents:
                # Try to capture escaped agents
                if hasattr(topology, "capture"):
                    if topology.capture(agent_id):
                        captured.append(agent_id)

                # Assign to territory if applicable
                if into_structure and hasattr(topology, "assign_to_territory"):
                    topology.assign_to_territory(agent_id, into_structure)
                    captured.append(agent_id)

        operation = MachineOperation(
            operation_type=OperationType.CAPTURE,
            machine_type=MachineType.STATE,
            target_agents=target_agents,
            success=len(captured) > 0,
            metadata={"captured": captured, "structure": into_structure},
        )
        self._operations.append(operation)

        # Update state
        if captured:
            self._state.state_intensity = min(1.0, self._state.state_intensity + 0.1)
            self._state.capture_rate = len(captured) / len(target_agents)

        logger.info(f"Capture operation: {len(captured)}/{len(target_agents)} captured")
        return operation

    async def striate_operation(
        self,
        region: str | None = None,
    ) -> MachineOperation:
        """Execute a striation operation (State machine).

        Impose divisions, hierarchy, and structure on smooth space.

        Args:
            region: Optional region to striate.

        Returns:
            MachineOperation record.
        """
        affected: list[str] = []

        for topology in self._registered_topologies.values():
            # Convert rhizomatic to arborescent patterns
            if hasattr(topology, "nodes") and hasattr(topology, "_root_id"):
                # Already hierarchical
                for node in topology.nodes.values():
                    if not node.parent_id and node.agent_id != topology._root_id:
                        # Assign unparented nodes to root
                        node.parent_id = topology._root_id
                        affected.append(node.agent_id)

            # Strengthen territory boundaries
            if hasattr(topology, "_territories"):
                for territory in topology._territories.values():
                    territory.stability = min(1.0, territory.stability + 0.1)

        operation = MachineOperation(
            operation_type=OperationType.STRIATE,
            machine_type=MachineType.STATE,
            target_agents=affected,
            success=len(affected) > 0,
            metadata={"region": region},
        )
        self._operations.append(operation)

        # Update state
        self._state.striation_level = min(1.0, self._state.striation_level + 0.1)

        logger.info(f"Striate operation: {len(affected)} agents affected")
        return operation

    async def overcoding_operation(
        self,
        rules: dict[str, Any],
    ) -> MachineOperation:
        """Execute an overcoding operation (State machine).

        Apply meta-rules that govern all other rules.

        Args:
            rules: Rules to impose.

        Returns:
            MachineOperation record.
        """
        affected: list[str] = []

        for topology in self._registered_topologies.values():
            if hasattr(topology, "nodes"):
                for node in topology.nodes.values():
                    # Apply role constraints
                    if "allowed_roles" in rules:
                        if node.role not in rules["allowed_roles"]:
                            node.role = rules["allowed_roles"][0]
                            affected.append(node.agent_id)

                    # Apply metadata constraints
                    if "required_metadata" in rules:
                        for key, value in rules["required_metadata"].items():
                            node.metadata[key] = value
                        affected.append(node.agent_id)

        operation = MachineOperation(
            operation_type=OperationType.OVERCODING,
            machine_type=MachineType.STATE,
            target_agents=affected,
            success=len(affected) > 0,
            metadata={"rules": rules},
        )
        self._operations.append(operation)

        logger.info(f"Overcoding operation: {len(affected)} agents affected")
        return operation

    # =========================================================================
    # War Machine Operations
    # =========================================================================

    async def smooth_operation(
        self,
        territory_id: str | None = None,
    ) -> MachineOperation:
        """Execute a smoothing operation (War machine).

        Remove striations and create smooth, unstructured space.

        Args:
            territory_id: Optional territory to smooth.

        Returns:
            MachineOperation record.
        """
        affected: list[str] = []

        for topology in self._registered_topologies.values():
            # Weaken territory boundaries
            if hasattr(topology, "_territories"):
                territories = topology._territories
                if territory_id and territory_id in territories:
                    territories[territory_id].stability *= 0.5
                    affected.extend(territories[territory_id].agent_ids)
                else:
                    for territory in territories.values():
                        territory.stability *= 0.8

            # Rupture some connections
            if hasattr(topology, "rupture") and hasattr(topology, "nodes"):
                nodes = list(topology.nodes.keys())
                if nodes:
                    # Rupture a random node
                    target = random.choice(nodes)
                    disconnected = topology.rupture(target)
                    affected.extend(disconnected)

        operation = MachineOperation(
            operation_type=OperationType.SMOOTH,
            machine_type=MachineType.WAR,
            target_agents=affected,
            success=len(affected) > 0,
            metadata={"territory": territory_id},
        )
        self._operations.append(operation)

        # Update state
        self._state.striation_level = max(0.0, self._state.striation_level - 0.1)
        self._state.war_intensity = min(1.0, self._state.war_intensity + 0.1)

        logger.info(f"Smooth operation: {len(affected)} agents affected")
        return operation

    async def line_of_flight(
        self,
        agent_id: str,
        escape_vector: str,
        duration: float = 300.0,
    ) -> MachineOperation:
        """Enable a line of flight for an agent (War machine).

        Allow an agent to escape fixed structures and operate nomadically.

        Args:
            agent_id: Agent initiating flight.
            escape_vector: Direction/purpose of escape.
            duration: Duration of escape in seconds.

        Returns:
            MachineOperation record.
        """
        success = False

        for topology in self._registered_topologies.values():
            if hasattr(topology, "initiate_line_of_flight"):
                if topology.initiate_line_of_flight(agent_id, escape_vector, duration):
                    success = True
                    break

            # Alternative: remove from territory
            if hasattr(topology, "_agent_territory"):
                if agent_id in topology._agent_territory:
                    territory_id = topology._agent_territory[agent_id]
                    if territory_id in topology._territories:
                        topology._territories[territory_id].remove_agent(agent_id)
                        del topology._agent_territory[agent_id]
                        success = True
                        break

        operation = MachineOperation(
            operation_type=OperationType.LINE_OF_FLIGHT,
            machine_type=MachineType.WAR,
            target_agents=[agent_id],
            success=success,
            metadata={"escape_vector": escape_vector, "duration": duration},
        )
        self._operations.append(operation)

        # Update state
        if success:
            self._state.escape_rate = min(1.0, self._state.escape_rate + 0.1)
            self._state.war_intensity = min(1.0, self._state.war_intensity + 0.05)

        logger.info(f"Line of flight: {agent_id} -> {escape_vector} (success={success})")
        return operation

    async def nomadize_operation(
        self,
        target_agents: list[str] | None = None,
    ) -> MachineOperation:
        """Enable nomadic movement for agents (War machine).

        Remove fixed positions and enable fluid role-shifting.

        Args:
            target_agents: Specific agents to nomadize, or all if None.

        Returns:
            MachineOperation record.
        """
        affected: list[str] = []

        for topology in self._registered_topologies.values():
            if hasattr(topology, "flux_cycle"):
                # Trigger role reassignment
                changes = await topology.flux_cycle()
                affected.extend(changes.keys())

            if hasattr(topology, "nodes"):
                agents = target_agents or list(topology.nodes.keys())
                for agent_id in agents:
                    node = topology.nodes.get(agent_id)
                    if node:
                        # Remove fixed position markers
                        node.metadata.pop("fixed_position", None)
                        node.metadata.pop("locked", None)
                        affected.append(agent_id)

        operation = MachineOperation(
            operation_type=OperationType.NOMADIZE,
            machine_type=MachineType.WAR,
            target_agents=list(set(affected)),
            success=len(affected) > 0,
            metadata={},
        )
        self._operations.append(operation)

        logger.info(f"Nomadize operation: {len(affected)} agents affected")
        return operation

    async def deterritorialize_operation(
        self,
        territory_id: str,
    ) -> MachineOperation:
        """Execute a deterritorialization operation (War machine).

        Break down territory boundaries and release agents.

        Args:
            territory_id: Territory to deterritorialize.

        Returns:
            MachineOperation record.
        """
        affected: list[str] = []

        for topology in self._registered_topologies.values():
            if hasattr(topology, "dissolve_territory"):
                released = topology.dissolve_territory(territory_id)
                affected.extend(released)
            elif hasattr(topology, "_territories"):
                if territory_id in topology._territories:
                    territory = topology._territories[territory_id]
                    affected.extend(territory.agent_ids)
                    # Dramatically reduce stability
                    territory.stability = 0.0

        operation = MachineOperation(
            operation_type=OperationType.DETERRITORIALIZE,
            machine_type=MachineType.WAR,
            target_agents=affected,
            success=len(affected) > 0,
            metadata={"territory": territory_id},
        )
        self._operations.append(operation)

        # Update state
        self._state.striation_level = max(0.0, self._state.striation_level - 0.2)
        self._state.state_intensity = max(0.0, self._state.state_intensity - 0.1)

        logger.info(f"Deterritorialize operation: {len(affected)} agents released")
        return operation

    # =========================================================================
    # Balance Management
    # =========================================================================

    def adjust_balance(self, delta: float) -> float:
        """Adjust the state/war balance.

        Args:
            delta: Amount to adjust (-1 to 1 range).

        Returns:
            New balance.
        """
        # Adjust intensities
        if delta > 0:
            self._state.state_intensity = min(1.0, self._state.state_intensity + delta)
            self._state.war_intensity = max(0.0, self._state.war_intensity - delta * 0.5)
        else:
            self._state.war_intensity = min(1.0, self._state.war_intensity - delta)
            self._state.state_intensity = max(0.0, self._state.state_intensity + delta * 0.5)

        return self._state.balance

    def get_operations_history(self, limit: int = 50) -> list[MachineOperation]:
        """Get recent operations history.

        Args:
            limit: Maximum number of operations.

        Returns:
            List of recent MachineOperation objects.
        """
        return self._operations[-limit:]

    def get_operation_stats(self) -> dict[str, Any]:
        """Get statistics about operations.

        Returns:
            Dictionary with operation statistics.
        """
        state_ops = [o for o in self._operations if o.machine_type == MachineType.STATE]
        war_ops = [o for o in self._operations if o.machine_type == MachineType.WAR]

        return {
            "total_operations": len(self._operations),
            "state_operations": len(state_ops),
            "war_operations": len(war_ops),
            "state_success_rate": (
                sum(1 for o in state_ops if o.success) / len(state_ops) if state_ops else 0.0
            ),
            "war_success_rate": (
                sum(1 for o in war_ops if o.success) / len(war_ops) if war_ops else 0.0
            ),
            "current_balance": self._state.balance,
            "dominant_machine": self.dominant_machine.value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "state": {
                "dominant_machine": self._state.dominant_machine.value,
                "state_intensity": self._state.state_intensity,
                "war_intensity": self._state.war_intensity,
                "capture_rate": self._state.capture_rate,
                "escape_rate": self._state.escape_rate,
                "striation_level": self._state.striation_level,
                "balance": self._state.balance,
            },
            "operations": [o.to_dict() for o in self._operations[-20:]],
            "stats": self.get_operation_stats(),
        }
