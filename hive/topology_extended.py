"""Extended Topology Types inspired by Deleuze & Guattari.

Implements four new topology types based on philosophical concepts:
- Rhizomatic: Any-to-any, no hierarchy, heterogeneous connections
- Arborescent: Strict tree, central control (extends Hierarchy)
- Territorialized: Bounded domains with controlled crossing
- Deterritorialized: Fluid, role-shifting, lines of flight

These topologies enable dynamic restructuring between structure and
emergence, capturing the tension between State and War Machine.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from hive.topology import AgentNode, BaseTopology, TopologyType

if TYPE_CHECKING:
    from hive.events import EventBus

logger = logging.getLogger("titan.hive.topology_extended")


# Extend TopologyType with new values
class ExtendedTopologyType(str, Enum):
    """Extended topology types including D&G-inspired patterns."""

    # Original types
    SWARM = "swarm"
    HIERARCHY = "hierarchy"
    PIPELINE = "pipeline"
    MESH = "mesh"
    RING = "ring"
    STAR = "star"

    # New D&G-inspired types
    RHIZOMATIC = "rhizomatic"        # Any-to-any, no hierarchy
    ARBORESCENT = "arborescent"      # Strict tree, central control
    TERRITORIALIZED = "territorialized"  # Bounded domains
    DETERRITORIALIZED = "deterritorialized"  # Fluid, role-shifting


class ConnectionType(str, Enum):
    """Types of connections in extended topologies."""

    PERMANENT = "permanent"  # Stable connection
    TEMPORARY = "temporary"  # Can be dissolved
    POTENTIAL = "potential"  # Not yet active
    RUPTURED = "ruptured"    # Broken connection


@dataclass
class Connection:
    """A connection between two agents in a topology."""

    from_agent: str
    to_agent: str
    connection_type: ConnectionType = ConnectionType.PERMANENT
    strength: float = 1.0  # 0-1, how strong the connection is
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def weaken(self, amount: float = 0.1) -> None:
        """Weaken the connection."""
        self.strength = max(0.0, self.strength - amount)

    def strengthen(self, amount: float = 0.1) -> None:
        """Strengthen the connection."""
        self.strength = min(1.0, self.strength + amount)


@dataclass
class Territory:
    """A bounded domain in a territorialized topology."""

    territory_id: str
    name: str
    agent_ids: list[str] = field(default_factory=list)
    boundary_agents: list[str] = field(default_factory=list)  # Can cross boundaries
    connections_to: list[str] = field(default_factory=list)  # Other territory IDs
    stability: float = 1.0  # 0-1, how stable the territory is
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_agent(self, agent_id: str, is_boundary: bool = False) -> None:
        """Add an agent to the territory."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
        if is_boundary and agent_id not in self.boundary_agents:
            self.boundary_agents.append(agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the territory."""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
        if agent_id in self.boundary_agents:
            self.boundary_agents.remove(agent_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "territory_id": self.territory_id,
            "name": self.name,
            "agent_ids": self.agent_ids,
            "boundary_agents": self.boundary_agents,
            "connections_to": self.connections_to,
            "stability": self.stability,
            "metadata": self.metadata,
        }


class RhizomaticTopology(BaseTopology):
    """Rhizomatic topology: any-to-any, no hierarchy, heterogeneous connections.

    Based on Deleuze & Guattari's concept of the rhizome:
    - Principles of connection and heterogeneity
    - Any point can connect to any other
    - No central authority
    - Multiple entry and exit points
    - Capable of rupture and repair
    """

    topology_type = TopologyType.MESH  # Closest existing type

    def __init__(
        self,
        max_connections_per_agent: int = 10,
        connection_decay_rate: float = 0.05,
    ) -> None:
        """Initialize rhizomatic topology.

        Args:
            max_connections_per_agent: Maximum connections each agent can have.
            connection_decay_rate: Rate at which unused connections weaken.
        """
        super().__init__()
        self._max_connections = max_connections_per_agent
        self._decay_rate = connection_decay_rate
        self._connections: dict[str, list[Connection]] = {}
        self._entry_points: list[str] = []  # Agents that can receive external input
        self._exit_points: list[str] = []   # Agents that can produce output

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        is_entry: bool = False,
        is_exit: bool = False,
        **kwargs: Any,
    ) -> AgentNode:
        """Add an agent to the rhizome."""
        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            metadata=kwargs.get("metadata", {}),
        )
        self.nodes[agent_id] = node
        self._connections[agent_id] = []

        if is_entry:
            self._entry_points.append(agent_id)
        if is_exit:
            self._exit_points.append(agent_id)

        # Form initial connections based on capability overlap
        self._form_heterogeneous_connections(agent_id)

        logger.debug(f"Added agent to rhizome: {agent_id}")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the rhizome."""
        if agent_id not in self.nodes:
            return False

        # Remove all connections involving this agent
        del self.nodes[agent_id]
        if agent_id in self._connections:
            del self._connections[agent_id]

        # Remove connections to this agent from others
        for conns in self._connections.values():
            conns[:] = [c for c in conns if c.to_agent != agent_id]

        if agent_id in self._entry_points:
            self._entry_points.remove(agent_id)
        if agent_id in self._exit_points:
            self._exit_points.remove(agent_id)

        return True

    def _form_heterogeneous_connections(self, agent_id: str) -> None:
        """Form connections based on heterogeneity principle."""
        agent = self.nodes.get(agent_id)
        if not agent:
            return

        # Connect to agents with different capabilities (heterogeneity)
        for other_id, other in self.nodes.items():
            if other_id == agent_id:
                continue

            # Favor connections to agents with different capabilities
            overlap = len(set(agent.capabilities) & set(other.capabilities))
            difference = len(set(agent.capabilities) ^ set(other.capabilities))

            if difference > overlap:
                self.connect(agent_id, other_id, strength=0.5)

    def connect(
        self,
        from_agent: str,
        to_agent: str,
        strength: float = 1.0,
        connection_type: ConnectionType = ConnectionType.TEMPORARY,
    ) -> Connection | None:
        """Create a connection between two agents."""
        if from_agent not in self.nodes or to_agent not in self.nodes:
            return None

        # Check connection limit
        current_connections = self._connections.get(from_agent, [])
        if len(current_connections) >= self._max_connections:
            # Remove weakest connection
            current_connections.sort(key=lambda c: c.strength)
            if current_connections and current_connections[0].strength < strength:
                current_connections.pop(0)
            else:
                return None

        connection = Connection(
            from_agent=from_agent,
            to_agent=to_agent,
            connection_type=connection_type,
            strength=strength,
        )

        if from_agent not in self._connections:
            self._connections[from_agent] = []
        self._connections[from_agent].append(connection)

        # Update node neighbors
        if to_agent not in self.nodes[from_agent].neighbors:
            self.nodes[from_agent].neighbors.append(to_agent)

        return connection

    def disconnect(self, from_agent: str, to_agent: str) -> bool:
        """Remove a connection between two agents."""
        if from_agent not in self._connections:
            return False

        conns = self._connections[from_agent]
        for i, conn in enumerate(conns):
            if conn.to_agent == to_agent:
                conns.pop(i)
                if to_agent in self.nodes[from_agent].neighbors:
                    self.nodes[from_agent].neighbors.remove(to_agent)
                return True

        return False

    def rupture(self, agent_id: str) -> list[str]:
        """Rupture connections at a node (rhizome can break).

        Args:
            agent_id: Agent at which to rupture connections.

        Returns:
            List of agent IDs that were disconnected.
        """
        disconnected: list[str] = []

        if agent_id in self._connections:
            for conn in self._connections[agent_id]:
                conn.connection_type = ConnectionType.RUPTURED
                disconnected.append(conn.to_agent)

        # Also rupture incoming connections
        for conns in self._connections.values():
            for conn in conns:
                if conn.to_agent == agent_id:
                    conn.connection_type = ConnectionType.RUPTURED

        logger.info(f"Ruptured connections at {agent_id}: {len(disconnected)} affected")
        return disconnected

    def repair(self, agent_id: str) -> int:
        """Repair ruptured connections at a node.

        Returns:
            Number of connections repaired.
        """
        repaired = 0

        if agent_id in self._connections:
            for conn in self._connections[agent_id]:
                if conn.connection_type == ConnectionType.RUPTURED:
                    conn.connection_type = ConnectionType.TEMPORARY
                    conn.strength = 0.5  # Start at reduced strength
                    repaired += 1

        return repaired

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        """Get agents to receive message.

        In rhizomatic topology, messages spread through connections.
        """
        if message_type == "broadcast":
            # All connected agents
            targets: set[str] = set()
            conns = self._connections.get(source_agent_id, [])
            for conn in conns:
                if conn.connection_type != ConnectionType.RUPTURED:
                    targets.add(conn.to_agent)
            return list(targets)

        elif message_type == "entry":
            return self._entry_points

        elif message_type == "exit":
            return self._exit_points

        else:
            # Direct connections only
            return [
                c.to_agent
                for c in self._connections.get(source_agent_id, [])
                if c.connection_type != ConnectionType.RUPTURED
            ]

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        """Get path between agents (BFS through connections)."""
        if source_agent_id == target_agent_id:
            return [source_agent_id]

        visited: set[str] = set()
        queue = [(source_agent_id, [source_agent_id])]

        while queue:
            current, path = queue.pop(0)
            if current == target_agent_id:
                return path

            if current in visited:
                continue
            visited.add(current)

            for conn in self._connections.get(current, []):
                if conn.connection_type != ConnectionType.RUPTURED:
                    if conn.to_agent not in visited:
                        queue.append((conn.to_agent, path + [conn.to_agent]))

        return []  # No path found

    def decay_connections(self) -> int:
        """Apply decay to all connections.

        Returns:
            Number of connections removed due to decay.
        """
        removed = 0

        for agent_id in list(self._connections.keys()):
            conns = self._connections[agent_id]
            remaining: list[Connection] = []

            for conn in conns:
                conn.weaken(self._decay_rate)
                if conn.strength > 0.1:
                    remaining.append(conn)
                else:
                    removed += 1

            self._connections[agent_id] = remaining

        return removed


class ArborealTopology(BaseTopology):
    """Arborescent (tree) topology: strict hierarchy, central control.

    Based on Deleuze & Guattari's critique of arborescent structures:
    - Single root with branching structure
    - Command flows downward
    - Reports flow upward
    - Strict parent-child relationships
    - State-like organization
    """

    topology_type = TopologyType.HIERARCHY

    def __init__(self, root_id: str | None = None) -> None:
        """Initialize arborescent topology.

        Args:
            root_id: ID of the root agent (if pre-assigned).
        """
        super().__init__()
        self._root_id = root_id
        self._depth: dict[str, int] = {}  # Agent -> depth in tree

    @property
    def root(self) -> AgentNode | None:
        """Get the root node."""
        if self._root_id:
            return self.nodes.get(self._root_id)
        return None

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        parent_id: str | None = None,
        **kwargs: Any,
    ) -> AgentNode:
        """Add an agent to the tree."""
        # First agent becomes root if no root set
        if not self._root_id and not self.nodes:
            self._root_id = agent_id
            parent_id = None

        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            parent_id=parent_id,
            role=kwargs.get("role", "worker" if parent_id else "root"),
            metadata=kwargs.get("metadata", {}),
        )
        self.nodes[agent_id] = node

        # Update parent's children
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].child_ids.append(agent_id)
            self._depth[agent_id] = self._depth.get(parent_id, 0) + 1
        else:
            self._depth[agent_id] = 0

        logger.debug(f"Added agent to tree: {agent_id} (depth={self._depth[agent_id]})")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the tree.

        Children are re-parented to the removed agent's parent.
        """
        node = self.nodes.get(agent_id)
        if not node:
            return False

        # Re-parent children
        new_parent = node.parent_id
        for child_id in node.child_ids:
            child = self.nodes.get(child_id)
            if child:
                child.parent_id = new_parent
                if new_parent and new_parent in self.nodes:
                    self.nodes[new_parent].child_ids.append(child_id)

        # Remove from parent's children
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if agent_id in parent.child_ids:
                parent.child_ids.remove(agent_id)

        del self.nodes[agent_id]
        del self._depth[agent_id]

        # Update root if needed
        if agent_id == self._root_id:
            if node.child_ids:
                self._root_id = node.child_ids[0]
            else:
                self._root_id = None

        return True

    def dispatch_command(
        self,
        from_agent: str,
        command: dict[str, Any],
    ) -> list[str]:
        """Dispatch a command downward in the tree.

        Commands flow from parent to children.

        Returns:
            List of agent IDs that received the command.
        """
        node = self.nodes.get(from_agent)
        if not node:
            return []

        recipients = list(node.child_ids)
        logger.debug(f"Dispatched command from {from_agent} to {len(recipients)} children")
        return recipients

    def report_up(
        self,
        from_agent: str,
        report: dict[str, Any],
    ) -> str | None:
        """Report upward in the tree.

        Reports flow from child to parent.

        Returns:
            Parent agent ID, or None if at root.
        """
        node = self.nodes.get(from_agent)
        if not node or not node.parent_id:
            return None

        return node.parent_id

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        """Get message targets based on tree structure."""
        node = self.nodes.get(source_agent_id)
        if not node:
            return []

        if message_type == "command":
            # Downward to all descendants
            descendants: list[str] = []
            to_visit = list(node.child_ids)
            while to_visit:
                child_id = to_visit.pop(0)
                descendants.append(child_id)
                child = self.nodes.get(child_id)
                if child:
                    to_visit.extend(child.child_ids)
            return descendants

        elif message_type == "report":
            # Upward to ancestors
            ancestors: list[str] = []
            current = node
            while current.parent_id:
                ancestors.append(current.parent_id)
                current = self.nodes.get(current.parent_id)
                if not current:
                    break
            return ancestors

        elif message_type == "broadcast":
            # All other nodes
            return [aid for aid in self.nodes if aid != source_agent_id]

        else:
            # Direct children only
            return list(node.child_ids)

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        """Get path between agents through the tree."""
        if source_agent_id == target_agent_id:
            return [source_agent_id]

        # Find common ancestor
        source_ancestors = self._get_ancestors(source_agent_id)
        target_ancestors = self._get_ancestors(target_agent_id)

        # Path: source -> common ancestor -> target
        common = None
        for ancestor in source_ancestors:
            if ancestor in target_ancestors:
                common = ancestor
                break

        if not common:
            # No common ancestor (disconnected)
            return []

        # Build path
        path_up = []
        current = source_agent_id
        while current != common:
            path_up.append(current)
            node = self.nodes.get(current)
            if not node or not node.parent_id:
                break
            current = node.parent_id

        path_down = []
        current = target_agent_id
        while current != common:
            path_down.append(current)
            node = self.nodes.get(current)
            if not node or not node.parent_id:
                break
            current = node.parent_id

        return path_up + [common] + list(reversed(path_down))

    def _get_ancestors(self, agent_id: str) -> list[str]:
        """Get all ancestors of an agent."""
        ancestors: list[str] = []
        node = self.nodes.get(agent_id)
        while node:
            ancestors.append(node.agent_id)
            if not node.parent_id:
                break
            node = self.nodes.get(node.parent_id)
        return ancestors

    def get_subtree_agents(self, root_agent: str) -> list[str]:
        """Get all agents in a subtree."""
        agents: list[str] = []
        to_visit = [root_agent]

        while to_visit:
            current = to_visit.pop(0)
            agents.append(current)
            node = self.nodes.get(current)
            if node:
                to_visit.extend(node.child_ids)

        return agents


class TerritorializedTopology(BaseTopology):
    """Territorialized topology: bounded domains with controlled crossing.

    Based on Deleuze & Guattari's concept of territorialization:
    - Space is divided into territories
    - Each territory has its own internal organization
    - Boundary agents can cross between territories
    - Communication between territories is controlled
    """

    topology_type = TopologyType.MESH

    def __init__(self) -> None:
        """Initialize territorialized topology."""
        super().__init__()
        self._territories: dict[str, Territory] = {}
        self._agent_territory: dict[str, str] = {}  # Agent -> territory ID

    def create_territory(
        self,
        name: str,
        territory_id: str | None = None,
    ) -> Territory:
        """Create a new territory."""
        if territory_id is None:
            territory_id = f"territory_{uuid.uuid4().hex[:8]}"

        territory = Territory(
            territory_id=territory_id,
            name=name,
        )
        self._territories[territory_id] = territory

        logger.info(f"Created territory: {name} ({territory_id})")
        return territory

    def dissolve_territory(self, territory_id: str) -> list[str]:
        """Dissolve a territory, returning agents to unassigned state.

        Returns:
            List of agent IDs that were in the territory.
        """
        territory = self._territories.get(territory_id)
        if not territory:
            return []

        agents = list(territory.agent_ids)
        for agent_id in agents:
            if agent_id in self._agent_territory:
                del self._agent_territory[agent_id]

        del self._territories[territory_id]

        logger.info(f"Dissolved territory: {territory_id}")
        return agents

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        territory_id: str | None = None,
        is_boundary: bool = False,
        **kwargs: Any,
    ) -> AgentNode:
        """Add an agent to a territory."""
        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role="boundary" if is_boundary else "internal",
            metadata=kwargs.get("metadata", {}),
        )
        self.nodes[agent_id] = node

        if territory_id and territory_id in self._territories:
            self._territories[territory_id].add_agent(agent_id, is_boundary)
            self._agent_territory[agent_id] = territory_id

        return node

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the topology."""
        if agent_id not in self.nodes:
            return False

        # Remove from territory
        if agent_id in self._agent_territory:
            territory_id = self._agent_territory[agent_id]
            if territory_id in self._territories:
                self._territories[territory_id].remove_agent(agent_id)
            del self._agent_territory[agent_id]

        del self.nodes[agent_id]
        return True

    def assign_to_territory(
        self,
        agent_id: str,
        territory_id: str,
        is_boundary: bool = False,
    ) -> bool:
        """Assign an agent to a territory."""
        if agent_id not in self.nodes:
            return False
        if territory_id not in self._territories:
            return False

        # Remove from current territory
        if agent_id in self._agent_territory:
            old_territory = self._agent_territory[agent_id]
            if old_territory in self._territories:
                self._territories[old_territory].remove_agent(agent_id)

        # Add to new territory
        self._territories[territory_id].add_agent(agent_id, is_boundary)
        self._agent_territory[agent_id] = territory_id
        self.nodes[agent_id].role = "boundary" if is_boundary else "internal"

        return True

    def connect_territories(
        self,
        territory_a: str,
        territory_b: str,
    ) -> bool:
        """Create a connection between two territories."""
        if territory_a not in self._territories or territory_b not in self._territories:
            return False

        if territory_b not in self._territories[territory_a].connections_to:
            self._territories[territory_a].connections_to.append(territory_b)
        if territory_a not in self._territories[territory_b].connections_to:
            self._territories[territory_b].connections_to.append(territory_a)

        return True

    def can_communicate(self, agent_a: str, agent_b: str) -> bool:
        """Check if two agents can communicate."""
        territory_a = self._agent_territory.get(agent_a)
        territory_b = self._agent_territory.get(agent_b)

        # Same territory
        if territory_a == territory_b:
            return True

        # No territory assigned
        if not territory_a or not territory_b:
            return True

        # Check if territories are connected and one agent is boundary
        t_a = self._territories.get(territory_a)
        t_b = self._territories.get(territory_b)

        if not t_a or not t_b:
            return False

        connected = territory_b in t_a.connections_to

        # At least one must be boundary agent
        is_boundary_a = agent_a in t_a.boundary_agents
        is_boundary_b = agent_b in t_b.boundary_agents

        return connected and (is_boundary_a or is_boundary_b)

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        """Get message targets respecting territory boundaries."""
        if message_type == "territory":
            # Only agents in same territory
            territory_id = self._agent_territory.get(source_agent_id)
            if not territory_id or territory_id not in self._territories:
                return []
            return [
                aid for aid in self._territories[territory_id].agent_ids
                if aid != source_agent_id
            ]

        elif message_type == "broadcast":
            # All agents that can be communicated with
            return [
                aid for aid in self.nodes
                if aid != source_agent_id and self.can_communicate(source_agent_id, aid)
            ]

        elif message_type == "cross_territory":
            # Only boundary agents in connected territories
            territory_id = self._agent_territory.get(source_agent_id)
            if not territory_id:
                return []

            territory = self._territories.get(territory_id)
            if not territory:
                return []

            targets: list[str] = []
            for connected_id in territory.connections_to:
                connected = self._territories.get(connected_id)
                if connected:
                    targets.extend(connected.boundary_agents)

            return targets

        else:
            return []

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        """Get path between agents, potentially crossing territories."""
        if source_agent_id == target_agent_id:
            return [source_agent_id]

        if not self.can_communicate(source_agent_id, target_agent_id):
            return []  # Cannot communicate

        # Simple path: source -> [boundary] -> target
        territory_s = self._agent_territory.get(source_agent_id)
        territory_t = self._agent_territory.get(target_agent_id)

        if territory_s == territory_t or not territory_s or not territory_t:
            return [source_agent_id, target_agent_id]

        # Need to go through boundary agents
        t_s = self._territories.get(territory_s)
        t_t = self._territories.get(territory_t)

        if not t_s or not t_t:
            return []

        # Find boundary path
        path = [source_agent_id]

        # If source is not boundary, go to boundary first
        if source_agent_id not in t_s.boundary_agents and t_s.boundary_agents:
            path.append(t_s.boundary_agents[0])

        # Cross to target territory boundary
        if t_t.boundary_agents:
            path.append(t_t.boundary_agents[0])

        # If target is not the boundary, add target
        if target_agent_id not in path:
            path.append(target_agent_id)

        return path

    def get_territories(self) -> list[Territory]:
        """Get all territories."""
        return list(self._territories.values())

    def get_agent_territory(self, agent_id: str) -> Territory | None:
        """Get the territory an agent belongs to."""
        territory_id = self._agent_territory.get(agent_id)
        if territory_id:
            return self._territories.get(territory_id)
        return None


class DeterritorializedTopology(BaseTopology):
    """Deterritorialized topology: fluid, role-shifting, lines of flight.

    Based on Deleuze & Guattari's concept of deterritorialization:
    - Roles are constantly reassigned
    - Agents can escape fixed positions (lines of flight)
    - Continuous flux between structure and chaos
    - War machine dynamics
    """

    topology_type = TopologyType.SWARM

    def __init__(
        self,
        flux_interval: float = 60.0,
        role_volatility: float = 0.3,
    ) -> None:
        """Initialize deterritorialized topology.

        Args:
            flux_interval: Seconds between role reassignment cycles.
            role_volatility: Probability of role change per cycle (0-1).
        """
        super().__init__()
        self._flux_interval = flux_interval
        self._role_volatility = role_volatility
        self._available_roles: list[str] = [
            "scout", "worker", "coordinator", "synthesizer",
            "critic", "explorer", "builder", "analyst",
        ]
        self._lines_of_flight: dict[str, dict[str, Any]] = {}  # Agent -> escape data
        self._flux_task: asyncio.Task[None] | None = None
        self._running = False

    async def start_flux(self) -> None:
        """Start the flux cycle for role reassignment."""
        if self._running:
            return

        self._running = True
        self._flux_task = asyncio.create_task(self._flux_loop())
        logger.info("Started deterritorialized flux cycle")

    async def stop_flux(self) -> None:
        """Stop the flux cycle."""
        self._running = False
        if self._flux_task:
            self._flux_task.cancel()
            try:
                await self._flux_task
            except asyncio.CancelledError:
                pass

    async def _flux_loop(self) -> None:
        """Background loop for role reassignment."""
        while self._running:
            try:
                await asyncio.sleep(self._flux_interval)
                await self.flux_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flux loop: {e}")

    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: list[str],
        initial_role: str | None = None,
        **kwargs: Any,
    ) -> AgentNode:
        """Add an agent to the fluid topology."""
        role = initial_role or random.choice(self._available_roles)

        node = AgentNode(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            role=role,
            metadata=kwargs.get("metadata", {}),
        )
        self.nodes[agent_id] = node

        logger.debug(f"Added agent to deterritorialized: {agent_id} (role={role})")
        return node

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the topology."""
        if agent_id not in self.nodes:
            return False

        del self.nodes[agent_id]
        if agent_id in self._lines_of_flight:
            del self._lines_of_flight[agent_id]

        return True

    async def flux_cycle(self) -> dict[str, str]:
        """Perform role reassignment cycle.

        Returns:
            Dict mapping agent_id -> new role for changed agents.
        """
        changes: dict[str, str] = {}

        for agent_id, node in self.nodes.items():
            # Skip agents on lines of flight
            if agent_id in self._lines_of_flight:
                continue

            # Probabilistic role change
            if random.random() < self._role_volatility:
                # Select new role (different from current)
                available = [r for r in self._available_roles if r != node.role]
                if available:
                    new_role = random.choice(available)
                    node.role = new_role
                    changes[agent_id] = new_role

        if changes:
            logger.info(f"Flux cycle changed {len(changes)} roles")

        return changes

    def initiate_line_of_flight(
        self,
        agent_id: str,
        escape_vector: str,
        duration_seconds: float = 300.0,
    ) -> bool:
        """Allow an agent to escape fixed roles (line of flight).

        During escape, the agent operates outside normal topology rules.

        Args:
            agent_id: Agent initiating escape.
            escape_vector: Direction/purpose of escape.
            duration_seconds: How long the escape lasts.

        Returns:
            True if escape initiated, False if agent not found.
        """
        if agent_id not in self.nodes:
            return False

        self._lines_of_flight[agent_id] = {
            "escape_vector": escape_vector,
            "started_at": datetime.now(timezone.utc),
            "duration": duration_seconds,
            "original_role": self.nodes[agent_id].role,
        }

        self.nodes[agent_id].role = "escaped"
        logger.info(f"Agent {agent_id} initiated line of flight: {escape_vector}")

        return True

    def capture(self, agent_id: str, new_role: str | None = None) -> bool:
        """Capture an escaped agent back into the topology.

        Args:
            agent_id: Agent to capture.
            new_role: Role to assign (or restore original).

        Returns:
            True if captured, False if not in flight.
        """
        if agent_id not in self._lines_of_flight:
            return False

        flight_data = self._lines_of_flight[agent_id]
        del self._lines_of_flight[agent_id]

        if new_role:
            self.nodes[agent_id].role = new_role
        else:
            self.nodes[agent_id].role = flight_data.get(
                "original_role", random.choice(self._available_roles)
            )

        logger.info(f"Captured agent {agent_id} (role={self.nodes[agent_id].role})")
        return True

    def check_escaped_agents(self) -> list[str]:
        """Check for agents whose escape duration has expired.

        Returns:
            List of agent IDs that should be captured.
        """
        now = datetime.now(timezone.utc)
        expired: list[str] = []

        for agent_id, flight_data in list(self._lines_of_flight.items()):
            started = flight_data["started_at"]
            duration = flight_data["duration"]
            if (now - started).total_seconds() > duration:
                expired.append(agent_id)

        return expired

    def get_agents_by_role(self, role: str) -> list[str]:
        """Get all agents with a specific role."""
        return [
            aid for aid, node in self.nodes.items()
            if node.role == role
        ]

    def get_escaped_agents(self) -> list[str]:
        """Get all agents currently on lines of flight."""
        return list(self._lines_of_flight.keys())

    def get_message_targets(
        self,
        source_agent_id: str,
        message_type: str = "broadcast",
    ) -> list[str]:
        """Get message targets (all agents in swarm-like fashion)."""
        if message_type == "role":
            # Same role agents
            node = self.nodes.get(source_agent_id)
            if not node:
                return []
            return [
                aid for aid, n in self.nodes.items()
                if n.role == node.role and aid != source_agent_id
            ]

        elif message_type == "escaped":
            # Only escaped agents
            return self.get_escaped_agents()

        else:
            # All non-escaped agents
            return [
                aid for aid in self.nodes
                if aid != source_agent_id and aid not in self._lines_of_flight
            ]

    def get_routing_path(
        self,
        source_agent_id: str,
        target_agent_id: str,
    ) -> list[str]:
        """Get path between agents (direct in deterritorialized)."""
        if source_agent_id not in self.nodes or target_agent_id not in self.nodes:
            return []
        return [source_agent_id, target_agent_id]

    def add_role(self, role: str) -> None:
        """Add a new available role."""
        if role not in self._available_roles:
            self._available_roles.append(role)

    def remove_role(self, role: str) -> int:
        """Remove a role and reassign agents.

        Returns:
            Number of agents reassigned.
        """
        if role not in self._available_roles:
            return 0

        self._available_roles.remove(role)
        reassigned = 0

        for node in self.nodes.values():
            if node.role == role:
                node.role = random.choice(self._available_roles) if self._available_roles else "worker"
                reassigned += 1

        return reassigned
