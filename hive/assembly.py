"""Assembly Manager for Dynamic Coordination.

Manages transitions between territorialized and deterritorialized states,
evaluating stability and triggering structural changes based on
coordination needs.

Based on Deleuze & Guattari's concepts:
- Territorialization: Processes that stabilize the assemblage
- Deterritorialization: Processes that destabilize/transform
- Lines of flight: Escape from fixed structures
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hive.events import EventBus

logger = logging.getLogger("titan.hive.assembly")


class AssemblyState(StrEnum):
    """State of an assembly (assemblage)."""

    STABLE = "stable"  # Well-territorialized, functioning
    UNSTABLE = "unstable"  # Deterritorializing, in flux
    TRANSITIONING = "transitioning"  # Moving between states
    RUPTURED = "ruptured"  # Broken connections, needs repair
    CRYSTALLIZED = "crystallized"  # Over-territorialized, rigid


class TerritorizationType(StrEnum):
    """Types of territorialization processes."""

    CODING = "coding"  # Establishing rules and patterns
    STRATIFICATION = "stratification"  # Creating layers/hierarchy
    SEGMENTATION = "segmentation"  # Dividing into bounded regions
    CAPTURE = "capture"  # State capture of war machine


class DeterritorializationType(StrEnum):
    """Types of deterritorialization processes."""

    DECODING = "decoding"  # Breaking down rules
    SMOOTHING = "smoothing"  # Removing striations
    NOMADIZATION = "nomadization"  # Becoming mobile/fluid
    FLIGHT = "flight"  # Line of flight/escape


@dataclass
class AssemblyEvent:
    """Record of an assembly state change."""

    event_type: str
    previous_state: AssemblyState
    new_state: AssemblyState
    trigger: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    affected_agents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "affected_agents": self.affected_agents,
            "metadata": self.metadata,
        }


@dataclass
class StabilityMetrics:
    """Metrics for evaluating assembly stability."""

    connection_density: float = 0.0  # How connected agents are (0-1)
    role_consistency: float = 0.0  # How stable roles are (0-1)
    communication_flow: float = 0.0  # Information flow efficiency (0-1)
    defection_rate: float = 0.0  # Rate of agents leaving (0-1)
    task_completion_rate: float = 0.0  # Task success rate (0-1)
    coordination_overhead: float = 0.0  # Cost of coordination (0-1)

    @property
    def overall_stability(self) -> float:
        """Calculate overall stability score (0-1)."""
        positive = (
            self.connection_density * 0.2
            + self.role_consistency * 0.2
            + self.communication_flow * 0.2
            + self.task_completion_rate * 0.3
        )
        negative = self.defection_rate * 0.5 + self.coordination_overhead * 0.5
        return max(0.0, min(1.0, positive - negative * 0.3))

    @property
    def suggested_state(self) -> AssemblyState:
        """Suggest assembly state based on metrics."""
        stability = self.overall_stability

        if stability > 0.9:
            return AssemblyState.CRYSTALLIZED
        elif stability > 0.7:
            return AssemblyState.STABLE
        elif stability > 0.4:
            return AssemblyState.TRANSITIONING
        elif stability > 0.2:
            return AssemblyState.UNSTABLE
        else:
            return AssemblyState.RUPTURED


class AssemblyManager:
    """Manages assembly dynamics and structural transitions.

    Monitors assembly stability and triggers territorialization or
    deterritorialization as needed to maintain effective coordination.
    """

    # Thresholds for state transitions
    STABILITY_LOW_THRESHOLD = 0.3
    STABILITY_HIGH_THRESHOLD = 0.85
    DEFECTION_THRESHOLD = 0.2

    def __init__(
        self,
        assembly_id: str | None = None,
        evaluation_interval: float = 30.0,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the assembly manager.

        Args:
            assembly_id: Identifier for this assembly.
            evaluation_interval: Seconds between stability evaluations.
            event_bus: Optional event bus for publishing events.
        """
        import uuid

        self._assembly_id = assembly_id or str(uuid.uuid4())[:8]
        self._evaluation_interval = evaluation_interval
        self._event_bus = event_bus

        self._state = AssemblyState.STABLE
        self._metrics = StabilityMetrics()
        self._history: list[AssemblyEvent] = []
        self._topologies: dict[str, Any] = {}  # Tracked topologies

        self._eval_task: asyncio.Task[None] | None = None
        self._running = False

        # Tracking for metrics
        self._role_changes: list[datetime] = []
        self._defections: list[datetime] = []
        self._task_results: list[bool] = []

    @property
    def assembly_id(self) -> str:
        """Get the assembly ID."""
        return self._assembly_id

    @property
    def state(self) -> AssemblyState:
        """Get current assembly state."""
        return self._state

    @property
    def metrics(self) -> StabilityMetrics:
        """Get current stability metrics."""
        return self._metrics

    async def start(self) -> None:
        """Start the assembly manager."""
        if self._running:
            return

        self._running = True
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        logger.info(f"Started assembly manager: {self._assembly_id}")

    async def stop(self) -> None:
        """Stop the assembly manager."""
        self._running = False
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped assembly manager: {self._assembly_id}")

    async def _evaluation_loop(self) -> None:
        """Background loop for stability evaluation."""
        while self._running:
            try:
                await asyncio.sleep(self._evaluation_interval)
                await self._evaluate_and_respond()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")

    async def _evaluate_and_respond(self) -> None:
        """Evaluate stability and respond if needed."""
        await self.evaluate_stability()

        if self.should_territorialize():
            await self.territorialize()
        elif self.should_deterritorialize():
            await self.deterritorialize()

    def register_topology(self, topology_id: str, topology: Any) -> None:
        """Register a topology to be managed.

        Args:
            topology_id: Unique ID for the topology.
            topology: The topology instance.
        """
        self._topologies[topology_id] = topology

    def unregister_topology(self, topology_id: str) -> None:
        """Unregister a topology."""
        if topology_id in self._topologies:
            del self._topologies[topology_id]

    async def evaluate_stability(self) -> StabilityMetrics:
        """Evaluate current assembly stability.

        Returns:
            Updated StabilityMetrics.
        """
        now = datetime.now(UTC)
        window_seconds = 300.0  # 5 minute window

        # Calculate connection density from topologies
        total_agents = 0
        total_connections = 0
        for topology in self._topologies.values():
            if hasattr(topology, "nodes"):
                nodes = topology.nodes
                total_agents += len(nodes)
                for node in nodes.values():
                    if hasattr(node, "neighbors"):
                        total_connections += len(node.neighbors)
                    if hasattr(node, "child_ids"):
                        total_connections += len(node.child_ids)

        max_connections = total_agents * (total_agents - 1) if total_agents > 1 else 1
        self._metrics.connection_density = min(1.0, total_connections / max_connections)

        # Calculate role consistency from recent role changes
        recent_changes = [
            t for t in self._role_changes if (now - t).total_seconds() < window_seconds
        ]
        if total_agents > 0:
            change_rate = len(recent_changes) / (total_agents * (window_seconds / 60))
            self._metrics.role_consistency = max(0.0, 1.0 - change_rate)
        else:
            self._metrics.role_consistency = 1.0

        # Calculate defection rate
        recent_defections = [
            t for t in self._defections if (now - t).total_seconds() < window_seconds
        ]
        if total_agents > 0:
            self._metrics.defection_rate = len(recent_defections) / total_agents
        else:
            self._metrics.defection_rate = 0.0

        # Calculate task completion rate
        if self._task_results:
            recent_results = self._task_results[-50:]  # Last 50 tasks
            self._metrics.task_completion_rate = sum(recent_results) / len(recent_results)
        else:
            self._metrics.task_completion_rate = 0.5  # Neutral

        # Estimate communication flow (based on connectivity)
        self._metrics.communication_flow = (
            self._metrics.connection_density * 0.5 + self._metrics.role_consistency * 0.5
        )

        # Coordination overhead increases with complexity
        if total_agents > 0:
            self._metrics.coordination_overhead = min(1.0, total_agents / 100.0)
        else:
            self._metrics.coordination_overhead = 0.0

        # Update state based on suggested state
        suggested = self._metrics.suggested_state
        if suggested != self._state:
            await self._transition_state(suggested, "stability_evaluation")

        return self._metrics

    def should_territorialize(self) -> bool:
        """Determine if territorialization is needed.

        Territorialize when:
        - Too unstable (low stability)
        - High defection rate
        - Low task completion

        Returns:
            True if territorialization recommended.
        """
        if self._state in [AssemblyState.CRYSTALLIZED, AssemblyState.STABLE]:
            return False

        return (
            self._metrics.overall_stability < self.STABILITY_LOW_THRESHOLD
            or self._metrics.defection_rate > self.DEFECTION_THRESHOLD
            or self._metrics.task_completion_rate < 0.3
        )

    def should_deterritorialize(self) -> bool:
        """Determine if deterritorialization is needed.

        Deterritorialize when:
        - Over-structured (crystallized)
        - High coordination overhead
        - Need for innovation/exploration

        Returns:
            True if deterritorialization recommended.
        """
        if self._state in [AssemblyState.UNSTABLE, AssemblyState.RUPTURED]:
            return False

        return (
            self._state == AssemblyState.CRYSTALLIZED
            or self._metrics.coordination_overhead > 0.7
            or (
                self._metrics.overall_stability > self.STABILITY_HIGH_THRESHOLD
                and self._metrics.task_completion_rate < 0.5
            )
        )

    async def territorialize(
        self,
        ttype: TerritorizationType = TerritorizationType.CODING,
    ) -> bool:
        """Apply territorialization to stabilize the assembly.

        Args:
            ttype: Type of territorialization to apply.

        Returns:
            True if successful.
        """
        logger.info(f"Territorializing assembly: {self._assembly_id} ({ttype.value})")

        affected_agents: list[str] = []

        for topology in self._topologies.values():
            # Apply different territorialization strategies
            if ttype == TerritorizationType.CODING:
                # Establish clearer rules and patterns
                if hasattr(topology, "nodes"):
                    for node in topology.nodes.values():
                        if not node.role:
                            node.role = "worker"
                        affected_agents.append(node.agent_id)

            elif ttype == TerritorizationType.STRATIFICATION:
                # Create layers/hierarchy
                if hasattr(topology, "_root_id") and hasattr(topology, "add_agent"):
                    # Arborescent topology: reinforce hierarchy
                    pass

            elif ttype == TerritorizationType.SEGMENTATION:
                # Create bounded regions
                if hasattr(topology, "create_territory"):
                    # Territorialized topology: strengthen boundaries
                    pass

            elif ttype == TerritorizationType.CAPTURE:
                # Capture escaped agents
                if hasattr(topology, "capture") and hasattr(topology, "get_escaped_agents"):
                    escaped = topology.get_escaped_agents()
                    for agent_id in escaped:
                        topology.capture(agent_id)
                        affected_agents.append(agent_id)

        # Record event
        event = AssemblyEvent(
            event_type=f"territorialize_{ttype.value}",
            previous_state=self._state,
            new_state=AssemblyState.STABLE,
            trigger="stability_evaluation",
            affected_agents=affected_agents,
        )
        self._history.append(event)

        await self._transition_state(AssemblyState.TRANSITIONING, "territorialization")

        return True

    async def deterritorialize(
        self,
        dtype: DeterritorializationType = DeterritorializationType.DECODING,
    ) -> bool:
        """Apply deterritorialization to introduce fluidity.

        Args:
            dtype: Type of deterritorialization to apply.

        Returns:
            True if successful.
        """
        logger.info(f"Deterritorializing assembly: {self._assembly_id} ({dtype.value})")

        affected_agents: list[str] = []

        for topology in self._topologies.values():
            if dtype == DeterritorializationType.DECODING:
                # Relax rules
                if hasattr(topology, "nodes"):
                    for node in topology.nodes.values():
                        if node.role and node.role != "escaped":
                            affected_agents.append(node.agent_id)

            elif dtype == DeterritorializationType.SMOOTHING:
                # Remove striations
                if hasattr(topology, "rupture"):
                    # Break some connections
                    pass

            elif dtype == DeterritorializationType.NOMADIZATION:
                # Increase mobility
                if hasattr(topology, "flux_cycle"):
                    await topology.flux_cycle()

            elif dtype == DeterritorializationType.FLIGHT:
                # Enable lines of flight
                if hasattr(topology, "initiate_line_of_flight") and hasattr(topology, "nodes"):
                    # Allow some agents to escape
                    import random

                    for node in topology.nodes.values():
                        if random.random() < 0.1:  # 10% chance
                            topology.initiate_line_of_flight(node.agent_id, "exploration", 300.0)
                            affected_agents.append(node.agent_id)

        # Record event
        event = AssemblyEvent(
            event_type=f"deterritorialize_{dtype.value}",
            previous_state=self._state,
            new_state=AssemblyState.UNSTABLE,
            trigger="stability_evaluation",
            affected_agents=affected_agents,
        )
        self._history.append(event)

        await self._transition_state(AssemblyState.TRANSITIONING, "deterritorialization")

        return True

    async def _transition_state(
        self,
        new_state: AssemblyState,
        trigger: str,
    ) -> None:
        """Transition to a new assembly state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state

        logger.info(f"Assembly {self._assembly_id} state: {old_state.value} -> {new_state.value}")

        # Publish event if bus available
        if self._event_bus:
            # Event publishing would go here
            pass

    def record_role_change(self, agent_id: str) -> None:
        """Record a role change for metrics."""
        self._role_changes.append(datetime.now(UTC))
        # Keep only recent changes
        if len(self._role_changes) > 1000:
            self._role_changes = self._role_changes[-500:]

    def record_defection(self, agent_id: str) -> None:
        """Record an agent leaving for metrics."""
        self._defections.append(datetime.now(UTC))
        if len(self._defections) > 1000:
            self._defections = self._defections[-500:]

    def record_task_result(self, success: bool) -> None:
        """Record a task result for metrics."""
        self._task_results.append(success)
        if len(self._task_results) > 1000:
            self._task_results = self._task_results[-500:]

    def get_history(self, limit: int = 50) -> list[AssemblyEvent]:
        """Get recent assembly events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of recent AssemblyEvent objects.
        """
        return self._history[-limit:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "assembly_id": self._assembly_id,
            "state": self._state.value,
            "metrics": {
                "connection_density": self._metrics.connection_density,
                "role_consistency": self._metrics.role_consistency,
                "communication_flow": self._metrics.communication_flow,
                "defection_rate": self._metrics.defection_rate,
                "task_completion_rate": self._metrics.task_completion_rate,
                "coordination_overhead": self._metrics.coordination_overhead,
                "overall_stability": self._metrics.overall_stability,
            },
            "history": [e.to_dict() for e in self._history[-20:]],
        }
