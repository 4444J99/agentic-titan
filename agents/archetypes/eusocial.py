"""
Eusocial Colony Agent - Superorganism coordination patterns.

Implements eusocial colony dynamics based on ant/bee/termite colonies:
- Reproductive division of labor (queen vs workers)
- Worker policing mechanisms
- Stigmergic coordination via pheromone-like signals
- Caste-based specialization

The colony functions as a superorganism where individual agents
sacrifice reproductive autonomy for collective benefit.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from agents.framework.base_agent import AgentState, BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.agents.eusocial")


class CasteType(StrEnum):
    """Castes in the eusocial colony."""

    QUEEN = "queen"  # Reproductive, decision-making
    WORKER = "worker"  # General tasks
    SOLDIER = "soldier"  # Defense and enforcement
    FORAGER = "forager"  # Resource gathering
    NURSE = "nurse"  # Care for developing agents
    SCOUT = "scout"  # Exploration and reconnaissance


class ColonySignal(StrEnum):
    """Types of colony-wide signals."""

    ALARM = "alarm"  # Danger detected
    FOOD_FOUND = "food_found"  # Resource located
    RECRUITMENT = "recruitment"  # Request for workers
    POLICING = "policing"  # Enforcement needed
    TRAIL = "trail"  # Path to follow


@dataclass
class ColonyTask:
    """A task within the colony."""

    task_id: str
    task_type: str
    description: str
    priority: float = 0.5
    assigned_caste: CasteType | None = None
    assigned_to: str | None = None
    completed: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ColonyState:
    """State of the colony as a superorganism."""

    health: float = 1.0  # 0-1, overall colony health
    food_stores: float = 0.5  # 0-1, resource level
    population: int = 0
    caste_counts: dict[str, int] = field(default_factory=dict)
    active_tasks: list[ColonyTask] = field(default_factory=list)
    threats_detected: int = 0


class EusocialColonyAgent(BaseAgent):
    """
    Agent implementing eusocial colony coordination patterns.

    The colony operates as a superorganism with:
    - Queen(s) that make reproductive/strategic decisions
    - Workers that perform tasks based on caste
    - Stigmergic coordination through pheromone-like signals
    - Worker policing to prevent defection

    Capabilities:
    - Caste-based task assignment
    - Stigmergic communication
    - Colony-level decision making
    - Resource allocation
    - Defense coordination
    """

    def __init__(
        self,
        caste: CasteType = CasteType.WORKER,
        pheromone_field: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize eusocial colony agent.

        Args:
            caste: Agent's caste within the colony.
            pheromone_field: Shared pheromone field for stigmergic communication.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", f"colony_{caste.value}")
        kwargs.setdefault(
            "capabilities",
            [
                "caste_behavior",
                "stigmergic_communication",
                "colony_coordination",
            ],
        )
        super().__init__(**kwargs)

        self._caste = caste
        self._pheromone_field = pheromone_field
        self._colony_state = ColonyState()
        self._colony_members: dict[str, CasteType] = {}
        self._current_task: ColonyTask | None = None
        self._carrying: dict[str, Any] = {}  # Resources being carried
        self._location: str = "nest"

    @property
    def caste(self) -> CasteType:
        """Get agent's caste."""
        return self._caste

    @property
    def is_queen(self) -> bool:
        """Check if this agent is a queen."""
        return self._caste == CasteType.QUEEN

    async def initialize(self) -> None:
        """Initialize the colony agent."""
        self._state = AgentState.READY
        logger.info(f"Colony agent initialized (caste={self._caste.value})")

    async def work(self) -> dict[str, Any]:
        """Perform caste-specific work."""
        result: dict[str, Any] = {
            "caste": self._caste.value,
            "location": self._location,
            "actions": [],
        }

        if self._caste == CasteType.QUEEN:
            result.update(await self._queen_work())
        elif self._caste == CasteType.WORKER:
            result.update(await self._worker_work())
        elif self._caste == CasteType.SOLDIER:
            result.update(await self._soldier_work())
        elif self._caste == CasteType.FORAGER:
            result.update(await self._forager_work())
        elif self._caste == CasteType.SCOUT:
            result.update(await self._scout_work())
        elif self._caste == CasteType.NURSE:
            result.update(await self._nurse_work())

        return result

    async def shutdown(self) -> None:
        """Shutdown colony agent."""
        logger.info(f"Colony agent shutdown (caste={self._caste.value})")

    # =========================================================================
    # Caste-Specific Behaviors
    # =========================================================================

    async def _queen_work(self) -> dict[str, Any]:
        """Queen behavior: strategic decisions and task assignment."""
        result: dict[str, Any] = {"role": "queen", "decisions": []}

        # Monitor colony state
        await self._update_colony_state()

        # Assign tasks based on needs
        if self._colony_state.food_stores < 0.3:
            task = await self._create_task("forage", "Gather food resources", CasteType.FORAGER)
            result["decisions"].append(f"Created foraging task: {task.task_id}")

        if self._colony_state.threats_detected > 0:
            task = await self._create_task("defend", "Defend against threat", CasteType.SOLDIER)
            result["decisions"].append(f"Created defense task: {task.task_id}")

        # Emit recruitment signal if workers needed
        if len(self._colony_state.active_tasks) > len(self._colony_members) * 0.5:
            await self._emit_signal(ColonySignal.RECRUITMENT, {"urgency": "high"})
            result["decisions"].append("Emitted recruitment signal")

        return result

    async def _worker_work(self) -> dict[str, Any]:
        """Worker behavior: general task execution."""
        result: dict[str, Any] = {"role": "worker", "task_status": None}

        # Check for assigned task
        if self._current_task:
            # Work on task
            progress = await self._work_on_task()
            result["task_status"] = {
                "task_id": self._current_task.task_id,
                "progress": progress,
            }
        else:
            # Look for available task
            task = await self._find_available_task()
            if task:
                self._current_task = task
                task.assigned_to = self.agent_id
                result["task_status"] = {"assigned": task.task_id}

        # Respond to signals
        signals = await self._sense_signals()
        result["signals_sensed"] = len(signals)

        return result

    async def _soldier_work(self) -> dict[str, Any]:
        """Soldier behavior: defense and policing."""
        result: dict[str, Any] = {"role": "soldier", "patrols": 0, "incidents": []}

        # Patrol for threats
        threats = await self._patrol_for_threats()
        result["patrols"] = 1
        result["threats_found"] = threats

        # Worker policing - check for defectors
        defectors = await self._check_for_defectors()
        for defector in defectors:
            result["incidents"].append(
                {
                    "type": "policing",
                    "target": defector,
                    "action": "warning_issued",
                }
            )

        # Emit alarm if threats detected
        if threats:
            await self._emit_signal(ColonySignal.ALARM, {"threat_level": len(threats)})

        return result

    async def _forager_work(self) -> dict[str, Any]:
        """Forager behavior: resource gathering."""
        result: dict[str, Any] = {"role": "forager", "gathered": [], "trails_followed": 0}

        # Follow food trails if available
        trail = await self._sense_signals([ColonySignal.FOOD_FOUND])
        if trail:
            # Follow the trail
            result["trails_followed"] = 1
            self._location = trail[0].get("location", "food_source")

        # Gather resources at current location
        if self._location != "nest":
            resource = await self._gather_resource()
            if resource:
                self._carrying["food"] = resource
                result["gathered"].append(resource)

                # Leave trail back to nest
                await self._emit_signal(
                    ColonySignal.TRAIL,
                    {"from": self._location, "to": "nest", "resource": "food"},
                )

        # Return to nest if carrying
        if self._carrying:
            await self._return_to_nest()
            result["returned"] = True

        return result

    async def _scout_work(self) -> dict[str, Any]:
        """Scout behavior: exploration and reconnaissance."""
        result: dict[str, Any] = {"role": "scout", "explored": [], "discoveries": []}

        # Explore new areas
        new_location = await self._explore()
        result["explored"].append(new_location)
        self._location = new_location

        # Check for resources or threats
        discovery = await self._survey_location()
        if discovery:
            result["discoveries"].append(discovery)

            # Signal discovery
            if discovery["type"] == "food":
                await self._emit_signal(
                    ColonySignal.FOOD_FOUND,
                    {"location": self._location, "amount": discovery.get("amount", 1)},
                )
            elif discovery["type"] == "threat":
                await self._emit_signal(
                    ColonySignal.ALARM,
                    {"location": self._location, "threat": discovery.get("threat", "unknown")},
                )

        return result

    async def _nurse_work(self) -> dict[str, Any]:
        """Nurse behavior: care for developing agents."""
        result = {"role": "nurse", "attended": 0}

        # Care for new agents (simplified)
        # In practice, this could involve initialization support
        result["attended"] = random.randint(0, 3)

        return result

    # =========================================================================
    # Stigmergic Communication
    # =========================================================================

    async def _emit_signal(
        self,
        signal_type: ColonySignal,
        payload: dict[str, Any],
    ) -> None:
        """Emit a pheromone-like signal.

        Args:
            signal_type: Type of signal.
            payload: Signal data.
        """
        if self._pheromone_field:
            # Use pheromone field for stigmergic communication
            from hive.stigmergy import TraceType

            trace_map = {
                ColonySignal.ALARM: TraceType.WARNING,
                ColonySignal.FOOD_FOUND: TraceType.RESOURCE,
                ColonySignal.RECRUITMENT: TraceType.COLLABORATION,
                ColonySignal.POLICING: TraceType.WARNING,
                ColonySignal.TRAIL: TraceType.PATH,
            }

            trace_type = trace_map.get(signal_type, TraceType.PATH)
            await self._pheromone_field.deposit(
                agent_id=self.agent_id,
                trace_type=trace_type,
                location=self._location,
                intensity=0.8,
                payload={**payload, "signal": signal_type.value},
            )

        # Also broadcast via hive mind if available
        if self._hive_mind:
            await self._hive_mind.broadcast(
                source_agent_id=self.agent_id,
                message={
                    "content": f"Colony signal: {signal_type.value}",
                    "importance": 0.7,
                },
                topic=f"colony_{signal_type.value}",
            )

    async def _sense_signals(
        self,
        signal_types: list[ColonySignal] | None = None,
    ) -> list[dict[str, Any]]:
        """Sense nearby signals.

        Args:
            signal_types: Types of signals to sense, or all if None.

        Returns:
            List of sensed signals with their payloads.
        """
        signals: list[dict[str, Any]] = []

        if self._pheromone_field:
            from hive.stigmergy import TraceType

            # Map signal types to trace types
            if signal_types is None:
                trace_types = None
            else:
                trace_map = {
                    ColonySignal.ALARM: TraceType.WARNING,
                    ColonySignal.FOOD_FOUND: TraceType.RESOURCE,
                    ColonySignal.RECRUITMENT: TraceType.COLLABORATION,
                    ColonySignal.POLICING: TraceType.WARNING,
                    ColonySignal.TRAIL: TraceType.PATH,
                }
                trace_types = [trace_map.get(st, TraceType.PATH) for st in signal_types]

            traces = await self._pheromone_field.sense(
                location=self._location,
                trace_types=trace_types,
            )

            for trace in traces:
                signals.append(
                    {
                        "type": trace.payload.get("signal", "unknown"),
                        "intensity": trace.intensity,
                        "location": trace.location,
                        **trace.payload,
                    }
                )

        return signals

    # =========================================================================
    # Task Management
    # =========================================================================

    async def _create_task(
        self,
        task_type: str,
        description: str,
        assigned_caste: CasteType | None = None,
        priority: float = 0.5,
    ) -> ColonyTask:
        """Create a new colony task (queen only).

        Args:
            task_type: Type of task.
            description: Task description.
            assigned_caste: Caste to assign.
            priority: Task priority.

        Returns:
            Created ColonyTask.
        """
        import uuid

        task = ColonyTask(
            task_id=f"CT-{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            description=description,
            assigned_caste=assigned_caste,
            priority=priority,
        )

        self._colony_state.active_tasks.append(task)
        return task

    async def _find_available_task(self) -> ColonyTask | None:
        """Find an available task matching this agent's caste.

        Returns:
            Available task or None.
        """
        for task in self._colony_state.active_tasks:
            if task.completed:
                continue
            if task.assigned_to:
                continue
            if task.assigned_caste and task.assigned_caste != self._caste:
                continue
            return task
        return None

    async def _work_on_task(self) -> float:
        """Work on the current task.

        Returns:
            Progress made (0-1).
        """
        if not self._current_task:
            return 0.0

        # Simulate work progress
        progress = random.uniform(0.2, 0.5)

        # Check if complete
        if random.random() < progress:
            self._current_task.completed = True
            self._current_task = None
            return 1.0

        return progress

    # =========================================================================
    # Colony Coordination
    # =========================================================================

    async def _update_colony_state(self) -> None:
        """Update knowledge of colony state."""
        self._colony_state.population = len(self._colony_members) + 1

        # Count castes
        self._colony_state.caste_counts = {}
        for caste in self._colony_members.values():
            caste_name = caste.value
            self._colony_state.caste_counts[caste_name] = (
                self._colony_state.caste_counts.get(caste_name, 0) + 1
            )
        self._colony_state.caste_counts[self._caste.value] = (
            self._colony_state.caste_counts.get(self._caste.value, 0) + 1
        )

    async def _patrol_for_threats(self) -> list[str]:
        """Patrol for threats (soldier).

        Returns:
            List of detected threats.
        """
        # Sense alarm signals
        signals = await self._sense_signals([ColonySignal.ALARM])
        return [s.get("threat", "unknown") for s in signals]

    async def _check_for_defectors(self) -> list[str]:
        """Check for defecting workers (policing).

        Returns:
            List of potential defector agent IDs.
        """
        # In practice, check for agents not following colony rules
        # Simplified: random detection
        defectors: list[str] = []

        for member_id, caste in self._colony_members.items():
            if caste != CasteType.QUEEN and random.random() < 0.05:
                defectors.append(member_id)

        return defectors

    async def _gather_resource(self) -> dict[str, Any] | None:
        """Gather resource at current location.

        Returns:
            Gathered resource or None.
        """
        if self._location == "nest":
            return None

        if random.random() < 0.7:  # 70% chance of finding resource
            return {
                "type": "food",
                "amount": random.uniform(0.1, 0.3),
                "location": self._location,
            }

        return None

    async def _return_to_nest(self) -> None:
        """Return to nest with carried resources."""
        if self._carrying.get("food"):
            # Deposit food
            self._colony_state.food_stores = min(
                1.0, self._colony_state.food_stores + self._carrying["food"]["amount"]
            )
            self._carrying.clear()

        self._location = "nest"

    async def _explore(self) -> str:
        """Explore and return new location.

        Returns:
            New location identifier.
        """
        locations = ["area_1", "area_2", "area_3", "area_4", "food_source", "danger_zone"]
        return random.choice(locations)

    async def _survey_location(self) -> dict[str, Any] | None:
        """Survey current location for discoveries.

        Returns:
            Discovery if any.
        """
        if self._location == "food_source":
            return {"type": "food", "amount": random.uniform(0.5, 1.0)}
        elif self._location == "danger_zone":
            return {"type": "threat", "threat": "predator"}
        elif random.random() < 0.2:
            return {"type": "food", "amount": random.uniform(0.1, 0.3)}
        return None

    # =========================================================================
    # Colony Management
    # =========================================================================

    def register_colony_member(self, agent_id: str, caste: CasteType) -> None:
        """Register a colony member.

        Args:
            agent_id: Member's agent ID.
            caste: Member's caste.
        """
        self._colony_members[agent_id] = caste

    def get_colony_state(self) -> ColonyState:
        """Get current colony state."""
        return self._colony_state

    def set_pheromone_field(self, field: Any) -> None:
        """Set the pheromone field for communication.

        Args:
            field: PheromoneField instance.
        """
        self._pheromone_field = field
