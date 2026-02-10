"""
Cell Agent - Multicellular coordination patterns.

Implements cellular-level coordination based on biological patterns:
- Apoptosis (programmed self-termination for collective good)
- Germ-soma separation (reproductive vs somatic roles)
- Cell signaling to neighbors
- Division of labor

Based on the major transitions in evolution that led to
multicellular organisms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from agents.framework.base_agent import AgentState, BaseAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.agents.cell")


class CellType(StrEnum):
    """Types of cells in the multicellular organism."""

    STEM = "stem"  # Can differentiate
    SOMATIC = "somatic"  # Specialized, non-reproductive
    GERMLINE = "germline"  # Reproductive lineage
    SIGNALING = "signaling"  # Coordination cells


class CellState(StrEnum):
    """States a cell can be in."""

    HEALTHY = "healthy"
    STRESSED = "stressed"
    DAMAGED = "damaged"
    APOPTOTIC = "apoptotic"  # Undergoing programmed death
    DEAD = "dead"


class SignalType(StrEnum):
    """Types of cell signals."""

    GROWTH = "growth"  # Promote growth/division
    INHIBITION = "inhibition"  # Inhibit growth
    APOPTOSIS = "apoptosis"  # Trigger cell death
    DIFFERENTIATION = "differentiation"  # Trigger specialization
    STRESS = "stress"  # Damage detected
    SURVIVAL = "survival"  # Support survival


@dataclass
class CellSignal:
    """A signal sent between cells."""

    signal_type: SignalType
    sender: str
    intensity: float = 1.0  # 0-1
    target: str | None = None  # Specific target or None for neighbors
    payload: dict[str, Any] = field(default_factory=dict)
    sent_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CellMemory:
    """Cell's internal state and history."""

    lineage: list[str] = field(default_factory=list)  # Parent cell IDs
    division_count: int = 0
    signals_received: list[CellSignal] = field(default_factory=list)
    stress_level: float = 0.0  # 0-1
    damage_level: float = 0.0  # 0-1
    energy: float = 1.0  # 0-1


class CellAgent(BaseAgent):
    """
    Agent implementing cellular coordination patterns.

    Operates as a cell in a multicellular organism with:
    - Self-termination capability (apoptosis) for collective good
    - Germ-soma separation for aligned incentives
    - Neighbor signaling for coordination
    - Specialization through differentiation

    Capabilities:
    - Cell signaling
    - Apoptosis (self-termination)
    - Differentiation
    - Neighbor coordination
    """

    # Thresholds for cellular decisions
    DAMAGE_APOPTOSIS_THRESHOLD = 0.7
    STRESS_WARNING_THRESHOLD = 0.5
    ENERGY_CRITICAL_THRESHOLD = 0.2

    def __init__(
        self,
        cell_type: CellType = CellType.SOMATIC,
        neighborhood: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize cell agent.

        Args:
            cell_type: Type of cell.
            neighborhood: Topological neighborhood for neighbor interactions.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", f"cell_{cell_type.value}")
        kwargs.setdefault(
            "capabilities",
            [
                "cell_signaling",
                "apoptosis",
                "differentiation",
                "neighbor_coordination",
            ],
        )
        super().__init__(**kwargs)

        self._cell_type = cell_type
        self._cell_state = CellState.HEALTHY
        self._memory = CellMemory()
        self._neighborhood = neighborhood
        self._neighbors: list[str] = []
        self._pending_signals: list[CellSignal] = []
        self._apoptosis_initiated = False

    @property
    def cell_type(self) -> CellType:
        """Get cell type."""
        return self._cell_type

    @property
    def cell_state(self) -> CellState:
        """Get current cell state."""
        return self._cell_state

    @property
    def is_alive(self) -> bool:
        """Check if cell is alive."""
        return self._cell_state not in [CellState.DEAD, CellState.APOPTOTIC]

    async def initialize(self) -> None:
        """Initialize cell agent."""
        self._state = AgentState.READY
        logger.info(f"Cell agent initialized (type={self._cell_type.value})")

    async def work(self) -> dict[str, Any]:
        """Perform cell work cycle."""
        if not self.is_alive:
            return {"status": self._cell_state.value, "alive": False}

        result: dict[str, Any] = {
            "cell_type": self._cell_type.value,
            "cell_state": self._cell_state.value,
            "actions": [],
        }

        # Process incoming signals
        await self._process_signals()

        # Check health and potentially initiate apoptosis
        if await self._should_apoptose():
            await self.apoptosis("self_check")
            result["actions"].append("initiated_apoptosis")
            return result

        # Perform type-specific work
        if self._cell_type == CellType.STEM:
            result.update(await self._stem_cell_work())
        elif self._cell_type == CellType.SOMATIC:
            result.update(await self._somatic_cell_work())
        elif self._cell_type == CellType.GERMLINE:
            result.update(await self._germline_cell_work())
        elif self._cell_type == CellType.SIGNALING:
            result.update(await self._signaling_cell_work())

        # Update energy
        self._memory.energy = max(0.0, self._memory.energy - 0.01)
        if self._memory.energy < self.ENERGY_CRITICAL_THRESHOLD:
            await self._signal_neighbors(SignalType.STRESS, {"reason": "low_energy"})
            result["actions"].append("signaled_stress")

        return result

    async def shutdown(self) -> None:
        """Shutdown cell agent."""
        # Perform clean death if not already apoptotic
        if self.is_alive and not self._apoptosis_initiated:
            await self.apoptosis("shutdown")
        logger.info(f"Cell agent shutdown (state={self._cell_state.value})")

    # =========================================================================
    # Cell Type-Specific Behaviors
    # =========================================================================

    async def _stem_cell_work(self) -> dict[str, Any]:
        """Stem cell behavior: can differentiate into other types."""
        result = {"role": "stem", "differentiation_potential": True}

        # Check for differentiation signals
        diff_signals = [
            s for s in self._memory.signals_received if s.signal_type == SignalType.DIFFERENTIATION
        ]

        if diff_signals:
            # Differentiate based on signal
            target_type = diff_signals[-1].payload.get("target_type", "somatic")
            if target_type == "somatic":
                self._cell_type = CellType.SOMATIC
            elif target_type == "germline":
                self._cell_type = CellType.GERMLINE
            elif target_type == "signaling":
                self._cell_type = CellType.SIGNALING

            result["differentiated_to"] = target_type

        return result

    async def _somatic_cell_work(self) -> dict[str, Any]:
        """Somatic cell behavior: specialized work, no reproduction."""
        result = {"role": "somatic", "work_performed": True}

        # Perform specialized function
        # Somatic cells sacrifice reproductive ability for specialization

        # Consume energy for work
        self._memory.energy = max(0.0, self._memory.energy - 0.02)

        # Signal neighbors about status
        if self._memory.stress_level > self.STRESS_WARNING_THRESHOLD:
            await self._signal_neighbors(
                SignalType.STRESS,
                {"level": self._memory.stress_level},
            )
            result["signaled_stress"] = True

        return result

    async def _germline_cell_work(self) -> dict[str, Any]:
        """Germline cell behavior: reproductive lineage maintenance."""
        result = {"role": "germline", "reproductive_capacity": True}

        # Germline cells maintain reproductive potential
        # They are protected from somatic mutations

        # Check for division signals
        growth_signals = [
            s for s in self._memory.signals_received if s.signal_type == SignalType.GROWTH
        ]

        if growth_signals and self._memory.energy > 0.5:
            # Could trigger reproduction/spawning
            result["division_ready"] = True
            self._memory.division_count += 1

        return result

    async def _signaling_cell_work(self) -> dict[str, Any]:
        """Signaling cell behavior: coordinate other cells."""
        result: dict[str, Any] = {"role": "signaling", "signals_sent": 0}

        # Aggregate information from neighbors
        stress_count = sum(
            1 for s in self._memory.signals_received if s.signal_type == SignalType.STRESS
        )

        # Coordinate response
        if stress_count > 2:
            # Multiple stressed neighbors - coordinate response
            await self._signal_neighbors(
                SignalType.SURVIVAL,
                {"coordinated_response": True},
            )
            result["signals_sent"] += 1

            # May need to trigger apoptosis in severely damaged cells
            if stress_count > 4:
                await self._signal_neighbors(
                    SignalType.APOPTOSIS,
                    {"reason": "collective_decision"},
                    target_stressed=True,
                )
                result["signals_sent"] += 1
                result["apoptosis_signals_sent"] = True

        return result

    # =========================================================================
    # Apoptosis (Programmed Cell Death)
    # =========================================================================

    async def apoptosis(self, reason: str) -> None:
        """Initiate programmed cell death for collective good.

        Apoptosis allows the cell to die cleanly without damaging
        neighbors, sacrificing itself for the organism's benefit.

        Args:
            reason: Reason for apoptosis.
        """
        if self._apoptosis_initiated:
            return

        self._apoptosis_initiated = True
        self._cell_state = CellState.APOPTOTIC

        logger.info(f"Cell {self.agent_id} initiating apoptosis: {reason}")

        # Signal neighbors about impending death
        await self._signal_neighbors(
            SignalType.STRESS,
            {"apoptosis": True, "reason": reason},
        )

        # Clean up resources (in practice, release resources back to pool)
        await self._cleanup()

        # Mark as dead
        self._cell_state = CellState.DEAD

    async def _should_apoptose(self) -> bool:
        """Check if cell should undergo apoptosis.

        Returns:
            True if apoptosis should be initiated.
        """
        # High damage level
        if self._memory.damage_level > self.DAMAGE_APOPTOSIS_THRESHOLD:
            return True

        # Received apoptosis signal
        apoptosis_signals = [
            s for s in self._memory.signals_received if s.signal_type == SignalType.APOPTOSIS
        ]
        if apoptosis_signals:
            # Check if signal is strong enough
            total_intensity = sum(s.intensity for s in apoptosis_signals)
            if total_intensity > 0.5:
                return True

        # Critical energy depletion
        if self._memory.energy < 0.05:
            return True

        return False

    async def _cleanup(self) -> None:
        """Clean up during apoptosis."""
        # Clear pending signals
        self._pending_signals.clear()

        # Signal completion
        await self._signal_neighbors(
            SignalType.STRESS,
            {"apoptosis_complete": True},
        )

    # =========================================================================
    # Cell Signaling
    # =========================================================================

    async def send_signal(
        self,
        signal_type: SignalType,
        payload: dict[str, Any] | None = None,
        target: str | None = None,
        intensity: float = 1.0,
    ) -> CellSignal:
        """Send a signal to target or neighbors.

        Args:
            signal_type: Type of signal.
            payload: Signal data.
            target: Specific target or None for neighbors.
            intensity: Signal intensity.

        Returns:
            The sent CellSignal.
        """
        signal = CellSignal(
            signal_type=signal_type,
            sender=self.agent_id,
            intensity=intensity,
            target=target,
            payload=payload or {},
        )

        self._pending_signals.append(signal)

        # Broadcast via hive mind if available
        if self._hive_mind:
            await self._hive_mind.broadcast(
                source_agent_id=self.agent_id,
                message={
                    "content": f"Cell signal: {signal_type.value}",
                    "importance": intensity * 0.5,
                },
                topic=f"cell_{signal_type.value}",
            )

        return signal

    async def _signal_neighbors(
        self,
        signal_type: SignalType,
        payload: dict[str, Any],
        target_stressed: bool = False,
    ) -> int:
        """Signal all neighbors.

        Args:
            signal_type: Type of signal.
            payload: Signal data.
            target_stressed: Only target stressed cells.

        Returns:
            Number of signals sent.
        """
        count = 0

        for neighbor_id in self._neighbors:
            if target_stressed:
                # Would need to check neighbor state
                pass

            await self.send_signal(signal_type, payload, target=neighbor_id)
            count += 1

        return count

    async def receive_signal(self, signal: CellSignal) -> None:
        """Receive a signal from another cell.

        Args:
            signal: The received signal.
        """
        self._memory.signals_received.append(signal)

        # Keep only recent signals
        if len(self._memory.signals_received) > 100:
            self._memory.signals_received = self._memory.signals_received[-50:]

    async def _process_signals(self) -> None:
        """Process received signals and update state."""
        for signal in self._memory.signals_received[-10:]:  # Process recent
            if signal.signal_type == SignalType.GROWTH:
                # Growth signals increase energy
                self._memory.energy = min(1.0, self._memory.energy + signal.intensity * 0.1)

            elif signal.signal_type == SignalType.INHIBITION:
                # Inhibition reduces activity
                pass

            elif signal.signal_type == SignalType.STRESS:
                # Stress from neighbors increases local stress
                self._memory.stress_level = min(
                    1.0, self._memory.stress_level + signal.intensity * 0.1
                )

            elif signal.signal_type == SignalType.SURVIVAL:
                # Survival signals help recover
                self._memory.stress_level = max(
                    0.0, self._memory.stress_level - signal.intensity * 0.1
                )
                self._memory.damage_level = max(
                    0.0, self._memory.damage_level - signal.intensity * 0.05
                )

    # =========================================================================
    # Cell Management
    # =========================================================================

    def add_neighbor(self, neighbor_id: str) -> None:
        """Add a neighbor cell.

        Args:
            neighbor_id: Neighbor's agent ID.
        """
        if neighbor_id not in self._neighbors:
            self._neighbors.append(neighbor_id)

    def remove_neighbor(self, neighbor_id: str) -> None:
        """Remove a neighbor cell.

        Args:
            neighbor_id: Neighbor's agent ID.
        """
        if neighbor_id in self._neighbors:
            self._neighbors.remove(neighbor_id)

    def set_neighborhood(self, neighborhood: Any) -> None:
        """Set the topological neighborhood.

        Args:
            neighborhood: TopologicalNeighborhood instance.
        """
        self._neighborhood = neighborhood

    def damage(self, amount: float) -> None:
        """Apply damage to the cell.

        Args:
            amount: Damage amount (0-1).
        """
        self._memory.damage_level = min(1.0, self._memory.damage_level + amount)
        self._memory.stress_level = min(1.0, self._memory.stress_level + amount * 0.5)

        if self._memory.damage_level > 0.3:
            self._cell_state = CellState.DAMAGED
        elif self._memory.stress_level > 0.3:
            self._cell_state = CellState.STRESSED

    def heal(self, amount: float) -> None:
        """Heal the cell.

        Args:
            amount: Healing amount (0-1).
        """
        self._memory.damage_level = max(0.0, self._memory.damage_level - amount)
        self._memory.stress_level = max(0.0, self._memory.stress_level - amount * 0.5)

        if self._memory.damage_level < 0.1 and self._memory.stress_level < 0.2:
            self._cell_state = CellState.HEALTHY

    def get_status(self) -> dict[str, Any]:
        """Get cell status.

        Returns:
            Dictionary with cell status.
        """
        return {
            "agent_id": self.agent_id,
            "cell_type": self._cell_type.value,
            "cell_state": self._cell_state.value,
            "energy": self._memory.energy,
            "damage_level": self._memory.damage_level,
            "stress_level": self._memory.stress_level,
            "neighbors": len(self._neighbors),
            "signals_pending": len(self._pending_signals),
            "is_alive": self.is_alive,
        }
