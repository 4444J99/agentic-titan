"""Assembly Theory data models.

Implements the core concepts from Assembly Theory:
- Assembly Index (AI): Minimum steps required to construct an object
- Total Assembly (A): Weighted sum of assembly indices
- Selection Signal: Indicator of whether an object was shaped by selection

These metrics help quantify the complexity and history of agent
coordination patterns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger("titan.metrics.assembly")


class SelectionSignal(StrEnum):
    """Strength of selection signal in assembly history.

    Based on Assembly Theory's approach to detecting selection:
    - High assembly index + high abundance = strong selection signal
    - High assembly index + low abundance = weak selection signal
    - Low assembly index = absent selection signal (random assembly)
    """

    STRONG = "strong"  # Clear evidence of selection
    MODERATE = "moderate"  # Some evidence of selection
    WEAK = "weak"  # Minimal evidence
    ABSENT = "absent"  # No selection signal


class StepType(StrEnum):
    """Types of assembly steps."""

    COORDINATION = "coordination"  # Agents coordinating
    TRANSFORMATION = "transformation"  # State change
    AGGREGATION = "aggregation"  # Combining elements
    SPECIALIZATION = "specialization"  # Role differentiation
    COMMUNICATION = "communication"  # Information exchange
    DECISION = "decision"  # Choice point
    SYNTHESIS = "synthesis"  # Creating new output


@dataclass
class AssemblyStep:
    """A single step in an assembly path.

    Represents one operation that transforms the current state
    into a new state during the assembly process.
    """

    step_id: str
    step_type: StepType
    description: str
    input_state: dict[str, Any] = field(default_factory=dict)
    output_state: dict[str, Any] = field(default_factory=dict)
    agent_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def complexity(self) -> float:
        """Estimate complexity of this step (0-1)."""
        # Simple heuristic based on state changes and agents involved
        state_delta = len(self.output_state) - len(self.input_state)
        agent_count = len(self.agent_ids)

        return min(1.0, 0.2 + (0.1 * max(0, state_delta)) + (0.1 * agent_count))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "description": self.description,
            "input_state": self.input_state,
            "output_state": self.output_state,
            "agent_ids": self.agent_ids,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyStep:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(UTC)

        return cls(
            step_id=data["step_id"],
            step_type=StepType(data["step_type"]),
            description=data["description"],
            input_state=data.get("input_state", {}),
            output_state=data.get("output_state", {}),
            agent_ids=data.get("agent_ids", []),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class AssemblyPath:
    """A complete path of assembly steps.

    Represents one way an object (like a coordination outcome) was
    assembled from its constituent parts through a sequence of steps.
    """

    path_id: str
    steps: list[AssemblyStep] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def assembly_index(self) -> int:
        """Minimum number of steps to construct (AI).

        The Assembly Index is the key metric from Assembly Theory,
        representing the minimum number of steps needed to construct
        the final object.
        """
        return len(self.steps)

    @property
    def duration_seconds(self) -> float | None:
        """Total duration of the assembly path."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def involved_agents(self) -> set[str]:
        """All agents involved in this path."""
        agents: set[str] = set()
        for step in self.steps:
            agents.update(step.agent_ids)
        return agents

    @property
    def total_complexity(self) -> float:
        """Sum of step complexities."""
        return sum(step.complexity for step in self.steps)

    def add_step(self, step: AssemblyStep) -> None:
        """Add a step to the path."""
        self.steps.append(step)

    def complete(self) -> None:
        """Mark the path as complete."""
        self.end_time = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path_id": self.path_id,
            "steps": [s.to_dict() for s in self.steps],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "assembly_index": self.assembly_index,
            "total_complexity": self.total_complexity,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyPath:
        """Create from dictionary."""
        start_time = data.get("start_time")
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        elif start_time is None:
            start_time = datetime.now(UTC)

        end_time = data.get("end_time")
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        steps = [AssemblyStep.from_dict(s) for s in data.get("steps", [])]

        return cls(
            path_id=data["path_id"],
            steps=steps,
            start_time=start_time,
            end_time=end_time,
            metadata=data.get("metadata", {}),
        )


@dataclass
class AssemblyMetrics:
    """Aggregate assembly metrics for an ensemble of paths.

    Implements the Total Assembly calculation from Assembly Theory:
    A = Σ(e^ai × (ni - 1)) / NT

    Where:
    - ai = assembly index of object i
    - ni = copy number (abundance) of object i
    - NT = total number of objects
    """

    paths: list[AssemblyPath] = field(default_factory=list)
    ensemble_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_objects(self) -> int:
        """Total number of assembly paths (NT)."""
        return len(self.paths)

    @property
    def mean_assembly_index(self) -> float:
        """Average assembly index across all paths."""
        if not self.paths:
            return 0.0
        return sum(p.assembly_index for p in self.paths) / len(self.paths)

    @property
    def max_assembly_index(self) -> int:
        """Maximum assembly index in the ensemble."""
        if not self.paths:
            return 0
        return max(p.assembly_index for p in self.paths)

    @property
    def total_assembly(self) -> float:
        """Calculate Total Assembly (A).

        A = Σ(e^ai × (ni - 1)) / NT

        For our purposes, we treat each unique assembly index as having
        ni = count of paths with that index.
        """
        if not self.paths:
            return 0.0

        # Group paths by assembly index
        index_counts: dict[int, int] = {}
        for path in self.paths:
            ai = path.assembly_index
            index_counts[ai] = index_counts.get(ai, 0) + 1

        # Calculate total assembly
        total = 0.0
        for ai, ni in index_counts.items():
            if ni > 1:  # Only count if abundance > 1
                total += math.exp(ai) * (ni - 1)

        nt = self.total_objects
        return total / nt if nt > 0 else 0.0

    @property
    def selection_signal(self) -> SelectionSignal:
        """Determine the selection signal strength.

        Based on Assembly Theory's approach:
        - High total assembly with high max AI = STRONG
        - Moderate values = MODERATE
        - Low values = WEAK or ABSENT
        """
        total_a = self.total_assembly
        max_ai = self.max_assembly_index

        if max_ai >= 15 and total_a > 100:
            return SelectionSignal.STRONG
        elif max_ai >= 10 or total_a > 50:
            return SelectionSignal.MODERATE
        elif max_ai >= 5 or total_a > 10:
            return SelectionSignal.WEAK
        else:
            return SelectionSignal.ABSENT

    @property
    def complexity_distribution(self) -> dict[int, int]:
        """Distribution of assembly indices."""
        dist: dict[int, int] = {}
        for path in self.paths:
            ai = path.assembly_index
            dist[ai] = dist.get(ai, 0) + 1
        return dict(sorted(dist.items()))

    def add_path(self, path: AssemblyPath) -> None:
        """Add a path to the ensemble."""
        self.paths.append(path)

    def get_paths_by_index(self, assembly_index: int) -> list[AssemblyPath]:
        """Get all paths with a specific assembly index."""
        return [p for p in self.paths if p.assembly_index == assembly_index]

    def get_most_complex_path(self) -> AssemblyPath | None:
        """Get the path with the highest assembly index."""
        if not self.paths:
            return None
        return max(self.paths, key=lambda p: p.assembly_index)

    def get_simplest_path(self) -> AssemblyPath | None:
        """Get the path with the lowest assembly index."""
        if not self.paths:
            return None
        return min(self.paths, key=lambda p: p.assembly_index)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ensemble_id": self.ensemble_id,
            "paths": [p.to_dict() for p in self.paths],
            "total_objects": self.total_objects,
            "mean_assembly_index": self.mean_assembly_index,
            "max_assembly_index": self.max_assembly_index,
            "total_assembly": self.total_assembly,
            "selection_signal": self.selection_signal.value,
            "complexity_distribution": self.complexity_distribution,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblyMetrics:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(UTC)

        paths = [AssemblyPath.from_dict(p) for p in data.get("paths", [])]

        return cls(
            ensemble_id=data.get("ensemble_id", ""),
            paths=paths,
            created_at=created_at,
        )

    def to_prometheus_format(self) -> dict[str, float]:
        """Export metrics in Prometheus-compatible format."""
        return {
            "titan_assembly_total_objects": float(self.total_objects),
            "titan_assembly_mean_index": self.mean_assembly_index,
            "titan_assembly_max_index": float(self.max_assembly_index),
            "titan_assembly_total": self.total_assembly,
            "titan_assembly_selection_signal": {
                SelectionSignal.STRONG: 3.0,
                SelectionSignal.MODERATE: 2.0,
                SelectionSignal.WEAK: 1.0,
                SelectionSignal.ABSENT: 0.0,
            }[self.selection_signal],
        }
