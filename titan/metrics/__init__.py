"""Assembly Theory Metrics Module.

Provides data models and tracking for Assembly Theory metrics,
which quantify the complexity and selection history of assembled objects.

Components:
- assembly: Core data models for Assembly Theory
- assembly_tracker: Decision history tracking and path analysis
"""

from titan.metrics.assembly import (
    AssemblyMetrics,
    AssemblyPath,
    AssemblyStep,
    SelectionSignal,
)
from titan.metrics.assembly_tracker import AssemblyTracker

__all__ = [
    "AssemblyMetrics",
    "AssemblyPath",
    "AssemblyStep",
    "AssemblyTracker",
    "SelectionSignal",
]
