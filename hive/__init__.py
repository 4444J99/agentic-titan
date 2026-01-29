"""
Hive Mind - Shared Intelligence Layer

Provides collective memory and real-time coordination for the agent swarm:
- Long-term memory (ChromaDB/Vector store)
- Working memory (Redis)
- Event bus (NATS/Redis Streams)
- Distributed state
- Event-driven topology transitions
- LLM-powered task analysis
- Episodic learning from outcomes
"""

from hive.memory import HiveMind, MemoryConfig
from hive.topology import (
    TopologyEngine,
    TopologyType,
    TaskProfile,
    AgentNode,
    BaseTopology,
    SwarmTopology,
    HierarchyTopology,
    PipelineTopology,
    MeshTopology,
    RingTopology,
    StarTopology,
)
from hive.events import (
    EventBus,
    Event,
    EventType,
    get_event_bus,
)
from hive.analyzer import (
    TaskAnalyzer,
    AnalysisResult,
    analyze_task,
)
from hive.learning import (
    EpisodicLearner,
    Episode,
    EpisodeOutcome,
    TopologyPreference,
    get_episodic_learner,
)
from hive.stigmergy import (
    PheromoneField,
    PheromoneTrace,
    TraceType,
    GradientInfo,
)
from hive.neighborhood import (
    TopologicalNeighborhood,
    InteractionRecord,
    InteractionType,
    AgentProfile,
    NeighborScore,
)
from hive.topology_extended import (
    ExtendedTopologyType,
    RhizomaticTopology,
    ArborealTopology,
    TerritorializedTopology,
    DeterritorializedTopology,
    Territory,
    Connection,
    ConnectionType,
)
from hive.assembly import (
    AssemblyManager,
    AssemblyState,
    AssemblyEvent,
    StabilityMetrics,
    TerritorizationType,
    DeterritorializationType,
)
from hive.machines import (
    MachineDynamics,
    MachineType,
    MachineState,
    MachineOperation,
    OperationType,
)

__all__ = [
    # Memory
    "HiveMind",
    "MemoryConfig",
    # Topology
    "TopologyEngine",
    "TopologyType",
    "TaskProfile",
    "AgentNode",
    "BaseTopology",
    "SwarmTopology",
    "HierarchyTopology",
    "PipelineTopology",
    "MeshTopology",
    "RingTopology",
    "StarTopology",
    # Events
    "EventBus",
    "Event",
    "EventType",
    "get_event_bus",
    # Analyzer
    "TaskAnalyzer",
    "AnalysisResult",
    "analyze_task",
    # Learning
    "EpisodicLearner",
    "Episode",
    "EpisodeOutcome",
    "TopologyPreference",
    "get_episodic_learner",
    # Stigmergy
    "PheromoneField",
    "PheromoneTrace",
    "TraceType",
    "GradientInfo",
    # Neighborhood
    "TopologicalNeighborhood",
    "InteractionRecord",
    "InteractionType",
    "AgentProfile",
    "NeighborScore",
    # Extended Topologies
    "ExtendedTopologyType",
    "RhizomaticTopology",
    "ArborealTopology",
    "TerritorializedTopology",
    "DeterritorializedTopology",
    "Territory",
    "Connection",
    "ConnectionType",
    # Assembly
    "AssemblyManager",
    "AssemblyState",
    "AssemblyEvent",
    "StabilityMetrics",
    "TerritorizationType",
    "DeterritorializationType",
    # Machines
    "MachineDynamics",
    "MachineType",
    "MachineState",
    "MachineOperation",
    "OperationType",
]
