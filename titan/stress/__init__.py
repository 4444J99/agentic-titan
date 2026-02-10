"""
Agentic Titan - Stress Testing Framework

Provides tools for stress testing agent swarms at scale (50-100+ agents).
Measures throughput, latency, memory usage, and identifies bottlenecks.
"""

from titan.stress.metrics import (
    LatencyHistogram,
    StressMetrics,
    ThroughputCounter,
)
from titan.stress.runner import (
    AgentFactory,
    StressTestConfig,
    StressTestResult,
    StressTestRunner,
)
from titan.stress.scenarios import (
    ChaosScenario,
    HierarchyDelegationScenario,
    PipelineWorkflowScenario,
    Scenario,
    SwarmBrainstormScenario,
)

__all__ = [
    # Runner
    "StressTestRunner",
    "StressTestConfig",
    "StressTestResult",
    "AgentFactory",
    # Scenarios
    "Scenario",
    "SwarmBrainstormScenario",
    "PipelineWorkflowScenario",
    "HierarchyDelegationScenario",
    "ChaosScenario",
    # Metrics
    "StressMetrics",
    "LatencyHistogram",
    "ThroughputCounter",
]
