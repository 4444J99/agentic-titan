"""
Test Configuration and Shared Fixtures.

Provides:
- Mock LLM responses for fast, deterministic tests
- Test database setup/teardown
- Shared agent fixtures
- Performance baseline tracking
- Test utilities
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """Test configuration."""
    return {
        "use_real_llm": os.getenv("TEST_USE_REAL_LLM", "false").lower() == "true",
        "llm_model": os.getenv("TEST_LLM_MODEL", "gpt-4o-mini"),
        "max_tokens": 100,
        "timeout_seconds": 30,
        "db_url": os.getenv("TEST_DATABASE_URL", "sqlite:///test_titan.db"),
    }


# ============================================================================
# Mock LLM Responses
# ============================================================================

@dataclass
class MockLLMResponse:
    """Mock response from LLM."""
    content: str
    model: str = "mock-model"
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 10, "output_tokens": 50})
    finish_reason: str = "stop"


class MockLLMRouter:
    """Mock LLM router for testing without API calls."""

    def __init__(self) -> None:
        self._responses: dict[str, str] = {}
        self._default_responses = {
            "decompose": """SUBTASK: Research the topic
AGENT: researcher
DEPENDS: none
---
SUBTASK: Implement the solution
AGENT: coder
DEPENDS: st-0
---
SUBTASK: Review the code
AGENT: reviewer
DEPENDS: st-1""",
            "research": "Based on my research, the key findings are: 1) Important insight, 2) Another discovery, 3) Final observation.",
            "code": """def solution():
    # Implementation
    return "result"
""",
            "review": "The code looks good. Minor suggestions: 1) Add type hints, 2) Improve documentation.",
            "analyze": "Analysis complete. Found 3 items of interest.",
            "default": "Task completed successfully.",
        }
        self._call_count = 0
        self._call_history: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize the mock router."""
        pass

    async def complete(
        self,
        messages: list[Any],
        system: str = "",
        max_tokens: int = 100,
        **kwargs: Any,
    ) -> MockLLMResponse:
        """Generate mock completion."""
        self._call_count += 1

        # Extract prompt content
        prompt = ""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                prompt = last_msg.content
            elif isinstance(last_msg, dict):
                prompt = last_msg.get("content", "")

        # Record call
        self._call_history.append({
            "prompt": prompt[:200],
            "system": system[:100],
            "timestamp": datetime.now().isoformat(),
        })

        # Find matching response
        prompt_lower = prompt.lower()
        for key, response in self._responses.items():
            if key in prompt_lower:
                return MockLLMResponse(content=response)

        # Use default responses based on context
        if "decompose" in prompt_lower or "subtask" in prompt_lower:
            return MockLLMResponse(content=self._default_responses["decompose"])
        elif "research" in prompt_lower or "investigate" in prompt_lower:
            return MockLLMResponse(content=self._default_responses["research"])
        elif "code" in prompt_lower or "implement" in prompt_lower:
            return MockLLMResponse(content=self._default_responses["code"])
        elif "review" in prompt_lower or "check" in prompt_lower:
            return MockLLMResponse(content=self._default_responses["review"])
        elif "analyze" in prompt_lower or "scan" in prompt_lower:
            return MockLLMResponse(content=self._default_responses["analyze"])

        return MockLLMResponse(content=self._default_responses["default"])

    def set_response(self, keyword: str, response: str) -> None:
        """Set a custom response for a keyword."""
        self._responses[keyword.lower()] = response

    def get_call_count(self) -> int:
        """Get number of calls made."""
        return self._call_count

    def get_call_history(self) -> list[dict[str, Any]]:
        """Get call history."""
        return self._call_history

    def reset(self) -> None:
        """Reset mock state."""
        self._call_count = 0
        self._call_history = []
        self._responses = {}


@pytest.fixture
def mock_router() -> Generator[MockLLMRouter, None, None]:
    """Create a fresh mock LLM router for each test."""
    router = MockLLMRouter()
    yield router
    # Note: No explicit cleanup needed since each test gets a fresh instance


@pytest.fixture
def mock_llm_router() -> Generator[MockLLMRouter, None, None]:
    """Patch the global router with a fresh mock for each test."""
    router = MockLLMRouter()

    # Need to patch in multiple locations due to imports
    patches = [
        patch("adapters.router.get_router", return_value=router),
        patch("agents.archetypes.orchestrator.get_router", return_value=router),
        patch("agents.archetypes.researcher.get_router", return_value=router),
        patch("agents.archetypes.cfo.get_router", return_value=router),
        patch("agents.archetypes.devops.get_router", return_value=router),
        patch("agents.archetypes.security_analyst.get_router", return_value=router),
        patch("agents.archetypes.data_engineer.get_router", return_value=router),
        patch("agents.archetypes.product_manager.get_router", return_value=router),
    ]

    for p in patches:
        p.start()

    yield router

    for p in patches:
        p.stop()


# ============================================================================
# Mock Hive Mind
# ============================================================================

class MockHiveMind:
    """Mock Hive Mind for testing."""

    def __init__(self) -> None:
        self._memories: dict[str, dict[str, Any]] = {}
        self._messages: list[dict[str, Any]] = []
        self._subscriptions: dict[str, list[Any]] = {}
        self._kv_store: dict[str, Any] = {}
        self._topology: dict[str, Any] = {}

    async def remember(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory."""
        memory_id = f"mem-{len(self._memories)}"
        self._memories[memory_id] = {
            "agent_id": agent_id,
            "content": content,
            "importance": importance,
            "tags": tags or [],
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        return memory_id

    async def recall(
        self,
        query: str,
        k: int = 10,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories (simple keyword match for testing)."""
        results = []
        query_lower = query.lower()
        for mem_id, mem in self._memories.items():
            if query_lower in mem["content"].lower():
                results.append({"id": mem_id, **mem})
            elif tags and any(t in mem["tags"] for t in tags):
                results.append({"id": mem_id, **mem})
        return results[:k]

    async def broadcast(
        self,
        source_agent_id: str,
        message: dict[str, Any],
        topic: str = "general",
    ) -> None:
        """Broadcast message."""
        self._messages.append({
            "source": source_agent_id,
            "topic": topic,
            "message": message,
            "type": "broadcast",
        })

    async def send(
        self,
        source_agent_id: str,
        target_agent_id: str,
        message: dict[str, Any],
    ) -> None:
        """Send direct message."""
        self._messages.append({
            "source": source_agent_id,
            "target": target_agent_id,
            "message": message,
            "type": "direct",
        })

    async def subscribe(
        self,
        agent_id: str,
        topic: str,
        handler: Any,
    ) -> None:
        """Subscribe to topic."""
        self._subscriptions.setdefault(topic, []).append({
            "agent_id": agent_id,
            "handler": handler,
        })

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set key-value."""
        self._kv_store[key] = {"value": value, "ttl": ttl}

    async def get(self, key: str) -> Any:
        """Get key-value."""
        entry = self._kv_store.get(key)
        return entry["value"] if entry else None

    async def set_topology(self, topology_data: dict[str, Any]) -> None:
        """Set topology."""
        self._topology = topology_data

    async def get_topology(self) -> dict[str, Any]:
        """Get topology."""
        return self._topology

    def get_memories(self) -> dict[str, dict[str, Any]]:
        """Get all memories (for testing)."""
        return self._memories

    def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages (for testing)."""
        return self._messages

    def reset(self) -> None:
        """Reset state."""
        self._memories = {}
        self._messages = []
        self._subscriptions = {}
        self._kv_store = {}
        self._topology = {}


@pytest.fixture
def mock_hive_mind() -> MockHiveMind:
    """Create a mock Hive Mind."""
    return MockHiveMind()


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
async def base_agent(mock_hive_mind: MockHiveMind, mock_llm_router: MockLLMRouter):
    """Create a basic test agent."""
    from agents.framework.base_agent import BaseAgent

    class TestAgent(BaseAgent):
        async def initialize(self) -> None:
            pass

        async def work(self) -> str:
            return "test work completed"

        async def shutdown(self) -> None:
            pass

    agent = TestAgent(
        name="test-agent",
        capabilities=["test"],
        hive_mind=mock_hive_mind,
    )
    return agent


@pytest.fixture
async def orchestrator_agent(mock_hive_mind: MockHiveMind, mock_llm_router: MockLLMRouter):
    """Create an orchestrator agent for testing."""
    from agents.archetypes.orchestrator import OrchestratorAgent

    agent = OrchestratorAgent(
        task="Test task for orchestration",
        hive_mind=mock_hive_mind,
    )
    return agent


@pytest.fixture
async def researcher_agent(mock_hive_mind: MockHiveMind, mock_llm_router: MockLLMRouter):
    """Create a researcher agent for testing."""
    from agents.archetypes.researcher import ResearcherAgent

    agent = ResearcherAgent(
        topic="Test research topic",
        hive_mind=mock_hive_mind,
    )
    return agent


# ============================================================================
# Topology Fixtures
# ============================================================================

@pytest.fixture
def topology_engine():
    """Create a topology engine for testing."""
    from hive.topology import TopologyEngine, TopologyType

    engine = TopologyEngine()
    return engine


@pytest.fixture
def swarm_topology(topology_engine):
    """Create a swarm topology with test agents."""
    from hive.topology import TopologyType

    topology = topology_engine.create_topology(TopologyType.SWARM)

    # Add test agents
    for i in range(3):
        topology.add_agent(
            agent_id=f"agent-{i}",
            name=f"TestAgent{i}",
            capabilities=["test"],
        )

    return topology


@pytest.fixture
def pipeline_topology(topology_engine):
    """Create a pipeline topology with stages."""
    from hive.topology import TopologyType

    topology = topology_engine.create_topology(TopologyType.PIPELINE)

    # Stage 0: Researcher
    topology.add_agent("researcher-1", "Researcher", ["research"], stage=0)

    # Stage 1: Coder
    topology.add_agent("coder-1", "Coder", ["code"], stage=1)

    # Stage 2: Reviewer
    topology.add_agent("reviewer-1", "Reviewer", ["review"], stage=2)

    return topology


# ============================================================================
# Performance Tracking
# ============================================================================

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    baseline_value: float
    unit: str
    tolerance_percent: float = 10.0  # Allow 10% deviation


class PerformanceTracker:
    """Track and compare performance metrics."""

    def __init__(self) -> None:
        self._measurements: list[dict[str, Any]] = []
        self._baselines: dict[str, PerformanceBaseline] = {}

    def set_baseline(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tolerance_percent: float = 10.0,
    ) -> None:
        """Set a performance baseline."""
        self._baselines[metric_name] = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=value,
            unit=unit,
            tolerance_percent=tolerance_percent,
        )

    def record(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a measurement."""
        self._measurements.append({
            "metric": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

    def check_baseline(self, metric_name: str, value: float) -> tuple[bool, str]:
        """
        Check if value is within baseline tolerance.

        Returns:
            Tuple of (is_within_tolerance, message)
        """
        baseline = self._baselines.get(metric_name)
        if not baseline:
            return True, f"No baseline set for {metric_name}"

        deviation = abs(value - baseline.baseline_value) / baseline.baseline_value * 100
        is_ok = deviation <= baseline.tolerance_percent

        msg = (
            f"{metric_name}: {value:.2f} {baseline.unit} "
            f"(baseline: {baseline.baseline_value:.2f}, deviation: {deviation:.1f}%)"
        )

        return is_ok, msg

    def get_summary(self) -> dict[str, Any]:
        """Get summary of measurements."""
        if not self._measurements:
            return {"count": 0}

        by_metric: dict[str, list[float]] = {}
        for m in self._measurements:
            by_metric.setdefault(m["metric"], []).append(m["value"])

        summary = {"count": len(self._measurements), "metrics": {}}
        for metric, values in by_metric.items():
            summary["metrics"][metric] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        return summary

    def save_report(self, path: Path) -> None:
        """Save performance report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "baselines": {
                k: {"value": v.baseline_value, "unit": v.unit, "tolerance": v.tolerance_percent}
                for k, v in self._baselines.items()
            },
            "measurements": self._measurements,
        }
        path.write_text(json.dumps(report, indent=2))


@pytest.fixture(scope="session")
def perf_tracker() -> PerformanceTracker:
    """Session-scoped performance tracker."""
    tracker = PerformanceTracker()

    # Set default baselines
    tracker.set_baseline("agent_init_ms", 50.0, "ms", tolerance_percent=20.0)
    tracker.set_baseline("llm_call_mock_ms", 5.0, "ms", tolerance_percent=50.0)
    tracker.set_baseline("memory_store_ms", 1.0, "ms", tolerance_percent=50.0)
    tracker.set_baseline("topology_create_ms", 10.0, "ms", tolerance_percent=20.0)

    return tracker


# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    test_dir = tmp_path / "titan_test"
    test_dir.mkdir(exist_ok=True)
    return test_dir


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self) -> None:
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "TimingContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@pytest.fixture
def timing() -> type[TimingContext]:
    """Provide timing context manager."""
    return TimingContext


async def async_timeout(coro: Any, timeout_seconds: float) -> Any:
    """Run coroutine with timeout."""
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


@pytest.fixture
def async_with_timeout():
    """Provide async timeout helper."""
    return async_timeout


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")
    config.addinivalue_line("markers", "chaos: marks chaos/resilience tests")
    config.addinivalue_line("markers", "performance: marks performance tests")
    config.addinivalue_line("markers", "live_llm: marks tests that require real LLM API")
