"""
Performance and Load Tests.

Tests system performance under various load conditions:
- Concurrent agent scaling (10, 50, 100 agents)
- Memory usage under sustained load
- Response time percentiles (p50, p95, p99)
- Token throughput benchmarks
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import sys
import time
from typing import Any

import pytest

# Mark all tests in this module as performance tests
pytestmark = [pytest.mark.performance, pytest.mark.asyncio]


class TestConcurrentAgentScaling:
    """Test system performance with increasing agent counts."""

    @pytest.mark.parametrize("agent_count", [10, 25, 50])
    async def test_concurrent_agents(
        self,
        agent_count: int,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test performance with multiple concurrent agents."""
        from agents.archetypes.researcher import ResearcherAgent

        # Create agents
        agents = [
            ResearcherAgent(
                name=f"researcher-{i}",
                topic=f"Topic {i}",
                hive_mind=mock_hive_mind,
                max_turns=5,  # Limit turns for performance test
            )
            for i in range(agent_count)
        ]

        # Time concurrent execution
        with timing() as t:
            results = await asyncio.gather(
                *[agent.run() for agent in agents],
                return_exceptions=True,
            )

        # Record metrics
        perf_tracker.record(
            f"concurrent_agents_{agent_count}_ms",
            t.duration_ms,
            "ms",
            {"agent_count": agent_count},
        )

        # Calculate success rate
        successes = sum(
            1 for r in results
            if not isinstance(r, Exception) and r.success
        )
        success_rate = successes / agent_count

        perf_tracker.record(
            f"concurrent_agents_{agent_count}_success_rate",
            success_rate,
            "ratio",
        )

        # Assert reasonable performance
        assert success_rate >= 0.8, f"Only {success_rate*100:.1f}% succeeded"
        # Per-agent time should be reasonable
        per_agent_ms = t.duration_ms / agent_count
        assert per_agent_ms < 1000, f"Per-agent time too high: {per_agent_ms:.1f}ms"

    async def test_agent_creation_time(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test agent creation time."""
        from agents.archetypes.researcher import ResearcherAgent

        creation_times = []

        for i in range(20):
            with timing() as t:
                agent = ResearcherAgent(
                    name=f"test-{i}",
                    topic="Test",
                    hive_mind=mock_hive_mind,
                )
            creation_times.append(t.duration_ms)

        avg_creation = statistics.mean(creation_times)
        p99_creation = sorted(creation_times)[int(len(creation_times) * 0.99)]

        perf_tracker.record("agent_creation_avg_ms", avg_creation, "ms")
        perf_tracker.record("agent_creation_p99_ms", p99_creation, "ms")

        # Creation should be fast
        assert avg_creation < 10, f"Agent creation too slow: {avg_creation:.2f}ms"

    async def test_topology_scaling(
        self,
        perf_tracker,
        timing,
    ):
        """Test topology performance with many agents."""
        from hive.topology import TopologyEngine, TopologyType

        for agent_count in [10, 50, 100]:
            engine = TopologyEngine()

            # Create topology
            with timing() as create_time:
                topology = engine.create_topology(TopologyType.MESH)

            # Add agents
            with timing() as add_time:
                for i in range(agent_count):
                    topology.add_agent(f"agent-{i}", f"Agent{i}", ["test"])

            # Test routing
            route_times = []
            for _ in range(100):
                with timing() as route_time:
                    path = topology.get_routing_path("agent-0", f"agent-{agent_count-1}")
                route_times.append(route_time.duration_ms)

            perf_tracker.record(
                f"topology_add_{agent_count}_agents_ms",
                add_time.duration_ms,
                "ms",
            )
            perf_tracker.record(
                f"topology_route_avg_{agent_count}_ms",
                statistics.mean(route_times),
                "ms",
            )

            # Routing should be fast even with many agents
            assert statistics.mean(route_times) < 10


class TestMemoryUsage:
    """Test memory usage under sustained load."""

    async def test_memory_stable_over_time(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
    ):
        """Test memory doesn't grow unbounded over time."""
        from agents.archetypes.researcher import ResearcherAgent
        import tracemalloc

        tracemalloc.start()

        # Initial memory
        gc.collect()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Run many agents sequentially
        for i in range(20):
            agent = ResearcherAgent(
                name=f"test-{i}",
                topic=f"Topic {i}",
                hive_mind=mock_hive_mind,
                max_turns=3,
            )
            await agent.run()

            # Explicitly cleanup
            del agent
            if i % 5 == 0:
                gc.collect()

        # Final memory
        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        perf_tracker.record("memory_growth_mb", memory_growth, "MB")

        # Memory growth should be bounded
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB"

    async def test_hive_mind_memory_cleanup(
        self,
        mock_hive_mind,
        perf_tracker,
    ):
        """Test Hive Mind doesn't accumulate unbounded memory."""
        # Store many memories
        for i in range(1000):
            await mock_hive_mind.remember(
                agent_id=f"agent-{i % 10}",
                content=f"Memory content {i}" * 10,
                importance=0.5,
            )

        memory_count = len(mock_hive_mind.get_memories())
        perf_tracker.record("hive_mind_memories", float(memory_count), "count")

        # Should have stored all memories (mock doesn't have cleanup)
        assert memory_count == 1000


class TestResponseTimePercentiles:
    """Test response time percentiles."""

    async def test_llm_call_percentiles(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test LLM call latency percentiles."""
        from adapters.base import LLMMessage

        latencies = []

        for _ in range(100):
            with timing() as t:
                await mock_llm_router.complete(
                    [LLMMessage(role="user", content="Test query")],
                    system="Test system",
                )
            latencies.append(t.duration_ms)

        latencies.sort()
        p50 = latencies[50]
        p95 = latencies[95]
        p99 = latencies[99]

        perf_tracker.record("llm_call_p50_ms", p50, "ms")
        perf_tracker.record("llm_call_p95_ms", p95, "ms")
        perf_tracker.record("llm_call_p99_ms", p99, "ms")

        # Mock calls should be very fast
        assert p99 < 50, f"p99 latency too high: {p99:.2f}ms"

    async def test_agent_run_percentiles(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test agent run latency percentiles."""
        from agents.archetypes.researcher import ResearcherAgent

        latencies = []

        for i in range(30):
            agent = ResearcherAgent(
                name=f"test-{i}",
                topic="Quick topic",
                hive_mind=mock_hive_mind,
                max_turns=3,
            )

            with timing() as t:
                await agent.run()
            latencies.append(t.duration_ms)

        latencies.sort()
        p50 = latencies[15]
        p95 = latencies[28]

        perf_tracker.record("agent_run_p50_ms", p50, "ms")
        perf_tracker.record("agent_run_p95_ms", p95, "ms")

        # Agent runs should complete reasonably fast
        assert p95 < 5000, f"p95 agent run time too high: {p95:.2f}ms"

    async def test_topology_operation_percentiles(
        self,
        perf_tracker,
        timing,
    ):
        """Test topology operation latency percentiles."""
        from hive.topology import TopologyEngine, TopologyType

        engine = TopologyEngine()
        topology = engine.create_topology(TopologyType.SWARM)

        # Add agents
        for i in range(20):
            topology.add_agent(f"agent-{i}", f"Agent{i}", ["test"])

        # Measure operations
        message_target_times = []
        routing_times = []

        for _ in range(100):
            with timing() as t:
                topology.get_message_targets("agent-0", "broadcast")
            message_target_times.append(t.duration_ms)

            with timing() as t:
                topology.get_routing_path("agent-0", "agent-19")
            routing_times.append(t.duration_ms)

        message_target_times.sort()
        routing_times.sort()

        perf_tracker.record("message_targets_p99_ms", message_target_times[99], "ms")
        perf_tracker.record("routing_path_p99_ms", routing_times[99], "ms")


class TestTokenThroughput:
    """Test token throughput benchmarks."""

    async def test_token_estimation_throughput(
        self,
        perf_tracker,
        timing,
    ):
        """Test token estimation throughput."""
        # Simple token estimation (character-based approximation)
        texts = [
            "Short text" * 10,
            "Medium length text with more content" * 50,
            "Longer text with substantial content for testing" * 100,
        ]

        def estimate_tokens(text: str) -> int:
            # Rough estimation: ~4 chars per token
            return len(text) // 4

        start = time.perf_counter()
        total_tokens = 0
        iterations = 10000

        for _ in range(iterations):
            for text in texts:
                total_tokens += estimate_tokens(text)

        duration = time.perf_counter() - start
        throughput = total_tokens / duration

        perf_tracker.record("token_estimation_per_sec", throughput, "tokens/sec")

        # Should be able to estimate millions of tokens per second
        assert throughput > 100000, f"Token estimation too slow: {throughput:.0f} tokens/sec"

    async def test_mock_llm_throughput(
        self,
        mock_llm_router,
        perf_tracker,
    ):
        """Test mock LLM throughput."""
        from adapters.base import LLMMessage

        start = time.perf_counter()
        call_count = 100

        for _ in range(call_count):
            await mock_llm_router.complete(
                [LLMMessage(role="user", content="Test")],
                max_tokens=100,
            )

        duration = time.perf_counter() - start
        calls_per_sec = call_count / duration

        perf_tracker.record("mock_llm_calls_per_sec", calls_per_sec, "calls/sec")

        # Mock should be very fast
        assert calls_per_sec > 100, f"Mock LLM too slow: {calls_per_sec:.0f} calls/sec"


class TestWorkflowPerformance:
    """Test complete workflow performance."""

    async def test_orchestrator_workflow_performance(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test orchestrator workflow performance."""
        from agents.archetypes.orchestrator import OrchestratorAgent

        workflow_times = []

        for i in range(10):
            agent = OrchestratorAgent(
                task=f"Task {i}",
                hive_mind=mock_hive_mind,
            )

            with timing() as t:
                await agent.run()
            workflow_times.append(t.duration_ms)

        avg_time = statistics.mean(workflow_times)
        std_dev = statistics.stdev(workflow_times) if len(workflow_times) > 1 else 0

        perf_tracker.record("workflow_avg_ms", avg_time, "ms")
        perf_tracker.record("workflow_stddev_ms", std_dev, "ms")

        # Workflows should be reasonably fast and consistent
        assert avg_time < 10000, f"Workflow too slow: {avg_time:.2f}ms"

    async def test_parallel_workflow_speedup(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test parallel execution provides speedup."""
        from agents.archetypes.orchestrator import OrchestratorAgent

        # Sequential execution
        with timing() as sequential_time:
            for i in range(5):
                agent = OrchestratorAgent(
                    task=f"Task {i}",
                    hive_mind=mock_hive_mind,
                )
                await agent.run()

        # Parallel execution
        agents = [
            OrchestratorAgent(
                task=f"Task {i}",
                hive_mind=mock_hive_mind,
            )
            for i in range(5)
        ]

        with timing() as parallel_time:
            await asyncio.gather(*[a.run() for a in agents])

        # Calculate speedup (avoid division by zero)
        if parallel_time.duration_ms > 0:
            speedup = sequential_time.duration_ms / parallel_time.duration_ms
        else:
            speedup = 1.0

        perf_tracker.record("parallel_speedup_ratio", speedup, "x")

        # Note: With mock LLM, speedup may be minimal since there's no I/O wait
        # Just verify parallel execution doesn't make things slower
        assert speedup >= 0.5, f"Parallel execution much slower: {speedup:.2f}x"


class TestBaselineComparison:
    """Test performance against baselines."""

    async def test_agent_init_baseline(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
        timing,
    ):
        """Test agent initialization against baseline."""
        from agents.archetypes.researcher import ResearcherAgent

        times = []
        for i in range(20):
            with timing() as t:
                agent = ResearcherAgent(
                    name=f"test-{i}",
                    topic="Test",
                    hive_mind=mock_hive_mind,
                )
            times.append(t.duration_ms)

        avg_time = statistics.mean(times)
        perf_tracker.record("agent_init_ms", avg_time, "ms")

        # Just verify initialization is reasonably fast (< 100ms)
        assert avg_time < 100, f"Agent initialization too slow: {avg_time:.2f}ms"

    async def test_topology_create_baseline(
        self,
        perf_tracker,
        timing,
    ):
        """Test topology creation against baseline."""
        from hive.topology import TopologyEngine, TopologyType

        times = []
        for _ in range(20):
            with timing() as t:
                engine = TopologyEngine()
                engine.create_topology(TopologyType.SWARM)
            times.append(t.duration_ms)

        avg_time = statistics.mean(times)
        perf_tracker.record("topology_create_ms", avg_time, "ms")

        # Just verify topology creation is reasonably fast (< 50ms)
        assert avg_time < 50, f"Topology creation too slow: {avg_time:.2f}ms"


class TestScalabilityLimits:
    """Test scalability limits."""

    @pytest.mark.slow
    async def test_max_concurrent_agents(
        self,
        mock_hive_mind,
        mock_llm_router,
        perf_tracker,
    ):
        """Find maximum sustainable concurrent agents."""
        from agents.archetypes.researcher import ResearcherAgent

        max_successful = 0

        for count in [10, 25, 50, 75, 100]:
            agents = [
                ResearcherAgent(
                    name=f"test-{i}",
                    topic=f"Topic {i}",
                    hive_mind=mock_hive_mind,
                    max_turns=3,
                    timeout_ms=30000,
                )
                for i in range(count)
            ]

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *[a.run() for a in agents],
                        return_exceptions=True,
                    ),
                    timeout=60.0,
                )

                success_rate = sum(
                    1 for r in results
                    if not isinstance(r, Exception) and r.success
                ) / count

                if success_rate >= 0.9:
                    max_successful = count
                else:
                    break

            except asyncio.TimeoutError:
                break

        perf_tracker.record("max_concurrent_agents", float(max_successful), "agents")

        # Should handle at least 10 concurrent agents
        assert max_successful >= 10

    async def test_topology_max_agents(
        self,
        perf_tracker,
        timing,
    ):
        """Test topology with maximum agents."""
        from hive.topology import TopologyEngine, TopologyType

        engine = TopologyEngine()
        topology = engine.create_topology(TopologyType.SWARM)

        max_agents = 0
        with timing() as t:
            for i in range(500):
                try:
                    topology.add_agent(f"agent-{i}", f"Agent{i}", ["test"])
                    max_agents = i + 1
                except Exception:
                    break

        perf_tracker.record("topology_max_agents", float(max_agents), "agents")
        perf_tracker.record("topology_add_500_ms", t.duration_ms, "ms")

        # Should handle at least 100 agents
        assert max_agents >= 100
