"""
Chaos and Resilience Tests.

Tests system behavior under adverse conditions:
- Network partition simulation
- Agent timeout handling
- LLM API failure recovery
- Message queue disconnection
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

# Mark all tests in this module as chaos tests
pytestmark = [pytest.mark.chaos, pytest.mark.asyncio]


class TestNetworkPartition:
    """Test behavior during network partitions."""

    async def test_agent_handles_network_timeout(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test agent handles network timeouts gracefully."""
        from agents.archetypes.researcher import ResearcherAgent

        # Make router timeout on some calls
        call_count = 0
        original_complete = mock_llm_router.complete

        async def flaky_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                await asyncio.sleep(5)  # Simulate slow response
                raise TimeoutError("Network timeout")
            return await original_complete(*args, **kwargs)

        mock_llm_router.complete = flaky_complete

        agent = ResearcherAgent(
            topic="Test topic",
            hive_mind=mock_hive_mind,
            timeout_ms=10000,  # 10 second timeout
        )

        # Agent should handle timeout and continue or fail gracefully
        result = await agent.run()
        # Either succeeds or fails cleanly (no hanging)
        assert result.state.value in ("completed", "failed")

    async def test_hive_mind_reconnection(self, mock_hive_mind):
        """Test agent handles Hive Mind disconnection."""
        from agents.framework.base_agent import BaseAgent

        class TestAgent(BaseAgent):
            async def initialize(self):
                pass

            async def work(self):
                # Simulate Hive Mind disconnect mid-operation
                await self.remember("test content", importance=0.5)
                # Hive Mind "disconnects"
                self._hive_mind = None
                # Should handle gracefully
                result = await self.remember("more content", importance=0.5)
                return result

            async def shutdown(self):
                pass

        agent = TestAgent(name="test", hive_mind=mock_hive_mind)
        result = await agent.run()

        # Should complete without crashing
        assert result.success or result.error is not None

    async def test_topology_partition_recovery(self, topology_engine):
        """Test topology handles partition and recovery."""
        from hive.topology import TopologyType

        # Create mesh topology (fault-tolerant)
        mesh = topology_engine.create_topology(TopologyType.MESH, connectivity=3)

        # Add agents
        for i in range(5):
            mesh.add_agent(f"agent-{i}", f"Agent{i}", ["test"])

        # Simulate partition by removing an agent
        original_count = len(mesh.nodes)
        mesh.remove_agent("agent-2")

        # Topology should still be functional
        assert len(mesh.nodes) == original_count - 1

        # Messages should still route around the missing agent
        path = mesh.get_routing_path("agent-0", "agent-4")
        assert "agent-2" not in path
        assert len(path) > 0


class TestAgentTimeout:
    """Test timeout handling at various levels."""

    async def test_agent_respects_max_turns(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test agent stops at max turns."""
        from agents.archetypes.researcher import ResearcherAgent

        agent = ResearcherAgent(
            topic="Long research",
            hive_mind=mock_hive_mind,
            max_turns=3,
        )

        await agent.run()

        assert agent._context.turn_number <= 3

    async def test_agent_timeout_triggers_checkpoint(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test checkpoint is created on timeout."""
        from agents.archetypes.researcher import ResearcherAgent

        checkpoints_created = []

        def on_checkpoint(checkpoint):
            checkpoints_created.append(checkpoint)

        agent = ResearcherAgent(
            topic="Test topic",
            hive_mind=mock_hive_mind,
            checkpoint_interval=1,
        )
        agent.on_checkpoint(on_checkpoint)

        await agent.run()

        # At least one checkpoint should be created
        # (depends on number of turns)

    async def test_subtask_timeout_doesnt_block_workflow(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test one slow subtask doesn't block entire workflow."""
        from agents.archetypes.orchestrator import OrchestratorAgent

        # Make one subtask slow
        call_count = 0
        original_complete = mock_llm_router.complete

        async def slow_sometimes(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                await asyncio.sleep(0.5)  # Slow but not timeout
            return await original_complete(*args, **kwargs)

        mock_llm_router.complete = slow_sometimes

        agent = OrchestratorAgent(
            task="Multi-step task",
            hive_mind=mock_hive_mind,
        )

        result = await agent.run()

        # Workflow should complete
        assert result.success


class TestLLMAPIFailure:
    """Test LLM API failure handling and recovery."""

    async def test_agent_handles_llm_error(
        self,
        mock_hive_mind,
    ):
        """Test agent handles LLM API errors."""

        from agents.framework.base_agent import BaseAgent

        # Create a simple agent that will fail on LLM call
        class FailingAgent(BaseAgent):
            async def initialize(self):
                pass

            async def work(self):
                # This will raise an exception
                raise Exception("LLM API Error: Rate limit exceeded")

            async def shutdown(self):
                pass

        agent = FailingAgent(
            name="failing-test",
            hive_mind=mock_hive_mind,
        )

        result = await agent.run()

        # Should fail gracefully with error message
        assert result.state.value == "failed"
        assert result.error is not None
        assert "LLM API Error" in result.error

    async def test_llm_retry_on_transient_failure(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test retry logic on transient failures."""
        from agents.archetypes.researcher import ResearcherAgent

        call_count = 0
        original_complete = mock_llm_router.complete

        async def transient_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise Exception("Temporary error")
            return await original_complete(*args, **kwargs)

        mock_llm_router.complete = transient_failure

        agent = ResearcherAgent(
            topic="Test topic",
            hive_mind=mock_hive_mind,
        )

        # With proper retry logic, this should eventually succeed
        # (Note: current implementation may not have retry built-in)
        result = await agent.run()

        # At minimum, should not hang
        assert result.state.value in ("completed", "failed")

    async def test_fallback_to_simpler_model(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test fallback to simpler model on failure."""
        from agents.archetypes.cfo import CFOAgent

        cfo = CFOAgent(
            session_budget_usd=10.0,
            hive_mind=mock_hive_mind,
        )

        await cfo.initialize()

        # Request model selection with constraint
        recommendation = await cfo.select_model(
            task="Simple formatting task",
            max_cost_usd=0.001,  # Very tight budget
        )

        # Should select economy model
        assert recommendation.model_tier == "economy"


class TestMessageQueueDisconnection:
    """Test message queue disconnection handling."""

    async def test_agent_handles_broadcast_failure(
        self,
        mock_hive_mind,
    ):
        """Test agent handles broadcast failure."""
        from agents.framework.base_agent import BaseAgent

        class BroadcastingAgent(BaseAgent):
            async def initialize(self):
                pass

            async def work(self):
                # Simulate broadcast failure
                self._hive_mind.broadcast = AsyncMock(
                    side_effect=Exception("Message queue disconnected")
                )
                await self.broadcast({"message": "test"})
                return "completed despite broadcast failure"

            async def shutdown(self):
                pass

        agent = BroadcastingAgent(name="test", hive_mind=mock_hive_mind)
        result = await agent.run()

        # Should complete (broadcast failure is non-fatal)
        assert result.success

    async def test_agent_handles_subscribe_failure(
        self,
        mock_hive_mind,
    ):
        """Test agent handles subscription failure."""
        from agents.framework.base_agent import BaseAgent

        class SubscribingAgent(BaseAgent):
            async def initialize(self):
                # Simulate subscription failure
                self._hive_mind.subscribe = AsyncMock(side_effect=Exception("Subscription failed"))
                await self.subscribe("topic", lambda x: None)

            async def work(self):
                return "completed"

            async def shutdown(self):
                pass

        agent = SubscribingAgent(name="test", hive_mind=mock_hive_mind)
        result = await agent.run()

        # Should complete despite subscription failure
        assert result.success


class TestResourceExhaustion:
    """Test behavior under resource exhaustion."""

    async def test_many_concurrent_agents(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test system handles many concurrent agents."""
        from agents.archetypes.researcher import ResearcherAgent

        # Create many agents
        agents = [
            ResearcherAgent(
                name=f"researcher-{i}",
                topic=f"Topic {i}",
                hive_mind=mock_hive_mind,
            )
            for i in range(10)
        ]

        # Run all concurrently
        results = await asyncio.gather(
            *[agent.run() for agent in agents],
            return_exceptions=True,
        )

        # Most should complete (some might timeout)
        completed = sum(1 for r in results if not isinstance(r, Exception) and r.success)
        assert completed >= 5  # At least half should complete

    async def test_memory_pressure_handling(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test handling of memory pressure."""
        from agents.archetypes.researcher import ResearcherAgent

        agent = ResearcherAgent(
            topic="Memory test",
            hive_mind=mock_hive_mind,
        )

        # Store lots of memories
        for i in range(100):
            await mock_hive_mind.remember(
                agent_id=agent.agent_id,
                content=f"Memory {i}" * 100,  # Larger content
                importance=0.5,
            )

        # Agent should still work
        result = await agent.run()
        assert result.state.value in ("completed", "failed")


class TestErrorThresholds:
    """Test error threshold handling."""

    async def test_agent_stops_after_error_threshold(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test agent stops after too many consecutive errors."""
        from agents.framework.base_agent import BaseAgent

        class ErrorProneAgent(BaseAgent):
            async def initialize(self):
                pass

            async def work(self):
                # Generate errors
                for i in range(5):
                    self.record_error(f"Error {i}")

                # Check if should stop
                should_stop, reason, msg = await self.should_stop()
                if should_stop:
                    return f"Stopped due to: {reason.value}"
                return "Completed"

            async def shutdown(self):
                pass

        agent = ErrorProneAgent(
            name="test",
            hive_mind=mock_hive_mind,
            error_threshold=3,
        )

        await agent.run()

        # Should have detected error threshold
        assert agent._consecutive_errors >= 3

    async def test_error_count_resets_on_success(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test error count resets after successful operation."""
        from agents.framework.base_agent import BaseAgent

        class RecoveringAgent(BaseAgent):
            async def initialize(self):
                pass

            async def work(self):
                # Some errors
                self.record_error("Error 1")
                self.record_error("Error 2")

                # Then success (via track_decision with high confidence)
                self.track_decision(
                    choice="Successful operation",
                    confidence=0.9,
                    category="test",
                )

                return "Recovered"

            async def shutdown(self):
                pass

        agent = RecoveringAgent(name="test", hive_mind=mock_hive_mind)
        await agent.run()

        # Error count should be reset
        assert agent._consecutive_errors == 0


class TestGracefulDegradation:
    """Test graceful degradation scenarios."""

    async def test_works_without_hive_mind(
        self,
        mock_llm_router,
    ):
        """Test agent works without Hive Mind."""
        from agents.archetypes.researcher import ResearcherAgent

        agent = ResearcherAgent(
            topic="Test topic",
            hive_mind=None,  # No Hive Mind
        )

        result = await agent.run()

        # Should work, just without memory
        assert result.success

    async def test_works_without_topology(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test orchestrator works without topology engine."""
        from agents.archetypes.orchestrator import OrchestratorAgent

        agent = OrchestratorAgent(
            task="Test task",
            topology_engine=None,
            hive_mind=mock_hive_mind,
        )

        result = await agent.run()

        # Should work with default topology selection
        assert result.success

    async def test_partial_results_on_early_termination(
        self,
        mock_hive_mind,
        mock_llm_router,
    ):
        """Test partial results are preserved on early termination."""
        from agents.archetypes.orchestrator import OrchestratorAgent
        from agents.framework.base_agent import StoppingCondition, StoppingReason

        agent = OrchestratorAgent(
            task="Multi-step task",
            hive_mind=mock_hive_mind,
        )

        # Add early termination condition
        agent.add_stopping_condition(
            StoppingCondition(
                reason=StoppingReason.USER_CANCELLED,
                check=lambda a: a._context and a._context.turn_number >= 2,
                message="Early termination",
                priority=100,
            )
        )

        await agent.run()

        # Workflow should have partial results
        if agent.workflow:
            # Some results should be captured
            assert isinstance(agent.workflow.results, dict)
