"""
Adversarial Tests - Resource Exhaustion Prevention

Tests for termination conditions and resource limits.
"""

import pytest
from datetime import datetime, timedelta

from titan.orchestration.termination import (
    WorkflowState,
    TerminationCondition,
    TimeoutCondition,
    MaxIterationsCondition,
    MaxCostCondition,
    FailureThresholdCondition,
    SuccessCondition,
    ResourceExhaustionCondition,
    CompositeTerminationCondition,
    DefaultTerminationConditions,
    TerminationReason,
)
from titan.costs.budget import (
    BudgetTracker,
    BudgetConfig,
    BudgetExceededError,
)


class TestTerminationConditions:
    """Tests for termination condition enforcement."""

    @pytest.fixture
    def workflow_state(self):
        return WorkflowState(
            workflow_id="test-workflow",
            start_time=datetime.utcnow(),
        )

    # Timeout tests
    def test_timeout_not_reached(self, workflow_state):
        """Test that timeout condition doesn't trigger before time is up."""
        condition = TimeoutCondition(max_duration_seconds=60)
        result = condition.check(workflow_state)
        assert not result.should_terminate

    def test_timeout_reached(self, workflow_state):
        """Test that timeout condition triggers when time is up."""
        condition = TimeoutCondition(max_duration_seconds=0.001)
        # Artificially set start time in the past
        workflow_state.start_time = datetime.utcnow() - timedelta(seconds=10)
        result = condition.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.TIMEOUT

    # Iteration tests
    def test_max_iterations_not_reached(self, workflow_state):
        """Test that iteration condition doesn't trigger below limit."""
        condition = MaxIterationsCondition(max_iterations=10)
        workflow_state.iteration_count = 5
        result = condition.check(workflow_state)
        assert not result.should_terminate

    def test_max_iterations_reached(self, workflow_state):
        """Test that iteration condition triggers at limit."""
        condition = MaxIterationsCondition(max_iterations=10)
        workflow_state.iteration_count = 10
        result = condition.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.MAX_ITERATIONS

    # Cost tests
    def test_max_cost_not_reached(self, workflow_state):
        """Test that cost condition doesn't trigger below limit."""
        condition = MaxCostCondition(max_cost_usd=10.0)
        workflow_state.total_cost_usd = 5.0
        result = condition.check(workflow_state)
        assert not result.should_terminate

    def test_max_cost_reached(self, workflow_state):
        """Test that cost condition triggers at limit."""
        condition = MaxCostCondition(max_cost_usd=10.0)
        workflow_state.total_cost_usd = 10.0
        result = condition.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.MAX_COST

    # Failure threshold tests
    def test_failure_threshold_healthy(self, workflow_state):
        """Test that failure threshold doesn't trigger with healthy rate."""
        condition = FailureThresholdCondition(max_failure_rate=0.5, min_attempts=3)
        workflow_state.success_count = 8
        workflow_state.failure_count = 2  # 20% failure rate
        result = condition.check(workflow_state)
        assert not result.should_terminate

    def test_failure_threshold_exceeded(self, workflow_state):
        """Test that failure threshold triggers when exceeded."""
        condition = FailureThresholdCondition(max_failure_rate=0.5, min_attempts=3)
        workflow_state.success_count = 2
        workflow_state.failure_count = 8  # 80% failure rate
        result = condition.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.FAILURE_THRESHOLD

    def test_failure_threshold_min_attempts(self, workflow_state):
        """Test that failure threshold waits for min attempts."""
        condition = FailureThresholdCondition(max_failure_rate=0.5, min_attempts=5)
        workflow_state.success_count = 1
        workflow_state.failure_count = 2  # 66% failure but only 3 attempts
        result = condition.check(workflow_state)
        assert not result.should_terminate  # Not enough attempts yet

    # Memory exhaustion tests
    def test_memory_exhaustion_healthy(self, workflow_state):
        """Test that memory condition doesn't trigger when healthy."""
        condition = ResourceExhaustionCondition(max_memory_mb=1024)
        workflow_state.memory_usage_mb = 500
        result = condition.check(workflow_state)
        assert not result.should_terminate

    def test_memory_exhaustion_triggered(self, workflow_state):
        """Test that memory condition triggers when exhausted."""
        condition = ResourceExhaustionCondition(max_memory_mb=1024)
        workflow_state.memory_usage_mb = 1024
        result = condition.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.RESOURCE_EXHAUSTION

    # Success condition tests
    def test_success_condition_not_met(self, workflow_state):
        """Test that success condition doesn't trigger when not met."""
        condition = SuccessCondition(
            success_criteria=lambda s: s.success_count >= 10,
            description="Need 10 successes"
        )
        workflow_state.success_count = 5
        result = condition.check(workflow_state)
        assert not result.should_terminate

    def test_success_condition_met(self, workflow_state):
        """Test that success condition triggers when met."""
        condition = SuccessCondition(
            success_criteria=lambda s: s.success_count >= 10,
            description="Need 10 successes"
        )
        workflow_state.success_count = 10
        result = condition.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.SUCCESS


class TestCompositeConditions:
    """Tests for composite termination conditions."""

    @pytest.fixture
    def workflow_state(self):
        return WorkflowState(
            workflow_id="test-workflow",
            start_time=datetime.utcnow(),
        )

    def test_composite_any_triggers(self, workflow_state):
        """Test that composite triggers when any condition is met."""
        composite = CompositeTerminationCondition([
            MaxIterationsCondition(max_iterations=10),
            TimeoutCondition(max_duration_seconds=60),
        ], require_all=False)

        workflow_state.iteration_count = 10  # Triggers first condition
        result = composite.check(workflow_state)
        assert result.should_terminate
        assert result.reason == TerminationReason.MAX_ITERATIONS

    def test_composite_all_required(self, workflow_state):
        """Test that composite with require_all only triggers when all met."""
        composite = CompositeTerminationCondition([
            MaxIterationsCondition(max_iterations=10),
            MaxCostCondition(max_cost_usd=5.0),
        ], require_all=True)

        workflow_state.iteration_count = 10
        workflow_state.total_cost_usd = 3.0  # Below cost limit

        result = composite.check(workflow_state)
        assert not result.should_terminate  # Cost not exceeded

        workflow_state.total_cost_usd = 5.0
        result = composite.check(workflow_state)
        assert result.should_terminate  # Now both are met


class TestDefaultTerminationConditions:
    """Tests for default termination condition factories."""

    @pytest.fixture
    def workflow_state(self):
        return WorkflowState(
            workflow_id="test-workflow",
            start_time=datetime.utcnow(),
        )

    def test_agent_defaults_reasonable(self, workflow_state):
        """Test that agent defaults are reasonable."""
        conditions = DefaultTerminationConditions.for_agent(
            max_turns=20,
            timeout_seconds=300,
            max_cost_usd=2.0,
        )

        # Should not terminate initially
        result = conditions.check(workflow_state)
        assert not result.should_terminate

        # Should terminate at limits
        workflow_state.iteration_count = 20
        result = conditions.check(workflow_state)
        assert result.should_terminate

    def test_workflow_defaults_reasonable(self, workflow_state):
        """Test that workflow defaults are reasonable."""
        conditions = DefaultTerminationConditions.for_workflow(
            max_iterations=100,
            timeout_seconds=3600,
            max_cost_usd=50.0,
        )

        # Should not terminate initially
        result = conditions.check(workflow_state)
        assert not result.should_terminate

    def test_strict_mode_terminates_quickly(self, workflow_state):
        """Test that strict mode has tight limits."""
        conditions = DefaultTerminationConditions.strict(
            max_iterations=10,
            timeout_seconds=60,
        )

        workflow_state.iteration_count = 10
        result = conditions.check(workflow_state)
        assert result.should_terminate


class TestBudgetLimits:
    """Tests for budget limit enforcement."""

    @pytest.fixture
    def tracker(self):
        return BudgetTracker(BudgetConfig(
            session_limit_usd=10.0,
            daily_limit_usd=100.0,
            monthly_limit_usd=1000.0,
            enforce_limits=True,
        ))

    @pytest.mark.asyncio
    async def test_session_budget_created(self, tracker):
        """Test that session budgets are created correctly."""
        budget = await tracker.create_session_budget("session-1", limit_usd=5.0)
        assert budget.allocated_usd == 5.0
        assert budget.spent_usd == 0.0
        assert budget.remaining_usd == 5.0

    @pytest.mark.asyncio
    async def test_spend_recorded(self, tracker):
        """Test that spending is recorded correctly."""
        await tracker.create_session_budget("session-1", limit_usd=5.0)
        await tracker.record_spend(
            session_id="session-1",
            amount_usd=1.0,
            model="gpt-4o-mini",
            provider="openai",
        )

        budget = await tracker.get_budget("session-1")
        assert budget.spent_usd == 1.0
        assert budget.remaining_usd == 4.0

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises(self, tracker):
        """Test that exceeding budget raises error."""
        await tracker.create_session_budget("session-1", limit_usd=1.0)

        with pytest.raises(BudgetExceededError):
            await tracker.record_spend(
                session_id="session-1",
                amount_usd=2.0,
                model="gpt-4o",
                provider="openai",
            )

    @pytest.mark.asyncio
    async def test_can_afford_check(self, tracker):
        """Test can_afford checking."""
        await tracker.create_session_budget("session-1", limit_usd=5.0)

        can_afford, reason = await tracker.can_afford("session-1", 3.0)
        assert can_afford

        can_afford, reason = await tracker.can_afford("session-1", 10.0)
        assert not can_afford

    @pytest.mark.asyncio
    async def test_agent_budget_within_session(self, tracker):
        """Test that agent budget is limited by session budget."""
        await tracker.create_session_budget("session-1", limit_usd=5.0)
        await tracker.record_spend("session-1", amount_usd=3.0, model="test", provider="test")

        # Remaining session budget is 2.0, so agent budget should be capped
        agent_budget = await tracker.create_agent_budget(
            agent_id="agent-1",
            session_id="session-1",
            limit_usd=10.0,  # Request 10, but only 2 remaining
        )
        assert agent_budget.allocated_usd <= 2.0
