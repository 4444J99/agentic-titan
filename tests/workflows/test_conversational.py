"""Tests for conversational interleaving in inquiry engine."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from titan.workflows.inquiry_config import (
    InfluenceMode,
    UserInterjection,
    InquiryWorkflow,
    InquiryStage,
    CognitiveStyle,
    QUICK_INQUIRY_WORKFLOW,
)
from titan.workflows.inquiry_engine import (
    InquiryEngine,
    InquirySession,
    InquiryStatus,
)


class TestUserInterjection:
    """Tests for UserInterjection dataclass."""

    def test_create_interjection(self) -> None:
        """Test creating a user interjection."""
        interjection = UserInterjection(
            content="Please focus on environmental aspects",
            injected_at_stage=2,
            influence_mode=InfluenceMode.REDIRECT,
        )

        assert interjection.content == "Please focus on environmental aspects"
        assert interjection.injected_at_stage == 2
        assert interjection.influence_mode == InfluenceMode.REDIRECT
        assert interjection.processed is False

    def test_default_influence_mode(self) -> None:
        """Test default influence mode is CONTEXT."""
        interjection = UserInterjection(
            content="Additional context",
            injected_at_stage=0,
        )

        assert interjection.influence_mode == InfluenceMode.CONTEXT

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        interjection = UserInterjection(
            content="Test content",
            injected_at_stage=1,
            influence_mode=InfluenceMode.CLARIFY,
        )

        d = interjection.to_dict()

        assert d["content"] == "Test content"
        assert d["injected_at_stage"] == 1
        assert d["influence_mode"] == "clarify"
        assert d["processed"] is False
        assert "timestamp" in d

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "content": "Clarification text",
            "injected_at_stage": 3,
            "influence_mode": "redirect",
            "timestamp": datetime.now().isoformat(),
            "processed": True,
        }

        interjection = UserInterjection.from_dict(data)

        assert interjection.content == "Clarification text"
        assert interjection.injected_at_stage == 3
        assert interjection.influence_mode == InfluenceMode.REDIRECT
        assert interjection.processed is True


class TestInfluenceMode:
    """Tests for InfluenceMode enum."""

    def test_influence_mode_values(self) -> None:
        """Test influence mode values."""
        assert InfluenceMode.CONTEXT.value == "context"
        assert InfluenceMode.REDIRECT.value == "redirect"
        assert InfluenceMode.CLARIFY.value == "clarify"


class TestSessionInterjections:
    """Tests for interjections in InquirySession."""

    @pytest.fixture
    def engine(self) -> InquiryEngine:
        """Create an inquiry engine for testing."""
        return InquiryEngine(llm_caller=None)

    @pytest.mark.asyncio
    async def test_session_has_interjections_list(self, engine: InquiryEngine) -> None:
        """Test sessions have interjections list."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        assert hasattr(session, "interjections")
        assert session.interjections == []

    @pytest.mark.asyncio
    async def test_session_get_unprocessed_interjections(self, engine: InquiryEngine) -> None:
        """Test getting unprocessed interjections."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        # Add interjections
        session.interjections.append(
            UserInterjection("First", 0, processed=False)
        )
        session.interjections.append(
            UserInterjection("Second", 1, processed=True)
        )
        session.interjections.append(
            UserInterjection("Third", 1, processed=False)
        )

        unprocessed = session.get_unprocessed_interjections()

        assert len(unprocessed) == 2
        assert unprocessed[0].content == "First"
        assert unprocessed[1].content == "Third"

    @pytest.mark.asyncio
    async def test_session_get_interjections_for_stage(self, engine: InquiryEngine) -> None:
        """Test getting interjections relevant to a stage."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        session.interjections = [
            UserInterjection("Stage 0 input", 0, processed=False),
            UserInterjection("Stage 1 input", 1, processed=False),
            UserInterjection("Stage 2 input", 2, processed=False),
        ]

        # For stage 2, should get interjections from stages 0 and 1
        relevant = session.get_interjections_for_stage(2)

        assert len(relevant) == 2


class TestPauseResume:
    """Tests for pause/resume functionality."""

    @pytest.fixture
    def engine(self) -> InquiryEngine:
        """Create an inquiry engine for testing."""
        return InquiryEngine(llm_caller=None)

    @pytest.mark.asyncio
    async def test_pause_session(self, engine: InquiryEngine) -> None:
        """Test pausing a session."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.status = InquiryStatus.RUNNING

        result = engine.pause_session(session.id)

        assert result is True
        assert session.pause_requested is True

    @pytest.mark.asyncio
    async def test_pause_nonexistent_session(self, engine: InquiryEngine) -> None:
        """Test pausing nonexistent session returns False."""
        result = engine.pause_session("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_pause_completed_session(self, engine: InquiryEngine) -> None:
        """Test cannot pause completed session."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.status = InquiryStatus.COMPLETED

        result = engine.pause_session(session.id)

        assert result is False

    @pytest.mark.asyncio
    async def test_resume_session(self, engine: InquiryEngine) -> None:
        """Test resuming a paused session."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.status = InquiryStatus.PAUSED

        result = engine.resume_session(session.id)

        assert result is True
        assert session.status == InquiryStatus.RUNNING
        assert session.pause_requested is False

    @pytest.mark.asyncio
    async def test_resume_running_session(self, engine: InquiryEngine) -> None:
        """Test cannot resume non-paused session."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.status = InquiryStatus.RUNNING

        result = engine.resume_session(session.id)

        assert result is False


class TestInjectUserInput:
    """Tests for inject_user_input functionality."""

    @pytest.fixture
    def engine(self) -> InquiryEngine:
        """Create an inquiry engine for testing."""
        return InquiryEngine(llm_caller=None)

    @pytest.mark.asyncio
    async def test_inject_user_input(self, engine: InquiryEngine) -> None:
        """Test injecting user input."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.current_stage = 1

        interjection = engine.inject_user_input(
            session.id,
            "Please focus on technical aspects",
            InfluenceMode.REDIRECT,
        )

        assert interjection is not None
        assert interjection.content == "Please focus on technical aspects"
        assert interjection.injected_at_stage == 1
        assert interjection.influence_mode == InfluenceMode.REDIRECT
        assert len(session.interjections) == 1

    @pytest.mark.asyncio
    async def test_inject_with_string_mode(self, engine: InquiryEngine) -> None:
        """Test injecting with string mode converts to enum."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        interjection = engine.inject_user_input(
            session.id,
            "Clarification text",
            "clarify",
        )

        assert interjection.influence_mode == InfluenceMode.CLARIFY

    @pytest.mark.asyncio
    async def test_inject_nonexistent_session(self, engine: InquiryEngine) -> None:
        """Test injecting to nonexistent session returns None."""
        result = engine.inject_user_input(
            "nonexistent-id",
            "Content",
            InfluenceMode.CONTEXT,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_injections(self, engine: InquiryEngine) -> None:
        """Test multiple injections accumulate."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        engine.inject_user_input(session.id, "First input", InfluenceMode.CONTEXT)
        engine.inject_user_input(session.id, "Second input", InfluenceMode.REDIRECT)

        assert len(session.interjections) == 2


class TestInterleavedWorkflow:
    """Tests for run_interleaved_workflow."""

    @pytest.fixture
    def engine(self) -> InquiryEngine:
        """Create an inquiry engine for testing."""
        return InquiryEngine(llm_caller=None)

    @pytest.mark.asyncio
    async def test_interleaved_workflow_yields_events(self, engine: InquiryEngine) -> None:
        """Test interleaved workflow yields correct events."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        events = []
        async for event in engine.run_interleaved_workflow(session):
            events.append(event)
            # Stop after first stage to keep test short
            if event.get("type") == "stage_completed":
                engine.cancel_session(session.id)
                break

        event_types = [e.get("type") for e in events]
        assert "session_started" in event_types
        assert "stage_started" in event_types

    @pytest.mark.asyncio
    async def test_interleaved_workflow_processes_interjections(
        self, engine: InquiryEngine
    ) -> None:
        """Test interleaved workflow processes interjections."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        # Add an interjection before starting
        session.interjections.append(
            UserInterjection("Focus on AI", 0, InfluenceMode.REDIRECT)
        )

        events = []
        async for event in engine.run_interleaved_workflow(session):
            events.append(event)
            if event.get("type") == "stage_completed":
                break

        # Check interjection was processed
        event_types = [e.get("type") for e in events]
        assert "interjection_processing" in event_types

    @pytest.mark.asyncio
    async def test_interleaved_workflow_handles_pause(self, engine: InquiryEngine) -> None:
        """Test interleaved workflow handles pause request."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        events = []

        async def run_workflow() -> None:
            async for event in engine.run_interleaved_workflow(session):
                events.append(event)
                # Request pause after first stage starts
                if event.get("type") == "stage_started":
                    session.pause_requested = True
                # Resume after pause
                elif event.get("type") == "session_paused":
                    # In real usage, user would resume; here we cancel
                    session.status = InquiryStatus.CANCELLED

        await run_workflow()

        event_types = [e.get("type") for e in events]
        assert "session_paused" in event_types or "session_cancelled" in event_types


class TestInterjectionContext:
    """Tests for _get_interjection_context method."""

    @pytest.fixture
    def engine(self) -> InquiryEngine:
        """Create an inquiry engine for testing."""
        return InquiryEngine(llm_caller=None)

    @pytest.mark.asyncio
    async def test_get_interjection_context_empty(self, engine: InquiryEngine) -> None:
        """Test empty context when no interjections."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.current_stage = 0

        context = engine._get_interjection_context(session)

        assert context == ""

    @pytest.mark.asyncio
    async def test_get_interjection_context_with_interjections(
        self, engine: InquiryEngine
    ) -> None:
        """Test context includes relevant interjections."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.current_stage = 1

        session.interjections = [
            UserInterjection("Context info", 0, InfluenceMode.CONTEXT),
            UserInterjection("Redirect request", 0, InfluenceMode.REDIRECT),
        ]

        context = engine._get_interjection_context(session)

        assert "User clarifications" in context
        assert "Context info" in context
        assert "Redirect request" in context
        assert "Additional context" in context
        assert "Direction change" in context


class TestSessionToDict:
    """Tests for session serialization with interjections."""

    @pytest.fixture
    def engine(self) -> InquiryEngine:
        """Create an inquiry engine for testing."""
        return InquiryEngine(llm_caller=None)

    @pytest.mark.asyncio
    async def test_session_to_dict_includes_interjections(
        self, engine: InquiryEngine
    ) -> None:
        """Test session serialization includes interjections."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)

        session.interjections.append(
            UserInterjection("Test content", 0, InfluenceMode.CONTEXT)
        )

        d = session.to_dict()

        assert "interjections" in d
        assert len(d["interjections"]) == 1
        assert d["interjections"][0]["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_session_to_dict_includes_temporal_fields(
        self, engine: InquiryEngine
    ) -> None:
        """Test session serialization includes temporal fields."""
        session = await engine.start_inquiry("Test topic", QUICK_INQUIRY_WORKFLOW)
        session.parent_session_id = "parent-123"
        session.chain_id = "chain-abc"
        session.version = 2

        d = session.to_dict()

        assert d["parent_session_id"] == "parent-123"
        assert d["chain_id"] == "chain-abc"
        assert d["version"] == 2
