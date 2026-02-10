"""Tests for narrative synthesizer module."""

from __future__ import annotations

from datetime import datetime

import pytest

from titan.workflows.inquiry_config import QUICK_INQUIRY_WORKFLOW
from titan.workflows.inquiry_engine import InquirySession, InquiryStatus, StageResult
from titan.workflows.narrative_synthesizer import (
    NarrativeConfig,
    NarrativeSection,
    NarrativeStyle,
    NarrativeSynthesis,
    NarrativeSynthesizer,
    TargetLength,
    generate_narrative,
    get_narrative_synthesizer,
)


class TestNarrativeStyle:
    """Tests for NarrativeStyle enum."""

    def test_narrative_style_values(self) -> None:
        """Test all narrative styles have correct values."""
        assert NarrativeStyle.ACADEMIC.value == "academic"
        assert NarrativeStyle.JOURNALISTIC.value == "journalistic"
        assert NarrativeStyle.CONVERSATIONAL.value == "conversational"
        assert NarrativeStyle.POETIC.value == "poetic"
        assert NarrativeStyle.EXECUTIVE.value == "executive"
        assert NarrativeStyle.TECHNICAL.value == "technical"


class TestTargetLength:
    """Tests for TargetLength enum."""

    def test_target_length_values(self) -> None:
        """Test target length values."""
        assert TargetLength.BRIEF.value == "brief"
        assert TargetLength.MEDIUM.value == "medium"
        assert TargetLength.COMPREHENSIVE.value == "comprehensive"


class TestNarrativeConfig:
    """Tests for NarrativeConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating narrative config."""
        config = NarrativeConfig(
            style=NarrativeStyle.ACADEMIC,
            target_length=TargetLength.COMPREHENSIVE,
            preserve_stage_voices=True,
            highlight_contradictions=False,
        )

        assert config.style == NarrativeStyle.ACADEMIC
        assert config.target_length == TargetLength.COMPREHENSIVE
        assert config.preserve_stage_voices is True

    def test_default_config(self) -> None:
        """Test default config values."""
        config = NarrativeConfig()

        assert config.style == NarrativeStyle.CONVERSATIONAL
        assert config.target_length == TargetLength.MEDIUM
        assert config.preserve_stage_voices is True
        assert config.use_transitions is True

    def test_to_dict(self) -> None:
        """Test converting config to dict."""
        config = NarrativeConfig(
            style=NarrativeStyle.TECHNICAL,
            target_length=TargetLength.BRIEF,
        )

        d = config.to_dict()

        assert d["style"] == "technical"
        assert d["target_length"] == "brief"


class TestNarrativeSection:
    """Tests for NarrativeSection dataclass."""

    def test_create_section(self) -> None:
        """Test creating a section."""
        section = NarrativeSection(
            title="Introduction",
            content="This is the introduction content.",
            source_stage="Scope Clarification",
            voice="Scope AI",
        )

        assert section.title == "Introduction"
        assert section.content == "This is the introduction content."
        assert section.source_stage == "Scope Clarification"
        assert section.voice == "Scope AI"


class TestNarrativeSynthesis:
    """Tests for NarrativeSynthesis dataclass."""

    def test_create_synthesis(self) -> None:
        """Test creating a synthesis result."""
        config = NarrativeConfig()
        synthesis = NarrativeSynthesis(
            title="Test Title",
            abstract="Test abstract",
            sections=[NarrativeSection(title="S1", content="Content")],
            full_text="Full text here",
            config=config,
            session_id="inq-123",
            topic="Test Topic",
            word_count=100,
            stage_count=3,
        )

        assert synthesis.title == "Test Title"
        assert len(synthesis.sections) == 1
        assert synthesis.word_count == 100

    def test_to_dict(self) -> None:
        """Test converting synthesis to dict."""
        config = NarrativeConfig()
        synthesis = NarrativeSynthesis(
            title="Title",
            abstract="Abstract",
            sections=[],
            full_text="Text",
            config=config,
            session_id="inq-123",
            topic="Topic",
        )

        d = synthesis.to_dict()

        assert d["title"] == "Title"
        assert d["session_id"] == "inq-123"
        assert "config" in d
        assert "created_at" in d

    def test_to_markdown(self) -> None:
        """Test converting to markdown format."""
        config = NarrativeConfig()
        synthesis = NarrativeSynthesis(
            title="Markdown Test",
            abstract="Test abstract.",
            sections=[
                NarrativeSection(title="Section 1", content="Content 1"),
                NarrativeSection(title="Section 2", content="Content 2", voice="Test Voice"),
            ],
            full_text="",
            config=config,
            session_id="inq-123",
            topic="Test Topic",
        )

        md = synthesis.to_markdown()

        assert "# Markdown Test" in md
        assert "*Test abstract.*" in md
        assert "## Section 1" in md
        assert "## Section 2" in md
        assert "*Voice: Test Voice*" in md
        assert "Session: inq-123" in md


class TestNarrativeSynthesizer:
    """Tests for NarrativeSynthesizer class."""

    @pytest.fixture
    def synthesizer(self) -> NarrativeSynthesizer:
        """Create a synthesizer for testing."""
        return NarrativeSynthesizer()

    @pytest.fixture
    def mock_session(self) -> InquirySession:
        """Create a mock inquiry session."""
        session = InquirySession(
            id="inq-test-123",
            topic="Artificial Intelligence and Society",
            workflow=QUICK_INQUIRY_WORKFLOW,
            status=InquiryStatus.COMPLETED,
        )

        # Add mock results
        session.results = [
            StageResult(
                stage_name="Scope Clarification",
                role="Scope AI",
                content="The topic of AI and society encompasses multiple dimensions...",
                model_used="claude-3",
                timestamp=datetime.now(),
                stage_index=0,
            ),
            StageResult(
                stage_name="Logical Analysis",
                role="Logic AI",
                content="From a logical standpoint, the impact of AI can be categorized...",
                model_used="gpt-4",
                timestamp=datetime.now(),
                stage_index=1,
            ),
            StageResult(
                stage_name="Pattern Recognition",
                role="Pattern AI",
                content="Patterns emerge across the analysis showing key themes...",
                model_used="claude-3",
                timestamp=datetime.now(),
                stage_index=2,
            ),
        ]

        return session

    @pytest.mark.asyncio
    async def test_synthesize_basic(
        self, synthesizer: NarrativeSynthesizer, mock_session: InquirySession
    ) -> None:
        """Test basic synthesis functionality."""
        result = await synthesizer.synthesize(mock_session)

        assert result.session_id == mock_session.id
        assert result.topic == mock_session.topic
        assert result.stage_count == 3
        assert result.word_count > 0
        assert len(result.sections) > 0

    @pytest.mark.asyncio
    async def test_synthesize_with_custom_config(
        self, synthesizer: NarrativeSynthesizer, mock_session: InquirySession
    ) -> None:
        """Test synthesis with custom config."""
        config = NarrativeConfig(
            style=NarrativeStyle.ACADEMIC,
            include_methodology=True,
        )

        result = await synthesizer.synthesize(mock_session, config)

        assert result.config.style == NarrativeStyle.ACADEMIC
        # Should have methodology section
        section_titles = [s.title for s in result.sections]
        assert "Methodology" in section_titles

    @pytest.mark.asyncio
    async def test_synthesize_preserves_voices(
        self, synthesizer: NarrativeSynthesizer, mock_session: InquirySession
    ) -> None:
        """Test that voice preservation works."""
        config = NarrativeConfig(preserve_stage_voices=True)

        result = await synthesizer.synthesize(mock_session, config)

        # Check at least one section has voice set
        voices = [s.voice for s in result.sections if s.voice]
        assert len(voices) > 0

    @pytest.mark.asyncio
    async def test_synthesize_without_voices(
        self, synthesizer: NarrativeSynthesizer, mock_session: InquirySession
    ) -> None:
        """Test synthesis without voice preservation."""
        config = NarrativeConfig(preserve_stage_voices=False)

        result = await synthesizer.synthesize(mock_session, config)

        # No sections should have voice
        voices = [s.voice for s in result.sections if s.voice]
        assert len(voices) == 0

    def test_generate_title_academic(self, synthesizer: NarrativeSynthesizer) -> None:
        """Test academic title generation."""
        title = synthesizer._generate_title("quantum computing", NarrativeStyle.ACADEMIC)

        assert "Analysis" in title
        assert "quantum computing" in title

    def test_generate_title_executive(self, synthesizer: NarrativeSynthesizer) -> None:
        """Test executive title generation."""
        title = synthesizer._generate_title("market trends", NarrativeStyle.EXECUTIVE)

        assert "Brief" in title
        assert "market trends" in title

    @pytest.mark.asyncio
    async def test_generate_sections(
        self, synthesizer: NarrativeSynthesizer, mock_session: InquirySession
    ) -> None:
        """Test section generation."""
        config = NarrativeConfig()
        sections = await synthesizer._generate_sections(mock_session, config)

        assert len(sections) >= len(mock_session.results)
        # Should have a synthesis section
        titles = [s.title for s in sections]
        assert "Synthesis" in titles

    def test_get_transition_academic(self, synthesizer: NarrativeSynthesizer) -> None:
        """Test transition phrases for academic style."""
        transition = synthesizer._get_transition(1, NarrativeStyle.ACADEMIC)

        assert transition in synthesizer.STYLE_TRANSITIONS[NarrativeStyle.ACADEMIC]

    def test_get_transition_conversational(self, synthesizer: NarrativeSynthesizer) -> None:
        """Test transition phrases for conversational style."""
        transition = synthesizer._get_transition(0, NarrativeStyle.CONVERSATIONAL)

        assert transition in synthesizer.STYLE_TRANSITIONS[NarrativeStyle.CONVERSATIONAL]


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_narrative_synthesizer_singleton(self) -> None:
        """Test singleton behavior."""
        synth1 = get_narrative_synthesizer()
        synth2 = get_narrative_synthesizer()

        assert synth1 is synth2

    @pytest.mark.asyncio
    async def test_generate_narrative_function(self) -> None:
        """Test the generate_narrative convenience function."""
        session = InquirySession(
            id="inq-test",
            topic="Test Topic",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        session.results = [
            StageResult(
                stage_name="Test Stage",
                role="Test AI",
                content="Test content",
                model_used="test-model",
                timestamp=datetime.now(),
                stage_index=0,
            )
        ]

        result = await generate_narrative(session, style="academic")

        assert result.config.style == NarrativeStyle.ACADEMIC
        assert result.session_id == session.id

    @pytest.mark.asyncio
    async def test_generate_narrative_invalid_style(self) -> None:
        """Test generate_narrative with invalid style falls back to conversational."""
        session = InquirySession(
            id="inq-test",
            topic="Test",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        session.results = [
            StageResult(
                stage_name="Stage",
                role="AI",
                content="Content",
                model_used="model",
                timestamp=datetime.now(),
                stage_index=0,
            )
        ]

        result = await generate_narrative(session, style="invalid_style")

        assert result.config.style == NarrativeStyle.CONVERSATIONAL
