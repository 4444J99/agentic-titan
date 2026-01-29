"""Tests for temporal inquiry tracking module."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from titan.workflows.inquiry_temporal import (
    DriftType,
    StageDiff,
    InquiryDiff,
    TemporalChain,
    TemporalTracker,
    get_temporal_tracker,
)
from titan.workflows.inquiry_engine import InquirySession, StageResult, InquiryStatus
from titan.workflows.inquiry_config import QUICK_INQUIRY_WORKFLOW


class TestDriftType:
    """Tests for DriftType enum."""

    def test_drift_type_values(self) -> None:
        """Test all drift types have correct values."""
        assert DriftType.EXPANSION.value == "expansion"
        assert DriftType.CONTRACTION.value == "contraction"
        assert DriftType.REFINEMENT.value == "refinement"
        assert DriftType.PIVOT.value == "pivot"
        assert DriftType.STABLE.value == "stable"


class TestStageDiff:
    """Tests for StageDiff dataclass."""

    def test_create_stage_diff(self) -> None:
        """Test creating a stage diff."""
        diff = StageDiff(
            stage_name="Logical Analysis",
            stage_index=1,
            base_summary="Original analysis focused on X",
            comparison_summary="New analysis explores Y",
            drift_type=DriftType.EXPANSION,
            drift_score=0.6,
            added_themes=["theme1", "theme2"],
            removed_themes=["theme3"],
            consistent_themes=["theme4"],
            content_similarity=0.4,
        )

        assert diff.stage_name == "Logical Analysis"
        assert diff.drift_type == DriftType.EXPANSION
        assert diff.drift_score == 0.6
        assert len(diff.added_themes) == 2

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        diff = StageDiff(
            stage_name="Test",
            stage_index=0,
            base_summary="Base",
            comparison_summary="Comparison",
            drift_type=DriftType.STABLE,
            drift_score=0.1,
        )

        d = diff.to_dict()

        assert d["stage_name"] == "Test"
        assert d["drift_type"] == "stable"
        assert d["drift_score"] == 0.1


class TestInquiryDiff:
    """Tests for InquiryDiff dataclass."""

    def test_create_inquiry_diff(self) -> None:
        """Test creating an inquiry diff."""
        now = datetime.now()
        diff = InquiryDiff(
            base_session_id="inq-base-123",
            comparison_session_id="inq-comp-456",
            topic="AI Safety",
            stage_diffs=[],
            overall_drift_score=0.3,
            key_changes=["Change 1", "Change 2"],
            drift_summary=DriftType.REFINEMENT,
            base_timestamp=now - timedelta(days=7),
            comparison_timestamp=now,
            time_elapsed=7 * 24 * 3600,
        )

        assert diff.base_session_id == "inq-base-123"
        assert diff.overall_drift_score == 0.3
        assert len(diff.key_changes) == 2

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        now = datetime.now()
        diff = InquiryDiff(
            base_session_id="base",
            comparison_session_id="comp",
            topic="Topic",
            stage_diffs=[],
            overall_drift_score=0.5,
            key_changes=[],
            drift_summary=DriftType.PIVOT,
            base_timestamp=now,
            comparison_timestamp=now,
            time_elapsed=0,
        )

        d = diff.to_dict()

        assert d["base_session_id"] == "base"
        assert d["drift_summary"] == "pivot"
        assert "base_timestamp" in d


class TestTemporalChain:
    """Tests for TemporalChain dataclass."""

    def test_create_chain(self) -> None:
        """Test creating a temporal chain."""
        chain = TemporalChain(
            chain_id="chain-123",
            topic="Machine Learning Advances",
            sessions=["inq-1", "inq-2", "inq-3"],
            total_inquiries=3,
        )

        assert chain.chain_id == "chain-123"
        assert len(chain.sessions) == 3
        assert chain.total_inquiries == 3

    def test_to_dict(self) -> None:
        """Test converting chain to dict."""
        chain = TemporalChain(
            chain_id="chain-test",
            topic="Test Topic",
            sessions=["s1", "s2"],
        )

        d = chain.to_dict()

        assert d["chain_id"] == "chain-test"
        assert d["sessions"] == ["s1", "s2"]
        assert "created_at" in d

    def test_from_dict(self) -> None:
        """Test creating chain from dict."""
        data = {
            "chain_id": "chain-restored",
            "topic": "Restored Topic",
            "sessions": ["s1", "s2", "s3"],
            "created_at": datetime.now().isoformat(),
            "total_inquiries": 3,
        }

        chain = TemporalChain.from_dict(data)

        assert chain.chain_id == "chain-restored"
        assert chain.total_inquiries == 3


class TestTemporalTracker:
    """Tests for TemporalTracker class."""

    @pytest.fixture
    def tracker(self) -> TemporalTracker:
        """Create a tracker for testing."""
        return TemporalTracker()

    def test_create_chain(self, tracker: TemporalTracker) -> None:
        """Test creating a new chain."""
        chain = tracker.create_chain("AI Ethics", "inq-first-123")

        assert chain is not None
        assert chain.topic == "AI Ethics"
        assert "inq-first-123" in chain.sessions
        assert chain.total_inquiries == 1

    def test_add_to_chain(self, tracker: TemporalTracker) -> None:
        """Test adding sessions to a chain."""
        chain = tracker.create_chain("Topic", "inq-1")

        result = tracker.add_to_chain(chain.chain_id, "inq-2")

        assert result is True
        assert len(chain.sessions) == 2
        assert chain.total_inquiries == 2

    def test_add_to_nonexistent_chain(self, tracker: TemporalTracker) -> None:
        """Test adding to nonexistent chain returns False."""
        result = tracker.add_to_chain("nonexistent-chain", "inq-1")

        assert result is False

    def test_get_chain(self, tracker: TemporalTracker) -> None:
        """Test getting a chain by ID."""
        chain = tracker.create_chain("Topic", "inq-1")

        retrieved = tracker.get_chain(chain.chain_id)

        assert retrieved is chain

    def test_get_chain_for_topic(self, tracker: TemporalTracker) -> None:
        """Test getting chain by topic."""
        tracker.create_chain("Machine Learning", "inq-1")

        chain = tracker.get_chain_for_topic("Machine Learning")
        assert chain is not None

        # Topic matching should be case-insensitive
        chain2 = tracker.get_chain_for_topic("machine learning")
        assert chain2 is chain

    def test_list_chains(self, tracker: TemporalTracker) -> None:
        """Test listing all chains."""
        tracker.create_chain("Topic 1", "inq-1")
        tracker.create_chain("Topic 2", "inq-2")

        chains = tracker.list_chains()

        assert len(chains) == 2

    @pytest.mark.asyncio
    async def test_compute_diff_basic(self, tracker: TemporalTracker) -> None:
        """Test computing diff between sessions."""
        base_session = InquirySession(
            id="inq-base",
            topic="AI Safety",
            workflow=QUICK_INQUIRY_WORKFLOW,
            created_at=datetime.now() - timedelta(days=7),
        )
        base_session.results = [
            StageResult(
                stage_name="Scope Clarification",
                role="Scope AI",
                content="AI safety involves alignment problems and robustness issues.",
                model_used="claude-3",
                timestamp=datetime.now(),
                stage_index=0,
            ),
        ]

        comp_session = InquirySession(
            id="inq-comp",
            topic="AI Safety",
            workflow=QUICK_INQUIRY_WORKFLOW,
            created_at=datetime.now(),
        )
        comp_session.results = [
            StageResult(
                stage_name="Scope Clarification",
                role="Scope AI",
                content="AI safety encompasses alignment, robustness, and interpretability.",
                model_used="claude-3",
                timestamp=datetime.now(),
                stage_index=0,
            ),
        ]

        diff = await tracker.compute_diff(base_session, comp_session)

        assert diff.base_session_id == "inq-base"
        assert diff.comparison_session_id == "inq-comp"
        assert len(diff.stage_diffs) >= 1
        assert diff.time_elapsed > 0

    @pytest.mark.asyncio
    async def test_compute_diff_detects_changes(self, tracker: TemporalTracker) -> None:
        """Test diff detects meaningful changes."""
        base_session = InquirySession(
            id="base",
            topic="Quantum Computing",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        base_session.results = [
            StageResult(
                stage_name="Analysis",
                role="AI",
                content="Quantum computing uses qubits and superposition for computation.",
                model_used="model",
                timestamp=datetime.now(),
                stage_index=0,
            ),
        ]

        comp_session = InquirySession(
            id="comp",
            topic="Quantum Computing",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        comp_session.results = [
            StageResult(
                stage_name="Analysis",
                role="AI",
                # Significantly different content
                content="Machine learning neural networks deep learning transformers GPT.",
                model_used="model",
                timestamp=datetime.now(),
                stage_index=0,
            ),
        ]

        diff = await tracker.compute_diff(base_session, comp_session)

        # Should detect high drift
        assert diff.overall_drift_score > 0.3

    def test_extract_themes(self, tracker: TemporalTracker) -> None:
        """Test theme extraction from content."""
        content = """
        Artificial intelligence and machine learning are transforming technology.
        Deep learning neural networks enable powerful pattern recognition.
        Machine learning algorithms learn from data.
        """

        themes = tracker._extract_themes(content)

        assert "artificial" in themes or "intelligence" in themes
        assert "machine" in themes or "learning" in themes
        assert len(themes) > 0

    def test_compute_similarity(self, tracker: TemporalTracker) -> None:
        """Test text similarity computation."""
        text1 = "AI safety alignment robustness"
        text2 = "AI safety alignment interpretability"

        similarity = tracker._compute_similarity(text1, text2)

        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should have high overlap

    def test_compute_similarity_different(self, tracker: TemporalTracker) -> None:
        """Test similarity for different texts."""
        text1 = "quantum computing qubits superposition"
        text2 = "machine learning neural networks deep"

        similarity = tracker._compute_similarity(text1, text2)

        assert similarity < 0.3  # Low overlap

    def test_classify_stage_drift_stable(self, tracker: TemporalTracker) -> None:
        """Test drift classification for stable content."""
        drift_type, score = tracker._classify_stage_drift(
            similarity=0.8,
            added_count=1,
            removed_count=1,
            consistent_count=10,
        )

        assert drift_type == DriftType.STABLE
        assert score < 0.2

    def test_classify_stage_drift_expansion(self, tracker: TemporalTracker) -> None:
        """Test drift classification for expanded content."""
        drift_type, score = tracker._classify_stage_drift(
            similarity=0.4,
            added_count=8,
            removed_count=2,
            consistent_count=5,
        )

        assert drift_type == DriftType.EXPANSION

    def test_classify_stage_drift_pivot(self, tracker: TemporalTracker) -> None:
        """Test drift classification for pivoted content."""
        drift_type, score = tracker._classify_stage_drift(
            similarity=0.1,
            added_count=10,
            removed_count=10,
            consistent_count=2,
        )

        assert drift_type == DriftType.PIVOT
        assert score > 0.7


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_temporal_tracker_singleton(self) -> None:
        """Test singleton behavior."""
        tracker1 = get_temporal_tracker()
        tracker2 = get_temporal_tracker()

        assert tracker1 is tracker2
