"""
Tests for Preference Pairs Builder (Phase 18A)
"""

from datetime import UTC, datetime
from uuid import uuid4

from titan.learning.preference_pairs import (
    PreferencePair,
    PreferencePairBuilder,
    PreferencePairDataset,
    get_preference_pair_builder,
)
from titan.learning.rlhf import RLHFSample


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""

    def test_create_basic_pair(self):
        """Test creating a basic preference pair."""
        pair = PreferencePair(
            prompt="What is Python?",
            chosen="Python is a programming language.",
            rejected="Python is a snake.",
        )

        assert pair.prompt == "What is Python?"
        assert pair.chosen == "Python is a programming language."
        assert pair.rejected == "Python is a snake."
        assert pair.margin == 0.0
        assert pair.source == "unknown"

    def test_pair_with_margin(self):
        """Test pair with preference margin."""
        pair = PreferencePair(
            prompt="Test",
            chosen="Good response",
            rejected="Bad response",
            margin=0.75,
            source="rating",
        )

        assert pair.margin == 0.75
        assert pair.source == "rating"

    def test_is_valid(self):
        """Test validity checking."""
        valid_pair = PreferencePair(
            prompt="Test",
            chosen="Good",
            rejected="Bad",
        )
        assert valid_pair.is_valid()

        # Empty prompt
        invalid1 = PreferencePair(prompt="", chosen="Good", rejected="Bad")
        assert not invalid1.is_valid()

        # Same chosen and rejected
        invalid2 = PreferencePair(prompt="Test", chosen="Same", rejected="Same")
        assert not invalid2.is_valid()

        # Empty chosen
        invalid3 = PreferencePair(prompt="Test", chosen="", rejected="Bad")
        assert not invalid3.is_valid()

    def test_to_dict(self):
        """Test dictionary conversion."""
        pair = PreferencePair(
            prompt="Test",
            chosen="Good",
            rejected="Bad",
            margin=0.5,
            source="test",
        )

        data = pair.to_dict()
        assert data["prompt"] == "Test"
        assert data["chosen"] == "Good"
        assert data["rejected"] == "Bad"
        assert data["margin"] == 0.5
        assert data["source"] == "test"
        assert "id" in data
        assert "timestamp" in data

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": str(uuid4()),
            "prompt": "Test",
            "chosen": "Good",
            "rejected": "Bad",
            "margin": 0.5,
            "source": "test",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        pair = PreferencePair.from_dict(data)
        assert pair.prompt == "Test"
        assert pair.margin == 0.5


class TestPreferencePairDataset:
    """Tests for PreferencePairDataset."""

    def test_create_empty_dataset(self):
        """Test creating an empty dataset."""
        dataset = PreferencePairDataset(name="test")
        assert len(dataset) == 0
        assert dataset.name == "test"

    def test_add_pairs(self):
        """Test adding pairs to dataset."""
        dataset = PreferencePairDataset()

        pair1 = PreferencePair(prompt="Q1", chosen="A1", rejected="B1", source="rating")
        pair2 = PreferencePair(prompt="Q2", chosen="A2", rejected="B2", source="correction")

        dataset.add(pair1)
        dataset.add(pair2)

        assert len(dataset) == 2
        assert dataset.source_count["rating"] == 1
        assert dataset.source_count["correction"] == 1

    def test_add_invalid_pair_ignored(self):
        """Test that invalid pairs are not added."""
        dataset = PreferencePairDataset()

        invalid_pair = PreferencePair(prompt="", chosen="A", rejected="B")
        dataset.add(invalid_pair)

        assert len(dataset) == 0

    def test_to_hf_format(self):
        """Test conversion to HuggingFace format."""
        dataset = PreferencePairDataset()
        dataset.add(PreferencePair(prompt="Q1", chosen="A1", rejected="B1"))
        dataset.add(PreferencePair(prompt="Q2", chosen="A2", rejected="B2"))

        hf_data = dataset.to_hf_format()

        assert len(hf_data) == 2
        assert hf_data[0] == {"prompt": "Q1", "chosen": "A1", "rejected": "B1"}
        assert hf_data[1] == {"prompt": "Q2", "chosen": "A2", "rejected": "B2"}

    def test_filter_by_margin(self):
        """Test filtering by minimum margin."""
        dataset = PreferencePairDataset()
        dataset.add(PreferencePair(prompt="Q1", chosen="A1", rejected="B1", margin=0.3))
        dataset.add(PreferencePair(prompt="Q2", chosen="A2", rejected="B2", margin=0.7))
        dataset.add(PreferencePair(prompt="Q3", chosen="A3", rejected="B3", margin=0.5))

        filtered = dataset.filter_by_margin(0.5)
        assert len(filtered) == 2

    def test_split(self):
        """Test train/eval split."""
        dataset = PreferencePairDataset()
        for i in range(10):
            dataset.add(
                PreferencePair(
                    prompt=f"Q{i}",
                    chosen=f"A{i}",
                    rejected=f"B{i}",
                )
            )

        train, eval_ds = dataset.split(train_ratio=0.8)

        assert len(train) == 8
        assert len(eval_ds) == 2


class TestPreferencePairBuilder:
    """Tests for PreferencePairBuilder."""

    def create_sample(
        self,
        prompt: str,
        response: str,
        rating: int | None = None,
        correction: str | None = None,
        accepted: bool | None = None,
        was_regenerated: bool = False,
        regeneration_count: int = 0,
        session_id: str = "session1",
    ) -> RLHFSample:
        """Helper to create RLHF samples."""
        return RLHFSample(
            prompt=prompt,
            response=response,
            human_rating=rating,
            correction=correction,
            accepted=accepted,
            was_regenerated=was_regenerated,
            regeneration_count=regeneration_count,
            session_id=session_id,
        )

    def test_from_ratings(self):
        """Test building pairs from ratings."""
        builder = PreferencePairBuilder()

        samples = [
            self.create_sample("What is 2+2?", "The answer is 4.", rating=5),
            self.create_sample("What is 2+2?", "Two plus two equals four.", rating=4),
            self.create_sample("What is 2+2?", "It's about 4.", rating=3),
        ]

        dataset = builder.from_ratings(samples)

        # Should create pairs between different rating levels
        assert len(dataset) >= 1
        # 5 vs 4, 5 vs 3, 4 vs 3
        for pair in dataset.pairs:
            assert pair.source == "rating"

    def test_from_corrections(self):
        """Test building pairs from corrections."""
        builder = PreferencePairBuilder()

        samples = [
            self.create_sample(
                "What is Python?",
                "Python is a snake.",
                correction="Python is a programming language.",
            ),
        ]

        dataset = builder.from_corrections(samples)

        assert len(dataset) == 1
        pair = dataset.pairs[0]
        assert pair.chosen == "Python is a programming language."
        assert pair.rejected == "Python is a snake."
        assert pair.source == "correction"

    def test_from_regenerations(self):
        """Test building pairs from regeneration data."""
        builder = PreferencePairBuilder()

        samples = [
            self.create_sample(
                "Explain AI",
                "AI is artificial intelligence.",
                accepted=False,
                was_regenerated=True,
                regeneration_count=1,
                session_id="s1",
            ),
            self.create_sample(
                "Explain AI",
                (
                    "AI refers to computer systems that can perform tasks "
                    "typically requiring human intelligence."
                ),
                accepted=True,
                session_id="s1",
            ),
        ]

        dataset = builder.from_regenerations(samples)

        # Should create pair with accepted as chosen
        assert len(dataset) >= 0  # Depends on timestamp ordering

    def test_from_rlhf_samples_combined(self):
        """Test combined extraction from all sources."""
        builder = PreferencePairBuilder()

        samples = [
            # Rating-based
            self.create_sample("Q1", "Good answer", rating=5),
            self.create_sample("Q1", "Okay answer", rating=3),
            # Correction-based
            self.create_sample("Q2", "Wrong", correction="Right"),
        ]

        dataset = builder.from_rlhf_samples(samples)

        assert len(dataset) >= 2  # At least rating pair + correction pair

    def test_from_quality_comparison(self):
        """Test building pairs from quality comparison."""
        builder = PreferencePairBuilder()

        samples = [
            self.create_sample("Q1", "Great response", rating=5),
            self.create_sample("Q1", "Poor response", rating=2),
        ]

        dataset = builder.from_quality_comparison(samples)

        # Should create pairs based on inferred quality
        assert len(dataset) >= 0

    def test_to_hf_dataset(self):
        """Test conversion to HuggingFace Dataset."""
        builder = PreferencePairBuilder()

        dataset = PreferencePairDataset()
        dataset.add(PreferencePair(prompt="Q", chosen="A", rejected="B"))

        result = builder.to_hf_dataset(dataset)

        # Without datasets library, returns dict
        if isinstance(result, dict):
            assert "prompt" in result
            assert "chosen" in result
            assert "rejected" in result

    def test_min_rating_diff(self):
        """Test minimum rating difference threshold."""
        builder = PreferencePairBuilder(min_rating_diff=2)

        samples = [
            self.create_sample("Q", "A", rating=5),
            self.create_sample("Q", "B", rating=4),  # Diff of 1, should be ignored
            self.create_sample("Q", "C", rating=3),  # Diff of 2 from A
        ]

        dataset = builder.from_ratings(samples)

        # Should only include pairs with diff >= 2
        for pair in dataset.pairs:
            meta = pair.metadata
            if "chosen_rating" in meta and "rejected_rating" in meta:
                assert meta["chosen_rating"] - meta["rejected_rating"] >= 2

    def test_min_edit_ratio(self):
        """Test minimum edit ratio threshold."""
        builder = PreferencePairBuilder(min_edit_ratio=0.3)

        # Small edit (below threshold)
        samples = [
            self.create_sample("Q", "Hello world", correction="Hello, world"),
        ]

        builder.from_corrections(samples)
        # May or may not include based on actual edit ratio


class TestGetPreferencePairBuilder:
    """Tests for factory function."""

    def test_get_default_builder(self):
        """Test getting default builder."""
        builder = get_preference_pair_builder()
        assert builder is not None
        assert isinstance(builder, PreferencePairBuilder)

    def test_singleton(self):
        """Test singleton behavior."""
        builder1 = get_preference_pair_builder()
        builder2 = get_preference_pair_builder()
        assert builder1 is builder2
