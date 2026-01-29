"""Tests for personality modulator module."""

from __future__ import annotations

import pytest

from titan.workflows.personality_modulator import (
    PersonalityVector,
    PersonalityModulator,
    PRESET_PERSONALITIES,
    get_preset_personality,
    list_preset_personalities,
    get_personality_modulator,
    modulate_prompt,
)


class TestPersonalityVector:
    """Tests for PersonalityVector dataclass."""

    def test_create_vector(self) -> None:
        """Test creating a personality vector."""
        vector = PersonalityVector(
            tone=0.5,
            abstraction=-0.3,
            verbosity=0.7,
            creativity=0.2,
            technicality=-0.1,
        )

        assert vector.tone == 0.5
        assert vector.abstraction == -0.3
        assert vector.verbosity == 0.7

    def test_default_values(self) -> None:
        """Test default neutral values."""
        vector = PersonalityVector()

        assert vector.tone == 0.0
        assert vector.abstraction == 0.0
        assert vector.verbosity == 0.0
        assert vector.creativity == 0.0
        assert vector.technicality == 0.0

    def test_value_clamping(self) -> None:
        """Test values are clamped to -1 to 1 range."""
        vector = PersonalityVector(
            tone=2.0,
            abstraction=-2.0,
        )

        assert vector.tone == 1.0
        assert vector.abstraction == -1.0

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        vector = PersonalityVector(tone=0.5, creativity=0.8)
        d = vector.to_dict()

        assert d["tone"] == 0.5
        assert d["creativity"] == 0.8
        assert len(d) == 5

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {"tone": 0.3, "verbosity": -0.5}
        vector = PersonalityVector.from_dict(data)

        assert vector.tone == 0.3
        assert vector.verbosity == -0.5
        assert vector.creativity == 0.0

    def test_blend(self) -> None:
        """Test blending two vectors."""
        v1 = PersonalityVector(tone=0.0, creativity=1.0)
        v2 = PersonalityVector(tone=1.0, creativity=0.0)

        blended = v1.blend(v2, weight=0.5)

        assert blended.tone == 0.5
        assert blended.creativity == 0.5

    def test_blend_weights(self) -> None:
        """Test blend respects weight parameter."""
        v1 = PersonalityVector(tone=0.0)
        v2 = PersonalityVector(tone=1.0)

        # 0.25 weight = 75% v1, 25% v2
        blended = v1.blend(v2, weight=0.25)
        assert blended.tone == 0.25

    def test_intensity(self) -> None:
        """Test intensity calculation."""
        neutral = PersonalityVector()
        assert neutral.intensity() == 0.0

        extreme = PersonalityVector(
            tone=1.0, abstraction=1.0, verbosity=1.0,
            creativity=1.0, technicality=1.0
        )
        assert extreme.intensity() == 1.0

    def test_dominant_trait(self) -> None:
        """Test finding dominant trait."""
        vector = PersonalityVector(
            tone=0.2,
            creativity=0.8,
            verbosity=-0.3,
        )

        trait, value = vector.dominant_trait()
        assert trait == "creativity"
        assert value == 0.8

    def test_dominant_trait_negative(self) -> None:
        """Test dominant trait can be negative."""
        vector = PersonalityVector(
            tone=0.2,
            verbosity=-0.9,
        )

        trait, value = vector.dominant_trait()
        assert trait == "verbosity"
        assert value == -0.9


class TestPresetPersonalities:
    """Tests for preset personalities."""

    def test_academic_preset(self) -> None:
        """Test academic preset exists and has expected traits."""
        academic = PRESET_PERSONALITIES.get("academic")

        assert academic is not None
        assert academic.tone < 0  # Formal
        assert academic.technicality > 0  # Technical

    def test_conversational_preset(self) -> None:
        """Test conversational preset."""
        conv = PRESET_PERSONALITIES.get("conversational")

        assert conv is not None
        assert conv.tone > 0  # Casual
        assert conv.technicality < 0  # Accessible

    def test_creative_preset(self) -> None:
        """Test creative preset."""
        creative = PRESET_PERSONALITIES.get("creative")

        assert creative is not None
        assert creative.creativity > 0.5

    def test_get_preset_personality(self) -> None:
        """Test getting preset by name."""
        academic = get_preset_personality("academic")
        assert academic is not None

        unknown = get_preset_personality("nonexistent")
        assert unknown is None

    def test_list_preset_personalities(self) -> None:
        """Test listing preset names."""
        names = list_preset_personalities()

        assert "academic" in names
        assert "conversational" in names
        assert "creative" in names
        assert "technical" in names
        assert len(names) >= 7


class TestPersonalityModulator:
    """Tests for PersonalityModulator class."""

    def test_modulator_initialization(self) -> None:
        """Test modulator initializes correctly."""
        modulator = PersonalityModulator()

        assert modulator._dimension_instructions is not None

    def test_modulate_neutral_vector(self) -> None:
        """Test neutral vector doesn't modify prompt."""
        modulator = PersonalityModulator()
        original = "Write about AI safety."
        neutral = PersonalityVector()

        result = modulator.modulate_prompt(original, neutral)

        assert result == original

    def test_modulate_with_tone(self) -> None:
        """Test modulation adds tone instructions."""
        modulator = PersonalityModulator()
        original = "Explain quantum computing."
        formal = PersonalityVector(tone=-0.8)

        result = modulator.modulate_prompt(original, formal)

        assert "formal" in result.lower() or "professional" in result.lower()
        assert original in result

    def test_modulate_with_verbosity(self) -> None:
        """Test modulation adds verbosity instructions."""
        modulator = PersonalityModulator()
        original = "Describe machine learning."
        verbose = PersonalityVector(verbosity=0.8)

        result = modulator.modulate_prompt(original, verbose)

        assert "comprehensive" in result.lower() or "thorough" in result.lower() or "detail" in result.lower()

    def test_modulate_combined_traits(self) -> None:
        """Test modulation with multiple traits."""
        modulator = PersonalityModulator()
        original = "Explain the concept."
        vector = PersonalityVector(tone=-0.5, technicality=0.8, verbosity=-0.5)

        result = modulator.modulate_prompt(original, vector)

        # Should have multiple instructions
        assert "Style instructions:" in result
        assert original in result

    def test_modulate_without_prefix(self) -> None:
        """Test modulation appends instead of prefixes."""
        modulator = PersonalityModulator()
        original = "Write about AI."
        vector = PersonalityVector(creativity=0.8)

        result = modulator.modulate_prompt(original, vector, include_prefix=False)

        assert result.startswith(original)
        assert "Note:" in result

    def test_get_style_description(self) -> None:
        """Test generating style description."""
        modulator = PersonalityModulator()

        academic = PersonalityVector(tone=-0.5, technicality=0.7)
        desc = modulator.get_style_description(academic)
        assert "formal" in desc
        assert "technical" in desc

        neutral = PersonalityVector()
        desc = modulator.get_style_description(neutral)
        assert desc == "balanced"

    def test_suggest_for_audience(self) -> None:
        """Test audience-based suggestions."""
        modulator = PersonalityModulator()

        exec_vector = modulator.suggest_for_audience("executives")
        assert exec_vector.verbosity < 0  # Concise

        student_vector = modulator.suggest_for_audience("students")
        assert student_vector.technicality < 0  # Accessible

    def test_suggest_for_task(self) -> None:
        """Test task-based suggestions."""
        modulator = PersonalityModulator()

        analysis_vector = modulator.suggest_for_task("analysis")
        assert analysis_vector.verbosity > 0  # Detailed

        brainstorm_vector = modulator.suggest_for_task("brainstorm")
        assert brainstorm_vector.creativity > 0


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_personality_modulator_singleton(self) -> None:
        """Test singleton behavior."""
        mod1 = get_personality_modulator()
        mod2 = get_personality_modulator()

        assert mod1 is mod2

    def test_modulate_prompt_with_vector(self) -> None:
        """Test modulate_prompt with vector."""
        vector = PersonalityVector(creativity=0.8)
        result = modulate_prompt("Write a story.", vector)

        assert "Write a story." in result

    def test_modulate_prompt_with_preset_name(self) -> None:
        """Test modulate_prompt with preset name."""
        result = modulate_prompt("Explain the algorithm.", "technical")

        assert "Explain the algorithm." in result

    def test_modulate_prompt_unknown_preset(self) -> None:
        """Test modulate_prompt with unknown preset returns original."""
        original = "Test prompt."
        result = modulate_prompt(original, "nonexistent_preset")

        assert result == original
