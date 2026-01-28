"""
Titan Learning - Reward Signal Extraction

Extracts reward signals from user interactions for RLHF training.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("titan.learning.reward_signals")


class SignalType(str, Enum):
    """Types of reward signals."""

    # Explicit signals
    EXPLICIT_RATING = "explicit_rating"
    THUMBS_UP_DOWN = "thumbs_up_down"
    TEXT_FEEDBACK = "text_feedback"

    # Implicit signals - user behavior
    TIME_TO_ACCEPT = "time_to_accept"
    EDIT_DISTANCE = "edit_distance"
    REGENERATION_COUNT = "regeneration_count"
    FOLLOW_UP_QUESTIONS = "follow_up_questions"
    TASK_COMPLETION = "task_completion"

    # Implicit signals - response characteristics
    RESPONSE_LENGTH = "response_length"
    CODE_EXECUTION_SUCCESS = "code_execution_success"
    ERROR_MENTIONS = "error_mentions"

    # Derived signals
    COMPOSITE = "composite"


@dataclass
class RewardSignal:
    """A single reward signal."""

    signal_type: SignalType
    value: float  # Normalized to [-1, 1]
    confidence: float  # 0-1 confidence in the signal
    raw_value: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "raw_value": self.raw_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RewardEstimate:
    """Aggregated reward estimate for a response."""

    reward: float  # Weighted average of signals, -1 to 1
    confidence: float  # Overall confidence
    signals: list[RewardSignal] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reward": self.reward,
            "confidence": self.confidence,
            "signals": [s.to_dict() for s in self.signals],
            "explanation": self.explanation,
        }


class RewardSignalExtractor:
    """
    Extracts and normalizes reward signals from interactions.

    Converts various feedback types into normalized reward signals
    that can be used for training reward models.
    """

    def __init__(
        self,
        signal_weights: dict[SignalType, float] | None = None,
    ) -> None:
        # Default signal weights
        self._weights = signal_weights or {
            SignalType.EXPLICIT_RATING: 1.0,
            SignalType.THUMBS_UP_DOWN: 0.9,
            SignalType.TEXT_FEEDBACK: 0.7,
            SignalType.TIME_TO_ACCEPT: 0.5,
            SignalType.EDIT_DISTANCE: 0.6,
            SignalType.REGENERATION_COUNT: 0.7,
            SignalType.TASK_COMPLETION: 0.8,
            SignalType.CODE_EXECUTION_SUCCESS: 0.8,
        }

    def extract_from_rating(self, rating: int) -> RewardSignal:
        """
        Extract signal from 1-5 rating.

        Args:
            rating: 1-5 rating

        Returns:
            RewardSignal with normalized value
        """
        # Normalize: 1 -> -1, 3 -> 0, 5 -> 1
        normalized = (rating - 3) / 2
        return RewardSignal(
            signal_type=SignalType.EXPLICIT_RATING,
            value=normalized,
            confidence=0.95,  # High confidence for explicit ratings
            raw_value=rating,
        )

    def extract_from_thumbs(self, thumbs_up: bool) -> RewardSignal:
        """
        Extract signal from thumbs up/down.

        Args:
            thumbs_up: True for thumbs up, False for thumbs down

        Returns:
            RewardSignal
        """
        return RewardSignal(
            signal_type=SignalType.THUMBS_UP_DOWN,
            value=1.0 if thumbs_up else -1.0,
            confidence=0.9,
            raw_value=thumbs_up,
        )

    def extract_from_time_to_accept(
        self,
        time_ms: int,
        response_length: int,
    ) -> RewardSignal:
        """
        Extract signal from time to accept.

        Fast acceptance suggests good response.
        Long time suggests hesitation or issues.

        Args:
            time_ms: Time to accept in milliseconds
            response_length: Response length for normalization

        Returns:
            RewardSignal
        """
        # Expected time based on response length (rough: 200ms per word + 1s base)
        words = response_length / 5  # Rough word count
        expected_ms = 1000 + (words * 200)

        # Ratio: < 1 means faster than expected
        ratio = time_ms / expected_ms

        if ratio < 0.5:
            # Very fast - good signal
            value = 0.5
            confidence = 0.6
        elif ratio < 1.5:
            # Normal range - neutral
            value = 0.0
            confidence = 0.4
        elif ratio < 3.0:
            # Slow - mild negative
            value = -0.3
            confidence = 0.5
        else:
            # Very slow - negative signal
            value = -0.6
            confidence = 0.6

        return RewardSignal(
            signal_type=SignalType.TIME_TO_ACCEPT,
            value=value,
            confidence=confidence,
            raw_value=time_ms,
            metadata={"expected_ms": expected_ms, "ratio": ratio},
        )

    def extract_from_edits(
        self,
        original: str,
        edited: str,
    ) -> RewardSignal:
        """
        Extract signal from edit distance.

        More edits suggest lower quality original response.

        Args:
            original: Original response
            edited: Edited response

        Returns:
            RewardSignal
        """
        if not edited or original == edited:
            return RewardSignal(
                signal_type=SignalType.EDIT_DISTANCE,
                value=0.5,  # No edits is a positive signal
                confidence=0.7,
                raw_value=0,
            )

        # Calculate word-level edit ratio
        original_words = set(original.split())
        edited_words = set(edited.split())
        diff = original_words.symmetric_difference(edited_words)
        total = len(original_words.union(edited_words))

        edit_ratio = len(diff) / total if total > 0 else 0

        # Normalize: 0% edits -> +0.5, 50% edits -> 0, 100% edits -> -0.5
        value = 0.5 - edit_ratio

        return RewardSignal(
            signal_type=SignalType.EDIT_DISTANCE,
            value=value,
            confidence=0.7,
            raw_value=edit_ratio,
            metadata={"words_changed": len(diff), "total_words": total},
        )

    def extract_from_regeneration(
        self,
        regeneration_count: int,
    ) -> RewardSignal:
        """
        Extract signal from regeneration count.

        More regenerations suggest unsatisfactory responses.

        Args:
            regeneration_count: Number of regenerations

        Returns:
            RewardSignal
        """
        if regeneration_count == 0:
            return RewardSignal(
                signal_type=SignalType.REGENERATION_COUNT,
                value=0.3,  # No regeneration is a mild positive
                confidence=0.5,
                raw_value=0,
            )

        # Each regeneration is a negative signal
        value = max(-1.0, -0.3 * regeneration_count)
        confidence = min(0.9, 0.5 + 0.1 * regeneration_count)

        return RewardSignal(
            signal_type=SignalType.REGENERATION_COUNT,
            value=value,
            confidence=confidence,
            raw_value=regeneration_count,
        )

    def extract_from_code_execution(
        self,
        success: bool,
        error_message: str | None = None,
    ) -> RewardSignal:
        """
        Extract signal from code execution result.

        Args:
            success: Whether code executed successfully
            error_message: Error message if failed

        Returns:
            RewardSignal
        """
        if success:
            return RewardSignal(
                signal_type=SignalType.CODE_EXECUTION_SUCCESS,
                value=0.8,
                confidence=0.9,
                raw_value=True,
            )
        else:
            return RewardSignal(
                signal_type=SignalType.CODE_EXECUTION_SUCCESS,
                value=-0.5,
                confidence=0.8,
                raw_value=False,
                metadata={"error": error_message},
            )

    def extract_from_text_feedback(self, feedback: str) -> RewardSignal:
        """
        Extract signal from text feedback using sentiment analysis.

        Args:
            feedback: Text feedback

        Returns:
            RewardSignal
        """
        feedback_lower = feedback.lower()

        # Simple keyword-based sentiment
        positive_keywords = [
            "great", "good", "excellent", "perfect", "helpful",
            "thanks", "awesome", "correct", "right", "useful",
        ]
        negative_keywords = [
            "wrong", "bad", "incorrect", "useless", "broken",
            "error", "mistake", "fail", "terrible", "awful",
        ]

        positive_count = sum(1 for kw in positive_keywords if kw in feedback_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in feedback_lower)

        if positive_count > negative_count:
            value = min(1.0, 0.3 * (positive_count - negative_count))
            confidence = min(0.8, 0.5 + 0.1 * positive_count)
        elif negative_count > positive_count:
            value = max(-1.0, -0.3 * (negative_count - positive_count))
            confidence = min(0.8, 0.5 + 0.1 * negative_count)
        else:
            value = 0.0
            confidence = 0.3

        return RewardSignal(
            signal_type=SignalType.TEXT_FEEDBACK,
            value=value,
            confidence=confidence,
            raw_value=feedback,
            metadata={
                "positive_keywords": positive_count,
                "negative_keywords": negative_count,
            },
        )

    def aggregate_signals(
        self,
        signals: list[RewardSignal],
    ) -> RewardEstimate:
        """
        Aggregate multiple signals into a single reward estimate.

        Uses weighted average based on signal type and confidence.

        Args:
            signals: List of reward signals

        Returns:
            RewardEstimate with aggregated reward
        """
        if not signals:
            return RewardEstimate(
                reward=0.0,
                confidence=0.0,
                signals=[],
                explanation="No signals available",
            )

        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0

        for signal in signals:
            base_weight = self._weights.get(signal.signal_type, 0.5)
            weight = base_weight * signal.confidence
            weighted_sum += signal.value * weight
            total_weight += weight
            confidence_sum += signal.confidence

        reward = weighted_sum / total_weight if total_weight > 0 else 0.0
        avg_confidence = confidence_sum / len(signals)

        # Build explanation
        explanations = []
        for signal in sorted(signals, key=lambda s: abs(s.value), reverse=True):
            direction = "positive" if signal.value > 0 else "negative" if signal.value < 0 else "neutral"
            explanations.append(f"{signal.signal_type.value}: {direction} ({signal.value:.2f})")

        return RewardEstimate(
            reward=reward,
            confidence=avg_confidence,
            signals=signals,
            explanation="; ".join(explanations[:5]),
        )


# Singleton instance
_default_extractor: RewardSignalExtractor | None = None


def get_reward_extractor() -> RewardSignalExtractor:
    """Get the default reward signal extractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = RewardSignalExtractor()
    return _default_extractor
