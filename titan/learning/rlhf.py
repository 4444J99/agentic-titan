"""
Titan Learning - RLHF Data Collection

Captures structured data for reward model training.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from titan.persistence.postgres import PostgresClient

logger = logging.getLogger("titan.learning.rlhf")


class FeedbackType(str, Enum):
    """Types of feedback signals."""

    EXPLICIT_RATING = "explicit_rating"      # User gives 1-5 rating
    THUMBS = "thumbs"                         # Thumbs up/down
    CORRECTION = "correction"                 # User corrects output
    ACCEPTANCE = "acceptance"                 # User accepts/rejects
    IMPLICIT_SIGNAL = "implicit_signal"       # Derived from behavior


class ResponseQuality(str, Enum):
    """Quality levels for responses."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class RLHFSample:
    """
    Sample for RLHF training.

    Captures the full context of an interaction for reward model training.
    """

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Input/Output
    prompt: str = ""
    response: str = ""
    system_prompt: str | None = None

    # Human feedback
    human_rating: int | None = None  # 1-5 scale
    feedback_type: FeedbackType | None = None
    correction: str | None = None
    accepted: bool | None = None

    # Implicit signals
    time_to_accept_ms: int | None = None
    edits_made: int = 0
    response_length: int = 0
    was_regenerated: bool = False
    regeneration_count: int = 0

    # Context
    session_id: str = ""
    agent_id: str = ""
    model: str = ""
    provider: str = ""
    task_type: str = ""

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_explicit_feedback(self) -> bool:
        """Check if sample has explicit human feedback."""
        return self.human_rating is not None or self.accepted is not None

    @property
    def has_implicit_signals(self) -> bool:
        """Check if sample has implicit feedback signals."""
        return self.time_to_accept_ms is not None or self.edits_made > 0

    @property
    def inferred_quality(self) -> ResponseQuality:
        """Infer quality from available signals."""
        # Explicit rating takes precedence
        if self.human_rating is not None:
            if self.human_rating >= 5:
                return ResponseQuality.EXCELLENT
            elif self.human_rating >= 4:
                return ResponseQuality.GOOD
            elif self.human_rating >= 3:
                return ResponseQuality.ACCEPTABLE
            elif self.human_rating >= 2:
                return ResponseQuality.POOR
            else:
                return ResponseQuality.UNACCEPTABLE

        # Infer from other signals
        if self.accepted is False or self.was_regenerated:
            return ResponseQuality.POOR

        if self.accepted is True and self.edits_made == 0:
            return ResponseQuality.GOOD

        if self.edits_made > 5:
            return ResponseQuality.POOR

        return ResponseQuality.ACCEPTABLE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "prompt": self.prompt,
            "response": self.response,
            "system_prompt": self.system_prompt,
            "human_rating": self.human_rating,
            "feedback_type": self.feedback_type.value if self.feedback_type else None,
            "correction": self.correction,
            "accepted": self.accepted,
            "time_to_accept_ms": self.time_to_accept_ms,
            "edits_made": self.edits_made,
            "response_length": self.response_length,
            "was_regenerated": self.was_regenerated,
            "regeneration_count": self.regeneration_count,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "model": self.model,
            "provider": self.provider,
            "task_type": self.task_type,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "inferred_quality": self.inferred_quality.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLHFSample:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data["timestamp"], str)
            else data["timestamp"],
            prompt=data.get("prompt", ""),
            response=data.get("response", ""),
            system_prompt=data.get("system_prompt"),
            human_rating=data.get("human_rating"),
            feedback_type=FeedbackType(data["feedback_type"])
            if data.get("feedback_type")
            else None,
            correction=data.get("correction"),
            accepted=data.get("accepted"),
            time_to_accept_ms=data.get("time_to_accept_ms"),
            edits_made=data.get("edits_made", 0),
            response_length=data.get("response_length", 0),
            was_regenerated=data.get("was_regenerated", False),
            regeneration_count=data.get("regeneration_count", 0),
            session_id=data.get("session_id", ""),
            agent_id=data.get("agent_id", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            task_type=data.get("task_type", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RLHFDatasetStats:
    """Statistics about the RLHF dataset."""

    total_samples: int = 0
    samples_with_explicit_feedback: int = 0
    samples_with_implicit_signals: int = 0
    quality_distribution: dict[str, int] = field(default_factory=dict)
    average_rating: float | None = None
    acceptance_rate: float | None = None
    model_distribution: dict[str, int] = field(default_factory=dict)


class RLHFCollector:
    """
    Collects RLHF training data.

    Features:
    - Capture prompts and responses
    - Record explicit human feedback
    - Track implicit signals
    - Store in PostgreSQL for persistence
    - Export for training
    """

    def __init__(
        self,
        postgres_client: PostgresClient | None = None,
        buffer_size: int = 100,
        auto_flush: bool = True,
    ) -> None:
        self._postgres = postgres_client
        self._buffer_size = buffer_size
        self._auto_flush = auto_flush
        self._buffer: list[RLHFSample] = []
        self._lock = asyncio.Lock()

        # Tracking for implicit signals
        self._pending_responses: dict[str, tuple[RLHFSample, datetime]] = {}

    async def start_response_tracking(
        self,
        prompt: str,
        response: str,
        session_id: str,
        agent_id: str = "",
        model: str = "",
        provider: str = "",
        system_prompt: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> str:
        """
        Start tracking a response for feedback collection.

        Returns:
            Tracking ID for this response
        """
        sample = RLHFSample(
            prompt=prompt,
            response=response,
            system_prompt=system_prompt,
            session_id=session_id,
            agent_id=agent_id,
            model=model,
            provider=provider,
            response_length=len(response),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        tracking_id = str(sample.id)
        self._pending_responses[tracking_id] = (sample, datetime.utcnow())

        logger.debug(f"Started tracking response: {tracking_id}")
        return tracking_id

    async def record_feedback(
        self,
        tracking_id: str,
        rating: int | None = None,
        accepted: bool | None = None,
        correction: str | None = None,
        feedback_type: FeedbackType = FeedbackType.EXPLICIT_RATING,
    ) -> bool:
        """
        Record feedback for a tracked response.

        Args:
            tracking_id: Response tracking ID
            rating: Human rating (1-5)
            accepted: Whether response was accepted
            correction: Corrected response text
            feedback_type: Type of feedback

        Returns:
            True if feedback was recorded
        """
        if tracking_id not in self._pending_responses:
            logger.warning(f"Unknown tracking ID: {tracking_id}")
            return False

        sample, start_time = self._pending_responses.pop(tracking_id)

        # Record timing
        sample.time_to_accept_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Record feedback
        sample.human_rating = rating
        sample.accepted = accepted
        sample.correction = correction
        sample.feedback_type = feedback_type

        if correction:
            sample.edits_made = self._count_edits(sample.response, correction)

        # Add to buffer
        await self._add_to_buffer(sample)

        logger.debug(f"Recorded feedback for {tracking_id}: rating={rating}, accepted={accepted}")
        return True

    async def record_regeneration(self, tracking_id: str) -> bool:
        """Record that a response was regenerated."""
        if tracking_id not in self._pending_responses:
            return False

        sample, start_time = self._pending_responses[tracking_id]
        sample.was_regenerated = True
        sample.regeneration_count += 1

        return True

    async def record_sample(self, sample: RLHFSample) -> None:
        """Directly record a complete RLHF sample."""
        await self._add_to_buffer(sample)

    async def _add_to_buffer(self, sample: RLHFSample) -> None:
        """Add sample to buffer, flushing if needed."""
        async with self._lock:
            self._buffer.append(sample)

            if self._auto_flush and len(self._buffer) >= self._buffer_size:
                await self._flush()

    async def _flush(self) -> int:
        """Flush buffer to storage."""
        if not self._buffer:
            return 0

        samples_to_flush = self._buffer.copy()
        self._buffer.clear()

        # Store in PostgreSQL if available
        if self._postgres and self._postgres.is_connected:
            for sample in samples_to_flush:
                try:
                    await self._postgres.execute(
                        """
                        INSERT INTO rlhf_samples
                        (id, timestamp, prompt, response, human_rating, accepted,
                         session_id, agent_id, model, provider, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        sample.id,
                        sample.timestamp,
                        sample.prompt,
                        sample.response,
                        sample.human_rating,
                        sample.accepted,
                        sample.session_id,
                        sample.agent_id,
                        sample.model,
                        sample.provider,
                        json.dumps(sample.to_dict()),
                    )
                except Exception as e:
                    logger.error(f"Failed to store RLHF sample: {e}")

        logger.info(f"Flushed {len(samples_to_flush)} RLHF samples")
        return len(samples_to_flush)

    async def flush(self) -> int:
        """Manually flush the buffer."""
        async with self._lock:
            return await self._flush()

    def _count_edits(self, original: str, corrected: str) -> int:
        """Count approximate edit distance between strings."""
        # Simple word-level diff count
        original_words = set(original.split())
        corrected_words = set(corrected.split())
        return len(original_words.symmetric_difference(corrected_words))

    async def get_samples(
        self,
        session_id: str | None = None,
        model: str | None = None,
        min_rating: int | None = None,
        limit: int = 100,
    ) -> list[RLHFSample]:
        """
        Get collected samples with optional filtering.

        Args:
            session_id: Filter by session
            model: Filter by model
            min_rating: Minimum rating filter
            limit: Maximum samples to return

        Returns:
            List of RLHF samples
        """
        # First check buffer
        samples = [s for s in self._buffer]

        # Apply filters
        if session_id:
            samples = [s for s in samples if s.session_id == session_id]
        if model:
            samples = [s for s in samples if s.model == model]
        if min_rating:
            samples = [s for s in samples if s.human_rating and s.human_rating >= min_rating]

        # Query PostgreSQL if available
        if self._postgres and self._postgres.is_connected:
            conditions = []
            params: list[Any] = []
            idx = 1

            if session_id:
                conditions.append(f"session_id = ${idx}")
                params.append(session_id)
                idx += 1
            if model:
                conditions.append(f"model = ${idx}")
                params.append(model)
                idx += 1
            if min_rating:
                conditions.append(f"human_rating >= ${idx}")
                params.append(min_rating)
                idx += 1

            where = " AND ".join(conditions) if conditions else "TRUE"

            try:
                rows = await self._postgres.fetch(
                    f"""
                    SELECT metadata FROM rlhf_samples
                    WHERE {where}
                    ORDER BY timestamp DESC
                    LIMIT ${idx}
                    """,
                    *params,
                    limit,
                )
                for row in rows:
                    sample = RLHFSample.from_dict(json.loads(row["metadata"]))
                    samples.append(sample)
            except Exception as e:
                logger.error(f"Failed to query RLHF samples: {e}")

        return samples[:limit]

    async def get_stats(self) -> RLHFDatasetStats:
        """Get statistics about collected data."""
        samples = await self.get_samples(limit=10000)

        stats = RLHFDatasetStats(
            total_samples=len(samples),
            samples_with_explicit_feedback=sum(1 for s in samples if s.has_explicit_feedback),
            samples_with_implicit_signals=sum(1 for s in samples if s.has_implicit_signals),
        )

        # Quality distribution
        for sample in samples:
            quality = sample.inferred_quality.value
            stats.quality_distribution[quality] = stats.quality_distribution.get(quality, 0) + 1

        # Average rating
        rated_samples = [s for s in samples if s.human_rating]
        if rated_samples:
            stats.average_rating = sum(s.human_rating for s in rated_samples) / len(rated_samples)

        # Acceptance rate
        acceptance_samples = [s for s in samples if s.accepted is not None]
        if acceptance_samples:
            stats.acceptance_rate = sum(1 for s in acceptance_samples if s.accepted) / len(acceptance_samples)

        # Model distribution
        for sample in samples:
            if sample.model:
                stats.model_distribution[sample.model] = stats.model_distribution.get(sample.model, 0) + 1

        return stats

    async def export_for_training(
        self,
        output_path: str,
        format: str = "jsonl",
        min_rating: int | None = None,
    ) -> int:
        """
        Export samples for reward model training.

        Args:
            output_path: Path to output file
            format: Export format (jsonl, json, csv)
            min_rating: Only export samples with this rating or higher

        Returns:
            Number of samples exported
        """
        samples = await self.get_samples(min_rating=min_rating, limit=100000)

        if format == "jsonl":
            with open(output_path, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")
        elif format == "json":
            with open(output_path, "w") as f:
                json.dump([s.to_dict() for s in samples], f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(samples)} RLHF samples to {output_path}")
        return len(samples)


# Singleton instance
_default_collector: RLHFCollector | None = None


def get_rlhf_collector() -> RLHFCollector:
    """Get the default RLHF collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = RLHFCollector()
    return _default_collector
