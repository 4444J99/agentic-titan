"""
Titan Learning - Preference Pairs Builder

Builds preference pair datasets from RLHF samples for reward model training.
Supports multiple extraction methods: ratings, corrections, and regenerations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from titan.learning.rlhf import RLHFSample, ResponseQuality

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.learning.preference_pairs")


@dataclass
class PreferencePair:
    """
    A preference pair for reward model training.

    Contains a prompt with two responses: chosen (preferred) and rejected.
    """

    id: UUID = field(default_factory=uuid4)
    prompt: str = ""
    chosen: str = ""  # Preferred response
    rejected: str = ""  # Non-preferred response
    margin: float = 0.0  # Preference strength (0-1)
    source: str = "unknown"  # How the pair was created
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "margin": self.margin,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencePair:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else uuid4(),
            prompt=data.get("prompt", ""),
            chosen=data.get("chosen", ""),
            rejected=data.get("rejected", ""),
            margin=data.get("margin", 0.0),
            source=data.get("source", "unknown"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )

    def is_valid(self) -> bool:
        """Check if this is a valid preference pair."""
        return bool(self.prompt and self.chosen and self.rejected and self.chosen != self.rejected)


@dataclass
class PreferencePairDataset:
    """Collection of preference pairs with metadata."""

    pairs: list[PreferencePair] = field(default_factory=list)
    name: str = "unnamed"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_count: dict[str, int] = field(default_factory=dict)

    def add(self, pair: PreferencePair) -> None:
        """Add a preference pair to the dataset."""
        if pair.is_valid():
            self.pairs.append(pair)
            self.source_count[pair.source] = self.source_count.get(pair.source, 0) + 1

    def __len__(self) -> int:
        return len(self.pairs)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [p.to_dict() for p in self.pairs]

    def to_hf_format(self) -> list[dict[str, str]]:
        """
        Convert to HuggingFace dataset format.

        Returns list of dicts with keys: prompt, chosen, rejected
        """
        return [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in self.pairs
        ]

    def filter_by_margin(self, min_margin: float = 0.0) -> PreferencePairDataset:
        """Create a new dataset filtered by minimum margin."""
        filtered = PreferencePairDataset(
            name=f"{self.name}_filtered",
        )
        for pair in self.pairs:
            if pair.margin >= min_margin:
                filtered.add(pair)
        return filtered

    def split(
        self, train_ratio: float = 0.8
    ) -> tuple[PreferencePairDataset, PreferencePairDataset]:
        """Split into train and eval datasets."""
        import random

        shuffled = self.pairs.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)

        train_dataset = PreferencePairDataset(name=f"{self.name}_train")
        eval_dataset = PreferencePairDataset(name=f"{self.name}_eval")

        for pair in shuffled[:split_idx]:
            train_dataset.add(pair)
        for pair in shuffled[split_idx:]:
            eval_dataset.add(pair)

        return train_dataset, eval_dataset


class PreferencePairBuilder:
    """
    Builds preference pairs from various RLHF data sources.

    Supports:
    - Rating-based pairs (higher rated vs lower rated)
    - Correction-based pairs (corrected vs original)
    - Regeneration-based pairs (accepted vs regenerated)
    - Combined extraction from all sources
    """

    def __init__(
        self,
        min_rating_diff: int = 1,
        min_edit_ratio: float = 0.1,
    ) -> None:
        """
        Initialize the preference pair builder.

        Args:
            min_rating_diff: Minimum rating difference to create a pair
            min_edit_ratio: Minimum edit ratio to consider a correction significant
        """
        self._min_rating_diff = min_rating_diff
        self._min_edit_ratio = min_edit_ratio

    def from_rlhf_samples(
        self,
        samples: list[RLHFSample],
    ) -> PreferencePairDataset:
        """
        Extract preference pairs from RLHF samples using all methods.

        Args:
            samples: List of RLHF samples

        Returns:
            PreferencePairDataset with extracted pairs
        """
        dataset = PreferencePairDataset(name="combined")

        # Extract from different sources
        for pair in self._from_ratings_internal(samples):
            dataset.add(pair)

        for pair in self._from_corrections_internal(samples):
            dataset.add(pair)

        for pair in self._from_regenerations_internal(samples):
            dataset.add(pair)

        logger.info(
            f"Built {len(dataset)} preference pairs from {len(samples)} samples. "
            f"Sources: {dataset.source_count}"
        )
        return dataset

    def from_ratings(
        self,
        samples: list[RLHFSample],
    ) -> PreferencePairDataset:
        """
        Create preference pairs from rating comparisons.

        Groups samples by prompt and creates pairs where higher-rated
        responses are preferred over lower-rated ones.

        Args:
            samples: List of RLHF samples with ratings

        Returns:
            PreferencePairDataset with rating-based pairs
        """
        dataset = PreferencePairDataset(name="ratings")
        for pair in self._from_ratings_internal(samples):
            dataset.add(pair)
        return dataset

    def _from_ratings_internal(
        self,
        samples: list[RLHFSample],
    ) -> list[PreferencePair]:
        """Internal method to extract rating-based pairs."""
        pairs = []

        # Filter to samples with ratings
        rated_samples = [s for s in samples if s.human_rating is not None]

        # Group by prompt
        prompt_groups: dict[str, list[RLHFSample]] = {}
        for sample in rated_samples:
            prompt_key = sample.prompt[:500]  # Truncate for grouping
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = []
            prompt_groups[prompt_key].append(sample)

        # Create pairs within each group
        for prompt, group in prompt_groups.items():
            if len(group) < 2:
                continue

            # Sort by rating
            sorted_group = sorted(group, key=lambda s: s.human_rating or 0, reverse=True)

            # Create pairs between different rating levels
            for i, better in enumerate(sorted_group):
                for worse in sorted_group[i + 1 :]:
                    rating_diff = (better.human_rating or 0) - (worse.human_rating or 0)
                    if rating_diff >= self._min_rating_diff:
                        margin = rating_diff / 4.0  # Normalize to 0-1 (max diff is 4)
                        pairs.append(
                            PreferencePair(
                                prompt=better.prompt,
                                chosen=better.response,
                                rejected=worse.response,
                                margin=min(1.0, margin),
                                source="rating",
                                metadata={
                                    "chosen_rating": better.human_rating,
                                    "rejected_rating": worse.human_rating,
                                    "chosen_sample_id": str(better.id),
                                    "rejected_sample_id": str(worse.id),
                                },
                            )
                        )

        logger.debug(f"Extracted {len(pairs)} pairs from ratings")
        return pairs

    def from_corrections(
        self,
        samples: list[RLHFSample],
    ) -> PreferencePairDataset:
        """
        Create preference pairs from user corrections.

        When a user corrects a response, the correction is preferred
        over the original.

        Args:
            samples: List of RLHF samples with corrections

        Returns:
            PreferencePairDataset with correction-based pairs
        """
        dataset = PreferencePairDataset(name="corrections")
        for pair in self._from_corrections_internal(samples):
            dataset.add(pair)
        return dataset

    def _from_corrections_internal(
        self,
        samples: list[RLHFSample],
    ) -> list[PreferencePair]:
        """Internal method to extract correction-based pairs."""
        pairs = []

        for sample in samples:
            if not sample.correction:
                continue

            # Calculate edit ratio
            edit_ratio = self._calculate_edit_ratio(sample.response, sample.correction)

            if edit_ratio < self._min_edit_ratio:
                continue

            # Higher edit ratio suggests stronger preference for correction
            margin = min(1.0, edit_ratio)

            pairs.append(
                PreferencePair(
                    prompt=sample.prompt,
                    chosen=sample.correction,
                    rejected=sample.response,
                    margin=margin,
                    source="correction",
                    metadata={
                        "edit_ratio": edit_ratio,
                        "sample_id": str(sample.id),
                    },
                )
            )

        logger.debug(f"Extracted {len(pairs)} pairs from corrections")
        return pairs

    def from_regenerations(
        self,
        samples: list[RLHFSample],
    ) -> PreferencePairDataset:
        """
        Create preference pairs from regeneration data.

        When a user accepts a response after regenerations, the accepted
        version is preferred over regenerated versions.

        Args:
            samples: List of RLHF samples with regeneration data

        Returns:
            PreferencePairDataset with regeneration-based pairs
        """
        dataset = PreferencePairDataset(name="regenerations")
        for pair in self._from_regenerations_internal(samples):
            dataset.add(pair)
        return dataset

    def _from_regenerations_internal(
        self,
        samples: list[RLHFSample],
    ) -> list[PreferencePair]:
        """Internal method to extract regeneration-based pairs."""
        pairs = []

        # Group samples by session and prompt to find regeneration chains
        session_prompts: dict[tuple[str, str], list[RLHFSample]] = {}

        for sample in samples:
            key = (sample.session_id, sample.prompt[:500])
            if key not in session_prompts:
                session_prompts[key] = []
            session_prompts[key].append(sample)

        for (session_id, prompt), group in session_prompts.items():
            if len(group) < 2:
                continue

            # Sort by timestamp
            sorted_group = sorted(group, key=lambda s: s.timestamp)

            # Find accepted response (if any)
            accepted = None
            regenerated = []

            for sample in sorted_group:
                if sample.accepted is True:
                    accepted = sample
                elif sample.was_regenerated or sample.accepted is False:
                    regenerated.append(sample)

            if accepted and regenerated:
                for rejected in regenerated:
                    # More regenerations before acceptance = stronger preference
                    margin = min(1.0, 0.3 + 0.2 * rejected.regeneration_count)
                    pairs.append(
                        PreferencePair(
                            prompt=accepted.prompt,
                            chosen=accepted.response,
                            rejected=rejected.response,
                            margin=margin,
                            source="regeneration",
                            metadata={
                                "session_id": session_id,
                                "regeneration_count": rejected.regeneration_count,
                                "chosen_sample_id": str(accepted.id),
                                "rejected_sample_id": str(rejected.id),
                            },
                        )
                    )

        logger.debug(f"Extracted {len(pairs)} pairs from regenerations")
        return pairs

    def from_quality_comparison(
        self,
        samples: list[RLHFSample],
    ) -> PreferencePairDataset:
        """
        Create preference pairs from inferred quality comparisons.

        Uses the inferred_quality property to create pairs.

        Args:
            samples: List of RLHF samples

        Returns:
            PreferencePairDataset with quality-based pairs
        """
        dataset = PreferencePairDataset(name="quality")

        # Quality ranking
        quality_rank = {
            ResponseQuality.EXCELLENT: 5,
            ResponseQuality.GOOD: 4,
            ResponseQuality.ACCEPTABLE: 3,
            ResponseQuality.POOR: 2,
            ResponseQuality.UNACCEPTABLE: 1,
        }

        # Group by prompt
        prompt_groups: dict[str, list[RLHFSample]] = {}
        for sample in samples:
            prompt_key = sample.prompt[:500]
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = []
            prompt_groups[prompt_key].append(sample)

        for prompt, group in prompt_groups.items():
            if len(group) < 2:
                continue

            # Sort by quality
            sorted_group = sorted(
                group, key=lambda s: quality_rank.get(s.inferred_quality, 0), reverse=True
            )

            for i, better in enumerate(sorted_group):
                for worse in sorted_group[i + 1 :]:
                    better_rank = quality_rank.get(better.inferred_quality, 0)
                    worse_rank = quality_rank.get(worse.inferred_quality, 0)

                    if better_rank > worse_rank:
                        margin = (better_rank - worse_rank) / 4.0
                        pair = PreferencePair(
                            prompt=better.prompt,
                            chosen=better.response,
                            rejected=worse.response,
                            margin=min(1.0, margin),
                            source="quality",
                            metadata={
                                "chosen_quality": better.inferred_quality.value,
                                "rejected_quality": worse.inferred_quality.value,
                            },
                        )
                        dataset.add(pair)

        return dataset

    def to_hf_dataset(
        self,
        dataset: PreferencePairDataset,
    ) -> Any:
        """
        Convert to HuggingFace Dataset format.

        Args:
            dataset: PreferencePairDataset to convert

        Returns:
            HuggingFace Dataset (or dict if datasets not installed)
        """
        data = dataset.to_hf_format()

        try:
            from datasets import Dataset

            return Dataset.from_list(data)
        except ImportError:
            logger.warning("datasets library not installed, returning dict format")
            return {
                "prompt": [d["prompt"] for d in data],
                "chosen": [d["chosen"] for d in data],
                "rejected": [d["rejected"] for d in data],
            }

    def _calculate_edit_ratio(self, original: str, edited: str) -> float:
        """Calculate the edit ratio between two strings."""
        if not original or not edited:
            return 0.0

        original_words = set(original.split())
        edited_words = set(edited.split())

        if not original_words and not edited_words:
            return 0.0

        diff = original_words.symmetric_difference(edited_words)
        total = len(original_words.union(edited_words))

        return len(diff) / total if total > 0 else 0.0


# Factory function
_builder: PreferencePairBuilder | None = None


def get_preference_pair_builder() -> PreferencePairBuilder:
    """Get the default preference pair builder."""
    global _builder
    if _builder is None:
        _builder = PreferencePairBuilder()
    return _builder
