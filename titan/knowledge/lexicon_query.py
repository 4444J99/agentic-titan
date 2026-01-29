"""Query interface for Body Lexicon.

Provides high-level query methods for searching and exploring
the lexicon, including semantic search, cross-category analogies,
and graph traversal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from titan.knowledge.lexicon import (
    BodyEntry,
    InteractionRule,
    InteractionType,
    LexiconCategory,
)
from titan.knowledge.lexicon_store import LexiconStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.knowledge.lexicon_query")


@dataclass
class AnalogousPattern:
    """Represents a cross-category analogous pattern."""

    source_entry: BodyEntry
    target_entry: BodyEntry
    similarity_score: float
    shared_interaction_types: list[InteractionType] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class QueryResult:
    """Result from a lexicon query."""

    entries: list[BodyEntry]
    scores: list[float]
    total_matches: int
    query: str
    category_filter: LexiconCategory | None = None

    @property
    def top(self) -> BodyEntry | None:
        """Get the top result."""
        return self.entries[0] if self.entries else None


class LexiconQueryInterface:
    """High-level query interface for the Body Lexicon.

    Provides methods for:
    - Semantic search across entries
    - Category-based filtering
    - Alias-based lookup
    - Cross-link traversal
    - Cross-category analogy finding
    """

    def __init__(
        self,
        store: LexiconStore | None = None,
        seed_file: str | Path | None = None,
    ) -> None:
        """Initialize the query interface.

        Args:
            store: LexiconStore instance. If None, creates a new one.
            seed_file: Path to seed lexicon YAML file to load on init.
        """
        self._store = store or LexiconStore()
        self._store.initialize()

        if seed_file:
            self.load_seed_data(seed_file)

    def load_seed_data(self, seed_file: str | Path) -> int:
        """Load seed lexicon data.

        Args:
            seed_file: Path to seed lexicon YAML file.

        Returns:
            Number of entries loaded.
        """
        return self._store.load_seed_data(seed_file)

    def search(
        self,
        query: str,
        limit: int = 10,
        category: LexiconCategory | None = None,
    ) -> QueryResult:
        """Search the lexicon semantically.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            category: Optional category filter.

        Returns:
            QueryResult with matching entries.
        """
        results = self._store.search_semantic(query, limit=limit, category_filter=category)

        entries = [r[0] for r in results]
        scores = [r[1] for r in results]

        return QueryResult(
            entries=entries,
            scores=scores,
            total_matches=len(entries),
            query=query,
            category_filter=category,
        )

    def search_by_category(
        self,
        category: LexiconCategory,
        limit: int = 100,
    ) -> list[BodyEntry]:
        """Get all entries in a category.

        Args:
            category: The category to filter by.
            limit: Maximum number of results.

        Returns:
            List of BodyEntry objects in the category.
        """
        return self._store.list_by_category(category, limit=limit)

    def search_by_alias(
        self,
        alias: str,
        fuzzy: bool = True,
    ) -> list[BodyEntry]:
        """Search for entries matching an alias.

        Args:
            alias: The alias to search for.
            fuzzy: Whether to use fuzzy matching.

        Returns:
            List of matching BodyEntry objects.
        """
        all_entries = self._store.list_all()
        matches: list[BodyEntry] = []

        alias_normalized = alias.lower().replace(" ", "_").replace("-", "_")

        for entry in all_entries:
            if entry.matches_alias(alias):
                matches.append(entry)
            elif fuzzy:
                # Additional fuzzy matching
                for entry_alias in entry.aliases:
                    entry_alias_normalized = entry_alias.lower()
                    if (
                        alias_normalized in entry_alias_normalized
                        or entry_alias_normalized in alias_normalized
                    ):
                        if entry not in matches:
                            matches.append(entry)
                        break

        return matches

    def get_by_id(self, entry_id: str) -> BodyEntry | None:
        """Get an entry by its ID.

        Args:
            entry_id: The entry ID.

        Returns:
            BodyEntry if found, None otherwise.
        """
        return self._store.get(entry_id)

    def get_linked_bodies(
        self,
        entry_id: str,
        max_depth: int = 2,
    ) -> list[BodyEntry]:
        """Get all bodies linked to an entry.

        Traverses cross-links up to max_depth levels.

        Args:
            entry_id: Starting entry ID.
            max_depth: Maximum link traversal depth.

        Returns:
            List of linked BodyEntry objects.
        """
        visited: set[str] = set()
        result: list[BodyEntry] = []

        def traverse(eid: str, depth: int) -> None:
            if depth > max_depth or eid in visited:
                return

            visited.add(eid)
            entry = self._store.get(eid)
            if not entry:
                return

            result.append(entry)

            for link in entry.links:
                # Links can be plain IDs or $CATEGORY:ID format
                if ":" in link:
                    _, linked_id = link.rsplit(":", 1)
                else:
                    linked_id = link

                traverse(linked_id, depth + 1)

        entry = self._store.get(entry_id)
        if entry:
            for link in entry.links:
                if ":" in link:
                    _, linked_id = link.rsplit(":", 1)
                else:
                    linked_id = link
                traverse(linked_id, 1)

        return result

    def find_analogies(
        self,
        source_entry_id: str,
        target_category: LexiconCategory | None = None,
        limit: int = 5,
    ) -> list[AnalogousPattern]:
        """Find analogous patterns across categories.

        Given a source entry, finds similar patterns in other categories
        based on shared interaction types and semantic similarity.

        Args:
            source_entry_id: The source entry ID.
            target_category: Optional target category to search in.
            limit: Maximum number of analogies to return.

        Returns:
            List of AnalogousPattern objects.
        """
        source = self._store.get(source_entry_id)
        if not source:
            return []

        analogies: list[AnalogousPattern] = []

        # Get source interaction types
        source_types = {r.interaction_type for r in source.interaction_rules}

        # Search semantically in other categories
        search_results = self._store.search_semantic(
            source.searchable_text,
            limit=limit * 3,  # Get more to filter
            category_filter=target_category,
        )

        for entry, score in search_results:
            # Skip same entry and same category (unless target specified)
            if entry.id == source_entry_id:
                continue
            if target_category is None and entry.category == source.category:
                continue

            # Find shared interaction types
            entry_types = {r.interaction_type for r in entry.interaction_rules}
            shared_types = list(source_types & entry_types)

            # Calculate combined score
            type_bonus = 0.2 * len(shared_types) / max(len(source_types), 1)
            combined_score = score + type_bonus

            reasoning = self._generate_analogy_reasoning(source, entry, shared_types)

            analogies.append(
                AnalogousPattern(
                    source_entry=source,
                    target_entry=entry,
                    similarity_score=combined_score,
                    shared_interaction_types=shared_types,
                    reasoning=reasoning,
                )
            )

        # Sort by combined score
        analogies.sort(key=lambda x: x.similarity_score, reverse=True)
        return analogies[:limit]

    def _generate_analogy_reasoning(
        self,
        source: BodyEntry,
        target: BodyEntry,
        shared_types: list[InteractionType],
    ) -> str:
        """Generate reasoning for why two entries are analogous."""
        if not shared_types:
            return (
                f"Semantic similarity between {source.id} ({source.category.value}) "
                f"and {target.id} ({target.category.value})"
            )

        type_names = [t.value for t in shared_types]
        return (
            f"{source.id} and {target.id} share interaction patterns: "
            f"{', '.join(type_names)}. "
            f"Source ({source.category.value}): {source.referent[:50]}... "
            f"Target ({target.category.value}): {target.referent[:50]}..."
        )

    def search_by_interaction_type(
        self,
        interaction_type: InteractionType,
        limit: int = 20,
    ) -> list[BodyEntry]:
        """Find entries that use a specific interaction type.

        Args:
            interaction_type: The interaction type to search for.
            limit: Maximum number of results.

        Returns:
            List of matching BodyEntry objects.
        """
        all_entries = self._store.list_all(limit=1000)
        matches: list[BodyEntry] = []

        for entry in all_entries:
            for rule in entry.interaction_rules:
                if rule.interaction_type == interaction_type:
                    matches.append(entry)
                    break

            if len(matches) >= limit:
                break

        return matches

    def get_interaction_rules(
        self,
        entry_id: str,
    ) -> list[InteractionRule]:
        """Get interaction rules for an entry.

        Args:
            entry_id: The entry ID.

        Returns:
            List of InteractionRule objects.
        """
        entry = self._store.get(entry_id)
        if not entry:
            return []
        return entry.interaction_rules

    def get_categories_summary(self) -> dict[LexiconCategory, int]:
        """Get a summary of entries per category.

        Returns:
            Dict mapping category to entry count.
        """
        summary: dict[LexiconCategory, int] = {}

        for category in LexiconCategory:
            entries = self._store.list_by_category(category, limit=10000)
            summary[category] = len(entries)

        return summary

    def add_learned_entry(
        self,
        body_id: str,
        category: LexiconCategory,
        referent: str,
        notes: str = "",
        aliases: list[str] | None = None,
        links: list[str] | None = None,
        interaction_rules: list[InteractionRule] | None = None,
    ) -> BodyEntry:
        """Add a new learned entry to the lexicon.

        Called when agents learn new assembly patterns.

        Args:
            body_id: Unique ID for the entry.
            category: The category this body belongs to.
            referent: Description of what this body refers to.
            notes: Additional notes.
            aliases: Alternative names for this body.
            links: Cross-links to other entries.
            interaction_rules: How units in this body interact.

        Returns:
            The created BodyEntry.
        """
        entry = BodyEntry(
            id=body_id,
            category=category,
            aliases=aliases or [],
            referent=referent,
            notes=notes,
            links=links or [],
            interaction_rules=interaction_rules or [],
            source="learned",
        )

        self._store.add(entry)
        logger.info(f"Added learned entry: {body_id}")

        return entry

    def total_entries(self) -> int:
        """Get total number of entries in the lexicon.

        Returns:
            Total entry count.
        """
        return self._store.count()
