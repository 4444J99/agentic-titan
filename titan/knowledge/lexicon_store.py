"""ChromaDB storage backend for Body Lexicon.

Provides persistent vector storage for lexicon entries with
semantic search capabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from titan.knowledge.lexicon import (
    BodyEntry,
    CATEGORY_INTERACTION_RULES,
    LexiconCategory,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("titan.knowledge.lexicon_store")

# Try to import chromadb, fallback to in-memory if not available
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed, using in-memory store")


class LexiconStore:
    """Storage backend for Body Lexicon entries.

    Uses ChromaDB for vector storage and semantic search when available,
    falls back to in-memory storage otherwise.
    """

    COLLECTION_NAME = "titan_body_lexicon"

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        """Initialize the lexicon store.

        Args:
            persist_directory: Directory for ChromaDB persistence.
                If None, uses in-memory storage.
            embedding_function: Custom embedding function for ChromaDB.
                If None, uses default sentence transformer.
        """
        self._persist_directory = Path(persist_directory) if persist_directory else None
        self._embedding_function = embedding_function
        self._client: Any = None
        self._collection: Any = None
        self._in_memory_store: dict[str, BodyEntry] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the storage backend."""
        if self._initialized:
            return

        if CHROMADB_AVAILABLE and self._persist_directory:
            try:
                self._persist_directory.mkdir(parents=True, exist_ok=True)
                settings = Settings(
                    persist_directory=str(self._persist_directory),
                    anonymized_telemetry=False,
                )
                self._client = chromadb.Client(settings)
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._embedding_function,
                    metadata={"description": "Body Lexicon entries for assembly patterns"},
                )
                logger.info(
                    f"Initialized ChromaDB store at {self._persist_directory}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}, using in-memory")
                self._client = None
                self._collection = None
        elif CHROMADB_AVAILABLE:
            # In-memory ChromaDB
            try:
                self._client = chromadb.Client()
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._embedding_function,
                )
                logger.info("Initialized in-memory ChromaDB store")
            except Exception as e:
                logger.warning(f"Failed to initialize in-memory ChromaDB: {e}")
                self._client = None
                self._collection = None

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure store is initialized before operations."""
        if not self._initialized:
            self.initialize()

    def add(self, entry: BodyEntry) -> None:
        """Add a body entry to the store.

        Args:
            entry: The BodyEntry to add.
        """
        self._ensure_initialized()

        if self._collection is not None:
            try:
                # Check if exists
                existing = self._collection.get(ids=[entry.id])
                if existing and existing["ids"]:
                    # Update existing
                    self._collection.update(
                        ids=[entry.id],
                        documents=[entry.searchable_text],
                        metadatas=[{
                            "category": entry.category.value,
                            "aliases": json.dumps(entry.aliases),
                            "referent": entry.referent,
                            "notes": entry.notes,
                            "links": json.dumps(entry.links),
                            "source": entry.source,
                            "content_hash": entry.content_hash,
                        }],
                    )
                else:
                    self._collection.add(
                        ids=[entry.id],
                        documents=[entry.searchable_text],
                        metadatas=[{
                            "category": entry.category.value,
                            "aliases": json.dumps(entry.aliases),
                            "referent": entry.referent,
                            "notes": entry.notes,
                            "links": json.dumps(entry.links),
                            "source": entry.source,
                            "content_hash": entry.content_hash,
                        }],
                    )
                logger.debug(f"Added entry to ChromaDB: {entry.id}")
            except Exception as e:
                logger.error(f"Failed to add entry to ChromaDB: {e}")
                self._in_memory_store[entry.id] = entry
        else:
            self._in_memory_store[entry.id] = entry

    def add_batch(self, entries: list[BodyEntry]) -> int:
        """Add multiple entries in batch.

        Args:
            entries: List of BodyEntry objects to add.

        Returns:
            Number of entries successfully added.
        """
        self._ensure_initialized()
        added = 0

        if self._collection is not None:
            try:
                ids = [e.id for e in entries]
                documents = [e.searchable_text for e in entries]
                metadatas = [
                    {
                        "category": e.category.value,
                        "aliases": json.dumps(e.aliases),
                        "referent": e.referent,
                        "notes": e.notes,
                        "links": json.dumps(e.links),
                        "source": e.source,
                        "content_hash": e.content_hash,
                    }
                    for e in entries
                ]

                # ChromaDB upsert behavior
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                added = len(entries)
                logger.info(f"Batch added {added} entries to ChromaDB")
            except Exception as e:
                logger.error(f"Failed batch add to ChromaDB: {e}")
                # Fallback to in-memory
                for entry in entries:
                    self._in_memory_store[entry.id] = entry
                    added += 1
        else:
            for entry in entries:
                self._in_memory_store[entry.id] = entry
                added += 1

        return added

    def get(self, entry_id: str) -> BodyEntry | None:
        """Get an entry by ID.

        Args:
            entry_id: The entry ID to retrieve.

        Returns:
            BodyEntry if found, None otherwise.
        """
        self._ensure_initialized()

        if self._collection is not None:
            try:
                result = self._collection.get(ids=[entry_id], include=["metadatas"])
                if result and result["ids"]:
                    metadata = result["metadatas"][0]
                    return BodyEntry(
                        id=entry_id,
                        category=LexiconCategory(metadata["category"]),
                        aliases=json.loads(metadata.get("aliases", "[]")),
                        referent=metadata.get("referent", ""),
                        notes=metadata.get("notes", ""),
                        links=json.loads(metadata.get("links", "[]")),
                        source=metadata.get("source", "seed"),
                    )
            except Exception as e:
                logger.error(f"Failed to get entry from ChromaDB: {e}")

        return self._in_memory_store.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID.

        Args:
            entry_id: The entry ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        self._ensure_initialized()

        if self._collection is not None:
            try:
                self._collection.delete(ids=[entry_id])
                logger.debug(f"Deleted entry from ChromaDB: {entry_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete from ChromaDB: {e}")

        if entry_id in self._in_memory_store:
            del self._in_memory_store[entry_id]
            return True

        return False

    def search_semantic(
        self,
        query: str,
        limit: int = 10,
        category_filter: LexiconCategory | None = None,
    ) -> list[tuple[BodyEntry, float]]:
        """Perform semantic search using embeddings.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            category_filter: Optional category to filter by.

        Returns:
            List of (BodyEntry, similarity_score) tuples.
        """
        self._ensure_initialized()
        results: list[tuple[BodyEntry, float]] = []

        if self._collection is not None:
            try:
                where_filter = None
                if category_filter:
                    where_filter = {"category": category_filter.value}

                search_results = self._collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where=where_filter,
                    include=["metadatas", "distances"],
                )

                if search_results and search_results["ids"]:
                    for i, entry_id in enumerate(search_results["ids"][0]):
                        metadata = search_results["metadatas"][0][i]
                        distance = search_results["distances"][0][i] if search_results.get("distances") else 0.0
                        # Convert distance to similarity (ChromaDB returns L2 distance)
                        similarity = 1.0 / (1.0 + distance)

                        entry = BodyEntry(
                            id=entry_id,
                            category=LexiconCategory(metadata["category"]),
                            aliases=json.loads(metadata.get("aliases", "[]")),
                            referent=metadata.get("referent", ""),
                            notes=metadata.get("notes", ""),
                            links=json.loads(metadata.get("links", "[]")),
                            source=metadata.get("source", "seed"),
                        )
                        results.append((entry, similarity))

                return results
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")

        # Fallback: simple text matching
        query_lower = query.lower()
        for entry in self._in_memory_store.values():
            if category_filter and entry.category != category_filter:
                continue

            score = 0.0
            if query_lower in entry.searchable_text.lower():
                score = 0.5
            if entry.matches_alias(query):
                score = 0.8
            if score > 0:
                results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def list_by_category(
        self,
        category: LexiconCategory,
        limit: int = 100,
    ) -> list[BodyEntry]:
        """List all entries in a category.

        Args:
            category: The category to filter by.
            limit: Maximum number of results.

        Returns:
            List of BodyEntry objects.
        """
        self._ensure_initialized()
        entries: list[BodyEntry] = []

        if self._collection is not None:
            try:
                results = self._collection.get(
                    where={"category": category.value},
                    include=["metadatas"],
                    limit=limit,
                )

                if results and results["ids"]:
                    for i, entry_id in enumerate(results["ids"]):
                        metadata = results["metadatas"][i]
                        entry = BodyEntry(
                            id=entry_id,
                            category=LexiconCategory(metadata["category"]),
                            aliases=json.loads(metadata.get("aliases", "[]")),
                            referent=metadata.get("referent", ""),
                            notes=metadata.get("notes", ""),
                            links=json.loads(metadata.get("links", "[]")),
                            source=metadata.get("source", "seed"),
                        )
                        entries.append(entry)
                return entries
            except Exception as e:
                logger.error(f"Failed to list by category: {e}")

        # Fallback to in-memory
        for entry in self._in_memory_store.values():
            if entry.category == category:
                entries.append(entry)
                if len(entries) >= limit:
                    break

        return entries

    def list_all(self, limit: int = 1000) -> list[BodyEntry]:
        """List all entries in the store.

        Args:
            limit: Maximum number of results.

        Returns:
            List of all BodyEntry objects.
        """
        self._ensure_initialized()
        entries: list[BodyEntry] = []

        if self._collection is not None:
            try:
                results = self._collection.get(
                    include=["metadatas"],
                    limit=limit,
                )

                if results and results["ids"]:
                    for i, entry_id in enumerate(results["ids"]):
                        metadata = results["metadatas"][i]
                        entry = BodyEntry(
                            id=entry_id,
                            category=LexiconCategory(metadata["category"]),
                            aliases=json.loads(metadata.get("aliases", "[]")),
                            referent=metadata.get("referent", ""),
                            notes=metadata.get("notes", ""),
                            links=json.loads(metadata.get("links", "[]")),
                            source=metadata.get("source", "seed"),
                        )
                        entries.append(entry)
                return entries
            except Exception as e:
                logger.error(f"Failed to list all: {e}")

        return list(self._in_memory_store.values())[:limit]

    def count(self) -> int:
        """Get total number of entries.

        Returns:
            Number of entries in the store.
        """
        self._ensure_initialized()

        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception as e:
                logger.error(f"Failed to count: {e}")

        return len(self._in_memory_store)

    def load_seed_data(self, seed_file: str | Path) -> int:
        """Load seed lexicon data from YAML file.

        Args:
            seed_file: Path to seed lexicon YAML file.

        Returns:
            Number of entries loaded.
        """
        import yaml

        seed_path = Path(seed_file)
        if not seed_path.exists():
            logger.warning(f"Seed file not found: {seed_path}")
            return 0

        try:
            with open(seed_path) as f:
                data = yaml.safe_load(f)

            entries: list[BodyEntry] = []
            body_lexicon = data.get("$BODY_LEXICON", data)

            for category_id, bodies in body_lexicon.items():
                if category_id == "$CROSS_LINKS_INDEX":
                    continue

                if not category_id.startswith("$CATEGORY_"):
                    continue

                category = LexiconCategory.from_category_id(category_id)

                # Get default interaction rules for this category
                default_rules = CATEGORY_INTERACTION_RULES.get(category, [])

                for body_id, body_data in bodies.items():
                    if not isinstance(body_data, dict):
                        continue

                    entry = BodyEntry.from_lexicon_entry(
                        body_id=body_id,
                        category=category,
                        entry_data=body_data,
                    )
                    # Assign default interaction rules based on category
                    entry.interaction_rules = list(default_rules)
                    entries.append(entry)

            return self.add_batch(entries)

        except Exception as e:
            logger.error(f"Failed to load seed data: {e}")
            return 0

    def clear(self) -> None:
        """Clear all entries from the store."""
        self._ensure_initialized()

        if self._collection is not None:
            try:
                # Delete and recreate collection
                self._client.delete_collection(self.COLLECTION_NAME)
                self._collection = self._client.create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._embedding_function,
                )
                logger.info("Cleared ChromaDB collection")
            except Exception as e:
                logger.error(f"Failed to clear ChromaDB: {e}")

        self._in_memory_store.clear()
