"""Body Lexicon Knowledge System.

This module provides a queryable taxonomy of assembly patterns based on
the "Morphodynamics of Assembly" framework.

Components:
- lexicon: Data models for Body Lexicon entries
- lexicon_query: Query interface for semantic search
- lexicon_store: ChromaDB storage backend
- lexicon_learner: Agent learning interface
"""

from titan.knowledge.lexicon import (
    BodyEntry,
    InteractionRule,
    LexiconCategory,
)
from titan.knowledge.lexicon_query import LexiconQueryInterface
from titan.knowledge.lexicon_store import LexiconStore

__all__ = [
    "BodyEntry",
    "InteractionRule",
    "LexiconCategory",
    "LexiconQueryInterface",
    "LexiconStore",
]
