"""
Decision Module - Multi-agent voting and consensus.

Provides:
- VotingSession: Manage agent votes
- ConsensusEngine: Aggregate votes into decisions
- DecisionProtocol: Different voting strategies
"""

from hive.decision.consensus import (
    ConsensusConfig,
    ConsensusEngine,
    ConsensusResult,
)
from hive.decision.voting import (
    Vote,
    VotingResult,
    VotingSession,
    VotingStrategy,
)

__all__ = [
    "Vote",
    "VotingSession",
    "VotingStrategy",
    "VotingResult",
    "ConsensusEngine",
    "ConsensusConfig",
    "ConsensusResult",
]
