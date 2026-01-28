"""
Titan Learning - Local learning, style adaptation, and RLHF.

Provides:
- LocalTrainer: Train on local code patterns
- StyleAdapter: Adapt to user coding style
- PatternExtractor: Extract coding patterns
- RLHFCollector: Collect RLHF training data
- FeedbackHandler: Handle human feedback
- RewardSignalExtractor: Extract reward signals
"""

from titan.learning.local_trainer import (
    LocalTrainer,
    TrainingConfig,
    TrainingResult,
    StyleAdapter,
    CodingPattern,
    extract_patterns,
)
from titan.learning.rlhf import (
    RLHFSample,
    RLHFCollector,
    RLHFDatasetStats,
    FeedbackType,
    ResponseQuality,
    get_rlhf_collector,
)
from titan.learning.feedback import (
    FeedbackHandler,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackChannel,
    get_feedback_handler,
)
from titan.learning.reward_signals import (
    RewardSignal,
    RewardEstimate,
    RewardSignalExtractor,
    SignalType,
    get_reward_extractor,
)

__all__ = [
    # Local trainer
    "LocalTrainer",
    "TrainingConfig",
    "TrainingResult",
    "StyleAdapter",
    "CodingPattern",
    "extract_patterns",
    # RLHF
    "RLHFSample",
    "RLHFCollector",
    "RLHFDatasetStats",
    "FeedbackType",
    "ResponseQuality",
    "get_rlhf_collector",
    # Feedback
    "FeedbackHandler",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackChannel",
    "get_feedback_handler",
    # Reward signals
    "RewardSignal",
    "RewardEstimate",
    "RewardSignalExtractor",
    "SignalType",
    "get_reward_extractor",
]
