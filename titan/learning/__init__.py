"""
Titan Learning - Local learning, style adaptation, and RLHF.

Provides:
- LocalTrainer: Train on local code patterns
- StyleAdapter: Adapt to user coding style
- PatternExtractor: Extract coding patterns
- RLHFCollector: Collect RLHF training data
- FeedbackHandler: Handle human feedback
- RewardSignalExtractor: Extract reward signals

Phase 18 RLHF Training Pipeline:
- PreferencePairBuilder: Build preference datasets
- RewardModelTrainer: Train reward models
- DPOTrainer: Direct Preference Optimization
- RLHFEvalSuite: Evaluation metrics
- ExperimentTracker: Experiment tracking
- RLHFDeployment: A/B testing and deployment
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

# Phase 18A: RLHF Training Pipeline
from titan.learning.preference_pairs import (
    PreferencePair,
    PreferencePairDataset,
    PreferencePairBuilder,
    get_preference_pair_builder,
)
from titan.learning.reward_model import (
    RewardModelConfig,
    RewardMetrics,
    RewardModel,
    RewardModelTrainer,
    get_reward_model_trainer,
)
from titan.learning.dpo_trainer import (
    DPOConfig,
    DPOMetrics,
    DPOTrainer,
    get_dpo_trainer,
)
from titan.learning.eval_suite import (
    EvalResult,
    EvalReport,
    RLHFEvalSuite,
    get_eval_suite,
)
from titan.learning.experiment import (
    ExperimentConfig,
    ExperimentRun,
    ExperimentTracker,
    get_experiment_tracker,
)
from titan.learning.deployment import (
    DeploymentConfig,
    ABTestStats,
    RLHFDeployment,
    get_rlhf_deployment,
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
    # Phase 18A: Preference pairs
    "PreferencePair",
    "PreferencePairDataset",
    "PreferencePairBuilder",
    "get_preference_pair_builder",
    # Phase 18A: Reward model
    "RewardModelConfig",
    "RewardMetrics",
    "RewardModel",
    "RewardModelTrainer",
    "get_reward_model_trainer",
    # Phase 18A: DPO trainer
    "DPOConfig",
    "DPOMetrics",
    "DPOTrainer",
    "get_dpo_trainer",
    # Phase 18A: Evaluation
    "EvalResult",
    "EvalReport",
    "RLHFEvalSuite",
    "get_eval_suite",
    # Phase 18A: Experiment tracking
    "ExperimentConfig",
    "ExperimentRun",
    "ExperimentTracker",
    "get_experiment_tracker",
    # Phase 18A: Deployment
    "DeploymentConfig",
    "ABTestStats",
    "RLHFDeployment",
    "get_rlhf_deployment",
]
