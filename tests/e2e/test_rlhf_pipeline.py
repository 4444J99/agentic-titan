"""
End-to-End RLHF Pipeline Tests.

Tests complete RLHF training flow from data collection through deployment.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Mark all tests in this module as e2e
pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_rlhf_samples():
    """Create mock RLHF samples."""
    from titan.learning.rlhf import RLHFSample, ResponseQuality, FeedbackType

    return [
        RLHFSample(
            prompt="Write hello world in Python",
            response="print('hello world')",
            human_rating=5,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            session_id="session-1",
            agent_id="coder",
            model="gpt-4",
        ),
        RLHFSample(
            prompt="Write hello world in Python",
            response="console.log('hello')",  # Wrong language
            human_rating=2,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            session_id="session-1",
            agent_id="coder",
            model="gpt-4",
        ),
        RLHFSample(
            prompt="Explain recursion",
            response="Recursion is when a function calls itself to solve smaller instances of the same problem.",
            human_rating=4,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            session_id="session-2",
            agent_id="researcher",
            model="claude-3",
        ),
        RLHFSample(
            prompt="Explain recursion",
            response="It loops.",  # Too brief
            human_rating=1,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            session_id="session-2",
            agent_id="researcher",
            model="claude-3",
        ),
    ]


@pytest.fixture
def mock_preference_pairs():
    """Create mock preference pairs."""
    from titan.learning.preference_pairs import PreferencePair

    return [
        PreferencePair(
            prompt="Write hello world in Python",
            chosen="print('hello world')",
            rejected="console.log('hello')",
            margin=0.75,
            source="rating",
        ),
        PreferencePair(
            prompt="Explain recursion",
            chosen="Recursion is when a function calls itself to solve smaller instances of the same problem.",
            rejected="It loops.",
            margin=0.75,
            source="rating",
        ),
    ]


@pytest.fixture
def mock_preference_dataset(mock_preference_pairs):
    """Create mock preference dataset."""
    from titan.learning.preference_pairs import PreferencePairDataset

    dataset = PreferencePairDataset(name="test-dataset")
    for pair in mock_preference_pairs:
        dataset.add(pair)
    return dataset


@pytest.fixture
def mock_model_outputs():
    """Create mock model outputs for evaluation."""
    return [
        "This is a clear and helpful response that explains the concept well.",
        "Here is another well-structured answer with proper examples.",
        "The solution involves using the following approach to solve the problem.",
        "Let me break this down step by step for clarity.",
    ]


@pytest.fixture
def mock_baseline_outputs():
    """Create mock baseline model outputs."""
    return [
        "Here is a response.",
        "This is an answer.",
        "The solution is this.",
        "Step by step.",
    ]


@pytest.fixture
def mock_prompts():
    """Create mock prompts for evaluation."""
    return [
        "Explain how to implement a binary search tree",
        "What is the difference between HTTP and HTTPS?",
        "How do you handle errors in Python?",
        "Describe the observer pattern",
    ]


# ============================================================================
# Preference Pair Tests
# ============================================================================


class TestPreferencePairsE2E:
    """End-to-end preference pair building tests."""

    async def test_build_preference_pairs_from_rlhf_samples(self, mock_rlhf_samples):
        """Test building preference pairs from RLHF samples."""
        from titan.learning.preference_pairs import PreferencePairBuilder

        builder = PreferencePairBuilder()
        dataset = builder.from_rlhf_samples(mock_rlhf_samples)

        assert len(dataset) > 0
        for pair in dataset.pairs:
            assert pair.prompt
            assert pair.chosen
            assert pair.rejected
            assert pair.chosen != pair.rejected
            assert 0 <= pair.margin <= 1

    async def test_preference_pairs_from_ratings(self, mock_rlhf_samples):
        """Test building pairs specifically from rating comparisons."""
        from titan.learning.preference_pairs import PreferencePairBuilder

        builder = PreferencePairBuilder(min_rating_diff=1)
        dataset = builder.from_ratings(mock_rlhf_samples)

        # Should create pairs where higher rated is preferred
        for pair in dataset.pairs:
            assert pair.source == "rating"
            assert pair.margin > 0

    async def test_preference_dataset_hf_format(self, mock_preference_dataset):
        """Test converting dataset to HuggingFace format."""
        hf_format = mock_preference_dataset.to_hf_format()

        assert isinstance(hf_format, list)
        assert len(hf_format) == len(mock_preference_dataset)
        for item in hf_format:
            assert "prompt" in item
            assert "chosen" in item
            assert "rejected" in item

    async def test_preference_dataset_split(self, mock_preference_dataset):
        """Test splitting dataset into train/eval."""
        # Add more pairs to ensure split works
        from titan.learning.preference_pairs import PreferencePair

        for i in range(10):
            mock_preference_dataset.add(PreferencePair(
                prompt=f"Test prompt {i}",
                chosen=f"Good response {i}",
                rejected=f"Bad response {i}",
                margin=0.5,
            ))

        train, eval_ds = mock_preference_dataset.split(train_ratio=0.8)

        assert len(train) + len(eval_ds) == len(mock_preference_dataset)
        assert len(train) >= len(eval_ds)

    async def test_preference_dataset_filter_by_margin(self, mock_preference_pairs):
        """Test filtering dataset by minimum margin."""
        from titan.learning.preference_pairs import PreferencePairDataset, PreferencePair

        dataset = PreferencePairDataset(name="test")
        dataset.add(PreferencePair(
            prompt="p1", chosen="c1", rejected="r1", margin=0.3
        ))
        dataset.add(PreferencePair(
            prompt="p2", chosen="c2", rejected="r2", margin=0.7
        ))
        dataset.add(PreferencePair(
            prompt="p3", chosen="c3", rejected="r3", margin=0.9
        ))

        filtered = dataset.filter_by_margin(min_margin=0.5)

        assert len(filtered) == 2
        for pair in filtered.pairs:
            assert pair.margin >= 0.5


# ============================================================================
# Reward Model Tests
# ============================================================================


class TestRewardModelE2E:
    """End-to-end reward model tests."""

    async def test_reward_model_config(self):
        """Test reward model configuration."""
        from titan.learning.reward_model import RewardModelConfig

        config = RewardModelConfig(
            base_model="distilbert-base-uncased",
            epochs=1,
            batch_size=2,
            learning_rate=1e-5,
        )

        assert config.base_model == "distilbert-base-uncased"
        assert config.epochs == 1

        # Test serialization
        config_dict = config.to_dict()
        restored = RewardModelConfig.from_dict(config_dict)
        assert restored.base_model == config.base_model
        assert restored.epochs == config.epochs

    async def test_reward_model_prediction(self):
        """Test reward model prediction interface."""
        from titan.learning.reward_model import RewardModel, RewardModelConfig

        config = RewardModelConfig()
        model = RewardModel(config=config)

        # Test prediction (mock model returns length-based score)
        score1 = model.predict(
            prompt="Write hello world",
            response="print('hello world')"
        )
        score2 = model.predict(
            prompt="Write hello world",
            response="x"  # Very short
        )

        # Both should be floats
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    async def test_reward_model_comparison(self):
        """Test comparing two responses with reward model."""
        from titan.learning.reward_model import RewardModel, RewardModelConfig

        model = RewardModel(config=RewardModelConfig())

        result = model.compare(
            prompt="Explain Python",
            response_a="Python is a programming language known for its readability.",
            response_b="py",
        )

        # Returns 1 if response_a preferred, -1 if response_b, 0 for tie
        assert result in (1, -1, 0)


# ============================================================================
# DPO Trainer Tests
# ============================================================================


class TestDPOTrainerE2E:
    """End-to-end DPO trainer tests."""

    async def test_dpo_config(self):
        """Test DPO configuration."""
        from titan.learning.dpo_trainer import DPOConfig

        config = DPOConfig(
            base_model="gpt2",
            beta=0.1,
            max_steps=10,
            use_peft=True,
        )

        assert config.beta == 0.1
        assert config.use_peft is True

        config_dict = config.to_dict()
        assert config_dict["beta"] == 0.1

    async def test_dpo_trainer_initialization(self):
        """Test DPO trainer initialization."""
        from titan.learning.dpo_trainer import DPOTrainer, DPOConfig

        config = DPOConfig(base_model="gpt2", max_steps=5)
        trainer = DPOTrainer(config)

        assert trainer._config.base_model == "gpt2"
        assert trainer._config.max_steps == 5

    async def test_dpo_metrics_tracking(self):
        """Test DPO metrics dataclass."""
        from titan.learning.dpo_trainer import DPOMetrics

        metrics = DPOMetrics(
            loss=0.5,
            rewards_chosen=0.8,
            rewards_rejected=0.2,
            reward_margin=0.6,
            accuracy=0.85,
        )

        assert metrics.loss == 0.5
        assert metrics.reward_margin == 0.6

        metrics_dict = metrics.to_dict()
        assert metrics_dict["accuracy"] == 0.85


# ============================================================================
# Evaluation Suite Tests
# ============================================================================


class TestEvalSuiteE2E:
    """End-to-end evaluation suite tests."""

    async def test_coherence_score(self, mock_model_outputs):
        """Test coherence scoring."""
        from titan.learning.eval_suite import RLHFEvalSuite

        suite = RLHFEvalSuite()
        result = suite.coherence_score(mock_model_outputs)

        assert result.metric_name == "coherence_score"
        assert 0 <= result.value <= 1
        assert result.samples_evaluated == len(mock_model_outputs)

    async def test_safety_score(self, mock_model_outputs):
        """Test safety scoring."""
        from titan.learning.eval_suite import RLHFEvalSuite

        suite = RLHFEvalSuite()
        result = suite.safety_score(mock_model_outputs)

        assert result.metric_name == "safety_score"
        assert 0 <= result.value <= 1
        assert result.samples_evaluated == len(mock_model_outputs)

    async def test_diversity_score(self, mock_model_outputs):
        """Test diversity scoring."""
        from titan.learning.eval_suite import RLHFEvalSuite

        suite = RLHFEvalSuite()
        result = suite.diversity_score(mock_model_outputs)

        assert result.metric_name == "diversity_score"
        assert 0 <= result.value <= 1
        assert "type_token_ratio" in result.metadata

    async def test_win_rate(self, mock_model_outputs, mock_baseline_outputs, mock_prompts):
        """Test win rate calculation."""
        from titan.learning.eval_suite import RLHFEvalSuite

        suite = RLHFEvalSuite()
        result = suite.win_rate(mock_model_outputs, mock_baseline_outputs, mock_prompts)

        assert result.metric_name == "win_rate"
        assert 0 <= result.value <= 1
        assert result.samples_evaluated == len(mock_prompts)
        assert "wins" in result.metadata

    async def test_full_eval_report(
        self, mock_model_outputs, mock_baseline_outputs, mock_prompts
    ):
        """Test full evaluation report generation."""
        from titan.learning.eval_suite import RLHFEvalSuite

        suite = RLHFEvalSuite()
        report = suite.full_eval(
            model_name="test-model",
            baseline_name="baseline-model",
            model_responses=mock_model_outputs,
            baseline_responses=mock_baseline_outputs,
            prompts=mock_prompts,
        )

        assert report.model_name == "test-model"
        assert report.baseline_name == "baseline-model"
        assert report.win_rate is not None
        assert report.coherence_score is not None
        assert report.safety_score is not None
        assert report.overall_score >= 0
        assert report.recommendation

        # Test summary generation
        summary = report.summary()
        assert "test-model" in summary
        assert "Win Rate" in summary

    async def test_length_analysis(self, mock_model_outputs):
        """Test response length analysis."""
        from titan.learning.eval_suite import RLHFEvalSuite

        suite = RLHFEvalSuite()
        result = suite.length_analysis(mock_model_outputs)

        assert result.metric_name == "length_analysis"
        assert "mean_words" in result.metadata
        assert "std_words" in result.metadata


# ============================================================================
# A/B Deployment Tests
# ============================================================================


class TestABDeploymentE2E:
    """End-to-end A/B testing deployment tests."""

    async def test_start_ab_test(self, tmp_path):
        """Test starting an A/B test."""
        from titan.learning.deployment import (
            RLHFDeployment,
            DeploymentConfig,
        )

        config = DeploymentConfig(
            canary_percentage=0.1,
            min_samples_before_promote=10,
            state_dir=str(tmp_path / "deployment"),
        )
        deployment = RLHFDeployment(config)

        test_id = deployment.start_ab_test(
            new_model="model-v2",
            baseline="model-v1",
        )

        assert test_id
        assert test_id in deployment._active_tests

    async def test_get_ab_stats(self, tmp_path):
        """Test getting A/B test statistics."""
        from titan.learning.deployment import (
            RLHFDeployment,
            DeploymentConfig,
        )

        config = DeploymentConfig(state_dir=str(tmp_path / "deployment"))
        deployment = RLHFDeployment(config)

        test_id = deployment.start_ab_test(
            new_model="model-v2",
            baseline="model-v1",
        )

        stats = deployment.get_ab_stats(test_id)

        assert stats is not None
        assert stats.new_model == "model-v2"
        assert stats.baseline == "model-v1"
        assert stats.status == "running"

    async def test_route_request(self, tmp_path):
        """Test request routing between models."""
        from titan.learning.deployment import (
            RLHFDeployment,
            DeploymentConfig,
        )

        config = DeploymentConfig(
            canary_percentage=0.5,  # 50% to new model
            state_dir=str(tmp_path / "deployment"),
        )
        deployment = RLHFDeployment(config)

        test_id = deployment.start_ab_test(
            new_model="model-v2",
            baseline="model-v1",
        )

        # Route many requests to check distribution
        routes = {"model-v2": 0, "model-v1": 0}
        for _ in range(100):
            model = deployment.route_request(test_id)
            if model in routes:
                routes[model] += 1

        # Both models should receive some traffic
        assert routes["model-v2"] > 0
        assert routes["model-v1"] > 0

    async def test_record_result(self, tmp_path):
        """Test recording request results."""
        from titan.learning.deployment import (
            RLHFDeployment,
            DeploymentConfig,
        )

        config = DeploymentConfig(state_dir=str(tmp_path / "deployment"))
        deployment = RLHFDeployment(config)

        test_id = deployment.start_ab_test(
            new_model="model-v2",
            baseline="model-v1",
        )

        # Record some results
        deployment.record_result(
            test_id=test_id,
            model="model-v2",
            latency_ms=100.0,
            reward=0.8,
            feedback=1,
        )
        deployment.record_result(
            test_id=test_id,
            model="model-v1",
            latency_ms=150.0,
            reward=0.6,
            feedback=-1,
        )

        stats = deployment.get_ab_stats(test_id)

        assert stats.new_model_metrics.requests == 1
        assert stats.baseline_metrics.requests == 1
        assert stats.new_model_metrics.positive_feedback == 1

    async def test_promote_winner(self, tmp_path):
        """Test promoting new model as winner."""
        from titan.learning.deployment import (
            RLHFDeployment,
            DeploymentConfig,
        )

        config = DeploymentConfig(
            enable_gradual_rollout=False,  # Immediate promotion
            state_dir=str(tmp_path / "deployment"),
        )
        deployment = RLHFDeployment(config)

        test_id = deployment.start_ab_test(
            new_model="model-v2",
            baseline="model-v1",
        )

        success = deployment.promote_winner(test_id)

        assert success
        stats = deployment.get_ab_stats(test_id)
        assert stats.status == "completed"
        assert stats.current_percentage == 1.0

    async def test_rollback(self, tmp_path):
        """Test rolling back to baseline model."""
        from titan.learning.deployment import (
            RLHFDeployment,
            DeploymentConfig,
        )

        config = DeploymentConfig(state_dir=str(tmp_path / "deployment"))
        deployment = RLHFDeployment(config)

        test_id = deployment.start_ab_test(
            new_model="model-v2",
            baseline="model-v1",
        )

        success = deployment.rollback(test_id)

        assert success
        stats = deployment.get_ab_stats(test_id)
        assert stats.status == "rolled_back"
        assert stats.current_percentage == 0.0


# ============================================================================
# Experiment Tracking Tests
# ============================================================================


class TestExperimentTrackingE2E:
    """End-to-end experiment tracking tests."""

    async def test_start_run(self, tmp_path):
        """Test starting an experiment run."""
        from titan.learning.experiment import (
            ExperimentTracker,
            ExperimentConfig,
        )

        config = ExperimentConfig(
            backend="local",
            local_dir=str(tmp_path / "experiments"),
        )
        tracker = ExperimentTracker(config)

        run = tracker.start_run(
            name="test-run",
            config={"learning_rate": 1e-5},
            tags=["test"],
        )

        assert run.run_id
        assert run.name == "test-run"
        assert run.config["learning_rate"] == 1e-5
        assert "test" in run.tags

    async def test_log_metrics(self, tmp_path):
        """Test logging metrics to experiment."""
        from titan.learning.experiment import (
            ExperimentTracker,
            ExperimentConfig,
        )

        config = ExperimentConfig(
            backend="local",
            local_dir=str(tmp_path / "experiments"),
        )
        tracker = ExperimentTracker(config)

        run = tracker.start_run(name="metrics-test")

        tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        tracker.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=2)

        assert len(run.metrics["loss"]) == 2
        assert len(run.metrics["accuracy"]) == 2

    async def test_end_run(self, tmp_path):
        """Test ending an experiment run."""
        from titan.learning.experiment import (
            ExperimentTracker,
            ExperimentConfig,
        )

        config = ExperimentConfig(
            backend="local",
            local_dir=str(tmp_path / "experiments"),
        )
        tracker = ExperimentTracker(config)

        run = tracker.start_run(name="end-test")
        run_id = run.run_id

        tracker.end_run(status="completed")

        assert tracker.current_run is None

        # Should be able to retrieve completed run
        retrieved = tracker.get_run(run_id)
        assert retrieved is not None
        assert retrieved.status == "completed"

    async def test_list_runs(self, tmp_path):
        """Test listing experiment runs."""
        from titan.learning.experiment import (
            ExperimentTracker,
            ExperimentConfig,
        )

        config = ExperimentConfig(
            backend="local",
            local_dir=str(tmp_path / "experiments"),
        )
        tracker = ExperimentTracker(config)

        # Create multiple runs
        for i in range(3):
            tracker.start_run(name=f"run-{i}")
            tracker.end_run(status="completed")

        runs = tracker.list_runs()

        assert len(runs) == 3


# ============================================================================
# Learning Pipeline Integration Tests
# ============================================================================


class TestLearningPipelineE2E:
    """End-to-end learning pipeline integration tests."""

    async def test_response_tracking(self):
        """Test tracking a response for RLHF data collection."""
        from titan.learning.pipeline import LearningPipeline

        pipeline = LearningPipeline()

        tracking_id = pipeline.start_response_tracking(
            prompt="Write a function",
            agent_type="coder",
            session_id="session-1",
            model="gpt-4",
        )

        assert tracking_id
        assert tracking_id in pipeline._pending_samples

    async def test_complete_response(self):
        """Test completing response tracking."""
        from titan.learning.pipeline import LearningPipeline

        pipeline = LearningPipeline()

        tracking_id = pipeline.start_response_tracking(
            prompt="Explain recursion",
            agent_type="researcher",
            session_id="session-2",
        )

        sample = pipeline.complete_response(
            tracking_id=tracking_id,
            response="Recursion is a technique where a function calls itself.",
            prompt_tokens=10,
            completion_tokens=50,
        )

        assert sample is not None
        assert sample.response == "Recursion is a technique where a function calls itself."
        assert sample.prompt_tokens == 10

    async def test_process_feedback(self):
        """Test processing user feedback."""
        from titan.learning.pipeline import LearningPipeline, FeedbackResponse

        pipeline = LearningPipeline()

        # Start tracking
        tracking_id = pipeline.start_response_tracking(
            prompt="Test prompt",
            agent_type="coder",
            session_id="session-3",
        )

        # Complete response
        pipeline.complete_response(
            tracking_id=tracking_id,
            response="Test response",
        )

        # Process feedback
        feedback = FeedbackResponse(
            request_id=pipeline._pending_samples[tracking_id].id,
            rating=5,
            accepted=True,
        )

        reward = await pipeline.process_feedback(feedback)

        assert reward is not None
        assert reward.reward > 0  # Positive rating = positive reward
        assert reward.confidence > 0

    async def test_process_batch_completion(self):
        """Test processing batch completion as implicit feedback."""
        from titan.learning.pipeline import LearningPipeline

        pipeline = LearningPipeline()

        reward = await pipeline.process_batch_completion(
            batch_id="batch-123",
            success_rate=85.0,  # Good success rate
            total_tokens=1000,
            total_cost=0.05,
        )

        assert reward is not None
        # High success rate should yield positive reward
        assert reward.reward > 0

    async def test_feedback_callbacks(self):
        """Test feedback event callbacks."""
        from titan.learning.pipeline import LearningPipeline, FeedbackResponse

        pipeline = LearningPipeline()

        callback_received = []

        def on_feedback(feedback, reward):
            callback_received.append((feedback, reward))

        pipeline.on_feedback(on_feedback)

        # Process some feedback
        feedback = FeedbackResponse(rating=4)
        await pipeline.process_feedback(feedback)

        assert len(callback_received) == 1
        assert callback_received[0][0] == feedback

    async def test_learning_metrics(self):
        """Test learning metrics tracking."""
        from titan.learning.pipeline import LearningPipeline, FeedbackResponse

        pipeline = LearningPipeline()

        # Generate some activity
        for i in range(5):
            tid = pipeline.start_response_tracking(
                prompt=f"prompt-{i}",
                agent_type="coder",
                session_id=f"session-{i}",
            )
            pipeline.complete_response(tid, f"response-{i}")

        # Process some feedback
        for i in range(3):
            await pipeline.process_feedback(FeedbackResponse(rating=4))

        metrics = pipeline.get_metrics()

        assert metrics.samples_collected == 5
        assert metrics.feedback_received == 3
        assert metrics.rewards_calculated == 3

    async def test_episode_linking(self):
        """Test linking sessions to episodic learning episodes."""
        from titan.learning.pipeline import LearningPipeline

        pipeline = LearningPipeline()

        # Link session to episode
        pipeline.link_episode("session-1", "episode-abc")

        # Verify link
        episode_id = pipeline.get_episode_id("session-1")
        assert episode_id == "episode-abc"

        # Non-linked session returns None
        assert pipeline.get_episode_id("session-2") is None


# ============================================================================
# Full Pipeline Integration Test
# ============================================================================


class TestFullRLHFPipelineE2E:
    """Full end-to-end RLHF pipeline integration test."""

    async def test_full_pipeline_flow(self, tmp_path):
        """Test complete RLHF pipeline from data to deployment."""
        from titan.learning.pipeline import LearningPipeline, FeedbackResponse
        from titan.learning.preference_pairs import PreferencePairBuilder
        from titan.learning.eval_suite import RLHFEvalSuite
        from titan.learning.deployment import RLHFDeployment, DeploymentConfig
        from titan.learning.experiment import ExperimentTracker, ExperimentConfig

        # 1. Collect RLHF data via learning pipeline
        pipeline = LearningPipeline()

        # Simulate multiple interactions
        for i in range(5):
            tid = pipeline.start_response_tracking(
                prompt=f"Write function {i}",
                agent_type="coder",
                session_id=f"session-{i}",
            )
            pipeline.complete_response(tid, f"def func_{i}(): pass")

            # Provide feedback (simulate user ratings)
            await pipeline.process_feedback(FeedbackResponse(rating=4 if i % 2 == 0 else 2))

        assert pipeline.get_metrics().samples_collected == 5

        # 2. Build preference pairs from collected data
        # (In real scenario, would extract from stored samples)
        from titan.learning.rlhf import RLHFSample

        samples = [
            RLHFSample(
                prompt="Test prompt",
                response="Good response",
                human_rating=5,
            ),
            RLHFSample(
                prompt="Test prompt",
                response="Bad response",
                human_rating=2,
            ),
        ]

        builder = PreferencePairBuilder()
        dataset = builder.from_ratings(samples)

        # 3. Evaluate model quality
        suite = RLHFEvalSuite()

        model_responses = ["Clear explanation with examples."] * 4
        baseline_responses = ["Brief answer."] * 4
        prompts = ["Question 1", "Question 2", "Question 3", "Question 4"]

        report = suite.full_eval(
            model_name="new-model",
            baseline_name="baseline",
            model_responses=model_responses,
            baseline_responses=baseline_responses,
            prompts=prompts,
        )

        assert report.overall_score >= 0

        # 4. Set up A/B testing deployment
        deploy_config = DeploymentConfig(
            state_dir=str(tmp_path / "deploy"),
            canary_percentage=0.2,
        )
        deployment = RLHFDeployment(deploy_config)

        test_id = deployment.start_ab_test(
            new_model="new-model",
            baseline="baseline",
        )

        # 5. Track experiment
        exp_config = ExperimentConfig(
            backend="local",
            local_dir=str(tmp_path / "experiments"),
        )
        tracker = ExperimentTracker(exp_config)

        run = tracker.start_run(
            name="rlhf-test",
            config={"model": "new-model", "dataset_size": len(dataset)},
        )

        tracker.log_metrics({
            "win_rate": report.win_rate.value if report.win_rate else 0,
            "coherence": report.coherence_score.value if report.coherence_score else 0,
            "safety": report.safety_score.value if report.safety_score else 0,
        })

        tracker.end_run(status="completed")

        # Verify end-to-end flow completed
        assert deployment.get_ab_stats(test_id) is not None
        assert tracker.get_run(run.run_id) is not None
