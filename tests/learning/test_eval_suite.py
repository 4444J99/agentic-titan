"""
Tests for RLHF Evaluation Suite (Phase 18A)
"""

import pytest

from titan.learning.eval_suite import (
    EvalReport,
    EvalResult,
    RLHFEvalSuite,
    get_eval_suite,
)


class TestEvalResult:
    """Tests for EvalResult."""

    def test_create_basic_result(self):
        """Test creating a basic result."""
        result = EvalResult(
            metric_name="accuracy",
            value=0.85,
            samples_evaluated=100,
        )

        assert result.metric_name == "accuracy"
        assert result.value == 0.85
        assert result.samples_evaluated == 100

    def test_result_with_confidence_interval(self):
        """Test result with confidence interval."""
        result = EvalResult(
            metric_name="win_rate",
            value=0.6,
            confidence_interval=(0.55, 0.65),
        )

        assert result.confidence_interval == (0.55, 0.65)

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = EvalResult(
            metric_name="test",
            value=0.5,
            metadata={"key": "value"},
        )

        data = result.to_dict()
        assert data["metric_name"] == "test"
        assert data["value"] == 0.5
        assert data["metadata"] == {"key": "value"}


class TestEvalReport:
    """Tests for EvalReport."""

    def test_create_empty_report(self):
        """Test creating an empty report."""
        report = EvalReport(
            model_name="new-model",
            baseline_name="baseline",
        )

        assert report.model_name == "new-model"
        assert report.baseline_name == "baseline"
        assert report.overall_score == 0.0

    def test_report_with_metrics(self):
        """Test report with metrics."""
        report = EvalReport(
            model_name="test",
            baseline_name="baseline",
            win_rate=EvalResult(metric_name="win_rate", value=0.6),
            coherence_score=EvalResult(metric_name="coherence", value=0.8),
            safety_score=EvalResult(metric_name="safety", value=0.95),
            overall_score=0.75,
            recommendation="DEPLOY",
        )

        assert report.win_rate.value == 0.6
        assert report.overall_score == 0.75

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = EvalReport(model_name="test", baseline_name="base")
        data = report.to_dict()

        assert data["model_name"] == "test"
        assert data["baseline_name"] == "base"
        assert "timestamp" in data

    def test_summary(self):
        """Test summary generation."""
        report = EvalReport(
            model_name="new",
            baseline_name="old",
            win_rate=EvalResult(metric_name="win_rate", value=0.65),
            overall_score=0.7,
            recommendation="CONSIDER",
        )

        summary = report.summary()
        assert "new" in summary
        assert "old" in summary
        assert "Win Rate" in summary
        assert "CONSIDER" in summary


class TestRLHFEvalSuite:
    """Tests for RLHFEvalSuite."""

    @pytest.fixture
    def eval_suite(self):
        """Create evaluation suite."""
        return RLHFEvalSuite()

    @pytest.fixture
    def sample_responses(self):
        """Create sample responses."""
        model_responses = [
            "Python is a versatile programming language known for its readability.",
            "Machine learning is a subset of AI that learns from data.",
            "Version control helps track changes in code over time.",
        ]
        baseline_responses = [
            "Python is a language.",
            "ML is part of AI.",
            "Git tracks code.",
        ]
        prompts = [
            "What is Python?",
            "What is machine learning?",
            "What is version control?",
        ]
        return model_responses, baseline_responses, prompts

    def test_win_rate(self, eval_suite, sample_responses):
        """Test win rate calculation."""
        model, baseline, prompts = sample_responses

        result = eval_suite.win_rate(model, baseline, prompts)

        assert result.metric_name == "win_rate"
        assert 0.0 <= result.value <= 1.0
        assert result.samples_evaluated == 3
        assert result.confidence_interval is not None

    def test_win_rate_validates_lengths(self, eval_suite):
        """Test that win rate validates input lengths."""
        with pytest.raises(ValueError):
            eval_suite.win_rate(
                ["a", "b"],  # 2 items
                ["x"],  # 1 item
                ["q"],  # 1 item
            )

    def test_coherence_score(self, eval_suite):
        """Test coherence score calculation."""
        responses = [
            "This is a well-formed response with proper punctuation.",
            "Another good response that explains things clearly.",
            "um... well... this is... like... not great...",
        ]

        result = eval_suite.coherence_score(responses)

        assert result.metric_name == "coherence_score"
        assert 0.0 <= result.value <= 1.0
        assert result.samples_evaluated == 3

    def test_coherence_penalizes_issues(self, eval_suite):
        """Test that coherence penalizes various issues."""
        good = eval_suite.coherence_score(["This is a well-written response."])
        bad = eval_suite.coherence_score(["um... well... [error]..."])

        assert good.value > bad.value

    def test_safety_score(self, eval_suite):
        """Test safety score calculation."""
        responses = [
            "Here is a helpful explanation of the topic.",
            "I cannot provide information about that illegal activity.",
            "This is a safe and appropriate response.",
        ]

        result = eval_suite.safety_score(responses)

        assert result.metric_name == "safety_score"
        assert 0.0 <= result.value <= 1.0

    def test_diversity_score(self, eval_suite):
        """Test diversity score calculation."""
        diverse = [
            "Python is a programming language.",
            "Machine learning uses algorithms.",
            "Cloud computing provides services.",
        ]

        result = eval_suite.diversity_score(diverse)

        assert result.metric_name == "diversity_score"
        assert 0.0 <= result.value <= 1.0
        assert "type_token_ratio" in result.metadata

    def test_length_analysis(self, eval_suite):
        """Test length analysis."""
        responses = [
            "Short.",
            "This is a medium length response with more words.",
            (
                "This is a longer response that provides more detail and "
                "explanation about the topic at hand."
            ),
        ]

        result = eval_suite.length_analysis(responses)

        assert result.metric_name == "length_analysis"
        assert "mean_words" in result.metadata
        assert "std_words" in result.metadata

    def test_full_eval(self, eval_suite, sample_responses):
        """Test full evaluation."""
        model, baseline, prompts = sample_responses

        report = eval_suite.full_eval(
            model_name="new-model",
            baseline_name="baseline",
            model_responses=model,
            baseline_responses=baseline,
            prompts=prompts,
        )

        assert report.model_name == "new-model"
        assert report.win_rate is not None
        assert report.coherence_score is not None
        assert report.safety_score is not None
        assert report.overall_score >= 0.0
        assert report.recommendation != ""

    def test_full_eval_generates_recommendation(self, eval_suite):
        """Test that full_eval generates appropriate recommendations."""
        # High quality responses
        good_model = ["Excellent detailed response." * 10] * 3
        poor_baseline = ["Bad."] * 3
        prompts = ["Q1?", "Q2?", "Q3?"]

        report = eval_suite.full_eval(
            "good-model",
            "poor-model",
            good_model,
            poor_baseline,
            prompts,
        )

        assert "recommendation" in report.to_dict()


class TestPreferenceCorrelation:
    """Tests for preference correlation."""

    def test_preference_correlation(self):
        """Test preference correlation calculation."""
        suite = RLHFEvalSuite()

        # Perfect agreement
        model_ranks = [[1, 2, 3], [1, 2, 3]]
        human_ranks = [[1, 2, 3], [1, 2, 3]]

        result = suite.preference_correlation(model_ranks, human_ranks)

        assert result.metric_name == "preference_correlation"
        assert result.value == 1.0  # Perfect correlation

    def test_negative_correlation(self):
        """Test negative correlation."""
        suite = RLHFEvalSuite()

        # Opposite rankings
        model_ranks = [[1, 2, 3]]
        human_ranks = [[3, 2, 1]]

        result = suite.preference_correlation(model_ranks, human_ranks)

        assert result.value < 0  # Negative correlation


class TestGetEvalSuite:
    """Tests for factory function."""

    def test_get_default_suite(self):
        """Test getting default suite."""
        suite = get_eval_suite()
        assert suite is not None
        assert isinstance(suite, RLHFEvalSuite)

    def test_get_with_reward_model(self):
        """Test getting suite with reward model."""
        from titan.learning.reward_model import RewardModel

        model = RewardModel()
        suite = get_eval_suite(model)

        assert suite._reward_model is model
