"""
Titan Costs - Cost-Aware Model Routing

Provides intelligent model selection based on task complexity and budget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from titan.costs.budget import BudgetTracker

logger = logging.getLogger("titan.costs.router")


class ModelTier(str, Enum):
    """Model tiers by capability/cost."""

    ECONOMY = "economy"      # Cheapest, simple tasks
    STANDARD = "standard"    # Balanced price/performance
    PREMIUM = "premium"      # High capability
    FLAGSHIP = "flagship"    # Best available


class TaskComplexity(str, Enum):
    """Complexity levels for tasks."""

    TRIVIAL = "trivial"      # Simple lookups, formatting
    SIMPLE = "simple"        # Basic transformations
    MODERATE = "moderate"    # Standard tasks
    COMPLEX = "complex"      # Multi-step reasoning
    EXPERT = "expert"        # Challenging tasks


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    provider: str
    tier: ModelTier
    input_cost_per_1k: float
    output_cost_per_1k: float
    context_window: int
    quality_rating: float  # 0-1
    speed_rating: float    # 0-1
    supports_tools: bool = True
    supports_vision: bool = False

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens."""
        return (self.input_cost_per_1k + self.output_cost_per_1k) / 2


# Default model catalog
DEFAULT_MODELS: list[ModelInfo] = [
    # Anthropic
    ModelInfo("claude-3-opus-20240229", "anthropic", ModelTier.FLAGSHIP, 0.015, 0.075, 200000, 0.95, 0.6),
    ModelInfo("claude-3-5-sonnet-20241022", "anthropic", ModelTier.PREMIUM, 0.003, 0.015, 200000, 0.90, 0.8),
    ModelInfo("claude-3-haiku-20240307", "anthropic", ModelTier.ECONOMY, 0.00025, 0.00125, 200000, 0.75, 0.95),
    # OpenAI
    ModelInfo("gpt-4-turbo", "openai", ModelTier.PREMIUM, 0.01, 0.03, 128000, 0.88, 0.7),
    ModelInfo("gpt-4o", "openai", ModelTier.PREMIUM, 0.005, 0.015, 128000, 0.87, 0.8),
    ModelInfo("gpt-4o-mini", "openai", ModelTier.STANDARD, 0.00015, 0.0006, 128000, 0.80, 0.9),
    ModelInfo("gpt-3.5-turbo", "openai", ModelTier.ECONOMY, 0.0005, 0.0015, 16000, 0.70, 0.95),
    # Groq
    ModelInfo("llama-3.1-70b-versatile", "groq", ModelTier.STANDARD, 0.0007, 0.0008, 131072, 0.78, 0.98),
    ModelInfo("mixtral-8x7b-32768", "groq", ModelTier.ECONOMY, 0.0005, 0.0005, 32768, 0.72, 0.98),
]


@dataclass
class TaskAnalysis:
    """Analysis of a task for routing decisions."""

    complexity: TaskComplexity
    estimated_input_tokens: int
    estimated_output_tokens: int
    requires_tools: bool
    requires_vision: bool
    requires_long_context: bool
    confidence: float
    reasoning: str


@dataclass
class RoutingDecision:
    """Decision on which model to use."""

    model_id: str
    provider: str
    tier: ModelTier
    estimated_cost: float
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    constraints_applied: list[str] = field(default_factory=list)


class TaskComplexityAnalyzer:
    """
    Analyzes tasks to determine complexity.

    Uses heuristics and patterns to estimate:
    - Task complexity level
    - Required capabilities
    - Token estimates
    """

    def __init__(self) -> None:
        # Complexity indicators
        self._trivial_patterns = [
            r"say\s+(hello|hi|ok)",
            r"(format|convert)\s+this",
            r"what\s+is\s+\d+\s*[+\-*/]\s*\d+",
        ]
        self._simple_patterns = [
            r"(summarize|explain)\s+briefly",
            r"list\s+\d+\s+things",
            r"translate\s+to",
        ]
        self._complex_patterns = [
            r"(analyze|compare|evaluate)",
            r"(implement|write|create)\s+(a|an|the)\s+\w+",
            r"(debug|fix|resolve)",
            r"(design|architect|plan)",
        ]
        self._expert_patterns = [
            r"(optimize|refactor)\s+(complex|large)",
            r"multi[- ]step\s+reasoning",
            r"(security|vulnerability)\s+analysis",
            r"(formal|mathematical)\s+proof",
        ]

    def analyze(self, task: str, context: dict[str, Any] | None = None) -> TaskAnalysis:
        """
        Analyze a task to determine its complexity and requirements.

        Args:
            task: Task description
            context: Optional context (e.g., current budget, history)

        Returns:
            TaskAnalysis with complexity and estimates
        """
        task_lower = task.lower()
        context = context or {}

        # Determine complexity
        complexity = TaskComplexity.MODERATE  # Default
        reasoning_parts = []

        # Check patterns
        for pattern in self._trivial_patterns:
            if re.search(pattern, task_lower):
                complexity = TaskComplexity.TRIVIAL
                reasoning_parts.append("Trivial pattern matched")
                break

        if complexity == TaskComplexity.MODERATE:
            for pattern in self._simple_patterns:
                if re.search(pattern, task_lower):
                    complexity = TaskComplexity.SIMPLE
                    reasoning_parts.append("Simple pattern matched")
                    break

        if complexity == TaskComplexity.MODERATE:
            for pattern in self._complex_patterns:
                if re.search(pattern, task_lower):
                    complexity = TaskComplexity.COMPLEX
                    reasoning_parts.append("Complex pattern matched")
                    break

        for pattern in self._expert_patterns:
            if re.search(pattern, task_lower):
                complexity = TaskComplexity.EXPERT
                reasoning_parts.append("Expert pattern matched")
                break

        # Estimate tokens
        # Rule of thumb: ~4 chars per token for English
        input_tokens = max(100, len(task) // 4)

        # Output estimation based on complexity
        output_estimates = {
            TaskComplexity.TRIVIAL: 50,
            TaskComplexity.SIMPLE: 200,
            TaskComplexity.MODERATE: 500,
            TaskComplexity.COMPLEX: 1500,
            TaskComplexity.EXPERT: 3000,
        }
        output_tokens = output_estimates.get(complexity, 500)

        # Check for special requirements
        requires_tools = any(
            kw in task_lower
            for kw in ["execute", "run", "call", "api", "search", "fetch"]
        )
        requires_vision = any(
            kw in task_lower
            for kw in ["image", "picture", "screenshot", "visual", "diagram"]
        )

        # Check for long context needs
        context_indicators = ["entire", "all of", "full", "complete", "document"]
        requires_long_context = any(kw in task_lower for kw in context_indicators)

        # Adjust for context
        if context.get("history_length", 0) > 10000:
            requires_long_context = True
            reasoning_parts.append("Long conversation history")

        if context.get("code_context", 0) > 5000:
            if complexity.value < TaskComplexity.COMPLEX.value:
                complexity = TaskComplexity.COMPLEX
                reasoning_parts.append("Large code context")

        # Confidence based on pattern matching
        confidence = 0.7 if reasoning_parts else 0.5

        return TaskAnalysis(
            complexity=complexity,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_long_context=requires_long_context,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts) or "Default complexity assessment",
        )


class CostAwareRouter:
    """
    Routes tasks to appropriate models based on complexity and budget.

    Features:
    - Task complexity analysis
    - Budget-aware model selection
    - Quality vs cost tradeoffs
    - Provider preferences
    """

    def __init__(
        self,
        models: list[ModelInfo] | None = None,
        budget_tracker: BudgetTracker | None = None,
        analyzer: TaskComplexityAnalyzer | None = None,
        preferred_providers: list[str] | None = None,
    ) -> None:
        self._models = {m.model_id: m for m in (models or DEFAULT_MODELS)}
        self._budget_tracker = budget_tracker
        self._analyzer = analyzer or TaskComplexityAnalyzer()
        self._preferred_providers = preferred_providers or ["anthropic", "openai"]

    def set_budget_tracker(self, tracker: BudgetTracker) -> None:
        """Set the budget tracker."""
        self._budget_tracker = tracker

    async def select_model(
        self,
        task: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        context: dict[str, Any] | None = None,
        force_tier: ModelTier | None = None,
    ) -> RoutingDecision:
        """
        Select the best model for a task.

        Args:
            task: Task description
            session_id: Session for budget tracking
            agent_id: Agent for budget tracking
            context: Additional context
            force_tier: Force a specific tier

        Returns:
            RoutingDecision with selected model
        """
        # Analyze task
        analysis = self._analyzer.analyze(task, context)

        # Determine target tier
        if force_tier:
            target_tier = force_tier
        else:
            target_tier = self._complexity_to_tier(analysis.complexity)

        constraints = []

        # Get budget if available
        remaining_budget = float("inf")
        if self._budget_tracker and session_id:
            remaining_budget = await self._budget_tracker.get_remaining(session_id)
            if agent_id:
                agent_budget = await self._budget_tracker.get_agent_budget(agent_id)
                if agent_budget:
                    remaining_budget = min(remaining_budget, agent_budget.remaining_usd)

        # Filter models
        candidates = self._filter_candidates(
            target_tier=target_tier,
            requires_tools=analysis.requires_tools,
            requires_vision=analysis.requires_vision,
            requires_long_context=analysis.requires_long_context,
            min_context=analysis.estimated_input_tokens + analysis.estimated_output_tokens,
        )

        if not candidates:
            # Fall back to any available model
            candidates = list(self._models.values())
            constraints.append("Relaxed requirements - no matching models")

        # Sort by preference and cost
        candidates.sort(key=lambda m: (
            self._preferred_providers.index(m.provider)
            if m.provider in self._preferred_providers else 99,
            m.avg_cost_per_1k,
            -m.quality_rating,
        ))

        # Select best model within budget
        selected = None
        alternatives = []

        for model in candidates:
            estimated_cost = (
                (analysis.estimated_input_tokens / 1000 * model.input_cost_per_1k) +
                (analysis.estimated_output_tokens / 1000 * model.output_cost_per_1k)
            )

            if estimated_cost <= remaining_budget:
                if selected is None:
                    selected = model
                else:
                    alternatives.append(model.model_id)
            else:
                constraints.append(f"Budget constraint: ${estimated_cost:.4f} > ${remaining_budget:.4f}")

        # If no model fits budget, pick cheapest
        if selected is None:
            candidates.sort(key=lambda m: m.avg_cost_per_1k)
            selected = candidates[0]
            constraints.append("Selected cheapest model due to budget")

        # Calculate estimated cost
        estimated_cost = (
            (analysis.estimated_input_tokens / 1000 * selected.input_cost_per_1k) +
            (analysis.estimated_output_tokens / 1000 * selected.output_cost_per_1k)
        )

        reasoning = (
            f"Task complexity: {analysis.complexity.value}. "
            f"Selected {selected.tier.value} tier model. "
            f"Estimated cost: ${estimated_cost:.4f}. "
            f"{analysis.reasoning}"
        )

        return RoutingDecision(
            model_id=selected.model_id,
            provider=selected.provider,
            tier=selected.tier,
            estimated_cost=estimated_cost,
            reasoning=reasoning,
            alternatives=alternatives[:3],
            constraints_applied=constraints,
        )

    def _complexity_to_tier(self, complexity: TaskComplexity) -> ModelTier:
        """Map task complexity to model tier."""
        mapping = {
            TaskComplexity.TRIVIAL: ModelTier.ECONOMY,
            TaskComplexity.SIMPLE: ModelTier.ECONOMY,
            TaskComplexity.MODERATE: ModelTier.STANDARD,
            TaskComplexity.COMPLEX: ModelTier.PREMIUM,
            TaskComplexity.EXPERT: ModelTier.FLAGSHIP,
        }
        return mapping.get(complexity, ModelTier.STANDARD)

    def _filter_candidates(
        self,
        target_tier: ModelTier,
        requires_tools: bool,
        requires_vision: bool,
        requires_long_context: bool,
        min_context: int,
    ) -> list[ModelInfo]:
        """Filter models based on requirements."""
        tier_order = [ModelTier.ECONOMY, ModelTier.STANDARD, ModelTier.PREMIUM, ModelTier.FLAGSHIP]
        min_tier_idx = tier_order.index(target_tier)

        candidates = []
        for model in self._models.values():
            # Check tier
            if tier_order.index(model.tier) < min_tier_idx:
                continue

            # Check requirements
            if requires_tools and not model.supports_tools:
                continue
            if requires_vision and not model.supports_vision:
                continue
            if model.context_window < min_context:
                continue

            candidates.append(model)

        return candidates

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get information about a specific model."""
        return self._models.get(model_id)

    def list_models(self, tier: ModelTier | None = None) -> list[ModelInfo]:
        """List available models, optionally filtered by tier."""
        models = list(self._models.values())
        if tier:
            models = [m for m in models if m.tier == tier]
        return sorted(models, key=lambda m: m.avg_cost_per_1k)


# Singleton instance
_default_router: CostAwareRouter | None = None


def get_cost_aware_router() -> CostAwareRouter:
    """Get the default cost-aware router."""
    global _default_router
    if _default_router is None:
        _default_router = CostAwareRouter()
    return _default_router
