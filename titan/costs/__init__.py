"""
Titan Costs - Budget Management and Cost-Aware Routing

Provides:
- Budget tracking and enforcement
- Cost-aware model selection
- Task complexity analysis
- CFO agent integration
"""

from titan.costs.budget import (
    Budget,
    BudgetAllocation,
    BudgetConfig,
    BudgetExceededError,
    BudgetTracker,
    get_budget_tracker,
)
from titan.costs.router import (
    CostAwareRouter,
    ModelTier,
    TaskComplexity,
    TaskComplexityAnalyzer,
    get_cost_aware_router,
)

__all__ = [
    # Budget
    "Budget",
    "BudgetConfig",
    "BudgetTracker",
    "BudgetAllocation",
    "BudgetExceededError",
    "get_budget_tracker",
    # Router
    "CostAwareRouter",
    "ModelTier",
    "TaskComplexity",
    "TaskComplexityAnalyzer",
    "get_cost_aware_router",
]
