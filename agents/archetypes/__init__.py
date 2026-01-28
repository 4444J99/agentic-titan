"""
Agent Archetypes - Pre-built agent types for common tasks.

Available archetypes:
- Orchestrator: Coordinates multi-agent workflows
- Researcher: Gathers and analyzes information
- Coder: Writes and tests code
- Reviewer: Reviews work for quality
- Paper2Code: Converts research papers to implementations
- CFO: Budget management and cost optimization
- DevOps: Infrastructure and deployment automation
- SecurityAnalyst: Code security and compliance
- DataEngineer: ETL, data quality, and schema management
- ProductManager: Requirements analysis and roadmap planning
"""

from agents.archetypes.orchestrator import OrchestratorAgent
from agents.archetypes.researcher import ResearcherAgent
from agents.archetypes.coder import CoderAgent
from agents.archetypes.reviewer import ReviewerAgent
from agents.archetypes.paper2code import Paper2CodeAgent
from agents.archetypes.cfo import CFOAgent
from agents.archetypes.devops import DevOpsAgent
from agents.archetypes.security_analyst import SecurityAnalystAgent
from agents.archetypes.data_engineer import DataEngineerAgent
from agents.archetypes.product_manager import ProductManagerAgent

__all__ = [
    "OrchestratorAgent",
    "ResearcherAgent",
    "CoderAgent",
    "ReviewerAgent",
    "Paper2CodeAgent",
    "CFOAgent",
    "DevOpsAgent",
    "SecurityAnalystAgent",
    "DataEngineerAgent",
    "ProductManagerAgent",
]
