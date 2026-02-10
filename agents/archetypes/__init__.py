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

Governance Archetypes:
- JuryAgent: Deliberative body with evidence evaluation and voting
- ExecutiveAgent: Implements decisions and leads execution
- LegislativeAgent: Proposes and debates policies
- JudicialAgent: Reviews compliance and resolves disputes
- BureaucracyAgent: Rule-based processing with specialized roles

Biological Archetypes:
- EusocialColonyAgent: Superorganism with castes and stigmergy
- CellAgent: Multicellular patterns with apoptosis and signaling

Philosophical Archetypes:
- AssemblageAgent: Heterogeneous assembly with territorialization
- ActorNetworkAgent: ANT-based actant enrollment and translation

Digital Archetypes:
- SwarmIntelligenceAgent: PSO/ACO optimization algorithms
- DAOAgent: Decentralized governance with proposals and voting
"""

from agents.archetypes.actor_network import ActorNetworkAgent

# Philosophical archetypes
from agents.archetypes.assemblage import AssemblageAgent
from agents.archetypes.bureaucracy import BureaucracyAgent
from agents.archetypes.cell import CellAgent
from agents.archetypes.cfo import CFOAgent
from agents.archetypes.coder import CoderAgent
from agents.archetypes.dao import DAOAgent
from agents.archetypes.data_engineer import DataEngineerAgent
from agents.archetypes.devops import DevOpsAgent

# Biological archetypes
from agents.archetypes.eusocial import EusocialColonyAgent
from agents.archetypes.government import (
    ExecutiveAgent,
    JudicialAgent,
    LegislativeAgent,
)

# Governance archetypes
from agents.archetypes.jury import JuryAgent
from agents.archetypes.orchestrator import OrchestratorAgent
from agents.archetypes.paper2code import Paper2CodeAgent
from agents.archetypes.product_manager import ProductManagerAgent
from agents.archetypes.researcher import ResearcherAgent
from agents.archetypes.reviewer import ReviewerAgent
from agents.archetypes.security_analyst import SecurityAnalystAgent

# Digital archetypes
from agents.archetypes.swarm_intelligence import SwarmIntelligenceAgent

__all__ = [
    # Core archetypes
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
    # Governance archetypes
    "JuryAgent",
    "ExecutiveAgent",
    "LegislativeAgent",
    "JudicialAgent",
    "BureaucracyAgent",
    # Biological archetypes
    "EusocialColonyAgent",
    "CellAgent",
    # Philosophical archetypes
    "AssemblageAgent",
    "ActorNetworkAgent",
    # Digital archetypes
    "SwarmIntelligenceAgent",
    "DAOAgent",
]
