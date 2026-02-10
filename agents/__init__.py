"""
Agentic Titan - Agent Framework

This module provides the core agent abstractions and archetypes for building
multi-agent systems that can self-organize into different topologies.
"""

from agents.framework.base_agent import AgentContext, AgentState, BaseAgent
from agents.framework.errors import (
    AgentError,
    HiveMindError,
    LLMAdapterError,
    TitanError,
    TopologyError,
)
from agents.personas import CODER, ORCHESTRATOR, RESEARCHER, REVIEWER, Persona, say

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentContext",
    "AgentState",
    # Errors
    "TitanError",
    "AgentError",
    "HiveMindError",
    "TopologyError",
    "LLMAdapterError",
    # Personas
    "Persona",
    "ORCHESTRATOR",
    "RESEARCHER",
    "CODER",
    "REVIEWER",
    "say",
]
