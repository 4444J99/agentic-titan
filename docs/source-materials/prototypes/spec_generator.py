"""
Titan Workflows - Spec Generator

Automates the creation of agent specification files from conceptual descriptions.
Implements the 'Grounding Loop' by bridging abstract inquiry and concrete code.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from titan.spec import AgentSpecModel, AgentMetadata, AgentSpecInner, LLMPreference, PersonalitySpec, MemorySpec, RuntimeSpec, ToolSpec

logger = logging.getLogger("titan.workflows.spec_generator")

class SpecGenerator:
    """
    Generates Agentic Titan specification files.
    
    Can be used by Meta AI to ground new agent concepts into executable files.
    """
    
    def __init__(self, output_dir: str = "specs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def generate_from_concept(self, concept: Dict[str, Any]) -> str:
        """
        Generate a .titan.yaml file from a concept dictionary.
        
        Args:
            concept: Dictionary containing agent metadata and specs.
            
        Returns:
            Path to the generated file.
        """
        name = concept.get("name", "unnamed-agent")
        filename = f"{name.lower().replace(' ', '-')}.titan.yaml"
        file_path = self.output_dir / filename
        
        # Build the model
        model = AgentSpecModel(
            apiVersion="titan/v1",
            kind="Agent",
            metadata=AgentMetadata(
                name=name,
                labels=concept.get("labels", {}),
                annotations=concept.get("annotations", {"description": "Auto-generated agent"})
            ),
            spec=AgentSpecInner(
                capabilities=concept.get("capabilities", ["summarization"]),
                personality=PersonalitySpec(
                    traits=concept.get("traits", ["neutral"]),
                    communication_style=concept.get("communication_style", "neutral")
                ),
                llm=LLMPreference(
                    preferred=concept.get("preferred_model", "claude-sonnet"),
                    min_context=concept.get("min_context", 16000)
                ),
                tools=[ToolSpec(**t) for t in concept.get("tools", [])],
                memory=MemorySpec(
                    short_term=concept.get("short_term_memory", 10),
                    long_term=concept.get("long_term_memory", "hive_mind")
                ),
                runtimes=RuntimeSpec(**concept.get("runtimes", {})),
                systemPrompt=concept.get("system_prompt", f"You are a {name} agent."),
                maxTurns=concept.get("max_turns", 20),
                timeoutMs=concept.get("timeout_ms", 300000)
            )
        )
        
        # Serialize to YAML
        yaml_data = model.model_dump(by_alias=True, exclude_none=True)
        
        with open(file_path, "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
            
        logger.info(f"Generated spec file: {file_path}")
        return str(file_path)

    async def grounding_loop(self, session_id: str, concept_name: str, inquiry_results: str) -> str:
        """
        Implementation of the Grounding Loop.
        Uses inquiry results to derive a concrete agent spec.
        """
        # In a real implementation, this would call an LLM to parse the inquiry_results
        # and extract the fields for generate_from_concept.
        # For now, we'll implement a heuristic-based extraction or a placeholder.
        
        logger.info(f"Grounding concept '{concept_name}' from session {session_id}")
        
        # Mock extracted concept
        concept = {
            "name": concept_name,
            "capabilities": ["planning", "research"],
            "traits": ["recursive", "analytical"],
            "system_prompt": f"Extracted from inquiry {session_id}: {inquiry_results[:200]}..."
        }
        
        return await self.generate_from_concept(concept)
