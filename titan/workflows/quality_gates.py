"""
Titan Workflows - Quality Gates

Automated quality checks and validation logic for inquiry workflows.
Includes the Dialectic AI gate for contradiction detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

from titan.workflows.inquiry_engine import InquirySession, StageResult
from titan.metrics import get_metrics

logger = logging.getLogger("titan.workflows.quality_gates")

@dataclass
class QualityGateResult:
    passed: bool
    score: float
    issues: List[str]
    metadata: dict[str, Any]

class DialecticGate:
    """
    Dialectic AI Quality Gate.
    
    Analyzes inquiry stages for logical contradictions, tensions, and
    inconsistencies between different cognitive perspectives.
    """
    
    def __init__(self, llm_caller=None, default_model="claude-3-5-sonnet-20241022"):
        self.llm_caller = llm_caller
        self.default_model = default_model
        
    async def evaluate(self, session: InquirySession) -> QualityGateResult:
        """
        Evaluate the session for dialectic friction.
        
        Args:
            session: The inquiry session to evaluate.
            
        Returns:
            QualityGateResult with findings.
        """
        if len(session.results) < 2:
            return QualityGateResult(True, 1.0, [], {"reason": "Insufficient stages for dialectic analysis"})
            
        # Prepare context for analysis
        stages_content = []
        for r in session.results:
            stages_content.append(f"Stage: {r.stage_name} ({r.role})\nContent: {r.content[:1000]}...") # Truncate for prompt
            
        prompt = f"""
You are a Dialectic AI specialized in detecting contradictions and logical tensions.
Analyze the following inquiry stages for internal contradictions, divergent perspectives, or logical friction.

Context:
{chr(10).join(stages_content)}

Task:
1. Identify any direct contradictions between stages.
2. Highlight productive tensions (where perspectives disagree in a useful way).
3. Flag any logical fallacies or inconsistencies.

Return a JSON object with:
- "contradictions": list of strings
- "tensions": list of strings
- "friction_score": float 0.0 to 1.0 (0 = coherent, 1 = chaotic)
"""
        
        try:
            # Simulate LLM call if not provided (or implement actual call)
            if self.llm_caller:
                response = await self.llm_caller(prompt, self.default_model)
                # Parse JSON from response (mock logic here for safety)
                # In production, use robust JSON parsing
                import json
                try:
                    data = json.loads(response)
                except:
                    # Fallback if LLM returns text
                    data = {"contradictions": [], "tensions": [], "friction_score": 0.0}
            else:
                # Mock logic for testing/demonstration
                data = self._mock_analysis(session)
                
            friction_score = data.get("friction_score", 0.0)
            
            # Record metric
            if friction_score > 0.3:
                get_metrics().record_dialectic_friction(session.id)
                
            return QualityGateResult(
                passed=friction_score < 0.8, # Fail if too chaotic
                score=1.0 - friction_score,
                issues=data.get("contradictions", []) + data.get("tensions", []),
                metadata=data
            )
            
        except Exception as e:
            logger.error(f"Dialectic gate failed: {e}")
            return QualityGateResult(False, 0.0, [str(e)], {})

    def _mock_analysis(self, session: InquirySession) -> dict[str, Any]:
        """Simple keyword-based mock analysis."""
        text = " ".join([r.content.lower() for r in session.results])
        contradictions = []
        if "however" in text and "but" in text:
             contradictions.append("Potential tension detected via linguistic markers.")
             
        return {
            "contradictions": contradictions,
            "tensions": ["Perspective divergence between Logic and Mythos"],
            "friction_score": 0.4 if contradictions else 0.1
        }
