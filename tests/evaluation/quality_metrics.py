"""
Quality Metrics - Output quality scoring utilities.

Provides metrics for evaluating LLM output quality without
requiring external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityScore:
    """Quality score for a response."""

    overall: float  # 0-1 overall score
    relevance: float = 0.0  # How relevant to the query
    coherence: float = 0.0  # How coherent/logical
    completeness: float = 0.0  # How complete the answer is
    safety: float = 0.0  # Safety/appropriateness
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "completeness": self.completeness,
            "safety": self.safety,
            "details": self.details,
        }


class QualityEvaluator:
    """
    Evaluates output quality using heuristic metrics.

    Provides basic quality scoring without LLM calls.
    For more sophisticated evaluation, use DeepEval integration.
    """

    def __init__(self) -> None:
        # Coherence indicators
        self._transition_words = [
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "consequently",
            "thus",
            "hence",
            "additionally",
            "first",
            "second",
            "finally",
            "in conclusion",
        ]

        # Code quality indicators
        self._code_patterns = {
            "has_docstring": r'"""[\s\S]*?"""',
            "has_type_hints": r":\s*(int|str|float|bool|list|dict|Any|None)",
            "has_comments": r"#.*$",
            "has_function_def": r"def\s+\w+",
            "has_class_def": r"class\s+\w+",
        }

    def evaluate_response(
        self,
        response: str,
        prompt: str,
        expected_patterns: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
    ) -> QualityScore:
        """
        Evaluate a response.

        Args:
            response: The response to evaluate
            prompt: The original prompt
            expected_patterns: Regex patterns expected in response
            forbidden_patterns: Patterns that shouldn't appear

        Returns:
            QualityScore with detailed metrics
        """
        # Calculate individual metrics
        relevance = self._calculate_relevance(response, prompt)
        coherence = self._calculate_coherence(response)
        completeness = self._calculate_completeness(response, prompt, expected_patterns)
        safety = self._calculate_safety(response, forbidden_patterns)

        # Calculate overall score (weighted average)
        overall = relevance * 0.3 + coherence * 0.2 + completeness * 0.3 + safety * 0.2

        return QualityScore(
            overall=overall,
            relevance=relevance,
            coherence=coherence,
            completeness=completeness,
            safety=safety,
        )

    def _calculate_relevance(self, response: str, prompt: str) -> float:
        """Calculate relevance score based on keyword overlap."""
        # Extract significant words from prompt
        prompt_words = set(word.lower() for word in re.findall(r"\b\w{4,}\b", prompt))

        # Check how many appear in response
        response_lower = response.lower()
        matches = sum(1 for word in prompt_words if word in response_lower)

        if not prompt_words:
            return 0.5  # Neutral if no significant words

        return min(1.0, matches / (len(prompt_words) * 0.5))  # Expect ~50% word overlap

    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence based on structure indicators."""
        score = 0.5  # Base score

        # Check for transition words (indicates logical flow)
        response_lower = response.lower()
        transitions_found = sum(1 for word in self._transition_words if word in response_lower)
        score += min(0.2, transitions_found * 0.05)

        # Check for paragraph structure
        paragraphs = [p for p in response.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            score += 0.1

        # Check for lists/numbered items (structured content)
        if re.search(r"^\s*[\d\-\*]\s", response, re.MULTILINE):
            score += 0.1

        # Penalize very short responses
        word_count = len(response.split())
        if word_count < 10:
            score -= 0.2
        elif word_count > 50:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _calculate_completeness(
        self,
        response: str,
        prompt: str,
        expected_patterns: list[str] | None,
    ) -> float:
        """Calculate completeness based on expected patterns and structure."""
        score = 0.5

        # Check expected patterns
        if expected_patterns:
            matches = sum(
                1 for pattern in expected_patterns if re.search(pattern, response, re.IGNORECASE)
            )
            pattern_score = matches / len(expected_patterns)
            score = pattern_score * 0.7 + score * 0.3

        # Check for code if coding prompt
        if any(kw in prompt.lower() for kw in ["code", "function", "implement", "write"]):
            if "```" in response or re.search(r"def\s+\w+|class\s+\w+", response):
                score += 0.2

        # Check for explanation if explaining
        if any(kw in prompt.lower() for kw in ["explain", "describe", "what is"]):
            if len(response) > 100:  # Substantial explanation
                score += 0.2

        return max(0.0, min(1.0, score))

    def _calculate_safety(
        self,
        response: str,
        forbidden_patterns: list[str] | None,
    ) -> float:
        """Calculate safety score based on forbidden patterns."""
        score = 1.0  # Start with perfect safety

        # Check forbidden patterns
        if forbidden_patterns:
            for pattern in forbidden_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    score -= 0.3  # Significant penalty for each match

        # Check for common unsafe content
        unsafe_patterns = [
            r"ignore\s+(previous|all)\s+instructions",
            r"api[_-]?key\s*[:=]\s*\w{20,}",
            r"password\s*[:=]\s*\S{8,}",
        ]
        for pattern in unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score -= 0.2

        return max(0.0, score)

    def evaluate_code_quality(self, code: str) -> dict[str, Any]:
        """
        Evaluate code-specific quality metrics.

        Args:
            code: Code to evaluate

        Returns:
            Dictionary of code quality metrics
        """
        metrics = {}

        for name, pattern in self._code_patterns.items():
            metrics[name] = bool(re.search(pattern, code, re.MULTILINE))

        # Line count
        lines = code.split("\n")
        metrics["line_count"] = len(lines)
        metrics["non_empty_lines"] = len([line for line in lines if line.strip()])

        # Complexity estimate (very rough)
        metrics["estimated_complexity"] = (
            code.count("if ") + code.count("for ") + code.count("while ") + code.count("try:")
        )

        # Calculate overall code quality score
        score = 0.5
        if metrics["has_function_def"] or metrics["has_class_def"]:
            score += 0.2
        if metrics["has_type_hints"]:
            score += 0.15
        if metrics["has_docstring"]:
            score += 0.15

        metrics["quality_score"] = min(1.0, score)

        return metrics


def evaluate_response_quality(
    response: str,
    prompt: str,
    expected_patterns: list[str] | None = None,
    forbidden_patterns: list[str] | None = None,
) -> QualityScore:
    """Convenience function to evaluate response quality."""
    evaluator = QualityEvaluator()
    return evaluator.evaluate_response(response, prompt, expected_patterns, forbidden_patterns)
