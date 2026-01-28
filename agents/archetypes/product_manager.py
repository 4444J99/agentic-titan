"""
Product Manager Agent - Requirements analysis and roadmap planning specialist.

Capabilities:
- Requirements analysis
- User story generation
- Feature prioritization
- Roadmap planning
- Stakeholder communication
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentState
from adapters.base import LLMMessage
from adapters.router import get_router

logger = logging.getLogger("titan.agents.product_manager")


class Priority(Enum):
    """Feature priority levels."""
    CRITICAL = "critical"  # P0 - Must have
    HIGH = "high"  # P1 - Should have
    MEDIUM = "medium"  # P2 - Nice to have
    LOW = "low"  # P3 - Future consideration


class StoryType(Enum):
    """Types of user stories."""
    FEATURE = "feature"
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    TECHNICAL = "technical"
    SPIKE = "spike"  # Research/exploration


class StoryStatus(Enum):
    """User story status."""
    DRAFT = "draft"
    REFINED = "refined"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class RoadmapQuarter(Enum):
    """Roadmap time horizons."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    NOW = "now"  # Current sprint
    NEXT = "next"  # Next sprint
    LATER = "later"  # Backlog


@dataclass
class UserStory:
    """A user story with acceptance criteria."""
    id: str
    title: str
    story_type: StoryType
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    story_points: int = 0
    status: StoryStatus = StoryStatus.DRAFT
    labels: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    persona: str = ""  # User persona this story serves


@dataclass
class Requirement:
    """A product requirement."""
    id: str
    title: str
    description: str
    rationale: str = ""
    priority: Priority = Priority.MEDIUM
    user_stories: list[str] = field(default_factory=list)  # Story IDs
    stakeholders: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)


@dataclass
class Feature:
    """A product feature for roadmap planning."""
    id: str
    name: str
    description: str
    priority: Priority = Priority.MEDIUM
    effort_estimate: str = ""  # S, M, L, XL
    business_value: int = 0  # 1-10 scale
    technical_risk: int = 0  # 1-10 scale
    quarter: RoadmapQuarter = RoadmapQuarter.LATER
    dependencies: list[str] = field(default_factory=list)
    user_stories: list[str] = field(default_factory=list)
    status: str = "planned"  # planned, in_progress, shipped


@dataclass
class Roadmap:
    """Product roadmap."""
    name: str
    vision: str
    features: dict[RoadmapQuarter, list[Feature]] = field(default_factory=dict)
    themes: list[str] = field(default_factory=list)
    milestones: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PRD:
    """Product Requirements Document."""
    title: str
    version: str = "1.0"
    overview: str = ""
    problem_statement: str = ""
    goals: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    user_personas: list[dict[str, str]] = field(default_factory=list)
    success_metrics: list[str] = field(default_factory=list)
    timeline: str = ""
    risks: list[str] = field(default_factory=list)


class ProductManagerAgent(BaseAgent):
    """
    Agent specialized in product management tasks.

    Capabilities:
    - Analyze requirements from stakeholder input
    - Generate user stories with acceptance criteria
    - Prioritize features using frameworks (RICE, MoSCoW)
    - Create roadmaps
    - Generate PRDs
    """

    def __init__(
        self,
        product_name: str = "Product",
        prioritization_framework: str = "RICE",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "product_manager")
        kwargs.setdefault("capabilities", [
            "requirements_analysis",
            "user_story_generation",
            "feature_prioritization",
            "roadmap_planning",
            "prd_generation",
        ])
        super().__init__(**kwargs)

        self.product_name = product_name
        self.prioritization_framework = prioritization_framework  # RICE, MoSCoW, WSJF
        self._router = get_router()
        self._stories: list[UserStory] = []
        self._requirements: list[Requirement] = []
        self._features: list[Feature] = []
        self._story_counter = 0

    async def initialize(self) -> None:
        """Initialize the Product Manager agent."""
        logger.info(f"Product Manager Agent '{self.name}' initializing")
        await self._router.initialize()

        if self._hive_mind:
            previous = await self.recall(f"product {self.product_name} requirements", k=5)
            if previous:
                logger.info(f"Found {len(previous)} previous product artifacts")

    async def work(self) -> dict[str, Any]:
        """
        Main work loop - PM agent is typically invoked for specific tasks.

        Returns:
            Summary of capabilities and current state
        """
        logger.info("Product Manager Agent ready for product tasks")

        return {
            "status": "ready",
            "product": self.product_name,
            "stories_created": len(self._stories),
            "requirements_analyzed": len(self._requirements),
            "features_planned": len(self._features),
        }

    async def shutdown(self) -> None:
        """Cleanup Product Manager agent."""
        logger.info("Product Manager Agent shutting down")

        if self._hive_mind and self._stories:
            summary = f"User stories created: {len(self._stories)}"
            if self._features:
                summary += f"\nFeatures planned: {len(self._features)}"
            await self.remember(
                content=summary,
                importance=0.8,
                tags=["product", "stories", self.product_name.lower()],
            )

    async def analyze_requirements(
        self,
        stakeholder_input: str,
        context: str = "",
    ) -> list[Requirement]:
        """
        Analyze stakeholder input to extract requirements.

        Args:
            stakeholder_input: Raw input from stakeholders
            context: Additional context about the product

        Returns:
            List of extracted requirements
        """
        self.increment_turn()

        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze this stakeholder input and extract product requirements:

INPUT:
{stakeholder_input}

CONTEXT:
{context or f'Product: {self.product_name}'}

For each requirement, provide:
REQ_ID: REQ-XXX
TITLE: Short title
DESCRIPTION: Detailed description
RATIONALE: Why this is needed
PRIORITY: critical/high/medium/low
STAKEHOLDERS: Who cares about this
SUCCESS_METRICS: How to measure success
CONSTRAINTS: Any limitations""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a product manager. Extract clear, actionable requirements.",
            max_tokens=1000,
        )

        requirements = self._parse_requirements(response.content)
        self._requirements.extend(requirements)

        await self.log_decision(
            decision=f"Analyzed requirements: {len(requirements)} extracted",
            category="requirements_analysis",
            rationale=f"From stakeholder input: {stakeholder_input[:100]}...",
            tags=["product", "requirements"],
        )

        return requirements

    def _parse_requirements(self, content: str) -> list[Requirement]:
        """Parse requirements from LLM response."""
        requirements = []
        current: dict[str, Any] = {}

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("REQ_ID:"):
                if current.get("id"):
                    requirements.append(self._create_requirement(current))
                current = {"id": line.replace("REQ_ID:", "").strip()}
            elif line.startswith("TITLE:") and current:
                current["title"] = line.replace("TITLE:", "").strip()
            elif line.startswith("DESCRIPTION:") and current:
                current["description"] = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("RATIONALE:") and current:
                current["rationale"] = line.replace("RATIONALE:", "").strip()
            elif line.startswith("PRIORITY:") and current:
                current["priority"] = line.replace("PRIORITY:", "").strip().lower()
            elif line.startswith("STAKEHOLDERS:") and current:
                stakeholders = line.replace("STAKEHOLDERS:", "").strip()
                current["stakeholders"] = [s.strip() for s in stakeholders.split(",")]
            elif line.startswith("SUCCESS_METRICS:") and current:
                metrics = line.replace("SUCCESS_METRICS:", "").strip()
                current["metrics"] = [m.strip() for m in metrics.split(",")]
            elif line.startswith("CONSTRAINTS:") and current:
                constraints = line.replace("CONSTRAINTS:", "").strip()
                current["constraints"] = [c.strip() for c in constraints.split(",")]

        if current.get("id"):
            requirements.append(self._create_requirement(current))

        return requirements

    def _create_requirement(self, data: dict[str, Any]) -> Requirement:
        """Create Requirement from parsed data."""
        priority_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW,
        }

        return Requirement(
            id=data.get("id", f"REQ-{len(self._requirements) + 1}"),
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
            priority=priority_map.get(data.get("priority", "medium"), Priority.MEDIUM),
            stakeholders=data.get("stakeholders", []),
            success_metrics=data.get("metrics", []),
            constraints=data.get("constraints", []),
        )

    async def generate_user_stories(
        self,
        requirement: Requirement | str,
        persona: str = "user",
    ) -> list[UserStory]:
        """
        Generate user stories from a requirement.

        Args:
            requirement: Requirement object or description
            persona: User persona for the stories

        Returns:
            List of generated user stories
        """
        self.increment_turn()

        req_text = requirement if isinstance(requirement, str) else (
            f"{requirement.title}\n{requirement.description}"
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate user stories for this requirement:

REQUIREMENT:
{req_text}

PERSONA: {persona}

For each story, provide:
STORY_TYPE: feature/bug/enhancement/technical/spike
TITLE: As a [persona], I want [goal] so that [benefit]
DESCRIPTION: Detailed description
ACCEPTANCE_CRITERIA:
- Given/When/Then criteria
STORY_POINTS: 1/2/3/5/8/13
LABELS: comma,separated,labels
DEPENDENCIES: any dependencies

Generate 2-5 user stories.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a product manager. Write clear, testable user stories.",
            max_tokens=1000,
        )

        stories = self._parse_stories(response.content, persona)

        # Link to requirement if provided
        if isinstance(requirement, Requirement):
            for story in stories:
                requirement.user_stories.append(story.id)

        self._stories.extend(stories)

        await self.log_decision(
            decision=f"Generated {len(stories)} user stories",
            category="story_generation",
            rationale=f"For persona: {persona}",
            tags=["product", "stories", persona],
        )

        return stories

    def _parse_stories(self, content: str, persona: str) -> list[UserStory]:
        """Parse user stories from LLM response."""
        stories = []
        current: dict[str, Any] = {}

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("STORY_TYPE:"):
                if current.get("title"):
                    stories.append(self._create_story(current, persona))
                story_type = line.replace("STORY_TYPE:", "").strip().lower()
                current = {"story_type": story_type}
            elif line.startswith("TITLE:") and current:
                current["title"] = line.replace("TITLE:", "").strip()
            elif line.startswith("DESCRIPTION:") and current:
                current["description"] = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("ACCEPTANCE_CRITERIA:"):
                current.setdefault("criteria", [])
            elif line.startswith("- ") and "criteria" in current:
                current["criteria"].append(line[2:].strip())
            elif line.startswith("STORY_POINTS:") and current:
                try:
                    current["points"] = int(line.replace("STORY_POINTS:", "").strip())
                except ValueError:
                    current["points"] = 3
            elif line.startswith("LABELS:") and current:
                labels = line.replace("LABELS:", "").strip()
                current["labels"] = [l.strip() for l in labels.split(",")]
            elif line.startswith("DEPENDENCIES:") and current:
                deps = line.replace("DEPENDENCIES:", "").strip()
                if deps.lower() != "none":
                    current["dependencies"] = [d.strip() for d in deps.split(",")]

        if current.get("title"):
            stories.append(self._create_story(current, persona))

        return stories

    def _create_story(self, data: dict[str, Any], persona: str) -> UserStory:
        """Create UserStory from parsed data."""
        self._story_counter += 1

        type_map = {
            "feature": StoryType.FEATURE,
            "bug": StoryType.BUG,
            "enhancement": StoryType.ENHANCEMENT,
            "technical": StoryType.TECHNICAL,
            "spike": StoryType.SPIKE,
        }

        return UserStory(
            id=f"STORY-{self._story_counter}",
            title=data.get("title", "Untitled Story"),
            story_type=type_map.get(data.get("story_type", "feature"), StoryType.FEATURE),
            description=data.get("description", ""),
            acceptance_criteria=data.get("criteria", []),
            priority=Priority.MEDIUM,
            story_points=data.get("points", 3),
            labels=data.get("labels", []),
            dependencies=data.get("dependencies", []),
            persona=persona,
        )

    async def prioritize_features(
        self,
        features: list[Feature] | list[str],
    ) -> list[Feature]:
        """
        Prioritize features using the configured framework.

        Args:
            features: List of features or feature descriptions

        Returns:
            Prioritized list of features
        """
        self.increment_turn()

        # Convert strings to features if needed
        feature_objs = []
        for i, f in enumerate(features):
            if isinstance(f, str):
                feature_objs.append(Feature(
                    id=f"FEAT-{i+1}",
                    name=f,
                    description=f,
                ))
            else:
                feature_objs.append(f)

        features_text = "\n".join(
            f"- {f.name}: {f.description[:100]}" for f in feature_objs
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Prioritize these features using {self.prioritization_framework}:

FEATURES:
{features_text}

Using {self.prioritization_framework} framework, provide for each:
FEATURE: feature_name
PRIORITY: critical/high/medium/low
EFFORT: S/M/L/XL
BUSINESS_VALUE: 1-10
RISK: 1-10
RATIONALE: Why this priority
QUARTER: now/next/Q1/Q2/Q3/Q4/later""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a product manager using {self.prioritization_framework} prioritization.",
            max_tokens=1000,
        )

        return self._apply_prioritization(feature_objs, response.content)

    def _apply_prioritization(
        self,
        features: list[Feature],
        content: str,
    ) -> list[Feature]:
        """Apply prioritization from LLM response."""
        priority_map = {}
        current: dict[str, Any] = {}

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("FEATURE:"):
                if current.get("name"):
                    priority_map[current["name"].lower()] = current
                current = {"name": line.replace("FEATURE:", "").strip()}
            elif line.startswith("PRIORITY:") and current:
                current["priority"] = line.replace("PRIORITY:", "").strip().lower()
            elif line.startswith("EFFORT:") and current:
                current["effort"] = line.replace("EFFORT:", "").strip().upper()
            elif line.startswith("BUSINESS_VALUE:") and current:
                try:
                    current["value"] = int(line.replace("BUSINESS_VALUE:", "").strip())
                except ValueError:
                    current["value"] = 5
            elif line.startswith("RISK:") and current:
                try:
                    current["risk"] = int(line.replace("RISK:", "").strip())
                except ValueError:
                    current["risk"] = 5
            elif line.startswith("QUARTER:") and current:
                current["quarter"] = line.replace("QUARTER:", "").strip().lower()

        if current.get("name"):
            priority_map[current["name"].lower()] = current

        # Apply to features
        prio_enum_map = {
            "critical": Priority.CRITICAL,
            "high": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "low": Priority.LOW,
        }
        quarter_map = {
            "now": RoadmapQuarter.NOW,
            "next": RoadmapQuarter.NEXT,
            "q1": RoadmapQuarter.Q1,
            "q2": RoadmapQuarter.Q2,
            "q3": RoadmapQuarter.Q3,
            "q4": RoadmapQuarter.Q4,
            "later": RoadmapQuarter.LATER,
        }

        for f in features:
            prio_data = priority_map.get(f.name.lower(), {})
            if prio_data:
                f.priority = prio_enum_map.get(prio_data.get("priority", "medium"), Priority.MEDIUM)
                f.effort_estimate = prio_data.get("effort", "M")
                f.business_value = prio_data.get("value", 5)
                f.technical_risk = prio_data.get("risk", 5)
                f.quarter = quarter_map.get(prio_data.get("quarter", "later"), RoadmapQuarter.LATER)

        # Sort by priority then business value
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}
        features.sort(key=lambda x: (priority_order.get(x.priority, 2), -x.business_value))

        self._features.extend(features)

        return features

    async def create_roadmap(
        self,
        vision: str,
        themes: list[str] | None = None,
        horizon: str = "year",
    ) -> Roadmap:
        """
        Create a product roadmap.

        Args:
            vision: Product vision statement
            themes: Strategic themes
            horizon: Planning horizon (quarter, half, year)

        Returns:
            Roadmap with features organized by quarter
        """
        self.increment_turn()

        # Group features by quarter
        roadmap = Roadmap(
            name=f"{self.product_name} Roadmap",
            vision=vision,
            themes=themes or [],
        )

        for quarter in RoadmapQuarter:
            roadmap.features[quarter] = [f for f in self._features if f.quarter == quarter]

        # Generate milestones
        messages = [
            LLMMessage(
                role="user",
                content=f"""Create milestones for this roadmap:

PRODUCT: {self.product_name}
VISION: {vision}
THEMES: {', '.join(themes or ['growth'])}
HORIZON: {horizon}

CURRENT FEATURES:
{chr(10).join(f'- {f.name} ({f.quarter.value})' for f in self._features[:20])}

Provide 3-5 milestones:
MILESTONE: Name
DATE: Approximate date or quarter
FEATURES: Key features included
SUCCESS_CRITERIA: How to measure""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a product strategist creating a roadmap.",
            max_tokens=600,
        )

        roadmap.milestones = self._parse_milestones(response.content)

        return roadmap

    def _parse_milestones(self, content: str) -> list[dict[str, Any]]:
        """Parse milestones from LLM response."""
        milestones = []
        current: dict[str, Any] = {}

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("MILESTONE:"):
                if current:
                    milestones.append(current)
                current = {"name": line.replace("MILESTONE:", "").strip()}
            elif line.startswith("DATE:") and current:
                current["date"] = line.replace("DATE:", "").strip()
            elif line.startswith("FEATURES:") and current:
                current["features"] = line.replace("FEATURES:", "").strip()
            elif line.startswith("SUCCESS_CRITERIA:") and current:
                current["criteria"] = line.replace("SUCCESS_CRITERIA:", "").strip()

        if current:
            milestones.append(current)

        return milestones

    async def generate_prd(
        self,
        feature_description: str,
        context: str = "",
    ) -> PRD:
        """
        Generate a Product Requirements Document.

        Args:
            feature_description: Description of the feature/product
            context: Additional context

        Returns:
            Complete PRD
        """
        self.increment_turn()

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate a PRD for:

FEATURE: {feature_description}
CONTEXT: {context or self.product_name}

Provide:
TITLE: PRD title
OVERVIEW: Brief overview
PROBLEM: Problem statement
GOALS:
- Goal 1
- Goal 2
NON_GOALS:
- What this is NOT
PERSONAS:
- Name: description
SUCCESS_METRICS:
- Metric 1
TIMELINE: Estimated timeline
RISKS:
- Risk 1""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a product manager writing a PRD.",
            max_tokens=1000,
        )

        prd = self._parse_prd(response.content)

        await self.log_decision(
            decision=f"Generated PRD: {prd.title}",
            category="prd_generation",
            rationale=f"For feature: {feature_description[:50]}",
            tags=["product", "prd"],
        )

        return prd

    def _parse_prd(self, content: str) -> PRD:
        """Parse PRD from LLM response."""
        prd = PRD(title="Untitled PRD")
        current_section = None

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("TITLE:"):
                prd.title = line.replace("TITLE:", "").strip()
            elif line.startswith("OVERVIEW:"):
                prd.overview = line.replace("OVERVIEW:", "").strip()
            elif line.startswith("PROBLEM:"):
                prd.problem_statement = line.replace("PROBLEM:", "").strip()
            elif line.startswith("GOALS:"):
                current_section = "goals"
            elif line.startswith("NON_GOALS:"):
                current_section = "non_goals"
            elif line.startswith("PERSONAS:"):
                current_section = "personas"
            elif line.startswith("SUCCESS_METRICS:"):
                current_section = "metrics"
            elif line.startswith("TIMELINE:"):
                prd.timeline = line.replace("TIMELINE:", "").strip()
            elif line.startswith("RISKS:"):
                current_section = "risks"
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "goals":
                    prd.goals.append(item)
                elif current_section == "non_goals":
                    prd.non_goals.append(item)
                elif current_section == "personas":
                    if ":" in item:
                        name, desc = item.split(":", 1)
                        prd.user_personas.append({"name": name.strip(), "description": desc.strip()})
                    else:
                        prd.user_personas.append({"name": item, "description": ""})
                elif current_section == "metrics":
                    prd.success_metrics.append(item)
                elif current_section == "risks":
                    prd.risks.append(item)

        # Add any analyzed requirements
        prd.requirements = self._requirements

        return prd

    def get_backlog_summary(self) -> dict[str, Any]:
        """Get summary of product backlog."""
        stories_by_type = {}
        for s in self._stories:
            stories_by_type[s.story_type.value] = stories_by_type.get(s.story_type.value, 0) + 1

        stories_by_status = {}
        for s in self._stories:
            stories_by_status[s.status.value] = stories_by_status.get(s.status.value, 0) + 1

        total_points = sum(s.story_points for s in self._stories)

        return {
            "total_stories": len(self._stories),
            "total_points": total_points,
            "by_type": stories_by_type,
            "by_status": stories_by_status,
            "requirements": len(self._requirements),
            "features_planned": len(self._features),
        }
