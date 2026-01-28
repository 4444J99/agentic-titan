"""
Titan Workflows - Multi-Perspective Collaborative Inquiry System

This module provides a configurable workflow engine for multi-AI collaborative
inquiry. It uses specialized personas that each approach topics from different
cognitive angles, with multi-model routing for optimal results.

Components:
- inquiry_config: Workflow and stage configuration dataclasses
- inquiry_engine: Core workflow execution engine
- inquiry_prompts: Stage-specific prompt templates
- cognitive_router: Multi-model routing based on cognitive task type
- inquiry_export: Markdown export utilities

Key Classes:
- InquiryWorkflow: Defines a sequence of inquiry stages
- InquiryEngine: Executes workflows with multi-model routing
- CognitiveRouter: Selects optimal models for cognitive tasks

Example:
    from titan.workflows import InquiryEngine, EXPANSIVE_INQUIRY_WORKFLOW

    engine = InquiryEngine()
    session = await engine.start_inquiry(
        topic="The nature of consciousness",
        workflow=EXPANSIVE_INQUIRY_WORKFLOW,
    )
    await engine.run_full_workflow(session)
"""

from titan.workflows.inquiry_config import (
    InquiryStage,
    InquiryWorkflow,
    EXPANSIVE_INQUIRY_WORKFLOW,
    DEFAULT_WORKFLOWS,
)
from titan.workflows.inquiry_engine import (
    StageResult,
    InquirySession,
    InquiryEngine,
    InquiryStatus,
)
from titan.workflows.cognitive_router import (
    CognitiveTaskType,
    CognitiveRouter,
    COGNITIVE_MODEL_MAP,
)
from titan.workflows.inquiry_export import (
    export_stage_to_markdown,
    export_session_to_markdown,
)
from titan.workflows.inquiry_prompts import STAGE_PROMPTS

__all__ = [
    # Config
    "InquiryStage",
    "InquiryWorkflow",
    "EXPANSIVE_INQUIRY_WORKFLOW",
    "DEFAULT_WORKFLOWS",
    # Engine
    "StageResult",
    "InquirySession",
    "InquiryEngine",
    "InquiryStatus",
    # Router
    "CognitiveTaskType",
    "CognitiveRouter",
    "COGNITIVE_MODEL_MAP",
    # Export
    "export_stage_to_markdown",
    "export_session_to_markdown",
    # Prompts
    "STAGE_PROMPTS",
]
