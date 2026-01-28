"""
Titan Workflows - Inquiry Engine

Core workflow execution engine for multi-perspective collaborative inquiry.
Orchestrates the execution of inquiry stages with multi-model routing,
context accumulation, and real-time progress updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

from titan.workflows.inquiry_config import (
    CognitiveStyle,
    InquiryStage,
    InquiryWorkflow,
    EXPANSIVE_INQUIRY_WORKFLOW,
)
from titan.workflows.inquiry_prompts import get_prompt
from titan.workflows.cognitive_router import (
    CognitiveRouter,
    CognitiveTaskType,
    get_cognitive_router,
)

if TYPE_CHECKING:
    from hive.memory import HiveMind

logger = logging.getLogger("titan.workflows.inquiry_engine")


class InquiryStatus(str, Enum):
    """Status of an inquiry session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StageResult:
    """Result of executing a single inquiry stage."""

    stage_name: str
    role: str
    content: str
    model_used: str
    timestamp: datetime
    tokens_used: int = 0
    duration_ms: int = 0
    stage_index: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the stage completed successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "role": self.role,
            "content": self.content,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "stage_index": self.stage_index,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class InquirySession:
    """
    An active inquiry session.

    Tracks the state of a multi-stage inquiry including the topic,
    workflow configuration, current progress, and accumulated results.
    """

    id: str
    topic: str
    workflow: InquiryWorkflow
    status: InquiryStatus = InquiryStatus.PENDING
    current_stage: int = 0
    results: list[StageResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_stages(self) -> int:
        """Total number of stages in the workflow."""
        return len(self.workflow.stages)

    @property
    def progress(self) -> float:
        """Progress as a percentage (0-100)."""
        if self.total_stages == 0:
            return 100.0
        return (len(self.results) / self.total_stages) * 100

    @property
    def is_complete(self) -> bool:
        """Whether all stages have been executed."""
        return len(self.results) >= self.total_stages

    def get_previous_context(self) -> str:
        """Get accumulated context from previous stages as JSON."""
        if not self.results:
            return ""

        context = {}
        for result in self.results:
            context[result.stage_name] = {
                "role": result.role,
                "content": result.content,
                "stage_index": result.stage_index,
            }

        return json.dumps(context, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "topic": self.topic,
            "workflow_name": self.workflow.name,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "total_stages": self.total_stages,
            "progress": self.progress,
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
        }


class InquiryEngine:
    """
    Core engine for executing multi-perspective inquiry workflows.

    Features:
    - Multi-model routing based on cognitive task type
    - Context accumulation between stages
    - Progress streaming via callbacks
    - Integration with Hive Mind for shared memory
    """

    def __init__(
        self,
        cognitive_router: CognitiveRouter | None = None,
        hive_mind: HiveMind | None = None,
        llm_caller: Callable[[str, str], Any] | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        """
        Initialize the inquiry engine.

        Args:
            cognitive_router: Router for selecting models per cognitive task
            hive_mind: Shared memory for agents (optional)
            llm_caller: Function to call LLM (async). Signature: (prompt, model) -> response
            default_model: Fallback model when routing unavailable
        """
        self._cognitive_router = cognitive_router or get_cognitive_router()
        self._hive_mind = hive_mind
        self._llm_caller = llm_caller
        self._default_model = default_model

        # Active sessions
        self._sessions: dict[str, InquirySession] = {}

        # Event handlers
        self._on_stage_started: list[Callable[[InquirySession, int], None]] = []
        self._on_stage_completed: list[Callable[[InquirySession, StageResult], None]] = []
        self._on_session_completed: list[Callable[[InquirySession], None]] = []

        logger.info("Inquiry engine initialized")

    # =========================================================================
    # Session Management
    # =========================================================================

    async def start_inquiry(
        self,
        topic: str,
        workflow: InquiryWorkflow | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InquirySession:
        """
        Start a new inquiry session.

        Args:
            topic: The topic to explore
            workflow: Workflow to use (defaults to EXPANSIVE_INQUIRY_WORKFLOW)
            metadata: Optional metadata to attach

        Returns:
            New InquirySession
        """
        session_id = f"inq-{uuid.uuid4().hex[:12]}"

        session = InquirySession(
            id=session_id,
            topic=topic,
            workflow=workflow or EXPANSIVE_INQUIRY_WORKFLOW,
            status=InquiryStatus.PENDING,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session
        logger.info(f"Started inquiry session {session_id} for topic: {topic[:50]}...")

        # Store in Hive Mind if available
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session_id}",
                session.to_dict(),
                ttl=3600 * 24,  # 24 hours
            )

        return session

    def get_session(self, session_id: str) -> InquirySession | None:
        """Get an inquiry session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        status: InquiryStatus | None = None,
    ) -> list[InquirySession]:
        """List all sessions, optionally filtered by status."""
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    # =========================================================================
    # Stage Execution
    # =========================================================================

    async def run_stage(
        self,
        session: InquirySession,
        stage_index: int | None = None,
    ) -> StageResult:
        """
        Execute a single stage of the inquiry.

        Args:
            session: The inquiry session
            stage_index: Index of stage to run (defaults to next stage)

        Returns:
            StageResult with the stage output
        """
        # Determine which stage to run
        if stage_index is None:
            stage_index = len(session.results)

        if stage_index >= len(session.workflow.stages):
            raise ValueError(f"Stage index {stage_index} out of range")

        stage = session.workflow.stages[stage_index]
        session.current_stage = stage_index

        # Update session status
        if session.status == InquiryStatus.PENDING:
            session.status = InquiryStatus.RUNNING
            session.started_at = datetime.now()

        # Notify handlers
        for handler in self._on_stage_started:
            try:
                handler(session, stage_index)
            except Exception as e:
                logger.warning(f"Stage started handler error: {e}")

        logger.info(f"Running stage {stage_index + 1}/{len(session.workflow.stages)}: {stage.name}")

        start_time = time.time()

        try:
            # Build the prompt
            prompt = self._build_stage_prompt(session, stage)

            # Route to appropriate model
            cognitive_type = self._style_to_cognitive_type(stage.cognitive_style)
            routing = await self._cognitive_router.route_for_task(
                cognitive_type,
                preferred_model=stage.preferred_model,
            )
            model = routing.model_id

            logger.debug(f"Using model {model} for {stage.name} (score: {routing.score})")

            # Call the LLM
            if self._llm_caller:
                response = await self._llm_caller(prompt, model)
                content = response if isinstance(response, str) else str(response)
                tokens_used = len(content.split()) * 2  # Rough estimate
            else:
                # Mock response for testing
                content = self._mock_stage_response(stage, session.topic)
                tokens_used = 0

            duration_ms = int((time.time() - start_time) * 1000)

            result = StageResult(
                stage_name=stage.name,
                role=stage.role,
                content=content,
                model_used=model,
                timestamp=datetime.now(),
                tokens_used=tokens_used,
                duration_ms=duration_ms,
                stage_index=stage_index,
                metadata={
                    "routing_score": routing.score,
                    "routing_reasoning": routing.reasoning,
                    "cognitive_style": stage.cognitive_style.value,
                },
            )

        except Exception as e:
            logger.error(f"Stage {stage.name} failed: {e}")
            duration_ms = int((time.time() - start_time) * 1000)

            result = StageResult(
                stage_name=stage.name,
                role=stage.role,
                content="",
                model_used=self._default_model,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                stage_index=stage_index,
                error=str(e),
            )

        # Store result
        session.results.append(result)

        # Notify handlers
        for handler in self._on_stage_completed:
            try:
                handler(session, result)
            except Exception as e:
                logger.warning(f"Stage completed handler error: {e}")

        # Update Hive Mind
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session.id}",
                session.to_dict(),
                ttl=3600 * 24,
            )

        return result

    async def run_full_workflow(
        self,
        session: InquirySession,
    ) -> InquirySession:
        """
        Run all stages of the workflow sequentially.

        Args:
            session: The inquiry session to run

        Returns:
            The completed session
        """
        logger.info(f"Running full workflow for session {session.id}")

        try:
            while len(session.results) < len(session.workflow.stages):
                if session.status == InquiryStatus.CANCELLED:
                    logger.info(f"Session {session.id} was cancelled")
                    break

                await self.run_stage(session)

            # Mark complete
            if session.status == InquiryStatus.RUNNING:
                session.status = InquiryStatus.COMPLETED
                session.completed_at = datetime.now()

                # Notify handlers
                for handler in self._on_session_completed:
                    try:
                        handler(session)
                    except Exception as e:
                        logger.warning(f"Session completed handler error: {e}")

            logger.info(
                f"Workflow complete for session {session.id}. "
                f"Ran {len(session.results)} stages."
            )

        except Exception as e:
            logger.error(f"Workflow failed for session {session.id}: {e}")
            session.status = InquiryStatus.FAILED
            session.error = str(e)
            session.completed_at = datetime.now()

        # Update Hive Mind
        if self._hive_mind:
            await self._hive_mind.set(
                f"inquiry:{session.id}",
                session.to_dict(),
                ttl=3600 * 24 * 7,  # Keep completed sessions for 7 days
            )

        return session

    async def stream_workflow(
        self,
        session: InquirySession,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run workflow with streaming progress updates.

        Yields dictionaries with event type and data for each stage.

        Args:
            session: The inquiry session

        Yields:
            Progress events
        """
        yield {
            "type": "session_started",
            "session_id": session.id,
            "topic": session.topic,
            "total_stages": session.total_stages,
        }

        try:
            while len(session.results) < session.total_stages:
                if session.status == InquiryStatus.CANCELLED:
                    yield {
                        "type": "session_cancelled",
                        "session_id": session.id,
                    }
                    break

                stage_index = len(session.results)
                stage = session.workflow.stages[stage_index]

                yield {
                    "type": "stage_started",
                    "session_id": session.id,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "role": stage.role,
                }

                result = await self.run_stage(session)

                yield {
                    "type": "stage_completed",
                    "session_id": session.id,
                    "stage_index": stage_index,
                    "result": result.to_dict(),
                }

            if session.status != InquiryStatus.CANCELLED:
                session.status = InquiryStatus.COMPLETED
                session.completed_at = datetime.now()

                yield {
                    "type": "session_completed",
                    "session_id": session.id,
                    "results_count": len(session.results),
                }

        except Exception as e:
            session.status = InquiryStatus.FAILED
            session.error = str(e)
            session.completed_at = datetime.now()

            yield {
                "type": "session_failed",
                "session_id": session.id,
                "error": str(e),
            }

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a running session."""
        session = self._sessions.get(session_id)
        if session and session.status == InquiryStatus.RUNNING:
            session.status = InquiryStatus.CANCELLED
            session.completed_at = datetime.now()
            logger.info(f"Cancelled session {session_id}")
            return True
        return False

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_stage_started(
        self,
        handler: Callable[[InquirySession, int], None],
    ) -> None:
        """Register a handler for stage start events."""
        self._on_stage_started.append(handler)

    def on_stage_completed(
        self,
        handler: Callable[[InquirySession, StageResult], None],
    ) -> None:
        """Register a handler for stage completion events."""
        self._on_stage_completed.append(handler)

    def on_session_completed(
        self,
        handler: Callable[[InquirySession], None],
    ) -> None:
        """Register a handler for session completion events."""
        self._on_session_completed.append(handler)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _build_stage_prompt(
        self,
        session: InquirySession,
        stage: InquiryStage,
    ) -> str:
        """Build the prompt for a stage with context."""
        previous_context = ""
        if session.workflow.context_accumulation and session.results:
            previous_context = session.get_previous_context()

        return get_prompt(
            template_key=stage.prompt_template,
            topic=session.topic,
            previous_context=previous_context,
            stage_number=len(session.results) + 1,
            total_stages=session.total_stages,
        )

    def _style_to_cognitive_type(
        self,
        style: CognitiveStyle,
    ) -> CognitiveTaskType:
        """Map cognitive style to cognitive task type for routing."""
        mapping = {
            CognitiveStyle.STRUCTURED_REASONING: CognitiveTaskType.STRUCTURED_REASONING,
            CognitiveStyle.CREATIVE_SYNTHESIS: CognitiveTaskType.CREATIVE_SYNTHESIS,
            CognitiveStyle.MATHEMATICAL_ANALYSIS: CognitiveTaskType.MATHEMATICAL_ANALYSIS,
            CognitiveStyle.CROSS_DOMAIN: CognitiveTaskType.CROSS_DOMAIN,
            CognitiveStyle.META_ANALYSIS: CognitiveTaskType.META_ANALYSIS,
            CognitiveStyle.PATTERN_RECOGNITION: CognitiveTaskType.PATTERN_RECOGNITION,
        }
        return mapping.get(style, CognitiveTaskType.STRUCTURED_REASONING)

    def _mock_stage_response(
        self,
        stage: InquiryStage,
        topic: str,
    ) -> str:
        """Generate mock response for testing without LLM."""
        return f"""## {stage.name} Analysis

**Topic:** {topic}

**Role:** {stage.role}

This is a mock response for the {stage.name} stage.
In production, this would contain the AI's actual analysis.

### Key Insights
- Insight 1 related to {topic}
- Insight 2 exploring different angles
- Insight 3 with deeper analysis

### {stage.description}

The {stage.role} has analyzed the topic through its unique cognitive lens,
revealing aspects that complement other perspectives.
"""


# =============================================================================
# Factory Functions
# =============================================================================

_default_engine: InquiryEngine | None = None


def get_inquiry_engine() -> InquiryEngine:
    """Get the default inquiry engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = InquiryEngine()
    return _default_engine


async def quick_inquiry(
    topic: str,
    workflow_name: str = "expansive",
) -> InquirySession:
    """
    Quick helper to run a full inquiry and return results.

    Args:
        topic: Topic to explore
        workflow_name: Name of workflow to use

    Returns:
        Completed InquirySession
    """
    from titan.workflows.inquiry_config import get_workflow

    engine = get_inquiry_engine()
    workflow = get_workflow(workflow_name) or EXPANSIVE_INQUIRY_WORKFLOW

    session = await engine.start_inquiry(topic, workflow)
    return await engine.run_full_workflow(session)
