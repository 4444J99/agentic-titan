"""
Titan Learning - Human Feedback Handlers

Provides interfaces for collecting human feedback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from titan.learning.rlhf import RLHFCollector, RLHFSample, FeedbackType

logger = logging.getLogger("titan.learning.feedback")


class FeedbackChannel(str, Enum):
    """Channels for receiving feedback."""

    DASHBOARD = "dashboard"
    CLI = "cli"
    API = "api"
    WEBHOOK = "webhook"
    EMAIL = "email"


@dataclass
class FeedbackRequest:
    """Request for human feedback."""

    id: UUID = field(default_factory=uuid4)
    tracking_id: str = ""
    prompt_preview: str = ""
    response_preview: str = ""
    channel: FeedbackChannel = FeedbackChannel.DASHBOARD
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "tracking_id": self.tracking_id,
            "prompt_preview": self.prompt_preview[:200] + "..." if len(self.prompt_preview) > 200 else self.prompt_preview,
            "response_preview": self.response_preview[:500] + "..." if len(self.response_preview) > 500 else self.response_preview,
            "channel": self.channel.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class FeedbackResponse:
    """Human feedback response."""

    request_id: UUID
    rating: int | None = None  # 1-5
    thumbs_up: bool | None = None
    accepted: bool | None = None
    correction: str | None = None
    comment: str | None = None
    responder_id: str | None = None
    responded_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": str(self.request_id),
            "rating": self.rating,
            "thumbs_up": self.thumbs_up,
            "accepted": self.accepted,
            "correction": self.correction,
            "comment": self.comment,
            "responder_id": self.responder_id,
            "responded_at": self.responded_at.isoformat(),
        }


# Type for feedback callbacks
FeedbackCallback = Callable[[FeedbackRequest], Coroutine[Any, Any, None]]


class FeedbackHandler:
    """
    Handles human feedback collection.

    Integrates with RLHF collector and provides multiple
    channels for feedback collection.
    """

    def __init__(
        self,
        rlhf_collector: RLHFCollector | None = None,
        default_channel: FeedbackChannel = FeedbackChannel.DASHBOARD,
    ) -> None:
        self._rlhf_collector = rlhf_collector
        self._default_channel = default_channel
        self._pending_requests: dict[UUID, FeedbackRequest] = {}
        self._callbacks: dict[FeedbackChannel, list[FeedbackCallback]] = {}
        self._response_events: dict[UUID, asyncio.Event] = {}
        self._responses: dict[UUID, FeedbackResponse] = {}

    def set_rlhf_collector(self, collector: RLHFCollector) -> None:
        """Set the RLHF collector."""
        self._rlhf_collector = collector

    def add_callback(
        self,
        channel: FeedbackChannel,
        callback: FeedbackCallback,
    ) -> None:
        """Add callback for a feedback channel."""
        if channel not in self._callbacks:
            self._callbacks[channel] = []
        self._callbacks[channel].append(callback)

    async def request_feedback(
        self,
        tracking_id: str,
        prompt: str,
        response: str,
        channel: FeedbackChannel | None = None,
        blocking: bool = False,
        timeout_seconds: float = 300,
    ) -> FeedbackResponse | None:
        """
        Request feedback for a response.

        Args:
            tracking_id: RLHF tracking ID
            prompt: The prompt
            response: The response
            channel: Feedback channel to use
            blocking: Wait for response if True
            timeout_seconds: Timeout for blocking wait

        Returns:
            FeedbackResponse if blocking and response received
        """
        request = FeedbackRequest(
            tracking_id=tracking_id,
            prompt_preview=prompt,
            response_preview=response,
            channel=channel or self._default_channel,
        )

        self._pending_requests[request.id] = request

        # Notify callbacks
        await self._notify_callbacks(request)

        if not blocking:
            return None

        # Wait for response
        event = asyncio.Event()
        self._response_events[request.id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            return self._responses.get(request.id)
        except asyncio.TimeoutError:
            logger.warning(f"Feedback request {request.id} timed out")
            return None
        finally:
            self._response_events.pop(request.id, None)
            self._pending_requests.pop(request.id, None)

    async def submit_feedback(
        self,
        request_id: UUID | str,
        rating: int | None = None,
        thumbs_up: bool | None = None,
        accepted: bool | None = None,
        correction: str | None = None,
        comment: str | None = None,
        responder_id: str | None = None,
    ) -> bool:
        """
        Submit feedback for a request.

        Args:
            request_id: Request ID
            rating: 1-5 rating
            thumbs_up: Thumbs up/down
            accepted: Accept/reject
            correction: Corrected text
            comment: Free-form comment
            responder_id: ID of responder

        Returns:
            True if feedback was recorded
        """
        if isinstance(request_id, str):
            request_id = UUID(request_id)

        request = self._pending_requests.get(request_id)
        if not request:
            logger.warning(f"Unknown feedback request: {request_id}")
            return False

        response = FeedbackResponse(
            request_id=request_id,
            rating=rating,
            thumbs_up=thumbs_up,
            accepted=accepted,
            correction=correction,
            comment=comment,
            responder_id=responder_id,
        )

        self._responses[request_id] = response

        # Signal waiting tasks
        if request_id in self._response_events:
            self._response_events[request_id].set()

        # Record in RLHF collector
        if self._rlhf_collector:
            from titan.learning.rlhf import FeedbackType

            feedback_type = FeedbackType.EXPLICIT_RATING
            if thumbs_up is not None:
                feedback_type = FeedbackType.THUMBS
            elif correction:
                feedback_type = FeedbackType.CORRECTION
            elif accepted is not None:
                feedback_type = FeedbackType.ACCEPTANCE

            await self._rlhf_collector.record_feedback(
                tracking_id=request.tracking_id,
                rating=rating,
                accepted=accepted if accepted is not None else (thumbs_up if thumbs_up is not None else None),
                correction=correction,
                feedback_type=feedback_type,
            )

        logger.info(f"Recorded feedback for request {request_id}")
        return True

    async def _notify_callbacks(self, request: FeedbackRequest) -> None:
        """Notify registered callbacks."""
        callbacks = self._callbacks.get(request.channel, [])
        for callback in callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Feedback callback error: {e}")

    def get_pending_requests(self) -> list[FeedbackRequest]:
        """Get all pending feedback requests."""
        return list(self._pending_requests.values())

    async def quick_thumbs(
        self,
        tracking_id: str,
        thumbs_up: bool,
    ) -> bool:
        """
        Quick thumbs up/down feedback.

        Args:
            tracking_id: RLHF tracking ID
            thumbs_up: True for thumbs up, False for thumbs down

        Returns:
            True if recorded
        """
        if not self._rlhf_collector:
            return False

        from titan.learning.rlhf import FeedbackType

        return await self._rlhf_collector.record_feedback(
            tracking_id=tracking_id,
            accepted=thumbs_up,
            feedback_type=FeedbackType.THUMBS,
        )

    async def quick_rating(
        self,
        tracking_id: str,
        rating: int,
    ) -> bool:
        """
        Quick 1-5 rating feedback.

        Args:
            tracking_id: RLHF tracking ID
            rating: 1-5 rating

        Returns:
            True if recorded
        """
        if not self._rlhf_collector:
            return False

        if not 1 <= rating <= 5:
            logger.warning(f"Invalid rating: {rating}, must be 1-5")
            return False

        from titan.learning.rlhf import FeedbackType

        return await self._rlhf_collector.record_feedback(
            tracking_id=tracking_id,
            rating=rating,
            feedback_type=FeedbackType.EXPLICIT_RATING,
        )


# Singleton instance
_default_handler: FeedbackHandler | None = None


def get_feedback_handler() -> FeedbackHandler:
    """Get the default feedback handler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = FeedbackHandler()
    return _default_handler
