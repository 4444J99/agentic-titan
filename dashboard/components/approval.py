"""
Titan Dashboard - Approval Component

WebSocket-based approval UI component for Human-in-the-Loop gates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from fastapi import WebSocket
    from titan.safety.hitl import HITLHandler

logger = logging.getLogger("titan.dashboard.approval")


@dataclass
class ApprovalNotification:
    """Notification for approval UI."""

    type: str = "approval_request"
    request_id: str = ""
    action: str = ""
    description: str = ""
    risk_level: str = ""
    agent_id: str = ""
    session_id: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    timeout_seconds: int = 300

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type,
            "request_id": self.request_id,
            "action": self.action,
            "description": self.description,
            "risk_level": self.risk_level,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "created_at": self.created_at,
            "timeout_seconds": self.timeout_seconds,
        })


class ApprovalWebSocketHandler:
    """
    WebSocket handler for approval notifications.

    Manages connections and broadcasts approval requests to connected clients.
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        logger.info(f"Approval WebSocket connected, total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        logger.info(f"Approval WebSocket disconnected, total: {len(self._connections)}")

    async def broadcast(self, message: str) -> None:
        """Broadcast message to all connected clients."""
        disconnected = []

        async with self._lock:
            for websocket in self._connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket: {e}")
                    disconnected.append(websocket)

            # Clean up disconnected
            for ws in disconnected:
                self._connections.remove(ws)

    async def send_approval_request(
        self,
        request_id: str,
        action: str,
        description: str,
        risk_level: str,
        agent_id: str,
        session_id: str,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        timeout_seconds: int = 300,
    ) -> None:
        """Send an approval request notification."""
        notification = ApprovalNotification(
            type="approval_request",
            request_id=request_id,
            action=action,
            description=description,
            risk_level=risk_level,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments or {},
            created_at=datetime.utcnow().isoformat(),
            timeout_seconds=timeout_seconds,
        )

        await self.broadcast(notification.to_json())

    async def send_approval_response(
        self,
        request_id: str,
        approved: bool,
        responder: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Send an approval response notification."""
        message = json.dumps({
            "type": "approval_response",
            "request_id": request_id,
            "approved": approved,
            "responder": responder,
            "reason": reason,
            "responded_at": datetime.utcnow().isoformat(),
        })

        await self.broadcast(message)


class ApprovalComponent:
    """
    Approval UI component for the dashboard.

    Integrates with HITLHandler to show and handle approval requests.
    """

    def __init__(
        self,
        hitl_handler: HITLHandler | None = None,
        websocket_handler: ApprovalWebSocketHandler | None = None,
    ) -> None:
        self._hitl = hitl_handler
        self._ws_handler = websocket_handler or ApprovalWebSocketHandler()

    def set_hitl_handler(self, handler: HITLHandler) -> None:
        """Set the HITL handler and register callback."""
        self._hitl = handler
        handler.add_notification_callback(self._on_approval_request)

    @property
    def websocket_handler(self) -> ApprovalWebSocketHandler:
        """Get the WebSocket handler."""
        return self._ws_handler

    async def _on_approval_request(self, request: Any) -> None:
        """Callback for new approval requests."""
        await self._ws_handler.send_approval_request(
            request_id=str(request.id),
            action=request.action,
            description=request.description,
            risk_level=request.risk_level.value,
            agent_id=request.agent_id,
            session_id=request.session_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            timeout_seconds=request.timeout_seconds,
        )

    async def respond(
        self,
        request_id: str,
        approved: bool,
        responder: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """
        Respond to an approval request.

        Args:
            request_id: ID of the request
            approved: Whether to approve
            responder: Who made the decision
            reason: Reason for decision

        Returns:
            True if response was recorded
        """
        if not self._hitl:
            logger.error("HITL handler not configured")
            return False

        success = await self._hitl.respond_to_approval(
            request_id=request_id,
            approved=approved,
            responder=responder,
            reason=reason,
        )

        if success:
            await self._ws_handler.send_approval_response(
                request_id=request_id,
                approved=approved,
                responder=responder,
                reason=reason,
            )

        return success

    async def get_pending(self) -> list[dict[str, Any]]:
        """Get all pending approval requests."""
        if not self._hitl:
            return []

        requests = await self._hitl.get_pending_requests()
        return [r.to_dict() for r in requests]

    def get_approval_html(self) -> str:
        """Generate HTML for the approval UI component."""
        return """
<div id="approval-panel" class="approval-panel">
    <h3>Pending Approvals</h3>
    <div id="approval-list"></div>
</div>

<style>
.approval-panel {
    padding: 1rem;
    background: #1e1e1e;
    border-radius: 8px;
    margin: 1rem 0;
}

.approval-item {
    background: #2d2d2d;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
    border-left: 4px solid #666;
}

.approval-item.critical { border-color: #ff4444; }
.approval-item.high { border-color: #ff8800; }
.approval-item.medium { border-color: #ffcc00; }
.approval-item.low { border-color: #44cc44; }

.approval-action {
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.approval-meta {
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 0.5rem;
}

.approval-buttons {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.btn-approve {
    background: #44cc44;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
}

.btn-deny {
    background: #ff4444;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
}

.btn-approve:hover { background: #33bb33; }
.btn-deny:hover { background: #ee3333; }
</style>

<script>
(function() {
    const approvalList = document.getElementById('approval-list');
    const wsUrl = `ws://${window.location.host}/ws/approvals`;
    let ws = null;

    function connect() {
        ws = new WebSocket(wsUrl);

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.type === 'approval_request') {
                addApprovalRequest(data);
            } else if (data.type === 'approval_response') {
                removeApprovalRequest(data.request_id);
            }
        };

        ws.onclose = function() {
            setTimeout(connect, 3000);
        };
    }

    function addApprovalRequest(request) {
        const item = document.createElement('div');
        item.className = `approval-item ${request.risk_level}`;
        item.id = `approval-${request.request_id}`;

        item.innerHTML = `
            <div class="approval-action">${escapeHtml(request.action)}</div>
            <div class="approval-meta">
                Risk: ${request.risk_level.toUpperCase()} |
                Agent: ${escapeHtml(request.agent_id)} |
                Tool: ${escapeHtml(request.tool_name || 'N/A')}
            </div>
            <div class="approval-meta">
                Arguments: <code>${escapeHtml(JSON.stringify(request.arguments))}</code>
            </div>
            <div class="approval-buttons">
                <button class="btn-approve" onclick="respondApproval('${request.request_id}', true)">
                    Approve
                </button>
                <button class="btn-deny" onclick="respondApproval('${request.request_id}', false)">
                    Deny
                </button>
            </div>
        `;

        approvalList.prepend(item);
    }

    function removeApprovalRequest(requestId) {
        const item = document.getElementById(`approval-${requestId}`);
        if (item) {
            item.remove();
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    window.respondApproval = async function(requestId, approved) {
        try {
            const response = await fetch('/api/approvals/respond', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_id: requestId,
                    approved: approved,
                    responder: 'dashboard_user'
                })
            });

            if (!response.ok) {
                console.error('Failed to respond to approval');
            }
        } catch (e) {
            console.error('Error responding to approval:', e);
        }
    };

    // Load initial pending requests
    async function loadPending() {
        try {
            const response = await fetch('/api/approvals/pending');
            const data = await response.json();
            data.forEach(addApprovalRequest);
        } catch (e) {
            console.error('Error loading pending approvals:', e);
        }
    }

    connect();
    loadPending();
})();
</script>
"""
