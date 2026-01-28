"""
Dashboard Components - Modular UI components for the Titan dashboard.

Provides:
- ConversationManager: Multi-conversation support
- DiffViewer: Code diff visualization
- FileBrowser: Workspace file navigation
- ApprovalComponent: Human-in-the-Loop approval UI
"""

from dashboard.components.conversations import ConversationManager, Conversation
from dashboard.components.diff import DiffViewer, FileDiff
from dashboard.components.filebrowser import FileBrowser, FileEntry
from dashboard.components.approval import ApprovalComponent, ApprovalWebSocketHandler

__all__ = [
    "ConversationManager",
    "Conversation",
    "DiffViewer",
    "FileDiff",
    "FileBrowser",
    "FileEntry",
    "ApprovalComponent",
    "ApprovalWebSocketHandler",
]
