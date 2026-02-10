"""
Dashboard Components - Modular UI components for the Titan dashboard.

Provides:
- ConversationManager: Multi-conversation support
- DiffViewer: Code diff visualization
- FileBrowser: Workspace file navigation
- ApprovalComponent: Human-in-the-Loop approval UI
"""

from dashboard.components.approval import ApprovalComponent, ApprovalWebSocketHandler
from dashboard.components.conversations import Conversation, ConversationManager
from dashboard.components.diff import DiffViewer, FileDiff
from dashboard.components.filebrowser import FileBrowser, FileEntry

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
