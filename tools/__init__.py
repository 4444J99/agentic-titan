"""
Titan Tools - Executable tools for agents.

Provides:
- Tool protocol and registry
- Built-in tools (file, web, shell)
- RAG (Retrieval Augmented Generation)
- Document processing (PDF, DOCX, XLSX, PPTX)
- Web search with citations
- MCP bridge for external tools
"""

from tools.base import (
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    get_registry,
    register_tool,
)
from tools.documents import DOCXTool, PPTXTool, XLSXTool
from tools.executor import ToolExecutor, get_executor
from tools.pdf import PDFTool, extract_pdf_text

# Import tools to register them
from tools.rag import RAGStore, RAGTool
from tools.rag import get_store as get_rag_store
from tools.search import CitationManager, SearchResults, SearchTool

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    "ToolParameter",
    "ToolRegistry",
    "get_registry",
    "register_tool",
    # Executor
    "ToolExecutor",
    "get_executor",
    # RAG
    "RAGTool",
    "RAGStore",
    "get_rag_store",
    # PDF
    "PDFTool",
    "extract_pdf_text",
    # Search
    "SearchTool",
    "SearchResults",
    "CitationManager",
    # Documents
    "DOCXTool",
    "XLSXTool",
    "PPTXTool",
]
