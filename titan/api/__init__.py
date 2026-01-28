"""
Titan API - FastAPI Application

Provides REST and WebSocket APIs for the Titan platform including:
- Inquiry workflow management
- Agent orchestration
- Memory system access
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("titan.api")

# Create the main FastAPI app
app = FastAPI(
    title="Titan API",
    description="Multi-Agent Orchestration and Collaborative Inquiry System",
    version="0.1.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the main API router
api_router = APIRouter(prefix="/api")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "titan-api",
        "version": "0.1.0",
    }


# Import and include routers
def register_routers() -> None:
    """Register all API routers."""
    from titan.api.inquiry_routes import inquiry_router
    from titan.api.inquiry_ws import ws_router
    from titan.api.batch_routes import batch_router
    from titan.api.batch_ws import batch_ws_router

    api_router.include_router(inquiry_router)
    api_router.include_router(batch_router)
    app.include_router(api_router)
    app.include_router(ws_router)  # WebSocket routes at root level
    app.include_router(batch_ws_router)  # Batch WebSocket/SSE routes


# Lazy registration to avoid circular imports
_routers_registered = False


@app.on_event("startup")
async def startup_event() -> None:
    """Handle app startup."""
    global _routers_registered
    if not _routers_registered:
        register_routers()
        _routers_registered = True
    logger.info("Titan API started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Handle app shutdown."""
    logger.info("Titan API shutting down")


__all__ = ["app", "api_router"]
