"""
Titan Tools - Advanced tools for AI agents.

Provides:
- ImageGenerationTool: AI image generation (Stable Diffusion, DALL-E)
- Microsoft365Tool: Microsoft 365 Graph API integration
"""

from titan.tools.image_gen import (
    DallEBackend,
    GeneratedImage,
    ImageGenerationResult,
    ImageGenerationTool,
    ImageRequest,
    StableDiffusionBackend,
    generate_image,
    get_image_tool,
)
from titan.tools.m365 import (
    CalendarEvent,
    DriveItem,
    EmailMessage,
    GraphClient,
    M365User,
    Microsoft365Tool,
    TeamsMessage,
    get_m365_tool,
)

__all__ = [
    # Image Generation
    "ImageGenerationTool",
    "ImageRequest",
    "GeneratedImage",
    "ImageGenerationResult",
    "StableDiffusionBackend",
    "DallEBackend",
    "generate_image",
    "get_image_tool",
    # Microsoft 365
    "Microsoft365Tool",
    "GraphClient",
    "M365User",
    "EmailMessage",
    "CalendarEvent",
    "DriveItem",
    "TeamsMessage",
    "get_m365_tool",
]
