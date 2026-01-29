"""
Firecracker MicroVM Runtime

Provides secure, lightweight virtualization for code execution using
Firecracker microVMs. Linux only - requires KVM support.

Features:
- Sub-second boot times
- Strong isolation through hardware virtualization
- Minimal resource overhead
- TAP device networking support
- Guest agent for bidirectional communication
"""

from __future__ import annotations

import logging
import os
import platform

logger = logging.getLogger("titan.runtime.firecracker")

# Check platform availability
FIRECRACKER_AVAILABLE = platform.system() == "Linux"
KVM_AVAILABLE = FIRECRACKER_AVAILABLE and os.path.exists("/dev/kvm")

if not FIRECRACKER_AVAILABLE:
    logger.info("Firecracker requires Linux - not available on this platform")
elif not KVM_AVAILABLE:
    logger.warning("KVM not available - Firecracker will not work")

# Always import config (works on all platforms)
from runtime.firecracker.config import FirecrackerConfig, VMState
from runtime.firecracker.vm import MicroVM, MicroVMManager, ExecutionResult, get_vm_manager
from runtime.firecracker.network import FirecrackerNetwork, get_network_manager
from runtime.firecracker.guest_agent import (
    GuestAgentProtocol,
    GuestConnection,
    CommandResult,
    VMMetrics,
)
from runtime.firecracker.runtime import FirecrackerRuntime
from runtime.firecracker.image_builder import ImageBuilder, ImageConfig, ImageInfo, get_image_builder

__all__ = [
    # Availability flags
    "FIRECRACKER_AVAILABLE",
    "KVM_AVAILABLE",
    # Config
    "FirecrackerConfig",
    "VMState",
    # VM management
    "MicroVM",
    "MicroVMManager",
    "ExecutionResult",
    "get_vm_manager",
    # Network
    "FirecrackerNetwork",
    "get_network_manager",
    # Guest agent
    "GuestAgentProtocol",
    "GuestConnection",
    "CommandResult",
    "VMMetrics",
    # Runtime
    "FirecrackerRuntime",
    # Image builder
    "ImageBuilder",
    "ImageConfig",
    "ImageInfo",
    "get_image_builder",
]
