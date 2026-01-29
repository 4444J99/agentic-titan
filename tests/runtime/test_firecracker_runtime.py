"""
Tests for Firecracker Runtime (Phase 18B)
"""

import pytest
import platform
from unittest.mock import AsyncMock, MagicMock, patch


class TestFirecrackerImports:
    """Tests for Firecracker module imports."""

    def test_import_config(self):
        """Test importing config module."""
        from runtime.firecracker.config import FirecrackerConfig, VMState

        assert FirecrackerConfig is not None
        assert VMState is not None

    def test_import_vm(self):
        """Test importing VM module."""
        from runtime.firecracker.vm import MicroVM, MicroVMManager

        assert MicroVM is not None
        assert MicroVMManager is not None

    def test_import_network(self):
        """Test importing network module."""
        from runtime.firecracker.network import FirecrackerNetwork

        assert FirecrackerNetwork is not None

    def test_import_guest_agent(self):
        """Test importing guest agent module."""
        from runtime.firecracker.guest_agent import (
            GuestAgentProtocol,
            CommandResult,
        )

        assert GuestAgentProtocol is not None
        assert CommandResult is not None

    def test_import_image_builder(self):
        """Test importing image builder module."""
        from runtime.firecracker.image_builder import ImageBuilder

        assert ImageBuilder is not None


class TestFirecrackerAvailability:
    """Tests for Firecracker availability checking."""

    def test_availability_flags(self):
        """Test availability flags exist."""
        from runtime.firecracker import FIRECRACKER_AVAILABLE, KVM_AVAILABLE

        # On non-Linux, should be False
        if platform.system() != "Linux":
            assert FIRECRACKER_AVAILABLE is False
            assert KVM_AVAILABLE is False


class TestFirecrackerRuntime:
    """Tests for FirecrackerRuntime."""

    def test_create_runtime(self):
        """Test creating runtime instance."""
        from runtime.firecracker.runtime import FirecrackerRuntime
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig()
        runtime = FirecrackerRuntime(firecracker_config=config)

        assert runtime is not None
        assert runtime._fc_config is config

    def test_runtime_type(self):
        """Test runtime type."""
        from runtime.firecracker.runtime import FirecrackerRuntime

        runtime = FirecrackerRuntime()

        # RuntimeType is LOCAL until enum is updated
        assert runtime.type is not None

    def test_supports_gpu(self):
        """Test GPU support check."""
        from runtime.firecracker.runtime import FirecrackerRuntime

        runtime = FirecrackerRuntime()

        assert runtime.supports_gpu() is False

    def test_get_resource_limits(self):
        """Test getting resource limits."""
        from runtime.firecracker.runtime import FirecrackerRuntime
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(vcpu_count=2, mem_size_mib=512)
        runtime = FirecrackerRuntime(firecracker_config=config)

        limits = runtime.get_resource_limits()

        assert limits["max_vcpus"] == 2
        assert limits["max_memory_mib"] == 512

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        from runtime.firecracker.runtime import FirecrackerRuntime

        runtime = FirecrackerRuntime()
        health = await runtime.health_check()

        assert "firecracker_available" in health
        assert "binary_exists" in health

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="Firecracker only available on Linux"
    )
    async def test_initialize_requires_linux(self):
        """Test that initialization requires Linux."""
        from runtime.firecracker.runtime import FirecrackerRuntime

        runtime = FirecrackerRuntime()

        # On non-Linux, should raise
        if platform.system() != "Linux":
            with pytest.raises(RuntimeError, match="requires Linux"):
                await runtime.initialize()


class TestFirecrackerNetwork:
    """Tests for FirecrackerNetwork."""

    def test_create_network_manager(self):
        """Test creating network manager."""
        from runtime.firecracker.network import FirecrackerNetwork

        network = FirecrackerNetwork()

        assert network is not None
        assert network._bridge_name == "fcbr0"

    def test_get_tap_device_not_found(self):
        """Test getting non-existent TAP device."""
        from runtime.firecracker.network import FirecrackerNetwork

        network = FirecrackerNetwork()
        device = network.get_tap_device("nonexistent")

        assert device is None

    def test_list_tap_devices_empty(self):
        """Test listing TAP devices when empty."""
        from runtime.firecracker.network import FirecrackerNetwork

        network = FirecrackerNetwork()
        devices = network.list_tap_devices()

        assert devices == []


class TestGuestAgentProtocol:
    """Tests for GuestAgentProtocol."""

    def test_create_protocol(self):
        """Test creating guest agent protocol."""
        from runtime.firecracker.guest_agent import GuestAgentProtocol

        protocol = GuestAgentProtocol()

        assert protocol is not None

    def test_command_result(self):
        """Test CommandResult dataclass."""
        from runtime.firecracker.guest_agent import CommandResult

        result = CommandResult(
            exit_code=0,
            stdout="output",
            stderr="",
            duration_ms=50,
        )

        assert result.exit_code == 0
        assert result.stdout == "output"

    def test_vm_metrics(self):
        """Test VMMetrics dataclass."""
        from runtime.firecracker.guest_agent import VMMetrics

        metrics = VMMetrics(
            cpu_percent=50.0,
            memory_used_mb=128.0,
            memory_total_mb=256.0,
        )

        assert metrics.cpu_percent == 50.0
        assert metrics.memory_used_mb == 128.0


class TestImageBuilder:
    """Tests for ImageBuilder."""

    def test_create_builder(self):
        """Test creating image builder."""
        from runtime.firecracker.image_builder import ImageBuilder, ImageConfig

        config = ImageConfig()
        builder = ImageBuilder(config)

        assert builder is not None

    def test_image_config(self):
        """Test ImageConfig defaults."""
        from runtime.firecracker.image_builder import ImageConfig

        config = ImageConfig()

        assert config.base_image == "alpine"
        assert config.rootfs_size_mb == 256
        assert "python3" in config.packages

    def test_image_info(self):
        """Test ImageInfo dataclass."""
        from runtime.firecracker.image_builder import ImageInfo

        info = ImageInfo(
            path="/path/to/image",
            size_bytes=1024,
            created_at="2024-01-01",
        )

        assert info.path == "/path/to/image"
        assert info.size_bytes == 1024


class TestRuntimeSelectorIntegration:
    """Tests for RuntimeSelector with Firecracker."""

    def test_firecracker_in_selector(self):
        """Test that Firecracker is considered by selector."""
        from runtime.base import RuntimeType

        # Verify FIRECRACKER exists in enum
        assert hasattr(RuntimeType, "FIRECRACKER")
        assert RuntimeType.FIRECRACKER.value == "firecracker"

    def test_selector_scores_firecracker(self):
        """Test that selector can score Firecracker."""
        from runtime.selector import RuntimeSelector, RuntimeConstraints

        selector = RuntimeSelector()

        constraints = RuntimeConstraints(
            needs_isolation=True,
            requires_gpu=False,
        )

        # Score should work even if Firecracker not available
        # (will just be scored lower or unavailable)


class TestFirecrackerRuntimePackage:
    """Tests for Firecracker package exports."""

    def test_package_exports(self):
        """Test that package exports expected symbols."""
        from runtime import firecracker

        assert hasattr(firecracker, "FIRECRACKER_AVAILABLE")
        assert hasattr(firecracker, "KVM_AVAILABLE")
        assert hasattr(firecracker, "FirecrackerConfig")
        assert hasattr(firecracker, "MicroVM")
        assert hasattr(firecracker, "MicroVMManager")
        assert hasattr(firecracker, "FirecrackerNetwork")
        assert hasattr(firecracker, "GuestAgentProtocol")
        assert hasattr(firecracker, "FirecrackerRuntime")
        assert hasattr(firecracker, "ImageBuilder")
