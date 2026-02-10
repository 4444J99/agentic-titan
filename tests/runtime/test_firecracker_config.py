"""
Tests for Firecracker Configuration (Phase 18B)
"""


class TestFirecrackerConfig:
    """Tests for FirecrackerConfig."""

    def test_import(self):
        """Test that config can be imported."""
        from runtime.firecracker.config import FirecrackerConfig, VMState

        assert FirecrackerConfig is not None
        assert VMState is not None

    def test_default_config(self):
        """Test default configuration values."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig()

        assert config.vcpu_count == 1
        assert config.mem_size_mib == 128
        assert config.ht_enabled is False
        assert config.enable_vsock is True
        assert config.timeout_seconds == 30

    def test_custom_config(self):
        """Test custom configuration."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(
            vcpu_count=2,
            mem_size_mib=256,
            timeout_seconds=60,
        )

        assert config.vcpu_count == 2
        assert config.mem_size_mib == 256
        assert config.timeout_seconds == 60

    def test_socket_path(self):
        """Test socket path generation."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig()
        path = config.get_socket_path("test-vm")

        assert "test-vm" in path
        assert path.endswith(".socket")

    def test_to_machine_config(self):
        """Test machine config conversion."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(vcpu_count=2, mem_size_mib=512)
        machine = config.to_machine_config()

        assert machine["vcpu_count"] == 2
        assert machine["mem_size_mib"] == 512

    def test_to_boot_source(self):
        """Test boot source conversion."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(kernel_path="/path/to/kernel")
        boot = config.to_boot_source()

        assert boot["kernel_image_path"] == "/path/to/kernel"
        assert "boot_args" in boot

    def test_to_drive_config(self):
        """Test drive config conversion."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(rootfs_path="/path/to/rootfs")
        drive = config.to_drive_config()

        assert drive["drive_id"] == "rootfs"
        assert drive["path_on_host"] == "/path/to/rootfs"
        assert drive["is_root_device"] is True

    def test_to_network_config_disabled(self):
        """Test network config when disabled."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(enable_network=False)
        network = config.to_network_config()

        assert network is None

    def test_to_network_config_enabled(self):
        """Test network config when enabled."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(
            enable_network=True,
            tap_device="tap0",
            guest_mac="AA:BB:CC:DD:EE:FF",
        )
        network = config.to_network_config()

        assert network is not None
        assert network["host_dev_name"] == "tap0"
        assert network["guest_mac"] == "AA:BB:CC:DD:EE:FF"

    def test_to_vsock_config(self):
        """Test VSOCK config conversion."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(enable_vsock=True, vsock_cid=5)
        vsock = config.to_vsock_config()

        assert vsock is not None
        assert vsock["guest_cid"] == 5

    def test_to_dict(self):
        """Test dictionary conversion."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig(vcpu_count=4)
        data = config.to_dict()

        assert data["vcpu_count"] == 4
        assert "mem_size_mib" in data
        assert "timeout_seconds" in data

    def test_from_dict(self):
        """Test creating from dictionary."""
        from runtime.firecracker.config import FirecrackerConfig

        data = {
            "vcpu_count": 2,
            "mem_size_mib": 256,
            "enable_network": True,
        }

        config = FirecrackerConfig.from_dict(data)
        assert config.vcpu_count == 2
        assert config.mem_size_mib == 256
        assert config.enable_network is True

    def test_minimal_preset(self):
        """Test minimal configuration preset."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig.minimal()

        assert config.vcpu_count == 1
        assert config.mem_size_mib == 64
        assert config.timeout_seconds == 10

    def test_code_execution_preset(self):
        """Test code execution preset."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig.for_code_execution()

        assert config.vcpu_count == 1
        assert config.mem_size_mib == 256
        assert config.enable_network is False

    def test_network_tasks_preset(self):
        """Test network tasks preset."""
        from runtime.firecracker.config import FirecrackerConfig

        config = FirecrackerConfig.for_network_tasks()

        assert config.vcpu_count == 2
        assert config.enable_network is True


class TestVMState:
    """Tests for VMState enum."""

    def test_all_states(self):
        """Test all VM states exist."""
        from runtime.firecracker.config import VMState

        assert VMState.CREATED.value == "created"
        assert VMState.STARTING.value == "starting"
        assert VMState.RUNNING.value == "running"
        assert VMState.PAUSED.value == "paused"
        assert VMState.STOPPING.value == "stopping"
        assert VMState.STOPPED.value == "stopped"
        assert VMState.ERROR.value == "error"


class TestVMResourceLimits:
    """Tests for VMResourceLimits."""

    def test_default_limits(self):
        """Test default resource limits."""
        from runtime.firecracker.config import DEFAULT_RESOURCE_LIMITS

        assert DEFAULT_RESOURCE_LIMITS.max_memory_mib == 1024
        assert DEFAULT_RESOURCE_LIMITS.max_vcpus == 4
        assert DEFAULT_RESOURCE_LIMITS.max_concurrent_vms == 10
