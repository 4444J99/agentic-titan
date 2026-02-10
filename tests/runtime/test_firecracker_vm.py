"""
Tests for Firecracker MicroVM Manager (Phase 18B)
"""

import pytest


class TestMicroVM:
    """Tests for MicroVM dataclass."""

    def test_import(self):
        """Test that MicroVM can be imported."""
        from runtime.firecracker.vm import ExecutionResult, MicroVM

        assert MicroVM is not None
        assert ExecutionResult is not None

    def test_create_vm(self):
        """Test creating a MicroVM instance."""
        from runtime.firecracker.config import FirecrackerConfig, VMState
        from runtime.firecracker.vm import MicroVM

        config = FirecrackerConfig()
        vm = MicroVM(config=config)

        assert vm.vm_id is not None
        assert vm.state == VMState.CREATED
        assert vm.process is None
        assert vm.socket_path != ""

    def test_vm_to_dict(self):
        """Test VM dictionary conversion."""
        from runtime.firecracker.vm import MicroVM

        vm = MicroVM()
        data = vm.to_dict()

        assert "vm_id" in data
        assert "state" in data
        assert "created_at" in data


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_create_result(self):
        """Test creating an execution result."""
        from runtime.firecracker.vm import ExecutionResult

        result = ExecutionResult(
            exit_code=0,
            stdout="Hello, World!",
            stderr="",
            duration_ms=100,
        )

        assert result.exit_code == 0
        assert result.stdout == "Hello, World!"
        assert result.duration_ms == 100

    def test_timeout_result(self):
        """Test creating a timeout result."""
        from runtime.firecracker.vm import ExecutionResult

        result = ExecutionResult(
            exit_code=-1,
            timed_out=True,
            error="Command timed out",
        )

        assert result.timed_out is True
        assert result.exit_code == -1

    def test_to_dict(self):
        """Test result dictionary conversion."""
        from runtime.firecracker.vm import ExecutionResult

        result = ExecutionResult(exit_code=0, stdout="test")
        data = result.to_dict()

        assert data["exit_code"] == 0
        assert data["stdout"] == "test"


class TestMicroVMManager:
    """Tests for MicroVMManager."""

    def test_create_manager(self):
        """Test creating a VM manager."""
        from runtime.firecracker.config import FirecrackerConfig
        from runtime.firecracker.vm import MicroVMManager

        config = FirecrackerConfig()
        manager = MicroVMManager(default_config=config)

        assert manager is not None

    def test_create_manager_with_pool(self):
        """Test creating a manager with pool size."""
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager(pool_size=5)

        assert manager._pool_size == 5

    @pytest.mark.asyncio
    async def test_create_vm(self):
        """Test creating a VM through manager."""
        from runtime.firecracker.config import VMState
        from runtime.firecracker.vm import MicroVM, MicroVMManager

        manager = MicroVMManager()
        vm = await manager.create()

        assert isinstance(vm, MicroVM)
        assert vm.state == VMState.CREATED
        assert vm.vm_id in manager._vms

    @pytest.mark.asyncio
    async def test_list_vms(self):
        """Test listing VMs."""
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager()
        await manager.create()
        await manager.create()

        vms = await manager.list_vms()
        assert len(vms) == 2

    @pytest.mark.asyncio
    async def test_start_vm_missing_binary(self):
        """Test starting VM when Firecracker binary is missing."""
        from runtime.firecracker.config import FirecrackerConfig
        from runtime.firecracker.vm import MicroVMManager

        config = FirecrackerConfig(firecracker_path="/nonexistent/firecracker")
        manager = MicroVMManager(default_config=config)
        vm = await manager.create()

        with pytest.raises(FileNotFoundError):
            await manager.start(vm)

    @pytest.mark.asyncio
    async def test_stop_vm(self):
        """Test stopping a running VM."""
        from runtime.firecracker.config import VMState
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager()
        vm = await manager.create()

        # Simulate a running VM (can't actually start without Firecracker)
        vm.state = VMState.RUNNING

        await manager.stop(vm)

        assert vm.state == VMState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_nonrunning_vm(self):
        """Test stopping a VM that's not running."""
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager()
        vm = await manager.create()

        # Should not raise
        await manager.stop(vm)

    @pytest.mark.asyncio
    async def test_execute_on_stopped_vm(self):
        """Test executing on a stopped VM."""
        from runtime.firecracker.config import VMState
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager()
        vm = await manager.create()
        vm.state = VMState.STOPPED

        result = await manager.execute(vm, "echo test")

        assert result.exit_code == -1
        assert "not running" in result.error.lower()


class TestVMManagerPool:
    """Tests for VM pool functionality."""

    @pytest.mark.asyncio
    async def test_return_to_pool(self):
        """Test returning VM to pool."""
        from runtime.firecracker.config import VMState
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager(pool_size=2)
        vm = await manager.create()
        vm.state = VMState.RUNNING

        await manager.return_to_pool(vm)

        assert len(manager._pool) == 1

    @pytest.mark.asyncio
    async def test_pool_full(self):
        """Test returning VM when pool is full."""
        from runtime.firecracker.config import VMState
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager(pool_size=1)

        vm1 = await manager.create()
        vm1.state = VMState.RUNNING
        await manager.return_to_pool(vm1)

        vm2 = await manager.create()
        vm2.state = VMState.RUNNING
        # Pool is full, should stop vm2
        await manager.return_to_pool(vm2)

        assert len(manager._pool) == 1

    @pytest.mark.asyncio
    async def test_shutdown_clears_pool(self):
        """Test that shutdown clears the pool."""
        from runtime.firecracker.config import VMState
        from runtime.firecracker.vm import MicroVMManager

        manager = MicroVMManager(pool_size=2)

        vm = await manager.create()
        vm.state = VMState.RUNNING
        await manager.return_to_pool(vm)

        await manager.shutdown()

        assert len(manager._pool) == 0


class TestGetVMManager:
    """Tests for factory function."""

    def test_get_vm_manager(self):
        """Test getting VM manager."""
        from runtime.firecracker.vm import MicroVMManager, get_vm_manager

        manager = get_vm_manager()

        assert manager is not None
        assert isinstance(manager, MicroVMManager)
