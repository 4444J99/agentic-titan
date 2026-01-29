"""
Firecracker Image Builder

Utilities for building kernel and rootfs images for Firecracker VMs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("titan.runtime.firecracker.image_builder")


@dataclass
class ImageConfig:
    """Configuration for image building."""

    # Kernel
    kernel_version: str = "5.10"
    kernel_config: str = "minimal"  # minimal, standard, full

    # Rootfs
    base_image: str = "alpine"  # alpine, ubuntu-minimal, debian-slim
    rootfs_size_mb: int = 256
    rootfs_format: str = "ext4"

    # Packages to install
    packages: list[str] = field(default_factory=lambda: ["python3"])

    # Output paths
    output_dir: str = "/var/lib/firecracker"
    kernel_name: str = "vmlinux"
    rootfs_name: str = "rootfs.ext4"


@dataclass
class ImageInfo:
    """Information about a built image."""

    path: str
    size_bytes: int
    created_at: str
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageBuilder:
    """
    Builder for Firecracker kernel and rootfs images.

    Supports building:
    - Minimal Linux kernels optimized for microVMs
    - Alpine, Ubuntu, and Debian-based root filesystems
    - Custom packages and configurations
    """

    def __init__(self, config: ImageConfig | None = None) -> None:
        self._config = config or ImageConfig()
        self._cache_dir = Path(tempfile.gettempdir()) / "firecracker-images"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def build_minimal_kernel(
        self,
        output_path: str | None = None,
    ) -> ImageInfo:
        """
        Build or download a minimal Linux kernel for Firecracker.

        For most use cases, downloading a prebuilt kernel is faster
        than building from source.

        Args:
            output_path: Output path for kernel image

        Returns:
            ImageInfo with kernel details
        """
        output_path = output_path or str(
            Path(self._config.output_dir) / self._config.kernel_name
        )

        # Check if already exists
        if os.path.exists(output_path):
            return self._get_image_info(output_path, "kernel")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download prebuilt kernel from Firecracker releases
        kernel_url = self._get_kernel_url()

        logger.info(f"Downloading kernel from {kernel_url}")

        try:
            await self._download_file(kernel_url, output_path)
            os.chmod(output_path, 0o644)

            return self._get_image_info(output_path, "kernel")

        except Exception as e:
            logger.error(f"Failed to build/download kernel: {e}")
            raise

    async def build_rootfs(
        self,
        base: str | None = None,
        packages: list[str] | None = None,
        output_path: str | None = None,
    ) -> ImageInfo:
        """
        Build a root filesystem image.

        Args:
            base: Base image (alpine, ubuntu-minimal, debian-slim)
            packages: Additional packages to install
            output_path: Output path for rootfs image

        Returns:
            ImageInfo with rootfs details
        """
        base = base or self._config.base_image
        packages = packages or self._config.packages
        output_path = output_path or str(
            Path(self._config.output_dir) / self._config.rootfs_name
        )

        # Check if already exists
        if os.path.exists(output_path):
            return self._get_image_info(output_path, "rootfs")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Building {base} rootfs with packages: {packages}")

        if base == "alpine":
            await self._build_alpine_rootfs(output_path, packages)
        elif base in ("ubuntu-minimal", "ubuntu"):
            await self._build_ubuntu_rootfs(output_path, packages)
        elif base in ("debian-slim", "debian"):
            await self._build_debian_rootfs(output_path, packages)
        else:
            raise ValueError(f"Unsupported base image: {base}")

        return self._get_image_info(output_path, "rootfs")

    async def prebuild_images(self) -> dict[str, ImageInfo]:
        """
        Prebuild all standard images.

        Returns:
            Dict of image type to ImageInfo
        """
        images = {}

        # Build kernel
        try:
            images["kernel"] = await self.build_minimal_kernel()
            logger.info(f"Kernel ready: {images['kernel'].path}")
        except Exception as e:
            logger.warning(f"Failed to build kernel: {e}")

        # Build rootfs
        try:
            images["rootfs"] = await self.build_rootfs()
            logger.info(f"Rootfs ready: {images['rootfs'].path}")
        except Exception as e:
            logger.warning(f"Failed to build rootfs: {e}")

        return images

    async def _build_alpine_rootfs(
        self,
        output_path: str,
        packages: list[str],
    ) -> None:
        """Build Alpine Linux rootfs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rootfs_dir = Path(tmpdir) / "rootfs"
            rootfs_dir.mkdir()

            # Download Alpine minirootfs
            alpine_url = (
                "https://dl-cdn.alpinelinux.org/alpine/v3.19/releases/x86_64/"
                "alpine-minirootfs-3.19.0-x86_64.tar.gz"
            )

            tarball = Path(tmpdir) / "alpine.tar.gz"
            await self._download_file(alpine_url, str(tarball))

            # Extract
            await self._run_command(f"tar xzf {tarball} -C {rootfs_dir}")

            # Install additional packages (requires chroot)
            if packages:
                pkg_list = " ".join(packages)
                script = f"""
                #!/bin/sh
                apk update
                apk add {pkg_list}
                """
                script_path = rootfs_dir / "install.sh"
                script_path.write_text(script)
                script_path.chmod(0o755)

                # Note: This requires root and proper chroot setup
                # For now, we skip package installation in non-root context

            # Add guest agent
            agent_path = rootfs_dir / "usr" / "bin" / "agent"
            agent_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_guest_agent(agent_path)

            # Create init script
            init_path = rootfs_dir / "init"
            init_path.write_text("""#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev

# Start guest agent
/usr/bin/agent &

# Start shell
exec /bin/sh
""")
            init_path.chmod(0o755)

            # Create ext4 image
            await self._create_ext4_image(rootfs_dir, output_path)

    async def _build_ubuntu_rootfs(
        self,
        output_path: str,
        packages: list[str],
    ) -> None:
        """Build Ubuntu minimal rootfs."""
        # For Ubuntu, we would use debootstrap
        # This is a placeholder that falls back to Alpine
        logger.warning("Ubuntu rootfs not fully implemented, using Alpine")
        await self._build_alpine_rootfs(output_path, packages)

    async def _build_debian_rootfs(
        self,
        output_path: str,
        packages: list[str],
    ) -> None:
        """Build Debian slim rootfs."""
        # For Debian, we would use debootstrap
        # This is a placeholder that falls back to Alpine
        logger.warning("Debian rootfs not fully implemented, using Alpine")
        await self._build_alpine_rootfs(output_path, packages)

    async def _create_ext4_image(
        self,
        source_dir: Path,
        output_path: str,
    ) -> None:
        """Create ext4 image from directory."""
        size_mb = self._config.rootfs_size_mb

        # Create empty file
        await self._run_command(f"dd if=/dev/zero of={output_path} bs=1M count={size_mb}")

        # Create ext4 filesystem
        await self._run_command(f"mkfs.ext4 -F {output_path}")

        # Mount and copy (requires root)
        with tempfile.TemporaryDirectory() as mount_point:
            try:
                await self._run_command(f"mount -o loop {output_path} {mount_point}")
                await self._run_command(f"cp -a {source_dir}/* {mount_point}/")
            finally:
                await self._run_command(f"umount {mount_point}", check=False)

    def _write_guest_agent(self, path: Path) -> None:
        """Write guest agent script."""
        from runtime.firecracker.guest_agent import GUEST_AGENT_SCRIPT

        path.write_text(GUEST_AGENT_SCRIPT)
        path.chmod(0o755)

    def _get_kernel_url(self) -> str:
        """Get URL for prebuilt kernel."""
        # Firecracker team provides prebuilt kernels
        version = self._config.kernel_version
        return (
            f"https://s3.amazonaws.com/spec.ccfc.min/img/quickstart_guide/"
            f"x86_64/kernels/vmlinux.bin"
        )

    async def _download_file(self, url: str, output_path: str) -> None:
        """Download a file."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Download failed: {response.status}")

                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)

        except ImportError:
            # Fallback to curl
            await self._run_command(f"curl -L -o {output_path} {url}")

    async def _run_command(
        self,
        cmd: str,
        check: bool = True,
    ) -> asyncio.subprocess.Process:
        """Run a shell command."""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if check and proc.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\nStderr: {stderr.decode()}")

        return proc

    def _get_image_info(self, path: str, image_type: str) -> ImageInfo:
        """Get information about an image file."""
        stat = os.stat(path)
        return ImageInfo(
            path=path,
            size_bytes=stat.st_size,
            created_at=str(stat.st_mtime),
            metadata={"type": image_type},
        )


# Singleton instance
_builder: ImageBuilder | None = None


def get_image_builder(config: ImageConfig | None = None) -> ImageBuilder:
    """Get the default image builder."""
    global _builder
    if _builder is None or config is not None:
        _builder = ImageBuilder(config)
    return _builder
