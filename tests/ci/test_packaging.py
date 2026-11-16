"""Tests for the distribution/packaging contract (issue #93 activation audit).

The package output is defined in pyproject.toml (hatchling, console script
``titan``) and built on every green CI run by the ``package`` job in
``.github/workflows/ci.yml``. These tests guard the contract hermetically:
no network, no build, no installs.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_metadata_is_publishable() -> None:
    project = _pyproject()["project"]
    assert project["name"] == "agentic-titan"
    assert project["version"]
    assert project["readme"] == "README.md"
    assert (ROOT / "README.md").exists()
    assert (ROOT / "LICENSE").exists()


def test_console_script_resolves_to_callable() -> None:
    scripts = _pyproject()["project"]["scripts"]
    assert "titan" in scripts
    module_name, attr = scripts["titan"].split(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr))


def test_wheel_package_dirs_all_exist() -> None:
    packages = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "titan" in packages
    missing = [
        name
        for name in packages
        if not (ROOT / name).is_dir() or not (ROOT / name / "__init__.py").exists()
    ]
    assert not missing, f"wheel package dirs missing or not packages: {missing}"


def test_readme_documents_package_artifacts_and_smoke_check() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "agentic-titan-dist" in readme
    assert "titan --help" in readme


def test_changelog_tracks_activation_work() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
