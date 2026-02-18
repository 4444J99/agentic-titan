# Contributing to agentic-titan

Thank you for your interest in contributing to this project.

## Overview

Agentic Titan is a Python-based AI agent framework for LLM orchestration. It provides abstractions for building autonomous agents that can coordinate tasks, manage context, and interact with external tools and APIs.

**Stack:** Python (AI agent framework, LLM orchestration)

## Prerequisites

- Git
- Python 3.11+
- A GitHub account

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/agentic-titan.git
cd agentic-titan

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Use the provided issue templates when available
- Include clear reproduction steps for bugs
- For documentation issues, specify which file and section

### Making Changes

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a branch** for your change:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. **Make your changes** following the code style guidelines below
5. **Test** your changes:
   ```bash
   pytest -v
   ```
6. **Commit** with a clear, imperative-mood message:
   ```bash
   git commit -m "add validation for edge case in parser"
   ```
7. **Push** your branch and open a Pull Request

### Code Style

Follow PEP 8. Use type hints for all function signatures. Prefer dataclasses or Pydantic for data structures. Write docstrings for public APIs. Agent interfaces should be well-documented with usage examples.

### Commit Messages

- Use imperative mood: "add feature" not "added feature"
- Keep the title under 72 characters
- Reference issue numbers when applicable: "fix auth bug (#42)"
- Keep commits atomic and focused on a single change

## Pull Request Process

1. Fill out the PR template with a description of your changes
2. Reference any related issues
3. Ensure all CI checks pass
4. Request review from a maintainer
5. Address review feedback promptly

PRs should be focused — one feature or fix per PR. Large changes should be
discussed in an issue first.

## Code of Conduct

Be respectful, constructive, and honest. This project is part of the
[organvm system](https://github.com/organvm-iv-taxis), which values transparency
and building in public. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/).

## Questions?

Open an issue on this repository or start a discussion if discussions are
enabled. For system-wide questions, see
[orchestration-start-here](https://github.com/organvm-iv-taxis/orchestration-start-here).
