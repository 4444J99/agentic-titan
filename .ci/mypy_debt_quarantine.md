# Mypy Debt Quarantine

This file tracks the now-completed mypy quarantine burn-down.

## Purpose
- Preserve an auditable record of the quarantine removal program.
- Confirm that full-repo mypy now runs without `ignore_errors` quarantine.

## Current Quarantine Scope
- Source of truth: `[[tool.mypy.overrides]]` block in `pyproject.toml`.
- Current size: 0 modules.

## Closure Criteria (Met)
1. Quarantine override module count reached zero.
2. Full-repo mypy passed:
   - `.venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard`
3. Core-scope mypy passed:
   - `xargs .venv/bin/mypy --ignore-missing-imports --follow-imports=skip < .ci/typecheck_core_targets.txt`

## Guardrail
- Do not reintroduce an `ignore_errors=true` quarantine override for project modules.
