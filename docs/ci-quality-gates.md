# CI Quality Gates

This repository uses a staged quality-gate model:

1. Blocking jobs protect the core import boundary and test reliability.
2. Advisory jobs keep full-repo debt visible while we ratchet enforcement.

## Blocking Jobs

1. `dependency-integrity`
- Verifies base install and import integrity.

2. `core-boundary-governance`
- Validates `.ci/core_import_boundary_files.txt` against scoped directories in
  `.ci/core_import_boundary_directories.txt`.

3. `import-boundary-tests`
- Runs `tests/integration/test_import_boundaries.py`.

4. `mypy-quarantine-governance`
- Validates that `.ci/completion_status.md` truthfully reports mypy quarantine
  state from `pyproject.toml`.

5. `lint`
- Runs `ruff check` and `ruff format --check` on files listed in
  `.ci/core_import_boundary_files.txt`.

6. `typecheck-core`
- Runs mypy only on `.ci/typecheck_core_targets.txt` with
  `--follow-imports=skip`.

7. `test-core-blocking`
- Runs `pytest tests/ -q` in a CI-like environment with Redis.

## Advisory Jobs

1. `lint-full-advisory`
- Full-repo Ruff check and format checks.

2. `typecheck-full-advisory`
- Full-repo mypy for broad visibility while debt is burned down.

## Ratchet Policy

1. Expand blocking lint/typecheck scopes only when current scope is green.
2. Land scope expansion and cleanup in the same PR.
3. Keep advisory jobs enabled until blocking scopes cover agreed core surfaces.

## Mypy Quarantine Semantics

1. A green full-repo mypy command is only considered "true blocking complete"
   when quarantine module count is zero.
2. If `[[tool.mypy.overrides]]` has `ignore_errors=true` modules, Tranche 3 is
   considered `PARTIAL` even if command output is green.
3. `.ci/check_mypy_quarantine.py` enforces status-file consistency with the
   current quarantine count.

## How To Add A New Core Boundary File

1. Add the file path to `.ci/core_import_boundary_files.txt` (keep sorted).
2. If the file is in a new core directory, add directory path to
   `.ci/core_import_boundary_directories.txt`.
3. Run local checks:
- `ruff check $(cat .ci/core_import_boundary_files.txt)`
- `ruff format --check $(cat .ci/core_import_boundary_files.txt)`
- `python .ci/check_core_boundary_manifest.py`
