# Agentic Titan Completion Program

## Objective
Drive the repository from partial hardening to full project completion with explicit stop/go gates per tranche.

## Operating Model
- Every tranche has:
  - `Goal`: What must be achieved.
  - `Work`: The implementation scope.
  - `Stop/Go Gate`: Commands that must pass before proceeding.
  - `Exit Criteria`: Objective completion condition.
- If any gate fails, the program stops in that tranche and creates a concrete remediation backlog before retry.

## Tranche 0: Environment Reproducibility
- Goal:
  - Re-establish deterministic local execution parity with CI expectations.
- Work:
  - Recreate `.venv`.
  - Install project with required extras and test/lint/type tools.
  - Verify `python`, `pytest`, `ruff`, and `mypy` resolve from `.venv`.
- Stop/Go Gate:
  - `.venv/bin/python --version`
  - `.venv/bin/pytest --version`
  - `.venv/bin/ruff --version`
  - `.venv/bin/mypy --version`
- Exit Criteria:
  - All four commands pass and `.venv/bin/pytest` no longer references stale interpreter paths.

## Tranche 1: Baseline Snapshot (Non-Blocking Debt Inventory)
- Goal:
  - Produce a trustworthy full-repo quality baseline.
- Work:
  - Run full-repo advisory lint, typecheck, and tests in CI-like mode.
  - Capture outputs to artifacts for deterministic tracking.
- Stop/Go Gate:
  - `.venv/bin/ruff check . > .ci/baseline_ruff.txt || true`
  - `.venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard > .ci/baseline_mypy.txt || true`
  - `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q > .ci/baseline_pytest.txt || true`
- Exit Criteria:
  - Three baseline files exist and are non-empty with current error/failure counts.

## Tranche 2: Full-Lint Ratchet to Blocking
- Goal:
  - Reduce full-repo ruff debt to zero and convert full lint job to blocking.
- Work:
  - Fix ruff categories in priority order: import order/unused imports, line length, pyupgrade, enum modernization, timeout aliasing.
  - Keep boundary-scope lint green throughout.
- Stop/Go Gate:
  - `.venv/bin/ruff check .`
  - `.venv/bin/ruff format --check .`
  - CI workflow contains no `continue-on-error` for full lint.
- Exit Criteria:
  - Full lint clean locally and in CI with blocking enforcement.

## Tranche 3: Full-Typecheck Ratchet to Blocking
- Goal:
  - Reduce full-repo mypy debt to zero or documented narrow excludes, then block on it.
- Work:
  - Fix by module clusters: `titan/api` and dependencies, `titan/batch`, `hive`, `agents`, `mcp`, `dashboard`.
  - Replace file-level ignores with typed wrappers/protocols where possible.
  - Introduce narrowly scoped config suppressions only with issue references and expiration notes.
- Stop/Go Gate:
  - `.venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard`
  - CI workflow contains no `continue-on-error` for full typecheck.
- Exit Criteria:
  - Full typecheck is blocking and green.

## Tranche 4: Runtime Test Completion
- Goal:
  - Make full suite reliable and green in CI-like environment.
- Work:
  - Triage non-collection test failures.
  - Stabilize infra-dependent tests with deterministic fixtures/guards.
  - Preserve warning-hardening pass.
- Stop/Go Gate:
  - `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q`
  - `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"`
- Exit Criteria:
  - Both full-suite commands pass consistently.

## Tranche 5: Security, Deploy, and Ops Completion
- Goal:
  - Verify production-readiness controls and operational completeness.
- Work:
  - Validate secret scanning and dependency integrity.
  - Execute deploy smoke checks from `deploy/`.
  - Confirm runbooks for rollback and incident handling are current.
- Stop/Go Gate:
  - CI security job pass.
  - Dependency-integrity and core-boundary-governance pass.
  - At least one documented deploy smoke run in repo artifacts.
- Exit Criteria:
  - Production operations checklist is complete and validated.

## Tranche 6: Documentation and Release Closure
- Goal:
  - Align docs with actual behavior and publish a release-quality status.
- Work:
  - Update README quickstart and path assumptions.
  - Add completion report and known constraints.
  - Create release notes/changelog entry.
- Stop/Go Gate:
  - Documentation lint/consistency checks pass (if configured).
  - Final completion report reviewed and committed.
- Exit Criteria:
  - Repo can be onboarded and operated from docs alone.

## Definition of Done (Omega)
Project is complete only when all tranches have passed stop/go gates and there are no open blocker-severity issues in lint, typecheck, tests, deploy, or security.
