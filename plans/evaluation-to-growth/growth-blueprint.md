# Evaluation-to-Growth Blueprint

## Horizon 1 (0-2 weeks): Governance Stabilization
Objective:
- Ensure every completion claim is machine-verifiable.

Actions:
1. Enforce quarantine/status consistency in CI (completed).
2. Keep Tranche 3B visible in status/program docs (completed).
3. Re-run full gate sequence and refresh status artifacts.

Success signals:
- No status/config parity failures.
- Gate sequence remains green after governance changes.

## Horizon 2 (2-6 weeks): Quarantine Burn-Down
Objective:
- Reduce mypy quarantine count steadily to zero.

Actions:
1. Burn down top-risk modules in batches of 3-5.
2. Remove fixed modules from `pyproject.toml` quarantine list each ratchet.
3. Validate with:
   - `python .ci/check_mypy_quarantine.py`
   - full mypy command
   - full pytest command set

Success signals:
- Quarantine count monotonically decreasing.
- No regression in blocking gates.

## Horizon 3 (6-12 weeks): Omega Operationalization
Objective:
- Shift from one-time completion to continuous completion discipline.

Actions:
1. Add weekly scheduled governance audit workflow.
2. Introduce quality SLOs:
   - time-to-green
   - advisory failure age
   - quarantine delta per release
3. Create release evidence bundle:
   - gate outputs
   - completion status snapshot
   - risk register delta

Success signals:
- Zero quarantine.
- Full-repo typecheck truly blocking.
- Repeatable release closure with evidence artifacts.

## Novel Expansion Tracks
1. Contractized status files:
   - Treat status documents as CI-validated control-plane artifacts.
2. Ratchet recommender:
   - Automatically suggest next modules using error density + blast radius.
3. Governance dashboard:
   - Visualize quarantine trend, gate pass rates, and advisory drift.
