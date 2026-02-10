# Release Closure Checklist

## Quality Gates
- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy --ignore-missing-imports hive agents titan mcp dashboard`
- [ ] `pytest tests/ -q`
- [ ] `pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"`

## Governance Gates
- [ ] `python .ci/check_core_boundary_manifest.py`
- [ ] `python .ci/check_mypy_quarantine.py`
- [ ] `.ci/completion_status.md` updated with current date and counts

## Security and Dependency Gates
- [ ] CI `security` job green
- [ ] CI `dependency-integrity` job green
- [ ] New `allow-secret` usages reviewed and justified

## Deployment Evidence
- [ ] Compose smoke run captured
- [ ] K3s/Helm smoke run captured (when applicable)
- [ ] `.ci/` artifact links included in release notes

## Decision Record
- [ ] Known risks reviewed against `plans/evaluation-to-growth/risk-register.md`
- [ ] Omega status explicitly declared (`COMPLETE` or `NOT COMPLETE`)
- [ ] Release approver signoff captured
