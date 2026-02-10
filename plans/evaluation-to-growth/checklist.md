# Evaluation-to-Growth Checklist

## Evaluation
- [x] Repository-wide strengths and weaknesses documented.
- [x] Logic contradictions identified with concrete file evidence.
- [x] Logos/Pathos/Ethos review completed.
- [x] Priority improvement areas ranked.

## Reinforcement
- [x] Add governance check: `.ci/check_mypy_quarantine.py`.
- [x] Add CI enforcement job: `mypy-quarantine-governance`.
- [x] Correct completion semantics in `.ci/completion_status.md`.
- [x] Update quality-gates documentation with quarantine rules.
- [x] Update completion program with explicit Tranche 3B.
- [x] Fix README quickstart clone/path assumptions.

## Risk Controls
- [x] Blind spots and shatter points documented.
- [x] Preventive controls tied to CI/plan artifacts.
- [ ] Add periodic scheduled governance audit workflow (weekly).
- [ ] Add explicit owner rotation for advisory job triage.
- [ ] Add policy check/report for `allow-secret` annotation review.

## Growth Program
- [x] Growth blueprint created (near, medium, long horizon).
- [ ] Define quality SLOs and thresholds in CI/docs.
- [ ] Publish release evidence bundle template.
- [ ] Automate status-file refresh from command outputs.

## Stop/Go Gates
- [x] `python .ci/check_mypy_quarantine.py`
- [ ] `.venv/bin/ruff check .`
- [ ] `.venv/bin/ruff format --check .`
- [ ] `.venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard`
- [ ] `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q`
- [ ] `REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"`

## Omega Exit Criteria
- [ ] Mypy quarantine module count is `0`.
- [ ] Tranche 3 is `GO` without quarantine caveat.
- [ ] Completion verdict upgraded to `OMEGA COMPLETE`.
