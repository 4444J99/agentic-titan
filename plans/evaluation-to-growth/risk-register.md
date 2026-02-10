# Evaluation-to-Growth Risk Register

| ID | Risk | Severity | Likelihood | Attack Vector / Failure Mode | Mitigation | Contingency |
|---|---|---|---|---|---|---|
| R1 | Completion claim drift from config reality | High | Medium | Status says complete while quarantine/debt remains | CI governance check (`.ci/check_mypy_quarantine.py`) + status semantics | Block merges until status/config parity restored |
| R2 | Mypy quarantine ossifies | High | High | `ignore_errors` list remains static and accepted as normal | Tranche 3B with explicit zero target and count tracking | Freeze scope expansion until quarantine decreases |
| R3 | Advisory debt reaccumulation | Medium | High | Advisory jobs fail repeatedly without ownership | Add owner rotation and weekly review cadence | Promote high-drift advisory checks to blocking |
| R4 | Secret bypass annotation abuse (`allow-secret`) | Medium | Medium | False positives bypassed without periodic review | Add review policy and periodic scanner report | Temporarily fail CI on unreviewed new annotations |
| R5 | CI/local environment skew | Medium | Medium | Local green differs from CI matrix behavior | Keep matrix test job and command parity docs | Add scheduled clean-room gate replay |
| R6 | Documentation confidence gap | Medium | Medium | README/program docs imply stronger guarantees than enforced | Keep docs tied to CI-governed semantics | Add release evidence bundle with command outputs |

## Current Top Risks
1. `R2` - Mypy quarantine ossification.
2. `R1` - Claim drift.
3. `R3` - Advisory drift.

## Immediate Next Mitigations
1. Start quarantine burn-down in Tranche 3B with per-module removal batches.
2. Add weekly governance-audit CI schedule.
3. Add owner rubric for advisory failure triage.
