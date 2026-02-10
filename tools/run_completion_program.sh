#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRANCHE="${1:-all}"

run_tranche_0() {
  echo "[Tranche 0] Environment Reproducibility"
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install -e ".[dashboard,metrics,auth,ratelimit]"
  .venv/bin/pip install pytest pytest-asyncio ruff mypy types-redis
  .venv/bin/python --version
  .venv/bin/pytest --version
  .venv/bin/ruff --version
  .venv/bin/mypy --version
}

run_tranche_1() {
  echo "[Tranche 1] Baseline Snapshot"
  mkdir -p .ci
  set +e
  .venv/bin/ruff check . > .ci/baseline_ruff.txt 2>&1
  RUFF_EXIT=$?
  .venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard > .ci/baseline_mypy.txt 2>&1
  MYPY_EXIT=$?
  REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q > .ci/baseline_pytest.txt 2>&1
  PYTEST_EXIT=$?
  set -e

  echo "baseline exits: ruff=$RUFF_EXIT mypy=$MYPY_EXIT pytest=$PYTEST_EXIT"
  wc -l .ci/baseline_ruff.txt .ci/baseline_mypy.txt .ci/baseline_pytest.txt
}

run_tranche_2() {
  echo "[Tranche 2] Full-Lint Blocking Gate"
  .venv/bin/ruff check .
  .venv/bin/ruff format --check .
}

run_tranche_3() {
  echo "[Tranche 3] Full-Typecheck Blocking Gate"
  .venv/bin/mypy --ignore-missing-imports hive agents titan mcp dashboard
}

run_tranche_4() {
  echo "[Tranche 4] Runtime Test Completion Gate"
  REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q
  REDIS_URL=redis://localhost:6379/0 .venv/bin/pytest tests/ -q -W error::RuntimeWarning -W "error:.*on_event.*:DeprecationWarning"
}

run_tranche_5() {
  echo "[Tranche 5] Security + Governance Gate"
  python .ci/check_core_boundary_manifest.py
  .venv/bin/pytest tests/integration/test_import_boundaries.py -q
}

run_tranche_6() {
  echo "[Tranche 6] Documentation Closure Gate"
  test -f plans/completion_program.md
  test -f README.md
}

run_single() {
  case "$1" in
    0) run_tranche_0 ;;
    1) run_tranche_1 ;;
    2) run_tranche_2 ;;
    3) run_tranche_3 ;;
    4) run_tranche_4 ;;
    5) run_tranche_5 ;;
    6) run_tranche_6 ;;
    *)
      echo "unknown tranche: $1"
      exit 2
      ;;
  esac
}

if [[ "$TRANCHE" == "all" ]]; then
  for t in 0 1 2 3 4 5 6; do
    run_single "$t"
  done
else
  run_single "$TRANCHE"
fi

echo "completion program tranche(s) finished"
