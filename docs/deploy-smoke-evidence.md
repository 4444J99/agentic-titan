# Deploy Smoke Evidence (2026-02-10)

## Scope
Evidence capture for deployment smoke verification defined in
`docs/deploy-smoke-runbook.md`.

## Artifacts
1. Compose smoke: `.ci/deploy_smoke_compose.txt`
2. K3s smoke: `.ci/deploy_smoke_k3s.txt`
3. Metrics sample: `.ci/deploy_smoke_metrics_sample.txt`

## Results
1. Docker Compose smoke: `GO`
- Smoke run executed in isolated compose project `titan_smoke`.
- API liveness and dashboard checks passed:
  - `GET /api/status` -> HTTP 200
  - `GET /` -> HTTP 200
- Metrics sample captured from `GET /api/metrics`.

2. K3s dry-run smoke: `GO`
- Local control plane reachable via `kubectl cluster-info`.
- `kubectl kustomize deploy/k3s/` render succeeded.
- `kubectl apply -k deploy/k3s/ --dry-run=client` validated all base resources.
- Traefik `Middleware` CRD is absent locally; optional middleware add-on is
  explicitly skipped without blocking base smoke.

## Operational Decision
1. Deployment evidence is captured and reproducible.
2. Tranche-5 deploy smoke gates are satisfied for Omega closure criteria.
