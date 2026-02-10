# Deploy Smoke Runbook

## Purpose
Provide a deterministic smoke verification path after deployment changes.

## Preconditions
1. Lint/type/test blocking gates are green.
2. Required secrets are configured for target environment.
3. Redis and ChromaDB are reachable in target environment.

## Docker Compose Smoke
1. Start stack:
```bash
docker compose -f deploy/compose.yaml up -d
```
2. Check container health:
```bash
docker compose -f deploy/compose.yaml ps
```
3. Verify API liveness/readiness:
```bash
curl -sf http://localhost:8000/health
curl -sf http://localhost:8000/ready
```
4. Verify metrics endpoint:
```bash
curl -sf http://localhost:8000/metrics | head -n 20
```
5. Verify basic CLI operation:
```bash
.venv/bin/titan status
```

## Kubernetes / K3s Smoke
1. Apply manifests:
```bash
kubectl apply -k deploy/k3s/
```
2. Wait for rollout:
```bash
kubectl -n titan rollout status deploy/titan-api
kubectl -n titan rollout status deploy/titan-agent
```
3. Verify probes and pods:
```bash
kubectl -n titan get pods
kubectl -n titan get events --sort-by=.metadata.creationTimestamp | tail -n 50
```
4. Port-forward and probe:
```bash
kubectl -n titan port-forward deploy/titan-api 8000:8000
curl -sf http://localhost:8000/health
curl -sf http://localhost:8000/ready
```

## Artifact Capture
Save command outputs to `.ci/` for traceability:
1. `.ci/deploy_smoke_compose.txt`
2. `.ci/deploy_smoke_k3s.txt`
3. `.ci/deploy_smoke_metrics_sample.txt`

## Stop/Go Criteria
Go:
- All smoke commands pass with HTTP 200 on `/health` and `/ready`.
- No crashloop/restart patterns in deployment logs.

Stop:
- Probe failures, repeated restarts, or dependency connectivity failures.

## Rollback Triggers
Rollback immediately if:
1. API readiness fails for more than 10 minutes.
2. Error rate spikes above baseline and does not recover.
3. Critical paths (`/health`, `/ready`, `/metrics`) fail repeatedly.
