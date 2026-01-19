# Agentic Titan Deployment

Production deployment configurations for the Titan agent swarm.

## Quick Start

### Docker Compose (Development)

```bash
# Start all services
docker compose up -d

# With monitoring
docker compose --profile monitoring up -d

# View logs
docker compose logs -f
```

### Kubernetes / K3s

```bash
# Apply manifests
kubectl apply -k deploy/k3s/

# Or with Helm
helm install titan deploy/helm/titan/
```

## Directory Structure

```
deploy/
├── compose.yaml         # Docker Compose for local development
├── Dockerfile.api       # API server image
├── Dockerfile.agent     # Agent worker image
├── k3s/                 # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── redis.yaml
│   ├── chromadb.yaml
│   ├── api-deployment.yaml
│   ├── agent-deployment.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   └── kustomization.yaml
├── helm/                # Helm chart
│   └── titan/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── grafana/             # Grafana dashboards
└── prometheus.yml       # Prometheus config
```

## Deployment Options

### 1. Docker Compose (Development)

Best for local development and testing.

```bash
cd deploy
docker compose up -d

# Services available at:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### 2. Kubernetes with Kustomize

Direct Kubernetes manifests using Kustomize.

```bash
# Preview
kubectl kustomize deploy/k3s/

# Apply
kubectl apply -k deploy/k3s/

# Check status
kubectl -n titan get pods
```

### 3. Helm Chart

Templated, configurable deployment.

```bash
# Install
helm install titan deploy/helm/titan/

# With custom values
helm install titan deploy/helm/titan/ -f my-values.yaml

# Upgrade
helm upgrade titan deploy/helm/titan/

# Uninstall
helm uninstall titan
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_DEFAULT_PROVIDER` | Default LLM provider | `ollama` |
| `OLLAMA_HOST` | Ollama server URL | `http://ollama:11434` |
| `REDIS_HOST` | Redis host | `redis` |
| `CHROMADB_HOST` | ChromaDB host | `chromadb` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `METRICS_ENABLED` | Enable metrics | `true` |

### Secrets

For production, use a secrets manager:

```yaml
# External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: titan-secrets
spec:
  secretStoreRef:
    name: vault
    kind: ClusterSecretStore
  target:
    name: titan-secrets
  data:
    - secretKey: ANTHROPIC_API_KEY
      remoteRef:
        key: secret/titan
        property: anthropic_api_key
```

## Scaling

### Horizontal Pod Autoscaler

The HPA is configured to scale based on CPU and memory:

- **API**: 2-10 replicas, scales at 70% CPU / 80% memory
- **Agent**: 3-50 replicas, scales at 60% CPU / 70% memory

Agent scaling is more aggressive with:
- 10 minute stabilization before scale down
- Instant scale up (30s stabilization)
- Up to 200% increase per scale event

### Manual Scaling

```bash
# Scale agents
kubectl -n titan scale deployment titan-agent --replicas=10

# Scale API
kubectl -n titan scale deployment titan-api --replicas=5
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` on port 8080:

- `titan_agent_tasks_total` - Total tasks processed
- `titan_agent_tasks_active` - Currently running tasks
- `titan_agent_task_duration_seconds` - Task duration histogram
- `titan_llm_calls_total` - LLM API calls
- `titan_tool_calls_total` - Tool invocations

### Grafana Dashboards

Import dashboards from `grafana/provisioning/dashboards/`:

1. Titan Overview - System health
2. Agent Performance - Task metrics
3. LLM Usage - Provider metrics

## Health Checks

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness probe |
| `/ready` | Readiness probe |
| `/metrics` | Prometheus metrics |

### Probes Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /ready
    port: http
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Troubleshooting

### Common Issues

**Pods not starting:**
```bash
kubectl -n titan describe pod <pod-name>
kubectl -n titan logs <pod-name>
```

**HPA not scaling:**
```bash
kubectl -n titan describe hpa titan-agent-hpa
kubectl top pods -n titan
```

**Redis connection issues:**
```bash
kubectl -n titan exec -it redis-0 -- redis-cli ping
```

**ChromaDB issues:**
```bash
kubectl -n titan logs chromadb-0
curl http://chromadb:8000/api/v2/heartbeat
```

## Security

### Network Policies

Consider adding network policies to restrict pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: titan-network-policy
  namespace: titan
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: titan
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: titan
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: titan
```

### Pod Security Standards

The deployment uses restricted security context:
- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Dropped capabilities
