# Agentic Titan

**Polymorphic Agent Swarm Architecture** - A model-agnostic, self-organizing multi-agent system.

```
    ╔═══════════════════════════════════════════════════════════╗
    ║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗║
    ║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝║
    ║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     ║
    ║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     ║
    ║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝║
    ║          ████████╗██╗████████╗ █████╗ ███╗   ██╗          ║
    ║          ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║          ║
    ║             ██║   ██║   ██║   ███████║██╔██╗ ██║          ║
    ║             ██║   ██║   ██║   ██╔══██║██║╚██╗██║          ║
    ║             ██║   ██║   ██║   ██║  ██║██║ ╚████║          ║
    ║             ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝          ║
    ╚═══════════════════════════════════════════════════════════╝
```

## Features

- **Model-Agnostic**: Works with Ollama, Claude, OpenAI, Groq, and local models
- **Self-Organizing Topologies**: Swarm, Hierarchy, Pipeline, Mesh, Ring, Star
- **Hive Mind**: Shared memory and real-time coordination (Redis + ChromaDB)
- **Agent Spec DSL**: Declarative YAML-based agent definitions
- **Scalable**: From 2 to 100+ agents
- **Production Ready**: Docker Compose, health checks, observability

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd /Users/4jp/agentic-titan

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Start Infrastructure

```bash
# Start Redis and ChromaDB
docker compose -f deploy/compose.yaml up -d redis chromadb

# Verify services
titan status
```

### 3. Run Your First Agent

```bash
# Initialize a project
titan init my-project
cd my-project

# Run an agent
titan run specs/researcher.titan.yaml -p "Research quantum computing applications"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC TITAN                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    🧠 HIVE MIND LAYER                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │ Vector Store │  │ Event Stream │  │ Distributed State    │   │   │
│  │  │ (ChromaDB)   │  │ (NATS/Redis) │  │ (Redis)              │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│  ┌────────────┬────────────────────┼────────────────────┬────────────┐ │
│  │  ┌─────────▼─────────┐  ┌──────▼──────┐  ┌─────────▼─────────┐   │ │
│  │  │  TOPOLOGY ENGINE  │  │ LLM ADAPTER │  │   AGENT FORGE     │   │ │
│  │  │  • Swarm          │  │ • Ollama    │  │  • Agent DSL      │   │ │
│  │  │  • Hierarchy      │  │ • Claude    │  │  • Capabilities   │   │ │
│  │  │  • Pipeline       │  │ • OpenAI    │  │  • Behaviors      │   │ │
│  │  │  • Mesh/Ring/Star │  │ • Groq      │  │  • Tool Bindings  │   │ │
│  │  └───────────────────┘  └─────────────┘  └───────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                    │
│  ┌─────────────────────────────────▼─────────────────────────────────┐ │
│  │                    🦠 AGENT SWARM (2-100+ Agents)                 │ │
│  │    [Orchestrator] [Researcher] [Coder] [Reviewer] [...]          │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Agent Spec DSL

Define agents declaratively in YAML:

```yaml
apiVersion: titan/v1
kind: Agent
metadata:
  name: researcher
  labels:
    tier: cognitive
spec:
  capabilities:
    - web_search
    - summarization

  personality:
    traits: [thorough, curious, skeptical]
    communication_style: academic

  llm:
    preferred: claude-sonnet
    fallback: [gpt-4o, llama3.2]

  tools:
    - name: web_search
      protocol: native

  memory:
    short_term: 10
    long_term: hive_mind
```

### Topologies

| Topology | Pattern | Use Case |
|----------|---------|----------|
| **Swarm** | All-to-all | Brainstorming, consensus |
| **Hierarchy** | Tree | Command chains, delegation |
| **Pipeline** | Sequential | Workflows with stages |
| **Mesh** | Resilient grid | Fault-tolerant tasks |
| **Ring** | Token passing | Voting, sequential processing |
| **Star** | Hub and spoke | Coordinator pattern |

### LLM Providers

| Provider | Type | Best For |
|----------|------|----------|
| **Ollama** | Local | Development, privacy |
| **Claude** | Cloud | Complex reasoning |
| **OpenAI** | Cloud | Broad compatibility |
| **Groq** | Cloud | Fast inference |

## CLI Commands

```bash
# Initialize project
titan init [directory]

# Run an agent
titan run <spec.yaml> [--prompt "task"]

# Start a swarm
titan swarm "Build a REST API" --topology auto --agents 5

# Check status
titan status

# List agents
titan list --dir ./specs

# Suggest topology
titan topology "Review and approve pull requests"

# Runtime management
titan runtime status           # Show runtime health
titan runtime suggest -t "task" # Suggest runtime for task
titan runtime spawn -s spec.yaml -t "task"  # Spawn on specific runtime

# Phase 3: Self-Organization
titan analyze "task description"     # LLM-powered task analysis
titan analyze "task" --no-llm        # Keyword-based analysis (faster)
titan learning stats                 # View learning statistics
titan learning export -o data.json   # Export learning data
titan events history                 # View event history
titan events history -t topology.changed  # Filter by event type

# Health check
titan health

# Phase 4: Observability & Stress Testing
titan stress swarm --agents 50 --duration 120    # Run 50-agent swarm stress test
titan stress pipeline --agents 20 --duration 60  # Pipeline workflow stress test
titan stress chaos --agents 30 --failure-rate 0.1  # Chaos testing with failures
titan dashboard start --port 8080                # Start web dashboard
titan metrics start --port 9100                  # Start Prometheus metrics endpoint
titan observe start                              # Start full observability stack
titan observe status                             # Check observability status
```

## Runtime Fabric

Agents can execute in different environments based on requirements:

| Runtime | Type | Best For |
|---------|------|----------|
| **Local** | Python process | Development, GPU access, low latency |
| **Docker** | Container | Production, isolation, resource limits |
| **OpenFaaS** | Serverless | Burst scaling, cost optimization |

The Runtime Selector automatically chooses based on:
- GPU requirements
- Scale needs (number of instances)
- Isolation requirements
- Cost sensitivity

## Project Structure

```
agentic-titan/
├── agents/                    # Agent implementations
│   ├── framework/             # Base classes and utilities
│   │   ├── base_agent.py      # BaseAgent ABC
│   │   ├── errors.py          # Error hierarchy
│   │   └── resilience.py      # Circuit breaker, retry
│   ├── archetypes/            # Pre-built agents
│   │   ├── orchestrator.py
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   ├── reviewer.py
│   │   ├── cfo.py             # Budget management
│   │   ├── devops.py          # Infrastructure automation
│   │   ├── security_analyst.py # Security scanning
│   │   ├── data_engineer.py   # ETL pipeline design
│   │   └── product_manager.py # User story generation
│   └── personas.py            # Persona system
│
├── hive/                      # Shared intelligence
│   ├── memory.py              # HiveMind (Redis + ChromaDB)
│   ├── topology.py            # Topology engine
│   ├── criticality.py         # Phase transition detection (Phase 16)
│   ├── fission_fusion.py      # Cluster dynamics (Phase 16)
│   ├── information_center.py  # Pattern aggregation (Phase 16)
│   └── neighborhood.py        # Multi-scale coupling (Phase 16)
│
├── adapters/                  # LLM adapters
│   ├── base.py                # LLMAdapter interface
│   └── router.py              # Multi-provider routing
│
├── runtime/                   # Runtime fabric
│   ├── base.py                # Runtime interface
│   ├── local.py               # Local Python runtime
│   ├── docker.py              # Docker container runtime
│   ├── openfaas.py            # OpenFaaS serverless runtime
│   ├── selector.py            # Intelligent runtime selection
│   └── firecracker/           # MicroVM isolation (Phase 18B)
│       ├── config.py          # VM configuration
│       ├── vm.py              # MicroVMManager
│       ├── network.py         # TAP/NAT management
│       ├── guest_agent.py     # VSOCK communication
│       ├── runtime.py         # FirecrackerRuntime
│       └── image_builder.py   # Rootfs/kernel builder
│
├── titan/                     # Core package
│   ├── spec.py                # Agent Spec DSL
│   ├── cli.py                 # CLI interface
│   ├── metrics.py             # Prometheus instrumentation
│   ├── analysis/              # Analysis tools (Phase 17C)
│   │   ├── contradictions.py  # Contradiction detection
│   │   ├── detector.py        # ContradictionDetector
│   │   └── dialectic.py       # DialecticSynthesizer
│   ├── auth/                  # Authentication (Phase 15)
│   │   ├── models.py          # User, APIKey models
│   │   ├── jwt.py             # JWT creation/verification
│   │   ├── api_keys.py        # API key management
│   │   ├── middleware.py      # FastAPI dependencies
│   │   └── storage.py         # PostgreSQL backend
│   ├── batch/                 # Batch processing (Phase 13)
│   │   ├── celery.py          # Celery integration
│   │   └── tasks.py           # Task definitions
│   ├── learning/              # RLHF pipeline (Phase 18)
│   │   ├── preference_pairs.py # Preference dataset builder
│   │   ├── reward_model.py    # Reward model training
│   │   ├── dpo_trainer.py     # Direct Preference Optimization
│   │   ├── eval_suite.py      # Evaluation metrics
│   │   ├── experiment.py      # Experiment tracking
│   │   ├── deployment.py      # A/B testing
│   │   └── pipeline.py        # Learning coordinator
│   ├── ray/                   # Optional Ray backend (Phase 14)
│   │   ├── config.py          # RayConfig
│   │   ├── serve.py           # Ray Serve deployments
│   │   ├── actors.py          # Ray actors
│   │   └── backend_selector.py # Celery/Ray/Local selection
│   ├── safety/                # Safety systems
│   │   ├── hitl.py            # Human-in-the-loop
│   │   ├── filters.py         # Content filtering
│   │   └── rbac.py            # Role-based access
│   ├── workflows/             # Inquiry engine (Phase 17)
│   │   ├── inquiry_engine.py  # Multi-AI orchestration
│   │   ├── inquiry_config.py  # Stage configuration
│   │   ├── inquiry_dag.py     # DAG execution
│   │   └── cognitive_router.py # Model selection
│   └── stress/                # Stress testing framework
│       ├── runner.py          # StressTestRunner
│       ├── scenarios.py       # Test scenarios
│       └── metrics.py         # Stress metrics
│
├── dashboard/                 # Web dashboard
│   ├── app.py                 # FastAPI application
│   └── templates/             # Jinja2 HTML templates
│       └── models.html        # Epistemic signatures radar chart
│
├── specs/                     # Agent specifications
│   └── *.titan.yaml           # Agent spec files
│
└── deploy/                    # Infrastructure
    ├── compose.yaml           # Docker Compose
    ├── prometheus.yml         # Prometheus config
    ├── Dockerfile.api         # Dashboard container
    ├── grafana/               # Grafana provisioning
    └── k3s/                   # Kubernetes manifests (Phase 15)
        ├── postgres.yaml      # PostgreSQL StatefulSet
        ├── rbac.yaml          # ServiceAccounts, Roles
        ├── network-policies.yaml
        └── resource-quotas.yaml
```

## Observability

### Metrics (Prometheus)

```bash
# Start metrics endpoint
titan metrics start --port 9100

# Metrics available:
# - titan_agent_spawned_total
# - titan_agent_completed_total
# - titan_agent_duration_seconds
# - titan_topology_switches_total
# - titan_llm_requests_total
# - titan_learning_episodes_total
# ... and 15+ more
```

### Dashboard

```bash
# Start web dashboard
titan dashboard start --port 8080

# Features:
# - Real-time agent monitoring (WebSocket)
# - Topology visualization (SVG)
# - Live topology switching
# - Agent spawn/status tracking
```

### Full Observability Stack

```bash
# Start everything (Prometheus + Grafana + metrics)
titan observe start

# Access:
# - Grafana: http://localhost:3000 (admin/titan)
# - Prometheus: http://localhost:9090
# - Dashboard: http://localhost:8080
```

### Docker Compose Profiles

```bash
# Minimal (Redis + ChromaDB)
docker compose -f deploy/compose.yaml up -d redis chromadb

# With monitoring
docker compose -f deploy/compose.yaml --profile monitoring up -d

# Full stack
docker compose -f deploy/compose.yaml --profile full up -d
```

## Stress Testing

```bash
# Available scenarios:
titan stress swarm      # All-to-all communication
titan stress pipeline   # Sequential stage processing
titan stress hierarchy  # Tree delegation pattern
titan stress chaos      # Random failures + topology switches
titan stress scale      # Maximum agents, minimal work

# Options:
--agents 50            # Target number of agents
--duration 120         # Test duration in seconds
--max-concurrent 20    # Max agents running simultaneously
--failure-rate 0.1     # Inject 10% failures (chaos mode)
--output results.json  # Export detailed results
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=titan --cov=agents --cov=hive

# Run specific test categories
pytest tests/ -m e2e          # End-to-end tests
pytest tests/ -m integration  # Integration tests
pytest tests/ -m "not slow"   # Skip slow tests
```

**1095+ tests** covering:
- Unit tests for all components
- Integration tests for safety chain
- E2E workflow tests (swarm, topology, budget)
- Agent archetype tests (92 tests)
- Stress testing framework (50-100 agents)
- RLHF pipeline tests
- Firecracker runtime tests
- Authentication/authorization tests

### Adding a New Agent Archetype

1. Create spec in `specs/myagent.titan.yaml`
2. Implement in `agents/archetypes/myagent.py`
3. Extend `BaseAgent` with custom logic
4. Register in `agents/archetypes/__init__.py`

### Adding a New LLM Provider

1. Implement `LLMAdapter` in `adapters/base.py`
2. Add to router detection in `adapters/router.py`
3. Update `DEFAULT_MODELS` and `PROVIDER_INFO`

## Sources

This project synthesizes patterns from:

- **agent--claude-smith**: Orchestrator, session management, security hooks
- **metasystem-core**: BaseAgent lifecycle, Circuit Breaker, Knowledge Graph patterns
- **my--father-mother**: Dual-persona logging, MCP bridge
- **a-i-council--coliseum**: Decision engine, voting, communication protocol
- **skills**: YAML DSL patterns
- **iGOR**: Episodic learning
- **aionui**: LLM auto-detect and fallback

## Roadmap

### Phase 1: Foundation ✅
- [x] Agent Spec DSL parser
- [x] LLM adapter (multi-provider)
- [x] Local runtime
- [x] Basic Hive Mind (Redis + ChromaDB)
- [x] CLI interface
- [x] Example agents

### Phase 2: Multi-Runtime ✅
- [x] Container runtime (Docker)
- [x] Serverless runtime (OpenFaaS)
- [x] Runtime selector logic

### Phase 3: Self-Organization ✅
- [x] Dynamic topology switching with event notifications
- [x] LLM-powered task analyzer for intelligent topology selection
- [x] Episodic learning system (learns from outcomes)
- [x] Event bus for agent coordination
- [x] CLI commands: `titan analyze`, `titan learning`, `titan events`

### Phase 4: Scale & Polish ✅
- [x] 50-100 agent stress testing framework with 5 scenarios
- [x] Prometheus metrics instrumentation (20+ metrics)
- [x] Grafana dashboards (auto-provisioned)
- [x] Web dashboard (FastAPI + WebSocket real-time updates)
- [x] CLI commands: `stress`, `dashboard`, `metrics`, `observe`

### Phase 13: Batch Processing ✅
- [x] Celery task queue integration
- [x] HiveMind-Celery worker registration
- [x] Stalled batch detection and recovery
- [x] Cleanup tasks (results, artifacts, old batches)

### Phase 14: Ray & Learning Pipeline ✅
- [x] Optional Ray Serve backend
- [x] Learning pipeline coordinator
- [x] Feedback processing with reward signals
- [x] Episode linking for topology learning

### Phase 15: Production Hardening ✅
- [x] PostgreSQL deployment
- [x] JWT authentication system
- [x] API key management
- [x] Rate limiting (Redis backend)
- [x] Admin API endpoints
- [x] Kubernetes RBAC and network policies

### Phase 16: Advanced Topology ✅
- [x] Criticality detection (phase transitions)
- [x] Multi-scale neighbor coupling
- [x] Fission-fusion dynamics
- [x] Information centers (pattern aggregation)

### Phase 17: Expansive Inquiry ✅
- [x] DAG-based workflow execution
- [x] Epistemic signatures visualization (radar charts)
- [x] Contradiction/dialectic detection

### Phase 18: RLHF & Sandbox ✅
- [x] Preference pair dataset builder
- [x] Reward model training
- [x] DPO (Direct Preference Optimization) trainer
- [x] A/B testing deployment
- [x] Firecracker MicroVM runtime

## License

MIT

---

*Built with patterns from production codebases, designed for the next generation of AI agent systems.*
