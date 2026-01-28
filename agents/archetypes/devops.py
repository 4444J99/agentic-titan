"""
DevOps Agent - Infrastructure and deployment automation specialist.

Capabilities:
- Infrastructure as code analysis
- Dockerfile and Kubernetes manifest generation
- CI/CD pipeline configuration
- Deployment health checks
- Container orchestration
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents.framework.base_agent import BaseAgent, AgentState
from adapters.base import LLMMessage
from adapters.router import get_router

logger = logging.getLogger("titan.agents.devops")


class InfrastructureType(Enum):
    """Types of infrastructure components."""
    DOCKERFILE = "dockerfile"
    KUBERNETES = "kubernetes"
    HELM = "helm"
    TERRAFORM = "terraform"
    DOCKER_COMPOSE = "docker_compose"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"


class DeploymentStatus(Enum):
    """Deployment health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class InfraAnalysis:
    """Analysis result for infrastructure code."""
    file_path: str
    infra_type: InfrastructureType
    issues: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    security_findings: list[dict[str, Any]] = field(default_factory=list)
    best_practices_score: float = 0.0


@dataclass
class DeploymentConfig:
    """Generated deployment configuration."""
    infra_type: InfrastructureType
    content: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    environment_vars: list[str] = field(default_factory=list)


@dataclass
class HealthCheck:
    """Deployment health check result."""
    service_name: str
    status: DeploymentStatus
    latency_ms: float = 0.0
    error_rate: float = 0.0
    replicas_ready: int = 0
    replicas_desired: int = 0
    last_deployment: str = ""
    issues: list[str] = field(default_factory=list)


class DevOpsAgent(BaseAgent):
    """
    Agent specialized in DevOps and infrastructure automation.

    Capabilities:
    - Analyze infrastructure code for issues and improvements
    - Generate Dockerfiles and Kubernetes manifests
    - Create CI/CD pipeline configurations
    - Perform deployment health checks
    """

    # Common patterns for infrastructure issues
    DOCKERFILE_PATTERNS = {
        "root_user": (r"^USER\s+root", "Running as root user is a security risk"),
        "latest_tag": (r"FROM\s+\S+:latest", "Using 'latest' tag makes builds non-reproducible"),
        "no_healthcheck": (None, "No HEALTHCHECK instruction found"),
        "apt_cache": (r"apt-get\s+install(?!.*--no-install-recommends)", "Consider using --no-install-recommends"),
        "multiple_run": (r"(RUN\s+[^\n]+\n){3,}", "Consider combining multiple RUN commands"),
    }

    K8S_PATTERNS = {
        "no_limits": (r"resources:", "Missing resource limits"),
        "no_probes": (None, "Missing liveness/readiness probes"),
        "privileged": (r"privileged:\s*true", "Privileged container is a security risk"),
        "host_network": (r"hostNetwork:\s*true", "Using host network is a security risk"),
    }

    def __init__(
        self,
        project_path: str | None = None,
        target_platform: str = "kubernetes",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "devops")
        kwargs.setdefault("capabilities", [
            "infrastructure_analysis",
            "dockerfile_generation",
            "kubernetes_manifests",
            "cicd_pipelines",
            "health_checks",
        ])
        super().__init__(**kwargs)

        self.project_path = project_path
        self.target_platform = target_platform
        self._router = get_router()
        self._analyses: list[InfraAnalysis] = []
        self._configs_generated: list[DeploymentConfig] = []

    async def initialize(self) -> None:
        """Initialize the DevOps agent."""
        logger.info(f"DevOps Agent '{self.name}' initializing")
        await self._router.initialize()

        # Load any previous analyses from Hive Mind
        if self._hive_mind and self.project_path:
            previous = await self.recall(f"devops analysis {self.project_path}", k=5)
            if previous:
                logger.info(f"Found {len(previous)} previous DevOps analyses")

    async def work(self) -> dict[str, Any]:
        """
        Main work loop - DevOps agent is typically invoked for specific tasks.

        Returns:
            Summary of available capabilities and current state
        """
        logger.info("DevOps Agent ready for infrastructure tasks")

        return {
            "status": "ready",
            "project_path": self.project_path,
            "target_platform": self.target_platform,
            "analyses_completed": len(self._analyses),
            "configs_generated": len(self._configs_generated),
        }

    async def shutdown(self) -> None:
        """Cleanup DevOps agent."""
        logger.info("DevOps Agent shutting down")

        # Store analyses in memory
        if self._hive_mind and self._analyses:
            summary = "\n".join(
                f"- {a.file_path}: {len(a.issues)} issues, score: {a.best_practices_score:.1f}"
                for a in self._analyses
            )
            await self.remember(
                content=f"DevOps Analysis Summary:\n{summary}",
                importance=0.7,
                tags=["devops", "infrastructure", "analysis"],
            )

    async def analyze_infrastructure(
        self,
        content: str,
        file_path: str = "unknown",
    ) -> InfraAnalysis:
        """
        Analyze infrastructure code for issues and improvements.

        Args:
            content: Infrastructure file content
            file_path: Path to the file (for context)

        Returns:
            InfraAnalysis with findings
        """
        self.increment_turn()

        # Detect infrastructure type
        infra_type = self._detect_infra_type(content, file_path)

        # Pattern-based analysis
        issues = self._pattern_analysis(content, infra_type)

        # LLM-based deep analysis
        llm_findings = await self._llm_analysis(content, infra_type)

        analysis = InfraAnalysis(
            file_path=file_path,
            infra_type=infra_type,
            issues=issues + llm_findings.get("issues", []),
            recommendations=llm_findings.get("recommendations", []),
            security_findings=llm_findings.get("security", []),
            best_practices_score=llm_findings.get("score", 0.5),
        )

        self._analyses.append(analysis)

        # Log decision
        await self.log_decision(
            decision=f"Analyzed {infra_type.value} file: {file_path}",
            category="infrastructure",
            rationale=f"Found {len(analysis.issues)} issues",
            tags=["devops", infra_type.value],
        )

        return analysis

    def _detect_infra_type(self, content: str, file_path: str) -> InfrastructureType:
        """Detect the type of infrastructure file."""
        file_lower = file_path.lower()

        if "dockerfile" in file_lower or content.strip().startswith("FROM "):
            return InfrastructureType.DOCKERFILE
        elif file_lower.endswith((".yaml", ".yml")):
            if "apiVersion:" in content and "kind:" in content:
                return InfrastructureType.KUBERNETES
            elif "services:" in content and ("version:" in content or "networks:" in content):
                return InfrastructureType.DOCKER_COMPOSE
            elif "stages:" in content or "jobs:" in content:
                if ".gitlab-ci" in file_lower:
                    return InfrastructureType.GITLAB_CI
                return InfrastructureType.GITHUB_ACTIONS
        elif "helm" in file_lower or "chart.yaml" in file_lower:
            return InfrastructureType.HELM
        elif file_lower.endswith(".tf"):
            return InfrastructureType.TERRAFORM
        elif "jenkinsfile" in file_lower:
            return InfrastructureType.JENKINS

        return InfrastructureType.KUBERNETES  # Default

    def _pattern_analysis(
        self,
        content: str,
        infra_type: InfrastructureType,
    ) -> list[dict[str, Any]]:
        """Perform pattern-based analysis."""
        issues = []

        patterns = {}
        if infra_type == InfrastructureType.DOCKERFILE:
            patterns = self.DOCKERFILE_PATTERNS
        elif infra_type == InfrastructureType.KUBERNETES:
            patterns = self.K8S_PATTERNS

        for issue_id, (pattern, message) in patterns.items():
            if pattern:
                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    issues.append({
                        "id": issue_id,
                        "severity": "medium",
                        "message": message,
                        "pattern_match": True,
                    })
            else:
                # Check for absence of something
                if issue_id == "no_healthcheck" and "HEALTHCHECK" not in content:
                    issues.append({
                        "id": issue_id,
                        "severity": "low",
                        "message": message,
                    })
                elif issue_id == "no_probes" and "livenessProbe" not in content and "readinessProbe" not in content:
                    issues.append({
                        "id": issue_id,
                        "severity": "medium",
                        "message": message,
                    })

        return issues

    async def _llm_analysis(
        self,
        content: str,
        infra_type: InfrastructureType,
    ) -> dict[str, Any]:
        """Perform LLM-based deep analysis."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze this {infra_type.value} configuration for issues and improvements:

```
{content[:3000]}
```

Provide analysis in this format:
ISSUES:
- [severity:high/medium/low] Issue description
RECOMMENDATIONS:
- Improvement suggestion
SECURITY:
- Security concern
SCORE: X.X (0-10 best practices score)""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a DevOps expert. Analyze infrastructure code for issues, security, and best practices.",
            max_tokens=500,
        )

        # Parse response
        result: dict[str, Any] = {"issues": [], "recommendations": [], "security": [], "score": 5.0}
        current_section = None

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("ISSUES:"):
                current_section = "issues"
            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"
            elif line.startswith("SECURITY:"):
                current_section = "security"
            elif line.startswith("SCORE:"):
                try:
                    score_str = line.replace("SCORE:", "").strip()
                    result["score"] = min(10.0, max(0.0, float(score_str.split()[0])))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "issues":
                    severity = "medium"
                    if "[severity:" in item.lower():
                        if "high]" in item.lower():
                            severity = "high"
                        elif "low]" in item.lower():
                            severity = "low"
                        item = re.sub(r"\[severity:\w+\]\s*", "", item, flags=re.IGNORECASE)
                    result["issues"].append({"severity": severity, "message": item})
                elif current_section == "security":
                    result["security"].append({"severity": "high", "message": item})
                else:
                    result["recommendations"].append(item)

        return result

    async def generate_dockerfile(
        self,
        language: str,
        framework: str | None = None,
        requirements: list[str] | None = None,
    ) -> DeploymentConfig:
        """
        Generate a Dockerfile for the given language/framework.

        Args:
            language: Programming language (python, node, go, etc.)
            framework: Optional framework (fastapi, express, gin, etc.)
            requirements: Additional requirements

        Returns:
            DeploymentConfig with Dockerfile content
        """
        self.increment_turn()

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate a production-ready Dockerfile for:
Language: {language}
Framework: {framework or 'none'}
Requirements: {', '.join(requirements or ['standard'])}

Follow these best practices:
1. Use multi-stage builds
2. Don't run as root
3. Include health check
4. Minimize image size
5. Use specific base image tags

Output only the Dockerfile content.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a DevOps expert. Generate production-ready Dockerfiles following best practices.",
            max_tokens=800,
        )

        # Extract Dockerfile content
        content = response.content
        if "```dockerfile" in content.lower():
            content = content.split("```dockerfile", 1)[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        config = DeploymentConfig(
            infra_type=InfrastructureType.DOCKERFILE,
            content=content.strip(),
            description=f"Dockerfile for {language}" + (f" with {framework}" if framework else ""),
            dependencies=[language] + (requirements or []),
        )

        self._configs_generated.append(config)
        return config

    async def generate_kubernetes_manifest(
        self,
        app_name: str,
        image: str,
        port: int = 8080,
        replicas: int = 2,
        resources: dict[str, Any] | None = None,
    ) -> DeploymentConfig:
        """
        Generate Kubernetes deployment manifest.

        Args:
            app_name: Application name
            image: Container image
            port: Container port
            replicas: Number of replicas
            resources: Resource limits/requests

        Returns:
            DeploymentConfig with K8s manifest
        """
        self.increment_turn()

        resources = resources or {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        }

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate Kubernetes manifests for:
App Name: {app_name}
Image: {image}
Port: {port}
Replicas: {replicas}
Resources: {resources}

Include:
1. Deployment with health probes
2. Service (ClusterIP)
3. HorizontalPodAutoscaler
4. NetworkPolicy (optional)

Follow security best practices. Output YAML only.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a Kubernetes expert. Generate secure, production-ready manifests.",
            max_tokens=1000,
        )

        content = response.content
        if "```yaml" in content.lower():
            content = content.split("```yaml", 1)[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        config = DeploymentConfig(
            infra_type=InfrastructureType.KUBERNETES,
            content=content.strip(),
            description=f"Kubernetes manifests for {app_name}",
            dependencies=[image],
            environment_vars=[],
        )

        self._configs_generated.append(config)
        return config

    async def generate_cicd_pipeline(
        self,
        platform: str = "github",
        stages: list[str] | None = None,
        language: str = "python",
    ) -> DeploymentConfig:
        """
        Generate CI/CD pipeline configuration.

        Args:
            platform: CI/CD platform (github, gitlab, jenkins)
            stages: Pipeline stages
            language: Primary language

        Returns:
            DeploymentConfig with pipeline config
        """
        self.increment_turn()

        stages = stages or ["lint", "test", "build", "deploy"]

        infra_type_map = {
            "github": InfrastructureType.GITHUB_ACTIONS,
            "gitlab": InfrastructureType.GITLAB_CI,
            "jenkins": InfrastructureType.JENKINS,
        }
        infra_type = infra_type_map.get(platform, InfrastructureType.GITHUB_ACTIONS)

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate a CI/CD pipeline for:
Platform: {platform}
Language: {language}
Stages: {', '.join(stages)}

Include:
1. Caching for dependencies
2. Parallel jobs where possible
3. Environment-specific deployments
4. Security scanning step

Output only the pipeline configuration file content.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {platform} CI/CD expert. Generate efficient, secure pipelines.",
            max_tokens=800,
        )

        content = response.content
        if "```yaml" in content.lower():
            content = content.split("```yaml", 1)[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        config = DeploymentConfig(
            infra_type=infra_type,
            content=content.strip(),
            description=f"{platform.title()} CI/CD pipeline for {language}",
            dependencies=stages,
        )

        self._configs_generated.append(config)
        return config

    async def check_deployment_health(
        self,
        service_name: str,
        namespace: str = "default",
    ) -> HealthCheck:
        """
        Check deployment health (simulated for now).

        Args:
            service_name: Name of the service
            namespace: Kubernetes namespace

        Returns:
            HealthCheck with status
        """
        self.increment_turn()

        # In a real implementation, this would query the Kubernetes API
        # For now, return simulated health check
        messages = [
            LLMMessage(
                role="user",
                content=f"""Simulate a health check response for service '{service_name}' in namespace '{namespace}'.

Provide realistic metrics in this format:
STATUS: healthy/degraded/unhealthy
LATENCY_MS: X
ERROR_RATE: X.XX
REPLICAS_READY: X
REPLICAS_DESIRED: X
ISSUES:
- Issue if any""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a Kubernetes monitoring system. Provide realistic health metrics.",
            max_tokens=200,
        )

        # Parse response
        status = DeploymentStatus.UNKNOWN
        latency = 0.0
        error_rate = 0.0
        ready = 0
        desired = 0
        issues = []

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("STATUS:"):
                status_str = line.replace("STATUS:", "").strip().lower()
                if "healthy" in status_str:
                    status = DeploymentStatus.HEALTHY
                elif "degraded" in status_str:
                    status = DeploymentStatus.DEGRADED
                elif "unhealthy" in status_str:
                    status = DeploymentStatus.UNHEALTHY
            elif line.startswith("LATENCY_MS:"):
                try:
                    latency = float(line.replace("LATENCY_MS:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("ERROR_RATE:"):
                try:
                    error_rate = float(line.replace("ERROR_RATE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REPLICAS_READY:"):
                try:
                    ready = int(line.replace("REPLICAS_READY:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REPLICAS_DESIRED:"):
                try:
                    desired = int(line.replace("REPLICAS_DESIRED:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("- "):
                issues.append(line[2:].strip())

        return HealthCheck(
            service_name=service_name,
            status=status,
            latency_ms=latency,
            error_rate=error_rate,
            replicas_ready=ready,
            replicas_desired=desired,
            issues=issues,
        )

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of all analyses performed."""
        if not self._analyses:
            return {"count": 0}

        total_issues = sum(len(a.issues) for a in self._analyses)
        total_security = sum(len(a.security_findings) for a in self._analyses)
        avg_score = sum(a.best_practices_score for a in self._analyses) / len(self._analyses)

        return {
            "count": len(self._analyses),
            "total_issues": total_issues,
            "total_security_findings": total_security,
            "average_score": round(avg_score, 2),
            "by_type": {
                t.value: sum(1 for a in self._analyses if a.infra_type == t)
                for t in InfrastructureType
            },
        }
