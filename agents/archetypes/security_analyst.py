"""
Security Analyst Agent - Code security and compliance specialist.

Capabilities:
- Code vulnerability scanning
- Dependency audit
- Security best practices review
- Compliance checking (OWASP, CIS)
- Secret detection
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from adapters.base import LLMMessage
from adapters.router import get_router
from agents.framework.base_agent import BaseAgent

logger = logging.getLogger("titan.agents.security_analyst")


class Severity(Enum):
    """Vulnerability severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityCategory(Enum):
    """Categories of security vulnerabilities."""

    INJECTION = "injection"  # SQL, Command, LDAP injection
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZE = "insecure_deserialization"
    VULNERABLE_COMPONENTS = "vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    SECRETS = "hardcoded_secrets"
    CRYPTO = "weak_cryptography"


class ComplianceFramework(Enum):
    """Compliance frameworks."""

    OWASP_TOP_10 = "owasp_top_10"
    CIS = "cis_benchmarks"
    NIST = "nist_csf"
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"


@dataclass
class Vulnerability:
    """A detected security vulnerability."""

    id: str
    category: VulnerabilityCategory
    severity: Severity
    title: str
    description: str
    file_path: str = ""
    line_number: int = 0
    code_snippet: str = ""
    recommendation: str = ""
    cwe_id: str = ""  # Common Weakness Enumeration
    cvss_score: float = 0.0  # 0-10


@dataclass
class DependencyAudit:
    """Dependency security audit result."""

    package_name: str
    version: str
    vulnerabilities: list[dict[str, Any]] = field(default_factory=list)
    is_outdated: bool = False
    latest_version: str = ""
    license: str = ""
    risk_level: Severity = Severity.LOW


@dataclass
class ComplianceCheck:
    """Compliance check result."""

    framework: ComplianceFramework
    control_id: str
    control_name: str
    status: str  # pass, fail, partial, not_applicable
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class SecurityReport:
    """Complete security analysis report."""

    scan_id: str
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    dependency_audits: list[DependencyAudit] = field(default_factory=list)
    compliance_checks: list[ComplianceCheck] = field(default_factory=list)
    risk_score: float = 0.0  # 0-100
    summary: str = ""


class SecurityAnalystAgent(BaseAgent):
    """
    Agent specialized in security analysis and compliance.

    Capabilities:
    - Scan code for vulnerabilities (OWASP Top 10)
    - Audit dependencies for known CVEs
    - Check compliance with security frameworks
    - Detect hardcoded secrets
    - Provide remediation guidance
    """

    # Patterns for common vulnerabilities
    SECRET_PATTERNS = {
        "aws_key": (
            r"(?:AWS|aws)[_\-]?(?:ACCESS|access)[_\-]?"
            r"(?:KEY|key)[_\-]?(?:ID|id)?\s*[:=]\s*['\"]?([A-Z0-9]{20})['\"]?"
        ),
        "aws_secret": (
            r"(?:AWS|aws)[_\-]?(?:SECRET|secret)[_\-]?(?:ACCESS|access)?"
            r"[_\-]?(?:KEY|key)?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
        ),
        "github_token": r"(?:gh[ps]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59})",
        "generic_api_key": (
            r"(?:api[_\-]?key|apikey|secret[_\-]?key)\s*[:=]\s*['\"]?"
            r"([A-Za-z0-9\-_]{16,})['\"]?"
        ),
        "password": r"(?:password|passwd|pwd)\s*[:=]\s*['\"]([^'\"]{8,})['\"]",
        "jwt": r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+",
        "private_key": r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",
    }

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"execute\s*\(\s*['\"].*%s",  # String formatting in SQL
        r"f['\"].*SELECT.*{.*}",  # f-string in SQL
        r"cursor\.execute\s*\(\s*['\"].*\+",  # String concatenation
        r"\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"innerHTML\s*=\s*[^']",  # Direct innerHTML assignment
        r"document\.write\s*\(",  # document.write usage
        r"\.html\s*\(\s*[^']",  # jQuery .html() with variable
        r"dangerouslySetInnerHTML",  # React dangerous pattern
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"subprocess\.(?:call|run|Popen)\s*\(\s*['\"].*\+",  # String concatenation
        r"os\.system\s*\(",  # os.system usage
        r"exec\s*\(\s*[^)]*\+",  # exec with concatenation
        r"eval\s*\(\s*[^)]*\+",  # eval with concatenation
    ]

    def __init__(
        self,
        scan_depth: str = "standard",
        frameworks: list[ComplianceFramework] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "security_analyst")
        kwargs.setdefault(
            "capabilities",
            [
                "vulnerability_scanning",
                "dependency_audit",
                "compliance_checking",
                "secret_detection",
                "remediation_guidance",
            ],
        )
        super().__init__(**kwargs)

        self.scan_depth = scan_depth  # quick, standard, deep
        self.frameworks = frameworks or [ComplianceFramework.OWASP_TOP_10]
        self._router = get_router()
        self._reports: list[SecurityReport] = []

    async def initialize(self) -> None:
        """Initialize the Security Analyst agent."""
        logger.info(f"Security Analyst Agent '{self.name}' initializing")
        await self._router.initialize()

    async def work(self) -> dict[str, Any]:
        """
        Main work loop - Security agent is typically invoked for specific scans.

        Returns:
            Summary of capabilities and current state
        """
        logger.info("Security Analyst Agent ready for security assessments")

        return {
            "status": "ready",
            "scan_depth": self.scan_depth,
            "frameworks": [f.value for f in self.frameworks],
            "reports_generated": len(self._reports),
        }

    async def shutdown(self) -> None:
        """Cleanup Security Analyst agent."""
        logger.info("Security Analyst Agent shutting down")

        if self._hive_mind and self._reports:
            summary = f"Security scans completed: {len(self._reports)}"
            total_vulns = sum(len(r.vulnerabilities) for r in self._reports)
            summary += f"\nTotal vulnerabilities found: {total_vulns}"
            await self.remember(
                content=summary,
                importance=0.8,
                tags=["security", "scan", "summary"],
            )

    async def scan_code(
        self,
        code: str,
        language: str = "python",
        file_path: str = "unknown",
    ) -> list[Vulnerability]:
        """
        Scan code for security vulnerabilities.

        Args:
            code: Source code to scan
            language: Programming language
            file_path: File path for context

        Returns:
            List of detected vulnerabilities
        """
        self.increment_turn()

        vulnerabilities: list[Vulnerability] = []

        # Pattern-based detection
        pattern_vulns = self._pattern_scan(code, file_path)
        vulnerabilities.extend(pattern_vulns)

        # LLM-based deep analysis
        if self.scan_depth in ("standard", "deep"):
            llm_vulns = await self._llm_security_scan(code, language, file_path)
            vulnerabilities.extend(llm_vulns)

        # Deduplicate
        seen = set()
        unique_vulns = []
        for v in vulnerabilities:
            key = (v.category, v.line_number, v.title[:50])
            if key not in seen:
                seen.add(key)
                unique_vulns.append(v)

        # Log finding
        await self.log_decision(
            decision=f"Scanned {file_path}: {len(unique_vulns)} vulnerabilities",
            category="security_scan",
            rationale=f"Depth: {self.scan_depth}, Language: {language}",
            tags=["security", "scan", language],
        )

        return unique_vulns

    def _pattern_scan(self, code: str, file_path: str) -> list[Vulnerability]:
        """Perform pattern-based vulnerability detection."""
        vulnerabilities = []
        lines = code.split("\n")

        # Check for secrets
        for secret_type, pattern in self.SECRET_PATTERNS.items():
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(
                        Vulnerability(
                            id=f"SEC-{secret_type.upper()}-{line_num}",
                            category=VulnerabilityCategory.SECRETS,
                            severity=Severity.CRITICAL,
                            title=f"Hardcoded {secret_type.replace('_', ' ').title()}",
                            description="Potential hardcoded secret detected",
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip()[:100],
                            recommendation="Use environment variables or a secrets manager",
                            cwe_id="CWE-798",
                        )
                    )

        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(
                        Vulnerability(
                            id=f"SEC-SQLI-{line_num}",
                            category=VulnerabilityCategory.INJECTION,
                            severity=Severity.HIGH,
                            title="Potential SQL Injection",
                            description=(
                                "SQL query constructed with string formatting/concatenation"
                            ),
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip()[:100],
                            recommendation="Use parameterized queries or an ORM",
                            cwe_id="CWE-89",
                        )
                    )

        # Check for XSS
        for pattern in self.XSS_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(
                        Vulnerability(
                            id=f"SEC-XSS-{line_num}",
                            category=VulnerabilityCategory.XSS,
                            severity=Severity.MEDIUM,
                            title="Potential Cross-Site Scripting (XSS)",
                            description="Unsafe HTML manipulation detected",
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip()[:100],
                            recommendation="Sanitize user input and use safe DOM APIs",
                            cwe_id="CWE-79",
                        )
                    )

        # Check for command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(
                        Vulnerability(
                            id=f"SEC-CMDI-{line_num}",
                            category=VulnerabilityCategory.INJECTION,
                            severity=Severity.CRITICAL,
                            title="Potential Command Injection",
                            description="Shell command constructed with user input",
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip()[:100],
                            recommendation="Use subprocess with shell=False and validate inputs",
                            cwe_id="CWE-78",
                        )
                    )

        return vulnerabilities

    async def _llm_security_scan(
        self,
        code: str,
        language: str,
        file_path: str,
    ) -> list[Vulnerability]:
        """Perform LLM-based security analysis."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze this {language} code for security vulnerabilities:

```{language}
{code[:4000]}
```

For each vulnerability found, provide:
VULN: [CATEGORY] [SEVERITY] Title
LINE: approximate line number
DESC: Description
REC: Recommendation
CWE: CWE-XXX

Categories: injection, broken_auth, sensitive_data, xss, access_control, crypto
Severities: critical, high, medium, low""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a security expert. Find vulnerabilities following OWASP guidelines.",
            max_tokens=800,
        )

        vulnerabilities = []
        current_vuln: dict[str, Any] = {}

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("VULN:"):
                if current_vuln:
                    vulnerabilities.append(self._parse_vuln(current_vuln, file_path))
                current_vuln = {"vuln": line.replace("VULN:", "").strip()}
            elif line.startswith("LINE:") and current_vuln:
                try:
                    match = re.search(r"\d+", line)
                    current_vuln["line"] = int(match.group()) if match else 0
                except (AttributeError, ValueError):
                    current_vuln["line"] = 0
            elif line.startswith("DESC:") and current_vuln:
                current_vuln["desc"] = line.replace("DESC:", "").strip()
            elif line.startswith("REC:") and current_vuln:
                current_vuln["rec"] = line.replace("REC:", "").strip()
            elif line.startswith("CWE:") and current_vuln:
                current_vuln["cwe"] = line.replace("CWE:", "").strip()

        if current_vuln:
            vulnerabilities.append(self._parse_vuln(current_vuln, file_path))

        return vulnerabilities

    def _parse_vuln(self, vuln_data: dict[str, Any], file_path: str) -> Vulnerability:
        """Parse vulnerability data from LLM response."""
        vuln_line = vuln_data.get("vuln", "")
        vuln_line_lower = vuln_line.lower()

        # Parse category
        category = VulnerabilityCategory.SECURITY_MISCONFIG
        for cat in VulnerabilityCategory:
            if (
                cat.value.replace("_", " ") in vuln_line_lower
                or cat.name.lower() in vuln_line_lower
            ):
                category = cat
                break

        # Parse severity
        severity = Severity.MEDIUM
        for sev in Severity:
            if sev.value in vuln_line_lower:
                severity = sev
                break

        # Extract title
        title = re.sub(r"\[.*?\]", "", vuln_line).strip()

        return Vulnerability(
            id=f"LLM-{category.name}-{vuln_data.get('line', 0)}",
            category=category,
            severity=severity,
            title=title or "Security Issue",
            description=vuln_data.get("desc", ""),
            file_path=file_path,
            line_number=vuln_data.get("line", 0),
            recommendation=vuln_data.get("rec", ""),
            cwe_id=vuln_data.get("cwe", ""),
        )

    async def audit_dependencies(
        self,
        dependencies: dict[str, str],
        ecosystem: str = "python",
    ) -> list[DependencyAudit]:
        """
        Audit dependencies for known vulnerabilities.

        Args:
            dependencies: Dict of package_name -> version
            ecosystem: Package ecosystem (python, npm, go, etc.)

        Returns:
            List of dependency audit results
        """
        self.increment_turn()

        audits = []

        # In a real implementation, this would query vulnerability databases
        # For now, use LLM to provide analysis
        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze these {ecosystem} dependencies for security issues:

{chr(10).join(f"- {pkg}: {ver}" for pkg, ver in dependencies.items())}

For each package with known issues:
PKG: package_name
VERSION: current_version
ISSUE: Brief description of vulnerability
SEVERITY: critical/high/medium/low
LATEST: latest_safe_version
LICENSE: license_type""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a security expert with knowledge of CVE databases.",
            max_tokens=800,
        )

        # Parse response
        current_pkg: dict[str, Any] = {}
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("PKG:"):
                if current_pkg:
                    audits.append(self._create_audit(current_pkg, dependencies))
                current_pkg = {"pkg": line.replace("PKG:", "").strip()}
            elif line.startswith("VERSION:") and current_pkg:
                current_pkg["version"] = line.replace("VERSION:", "").strip()
            elif line.startswith("ISSUE:") and current_pkg:
                current_pkg.setdefault("issues", []).append(line.replace("ISSUE:", "").strip())
            elif line.startswith("SEVERITY:") and current_pkg:
                current_pkg["severity"] = line.replace("SEVERITY:", "").strip()
            elif line.startswith("LATEST:") and current_pkg:
                current_pkg["latest"] = line.replace("LATEST:", "").strip()
            elif line.startswith("LICENSE:") and current_pkg:
                current_pkg["license"] = line.replace("LICENSE:", "").strip()

        if current_pkg:
            audits.append(self._create_audit(current_pkg, dependencies))

        # Add entries for packages not flagged (they're safe)
        flagged = {a.package_name for a in audits}
        for pkg, ver in dependencies.items():
            if pkg not in flagged:
                audits.append(
                    DependencyAudit(
                        package_name=pkg,
                        version=ver,
                        risk_level=Severity.LOW,
                    )
                )

        return audits

    def _create_audit(
        self,
        data: dict[str, Any],
        dependencies: dict[str, str],
    ) -> DependencyAudit:
        """Create DependencyAudit from parsed data."""
        pkg_name = data.get("pkg", "unknown")
        version = data.get("version", dependencies.get(pkg_name, "unknown"))

        severity = Severity.MEDIUM
        sev_str = data.get("severity", "").lower()
        for s in Severity:
            if s.value in sev_str:
                severity = s
                break

        vulns = [{"description": issue} for issue in data.get("issues", [])]

        return DependencyAudit(
            package_name=pkg_name,
            version=version,
            vulnerabilities=vulns,
            is_outdated=bool(data.get("latest")),
            latest_version=data.get("latest", ""),
            license=data.get("license", ""),
            risk_level=severity,
        )

    async def check_compliance(
        self,
        code: str,
        framework: ComplianceFramework = ComplianceFramework.OWASP_TOP_10,
    ) -> list[ComplianceCheck]:
        """
        Check code compliance against a security framework.

        Args:
            code: Source code to check
            framework: Compliance framework to check against

        Returns:
            List of compliance check results
        """
        self.increment_turn()

        messages = [
            LLMMessage(
                role="user",
                content=f"""Check this code against {framework.value} controls:

```
{code[:3000]}
```

For each relevant control:
CONTROL: Control ID and Name
STATUS: pass/fail/partial
FINDING: What was found
RECOMMENDATION: How to fix""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {framework.value} compliance expert.",
            max_tokens=800,
        )

        checks = []
        current: dict[str, Any] = {}

        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("CONTROL:"):
                if current:
                    checks.append(self._create_compliance_check(current, framework))
                current = {"control": line.replace("CONTROL:", "").strip()}
            elif line.startswith("STATUS:") and current:
                current["status"] = line.replace("STATUS:", "").strip().lower()
            elif line.startswith("FINDING:") and current:
                current.setdefault("findings", []).append(line.replace("FINDING:", "").strip())
            elif line.startswith("RECOMMENDATION:") and current:
                recommendation = line.replace("RECOMMENDATION:", "").strip()
                current.setdefault("recommendations", []).append(recommendation)

        if current:
            checks.append(self._create_compliance_check(current, framework))

        return checks

    def _create_compliance_check(
        self,
        data: dict[str, Any],
        framework: ComplianceFramework,
    ) -> ComplianceCheck:
        """Create ComplianceCheck from parsed data."""
        control = data.get("control", "Unknown")
        parts = control.split(" ", 1)

        return ComplianceCheck(
            framework=framework,
            control_id=parts[0] if parts else "N/A",
            control_name=parts[1] if len(parts) > 1 else control,
            status=data.get("status", "unknown"),
            findings=data.get("findings", []),
            recommendations=data.get("recommendations", []),
        )

    async def generate_report(
        self,
        code: str,
        dependencies: dict[str, str] | None = None,
        language: str = "python",
    ) -> SecurityReport:
        """
        Generate a comprehensive security report.

        Args:
            code: Source code to analyze
            dependencies: Optional dependencies to audit
            language: Programming language

        Returns:
            Complete SecurityReport
        """
        import uuid

        self.increment_turn()

        scan_id = f"SEC-{uuid.uuid4().hex[:8]}"

        # Scan code
        vulnerabilities = await self.scan_code(code, language)

        # Audit dependencies if provided
        audits = []
        if dependencies:
            audits = await self.audit_dependencies(dependencies)

        # Check compliance
        compliance = []
        for framework in self.frameworks:
            checks = await self.check_compliance(code, framework)
            compliance.extend(checks)

        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities, audits, compliance)

        # Generate summary
        summary = self._generate_summary(vulnerabilities, audits, compliance, risk_score)

        report = SecurityReport(
            scan_id=scan_id,
            vulnerabilities=vulnerabilities,
            dependency_audits=audits,
            compliance_checks=compliance,
            risk_score=risk_score,
            summary=summary,
        )

        self._reports.append(report)

        return report

    def _calculate_risk_score(
        self,
        vulnerabilities: list[Vulnerability],
        audits: list[DependencyAudit],
        compliance: list[ComplianceCheck],
    ) -> float:
        """Calculate overall risk score (0-100, higher = more risk)."""
        score = 0.0

        # Vulnerability scoring
        severity_scores = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 15,
            Severity.MEDIUM: 8,
            Severity.LOW: 3,
            Severity.INFO: 1,
        }
        for v in vulnerabilities:
            score += severity_scores.get(v.severity, 5)

        # Dependency scoring
        for a in audits:
            if a.vulnerabilities:
                score += severity_scores.get(a.risk_level, 5) * len(a.vulnerabilities)

        # Compliance scoring
        failed = sum(1 for c in compliance if c.status == "fail")
        partial = sum(1 for c in compliance if c.status == "partial")
        score += failed * 10 + partial * 5

        return min(100.0, score)

    def _generate_summary(
        self,
        vulnerabilities: list[Vulnerability],
        audits: list[DependencyAudit],
        compliance: list[ComplianceCheck],
        risk_score: float,
    ) -> str:
        """Generate human-readable summary."""
        parts = []

        # Risk level
        if risk_score >= 75:
            parts.append("CRITICAL: Immediate action required")
        elif risk_score >= 50:
            parts.append("HIGH: Significant security issues found")
        elif risk_score >= 25:
            parts.append("MEDIUM: Some security improvements needed")
        else:
            parts.append("LOW: Minor issues found")

        # Vulnerability summary
        by_severity: dict[str, int] = {}
        for v in vulnerabilities:
            by_severity[v.severity.value] = by_severity.get(v.severity.value, 0) + 1
        if by_severity:
            severity_summary = ", ".join(f"{k}:{v}" for k, v in by_severity.items())
            parts.append(f"Vulnerabilities: {severity_summary}")

        # Dependency summary
        risky_deps = [a for a in audits if a.vulnerabilities]
        if risky_deps:
            parts.append(f"Vulnerable dependencies: {len(risky_deps)}")

        # Compliance summary
        failed = sum(1 for c in compliance if c.status == "fail")
        if failed:
            parts.append(f"Failed compliance checks: {failed}")

        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about scans performed."""
        if not self._reports:
            return {"reports": 0}

        total_vulns = sum(len(r.vulnerabilities) for r in self._reports)
        avg_risk = sum(r.risk_score for r in self._reports) / len(self._reports)

        return {
            "reports": len(self._reports),
            "total_vulnerabilities": total_vulns,
            "average_risk_score": round(avg_risk, 1),
            "frameworks_checked": [f.value for f in self.frameworks],
        }
