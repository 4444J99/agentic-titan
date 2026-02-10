"""
Titan Safety Layer

Provides safety mechanisms for the agent system:
- Human-in-the-Loop (HITL) approval gates
- Content filtering and guardrails
- Role-based access control (RBAC)
- Action risk classification
- Output sanitization
"""

from titan.safety.filters import (
    CommandInjectionFilter,
    ContentFilter,
    CredentialLeakFilter,
    FilterMatch,
    FilterPipeline,
    FilterResult,
    PromptInjectionFilter,
    create_default_pipeline,
)
from titan.safety.gates import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalResult,
    ApprovalStatus,
    GateRegistry,
    get_gate_registry,
)
from titan.safety.hitl import (
    HITLConfig,
    HITLHandler,
    get_hitl_handler,
    init_hitl_handler,
)
from titan.safety.patterns import (
    DangerousPattern,
    PatternCategory,
    PatternSeverity,
    get_all_patterns,
    get_patterns_by_category,
    get_patterns_by_severity,
)
from titan.safety.permissions import (
    Permission,
    PersonaRole,
    RolePermissions,
    create_custom_role,
    get_role_permissions,
)
from titan.safety.policies import (
    ActionCategory,
    ActionClassifier,
    ActionPolicy,
    RiskLevel,
    get_action_classifier,
)
from titan.safety.rbac import (
    ActionValidation,
    RBACEnforcer,
    ValidationResult,
    get_rbac_enforcer,
)
from titan.safety.sanitizer import (
    OutputSanitizer,
    SanitizationConfig,
    SanitizationResult,
    get_sanitizer,
    sanitize_output,
)

__all__ = [
    # Policies
    "RiskLevel",
    "ActionPolicy",
    "ActionCategory",
    "ActionClassifier",
    "get_action_classifier",
    # Gates
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalResult",
    "ApprovalStatus",
    "GateRegistry",
    "get_gate_registry",
    # HITL
    "HITLHandler",
    "HITLConfig",
    "get_hitl_handler",
    "init_hitl_handler",
    # Patterns
    "PatternCategory",
    "PatternSeverity",
    "DangerousPattern",
    "get_all_patterns",
    "get_patterns_by_category",
    "get_patterns_by_severity",
    # Filters
    "ContentFilter",
    "FilterResult",
    "FilterMatch",
    "FilterPipeline",
    "PromptInjectionFilter",
    "CredentialLeakFilter",
    "CommandInjectionFilter",
    "create_default_pipeline",
    # Sanitizer
    "OutputSanitizer",
    "SanitizationConfig",
    "SanitizationResult",
    "get_sanitizer",
    "sanitize_output",
    # Permissions
    "Permission",
    "PersonaRole",
    "RolePermissions",
    "get_role_permissions",
    "create_custom_role",
    # RBAC
    "RBACEnforcer",
    "ValidationResult",
    "ActionValidation",
    "get_rbac_enforcer",
]
