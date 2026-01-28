"""
Titan Safety - Permission Definitions

Defines permissions for role-based access control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Permission(str, Enum):
    """Available permissions in the system."""

    # Agent management
    SPAWN_AGENTS = "spawn_agents"
    TERMINATE_AGENTS = "terminate_agents"
    CONFIGURE_AGENTS = "configure_agents"

    # Code execution
    EXECUTE_CODE = "execute_code"
    EXECUTE_SHELL = "execute_shell"
    EXECUTE_SANDBOXED = "execute_sandboxed"

    # File operations
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    DELETE_FILES = "delete_files"
    MODIFY_PERMISSIONS = "modify_permissions"

    # Network operations
    MAKE_HTTP_REQUESTS = "make_http_requests"
    CONNECT_EXTERNAL_APIS = "connect_external_apis"
    SEND_EMAILS = "send_emails"

    # Database operations
    READ_DATABASE = "read_database"
    WRITE_DATABASE = "write_database"
    MODIFY_SCHEMA = "modify_schema"

    # Approval operations
    APPROVE_ACTIONS = "approve_actions"
    BYPASS_APPROVAL = "bypass_approval"

    # Budget operations
    VIEW_BUDGET = "view_budget"
    ALLOCATE_BUDGET = "allocate_budget"
    MODIFY_LIMITS = "modify_limits"

    # System operations
    MODIFY_CONFIG = "modify_config"
    VIEW_AUDIT_LOG = "view_audit_log"
    MODIFY_TOPOLOGY = "modify_topology"


class PersonaRole(str, Enum):
    """Pre-defined persona roles."""

    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    CFO = "cfo"
    ADMIN = "admin"
    CUSTOM = "custom"


@dataclass
class RolePermissions:
    """Permission set for a role."""

    role: PersonaRole
    permissions: set[Permission] = field(default_factory=set)
    description: str = ""
    can_delegate: bool = False  # Can this role grant permissions to others?
    max_budget_usd: float | None = None  # Maximum budget this role can use

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        return permission in self.permissions

    def has_all_permissions(self, permissions: list[Permission]) -> bool:
        """Check if role has all specified permissions."""
        return all(p in self.permissions for p in permissions)

    def has_any_permission(self, permissions: list[Permission]) -> bool:
        """Check if role has any of the specified permissions."""
        return any(p in self.permissions for p in permissions)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "description": self.description,
            "can_delegate": self.can_delegate,
            "max_budget_usd": self.max_budget_usd,
        }


# Default permission sets for each role
DEFAULT_ROLE_PERMISSIONS: dict[PersonaRole, RolePermissions] = {
    PersonaRole.ORCHESTRATOR: RolePermissions(
        role=PersonaRole.ORCHESTRATOR,
        permissions={
            Permission.SPAWN_AGENTS,
            Permission.TERMINATE_AGENTS,
            Permission.CONFIGURE_AGENTS,
            Permission.READ_FILES,
            Permission.APPROVE_ACTIONS,
            Permission.VIEW_BUDGET,
            Permission.ALLOCATE_BUDGET,
            Permission.MODIFY_TOPOLOGY,
            Permission.MAKE_HTTP_REQUESTS,
        },
        description="Coordinates multi-agent workflows, can spawn and manage agents",
        can_delegate=True,
        max_budget_usd=50.0,
    ),
    PersonaRole.RESEARCHER: RolePermissions(
        role=PersonaRole.RESEARCHER,
        permissions={
            Permission.READ_FILES,
            Permission.MAKE_HTTP_REQUESTS,
            Permission.CONNECT_EXTERNAL_APIS,
            Permission.READ_DATABASE,
            Permission.VIEW_BUDGET,
        },
        description="Gathers and analyzes information, read-only access",
        can_delegate=False,
        max_budget_usd=5.0,
    ),
    PersonaRole.CODER: RolePermissions(
        role=PersonaRole.CODER,
        permissions={
            Permission.READ_FILES,
            Permission.WRITE_FILES,
            Permission.EXECUTE_CODE,
            Permission.EXECUTE_SANDBOXED,
            Permission.MAKE_HTTP_REQUESTS,
            Permission.READ_DATABASE,
            Permission.WRITE_DATABASE,
            Permission.VIEW_BUDGET,
        },
        description="Writes and tests code, can execute in sandbox",
        can_delegate=False,
        max_budget_usd=10.0,
    ),
    PersonaRole.REVIEWER: RolePermissions(
        role=PersonaRole.REVIEWER,
        permissions={
            Permission.READ_FILES,
            Permission.APPROVE_ACTIONS,
            Permission.VIEW_AUDIT_LOG,
            Permission.VIEW_BUDGET,
        },
        description="Reviews work for quality, can approve actions",
        can_delegate=False,
        max_budget_usd=2.0,
    ),
    PersonaRole.CFO: RolePermissions(
        role=PersonaRole.CFO,
        permissions={
            Permission.VIEW_BUDGET,
            Permission.ALLOCATE_BUDGET,
            Permission.MODIFY_LIMITS,
            Permission.VIEW_AUDIT_LOG,
            Permission.APPROVE_ACTIONS,
        },
        description="Manages budgets and cost optimization",
        can_delegate=True,
        max_budget_usd=100.0,
    ),
    PersonaRole.ADMIN: RolePermissions(
        role=PersonaRole.ADMIN,
        permissions=set(Permission),  # All permissions
        description="Full system access",
        can_delegate=True,
        max_budget_usd=None,  # Unlimited
    ),
}


def get_role_permissions(role: PersonaRole) -> RolePermissions:
    """Get default permissions for a role."""
    return DEFAULT_ROLE_PERMISSIONS.get(
        role,
        RolePermissions(role=PersonaRole.CUSTOM, description="Custom role"),
    )


def create_custom_role(
    name: str,
    permissions: set[Permission],
    description: str = "",
    can_delegate: bool = False,
    max_budget_usd: float | None = None,
) -> RolePermissions:
    """Create a custom role with specific permissions."""
    return RolePermissions(
        role=PersonaRole.CUSTOM,
        permissions=permissions,
        description=description or f"Custom role: {name}",
        can_delegate=can_delegate,
        max_budget_usd=max_budget_usd,
    )
