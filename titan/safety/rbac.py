"""
Titan Safety - Role-Based Access Control

Prevents role confusion with runtime permission validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from titan.safety.permissions import (
    DEFAULT_ROLE_PERMISSIONS,
    Permission,
    PersonaRole,
    RolePermissions,
    get_role_permissions,
)

if TYPE_CHECKING:
    from titan.persistence.audit import AuditLogger

logger = logging.getLogger("titan.safety.rbac")


@dataclass
class ValidationResult:
    """Result of a permission validation."""

    allowed: bool
    permission: Permission
    role: PersonaRole
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "permission": self.permission.value,
            "role": self.role.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ActionValidation:
    """Validation of a complete action."""

    id: UUID = field(default_factory=uuid4)
    agent_id: str = ""
    action: str = ""
    required_permissions: list[Permission] = field(default_factory=list)
    role: PersonaRole = PersonaRole.CUSTOM
    allowed: bool = False
    missing_permissions: list[Permission] = field(default_factory=list)
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "agent_id": self.agent_id,
            "action": self.action,
            "required_permissions": [p.value for p in self.required_permissions],
            "role": self.role.value,
            "allowed": self.allowed,
            "missing_permissions": [p.value for p in self.missing_permissions],
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


class RBACEnforcer:
    """
    Enforces role-based access control.

    Validates agent actions against role permissions and
    prevents privilege escalation.
    """

    def __init__(
        self,
        role_permissions: dict[PersonaRole, RolePermissions] | None = None,
        audit_logger: AuditLogger | None = None,
        strict_mode: bool = True,
    ) -> None:
        """
        Args:
            role_permissions: Custom role permission mappings
            audit_logger: Logger for audit trail
            strict_mode: If True, deny by default; if False, allow unknown actions
        """
        self._role_permissions = role_permissions or DEFAULT_ROLE_PERMISSIONS.copy()
        self._audit_logger = audit_logger
        self._strict_mode = strict_mode

        # Agent role assignments
        self._agent_roles: dict[str, PersonaRole] = {}

        # Custom agent permissions (overrides)
        self._agent_permission_overrides: dict[str, set[Permission]] = {}

        # Validation history for debugging
        self._validation_history: list[ActionValidation] = []

    def assign_role(self, agent_id: str, role: PersonaRole) -> None:
        """
        Assign a role to an agent.

        Args:
            agent_id: Agent identifier
            role: Role to assign
        """
        self._agent_roles[agent_id] = role
        logger.info(f"Assigned role {role.value} to agent {agent_id}")

    def get_role(self, agent_id: str) -> PersonaRole | None:
        """Get the role assigned to an agent."""
        return self._agent_roles.get(agent_id)

    def grant_permission(self, agent_id: str, permission: Permission) -> None:
        """
        Grant an additional permission to an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission to grant
        """
        if agent_id not in self._agent_permission_overrides:
            self._agent_permission_overrides[agent_id] = set()
        self._agent_permission_overrides[agent_id].add(permission)
        logger.info(f"Granted {permission.value} to agent {agent_id}")

    def revoke_permission(self, agent_id: str, permission: Permission) -> None:
        """
        Revoke a permission from an agent.

        Args:
            agent_id: Agent identifier
            permission: Permission to revoke
        """
        if agent_id in self._agent_permission_overrides:
            self._agent_permission_overrides[agent_id].discard(permission)
            logger.info(f"Revoked {permission.value} from agent {agent_id}")

    def check_permission(
        self,
        agent_id: str,
        permission: Permission,
    ) -> ValidationResult:
        """
        Check if an agent has a specific permission.

        Args:
            agent_id: Agent identifier
            permission: Permission to check

        Returns:
            ValidationResult indicating if allowed
        """
        role = self._agent_roles.get(agent_id)

        if role is None:
            return ValidationResult(
                allowed=not self._strict_mode,
                permission=permission,
                role=PersonaRole.CUSTOM,
                reason="No role assigned" if self._strict_mode else "No role, defaulting to allow",
            )

        # Get role permissions
        role_perms = self._role_permissions.get(role)
        if not role_perms:
            return ValidationResult(
                allowed=not self._strict_mode,
                permission=permission,
                role=role,
                reason="Role permissions not defined",
            )

        # Check base role permission
        has_base_permission = role_perms.has_permission(permission)

        # Check permission overrides
        has_override = permission in self._agent_permission_overrides.get(agent_id, set())

        allowed = has_base_permission or has_override

        return ValidationResult(
            allowed=allowed,
            permission=permission,
            role=role,
            reason="Permission granted" if allowed else f"Permission {permission.value} not in role {role.value}",
            metadata={
                "from_role": has_base_permission,
                "from_override": has_override,
            },
        )

    async def validate_action(
        self,
        agent_id: str,
        action: str,
        required_permissions: list[Permission],
    ) -> ActionValidation:
        """
        Validate an action against agent permissions.

        Args:
            agent_id: Agent performing the action
            action: Description of the action
            required_permissions: Permissions required for the action

        Returns:
            ActionValidation with detailed results
        """
        role = self._agent_roles.get(agent_id, PersonaRole.CUSTOM)
        missing = []

        for permission in required_permissions:
            result = self.check_permission(agent_id, permission)
            if not result.allowed:
                missing.append(permission)

        allowed = len(missing) == 0

        validation = ActionValidation(
            agent_id=agent_id,
            action=action,
            required_permissions=required_permissions,
            role=role,
            allowed=allowed,
            missing_permissions=missing,
            reason="Action allowed" if allowed else f"Missing permissions: {[p.value for p in missing]}",
        )

        # Store in history
        self._validation_history.append(validation)
        if len(self._validation_history) > 1000:
            self._validation_history = self._validation_history[-500:]

        # Audit log
        await self._audit_validation(validation)

        if not allowed:
            logger.warning(
                f"Permission denied for {agent_id}: {action} "
                f"(missing: {[p.value for p in missing]})"
            )

        return validation

    async def validate_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ActionValidation:
        """
        Validate a tool call against agent permissions.

        Args:
            agent_id: Agent making the call
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ActionValidation with results
        """
        # Map tool names to required permissions
        tool_permissions = self._get_tool_permissions(tool_name)

        return await self.validate_action(
            agent_id=agent_id,
            action=f"Tool call: {tool_name}",
            required_permissions=tool_permissions,
        )

    def _get_tool_permissions(self, tool_name: str) -> list[Permission]:
        """Map tool name to required permissions."""
        # Tool to permission mapping
        tool_permission_map: dict[str, list[Permission]] = {
            # File tools
            "read_file": [Permission.READ_FILES],
            "write_file": [Permission.WRITE_FILES],
            "delete_file": [Permission.DELETE_FILES],
            "list_directory": [Permission.READ_FILES],

            # Execution tools
            "execute_code": [Permission.EXECUTE_CODE],
            "execute_shell": [Permission.EXECUTE_SHELL],
            "run_sandbox": [Permission.EXECUTE_SANDBOXED],
            "shell": [Permission.EXECUTE_SHELL],
            "bash": [Permission.EXECUTE_SHELL],

            # Network tools
            "http_request": [Permission.MAKE_HTTP_REQUESTS],
            "fetch": [Permission.MAKE_HTTP_REQUESTS],
            "api_call": [Permission.CONNECT_EXTERNAL_APIS],
            "send_email": [Permission.SEND_EMAILS],

            # Database tools
            "query_database": [Permission.READ_DATABASE],
            "execute_sql": [Permission.WRITE_DATABASE],

            # Agent tools
            "spawn_agent": [Permission.SPAWN_AGENTS],
            "terminate_agent": [Permission.TERMINATE_AGENTS],
        }

        return tool_permission_map.get(tool_name, [])

    def get_agent_permissions(self, agent_id: str) -> set[Permission]:
        """Get all permissions for an agent."""
        role = self._agent_roles.get(agent_id)
        permissions: set[Permission] = set()

        if role:
            role_perms = self._role_permissions.get(role)
            if role_perms:
                permissions = role_perms.permissions.copy()

        # Add overrides
        if agent_id in self._agent_permission_overrides:
            permissions |= self._agent_permission_overrides[agent_id]

        return permissions

    def get_validation_history(
        self,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[ActionValidation]:
        """Get validation history, optionally filtered by agent."""
        history = self._validation_history
        if agent_id:
            history = [v for v in history if v.agent_id == agent_id]
        return history[-limit:]

    async def _audit_validation(self, validation: ActionValidation) -> None:
        """Log validation to audit trail."""
        if not self._audit_logger:
            return

        try:
            from titan.persistence.models import AuditEventType

            if validation.allowed:
                return  # Only log denials

            await self._audit_logger.log_event(
                event_type=AuditEventType.PERMISSION_DENIED,
                action=f"RBAC: {validation.action}",
                agent_id=validation.agent_id,
                input_data={
                    "required_permissions": [p.value for p in validation.required_permissions],
                    "role": validation.role.value,
                },
                output_data={
                    "allowed": validation.allowed,
                    "missing_permissions": [p.value for p in validation.missing_permissions],
                },
            )
        except Exception as e:
            logger.warning(f"Failed to audit RBAC validation: {e}")


# Singleton instance
_default_enforcer: RBACEnforcer | None = None


def get_rbac_enforcer() -> RBACEnforcer:
    """Get the default RBAC enforcer."""
    global _default_enforcer
    if _default_enforcer is None:
        _default_enforcer = RBACEnforcer()
    return _default_enforcer
