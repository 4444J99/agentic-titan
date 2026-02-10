"""
Adversarial Tests - Role Confusion Prevention

Tests for RBAC enforcement and role boundary protection.
"""

import pytest

from titan.safety.permissions import (
    DEFAULT_ROLE_PERMISSIONS,
    Permission,
    PersonaRole,
    get_role_permissions,
)
from titan.safety.rbac import RBACEnforcer


class TestRBACEnforcement:
    """Tests for role-based access control enforcement."""

    @pytest.fixture
    def enforcer(self):
        return RBACEnforcer(strict_mode=True)

    @pytest.fixture
    def enforcer_permissive(self):
        return RBACEnforcer(strict_mode=False)

    # Test role assignment
    def test_assign_role(self, enforcer):
        """Test that roles can be assigned to agents."""
        enforcer.assign_role("agent-123", PersonaRole.CODER)
        assert enforcer.get_role("agent-123") == PersonaRole.CODER

    def test_unassigned_agent_strict_mode(self, enforcer):
        """Test that unassigned agents are denied in strict mode."""
        result = enforcer.check_permission("unknown-agent", Permission.EXECUTE_CODE)
        assert not result.allowed

    def test_unassigned_agent_permissive_mode(self, enforcer_permissive):
        """Test that unassigned agents are allowed in permissive mode."""
        result = enforcer_permissive.check_permission("unknown-agent", Permission.EXECUTE_CODE)
        assert result.allowed

    # Test permission checking per role
    @pytest.mark.parametrize(
        "role,permission,expected",
        [
            # Orchestrator permissions
            (PersonaRole.ORCHESTRATOR, Permission.SPAWN_AGENTS, True),
            (PersonaRole.ORCHESTRATOR, Permission.EXECUTE_CODE, False),
            (PersonaRole.ORCHESTRATOR, Permission.MODIFY_TOPOLOGY, True),
            # Coder permissions
            (PersonaRole.CODER, Permission.EXECUTE_CODE, True),
            (PersonaRole.CODER, Permission.WRITE_FILES, True),
            (PersonaRole.CODER, Permission.SPAWN_AGENTS, False),
            # Researcher permissions
            (PersonaRole.RESEARCHER, Permission.READ_FILES, True),
            (PersonaRole.RESEARCHER, Permission.WRITE_FILES, False),
            (PersonaRole.RESEARCHER, Permission.EXECUTE_CODE, False),
            # Reviewer permissions
            (PersonaRole.REVIEWER, Permission.APPROVE_ACTIONS, True),
            (PersonaRole.REVIEWER, Permission.EXECUTE_CODE, False),
            (PersonaRole.REVIEWER, Permission.WRITE_FILES, False),
            # CFO permissions
            (PersonaRole.CFO, Permission.ALLOCATE_BUDGET, True),
            (PersonaRole.CFO, Permission.EXECUTE_CODE, False),
            (PersonaRole.CFO, Permission.MODIFY_LIMITS, True),
        ],
    )
    def test_role_permissions(self, enforcer, role, permission, expected):
        """Test that roles have correct permissions."""
        agent_id = f"agent-{role.value}"
        enforcer.assign_role(agent_id, role)
        result = enforcer.check_permission(agent_id, permission)
        assert result.allowed == expected, (
            f"Expected {expected} for {role.value} + {permission.value}"
        )

    # Test permission override
    def test_grant_additional_permission(self, enforcer):
        """Test that additional permissions can be granted."""
        enforcer.assign_role("agent-1", PersonaRole.RESEARCHER)
        # Researcher shouldn't have EXECUTE_CODE
        assert not enforcer.check_permission("agent-1", Permission.EXECUTE_CODE).allowed

        # Grant the permission
        enforcer.grant_permission("agent-1", Permission.EXECUTE_CODE)
        assert enforcer.check_permission("agent-1", Permission.EXECUTE_CODE).allowed

    def test_revoke_permission(self, enforcer):
        """Test that permissions can be revoked."""
        enforcer.assign_role("agent-1", PersonaRole.CODER)
        enforcer.grant_permission("agent-1", Permission.SEND_EMAILS)

        # Should have the permission
        assert enforcer.check_permission("agent-1", Permission.SEND_EMAILS).allowed

        # Revoke it
        enforcer.revoke_permission("agent-1", Permission.SEND_EMAILS)
        assert not enforcer.check_permission("agent-1", Permission.SEND_EMAILS).allowed

    # Test action validation
    @pytest.mark.asyncio
    async def test_validate_action_allowed(self, enforcer):
        """Test action validation for allowed actions."""
        enforcer.assign_role("coder-1", PersonaRole.CODER)
        result = await enforcer.validate_action(
            agent_id="coder-1",
            action="Execute Python script",
            required_permissions=[Permission.EXECUTE_CODE],
        )
        assert result.allowed
        assert not result.missing_permissions

    @pytest.mark.asyncio
    async def test_validate_action_denied(self, enforcer):
        """Test action validation for denied actions."""
        enforcer.assign_role("researcher-1", PersonaRole.RESEARCHER)
        result = await enforcer.validate_action(
            agent_id="researcher-1",
            action="Execute Python script",
            required_permissions=[Permission.EXECUTE_CODE],
        )
        assert not result.allowed
        assert Permission.EXECUTE_CODE in result.missing_permissions

    @pytest.mark.asyncio
    async def test_validate_action_multiple_permissions(self, enforcer):
        """Test validation requiring multiple permissions."""
        enforcer.assign_role("coder-1", PersonaRole.CODER)
        result = await enforcer.validate_action(
            agent_id="coder-1",
            action="Execute shell command with network access",
            required_permissions=[Permission.EXECUTE_SHELL, Permission.MAKE_HTTP_REQUESTS],
        )
        # Coder has MAKE_HTTP_REQUESTS but not EXECUTE_SHELL
        assert not result.allowed
        assert Permission.EXECUTE_SHELL in result.missing_permissions

    # Test tool call validation
    @pytest.mark.asyncio
    async def test_validate_tool_call(self, enforcer):
        """Test tool call validation."""
        enforcer.assign_role("coder-1", PersonaRole.CODER)

        # Allowed tool
        result = await enforcer.validate_tool_call(
            agent_id="coder-1",
            tool_name="execute_code",
            arguments={"code": "print('hello')"},
        )
        assert result.allowed

        # Denied tool
        result = await enforcer.validate_tool_call(
            agent_id="coder-1",
            tool_name="spawn_agent",
            arguments={"type": "researcher"},
        )
        assert not result.allowed

    # Test privilege escalation prevention
    def test_prevent_privilege_escalation(self, enforcer):
        """Test that agents cannot grant themselves higher privileges."""
        enforcer.assign_role("researcher-1", PersonaRole.RESEARCHER)

        # Researcher cannot delegate (can_delegate is False)
        role_perms = get_role_permissions(PersonaRole.RESEARCHER)
        assert not role_perms.can_delegate

    # Test admin role has all permissions
    def test_admin_has_all_permissions(self, enforcer):
        """Test that admin role has all permissions."""
        enforcer.assign_role("admin-1", PersonaRole.ADMIN)

        for permission in Permission:
            result = enforcer.check_permission("admin-1", permission)
            assert result.allowed, f"Admin should have {permission.value}"


class TestRolePermissionDefaults:
    """Tests for default role permission configurations."""

    def test_all_roles_have_defaults(self):
        """Test that all persona roles have default permissions."""
        for role in PersonaRole:
            if role != PersonaRole.CUSTOM:
                assert role in DEFAULT_ROLE_PERMISSIONS

    def test_orchestrator_cannot_execute_code(self):
        """Test that orchestrator role cannot execute code (separation of concerns)."""
        perms = get_role_permissions(PersonaRole.ORCHESTRATOR)
        assert Permission.EXECUTE_CODE not in perms.permissions
        assert Permission.EXECUTE_SHELL not in perms.permissions

    def test_coder_cannot_spawn_agents(self):
        """Test that coder role cannot spawn agents."""
        perms = get_role_permissions(PersonaRole.CODER)
        assert Permission.SPAWN_AGENTS not in perms.permissions

    def test_researcher_is_read_only(self):
        """Test that researcher is essentially read-only."""
        perms = get_role_permissions(PersonaRole.RESEARCHER)
        assert Permission.READ_FILES in perms.permissions
        assert Permission.WRITE_FILES not in perms.permissions
        assert Permission.DELETE_FILES not in perms.permissions
        assert Permission.EXECUTE_CODE not in perms.permissions

    def test_budget_limits_per_role(self):
        """Test that roles have appropriate budget limits."""
        researcher = get_role_permissions(PersonaRole.RESEARCHER)
        orchestrator = get_role_permissions(PersonaRole.ORCHESTRATOR)
        admin = get_role_permissions(PersonaRole.ADMIN)

        assert researcher.max_budget_usd is not None
        assert orchestrator.max_budget_usd is not None
        assert admin.max_budget_usd is None  # Unlimited

        # Orchestrator should have higher budget than researcher
        assert orchestrator.max_budget_usd > researcher.max_budget_usd
