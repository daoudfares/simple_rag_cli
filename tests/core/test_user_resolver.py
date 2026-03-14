"""
Unit tests for SimpleUserResolver.
"""

import asyncio
from unittest.mock import MagicMock

from src.security.simple_user_resolver import SimpleUserResolver


class TestSimpleUserResolver:
    """Tests for user resolution from request context."""

    def _make_context(self, cookies=None):
        ctx = MagicMock()
        ctx.get_cookie = MagicMock(side_effect=lambda k: (cookies or {}).get(k))
        return ctx

    def test_admin_email_resolves_admin_group(self):
        """An email containing 'admin' should resolve to the admin group."""
        resolver = SimpleUserResolver()
        ctx = self._make_context({"vanna_email": "admin@example.com"})
        user = asyncio.run(resolver.resolve_user(ctx))

        assert user.id == "admin@example.com"
        assert user.email == "admin@example.com"
        assert "admin" in user.group_memberships

    def test_regular_email_resolves_user_group(self):
        """A non-admin email should resolve to the user group."""
        resolver = SimpleUserResolver()
        ctx = self._make_context({"vanna_email": "alice@company.com"})
        user = asyncio.run(resolver.resolve_user(ctx))

        assert user.id == "alice@company.com"
        assert "user" in user.group_memberships
        assert "admin" not in user.group_memberships

    def test_missing_cookie_defaults_to_admin(self):
        """Missing vanna_email cookie should default to admin@example.com."""
        resolver = SimpleUserResolver()
        ctx = self._make_context({})
        user = asyncio.run(resolver.resolve_user(ctx))

        assert user.email == "admin@example.com"
        assert "admin" in user.group_memberships
