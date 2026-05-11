"""Coverage close-out for backend/config.py.

The Settings dataclass is mostly trivial env-var reads — what matters is
the AUTH_ENABLED property, which is the only thing in the module that
actually computes anything. Lock in its truth table so a future refactor
can't quietly invert the gate.
"""
from unittest.mock import patch

from backend.config import Settings


class TestSettingsAuthEnabled:
    """AUTH_ENABLED is True iff BOTH username and password are non-empty."""

    def test_both_set_returns_true(self):
        s = Settings()
        with patch.object(s, "AUTH_USERNAME", "user"), \
             patch.object(s, "AUTH_PASSWORD", "pw"):
            assert s.AUTH_ENABLED is True

    def test_empty_username_returns_false(self):
        s = Settings()
        with patch.object(s, "AUTH_USERNAME", ""), \
             patch.object(s, "AUTH_PASSWORD", "pw"):
            assert s.AUTH_ENABLED is False

    def test_empty_password_returns_false(self):
        s = Settings()
        with patch.object(s, "AUTH_USERNAME", "user"), \
             patch.object(s, "AUTH_PASSWORD", ""):
            assert s.AUTH_ENABLED is False

    def test_both_empty_returns_false(self):
        s = Settings()
        with patch.object(s, "AUTH_USERNAME", ""), \
             patch.object(s, "AUTH_PASSWORD", ""):
            assert s.AUTH_ENABLED is False
