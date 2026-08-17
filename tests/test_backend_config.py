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


def test_position_cap_keys_agree():
    """Aug-6 audit misc: buy sizing reads each profile's max_single_position_pct
    while pyramid caps read the global ai_trader.allocation.max_single_position
    (trading_utils.MAX_POSITION_ALLOCATION). The values are equal today; this
    guard fails loudly if either side drifts, because pyramids would silently
    cap at a different level than buys."""
    from config_loader import config
    from backend.trading_utils import MAX_POSITION_ALLOCATION

    profiles = config.get('strategy_profiles', default={}) or {}
    assert profiles, "strategy_profiles missing from config"
    for name, prof in profiles.items():
        pct = (prof or {}).get('max_single_position_pct')
        if pct is None:
            continue
        assert abs(MAX_POSITION_ALLOCATION * 100 - pct) < 1e-6, (
            f"profile '{name}' max_single_position_pct={pct} but the global "
            f"allocation cap (used by pyramid paths) is {MAX_POSITION_ALLOCATION * 100:.1f}"
        )
