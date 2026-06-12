"""Unit tests for the trailing-stop cadence gate (June-19 exit-parity item 2).

`trailing_stops_allowed_now` decides whether LIVE trailing stops may fire on
the current cycle. Default OFF = always allow (current behavior). When the
`daily_only` lever is on, trailing stops fire only inside the close window so
live matches the backtester's daily cadence. Hard stops are unaffected (they
never call this gate). These pin the window math so the June-19 landing is safe.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from backend.trading_engine import trailing_stops_allowed_now

ET = ZoneInfo("America/New_York")


class _Cfg:
    """Minimal config stub exposing the dotted-key .get the helper uses."""
    def __init__(self, cadence):
        self._cadence = cadence

    def get(self, key, default=None):
        if key == "ai_trader.trailing_cadence":
            return self._cadence
        return default


def _et(h, m=0):
    return datetime(2026, 6, 12, h, m, tzinfo=ET)


# ── Lever OFF: always allow (preserves current every-scan behavior) ──────────

def test_lever_off_allows_regardless_of_time():
    cfg = _Cfg({"daily_only": False})
    assert trailing_stops_allowed_now(cfg, now_et=_et(2)) is True
    assert trailing_stops_allowed_now(cfg, now_et=_et(10)) is True
    assert trailing_stops_allowed_now(cfg, now_et=_et(23)) is True


def test_missing_config_allows():
    # No trailing_cadence block at all → default off → always allow.
    assert trailing_stops_allowed_now(_Cfg({}), now_et=_et(10)) is True


# ── Lever ON: only the close window (default 15:00–16:00 ET) ─────────────────

DAILY = {"daily_only": True}


def test_in_close_window_allows():
    # The ~15:30 ET close scan lands here.
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(15, 30)) is True


def test_before_window_blocks():
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(14, 0)) is False
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(11, 0)) is False


def test_after_close_blocks():
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(16, 30)) is False


def test_window_edges_inclusive():
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(15, 0)) is True   # window start
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(16, 0)) is True   # close
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(14, 59)) is False
    assert trailing_stops_allowed_now(_Cfg(DAILY), now_et=_et(16, 1)) is False


def test_custom_window_start_honored():
    cfg = _Cfg({"daily_only": True, "window_start_hour_et": 14, "window_start_minute_et": 30})
    assert trailing_stops_allowed_now(cfg, now_et=_et(14, 30)) is True
    assert trailing_stops_allowed_now(cfg, now_et=_et(14, 0)) is False
