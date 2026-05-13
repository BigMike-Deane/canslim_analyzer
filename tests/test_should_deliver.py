"""Tests for backend.email_utils._should_deliver — the notification gate.

Covers every branch added by 741ea59 (per-kind mute + quiet hours) and
1b4277c (fail-open-on-tz-error contract). Before this file the function
had ZERO test coverage despite sitting on the trade-emission path.

Test mechanics: `_should_deliver` defers `from datetime import datetime`
and `from zoneinfo import ZoneInfo` to call time (inside its try block),
so we patch `datetime.datetime` to control the hour and patch
`zoneinfo.ZoneInfo` to simulate tz failure.
"""
from datetime import datetime as real_dt
from unittest.mock import MagicMock, patch

from backend.email_utils import _should_deliver


def _user(mute=None, qs=None, qe=None, score_threshold=None):
    u = MagicMock()
    u.mute_kinds = mute
    u.quiet_hours_start = qs
    u.quiet_hours_end = qe
    u.score_alert_threshold = score_threshold
    return u


def _freeze_hour(hour: int):
    """Patch datetime.datetime so `.now(tz).hour` returns the given value."""
    fixed = real_dt(2026, 1, 15, hour, 30, 0)
    mock_dt = MagicMock()
    mock_dt.now.side_effect = lambda tz=None: fixed
    return patch("datetime.datetime", mock_dt)


class TestUrgentBypass:
    def test_urgent_bypasses_mute(self):
        assert _should_deliver(_user(mute=["trade"]), "trade", "urgent") is True

    def test_urgent_bypasses_quiet_hours(self):
        with _freeze_hour(10):
            assert _should_deliver(_user(qs=9, qe=17), "trade", "urgent") is True

    def test_urgent_bypasses_missing_user(self):
        assert _should_deliver(None, "trade", "urgent") is True


class TestFailOpenOnMissingUser:
    def test_none_user_passes(self):
        assert _should_deliver(None, "trade", "high") is True

    def test_mute_kinds_none_passes(self):
        assert _should_deliver(_user(mute=None), "trade", "high") is True

    def test_mute_kinds_empty_list_passes(self):
        assert _should_deliver(_user(mute=[]), "trade", "high") is True


class TestMute:
    def test_muted_kind_blocks(self):
        assert _should_deliver(_user(mute=["trade"]), "trade", "high") is False

    def test_other_kind_not_muted(self):
        assert _should_deliver(_user(mute=["trade"]), "breakout", "high") is True

    def test_multiple_muted_kinds(self):
        u = _user(mute=["trade", "breakout"])
        assert _should_deliver(u, "trade", "high") is False
        assert _should_deliver(u, "breakout", "high") is False
        assert _should_deliver(u, "system", "high") is True

    def test_non_list_mute_is_ignored(self):
        # mute_kinds is a JSON column; if a bad write lands a string the
        # isinstance(mute, list) guard means we allow rather than crash.
        assert _should_deliver(_user(mute="trade"), "trade", "high") is True


class TestQuietHoursDisabled:
    def test_qs_none_skips_window(self):
        assert _should_deliver(_user(qs=None, qe=7), "trade", "high") is True

    def test_qe_none_skips_window(self):
        assert _should_deliver(_user(qs=22, qe=None), "trade", "high") is True

    def test_qs_eq_qe_skips_window(self):
        assert _should_deliver(_user(qs=5, qe=5), "trade", "high") is True


class TestQuietHoursNonCrossing:
    """qs < qe → in_quiet iff qs <= hour < qe (inclusive start, exclusive end)."""

    def test_hour_inside_window_blocks(self):
        with _freeze_hour(10):
            assert _should_deliver(_user(qs=9, qe=17), "trade", "high") is False

    def test_hour_at_start_inclusive_blocks(self):
        with _freeze_hour(9):
            assert _should_deliver(_user(qs=9, qe=17), "trade", "high") is False

    def test_hour_at_end_exclusive_passes(self):
        with _freeze_hour(17):
            assert _should_deliver(_user(qs=9, qe=17), "trade", "high") is True

    def test_hour_outside_window_passes(self):
        with _freeze_hour(18):
            assert _should_deliver(_user(qs=9, qe=17), "trade", "high") is True


class TestQuietHoursCrossingMidnight:
    """qs > qe → in_quiet iff hour >= qs OR hour < qe."""

    def test_hour_after_qs_blocks(self):
        with _freeze_hour(23):
            assert _should_deliver(_user(qs=22, qe=7), "trade", "high") is False

    def test_hour_before_qe_blocks(self):
        with _freeze_hour(3):
            assert _should_deliver(_user(qs=22, qe=7), "trade", "high") is False

    def test_hour_at_qs_inclusive_blocks(self):
        with _freeze_hour(22):
            assert _should_deliver(_user(qs=22, qe=7), "trade", "high") is False

    def test_hour_at_qe_exclusive_passes(self):
        with _freeze_hour(7):
            assert _should_deliver(_user(qs=22, qe=7), "trade", "high") is True

    def test_hour_between_qe_and_qs_passes(self):
        with _freeze_hour(12):
            assert _should_deliver(_user(qs=22, qe=7), "trade", "high") is True


class TestFailOpenOnTzError:
    """Contract shipped in 1b4277c: when ZoneInfo / clock fails, skip the
    gate (return True) rather than apply a UTC-shifted window."""

    def test_zoneinfo_failure_fails_open(self):
        with patch("zoneinfo.ZoneInfo", side_effect=Exception("tz unavailable")):
            assert _should_deliver(_user(qs=22, qe=7), "trade", "high") is True

    def test_clock_failure_fails_open(self):
        with patch("datetime.datetime") as MockDT:
            MockDT.now.side_effect = OSError("clock failed")
            assert _should_deliver(_user(qs=9, qe=17), "trade", "high") is True


class TestScoreAlertThreshold:
    """Per-user min-score gate: when a user sets a threshold and an alert
    carries a 'score' field, scores below the threshold are suppressed.
    Alerts without a score field are unaffected.
    """

    def test_none_threshold_passes_low_score(self):
        u = _user(score_threshold=None)
        assert _should_deliver(u, "breakout", "high", {"score": 30}) is True

    def test_score_below_threshold_blocks(self):
        u = _user(score_threshold=80)
        assert _should_deliver(u, "breakout", "high", {"score": 72}) is False

    def test_score_at_threshold_passes(self):
        # >= threshold passes (inclusive). 80 should not be considered "below 80".
        u = _user(score_threshold=80)
        assert _should_deliver(u, "breakout", "high", {"score": 80}) is True

    def test_score_above_threshold_passes(self):
        u = _user(score_threshold=80)
        assert _should_deliver(u, "breakout", "high", {"score": 85}) is True

    def test_alert_without_score_field_passes(self):
        # System-wide alerts (e.g. spy_gate_change, morning_briefing) don't
        # carry a 'score' — the threshold must not affect them.
        u = _user(score_threshold=99)
        assert _should_deliver(u, "spy_gate_change", "high", {"ticker": "SPY"}) is True

    def test_alert_with_no_data_passes(self):
        u = _user(score_threshold=99)
        assert _should_deliver(u, "trade", "high", None) is True

    def test_urgent_bypasses_threshold(self):
        u = _user(score_threshold=99)
        assert _should_deliver(u, "trade", "urgent", {"score": 10}) is True

    def test_malformed_score_doesnt_suppress(self):
        # Fail-safe: if score is unparseable, deliver rather than swallow.
        u = _user(score_threshold=80)
        assert _should_deliver(u, "breakout", "high", {"score": "n/a"}) is True

    def test_string_numeric_score_works(self):
        # Some emit sites stringify numbers; float() conversion handles it.
        u = _user(score_threshold=80)
        assert _should_deliver(u, "breakout", "high", {"score": "72"}) is False
        assert _should_deliver(u, "breakout", "high", {"score": "85"}) is True

    def test_threshold_zero_blocks_negative_scores_only(self):
        # Threshold of 0 means "no threshold" in practice — every realistic
        # score is >= 0. Pin this so we don't accidentally treat 0 as falsy.
        u = _user(score_threshold=0)
        assert _should_deliver(u, "breakout", "high", {"score": 0}) is True
        assert _should_deliver(u, "breakout", "high", {"score": 50}) is True
