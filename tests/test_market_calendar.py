"""
NYSE session calendar.

Origin (2026-09-09): `_get_us_market_holidays()` hardcoded NYSE closures for
2026 and 2027 ONLY, baked into a module-level constant. From 2028-01-01 that
set holds nothing for the current year, so `is_trading_day()` and
`is_market_open()` would have reported every holiday as a normal session and
the scheduler would have run trading cycles, stop-loss checks and snapshots
against a closed market. Silent -- no crash, no alert.

These tests pin three things:
  1. the swap changed NOTHING for 2026/2027 (parity with the old table),
  2. it is correct for years past the old table (the actual bug),
  3. it degrades to the old table if the calendar library is missing.
"""

import os
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.ai_trader import (
    EASTERN_TZ,
    _get_us_market_holidays,
    _is_market_holiday,
    _nyse_holidays_for_year,
    is_trading_day,
)

pmc = pytest.importorskip(
    "pandas_market_calendars",
    reason="NYSE calendar backend not installed; the fallback table is covered separately",
)


def _et(y, m, d, hour=12):
    return datetime(y, m, d, hour, 0, tzinfo=EASTERN_TZ)


class TestParityWithTheOldHardcodedTable:
    """The swap must not move any date the old table already covered."""

    @pytest.mark.parametrize("year", [2026, 2027])
    def test_identical_to_the_hardcoded_table(self, year):
        hardcoded = {d for d in _get_us_market_holidays() if d.year == year}
        derived = set(_nyse_holidays_for_year(year))
        assert derived == hardcoded, (
            f"{year} differs: only-hardcoded={sorted(hardcoded - derived)} "
            f"only-derived={sorted(derived - hardcoded)}"
        )

    def test_labor_day_2026_still_closed(self):
        # The day the Sep-8 snapshot gap was first misread as an outage.
        assert is_trading_day(_et(2026, 9, 7)) is False

    def test_ordinary_weekday_still_open(self):
        assert is_trading_day(_et(2026, 9, 9)) is True

    def test_weekend_still_closed(self):
        assert is_trading_day(_et(2026, 9, 5)) is False   # Saturday
        assert is_trading_day(_et(2026, 9, 6)) is False   # Sunday


class TestBeyondTheOldTable:
    """The actual bug: years the hardcoded set never covered."""

    def test_hardcoded_table_really_did_stop(self):
        # Guards the premise -- if someone extends the table by hand, the
        # tests below stop proving anything.
        assert not [d for d in _get_us_market_holidays() if d.year >= 2028]

    @pytest.mark.parametrize(
        "day",
        [
            date(2028, 1, 17),   # MLK Day
            date(2028, 5, 29),   # Memorial Day
            date(2028, 7, 4),    # Independence Day
            date(2028, 9, 4),    # Labor Day
            date(2028, 11, 23),  # Thanksgiving
            date(2028, 12, 25),  # Christmas
        ],
    )
    def test_2028_holidays_are_not_trading_days(self, day):
        # Every one of these returned True before the fix.
        assert is_trading_day(_et(day.year, day.month, day.day)) is False

    def test_new_years_day_on_a_saturday_does_not_close_the_friday(self):
        # 2028-01-01 is a Saturday. The NYSE does NOT observe it on the
        # preceding Friday, so 2027-12-31 is a normal session -- exactly the
        # rule a handwritten table tends to get wrong.
        assert is_trading_day(_et(2027, 12, 31)) is True
        assert len(_nyse_holidays_for_year(2028)) == 9

    def test_still_correct_a_decade_out(self):
        closures = _nyse_holidays_for_year(2035)
        assert closures, "no closures computed for 2035"
        assert date(2035, 12, 25) in closures
        assert date(2035, 7, 4) in closures


class TestFallbackWhenCalendarUnavailable:
    """An import failure must degrade to the old table, never to
    'every day is a trading day'."""

    def test_falls_back_to_the_hardcoded_table(self):
        _nyse_holidays_for_year.cache_clear()
        try:
            with patch("backend.ai_trader._nyse_holidays_for_year", return_value=None):
                assert _is_market_holiday(date(2026, 9, 7)) is True    # in the table
                assert _is_market_holiday(date(2026, 9, 9)) is False
        finally:
            _nyse_holidays_for_year.cache_clear()

    def test_no_network_needed(self, monkeypatch):
        # The container must not need egress to compute a session calendar.
        import socket

        def _blocked(*a, **k):
            raise AssertionError("calendar attempted a network connection")

        monkeypatch.setattr(socket.socket, "connect", _blocked)
        _nyse_holidays_for_year.cache_clear()
        try:
            assert date(2029, 12, 25) in _nyse_holidays_for_year(2029)
        finally:
            _nyse_holidays_for_year.cache_clear()


class TestSessionClockUnchanged:
    """The 9:30-16:00 clock is deliberately NOT touched by this change.

    The NYSE closes at 13:00 on ~2 days a year (day after Thanksgiving,
    Christmas Eve) and this code has never modelled that. Moving to
    schedule-based hours would change LIVE TRADING behaviour on those days,
    so it is a separate, owner-approved decision. This test pins the current
    behaviour so that change cannot land by accident.
    """

    def test_half_day_still_treated_as_a_full_session(self):
        # 2026-11-27, the day after Thanksgiving: real NYSE close is 13:00 ET.
        assert is_trading_day(_et(2026, 11, 27)) is True
