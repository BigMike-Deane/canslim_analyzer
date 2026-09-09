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
    _nyse_session_bounds,
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


class TestEarlyCloses:
    """The NYSE closes at 13:00 ET the day after Thanksgiving and on
    Christmas Eve. Hours used to be hardcoded 9:30-16:00, so the trader
    believed the market was open for three hours after it shut and could
    record simulated fills at what was by then the day's closing price --
    roughly 6-8 trade cycles a year that no broker could have filled.

    Owner-approved 2026-09-09. The change only ever makes the system do
    LESS: a 2pm signal on a half-day is deferred to the next session, never
    filled wrongly.
    """

    def _freeze(self, monkeypatch, when):
        from backend import ai_trader

        class FixedDt(datetime):
            @classmethod
            def now(cls, tz=None):
                return when if tz else when.replace(tzinfo=None)

        monkeypatch.setattr(ai_trader, "datetime", FixedDt)
        return ai_trader

    @pytest.mark.parametrize(
        "day", [date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)],
        ids=["black-friday-2026", "christmas-eve-2026", "black-friday-2027"],
    )
    def test_half_day_closes_at_1pm(self, day):
        bounds = _nyse_session_bounds(day)
        assert bounds is not None
        close_et = bounds[1].astimezone(EASTERN_TZ)
        assert (close_et.hour, close_et.minute) == (13, 0), (
            f"{day} closes at {close_et:%H:%M} ET, expected 13:00"
        )

    def test_open_before_the_early_close(self, monkeypatch):
        ait = self._freeze(monkeypatch, _et(2026, 11, 27, 11))
        assert ait.is_market_open() is True

    def test_closed_after_the_early_close(self, monkeypatch):
        # 14:00 on Black Friday. This returned True before the fix.
        ait = self._freeze(monkeypatch, _et(2026, 11, 27, 14))
        assert ait.is_market_open() is False

    def test_a_normal_day_still_runs_to_4pm(self, monkeypatch):
        # Guards against over-correcting: ordinary sessions are unchanged.
        ait = self._freeze(monkeypatch, _et(2026, 11, 25, 15))
        assert ait.is_market_open() is True

    def test_still_closed_before_the_open(self, monkeypatch):
        ait = self._freeze(monkeypatch, _et(2026, 11, 27, 8))
        assert ait.is_market_open() is False

    def test_half_day_is_still_a_trading_day(self, monkeypatch):
        # It is a session, just a short one -- the scan throttle must not
        # treat it as a weekend.
        assert is_trading_day(_et(2026, 11, 27, 11)) is True

    def test_falls_back_to_a_fixed_session_if_hours_unavailable(self, monkeypatch):
        # Degrade to the historical 9:30-16:00 rather than refusing to trade.
        from unittest.mock import patch as _patch

        ait = self._freeze(monkeypatch, _et(2026, 11, 25, 15))
        with _patch.object(ait, "_nyse_session_bounds", return_value=None):
            assert ait.is_market_open() is True


class TestTheDependencyShipsInTheImage:
    """The container installs backend/requirements.txt, NOT the root one.

    2026-09-09: the dependency was first added to the ROOT requirements.txt,
    so the image built without it. Nothing failed loudly -- the fallback in
    _nyse_holidays_for_year() quietly returned None and production kept using
    the hardcoded 2026-2027 table, i.e. the bug was still live while the
    tests were green. Graceful degradation hides packaging mistakes, so pin
    the packaging too.
    """

    def test_calendar_dep_is_in_the_file_the_dockerfile_installs(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        dockerfile = (root / "Dockerfile").read_text()
        assert "COPY backend/requirements.txt" in dockerfile, (
            "Dockerfile no longer installs backend/requirements.txt; update "
            "this test to point at whatever it installs now"
        )
        installed = (root / "backend" / "requirements.txt").read_text()
        assert "pandas-market-calendars" in installed, (
            "pandas-market-calendars missing from backend/requirements.txt -- "
            "the image would silently fall back to the hardcoded holiday table"
        )
