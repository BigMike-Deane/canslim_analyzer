"""Buy-time stale-quote guard (ai_trader.stale_quote_reason), 2026-09-24.

A bought-out or delisted name keeps quoting its final trade -- ATAI sat at
$7.35 for two weeks after its Sep-11 cash merger -- so the price alone can't
tell a live stock from a dead one. fetch_live_price records the ET date of the
trade behind each quote; a name that hasn't traded since the previous NYSE
session is never bought or pyramided.
"""
from datetime import date, datetime, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

import backend.ai_trader as ai_trader
from backend.ai_trader import fetch_live_price, stale_quote_reason

ET = ZoneInfo("America/New_York")
THU_OPEN = datetime(2026, 9, 24, 10, 0, tzinfo=ET)     # previous session: Wed Sep-23
MON_OPEN = datetime(2026, 9, 28, 10, 0, tzinfo=ET)     # previous session: Fri Sep-25


@pytest.fixture(autouse=True)
def _clean():
    ai_trader._quote_trade_dates.clear()
    yield
    ai_trader._quote_trade_dates.clear()


@pytest.mark.parametrize("now,last,stale", [
    (THU_OPEN, date(2026, 9, 24), False),   # traded today
    (THU_OPEN, date(2026, 9, 23), False),   # hasn't printed yet this morning
    (THU_OPEN, date(2026, 9, 22), True),
    (THU_OPEN, date(2026, 9, 11), True),    # ATAI
    (MON_OPEN, date(2026, 9, 25), False),   # Friday print, weekend skipped
    (MON_OPEN, date(2026, 9, 24), True),
])
def test_stale_is_older_than_the_previous_session(now, last, stale):
    ai_trader._quote_trade_dates["X"] = last
    assert (stale_quote_reason("X", now=now) is not None) is stale


def test_unknown_trade_date_is_never_called_stale():
    assert stale_quote_reason("NEVER_FETCHED", now=THU_OPEN) is None
    ai_trader._quote_trade_dates["X"] = None
    assert stale_quote_reason("X", now=THU_OPEN) is None


def _resp(status, payload):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


def test_fmp_quote_records_its_trade_date(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    ts = int(datetime(2026, 9, 11, 13, 30, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr("requests.get", lambda *a, **kw: _resp(200, [{"price": 7.35, "timestamp": ts}]))
    assert fetch_live_price("ATAI") == 7.35
    assert ai_trader._quote_trade_dates["ATAI"] == date(2026, 9, 11)
    assert stale_quote_reason("ATAI", now=THU_OPEN) == "stale quote: last trade 2026-09-11"


def test_yahoo_fallback_records_its_trade_date(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "k")
    ts = int(datetime(2026, 9, 23, 20, 0, tzinfo=timezone.utc).timestamp())

    def get(url, *a, **kw):
        if "financialmodelingprep" in url:
            return _resp(500, None)
        return _resp(200, {"chart": {"result": [{"meta": {"regularMarketPrice": 20.0,
                                                           "regularMarketTime": ts}}]}})
    monkeypatch.setattr("requests.get", get)
    assert fetch_live_price("YY") == 20.0
    assert ai_trader._quote_trade_dates["YY"] == date(2026, 9, 23)


def test_a_failed_fetch_clears_the_previous_date(monkeypatch):
    ai_trader._quote_trade_dates["ZZ"] = date(2020, 1, 2)
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr("requests.get", lambda *a, **kw: _resp(500, None))
    monkeypatch.setattr("backend.alpaca_data.last_trade_price", lambda *a, **kw: None)
    assert fetch_live_price("ZZ") is None
    assert "ZZ" not in ai_trader._quote_trade_dates
