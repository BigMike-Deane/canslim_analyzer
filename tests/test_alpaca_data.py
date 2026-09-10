"""
Alpaca market data (2026-09-10): the ATR stop's daily bars move from Yahoo's
throttled chart API to Alpaca's consolidated tape, and IEX becomes a
last-resort live price.

What must hold:

  * PARITY -- the same bars give the same stop whichever source served them.
    Only the data source changed, never the ATR math.
  * FALLBACK ORDER -- ATR: Alpaca, then Yahoo. Live price: FMP, then Yahoo,
    then Alpaca IEX (thin, one venue -- last resort only).
  * NO KEYS, NO CALLS -- without credentials the old path runs untouched.
  * THE FREE PLAN'S LIMITS ARE RESPECTED -- SIP bars end 16 min ago (newer
    is a 403), and an IEX trade older than 15 min is not a live price.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from backend import alpaca_data
from backend.trading_engine import _atr_stop_cache

NOW = datetime(2026, 9, 10, 15, 30, tzinfo=timezone.utc)


def _bars(n=20):
    # Rising, volatile candles: ATR ~8 on a ~100 stock -> widened stop.
    return [{"t": f"2026-08-{i + 1:02d}T04:00:00Z", "o": 100 + i * 0.1,
             "h": 105.0 + i * 0.1, "l": 97.0 + i * 0.1, "c": 100.0 + i * 0.1, "v": 1000}
            for i in range(n)]


def _yahoo_payload(bars):
    return {"chart": {"result": [{"indicators": {"quote": [{
        "high": [b["h"] for b in bars], "low": [b["l"] for b in bars],
        "close": [b["c"] for b in bars]}]}}]}}


class Router:
    """requests.get stand-in: answers per host, records every call."""

    def __init__(self, alpaca=None, yahoo=None, fmp=None, alpaca_status=200,
                 yahoo_status=200, fmp_status=200, latest_trade=None):
        self.calls = []
        self.alpaca, self.yahoo, self.fmp = alpaca, yahoo, fmp
        self.alpaca_status, self.yahoo_status, self.fmp_status = alpaca_status, yahoo_status, fmp_status
        self.latest_trade = latest_trade

    def __call__(self, url, *a, params=None, **kw):
        self.calls.append((url, params))
        resp = MagicMock()
        if "data.alpaca.markets" in url:
            resp.status_code = self.alpaca_status
            if "/trades/latest" in url:
                resp.json.return_value = {"trade": self.latest_trade}
            else:
                sym = params["symbols"]
                resp.json.return_value = {"bars": {sym: self.alpaca} if self.alpaca else {}}
            resp.text = ""
        elif "financialmodelingprep" in url:
            resp.status_code = self.fmp_status
            resp.json.return_value = [{"price": self.fmp}] if self.fmp else []
        else:
            resp.status_code = self.yahoo_status
            resp.json.return_value = self.yahoo or {}
        return resp

    def hosts(self):
        return ["alpaca" if "alpaca" in u else "fmp" if "financialmodelingprep" in u else "yahoo"
                for u, _ in self.calls]


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    _atr_stop_cache.clear()
    import time
    monkeypatch.setattr(time, "sleep", lambda *a, **k: None)
    yield
    _atr_stop_cache.clear()


# ---------------------------------------------------------------- the data client

class TestDailyBars:

    def test_no_keys_makes_no_call(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
        router = Router(alpaca=_bars())
        monkeypatch.setattr("requests.get", router)
        assert alpaca_data.daily_bars("AAA") is None and router.calls == []

    def test_asks_for_split_adjusted_sip_bars_ending_16_minutes_ago(self, keys, monkeypatch):
        router = Router(alpaca=_bars())
        monkeypatch.setattr("requests.get", router)
        assert len(alpaca_data.daily_bars("BRK-B", now=NOW)) == 20
        url, params = router.calls[0]
        assert url == "https://data.alpaca.markets/v2/stocks/bars"
        assert params["symbols"] == "BRK.B"                       # Alpaca spelling
        assert params["feed"] == "sip" and params["adjustment"] == "split"
        assert params["end"] == (NOW - timedelta(minutes=16)).isoformat()

    def test_errors_return_none_never_raise(self, keys, monkeypatch):
        monkeypatch.setattr("requests.get", Router(alpaca=_bars(), alpaca_status=403))
        assert alpaca_data.daily_bars("AAA") is None

        def boom(*a, **k):
            raise ConnectionError("down")
        monkeypatch.setattr("requests.get", boom)
        assert alpaca_data.daily_bars("AAA") is None


class TestLastTrade:

    def test_fresh_iex_trade_is_used(self, keys, monkeypatch):
        t = (NOW - timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S") + ".929936936Z"
        monkeypatch.setattr("requests.get", Router(latest_trade={"p": 31.2, "t": t}))
        assert alpaca_data.last_trade_price("CARE", now=NOW) == 31.2

    def test_stale_iex_trade_is_not_a_live_price(self, keys, monkeypatch):
        t = (NOW - timedelta(minutes=40)).strftime("%Y-%m-%dT%H:%M:%SZ")
        monkeypatch.setattr("requests.get", Router(latest_trade={"p": 31.2, "t": t}))
        assert alpaca_data.last_trade_price("CARE", now=NOW) is None

    def test_nanosecond_timestamps_parse(self):
        ts = alpaca_data.parse_ts("2026-09-10T15:24:00.929936936Z")
        assert ts == datetime(2026, 9, 10, 15, 24, 0, 929936, tzinfo=timezone.utc)
        assert alpaca_data.parse_ts("2026-09-10T15:24:00Z").second == 0
        assert alpaca_data.parse_ts("garbage") is None


# ---------------------------------------------------------------- ATR stop wiring

class TestAtrSource:

    def test_same_bars_same_stop_from_either_source(self, keys, monkeypatch):
        from backend import ai_trader
        bars = _bars()
        monkeypatch.setattr("requests.get", Router(alpaca=bars))
        via_alpaca = ai_trader.calculate_atr_stop("AAA", current_price=100.0, base_stop_pct=7.0)
        _atr_stop_cache.clear()
        monkeypatch.setattr("requests.get", Router(alpaca=None, yahoo=_yahoo_payload(bars)))
        via_yahoo = ai_trader.calculate_atr_stop("AAA", current_price=100.0, base_stop_pct=7.0)
        assert via_alpaca == via_yahoo
        assert via_alpaca > 7.0                   # the widening actually ran

    def test_alpaca_answers_so_yahoo_is_never_called(self, keys, monkeypatch):
        from backend import ai_trader
        router = Router(alpaca=_bars(), yahoo=_yahoo_payload(_bars()))
        monkeypatch.setattr("requests.get", router)
        ai_trader.calculate_atr_stop("AAA", current_price=100.0, base_stop_pct=7.0)
        assert router.hosts() == ["alpaca"]

    @pytest.mark.parametrize("alpaca,status", [(None, 200), (_bars(10), 200), (_bars(), 429)])
    def test_falls_back_to_yahoo(self, keys, monkeypatch, alpaca, status):
        # no bars / too few for a 14-day ATR / throttled
        from backend import ai_trader
        router = Router(alpaca=alpaca, alpaca_status=status, yahoo=_yahoo_payload(_bars()))
        monkeypatch.setattr("requests.get", router)
        stop = ai_trader.calculate_atr_stop("AAA", current_price=100.0, base_stop_pct=7.0)
        assert router.hosts() == ["alpaca", "yahoo"] and stop > 7.0

    def test_both_sources_down_keeps_the_last_good_stop(self, keys, monkeypatch):
        from backend import ai_trader
        from backend.trading_engine import cache_atr_stop
        cache_atr_stop("AAA", 14.0)
        monkeypatch.setattr("requests.get", Router(alpaca=None, yahoo_status=503))
        assert ai_trader.calculate_atr_stop("AAA", current_price=100.0, base_stop_pct=7.0) == 14.0

    def test_success_caches_for_the_broker_mirror(self, keys, monkeypatch):
        from backend import ai_trader
        from backend.trading_engine import get_cached_atr_stop
        monkeypatch.setattr("requests.get", Router(alpaca=_bars()))
        stop = ai_trader.calculate_atr_stop("AAA", current_price=100.0, base_stop_pct=7.0)
        assert get_cached_atr_stop("AAA") == stop


# ---------------------------------------------------------------- live price order

class TestLivePriceOrder:

    def test_fmp_first_nothing_else_called(self, keys, monkeypatch):
        from backend.ai_trader import fetch_live_price
        monkeypatch.setenv("FMP_API_KEY", "x")
        router = Router(fmp=42.0)
        monkeypatch.setattr("requests.get", router)
        assert fetch_live_price("AAA") == 42.0 and router.hosts() == ["fmp"]

    def test_iex_only_after_fmp_and_yahoo_fail(self, keys, monkeypatch):
        from backend.ai_trader import fetch_live_price
        monkeypatch.setenv("FMP_API_KEY", "x")
        t = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        router = Router(fmp_status=500, yahoo_status=503, latest_trade={"p": 31.2, "t": t})
        monkeypatch.setattr("requests.get", router)
        assert fetch_live_price("CARE") == 31.2
        assert router.hosts() == ["fmp", "yahoo", "alpaca"]

    def test_no_keys_all_down_is_still_none(self, monkeypatch):
        from backend.ai_trader import fetch_live_price
        monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
        monkeypatch.setenv("FMP_API_KEY", "x")
        monkeypatch.setattr("requests.get", Router(fmp_status=500, yahoo_status=503))
        assert fetch_live_price("AAA") is None
