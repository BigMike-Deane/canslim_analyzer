"""
Coverage push #2 for data_fetcher.py — squeeze the gaps left by 2866c89.

After 2866c89 (44% → 74%), the genuinely-uncovered SAFE branches are mostly
exception-handler bodies in the 13 FMP fetcher wrappers, the insider-title
detection buckets at lines 1301-1312, count-based sentiment fallbacks at 1327-
1329, save_ticker yahoo_info partial fields, the _fmp_get retry-then-success
path, and a handful of load_cache_from_db field paths whose `_updated_at`
columns weren't traversed in the Round-1 record fixtures.

Eval-window-safe: HTTP-only mocks via monkeypatch on data_fetcher._fmp_get and
on the bare `requests.get` used by Yahoo wrappers; in-memory SQLite for any
helper that touches StockDataCache (per project rule, no Mock(spec=Session)).

Triage:
  Tier 1 (silent-failure surface — exception bodies that swallow API errors
  and return {} so the scorer keeps seeing stale data):
    - 13 FMP fetcher exception handlers
    - load_cache_from_db field-by-field _updated_at branches
    - save_ticker_to_db_cache yahoo_info partial-update branches
    - _fmp_get 429-then-200 retry path

  Tier 2 (display/UX edges, real but lower blast radius):
    - calculate_index_m_score ma_50<=0 fallback
    - fetch_market_direction_data trend-bucket boundaries
    - insider title detection (CFO/COO/PRESIDENT/DIRECTOR/10%/OFFICER)
    - insider count-based sentiment when net_value below ±$100k threshold

  Tier 3 (DELIBERATELY skipped — covered by other tests or unmockable):
    - DataFetcher.get_stock_data orchestration (1771-2090): existing
      test_data_fetcher.py header explicitly defers it as integration-tested.
    - fetch_finviz_institutional HTML scraping: brittle by intent.
    - Module-import error paths (27-29): require unloading the module.
    - Cache eviction at 25k entries (403-405): too expensive even with mocks.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

import data_fetcher
from backend.database import Base, DelistedTicker, StockDataCache


# ── Fixtures (mirror test_data_fetcher.py to stay self-contained) ────────────


@pytest.fixture
def db_session(monkeypatch):
    """Fresh in-memory SQLite, wired into backend.database.SessionLocal."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    import backend.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", Session)
    return Session()


@pytest.fixture
def fmp_key(monkeypatch):
    monkeypatch.setattr(data_fetcher, "FMP_API_KEY", "test-key-1234")


@pytest.fixture(autouse=True)
def _reset_module_caches():
    """Clear all module-level caches between tests."""
    with data_fetcher._cached_data_lock:
        data_fetcher._cached_data.clear()
    with data_fetcher._freshness_lock:
        data_fetcher._data_freshness_cache.clear()
    with data_fetcher._stats_lock:
        data_fetcher._rate_limit_stats["errors_429"] = 0
        data_fetcher._rate_limit_stats["total_requests"] = 0
    data_fetcher._cache_hit_count = 0
    data_fetcher._cache_miss_count = 0
    data_fetcher._known_delisted_cache.clear()
    data_fetcher._delisted_cache_loaded = False
    data_fetcher._db_cache_loaded = False
    yield


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else []
    resp.text = text
    return resp


def _raises(*_a, **_k):
    """Force a fetcher's _fmp_get call to bubble through its except block."""
    raise RuntimeError("simulated network failure")


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Fetcher exception-handler bodies (13 helpers, 1 mock pattern)
#
# Each helper wraps its FMP/yahoo call in a `try/except Exception` that logs
# debug and returns {}. Without these tests, a regression that turned an
# error into something other than {} (e.g. a partial dict, or a re-raise)
# would silently propagate into the scorer.
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetcherExceptionHandlers:
    """Tier 1: every FMP fetcher must swallow exceptions and return {} so a
    transient API hiccup doesn't poison the in-memory cache or crash the scan."""

    def test_fetch_fmp_profile_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:769-770 (profile outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_profile("AAPL") == {}

    def test_fetch_fmp_quote_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:796-797 (quote outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_quote("AAPL") == {}

    def test_fetch_fmp_key_metrics_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:821-822 (key_metrics outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_key_metrics("AAPL") == {}

    def test_fetch_fmp_earnings_swallows_quarterly_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:850-852/863-864 — both quarterly AND annual
        try/except blocks must independently swallow, returning the default
        empty-list result dict."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        result = data_fetcher.fetch_fmp_earnings("AAPL")
        # Even on full failure, the function returns the seeded result dict.
        assert result == {"quarterly_eps": [], "annual_eps": []}

    def test_fetch_fmp_price_target_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:951-952 (price_target outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_price_target("AAPL") == {}

    def test_fetch_fmp_earnings_surprise_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:1005-1006 (earnings_surprise outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_earnings_surprise("AAPL") == {}

    def test_fetch_fmp_earnings_calendar_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:1107-1109 (earnings_calendar outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_earnings_calendar("AAPL") == {}

    def test_fetch_fmp_analyst_estimates_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:1172-1173 (analyst_estimates outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_analyst_estimates("AAPL") == {}

    def test_fetch_fmp_revenue_swallows_both_branches(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:1192-1193/1203-1204 — quarterly AND annual
        wrapped independently. Returns the empty seed dict on full failure."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        result = data_fetcher.fetch_fmp_revenue("AAPL")
        assert result == {"quarterly_revenue": [], "annual_revenue": []}

    def test_fetch_fmp_balance_sheet_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:1227-1228 (balance_sheet outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_balance_sheet("AAPL") == {}

    def test_fetch_fmp_insider_trading_returns_empty_on_exception(self, fmp_key, monkeypatch):
        """Branch: data_fetcher.py:1348-1351 (insider_trading outer except)."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        assert data_fetcher.fetch_fmp_insider_trading("AAPL") == {}

    def test_fetch_fmp_institutional_falls_back_to_finviz_on_exception(
        self, fmp_key, monkeypatch
    ):
        """Branch: data_fetcher.py:902-903 — when FMP raises, the function
        must fall through to fetch_finviz_institutional rather than re-raise."""
        monkeypatch.setattr(data_fetcher, "_fmp_get", _raises)
        # Stub finviz so we don't actually hit the web.
        monkeypatch.setattr(
            data_fetcher, "fetch_finviz_institutional", lambda t: 42.5
        )
        assert data_fetcher.fetch_fmp_institutional("AAPL") == 42.5


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Insider title detection buckets (1301-1312)
#
# Each `elif` branch is keyed off case-insensitive substring match in the
# reportingName / typeOfOwner. Every branch is currently uncovered.
# ═══════════════════════════════════════════════════════════════════════════════


def _insider_record(reporting_name, type_of_owner="officer", price=200,
                    shares=1000, days_ago=10, transaction_type="P-Purchase"):
    """Build one FMP insider-trading record matching the production schema."""
    when = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {
        "transactionDate": when,
        "transactionType": transaction_type,
        "securitiesTransacted": shares,
        "price": price,
        "typeOfOwner": type_of_owner,
        "reportingName": reporting_name,
    }


class TestInsiderTitleDetection:
    """Tier 2: each title bucket at data_fetcher.py:1301-1312. The buckets are
    branches off the LARGEST single buy — so each test makes the named title's
    purchase the largest one. CEO is already covered by the existing suite."""

    def test_cfo_detected_from_reporting_name(self, fmp_key, monkeypatch):
        """Branch: 1301-1302 — 'CFO' or 'CHIEF FINANCIAL' substring."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Jane Doe CFO"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "CFO"

    def test_cfo_detected_from_chief_financial_phrase(self, fmp_key, monkeypatch):
        """Branch: 1301-1302 — 'CHIEF FINANCIAL' phrase variant."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Jane Doe Chief Financial Officer"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "CFO"

    def test_coo_detected_from_reporting_name(self, fmp_key, monkeypatch):
        """Branch: 1303-1304 — 'COO' or 'CHIEF OPERATING' substring."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Sam Lee COO"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "COO"

    def test_president_detected(self, fmp_key, monkeypatch):
        """Branch: 1305-1306 — 'PRESIDENT' substring."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Alice Z. President"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "PRESIDENT"

    def test_director_detected_from_type_of_owner(self, fmp_key, monkeypatch):
        """Branch: 1307-1308 — 'DIRECTOR' in typeOfOwner field."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Bob Smith", type_of_owner="director"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "DIRECTOR"

    def test_director_detected_from_reporting_name(self, fmp_key, monkeypatch):
        """Branch: 1307-1308 — 'DIRECTOR' substring fallback in reportingName
        (some FMP records put title in name, not typeOfOwner)."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Bob Smith Director", type_of_owner="officer"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "DIRECTOR"

    def test_ten_percent_owner_detected(self, fmp_key, monkeypatch):
        """Branch: 1309-1310 — '10%' substring in typeOfOwner."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Beneficial Holder", type_of_owner="10% owner"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "10% OWNER"

    def test_officer_fallback_when_no_keywords_match(self, fmp_key, monkeypatch):
        """Branch: 1311-1312 — final else uses typeOfOwner verbatim, or 'OFFICER'.
        This is the riskiest branch in the chain because every unrecognized
        title silently lands here."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Anonymous Holder", type_of_owner="senior vp"),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # typeOfOwner is non-empty truthy → use it verbatim.
        assert result["largest_buyer_title"] == "senior vp"

    def test_officer_fallback_when_type_of_owner_empty(self, fmp_key, monkeypatch):
        """Branch: 1311-1312 — `insider_title or "OFFICER"` final fallback."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(json_data=[
                _insider_record("Anonymous Holder", type_of_owner=""),
            ]),
        )
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["largest_buyer_title"] == "OFFICER"


class TestInsiderCountBasedSentiment:
    """Tier 2: count-based sentiment fallback at 1326-1331. Triggers when
    net_value is between ±$100k (ambiguous on $) but counts heavily skew."""

    def test_bullish_when_buy_count_dominates_but_net_value_small(
        self, fmp_key, monkeypatch
    ):
        """Branch: 1326-1327 — `buy_count > sell_count * 1.5` AND |net_value|<100k."""
        records = [
            # Many small buys, one tiny sell. Net $ ~ $20k (below threshold).
            _insider_record("A", price=10, shares=500, days_ago=10),  # +5k
            _insider_record("B", price=10, shares=500, days_ago=11),  # +5k
            _insider_record("C", price=10, shares=500, days_ago=12),  # +5k
            _insider_record("D", price=10, shares=500, days_ago=13),  # +5k
            _insider_record("E", price=10, shares=500, days_ago=14),  # +5k
            _insider_record("F", price=10, shares=500, days_ago=15,
                            transaction_type="S-Sale"),  # -5k
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # 5 buys vs 1 sell → 5 > 1.5 → bullish even though net_value < $100k
        assert result["sentiment"] == "bullish"
        assert result["buy_count"] == 5
        assert result["sell_count"] == 1

    def test_bearish_when_sell_count_dominates_but_net_value_small(
        self, fmp_key, monkeypatch
    ):
        """Branch: 1328-1329 — `sell_count > buy_count * 1.5`."""
        records = [
            _insider_record("A", price=10, shares=500, days_ago=10),  # buy +5k
            _insider_record("B", price=10, shares=500, days_ago=11,
                            transaction_type="S-Sale"),
            _insider_record("C", price=10, shares=500, days_ago=12,
                            transaction_type="S-Sale"),
            _insider_record("D", price=10, shares=500, days_ago=13,
                            transaction_type="S-Sale"),
            _insider_record("E", price=10, shares=500, days_ago=14,
                            transaction_type="S-Sale"),
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # 4 sells vs 1 buy → 4 > 1.5 → bearish.
        assert result["sentiment"] == "bearish"

    def test_neutral_when_counts_balanced_and_net_value_small(
        self, fmp_key, monkeypatch
    ):
        """Branch: 1330-1331 — final neutral fallback."""
        records = [
            _insider_record("A", price=10, shares=500, days_ago=10),
            _insider_record("B", price=10, shares=500, days_ago=11,
                            transaction_type="S-Sale"),
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # 1 buy, 1 sell, net $0 → neutral.
        assert result["sentiment"] == "neutral"


class TestInsiderTradingEdgeCases:
    """Tier 2: data branches around insider_trading not covered by the round-1
    suite. All sit in the per-record loop at 1267-1318."""

    def test_returns_empty_dict_when_no_data(self, fmp_key, monkeypatch):
        """Branch: 1255 — empty array short-circuit before the loop."""
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=[]))
        assert data_fetcher.fetch_fmp_insider_trading("AAPL") == {}

    def test_skips_record_with_missing_transaction_date(self, fmp_key, monkeypatch):
        """Branch: 1271 — `if not trade_date_str: continue`."""
        records = [
            {"transactionDate": "", "transactionType": "P-Purchase",
             "securitiesTransacted": 1000, "price": 10,
             "typeOfOwner": "officer", "reportingName": "X"},
            _insider_record("Real Buyer"),
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        # Only the second record should have been counted.
        assert result["buy_count"] == 1

    def test_skips_record_with_unparseable_transaction_date(
        self, fmp_key, monkeypatch
    ):
        """Branch: 1277-1278 — ValueError from datetime.strptime ⇒ continue."""
        records = [
            {"transactionDate": "not-a-date", "transactionType": "P-Purchase",
             "securitiesTransacted": 1000, "price": 10,
             "typeOfOwner": "officer", "reportingName": "X"},
            _insider_record("Real Buyer"),
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_insider_trading("AAPL")
        assert result["buy_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_earnings_calendar additional branches
# ═══════════════════════════════════════════════════════════════════════════════


class TestEarningsCalendarBranches:
    """Tier 1: gaps in calendar parsing — empty payloads, missing fields,
    forward-projection fallback when no future earnings record exists yet."""

    def test_returns_empty_on_empty_payload(self, fmp_key, monkeypatch):
        """Branch: 1034 — `if not all_data: return {}`."""
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=[]))
        assert data_fetcher.fetch_fmp_earnings_calendar("AAPL") == {}

    def test_returns_empty_when_filtered_to_zero_records(self, fmp_key, monkeypatch):
        """Branch: 1041 — payload non-empty but no records for our symbol."""
        records = [
            {"symbol": "MSFT", "date": "2026-07-01",
             "epsActual": 1.0, "epsEstimated": 0.9},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        assert data_fetcher.fetch_fmp_earnings_calendar("AAPL") == {}

    def test_skips_record_with_missing_date_field(self, fmp_key, monkeypatch):
        """Branch: 1056 — `if not item_date_str: continue`. The sort step at
        1038 will place '' last, so we put a real future record first to keep
        the helper from returning {}."""
        future = (date.today() + timedelta(days=30)).isoformat()
        records = [
            {"symbol": "AAPL", "date": future,
             "epsActual": None, "epsEstimated": 1.5},
            # Empty date — must be skipped without crashing.
            {"symbol": "AAPL", "date": "",
             "epsActual": 1.2, "epsEstimated": 1.0},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_calendar("AAPL")
        assert result["next_earnings_date"] == future

    def test_estimates_next_earnings_when_only_past_records(self, fmp_key, monkeypatch):
        """Branch: 1085-1098 — when no future earnings record exists, the
        helper projects forward 90 days at a time from the most recent past."""
        # Last earnings 30 days ago, no future record.
        past = (date.today() - timedelta(days=30)).isoformat()
        records = [
            {"symbol": "AAPL", "date": past,
             "epsActual": 2.0, "epsEstimated": 1.8},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_earnings_calendar("AAPL")
        # Projected next earnings should be ~60 days out (90 - 30).
        assert result["next_earnings_date"] is not None
        # And days_to_earnings is positive.
        assert result["days_to_earnings"] > 0
        assert result["days_to_earnings"] <= 90


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_fmp_analyst_estimates fallback paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalystEstimatesFallbacks:
    """Tier 1: when current-year or prior-year aren't found in the payload, the
    helper falls back to data[0] / data[1] (lines 1146-1149) so it still
    produces a reading."""

    def test_falls_back_to_first_two_records_when_year_filter_misses(
        self, fmp_key, monkeypatch
    ):
        """Branch: 1147 + 1149 — neither current_year nor (current_year-1)
        present, so `current = data[0]` and `prior = data[1]`."""
        # Use stale dates (well in the past) so the year-match loop misses.
        records = [
            {"date": "2020-12-31", "epsAvg": 5.5,
             "numberAnalystsEstimatedEps": 10},
            {"date": "2019-12-31", "epsAvg": 5.0,
             "numberAnalystsEstimatedEps": 8},
        ]
        monkeypatch.setattr(data_fetcher, "_fmp_get",
                            lambda *a, **k: _mock_response(json_data=records))
        result = data_fetcher.fetch_fmp_analyst_estimates("AAPL")
        assert result["eps_estimate_current"] == pytest.approx(5.5)
        assert result["eps_estimate_prior"] == pytest.approx(5.0)
        # Revision (5.5-5.0)/5.0 = 10% → "up"
        assert result["estimate_revision_trend"] == "up"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — save_ticker_to_db_cache uncovered branches
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveTickerToDbCacheBranches:
    """Tier 1: branches inside save_ticker_to_db_cache that the round-1 suite
    didn't traverse — yahoo_info partial fields, key_metrics forward_pe
    non-None path, institutional non-numeric data, no-DB-session early
    return, and the earnings_calendar `else` branch when next_date is
    already a date object."""

    def test_yahoo_info_writes_roe_and_metrics_timestamp(self, db_session):
        """Branch: 296-297 — yahoo_info path with `roe is not None`."""
        data_fetcher.save_ticker_to_db_cache("YH1", "yahoo_info", {
            "roe": 0.18,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="YH1").one()
        assert rec.roe == pytest.approx(0.18)
        assert rec.metrics_updated_at is not None

    def test_yahoo_info_writes_institutional_pct(self, db_session):
        """Branch: 298-300 — institutional_holders_pct write + timestamp."""
        data_fetcher.save_ticker_to_db_cache("YH2", "yahoo_info", {
            "institutional_holders_pct": 72.4,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="YH2").one()
        assert rec.institutional_holders_pct == pytest.approx(72.4)
        assert rec.institutional_updated_at is not None

    def test_yahoo_info_writes_cash_and_debt_when_either_present(self, db_session):
        """Branch: 305-308 — `if cash_and_equivalents OR total_debt` → both
        get assigned, and balance_updated_at is touched. Even if only debt
        is present, cash will be assigned (its None value)."""
        data_fetcher.save_ticker_to_db_cache("YH3", "yahoo_info", {
            "total_debt": 5_000_000_000,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="YH3").one()
        assert rec.total_debt == 5_000_000_000
        assert rec.balance_updated_at is not None

    def test_key_metrics_writes_forward_pe(self, db_session):
        """Branch: 287-288 — forward_pe non-None path. The round-1 suite has
        a None case but no positive case for forward_pe."""
        data_fetcher.save_ticker_to_db_cache("METRICS_FWD", "key_metrics", {
            "forward_pe": 18.5,
        })
        rec = db_session.query(StockDataCache).filter_by(
            ticker="METRICS_FWD").one()
        assert rec.forward_pe == pytest.approx(18.5)

    def test_institutional_with_string_data_is_ignored(self, db_session):
        """Branch: 277-279 — the isinstance(int|float) guard. A non-numeric
        payload must NOT crash and must NOT write institutional_holders_pct."""
        data_fetcher.save_ticker_to_db_cache(
            "INST_STR", "institutional", "not-a-number"
        )
        # The record may still be created (line 242-244) but institutional
        # fields must be untouched.
        rec = db_session.query(StockDataCache).filter_by(
            ticker="INST_STR").first()
        assert rec is not None
        assert rec.institutional_holders_pct is None
        assert rec.institutional_updated_at is None

    def test_returns_quietly_when_db_session_unavailable(self, monkeypatch):
        """Branch: 233-235 — `if not db: return`. SessionLocal returning None
        must NOT raise."""
        monkeypatch.setattr(data_fetcher, "_get_db_session", lambda: None)
        # Must complete without raising — there's nothing else to assert.
        data_fetcher.save_ticker_to_db_cache(
            "NODB", "earnings", {"quarterly_eps": [1.0]}
        )

    def test_earnings_calendar_accepts_pre_parsed_date_object(self, db_session):
        """Branch: 319-320 — `else` path when next_earnings_date is already
        a `date` object (not a string)."""
        data_fetcher.save_ticker_to_db_cache("PARSED", "earnings_calendar", {
            "next_earnings_date": date(2026, 8, 20),
            "days_to_earnings": 100,
            "earnings_beat_streak": 2,
        })
        rec = db_session.query(StockDataCache).filter_by(ticker="PARSED").one()
        assert rec.next_earnings_date == date(2026, 8, 20)
        assert rec.earnings_beat_streak == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — load_cache_from_db field-by-field freshness branches
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadCacheFromDbFieldPaths:
    """Tier 1: each of the 9 data types loaded by load_cache_from_db has its
    own freshness-update block (lines 132/144/155/163/176/201/212). The
    round-1 suite covers earnings + the full P1 record but not the
    standalone single-field records that hit each freshness block in
    isolation."""

    def test_balance_sheet_only_record_loads_with_freshness(self, db_session):
        """Branch: 144 — balance_sheet freshness lock update."""
        rec = StockDataCache(
            ticker="BAL",
            total_cash=100e9,
            total_debt=50e9,
            balance_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("BAL", "balance_sheet")
        assert cached is not None
        assert cached["total_cash"] == 100e9
        assert data_fetcher.is_data_fresh("BAL", "balance_sheet") is True

    def test_analyst_only_record_loads_with_freshness(self, db_session):
        """Branch: 155 — analyst freshness lock update."""
        rec = StockDataCache(
            ticker="ANL",
            analyst_target_price=180.0,
            analyst_count=20,
            analyst_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("ANL", "analyst")
        assert cached["target_price"] == 180.0
        assert cached["count"] == 20
        assert data_fetcher.is_data_fresh("ANL", "analyst") is True

    def test_institutional_only_record_loads_with_freshness(self, db_session):
        """Branch: 163 — institutional freshness lock update."""
        rec = StockDataCache(
            ticker="INST",
            institutional_holders_pct=68.5,
            institutional_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("INST", "institutional")
        assert cached == pytest.approx(68.5)
        assert data_fetcher.is_data_fresh("INST", "institutional") is True

    def test_key_metrics_only_record_loads_with_freshness(self, db_session):
        """Branch: 176 — key_metrics freshness lock update."""
        rec = StockDataCache(
            ticker="MET",
            roe=0.22,
            trailing_pe=24.0,
            forward_pe=20.0,
            peg_ratio=1.2,
            metrics_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("MET", "key_metrics")
        assert cached["roe"] == pytest.approx(0.22)
        assert data_fetcher.is_data_fresh("MET", "key_metrics") is True

    def test_analyst_estimates_only_record_loads_with_freshness(self, db_session):
        """Branch: 201 — analyst_estimates freshness lock update."""
        rec = StockDataCache(
            ticker="EST",
            eps_estimate_current=11.0,
            eps_estimate_prior=10.0,
            eps_estimate_revision_pct=10.0,
            analyst_estimates_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("EST", "analyst_estimates")
        assert cached["eps_estimate_current"] == pytest.approx(11.0)
        assert data_fetcher.is_data_fresh("EST", "analyst_estimates") is True

    def test_short_interest_only_record_loads_with_freshness(self, db_session):
        """Branch: 212 — short_interest freshness lock update."""
        rec = StockDataCache(
            ticker="SHRT",
            short_interest_pct=15.0,
            short_ratio=4.5,
            short_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("SHRT", "short_interest")
        assert cached["short_interest_pct"] == pytest.approx(15.0)
        assert data_fetcher.is_data_fresh("SHRT", "short_interest") is True

    def test_revenue_only_record_loads_with_freshness(self, db_session):
        """Branch: 132 — revenue freshness lock update."""
        rec = StockDataCache(
            ticker="REV",
            quarterly_revenue=[100e9, 95e9, 90e9, 85e9],
            annual_revenue=[380e9, 360e9],
            revenue_updated_at=datetime.now(),
        )
        db_session.add(rec)
        db_session.commit()
        data_fetcher.load_cache_from_db()
        cached = data_fetcher.get_cached_data("REV", "revenue")
        assert cached is not None
        assert cached["quarterly_revenue"] == [100e9, 95e9, 90e9, 85e9]
        assert data_fetcher.is_data_fresh("REV", "revenue") is True

    def test_db_unavailable_marks_loaded_without_raising(self, monkeypatch):
        """Branch: 99-102 — `if not db: ... return` early exit. Ensures the
        cold-start path tolerates a missing database connection."""
        monkeypatch.setattr(data_fetcher, "_get_db_session", lambda: None)
        # Must mark loaded so callers don't retry forever.
        data_fetcher.load_cache_from_db()
        assert data_fetcher._db_cache_loaded is True


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — _fmp_get retry path (429 → success)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFmpGetRetryThenSuccess:
    """Tier 1: 429 → backoff → 200 retry path at lines 562-571. Patches
    fmp_rate_limiter._config + time.sleep so we don't actually wait."""

    def test_retries_after_429_and_returns_200(self, monkeypatch):
        """Branch: 562-571 — backoff calculated, time.sleep called, retry."""
        import fmp_rate_limiter
        import time as _time

        monkeypatch.setattr(fmp_rate_limiter, "acquire_sync", lambda: 0.0)
        monkeypatch.setattr(fmp_rate_limiter, "record_429", lambda: None)
        monkeypatch.setattr(fmp_rate_limiter, "record_success", lambda *a, **k: None)
        monkeypatch.setattr(fmp_rate_limiter, "calculate_backoff",
                            lambda *_a, **_k: 0.0)
        # No real sleeping.
        monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
        # Allow at least 2 retries.
        monkeypatch.setitem(fmp_rate_limiter._config, "max_retries", 3)

        # First call returns 429, second returns 200.
        responses = iter([
            _mock_response(status_code=429),
            _mock_response(status_code=200, json_data=[{"x": 1}]),
        ])
        monkeypatch.setattr(data_fetcher.requests, "get",
                            lambda *a, **k: next(responses))

        resp = data_fetcher._fmp_get("https://example.com")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — calculate_index_m_score ma_50<=0 fallbacks
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateIndexMScoreFallbacks:
    """Tier 2: ma_50<=0 short-circuits the 50MA distance and the structure
    components to neutral 0.5 (lines 1499 and 1513)."""

    def test_zero_ma_50_uses_neutral_distance_and_structure(self):
        """Branches: 1499 (ma50_score=0.5) + 1513 (structure_score=0.5).
        With ma_50=0, only the 200MA term carries information.

        Math: pct_from_200 = (price - ma_200) / ma_200
              ma200_score = 0.5 + pct_from_200 / 0.20
              composite = 0.5*0.60 + ma200_score*0.30 + 0.5*0.10 = 0.35 + ma200_score*0.30
        """
        # price > 200MA by 10% → ma200_score = 0.5 + 0.5 = 1.0
        score = data_fetcher.calculate_index_m_score(
            price=110.0, ma_50=0.0, ma_200=100.0
        )
        # composite = 0.35 + 1.0*0.30 = 0.65
        assert score == pytest.approx(0.65)

    def test_zero_ma_200_returns_neutral_half(self):
        """Branch: 1489-1490 — ma_200<=0 short-circuits to 0.5."""
        assert data_fetcher.calculate_index_m_score(100.0, 99.0, 0.0) == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — fetch_market_direction_data trend buckets
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchMarketDirectionTrendBuckets:
    """Tier 2: the four trend boundaries at 1634-1643 (cautious / neutral /
    bearish / severe_bear). Round-1 covered bullish + severe_bear; this
    fills in the middle three."""

    def _make_chart_data(self, price, ma_50, ma_200, history_days=200):
        """Build chart_data with `history_days` close prices that average to
        ma_50 over the last 50 entries and ma_200 over the last 200."""
        # Easiest synthesis: build a constant array at the desired ma_200,
        # then bias the last 50 to land at ma_50, and the very last entry
        # at price. The exact distribution doesn't matter for the
        # calculate_index_m_score math — the helper only reads the mean.
        prices = [ma_200] * (history_days - 50)
        prices.extend([ma_50] * 49)
        prices.append(price)
        return {
            "current_price": price,
            "high_52w": price * 1.1,
            "low_52w": price * 0.9,
            "close_prices": prices,
            "volumes": [100_000_000] * len(prices),
        }

    def test_cautious_trend_when_composite_in_50_to_65_band(self, monkeypatch):
        """Branch: 1636-1637 — composite_m in [0.50, 0.65). Equal MAs and
        price slightly above produce composite ~ 0.53."""
        # price=101, ma_50=100, ma_200=100 → ma50_score≈0.6, ma200_score≈0.55,
        # structure_score=0.5 → composite ≈ 0.6*0.60 + 0.55*0.30 + 0.5*0.10 = 0.575
        chart = self._make_chart_data(price=101, ma_50=100, ma_200=100)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda *_a, **_k: chart)
        result = data_fetcher.fetch_market_direction_data()
        assert result["market_trend"] == "cautious"

    def test_neutral_trend_when_composite_in_35_to_50_band(self, monkeypatch):
        """Branch: 1638-1639 — composite_m in [0.35, 0.50)."""
        # price=98, ma_50=100, ma_200=100 → ma50_score≈0.3, ma200_score≈0.45,
        # structure_score=0.5 → composite ≈ 0.3*0.60+0.45*0.30+0.5*0.10 = 0.365
        chart = self._make_chart_data(price=98, ma_50=100, ma_200=100)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda *_a, **_k: chart)
        result = data_fetcher.fetch_market_direction_data()
        assert result["market_trend"] == "neutral"

    def test_bearish_trend_when_composite_in_20_to_35_band(self, monkeypatch):
        """Branch: 1640-1641 — composite_m in [0.20, 0.35)."""
        # price=96, ma_50=100, ma_200=100 → ma50_score=0.1, ma200_score=0.3,
        # structure_score=0.5 → composite = 0.1*0.6+0.3*0.3+0.5*0.1 = 0.2
        # (boundary 0.20 → bearish bucket)
        chart = self._make_chart_data(price=96, ma_50=100, ma_200=100)
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda *_a, **_k: chart)
        result = data_fetcher.fetch_market_direction_data()
        assert result["market_trend"] == "bearish"

    def test_no_index_data_returns_default_neutral(self, monkeypatch):
        """Branch: 1648-1650 — total_weight==0 fallback when chart fetch
        fails for every index."""
        # Always-empty chart_data → loop never adds to total_weight.
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                            lambda *_a, **_k: {})
        result = data_fetcher.fetch_market_direction_data()
        assert result["success"] is False
        assert result["error"] == "Could not fetch any index data"
        assert result["market_trend"] == "neutral"

    def test_per_index_exception_marked_error(self, monkeypatch):
        """Branch: 1614-1617 — per-index `try/except` traps fetch crashes
        and marks status='error', without aborting the rest of the loop."""
        def _crash(*_a, **_k):
            raise RuntimeError("boom")
        monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api", _crash)
        result = data_fetcher.fetch_market_direction_data()
        # All indexes crashed → no usable data.
        assert result["success"] is False
        for ticker_data in result["indexes"].values():
            assert ticker_data["status"] == "error"
            assert ticker_data["error"] == "boom"


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Delisted-ticker lifecycle DB-unavailable / exception edges
# ═══════════════════════════════════════════════════════════════════════════════


class TestDelistedLifecycleErrorPaths:
    """Tier 2: every helper in the delisted lifecycle has a `if not db: return`
    bail-out at the top. Hit each one to lock in the contract: failure to
    connect MUST NOT raise."""

    def test_refresh_delisted_cache_returns_quietly_without_db(self, monkeypatch):
        """Branch: 599-600 — refresh_delisted_cache no-DB exit."""
        monkeypatch.setattr(data_fetcher, "_get_db_session", lambda: None)
        data_fetcher.refresh_delisted_cache()  # Must not raise.

    def test_mark_ticker_as_delisted_returns_quietly_without_db(self, monkeypatch):
        """Branch: 619-621 — mark_ticker_as_delisted no-DB exit."""
        monkeypatch.setattr(data_fetcher, "_get_db_session", lambda: None)
        data_fetcher.mark_ticker_as_delisted("ZZZZ")  # Must not raise.

    def test_get_delisted_tickers_returns_empty_set_without_db(self, monkeypatch):
        """Branch: 693-695 — get_delisted_tickers no-DB exit returns set()."""
        monkeypatch.setattr(data_fetcher, "_get_db_session", lambda: None)
        assert data_fetcher.get_delisted_tickers() == set()

    def test_clear_delisted_ticker_returns_quietly_without_db(self, monkeypatch):
        """Branch: 730-732 — clear_delisted_ticker no-DB exit (after the
        in-memory cache short-circuit)."""
        # Mark the ticker as known-delisted so the in-memory short-circuit
        # at 727-728 doesn't fire — we want to reach the DB path.
        data_fetcher._delisted_cache_loaded = True
        data_fetcher._known_delisted_cache.add("CLR")
        monkeypatch.setattr(data_fetcher, "_get_db_session", lambda: None)
        data_fetcher.clear_delisted_ticker("CLR")  # Must not raise.

    def test_fmp_confirms_delisted_returns_true_on_request_exception(
        self, monkeypatch
    ):
        """Branch: 684-685 — `except Exception: return True` (err on
        side of caution when FMP itself is unreachable)."""
        # Need API key set so the early-return at 673-674 doesn't fire.
        monkeypatch.setenv("FMP_API_KEY", "test-key-1234")

        def _bang(*_a, **_k):
            raise ConnectionError("no route to host")
        monkeypatch.setattr(data_fetcher.requests, "get", _bang)

        assert data_fetcher._fmp_confirms_delisted("ZZZZ") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — fetch_with_cache Redis hot path
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchWithCacheRedisPath:
    """Tier 1: Redis cache layer — hot path at 450-461 lifts from Redis when
    memory misses, and writes the result back to memory cache. This is the
    layer that survives container restarts before the DB load completes."""

    def test_redis_hit_populates_memory_and_skips_fetch(self, monkeypatch):
        """Branch: 450-459 — Redis hit returns cached value, populates the
        in-memory cache, and the underlying fetch_func is NEVER called."""
        # Make the helper believe Redis is enabled.
        monkeypatch.setattr(data_fetcher, "REDIS_AVAILABLE", True)

        fake_redis = MagicMock()
        fake_redis.enabled = True
        fake_redis.get.return_value = {"hello": "world"}
        monkeypatch.setattr(data_fetcher, "redis_cache", fake_redis)

        # DB-cache load no-op.
        monkeypatch.setattr(data_fetcher, "load_cache_from_db", lambda: None)
        data_fetcher._db_cache_loaded = True

        called = []
        def fetch_func():
            called.append(1)
            return {"should": "not-be-used"}

        result = data_fetcher.fetch_with_cache("AAPL", "earnings", fetch_func)
        assert result == {"hello": "world"}
        assert called == []
        # And the result was written into the in-memory cache.
        assert data_fetcher.get_cached_data("AAPL", "earnings") == {"hello": "world"}

    def test_redis_get_exception_falls_through_to_fetch_func(self, monkeypatch):
        """Branch: 460-461 — `try/except` around redis_cache.get falls
        through to the API fetch on Redis errors."""
        monkeypatch.setattr(data_fetcher, "REDIS_AVAILABLE", True)

        fake_redis = MagicMock()
        fake_redis.enabled = True
        fake_redis.get.side_effect = RuntimeError("redis down")
        # set must not raise either.
        fake_redis.set = MagicMock()
        monkeypatch.setattr(data_fetcher, "redis_cache", fake_redis)

        monkeypatch.setattr(data_fetcher, "load_cache_from_db", lambda: None)
        data_fetcher._db_cache_loaded = True

        called = []
        def fetch_func():
            called.append(1)
            return {"fresh": "data"}

        result = data_fetcher.fetch_with_cache("AAPL", "earnings", fetch_func)
        assert result == {"fresh": "data"}
        assert called == [1]

    def test_redis_set_exception_does_not_break_fetch(self, monkeypatch):
        """Branch: 477-478 — when redis_cache.set raises (e.g. Redis OOM),
        the helper must still return the freshly-fetched data. The Redis
        write is best-effort."""
        monkeypatch.setattr(data_fetcher, "REDIS_AVAILABLE", True)

        fake_redis = MagicMock()
        fake_redis.enabled = True
        fake_redis.get.return_value = None  # Redis miss → fall through to fetch
        fake_redis.set.side_effect = RuntimeError("redis OOM")
        monkeypatch.setattr(data_fetcher, "redis_cache", fake_redis)

        monkeypatch.setattr(data_fetcher, "load_cache_from_db", lambda: None)
        data_fetcher._db_cache_loaded = True

        result = data_fetcher.fetch_with_cache(
            "AAPL", "earnings", lambda: {"fresh": "data"}
        )
        assert result == {"fresh": "data"}
        # The Redis set was attempted (and swallowed).
        fake_redis.set.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Earnings response branches (empty 200, non-200) and saver edge
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchFmpEarningsResponseBranches:
    """Tier 1: `fetch_fmp_earnings` has three response paths inside the
    quarterly try-block — 200+data, 200+empty, non-200. The 200+data path
    is covered by the round-1 suite; the other two land in logger.debug()
    bodies (lines 848 and 850) that are currently uncovered."""

    def test_quarterly_empty_200_response_logs_and_returns_empty_lists(
        self, fmp_key, monkeypatch
    ):
        """Branch: 847-848 — `if data: ... else: logger.debug("empty
        response")`. Empty payload must NOT crash and result must remain
        the seeded {quarterly_eps: [], annual_eps: []}."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(status_code=200, json_data=[]),
        )
        result = data_fetcher.fetch_fmp_earnings("AAPL")
        assert result == {"quarterly_eps": [], "annual_eps": []}

    def test_quarterly_non_200_response_logs_error_text(
        self, fmp_key, monkeypatch
    ):
        """Branch: 849-850 — non-200 path reads `resp.text[:200]` for the
        debug log. A regression that strips the text attribute would crash
        the wrapper instead of silently returning empty."""
        monkeypatch.setattr(
            data_fetcher, "_fmp_get",
            lambda *a, **k: _mock_response(
                status_code=500, text="upstream broke"
            ),
        )
        result = data_fetcher.fetch_fmp_earnings("AAPL")
        assert result == {"quarterly_eps": [], "annual_eps": []}


class TestSaveTickerInstitutionalNumeric:
    """Tier 1: the positive numeric path through save_ticker_to_db_cache for
    institutional data (lines 277-279). The round-1 suite has the string-
    rejection negative case but no positive scalar test."""

    def test_institutional_float_writes_field_and_timestamp(self, db_session):
        """Branch: 277-279 — `if data and isinstance(data, (int, float))`."""
        data_fetcher.save_ticker_to_db_cache("INST_NUM", "institutional", 67.5)
        rec = db_session.query(StockDataCache).filter_by(
            ticker="INST_NUM").one()
        assert rec.institutional_holders_pct == pytest.approx(67.5)
        assert rec.institutional_updated_at is not None

    def test_institutional_int_writes_field_and_timestamp(self, db_session):
        """Branch: 277-279 — int path (some Yahoo callers pass int)."""
        data_fetcher.save_ticker_to_db_cache("INST_INT", "institutional", 80)
        rec = db_session.query(StockDataCache).filter_by(
            ticker="INST_INT").one()
        assert rec.institutional_holders_pct == 80


class TestFmpGetRetryExhaustion:
    """Tier 1: `_fmp_get` after max_retries 429s has been hit must return
    the final 429 response (line 572 inside the elif). This is the sad
    path that makes the helper graceful rather than raising or looping
    forever."""

    def test_returns_final_429_after_retries_exhausted(self, monkeypatch):
        """Branch: 562-572 — every attempt 429s, final iteration falls out
        of the loop with a 429 in hand."""
        import fmp_rate_limiter
        import time as _time

        monkeypatch.setattr(fmp_rate_limiter, "acquire_sync", lambda: 0.0)
        monkeypatch.setattr(fmp_rate_limiter, "record_429", lambda: None)
        monkeypatch.setattr(fmp_rate_limiter, "calculate_backoff",
                            lambda *_a, **_k: 0.0)
        monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
        monkeypatch.setitem(fmp_rate_limiter._config, "max_retries", 3)

        # Always 429.
        monkeypatch.setattr(data_fetcher.requests, "get",
                            lambda *a, **k: _mock_response(status_code=429))

        resp = data_fetcher._fmp_get("https://example.com")
        assert resp.status_code == 429
        # 3 attempts × 1 request/attempt = 3 total requests tracked.
        assert data_fetcher.get_rate_limit_stats()["total_requests"] == 3
