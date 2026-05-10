"""Tests for backend/historical_data.py — closes the 28.12% baseline
(529/736 statements uncovered as of c6db1dd).

The historical data provider is the read-side companion to
backend/price_cache.py: every backtest builds one, preloads price
history for the universe + market indexes, and then queries it day-by-day
during the simulated walk forward. Silent failures here corrupt every
downstream signal (52-week highs, RS, breakouts, market direction, FTD
status), and would silently corrupt the 2026-06-18 Approach 2 verdict
read.

Eval-safe posture: every test injects synthetic DataFrames directly
into `provider._price_cache` / `provider._index_cache`, bypassing
yfinance entirely. Disk cache is neutered post-construction so tests
don't accidentally hit a real PriceHistoryCache singleton or write to
disk. No scoring touch, no live trading touch, no async.

Scope: Tier 1 read-only helpers — pure math, cache-key lookups, date
arithmetic, branch coverage on guard clauses. The yfinance fetcher,
ThreadPoolExecutor preload, and disk-cache integration paths are
deliberately deferred to Tier 2/3.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.historical_data import (  # noqa: E402
    EARNINGS_REPORT_DELAY_DAYS,
    HistoricalDataProvider,
    HistoricalStockData,
)


# ============================================================================
# Helpers
# ============================================================================
def _make_provider(tickers=None, ref_date=None):
    """Construct a provider with disk cache disabled.

    Disk cache is neutered after construction so tests don't pin a
    PriceHistoryCache singleton to ephemeral test state. We DO let
    `_init_disk_cache` run once during __init__ to exercise that code
    path under the standard import-resolves branch.
    """
    if ref_date is None:
        ref_date = date(2026, 5, 9)
    if tickers is None:
        tickers = ["AAPL"]
    provider = HistoricalDataProvider(tickers, data_reference_date=ref_date)
    provider._disk_cache = None
    provider._disk_cache_enabled = False
    return provider


def _make_price_df(start=date(2025, 1, 1), days=300, base=100.0,
                   slope=0.05, vol_base=1_000_000, vol_step=1000):
    """Build a synthetic OHLCV DataFrame in the schema the provider expects.

    Linear price slope keeps prices monotonically increasing so behavior
    is predictable: 52-week high is always the latest bar, MAs sit below
    current price, no division-by-zero edge cases.
    """
    rows = []
    d = start
    for i in range(days):
        close = base + slope * i
        rows.append({
            "date": d,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": vol_base + i * vol_step,
        })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def _make_pullback_df(start=date(2025, 1, 1), days=60, drop_index=30,
                       drop_pct=-3.0):
    """Build a SPY-shaped df with a single forced large daily drop on
    the bar at `drop_index`. Used to exercise pullback / distribution
    detection."""
    rows = []
    prev_close = 100.0
    d = start
    for i in range(days):
        if i == drop_index:
            close = prev_close * (1 + drop_pct / 100.0)
        else:
            close = prev_close * 1.001  # gentle uptrend baseline
        rows.append({
            "date": d,
            "open": prev_close,
            "high": max(close, prev_close) + 0.5,
            "low": min(close, prev_close) - 0.5,
            "close": close,
            "volume": 50_000_000 + (5_000_000 if i == drop_index else 0),
        })
        prev_close = close
        d += timedelta(days=1)
    return pd.DataFrame(rows)


# ============================================================================
# HistoricalStockData dataclass
# ============================================================================
class TestHistoricalStockData:
    def test_default_fields_initialized(self):
        d = HistoricalStockData(ticker="AAPL")
        assert d.ticker == "AAPL"
        assert d.price == 0.0
        assert d.high_52w == 0.0
        assert d.quarterly_earnings == []
        assert d.annual_earnings == []
        assert d.is_growth_stock is False
        assert d.eps_estimate_revision_pct is None
        # default_factory should produce independent lists per instance
        d2 = HistoricalStockData(ticker="MSFT")
        d.quarterly_earnings.append(1.0)
        assert d2.quarterly_earnings == []

    def test_field_assignment_works(self):
        d = HistoricalStockData(ticker="NVDA", price=150.0, sector="Technology")
        d.quarterly_earnings = [1.5, 1.4, 1.3, 1.2]
        d.eps_beat_streak = 4
        assert d.price == 150.0
        assert d.sector == "Technology"
        assert d.eps_beat_streak == 4


# ============================================================================
# _calculate_index_signal (discrete regime score)
# ============================================================================
class TestCalculateIndexSignal:
    def setup_method(self):
        self.provider = _make_provider()

    def test_bullish_when_above_both_mas(self):
        assert self.provider._calculate_index_signal(110, 100, 95) == 2

    def test_cautious_when_above_200_below_50(self):
        assert self.provider._calculate_index_signal(98, 100, 95) == 1

    def test_neutral_when_below_200_above_50(self):
        # price > 50MA but below 200MA → neutral (0)
        assert self.provider._calculate_index_signal(101, 100, 110) == 0

    def test_bearish_when_below_both_mas(self):
        assert self.provider._calculate_index_signal(80, 100, 95) == -1

    def test_returns_zero_when_ma200_invalid(self):
        assert self.provider._calculate_index_signal(100, 100, 0) == 0
        assert self.provider._calculate_index_signal(0, 100, 95) == 0


# ============================================================================
# _calculate_index_m_score (continuous M score)
# ============================================================================
class TestCalculateIndexMScore:
    def test_returns_half_when_ma200_invalid(self):
        assert HistoricalDataProvider._calculate_index_m_score(0, 100, 0) == 0.5

    def test_perfect_uptrend_above_caps(self):
        # Price >> 50MA, 50MA >> 200MA → all components saturate to 1.0
        score = HistoricalDataProvider._calculate_index_m_score(150, 120, 100)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_deep_downtrend_clamps_price_components(self):
        # Far below both MAs with ma_50 == ma_200: price-vs-MA components
        # clamp to 0, but structure_score defaults to 0.5 because
        # ma_spread = 0 → final = 0*0.6 + 0*0.3 + 0.5*0.1 = 0.05.
        # This pins the floor for "deep downtrend" — it is NOT zero.
        score = HistoricalDataProvider._calculate_index_m_score(50, 100, 100)
        assert score == pytest.approx(0.05, abs=1e-9)

    def test_at_neutral_when_price_equals_mas(self):
        # price == ma_50 == ma_200 → all components 0.5 → weighted 0.5
        assert HistoricalDataProvider._calculate_index_m_score(
            100, 100, 100
        ) == pytest.approx(0.5, abs=1e-9)

    def test_falls_back_when_ma50_zero(self):
        # ma_50 == 0 → ma50_score and structure_score default to 0.5
        score = HistoricalDataProvider._calculate_index_m_score(100, 0, 100)
        # ma50 contrib (0.5*0.6) + ma200 contrib (0.5*0.3) + structure (0.5*0.1) = 0.5
        assert score == pytest.approx(0.5, abs=1e-9)


# ============================================================================
# _filter_available_earnings — point-in-time earnings filter
# ============================================================================
class TestFilterAvailableEarnings:
    def setup_method(self):
        self.provider = _make_provider(ref_date=date(2026, 5, 9))

    def test_empty_list_returns_empty(self):
        assert self.provider._filter_available_earnings([], date(2025, 1, 1)) == []

    def test_returns_all_for_today_or_future(self):
        # days_diff <= 0 short-circuits to "return all"
        eps = [1.5, 1.4, 1.3, 1.2]
        out = self.provider._filter_available_earnings(eps, date(2026, 5, 9))
        assert out == eps
        out_future = self.provider._filter_available_earnings(eps, date(2026, 6, 1))
        assert out_future == eps

    def test_filters_recent_quarterly_periods(self):
        # As-of date is 365 days before reference → skip ~ (365+45)/91 = 4 quarters
        eps = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        out = self.provider._filter_available_earnings(
            eps, date(2025, 5, 9), quarterly=True
        )
        # Most-recent 4 quarters skipped, leaving entries from index 4 onwards
        assert out == [1.1, 1.0]

    def test_handles_dict_format_with_eps_key(self):
        eps = [{"eps": 2.0}, {"eps": 1.8}, {"eps": 1.6}]
        out = self.provider._filter_available_earnings(
            eps, date(2026, 5, 9), quarterly=True
        )
        assert out == [2.0, 1.8, 1.6]

    def test_skips_dicts_without_eps_key(self):
        # Entries without numeric value or "eps" key should be silently dropped
        mixed = [1.5, {"other": 99}, {"eps": 1.2}]
        out = self.provider._filter_available_earnings(
            mixed, date(2026, 5, 9), quarterly=True
        )
        assert out == [1.5, 1.2]

    def test_returns_empty_when_all_skipped(self):
        # 5 years in the past with annual data: skip 5 → all gone
        eps = [3.0, 2.5, 2.0]
        out = self.provider._filter_available_earnings(
            eps, date(2021, 5, 9), quarterly=False
        )
        assert out == []

    def test_filters_annual_by_year(self):
        # 366 days back → skip exactly 1 annual entry
        eps = [3.0, 2.5, 2.0, 1.5]
        out = self.provider._filter_available_earnings(
            eps, date(2025, 5, 8), quarterly=False
        )
        assert out == [2.5, 2.0, 1.5]


# ============================================================================
# Price / volume / history accessors
# ============================================================================
class TestPriceVolumeAccessors:
    def setup_method(self):
        self.provider = _make_provider()
        self.df = _make_price_df(start=date(2025, 1, 1), days=10, base=100, slope=1)
        self.provider._price_cache["AAPL"] = self.df

    def test_get_price_on_date_exact_match(self):
        assert self.provider.get_price_on_date("AAPL", date(2025, 1, 1)) == 100.0
        assert self.provider.get_price_on_date("AAPL", date(2025, 1, 5)) == 104.0

    def test_get_price_on_date_falls_back_to_prior(self):
        # 1/15 isn't in the df (only 1/1 - 1/10) — should return latest prior close
        result = self.provider.get_price_on_date("AAPL", date(2025, 1, 15))
        assert result == 109.0  # close on 2025-01-10

    def test_get_price_on_date_returns_none_for_unknown_ticker(self):
        assert self.provider.get_price_on_date("UNKNOWN", date(2025, 1, 1)) is None

    def test_get_price_on_date_returns_none_when_date_predates_history(self):
        # No exact match, no prior date — should return None
        assert self.provider.get_price_on_date("AAPL", date(2024, 1, 1)) is None

    def test_get_volume_on_date_exact_match(self):
        assert self.provider.get_volume_on_date("AAPL", date(2025, 1, 1)) == 1_000_000

    def test_get_volume_on_date_returns_none_for_unknown_ticker(self):
        assert self.provider.get_volume_on_date("UNKNOWN", date(2025, 1, 1)) is None

    def test_get_volume_on_date_returns_none_for_no_match(self):
        # Volume getter has no fall-back-to-prior logic: returns None when
        # the exact date isn't found.
        assert self.provider.get_volume_on_date("AAPL", date(2025, 2, 1)) is None

    def test_get_price_history_up_to_truncates_to_lookback(self):
        result = self.provider.get_price_history_up_to(
            "AAPL", date(2025, 1, 10), lookback_days=3
        )
        assert len(result) == 3
        assert list(result["close"]) == [107.0, 108.0, 109.0]

    def test_get_price_history_up_to_empty_for_unknown_ticker(self):
        result = self.provider.get_price_history_up_to(
            "UNKNOWN", date(2025, 1, 1), lookback_days=10
        )
        assert result.empty

    def test_get_price_history_returns_list(self):
        prices = self.provider.get_price_history(
            "AAPL", date(2025, 1, 10), lookback_days=5
        )
        assert prices == [105.0, 106.0, 107.0, 108.0, 109.0]

    def test_get_price_history_none_for_unknown_ticker(self):
        assert self.provider.get_price_history(
            "UNKNOWN", date(2025, 1, 10), lookback_days=5
        ) is None

    def test_get_price_history_none_when_too_few_points(self):
        # Only 1 day of history available before lookback applies
        small_df = _make_price_df(days=1)
        self.provider._price_cache["TINY"] = small_df
        assert self.provider.get_price_history(
            "TINY", small_df.iloc[0]["date"], lookback_days=5
        ) is None

    def test_get_price_series_aliased_to_get_price_history(self):
        # Correlation-guard compatibility alias
        assert self.provider.get_price_series == self.provider.get_price_history


# ============================================================================
# 52-week / 50-day / volume-ratio aggregations
# ============================================================================
class TestRollingAggregations:
    def setup_method(self):
        self.provider = _make_provider()
        self.df = _make_price_df(start=date(2025, 1, 1), days=300, base=50,
                                  slope=0.5)
        self.provider._price_cache["AAPL"] = self.df

    def test_52w_high_low_from_history(self):
        as_of = date(2025, 10, 1)  # well into the 300-day df
        high, low = self.provider.get_52_week_high_low("AAPL", as_of)
        # high is highest "high" column over 252-day lookback,
        # low is lowest "low" column. Slope=0.5, base=50, days=300:
        # expect high > low > 0
        assert high > low > 0
        # Cache hit returns same value without recomputing
        assert self.provider.get_52_week_high_low("AAPL", as_of) == (high, low)

    def test_52w_high_low_zero_for_unknown_ticker(self):
        assert self.provider.get_52_week_high_low(
            "UNKNOWN", date(2025, 1, 1)
        ) == (0.0, 0.0)

    def test_50_day_avg_volume_happy(self):
        as_of = date(2025, 6, 1)
        result = self.provider.get_50_day_avg_volume("AAPL", as_of)
        assert result > 0

    def test_50_day_avg_volume_zero_for_insufficient(self):
        # First few days of df: < 20 days available
        as_of = date(2025, 1, 5)
        assert self.provider.get_50_day_avg_volume("AAPL", as_of) == 0.0

    def test_moving_average_happy(self):
        as_of = date(2025, 6, 1)
        result = self.provider.get_moving_average("AAPL", as_of, period=20)
        assert result > 0

    def test_moving_average_zero_when_insufficient(self):
        # Period larger than available history
        as_of = date(2025, 1, 5)
        assert self.provider.get_moving_average(
            "AAPL", as_of, period=200
        ) == 0.0

    def test_volume_ratio_normal(self):
        as_of = date(2025, 6, 1)
        ratio = self.provider.get_volume_ratio("AAPL", as_of)
        assert ratio > 0

    def test_volume_ratio_one_when_avg_zero(self):
        # If 50-day-avg is zero (insufficient history), ratio defaults to 1.0
        as_of = date(2025, 1, 1)  # only 1 day of data
        assert self.provider.get_volume_ratio("AAPL", as_of) == 1.0


# ============================================================================
# Accumulation / distribution helper
# ============================================================================
class TestAccumulationDistribution:
    def setup_method(self):
        self.provider = _make_provider()

    def test_returns_none_for_unknown_ticker(self):
        assert self.provider.get_accumulation_distribution(
            "UNKNOWN", date(2025, 1, 1)
        ) is None

    def test_returns_none_for_insufficient_history(self):
        # Need >= 10 rows
        df = _make_price_df(days=5)
        self.provider._price_cache["TINY"] = df
        assert self.provider.get_accumulation_distribution(
            "TINY", df.iloc[-1]["date"]
        ) is None

    def test_happy_path_returns_metrics(self):
        df = _make_price_df(days=25, base=100, slope=1.0,
                            vol_base=1_000_000, vol_step=10_000)
        self.provider._price_cache["AAPL"] = df
        result = self.provider.get_accumulation_distribution(
            "AAPL", df.iloc[-1]["date"], lookback_days=20
        )
        assert result is not None
        assert "ad_ratio" in result
        assert "volume_dry_up" in result
        assert "volume_dry_up_score" in result
        assert "institutional_accumulation" in result
        # All-up days → ad_ratio defaults to 2.0 (no down volume)
        assert result["ad_ratio"] == 2.0

    def test_neutral_ratio_when_no_movement(self):
        # Flat prices: no up or down volume → ad_ratio defaults to 1.0
        rows = [{"date": date(2025, 1, 1) + timedelta(days=i),
                 "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                 "volume": 1_000_000}
                for i in range(15)]
        df = pd.DataFrame(rows)
        self.provider._price_cache["FLAT"] = df
        result = self.provider.get_accumulation_distribution(
            "FLAT", df.iloc[-1]["date"]
        )
        assert result["ad_ratio"] == 1.0


# ============================================================================
# SPY helpers (price + daily data + pullback distance)
# ============================================================================
class TestSpyHelpers:
    def setup_method(self):
        self.provider = _make_provider()
        self.spy_df = _make_price_df(start=date(2025, 1, 1), days=260, base=400,
                                      slope=0.2)
        self.provider._index_cache["SPY"] = self.spy_df

    def test_get_spy_price_exact_match(self):
        assert self.provider.get_spy_price_on_date(date(2025, 1, 1)) == 400.0

    def test_get_spy_price_falls_back_to_prior(self):
        # Date past end of df → returns most-recent close
        far_future = date(2030, 1, 1)
        result = self.provider.get_spy_price_on_date(far_future)
        assert result == self.spy_df.iloc[-1]["close"]

    def test_get_spy_price_zero_for_pre_history_date(self):
        # No exact match, no prior date in df
        assert self.provider.get_spy_price_on_date(date(2024, 1, 1)) == 0.0

    def test_get_spy_price_zero_when_no_spy_cached(self):
        provider = _make_provider()
        assert provider.get_spy_price_on_date(date(2025, 1, 1)) == 0.0

    def test_get_spy_daily_data_happy(self):
        as_of = date(2025, 6, 1)
        data = self.provider.get_spy_daily_data(as_of)
        assert data["close"] > 0
        assert data["prev_close"] > 0
        assert data["ma50"] > 0
        assert data["ema21"] > 0

    def test_get_spy_daily_data_default_when_no_spy(self):
        provider = _make_provider()
        data = provider.get_spy_daily_data(date(2025, 1, 1))
        assert data == {"close": 0, "prev_close": 0, "volume": 0,
                        "prev_volume": 0, "ma50": 0, "ema21": 0, "ma200": 0}

    def test_get_spy_daily_data_default_when_too_few_days(self):
        provider = _make_provider()
        provider._index_cache["SPY"] = _make_price_df(days=1)
        data = provider.get_spy_daily_data(date(2025, 1, 1))
        assert data["close"] == 0
        assert data["ma50"] == 0

    def test_get_days_since_spy_pullback_finds_drop(self):
        # Replace SPY df with one containing a deliberate -3% drop
        provider = _make_provider()
        provider._index_cache["SPY"] = _make_pullback_df(
            start=date(2025, 1, 1), days=60, drop_index=30, drop_pct=-3.0
        )
        # As-of-date is well after the drop
        as_of = date(2025, 1, 1) + timedelta(days=59)
        days_since = provider.get_days_since_spy_pullback(as_of, threshold_pct=-2.0)
        # 60 calendar days from drop_index=30 to day 59 → 29 days
        assert 0 < days_since <= 60

    def test_get_days_since_spy_pullback_returns_60_when_no_drop(self):
        # Pure uptrend df → no drop in window → returns 60 (window cap)
        as_of = date(2025, 6, 1)
        result = self.provider.get_days_since_spy_pullback(as_of)
        assert result == 60

    def test_get_days_since_spy_pullback_default_when_no_spy(self):
        provider = _make_provider()
        # No SPY cached → returns default 30
        assert provider.get_days_since_spy_pullback(date(2025, 1, 1)) == 30


# ============================================================================
# Market direction (per-index + composite + cache + precompute)
# ============================================================================
class TestMarketDirection:
    def setup_method(self):
        self.provider = _make_provider()
        for idx in ["SPY", "QQQ", "DIA"]:
            self.provider._index_cache[idx] = _make_price_df(
                start=date(2025, 1, 1), days=260, base=400, slope=0.5
            )

    def test_compute_market_direction_with_three_indexes(self):
        as_of = date(2025, 8, 1)
        result = self.provider._compute_market_direction(as_of)
        for key in ["spy", "qqq", "dia"]:
            assert key in result
            assert "price" in result[key]
            assert "ma_50" in result[key]
            assert "ma_200" in result[key]
            assert "signal" in result[key]
            assert "m_score" in result[key]
        assert "weighted_signal" in result
        assert "composite_m" in result
        assert "is_bullish" in result
        # Steady uptrend → bullish
        assert result["is_bullish"] is True

    def test_compute_market_direction_handles_missing_index(self):
        provider = _make_provider()
        # Only SPY cached — QQQ and DIA missing entirely
        provider._index_cache["SPY"] = _make_price_df(days=260)
        result = provider._compute_market_direction(date(2025, 6, 1))
        assert result["qqq"]["signal"] == 0
        assert result["dia"]["signal"] == 0
        # SPY should still compute
        assert result["spy"]["signal"] != 0 or result["spy"]["price"] > 0

    def test_compute_market_direction_handles_short_history(self):
        provider = _make_provider()
        provider._index_cache["SPY"] = _make_price_df(days=10)
        provider._index_cache["QQQ"] = _make_price_df(days=10)
        provider._index_cache["DIA"] = _make_price_df(days=10)
        result = provider._compute_market_direction(date(2025, 1, 5))
        # Less than 50 days → defaults
        assert result["spy"]["signal"] == 0
        assert result["spy"]["price"] == 0

    def test_get_market_direction_uses_cache_when_present(self):
        as_of = date(2025, 6, 1)
        sentinel = {"weighted_signal": 99, "composite_m": 0.42, "is_bullish": True}
        self.provider._market_direction_cache[as_of] = sentinel
        assert self.provider.get_market_direction(as_of) is sentinel

    def test_get_market_direction_falls_through_when_uncached(self):
        as_of = date(2025, 6, 1)
        result = self.provider.get_market_direction(as_of)
        assert "weighted_signal" in result

    def test_precompute_market_direction_populates_cache(self):
        self.provider._trading_days = [
            date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)
        ]
        self.provider.precompute_market_direction()
        assert len(self.provider._market_direction_cache) == 3
        for d in self.provider._trading_days:
            assert d in self.provider._market_direction_cache


# ============================================================================
# Relative strength (vs SPY)
# ============================================================================
class TestRelativeStrength:
    def setup_method(self):
        self.provider = _make_provider()
        # SPY: gentle uptrend
        self.provider._index_cache["SPY"] = _make_price_df(
            start=date(2024, 1, 1), days=300, base=400, slope=0.1
        )

    def test_default_when_stock_missing(self):
        # No stock data cached → defaults to 1.0
        result = self.provider.get_relative_strength(
            "MISSING", date(2025, 1, 1), period_months=12
        )
        assert result == 1.0

    def test_outperformer_returns_above_one(self):
        # Stock with steeper slope than SPY
        self.provider._price_cache["WINNER"] = _make_price_df(
            start=date(2024, 1, 1), days=300, base=100, slope=2.0
        )
        as_of = date(2024, 12, 1)
        rs = self.provider.get_relative_strength("WINNER", as_of, period_months=6)
        # Steep stock vs gentle SPY → RS > 1.0
        assert rs > 1.0
        # And cached so a second call returns the same value
        rs2 = self.provider.get_relative_strength("WINNER", as_of, period_months=6)
        assert rs2 == rs

    def test_capped_at_three_during_severe_spy_crash(self):
        # When SPY crashes, the floor on spy_denom (max(1+spy_return, 0.5))
        # caps RS inflation. We force a crash by giving SPY a steep
        # downtrend AND the stock a steep uptrend; the cap kicks in once
        # (1+stock_return)/0.5 exceeds 3.0.
        provider = _make_provider()
        # SPY: drops from 400 to ~50 over the lookback window
        crash_rows = []
        for i in range(300):
            close = 400.0 - i * 1.2
            crash_rows.append({"date": date(2024, 1, 1) + timedelta(days=i),
                                "open": close, "high": close + 1, "low": close - 1,
                                "close": close, "volume": 50_000_000})
        provider._index_cache["SPY"] = pd.DataFrame(crash_rows)
        provider._price_cache["MOON"] = _make_price_df(
            start=date(2024, 1, 1), days=300, base=10, slope=10.0
        )
        rs = provider.get_relative_strength(
            "MOON", date(2024, 12, 1), period_months=6
        )
        assert rs == 3.0

    def test_default_when_too_short_history(self):
        # < 20 stock bars → default to 1.0
        self.provider._price_cache["SHORT"] = _make_price_df(days=5)
        rs = self.provider.get_relative_strength(
            "SHORT", date(2024, 1, 5), period_months=12
        )
        assert rs == 1.0


# ============================================================================
# Ticker-set helpers
# ============================================================================
class TestTickerSetHelpers:
    def setup_method(self):
        self.provider = _make_provider(tickers=["AAPL", "MSFT", "NVDA"])
        for t in ["AAPL", "MSFT", "NVDA"]:
            self.provider._price_cache[t] = _make_price_df(days=10)

    def test_has_data_for_ticker_present(self):
        assert self.provider.has_data_for_ticker("AAPL") is True

    def test_has_data_for_ticker_absent(self):
        assert self.provider.has_data_for_ticker("UNKNOWN") is False

    def test_get_available_tickers_lists_cached(self):
        assert set(self.provider.get_available_tickers()) == {"AAPL", "MSFT", "NVDA"}

    def test_filter_tickers_drops_excluded(self):
        self.provider.filter_tickers(["AAPL", "MSFT"])
        assert set(self.provider.get_available_tickers()) == {"AAPL", "MSFT"}
        assert "NVDA" not in self.provider._price_cache

    def test_get_trading_days_returns_copy(self):
        self.provider._trading_days = [date(2025, 1, 1), date(2025, 1, 2)]
        copy = self.provider.get_trading_days()
        copy.append(date(2099, 1, 1))
        # Original list untouched
        assert self.provider._trading_days == [date(2025, 1, 1), date(2025, 1, 2)]


# ============================================================================
# Weekly aggregation + ATR
# ============================================================================
class TestWeeklyAndAtr:
    def setup_method(self):
        self.provider = _make_provider()
        self.provider._price_cache["AAPL"] = _make_price_df(
            start=date(2025, 1, 1), days=200, base=100, slope=0.5
        )

    def test_weekly_price_history_aggregates(self):
        as_of = date(2025, 5, 1)
        weeks = self.provider.get_weekly_price_history("AAPL", as_of, weeks=10)
        assert len(weeks) <= 10
        for w in weeks:
            assert "high" in w
            assert "low" in w
            assert "close" in w
            assert "volume" in w
            # Volume is summed across the week → larger than any single bar
            assert w["volume"] > 0

    def test_weekly_price_history_empty_for_unknown(self):
        result = self.provider.get_weekly_price_history(
            "UNKNOWN", date(2025, 1, 1), weeks=10
        )
        assert result == []

    def test_atr_happy_path(self):
        as_of = date(2025, 5, 1)
        atr = self.provider.get_atr("AAPL", as_of, period=14)
        # Synthetic data has fixed 1-pt high-low range; ATR > 0
        assert atr > 0

    def test_atr_zero_for_insufficient_history(self):
        provider = _make_provider()
        provider._price_cache["TINY"] = _make_price_df(days=5)
        assert provider.get_atr(
            "TINY", date(2025, 1, 5), period=14
        ) == 0.0

    def test_atr_zero_when_unknown_ticker(self):
        assert self.provider.get_atr(
            "UNKNOWN", date(2025, 1, 1)
        ) == 0.0


# ============================================================================
# Base patterns: flat base, cup, dispatcher + caching
# ============================================================================
class TestBasePatternDetection:
    def setup_method(self):
        self.provider = _make_provider()

    def _flat_weekly(self, n=10, base=100, jitter=2):
        # Tight range — should qualify as flat base
        return [{"high": base + jitter, "low": base - jitter,
                 "close": base, "volume": 1_000_000} for _ in range(n)]

    def _cup_weekly(self, n=20):
        # First third high, middle low, recovery to within 15% of high
        weekly = []
        for i in range(n):
            if i < n // 3:
                base = 100
            elif i < 2 * n // 3:
                base = 75  # ~25% drop = valid cup depth
            else:
                base = 95  # recovery within 15%
            weekly.append({"high": base + 1, "low": base - 1,
                           "close": base, "volume": 1_000_000})
        return weekly

    def test_detect_flat_base_finds_tight_range(self):
        result = self.provider._detect_flat_base(self._flat_weekly(n=10))
        assert result["type"] == "flat"
        assert result["weeks"] >= 5

    def test_detect_flat_base_returns_none_for_too_few_weeks(self):
        result = self.provider._detect_flat_base(self._flat_weekly(n=3))
        assert result["type"] == "none"

    def test_detect_flat_base_returns_none_for_wide_range(self):
        # Big range — should NOT qualify as flat base
        weekly = [{"high": 100 + i * 5, "low": 100 - i * 5,
                   "close": 100, "volume": 1_000_000} for i in range(10)]
        result = self.provider._detect_flat_base(weekly)
        assert result["type"] == "none"

    def test_detect_cup_pattern_finds_recovery(self):
        result = self.provider._detect_cup_pattern(self._cup_weekly())
        assert result["type"] == "cup"
        assert result["pivot_price"] > 0

    def test_detect_cup_pattern_returns_none_for_shallow(self):
        # All prices flat — not enough depth for a cup
        weekly = [{"high": 101, "low": 99, "close": 100, "volume": 1_000_000}
                  for _ in range(15)]
        result = self.provider._detect_cup_pattern(weekly)
        assert result["type"] == "none"

    def test_detect_cup_pattern_returns_none_for_too_few_weeks(self):
        result = self.provider._detect_cup_pattern(self._cup_weekly(n=5))
        assert result["type"] == "none"

    def test_detect_base_pattern_caches_within_a_week(self):
        # Flat-base setup
        self.provider._price_cache["AAPL"] = _make_price_df(
            start=date(2025, 1, 1), days=120, base=100, slope=0.0
        )
        as_of = date(2025, 4, 1)
        first = self.provider.detect_base_pattern("AAPL", as_of)
        # Mutate cache entry to verify second call returns cached pattern
        self.provider._base_pattern_cache["AAPL"] = (
            as_of, {"type": "sentinel", "weeks": 99, "pivot_price": 99.0}
        )
        again = self.provider.detect_base_pattern(
            "AAPL", as_of + timedelta(days=3)
        )
        assert again["type"] == "sentinel"

    def test_detect_base_pattern_returns_none_for_too_few_weekly(self):
        # Only a few days of price history → < 5 weekly bars
        self.provider._price_cache["TINY"] = _make_price_df(days=10)
        result = self.provider.detect_base_pattern(
            "TINY", date(2025, 1, 10)
        )
        assert result["type"] == "none"


# ============================================================================
# Accumulation/distribution day count + FTD state machine
# ============================================================================
class TestAccumulationDistributionDays:
    def setup_method(self):
        self.provider = _make_provider()

    def test_returns_zeros_when_no_spy(self):
        result = self.provider.get_accumulation_distribution_days(
            date(2025, 1, 1)
        )
        assert result["distribution_days"] == 0
        assert result["accumulation_days"] == 0
        assert result["is_under_pressure"] is False
        assert result["is_critical"] is False

    def test_counts_distribution_days(self):
        # SPY df with several -1% drops on rising volume → distribution days
        rows = []
        prev = 400.0
        prev_vol = 50_000_000
        for i in range(30):
            if i % 3 == 0:
                close = prev * 0.99  # -1% drop
                vol = prev_vol * 1.5  # higher volume
            else:
                close = prev * 1.001
                vol = prev_vol * 0.9
            rows.append({"date": date(2025, 1, 1) + timedelta(days=i),
                         "open": prev, "high": max(prev, close),
                         "low": min(prev, close),
                         "close": close, "volume": vol})
            prev = close
            prev_vol = vol
        self.provider._index_cache["SPY"] = pd.DataFrame(rows)
        result = self.provider.get_accumulation_distribution_days(
            date(2025, 1, 30), lookback_days=25
        )
        assert result["distribution_days"] >= 4
        # 4+ distribution days → under_pressure flag
        assert result["is_under_pressure"] is True


class TestFollowThroughDayStatus:
    def setup_method(self):
        self.provider = _make_provider()

    def test_default_when_no_spy(self):
        result = self.provider.get_follow_through_day_status(date(2025, 1, 1))
        assert result["state"] == "CONFIRMED_UPTREND"
        assert result["can_buy"] is True

    def test_default_when_too_few_bars(self):
        self.provider._index_cache["SPY"] = _make_price_df(days=5)
        result = self.provider.get_follow_through_day_status(
            date(2025, 1, 5)
        )
        assert result["state"] == "CONFIRMED_UPTREND"
        assert result["can_buy"] is True

    def test_returns_dict_with_uptrend_for_steady_climb(self):
        # Steady uptrend: above 50MA, no distribution days → CONFIRMED_UPTREND
        self.provider._index_cache["SPY"] = _make_price_df(
            start=date(2025, 1, 1), days=120, base=400, slope=0.5,
            vol_base=50_000_000
        )
        result = self.provider.get_follow_through_day_status(date(2025, 5, 1))
        assert result["state"] in (
            "CONFIRMED_UPTREND", "UPTREND_UNDER_PRESSURE"
        )


# ============================================================================
# RS-line new high
# ============================================================================
class TestRsLineNewHigh:
    def setup_method(self):
        self.provider = _make_provider()

    def test_returns_no_high_when_no_stock_data(self):
        self.provider._index_cache["SPY"] = _make_price_df(days=300)
        result = self.provider.get_rs_line_new_high(
            "MISSING", date(2025, 6, 1)
        )
        assert result == {"rs_at_new_high": False, "price_at_new_high": False,
                          "rs_leading": False, "rs_lagging": False}

    def test_returns_no_high_when_no_spy(self):
        self.provider._price_cache["AAPL"] = _make_price_df(days=300)
        result = self.provider.get_rs_line_new_high("AAPL", date(2025, 6, 1))
        assert result["rs_at_new_high"] is False

    def test_returns_no_high_when_history_too_short(self):
        # < 60 stock bars
        self.provider._price_cache["SHORT"] = _make_price_df(days=30)
        self.provider._index_cache["SPY"] = _make_price_df(days=300)
        result = self.provider.get_rs_line_new_high("SHORT", date(2025, 1, 30))
        assert result["rs_at_new_high"] is False

    def test_detects_rs_at_new_high_for_outperformer(self):
        # Stock outperforms SPY → both RS line and price hit new highs
        self.provider._price_cache["WINNER"] = _make_price_df(
            start=date(2024, 1, 1), days=300, base=100, slope=2.0
        )
        self.provider._index_cache["SPY"] = _make_price_df(
            start=date(2024, 1, 1), days=300, base=400, slope=0.1
        )
        result = self.provider.get_rs_line_new_high(
            "WINNER", date(2024, 10, 1)
        )
        assert result["rs_at_new_high"] is True
        assert result["price_at_new_high"] is True


# ============================================================================
# VIX proxy + stock correlation
# ============================================================================
class TestVixAndCorrelation:
    def setup_method(self):
        self.provider = _make_provider()

    def test_vix_proxy_default_when_no_spy(self):
        assert self.provider.get_vix_proxy(date(2025, 1, 1)) == 18.0

    def test_vix_proxy_default_when_too_few_bars(self):
        self.provider._index_cache["SPY"] = _make_price_df(days=5)
        assert self.provider.get_vix_proxy(date(2025, 1, 5)) == 18.0

    def test_vix_proxy_calculates_volatility(self):
        # SPY df with deliberate alternating moves so daily-return std-dev
        # is non-zero. A pure linear slope produces near-constant returns
        # and the annualized vol collapses to ~0.
        rows = []
        prev = 400.0
        for i in range(60):
            close = prev * (1.01 if i % 2 == 0 else 0.99)
            rows.append({"date": date(2025, 1, 1) + timedelta(days=i),
                         "open": prev, "high": max(prev, close),
                         "low": min(prev, close),
                         "close": close, "volume": 50_000_000})
            prev = close
        self.provider._index_cache["SPY"] = pd.DataFrame(rows)
        result = self.provider.get_vix_proxy(date(2025, 2, 28),
                                              lookback_days=20)
        assert result > 0

    def test_correlation_zero_for_unknown_tickers(self):
        assert self.provider.get_stock_correlation(
            "A", "B", date(2025, 1, 1)
        ) == 0.0

    def test_correlation_zero_for_insufficient_data(self):
        self.provider._price_cache["A"] = _make_price_df(days=5)
        self.provider._price_cache["B"] = _make_price_df(days=5)
        assert self.provider.get_stock_correlation(
            "A", "B", date(2025, 1, 5)
        ) == 0.0

    def test_correlation_high_for_lockstep_returns(self):
        # Two identical price series → correlation ~1.0
        df = _make_price_df(days=60, base=100, slope=0.5)
        self.provider._price_cache["A"] = df.copy()
        self.provider._price_cache["B"] = df.copy()
        result = self.provider.get_stock_correlation(
            "A", "B", df.iloc[-1]["date"], lookback_days=30
        )
        assert result == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# get_stock_data_on_date orchestration (happy path + minimal)
# ============================================================================
class TestGetStockDataOnDate:
    def setup_method(self):
        self.provider = _make_provider(ref_date=date(2026, 5, 9))
        self.provider._price_cache["AAPL"] = _make_price_df(
            start=date(2025, 1, 1), days=300, base=100, slope=0.5
        )

    def test_minimal_when_no_static_data(self):
        as_of = date(2025, 6, 1)
        data = self.provider.get_stock_data_on_date("AAPL", as_of)
        assert data.ticker == "AAPL"
        assert data.price > 0
        assert data.volume > 0
        # No static_data → defaults
        assert data.sector == ""
        assert data.quarterly_earnings == []

    def test_populates_from_static_data(self):
        as_of = date(2026, 4, 1)  # close to reference → minimal earnings filter
        static = {
            "sector": "Technology",
            "name": "Apple",
            "institutional_holders_pct": 65.0,
            "roe": 35.0,
            "analyst_target_price": 180.0,
            "num_analyst_opinions": 25,
            "earnings_surprise_pct": 12.5,
            "earnings_beat_streak": 4,
            "eps_estimate_revision_pct": 8.0,
            "quarterly_earnings": [1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
            "annual_earnings": [5.0, 4.5, 4.0],
            "quarterly_revenue": [100, 95, 90, 85, 80, 75],
        }
        data = self.provider.get_stock_data_on_date("AAPL", as_of, static)
        assert data.sector == "Technology"
        assert data.name == "Apple"
        assert data.roe == 35.0
        assert data.num_analyst_opinions == 25
        assert data.earnings_surprise_pct == 12.5
        # Both eps_beat_streak and earnings_beat_streak mirror the streak
        assert data.eps_beat_streak == 4
        assert data.earnings_beat_streak == 4
        assert data.eps_estimate_revision_pct == 8.0
        # Earnings filter ran, returning some subset
        assert isinstance(data.quarterly_earnings, list)
        assert isinstance(data.annual_earnings, list)
        assert isinstance(data.quarterly_revenue, list)

    def test_handles_none_surprise_pct(self):
        # earnings_surprise_pct=None should coerce to 0.0
        as_of = date(2026, 4, 1)
        static = {
            "sector": "Energy",
            "earnings_surprise_pct": None,
            "earnings_beat_streak": None,
            "quarterly_earnings": [],
            "annual_earnings": [],
            "quarterly_revenue": [],
        }
        data = self.provider.get_stock_data_on_date("AAPL", as_of, static)
        assert data.earnings_surprise_pct == 0.0
        assert data.eps_beat_streak == 0
        assert data.earnings_beat_streak == 0


# ============================================================================
# is_breaking_out (composes price/volume/base detection)
# ============================================================================
class TestIsBreakingOut:
    def setup_method(self):
        self.provider = _make_provider()

    def test_no_breakout_for_unknown_ticker(self):
        is_bo, vol_ratio, pattern = self.provider.is_breaking_out(
            "UNKNOWN", date(2025, 1, 1)
        )
        assert is_bo is False
        assert pattern["type"] == "none"

    def test_no_breakout_when_no_price(self):
        # Empty df cached for the ticker
        self.provider._price_cache["EMPTY"] = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )
        is_bo, _, _ = self.provider.is_breaking_out("EMPTY", date(2025, 1, 1))
        assert is_bo is False


# ============================================================================
# get_cache_stats branches
# ============================================================================
class TestCacheStats:
    def test_returns_disabled_when_no_disk_cache(self):
        provider = _make_provider()
        # _disk_cache forced to None in helper
        assert provider.get_cache_stats() == {"enabled": False}

    def test_delegates_to_disk_cache_when_present(self):
        provider = _make_provider()
        sentinel = {"enabled": True, "size": 42}

        class _FakeCache:
            def get_stats(self_inner):
                return sentinel

        provider._disk_cache = _FakeCache()
        assert provider.get_cache_stats() is sentinel
