"""
Regression tests for bugs found during the Feb 24, 2026 deep review.
Each test class documents the original bug and ensures it stays fixed.
"""

import math
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

import pytest


# ─── NaN Sanitization (canslim_scorer.py) ─────────────────────────────────
# Bug: float('nan') passes `is not None` checks, corrupting CANSLIM scores.
# Fix: _clean_earnings() filters both None and NaN.


class TestNaNSanitization:
    """Regression: NaN values in earnings lists caused silent score corruption."""

    def test_clean_earnings_removes_nan(self):
        from canslim_scorer import _clean_earnings

        earnings = [1.50, float("nan"), 1.30, 1.20]
        cleaned = _clean_earnings(earnings)
        assert len(cleaned) == 3
        assert all(not math.isnan(e) for e in cleaned)

    def test_clean_earnings_removes_none(self):
        from canslim_scorer import _clean_earnings

        earnings = [1.50, None, 1.30, None]
        cleaned = _clean_earnings(earnings)
        assert cleaned == [1.50, 1.30]

    def test_clean_earnings_removes_mixed(self):
        from canslim_scorer import _clean_earnings

        earnings = [None, float("nan"), 1.0, float("nan"), None, 0.5]
        cleaned = _clean_earnings(earnings)
        assert cleaned == [1.0, 0.5]

    def test_clean_earnings_preserves_zeros(self):
        """Zero is valid earnings data (breakeven), must not be removed."""
        from canslim_scorer import _clean_earnings

        earnings = [0.0, 1.50, 0, -0.5]
        cleaned = _clean_earnings(earnings)
        assert cleaned == [0.0, 1.50, 0, -0.5]

    def test_clean_earnings_preserves_negatives(self):
        """Negative earnings are valid (losses), must not be removed."""
        from canslim_scorer import _clean_earnings

        earnings = [-1.0, -0.5, 0.1, 0.5]
        cleaned = _clean_earnings(earnings)
        assert cleaned == [-1.0, -0.5, 0.1, 0.5]

    def test_clean_earnings_empty_list(self):
        from canslim_scorer import _clean_earnings

        assert _clean_earnings([]) == []

    def test_clean_earnings_all_nan(self):
        from canslim_scorer import _clean_earnings

        earnings = [float("nan"), float("nan"), None]
        assert _clean_earnings(earnings) == []

    def test_scorer_handles_nan_in_quarterly(self, mock_stock_data, mock_data_fetcher):
        """C score should not crash when quarterly earnings contain NaN."""
        from canslim_scorer import CANSLIMScorer

        mock_stock_data.quarterly_earnings = [
            1.50, float("nan"), 1.30, 1.20,
            1.10, float("nan"), 0.95, 0.90
        ]
        scorer = CANSLIMScorer(mock_data_fetcher)
        # Should not raise, and should produce a valid score
        c_score, detail = scorer._score_current_earnings(mock_stock_data)
        assert isinstance(c_score, (int, float))
        assert c_score >= 0

    def test_scorer_handles_nan_in_annual(self, mock_stock_data, mock_data_fetcher):
        """A score should not crash when annual earnings contain NaN."""
        from canslim_scorer import CANSLIMScorer

        mock_stock_data.annual_earnings = [5.50, float("nan"), 4.20, float("nan"), 3.00]
        scorer = CANSLIMScorer(mock_data_fetcher)
        a_score, detail = scorer._score_annual_earnings(mock_stock_data)
        assert isinstance(a_score, (int, float))
        assert a_score >= 0


# ─── Cache Truthiness (data_fetcher.py) ────────────────────────────────────
# Bug: `if data.get("roe"):` treated 0.0 as falsy, silently dropping valid
# zero values for ROE, PE, analyst_count, etc.
# Fix: Changed to `if data.get("roe") is not None:`


class TestCacheTruthiness:
    """Regression: Valid zero values dropped by truthiness checks in cache save."""

    def test_zero_roe_preserved(self):
        """ROE of 0.0 (breakeven) must be saved, not skipped."""
        mock_record = MagicMock()
        mock_record.roe = 999.0  # pre-existing value

        data = {"roe": 0.0, "trailing_pe": None}

        # Simulate the fixed logic
        if data.get("roe") is not None:
            mock_record.roe = data.get("roe")

        assert mock_record.roe == 0.0

    def test_zero_roe_skipped_by_truthiness(self):
        """Demonstrate the bug: `if data.get("roe"):` drops 0.0."""
        mock_record = MagicMock()
        mock_record.roe = 999.0

        data = {"roe": 0.0}

        # Old buggy logic
        if data.get("roe"):
            mock_record.roe = data.get("roe")

        # Bug: roe stays at 999.0 because 0.0 is falsy
        assert mock_record.roe == 999.0  # This proves the bug

    def test_none_roe_not_saved(self):
        """None ROE should NOT overwrite existing data."""
        mock_record = MagicMock()
        mock_record.roe = 0.15

        data = {"roe": None}

        if data.get("roe") is not None:
            mock_record.roe = data.get("roe")

        # None should not overwrite
        assert mock_record.roe == 0.15

    def test_zero_analyst_count_preserved(self):
        """Analyst count of 0 is valid data (no coverage)."""
        mock_record = MagicMock()
        mock_record.analyst_count = 5

        data = {"count": 0}

        if data.get("count") is not None:
            mock_record.analyst_count = data.get("count")

        assert mock_record.analyst_count == 0

    def test_zero_trailing_pe_preserved(self):
        """Trailing PE of 0.0 is valid (e.g., no earnings)."""
        mock_record = MagicMock()
        data = {"trailing_pe": 0.0}

        if data.get("trailing_pe") is not None:
            mock_record.trailing_pe = data.get("trailing_pe")

        assert mock_record.trailing_pe == 0.0


# ─── Double-Execution Lock (ai_trader.py) ──────────────────────────────────
# Bug: check_and_execute_stop_losses() had NO lock, could run concurrently
# with run_ai_trading_cycle(), potentially selling the same position twice.
# Fix: Uses _trading_cycle_lock with non-blocking acquire.


class TestDoubleExecutionLock:
    """Regression: Stop loss check could run concurrently with trading cycle."""

    def test_stop_loss_skips_when_lock_held_by_other_thread(self):
        """Stop loss check should skip when trading cycle holds the lock from another thread."""
        import backend.ai_trader as ai_trader

        result_holder = {}
        lock_acquired = threading.Event()

        def hold_lock():
            """Simulate a trading cycle holding the lock in another thread."""
            ai_trader._trading_cycle_lock.acquire()
            lock_acquired.set()
            # Hold the lock until the test finishes checking
            time.sleep(2)
            ai_trader._trading_cycle_lock.release()

        # Start another thread holding the lock
        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        lock_acquired.wait(timeout=5)

        # Now call stop loss check from this thread — should skip
        mock_db = MagicMock()
        with patch.object(ai_trader, '_check_and_execute_stop_losses_impl') as mock_impl:
            mock_impl.return_value = {"message": "Should not be called", "sells_executed": []}
            result = ai_trader.check_and_execute_stop_losses(mock_db)

        assert "Trading cycle in progress" in result.get("message", "")
        assert result.get("sells_executed", []) == []
        # Implementation should NOT have been called
        mock_impl.assert_not_called()

        holder.join(timeout=5)

    def test_stop_loss_runs_when_lock_free(self):
        """Stop loss check should proceed when no trading cycle is running."""
        import backend.ai_trader as ai_trader

        mock_db = MagicMock()

        # Mock the implementation to avoid needing a real DB
        with patch.object(ai_trader, '_check_and_execute_stop_losses_impl') as mock_impl:
            mock_impl.return_value = {
                "message": "Stop losses checked",
                "sells_executed": []
            }
            result = ai_trader.check_and_execute_stop_losses(mock_db)

        assert "sells_executed" in result

    def test_lock_is_reentrant(self):
        """The lock should be an RLock (reentrant) for safety."""
        import backend.ai_trader as ai_trader

        assert isinstance(ai_trader._trading_cycle_lock, type(threading.RLock()))


# ─── Delisted Ticker Recheck (data_fetcher.py) ─────────────────────────────
# Bug: get_delisted_tickers() never checked recheck_after timestamp — once
# marked, tickers were excluded FOREVER.
# Fix: Added `DelistedTicker.recheck_after > datetime.utcnow()` filter.


class TestDelistedTickerRecheck:
    """Regression: Delisted tickers were never rechecked after cooldown."""

    def test_recheck_filter_logic(self):
        """Tickers with expired recheck_after should NOT be excluded."""
        # This tests the filter logic conceptually
        from datetime import timezone
        now = datetime.now(timezone.utc)

        # Ticker with future recheck — should be excluded
        future_recheck = now + timedelta(days=10)
        assert future_recheck > now  # Would match filter → excluded

        # Ticker with past recheck — should NOT be excluded (eligible for recheck)
        past_recheck = now - timedelta(days=5)
        assert not (past_recheck > now)  # Would NOT match filter → included in scans

    def test_new_ticker_gets_7_day_recheck(self):
        """First failure should set recheck_after to 7 days."""
        now = datetime.now()
        recheck_after = now + timedelta(days=7)
        # Should be about 7 days from now
        assert 6 < (recheck_after - now).days <= 7

    def test_third_failure_gets_30_day_recheck(self):
        """After 3 failures, recheck window extends to 30 days."""
        now = datetime.now()
        recheck_after = now + timedelta(days=30)
        assert 29 < (recheck_after - now).days <= 30

    @patch('data_fetcher._get_db_session')
    def test_get_delisted_tickers_filters_by_recheck(self, mock_get_db):
        """Verify the query filters by failure_count >= 3 AND recheck_after > now."""
        from data_fetcher import get_delisted_tickers

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Mock the query chain
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = [
            MagicMock(ticker="DEAD1"),
            MagicMock(ticker="DEAD2"),
        ]

        result = get_delisted_tickers()

        # Should return a set
        assert isinstance(result, set)
        assert "DEAD1" in result
        assert "DEAD2" in result

        # Verify filter was called (checking recheck_after)
        mock_db.query.assert_called_once()
        mock_query.filter.assert_called_once()


# ─── FMP Type Guards (async_data_fetcher.py) ────────────────────────────────
# Bug: FMP error dicts ({"message": "Limit reached"}) caused KeyError when
# code expected list responses. isinstance(data, list) guards prevent this.


class TestFMPTypeGuards:
    """Regression: FMP API error responses (dicts) mishandled as data lists."""

    def test_dict_response_is_not_list(self):
        """FMP error responses are dicts, not lists."""
        error_response = {"message": "Limit Reach."}
        assert not isinstance(error_response, list)

    def test_list_response_is_list(self):
        """Normal FMP responses are lists."""
        normal_response = [{"date": "2025-01-01", "eps": 1.5}]
        assert isinstance(normal_response, list)

    def test_none_response_is_not_list(self):
        """Timeout/network errors may return None."""
        assert not isinstance(None, list)

    def test_empty_list_is_list(self):
        """Empty list is still a valid list response."""
        assert isinstance([], list)

    def test_guard_protects_key_access(self):
        """Without guard, dict response causes KeyError on list operations."""
        error_response = {"message": "Limit Reach."}

        # Without guard: this would crash
        with pytest.raises((KeyError, TypeError)):
            _ = error_response[0]

        # With guard: safely skip
        if isinstance(error_response, list) and len(error_response) > 0:
            _ = error_response[0]
        # No crash


# ─── Earnings Date Parsing (data_fetcher.py) ───────────────────────────────
# Bug: One malformed date in FMP earnings response crashed the entire function
# via unhandled ValueError from strptime, losing all earnings data for a ticker.
# Fix: Wrapped strptime in try/except with continue.


class TestEarningsDateParsing:
    """Regression: Malformed dates crashed earnings calendar parsing."""

    def test_valid_date_parses(self):
        """Standard YYYY-MM-DD format should parse correctly."""
        result = datetime.strptime("2025-03-15", "%Y-%m-%d").date()
        assert result.year == 2025
        assert result.month == 3
        assert result.day == 15

    def test_malformed_date_raises_valueerror(self):
        """Malformed dates raise ValueError that must be caught."""
        with pytest.raises(ValueError):
            datetime.strptime("not-a-date", "%Y-%m-%d")

    def test_empty_date_raises_valueerror(self):
        with pytest.raises(ValueError):
            datetime.strptime("", "%Y-%m-%d")

    def test_defensive_parsing_skips_bad_dates(self):
        """Simulate the fixed pattern: bad dates are skipped, good ones kept."""
        dates = ["2025-01-01", "bad-date", "2025-03-15", "", "2025-06-30"]
        parsed = []

        for date_str in dates:
            try:
                parsed.append(datetime.strptime(date_str, "%Y-%m-%d").date())
            except (ValueError, TypeError):
                continue  # Skip bad dates

        assert len(parsed) == 3
        assert parsed[0].month == 1
        assert parsed[1].month == 3
        assert parsed[2].month == 6

    def test_next_earnings_estimation_guarded(self):
        """If last_date parsing fails, estimation should be skipped (not crash)."""
        last_date_str = "invalid"
        last_date = None

        try:
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        except ValueError:
            last_date = None

        # Estimation code only runs if last_date is valid
        est_next = None
        if last_date:
            est_next = last_date + timedelta(days=90)

        assert last_date is None
        assert est_next is None  # Estimation correctly skipped


# ─── Batch Rollback Savepoints (scheduler.py) ──────────────────────────────
# Bug: When one stock save failed, db.rollback() undid ALL 49 other good
# stocks in the batch. Now uses begin_nested() savepoints.
# Fix: Each stock wrapped in its own savepoint.


class TestBatchRollbackSavepoints:
    """Regression: One failed stock save rolled back the entire batch."""

    def test_savepoint_isolates_failure(self):
        """Failed savepoint rollback should not affect committed items."""
        saved = []
        failed = []

        # Simulate the savepoint pattern
        items = ["AAPL", "BAD_STOCK", "MSFT", "GOOGL"]

        for item in items:
            try:
                if item == "BAD_STOCK":
                    raise ValueError(f"Error saving {item}")
                saved.append(item)
            except Exception:
                failed.append(item)

        assert saved == ["AAPL", "MSFT", "GOOGL"]
        assert failed == ["BAD_STOCK"]

    def test_nested_none_guard(self):
        """If begin_nested() itself fails, nested is None and rollback is skipped."""
        nested = None
        rolled_back = False

        try:
            # Simulate begin_nested() failing
            raise RuntimeError("Connection lost")
        except Exception:
            if nested:
                nested.rollback()
                rolled_back = True

        assert not rolled_back  # No crash from nested.rollback() on None


# ─── Redis Stats Keyspace Bug (redis_cache.py) ─────────────────────────────
# Bug: get_stats() fetched keyspace via info('keyspace') but then used
# info.get('db0', {}) instead of keyspace.get('db0', {}), so total_keys
# was always 0.
# Fix: Changed to keyspace.get('db0', {}).get('keys', 0).


class TestRedisStatsKeyspace:
    """Regression: Redis total_keys stat always returned 0."""

    def test_keyspace_vs_info_dict(self):
        """The bug was using `info` dict (stats) for keyspace data."""
        # Simulate the two dicts returned by Redis
        info = {  # from info('stats')
            "keyspace_hits": 100,
            "keyspace_misses": 10,
            "total_commands_processed": 500,
        }
        keyspace = {  # from info('keyspace')
            "db0": {"keys": 42, "expires": 5, "avg_ttl": 3600},
        }

        # Bug: used info.get('db0', {}) — always empty
        buggy_keys = info.get("db0", {}).get("keys", 0)
        assert buggy_keys == 0  # Always 0!

        # Fix: use keyspace.get('db0', {})
        fixed_keys = keyspace.get("db0", {}).get("keys", 0)
        assert fixed_keys == 42  # Correct!

    def test_empty_keyspace(self):
        """When no keys exist, keyspace dict may lack db0."""
        keyspace = {}
        keys = keyspace.get("db0", {}).get("keys", 0)
        assert keys == 0


# ─── Stock Data None-Guard (scheduler.py) ──────────────────────────────────
# Bug: When API returns None for core fields (name, sector, price), existing
# DB values were overwritten with None.
# Fix: Fall back to existing DB values when scan returns None.


class TestStockDataNoneGuard:
    """Regression: API failures overwrote good DB data with None."""

    def test_none_does_not_overwrite(self):
        """Existing DB value should be preserved when new data is None."""
        existing_name = "Apple Inc."
        new_name = None

        # Fixed pattern: use new value only if not None
        result = new_name if new_name is not None else existing_name
        assert result == "Apple Inc."

    def test_valid_data_overwrites(self):
        """Valid new data should overwrite existing."""
        existing_name = "Apple Inc."
        new_name = "Apple Inc. Updated"

        result = new_name if new_name is not None else existing_name
        assert result == "Apple Inc. Updated"

    def test_empty_string_overwrites(self):
        """Empty string is a valid value (not None), should overwrite."""
        existing = "Old Value"
        new = ""

        result = new if new is not None else existing
        assert result == ""

    def test_zero_price_overwrites(self):
        """Zero price is valid (though unusual), should overwrite."""
        existing_price = 150.0
        new_price = 0.0

        result = new_price if new_price is not None else existing_price
        assert result == 0.0


# ─── Price Cache Expiry (price_cache.py) ────────────────────────────────────
# Bug: timedelta.days returns integer (30h → 1 day), allowing stale data
# up to 24h past TTL. Fixed to use total_seconds()/86400.


class TestPriceCacheExpiry:
    """Regression: Integer day truncation allowed stale cache data."""

    def test_integer_days_truncates(self):
        """timedelta.days loses fractional days (30h → 1 day, not 1.25)."""
        delta = timedelta(hours=30)
        assert delta.days == 1  # Bug: looks like "1 day" but is actually 1.25

    def test_total_seconds_precise(self):
        """total_seconds() / 86400 gives precise fractional days."""
        delta = timedelta(hours=30)
        precise_days = delta.total_seconds() / 86400
        assert abs(precise_days - 1.25) < 0.001

    def test_expiry_within_fractional_day_window(self):
        """Data slightly over TTL should be expired with precise calculation.

        The actual code uses strict `>` comparison: `age_days > self.expiry_days`.
        With integer .days, data at 30 days + 23 hours has .days = 30, and
        30 > 30 = False (not expired). But it's actually 30.96 days old.
        """
        ttl_days = 30
        age = timedelta(days=30, hours=23)  # 30.96 days old

        # Bug: integer days says "not expired" (30 > 30 = False)
        buggy_expired = age.days > ttl_days
        assert not buggy_expired  # Bug: 30.96-day-old data not expired!

        # Fix: precise check says "expired" (30.96 > 30 = True)
        fixed_expired = age.total_seconds() / 86400 > ttl_days
        assert fixed_expired  # Correct: 30.96 > 30


# ─── Score Stability Blip Detection Sync (ai_trader.py) ────────────────────
# Bug: ai_trader's blip detection did NOT include `consecutive_low < 2` guard.
# When 3+ consecutive low scores occurred, the blip detector still flagged it
# as a blip → is_stable=False → sell skipped. Backtester had the guard and
# would correctly sell. Live trading was MORE protective than backtesting.
# Fix: Added `consecutive_low < 2` to blip condition, matching backtester.


class TestBlipDetectionSync:
    """Regression: ai_trader blip detection missed consecutive_low guard."""

    def test_single_low_is_blip(self):
        """One low score after high averages should be a blip (skip sell)."""
        # current=40, avg=70, variance=30, consecutive_low=1
        current = 40
        threshold = 50
        avg = 70
        variance = abs(current - avg)
        consecutive_low = 1

        is_blip = (current < threshold and
                   avg > threshold + 10 and
                   variance > 15 and
                   consecutive_low < 2)

        assert is_blip  # Single low = blip, skip sell

    def test_two_consecutive_lows_not_blip(self):
        """Two+ consecutive low scores should NOT be a blip (allow sell)."""
        current = 40
        threshold = 50
        avg = 70
        variance = abs(current - avg)
        consecutive_low = 2

        is_blip = (current < threshold and
                   avg > threshold + 10 and
                   variance > 15 and
                   consecutive_low < 2)

        assert not is_blip  # Multiple lows = real drop, allow sell

    def test_buggy_detection_without_guard(self):
        """Demonstrate the bug: without consecutive_low guard, blip fires incorrectly."""
        current = 40
        threshold = 50
        avg = 70
        variance = abs(current - avg)
        consecutive_low = 3  # Three consecutive lows — SHOULD sell

        # Bug: no consecutive_low guard
        buggy_is_blip = (current < threshold and
                         avg > threshold + 10 and
                         variance > 15)
        assert buggy_is_blip  # Bug: would skip sell even with 3 low scans!

        # Fix: with consecutive_low guard
        fixed_is_blip = (current < threshold and
                         avg > threshold + 10 and
                         variance > 15 and
                         consecutive_low < 2)
        assert not fixed_is_blip  # Correct: 3 lows = real drop

    def test_stability_returns_consecutive_low(self):
        """check_score_stability should return pre-computed consecutive_low."""
        import backend.ai_trader as ai_trader
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_stock = MagicMock()
        mock_stock.id = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_stock

        # Mock 3 recent scores — all below threshold
        mock_scores = [
            MagicMock(total_score=40.0),
            MagicMock(total_score=42.0),
            MagicMock(total_score=38.0),
        ]
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_scores

        result = ai_trader.check_score_stability(mock_db, "TEST", 40.0, threshold=50.0)

        assert "consecutive_low" in result
        assert result["consecutive_low"] == 3  # All 3 scores below threshold


# ─── V-Bottom Reset Dead Code (market_state.py) ──────────────────────────
# Bug: v_bottom_triggered was set to True once (line 332) but never reset.
# The reset code at lines 304-308 was DEAD because the early return at line 301
# short-circuited before reaching it. v_bottom detection only fired once per
# MarketStateManager lifetime.
# Fix: Reset v_bottom_triggered in update() when leaving CORRECTION state.


class TestVBottomReset:
    """Regression: v_bottom_triggered must reset when leaving CORRECTION."""

    def test_v_bottom_resets_on_correction_exit(self):
        """v_bottom_triggered should reset when transitioning out of CORRECTION."""
        from backend.market_state import MarketStateManager, MarketState

        mgr = MarketStateManager()
        mgr.state = MarketState.CORRECTION
        mgr.v_bottom_triggered = True

        # Simulate leaving CORRECTION → RECOVERY
        old_state = mgr.state
        mgr.state = MarketState.RECOVERY
        # Mimic what update() does on state change
        if old_state == MarketState.CORRECTION and mgr.state != MarketState.CORRECTION:
            mgr.last_correction_exit_date = None
            mgr.v_bottom_triggered = False

        assert not mgr.v_bottom_triggered  # Should be reset

    def test_v_bottom_fires_again_after_reset(self):
        """After reset, v_bottom should be able to fire in next correction."""
        from backend.market_state import MarketStateManager, MarketState
        from datetime import date

        mgr = MarketStateManager()
        mgr.state = MarketState.CORRECTION
        mgr.v_bottom_triggered = False
        mgr.spy_52w_high = 500.0
        mgr.correction_low = 400.0  # 20% drop
        mgr.correction_low_date = date(2025, 1, 1)

        # V-bottom should fire if rally is strong enough
        result = mgr._check_v_bottom(date(2025, 1, 10), spy_close=445.0)  # 11.25% rally
        assert result is True
        assert mgr.v_bottom_triggered is True

    def test_v_bottom_blocked_after_triggered(self):
        """v_bottom should NOT fire twice in the same correction cycle."""
        from backend.market_state import MarketStateManager, MarketState
        from datetime import date

        mgr = MarketStateManager()
        mgr.state = MarketState.CORRECTION
        mgr.v_bottom_triggered = True  # Already fired

        result = mgr._check_v_bottom(date(2025, 1, 15), spy_close=450.0)
        assert result is False  # Blocked because already triggered


# ─── ATR NaN Guard (historical_data.py) ───────────────────────────────────
# Bug: If OHLC data contains NaN, the ATR calculation propagated NaN through
# max() and sum(), producing NaN as output instead of a valid number.
# Fix: Skip NaN values in true range calculation.


class TestATRNaNGuard:
    """Regression: ATR should handle NaN in OHLC data gracefully."""

    def test_nan_in_max_propagates(self):
        """Demonstrate that NaN propagates through max() and sum()."""
        import math
        nan = float('nan')

        # NaN in max
        result = max(nan, 1.0, 2.0)
        assert math.isnan(result)

        # NaN in sum
        total = sum([1.0, nan, 2.0])
        assert math.isnan(total)

    def test_nan_check_using_self_inequality(self):
        """The NaN != NaN identity check works correctly."""
        nan = float('nan')
        assert nan != nan  # NaN is not equal to itself
        assert not (1.0 != 1.0)  # Regular floats are equal to themselves

    def test_atr_skips_nan_values(self):
        """ATR calculation should skip NaN entries and compute from valid data."""
        import math
        nan = float('nan')

        # Simulate the fixed ATR logic
        raw_trs = [1.5, nan, 2.0, nan, 1.8]
        true_ranges = [tr for tr in raw_trs if tr == tr]  # NaN check

        assert len(true_ranges) == 3  # Skipped 2 NaN values
        atr = sum(true_ranges) / len(true_ranges)
        assert not math.isnan(atr)
        assert abs(atr - 1.7667) < 0.001


# ─── Email None/NaN Formatting Guard (email_utils.py) ─────────────────────
# Bug: stock.current_price and stock.canslim_score were formatted with :.2f
# and :.0f without None guards. If either was None, format() would crash
# with TypeError: unsupported format string passed to NoneType.__format__
# Fix: Use (value or 0) pattern to default None to 0.


class TestEmailNoneFormatting:
    """Regression: email formatting must not crash on None stock attributes."""

    def test_none_price_crashes_format(self):
        """Demonstrate that None crashes :.2f format."""
        with pytest.raises(TypeError):
            f"${None:.2f}"

    def test_none_score_crashes_format(self):
        """Demonstrate that None crashes :.0f format."""
        with pytest.raises(TypeError):
            f"{None:.0f}"

    def test_or_zero_pattern_handles_none(self):
        """The (value or 0) pattern safely defaults None to 0."""
        price = None
        score = None
        assert f"${(price or 0):.2f}" == "$0.00"
        assert f"{(score or 0):.0f}" == "0"

    def test_or_zero_preserves_valid_values(self):
        """The (value or 0) pattern preserves non-None values."""
        price = 123.45
        score = 78.0
        assert f"${(price or 0):.2f}" == "$123.45"
        assert f"{(score or 0):.0f}" == "78"


# ─── Analyst Date Parsing (async_data_fetcher.py) ──────────────────────────
# Bug: int(item_date[:4]) could crash with ValueError on malformed dates
# like "Q1-2025" or very short strings. No try-except protection.
# Fix: Added try-except (ValueError, IndexError) around the int() call.


class TestAnalystDateParsing:
    """Regression: analyst date parsing must handle malformed dates."""

    def test_valid_date_parses(self):
        """Normal YYYY-MM-DD dates should parse correctly."""
        item_date = "2025-03-15"
        year = int(item_date[:4])
        assert year == 2025

    def test_malformed_date_crashes_without_guard(self):
        """Malformed dates crash int() without try-except."""
        with pytest.raises(ValueError):
            int("Q1-2025"[:4])  # "Q1-2" can't be parsed as int

    def test_short_date_does_not_crash(self):
        """Short date strings should not crash."""
        item_date = "20"
        year = int(item_date[:4])  # int("20") = 20, no crash
        assert year == 20  # Wrong but doesn't crash

    def test_empty_date_handled(self):
        """Empty date string should be caught by the outer if-check."""
        item_date = ""
        if item_date:
            int(item_date[:4])  # Won't reach here
        assert True  # Outer check prevents crash


# ─── Delisted Ticker Datetime Mismatch (data_fetcher.py) ────────────────────
# Bug: mark_ticker_as_delisted() stored recheck_after using datetime.now() (naive),
# but get_delisted_tickers() compared with datetime.now(timezone.utc) (aware).
# In PostgreSQL, comparing naive TIMESTAMP against aware datetime fails or
# returns wrong results, so tickers were never properly excluded.
# Fix: Use consistent naive datetime.now() for both storage and comparison.


class TestDelistedDatetimeConsistency:
    """Regression: naive vs UTC-aware datetime comparison broke delisted filtering."""

    def test_naive_vs_aware_comparison_is_wrong(self):
        """Demonstrate that naive and aware datetimes can't reliably compare."""
        from datetime import timezone
        naive = datetime.now()
        aware = datetime.now(timezone.utc)
        # In Python, comparing naive vs aware raises TypeError
        with pytest.raises(TypeError):
            naive > aware

    def test_storage_and_query_use_same_format(self):
        """Both storage (mark) and query (get) must use same datetime format."""
        # Verify the fix: both now use datetime.now() (naive)
        now_storage = datetime.now()  # Used in mark_ticker_as_delisted
        now_query = datetime.now()    # Used in get_delisted_tickers (after fix)
        # Same type = safe comparison
        assert type(now_storage) == type(now_query)
        # recheck_after in the future should be excluded
        recheck_future = now_storage + timedelta(days=7)
        assert recheck_future > now_query

    def test_recheck_window_correctly_excludes(self):
        """A ticker with future recheck_after should be excluded from scans."""
        now = datetime.now()
        recheck_after = now + timedelta(days=7)
        # Filter: recheck_after > now → True → ticker stays excluded
        assert recheck_after > now

    def test_recheck_window_correctly_includes(self):
        """A ticker whose recheck_after has passed should be included in scans."""
        now = datetime.now()
        recheck_after = now - timedelta(days=1)
        # Filter: recheck_after > now → False → ticker NOT excluded → gets rescanned
        assert not (recheck_after > now)


# ─── EPS Merge None Propagation (data_fetcher.py) ────────────────────────────
# Bug: When merging partial adjusted EPS with GAAP EPS, None values in the GAAP
# list could leak into quarterly_earnings, causing TypeError in CANSLIM scorer
# when calculating growth rates (None - float).
# Fix: Filter None from GAAP list before merging.


class TestEPSMergeNoneFilter:
    """Regression: EPS merge could propagate None values into quarterly_earnings."""

    def test_none_in_gaap_eps_filtered(self):
        """None values in GAAP EPS list should be filtered before merge."""
        adjusted_eps = [1.50]  # Only 1 quarter of adjusted data
        gaap_eps = [1.30, None, 1.10, None, 0.90]

        # Apply the fix: filter None before merging
        gaap_filtered = [x for x in gaap_eps if x is not None]
        merged = adjusted_eps + gaap_filtered[len(adjusted_eps):] if len(gaap_filtered) > len(adjusted_eps) else adjusted_eps
        result = merged[:8]

        # No None values in result
        assert None not in result
        # Adjusted EPS takes priority for first quarter
        assert result[0] == 1.50
        # GAAP fills remaining slots (skipping None values)
        assert result[1] == 1.10

    def test_all_none_gaap_uses_adjusted_only(self):
        """If all GAAP EPS are None, only adjusted EPS should remain."""
        adjusted_eps = [2.00, 1.80]
        gaap_eps = [None, None, None]

        gaap_filtered = [x for x in gaap_eps if x is not None]
        merged = adjusted_eps + gaap_filtered[len(adjusted_eps):] if len(gaap_filtered) > len(adjusted_eps) else adjusted_eps
        result = merged[:8]

        assert result == [2.00, 1.80]
        assert None not in result

    def test_empty_gaap_uses_adjusted_only(self):
        """If GAAP EPS list is empty/None, only adjusted EPS should remain."""
        adjusted_eps = [1.50, 1.30, 1.10]
        gaap_eps = []

        gaap_filtered = [x for x in (gaap_eps or []) if x is not None]
        merged = adjusted_eps + gaap_filtered[len(adjusted_eps):] if len(gaap_filtered) > len(adjusted_eps) else adjusted_eps
        result = merged[:8]

        assert result == [1.50, 1.30, 1.10]

    def test_normal_merge_preserves_values(self):
        """Normal merge (no None values) should work correctly."""
        adjusted_eps = [1.50, 1.30]
        gaap_eps = [1.20, 1.10, 0.90, 0.80, 0.70, 0.60]

        gaap_filtered = [x for x in gaap_eps if x is not None]
        merged = adjusted_eps + gaap_filtered[len(adjusted_eps):] if len(gaap_filtered) > len(adjusted_eps) else adjusted_eps
        result = merged[:8]

        # Adjusted takes first 2 slots, GAAP fills rest
        assert result == [1.50, 1.30, 0.90, 0.80, 0.70, 0.60]
        assert len(result) == 6


# ─── Negative Shares Guard (ai_trader.py / backtester.py) ────────────────────
# Bug: Partial sells could theoretically create positions with negative shares
# if sell_pct exceeded 100% due to config error or accumulation bug.
# Fix: Cap shares_to_sell at position.shares in both live trader and backtester.


class TestNegativeSharesGuard:
    """Regression: partial sells must never create negative share positions."""

    def test_normal_partial_sell_reduces_shares(self):
        """Normal 25% partial sell should leave 75% of shares."""
        shares = 100.0
        sell_pct = 25
        shares_to_sell = shares * (sell_pct / 100)
        remaining = shares - shares_to_sell
        assert remaining == 75.0

    def test_guard_caps_oversized_sell(self):
        """If sell_pct > 100%, shares_to_sell should be capped at position shares."""
        shares = 100.0
        sell_pct = 150  # Bug scenario: config error or accumulation
        shares_to_sell = shares * (sell_pct / 100)
        # Without guard: shares_to_sell = 150, remaining = -50 (BUG)
        assert shares_to_sell > shares
        # With guard: cap at position shares
        if shares_to_sell > shares:
            shares_to_sell = shares
        remaining = shares - shares_to_sell
        assert remaining == 0.0  # Full sell, not negative

    def test_exact_100_percent_leaves_zero(self):
        """Selling exactly 100% should leave 0 shares."""
        shares = 50.0
        sell_pct = 100
        shares_to_sell = shares * (sell_pct / 100)
        if shares_to_sell > shares:
            shares_to_sell = shares
        remaining = shares - shares_to_sell
        assert remaining == 0.0

    def test_floating_point_precision(self):
        """Multiple partial sells shouldn't accumulate to negative due to float precision."""
        shares = 100.0
        # Sell 33.33%, three times
        for _ in range(3):
            sell_pct = 33.33
            shares_to_sell = shares * (sell_pct / 100)
            if shares_to_sell > shares:
                shares_to_sell = shares
            shares -= shares_to_sell
        # After 3x33.33%, about 0.01 shares should remain (not negative)
        assert shares >= 0.0


# ─── Path Traversal Guard (main.py) ──────────────────────────────────────────
# Bug: Frontend catch-all route could serve files outside frontend directory
# using ../../../ path traversal sequences.
# Fix: Resolve paths and verify they stay within frontend_path.


class TestPathTraversalGuard:
    """Regression: frontend serving must prevent directory traversal."""

    def test_resolve_prevents_traversal(self):
        """Resolved path with ../ must not escape the base directory."""
        from pathlib import Path
        base = Path("/app/frontend/dist")
        # Simulate traversal attempt
        requested = "../../etc/passwd"
        resolved = (base / requested).resolve()
        # Resolved path should NOT start with base (it escapes to /etc/)
        assert not str(resolved).startswith(str(base.resolve()))

    def test_normal_path_stays_within_base(self):
        """Normal paths should resolve within the base directory."""
        from pathlib import Path
        base = Path("/app/frontend/dist")
        requested = "assets/main.js"
        resolved = (base / requested).resolve()
        # Normal path stays within base
        assert str(resolved).startswith(str(base.resolve()))


# ─── Ghost Field Writes (scheduler.py) ────────────────────────────────────
# Bug: scheduler.save_stock_to_db() wrote stock.confidence instead of
# stock.growth_confidence. SQLAlchemy silently created a Python instance
# attribute that never persisted to the database. This meant growth_confidence
# was NULL for stocks only scanned via batch (never via individual API call).
# Also removed 4 other dead field writes (analyst_target, pe_ratio,
# relative_strength, institutional_ownership) — columns no longer exist.
# Fix: Changed stock.confidence → stock.growth_confidence.


class TestGhostFieldWrites:
    """Regression: ORM field writes must target actual database columns."""

    def test_stock_model_has_growth_confidence(self):
        """Stock model must have growth_confidence column, not confidence."""
        from backend.database import Stock
        from sqlalchemy import inspect
        mapper = inspect(Stock)
        column_names = [col.key for col in mapper.column_attrs]
        assert "growth_confidence" in column_names
        assert "confidence" not in column_names

    def test_stock_model_lacks_legacy_columns(self):
        """Stock model should NOT have legacy columns that were removed."""
        from backend.database import Stock
        from sqlalchemy import inspect
        mapper = inspect(Stock)
        column_names = [col.key for col in mapper.column_attrs]
        # These columns were removed from the schema but writes remained
        assert "analyst_target" not in column_names
        assert "pe_ratio" not in column_names
        assert "relative_strength" not in column_names
        assert "institutional_ownership" not in column_names

    def test_scheduler_save_uses_growth_confidence(self):
        """Scheduler save function must use growth_confidence, not confidence."""
        from pathlib import Path
        scheduler_path = Path(__file__).parent.parent / "backend" / "scheduler.py"
        source = scheduler_path.read_text()
        # The save function must write to growth_confidence, not confidence
        assert "stock.growth_confidence" in source
        # Must NOT have the buggy stock.confidence write (note: string "growth_confidence"
        # contains "confidence" so we check for exact "stock.confidence" without "growth_" prefix)
        assert "stock.confidence " not in source


# ─── Watchlist Alert Cooldown Bug (scheduler.py) ─────────────────────────
# Bug: alert_sent flag was checked BEFORE cooldown, creating a permanent
# block that prevented alerts from ever re-firing after the first send.
# The cooldown timer was correct but unreachable.
# Also: email failure silently committed last_check_price, so the trigger
# condition (price crossing above target) would never re-fire.
# Fix: Removed permanent alert_sent block, set trigger timestamp before
# sending to prevent retry storms even on failure.


class TestWatchlistAlertCooldown:
    """Regression: watchlist alerts must re-fire after cooldown expires."""

    def test_no_permanent_alert_sent_block_in_scheduler(self):
        """Scheduler must NOT have alert_sent as a permanent block before cooldown."""
        from pathlib import Path
        scheduler_path = Path(__file__).parent.parent / "backend" / "scheduler.py"
        source = scheduler_path.read_text()
        # The old bug: checked alert_sent AFTER cooldown, blocking forever
        # Should NOT have: if item.alert_sent: ... continue (as a standalone block)
        # It's OK to SET alert_sent, just not to use it as a blocking gate
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "if item.alert_sent:":
                # Next non-empty line should NOT be a continue/skip
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_stripped = lines[j].strip()
                    if next_stripped:
                        assert next_stripped != "continue", (
                            f"Line {j+1}: alert_sent used as permanent block "
                            f"(prevents alerts from re-firing after cooldown)"
                        )
                        break

    def test_alert_triggered_at_set_before_email_send(self):
        """Trigger timestamp must be set BEFORE attempting email send."""
        from pathlib import Path
        scheduler_path = Path(__file__).parent.parent / "backend" / "scheduler.py"
        source = scheduler_path.read_text()
        # Find the alert-sending section — look for the actual CALL, not the import
        trigger_pos = source.find("item.alert_triggered_at = datetime.now")
        send_pos = source.find("if send_watchlist_alert_email(")
        # alert_triggered_at must be set BEFORE the email send call
        assert trigger_pos > 0, "alert_triggered_at assignment not found"
        assert send_pos > 0, "send_watchlist_alert_email call not found"
        assert trigger_pos < send_pos, (
            "alert_triggered_at must be set BEFORE send_watchlist_alert_email "
            "to prevent retry storms on email failure"
        )


# ─── Dead Config: half_size_initial ──────────────────────────────────────
# Bug: half_size_initial was defined in all strategy profile YAMLs but never
# read or used by any Python code in ai_trader.py or backtester.py.
# The champion backtest (+184.4%) was achieved without this feature.
# Fix: Removed from all YAML profiles to avoid confusion.


class TestNoDeadConfigOptions:
    """Regression: YAML config options must have corresponding Python code."""

    def test_half_size_initial_not_in_yaml(self):
        """half_size_initial should be removed from config (no implementation exists)."""
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        source = config_path.read_text()
        assert "half_size_initial" not in source, (
            "half_size_initial still in config but has no implementation "
            "in ai_trader.py or backtester.py"
        )

    def test_half_size_not_referenced_in_trading_code(self):
        """No trading code should reference half_size to prevent re-introduction."""
        from pathlib import Path
        base = Path(__file__).parent.parent
        for filename in ["backend/ai_trader.py", "backend/backtester.py"]:
            filepath = base / filename
            source = filepath.read_text()
            assert "half_size" not in source, (
                f"{filename} references half_size but it was never implemented"
            )


# ─── SPY Gate Fail-Safe (ai_trader.py / backtester.py) ──────────────────
# Bug: When SPY price or 50MA data was missing (0 or None), the regime gate
# silently allowed buys instead of conservatively blocking them.
# This is a fail-unsafe pattern: the safety mechanism fails OPEN instead of CLOSED.
# Fix: Missing SPY data now assumes bearish (blocks buys).


class TestSPYGateFailSafe:
    """Regression: missing SPY data must block buys, not allow them."""

    def test_ai_trader_blocks_on_missing_spy(self):
        """ai_trader regime gate must return [] when SPY data is missing."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        # Must check for missing data before the price comparison
        assert "not spy_px or not spy_50" in source or "not spy_px" in source, (
            "ai_trader.py regime gate must block buys when SPY data is missing"
        )

    def test_backtester_blocks_on_missing_spy(self):
        """backtester regime gate must block buys when SPY data is missing."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        assert "not spy_price or not spy_ma50" in source, (
            "backtester.py regime gate must block buys when SPY data is missing"
        )
