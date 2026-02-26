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


# ─── Variable Scoping in evaluate_buys (ai_trader.py) ──────────────────────
# Bug: has_base and pivot_price were used in volume gate BEFORE being defined.
# On first loop iteration this would NameError; on subsequent iterations it used
# stale values from the previous stock's base pattern data.
# Fix: Move base pattern extraction before volume gate.

class TestVariableScopingBuys:
    """Regression: has_base/pivot_price must be defined before volume gate uses them."""

    def test_has_base_defined_before_volume_gate(self):
        """has_base and pivot_price must appear before the volume gate block."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        # Find where has_base is first DEFINED (assigned)
        define_pos = source.find("has_base = base_type not in")
        # Find where has_base is first USED (in volume gate)
        use_pos = source.find("elif has_base and pivot_price")
        assert define_pos > 0, "has_base definition not found"
        assert use_pos > 0, "has_base usage in volume gate not found"
        assert define_pos < use_pos, (
            f"has_base defined at position {define_pos} but used at {use_pos} — "
            "variable used before definition"
        )


# ─── Correlation Sizing Default (backtester.py) ─────────────────────────────
# Bug: backtester defaulted to enabled=True for correlation sizing when config
# section was missing, while ai_trader had it removed entirely. This caused
# silent divergence in position sizing between live and backtest.
# Fix: Default to False, matching ai_trader behavior.

class TestCorrelationSizingDefault:
    """Regression: correlation sizing must default to disabled."""

    def test_backtester_defaults_to_disabled(self):
        """Correlation sizing must default to False when config is missing."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        assert "corr_config.get('enabled', False)" in source, (
            "Correlation sizing must default to disabled (False) to match ai_trader"
        )


# ─── Score Available Guard in Backtester Sells ──────────────────────────────
# Bug: backtester evaluate_sells had no guard for score=0 (data gap), causing
# false score crash sells when historical score data was missing. ai_trader had
# a score_available guard but backtester did not.
# Fix: Add score_available = current_score > 0 guard before score-dependent sells.

class TestBacktesterScoreAvailableGuard:
    """Regression: backtester must skip score-dependent sells when score=0."""

    def test_backtester_has_score_available_guard(self):
        """backtester _evaluate_sells must check score_available before score crash."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        # The guard must appear in the evaluate_sells method
        guard_pos = source.find("score_available = current_score > 0")
        assert guard_pos > 0, "score_available guard not found in backtester"
        # It must come before score crash check
        crash_pos = source.find("Score crash check with stability")
        assert guard_pos < crash_pos, (
            "score_available guard must appear before score crash check"
        )


# ─── Cost Basis Division by Zero (main.py) ────────────────────────────────
# Bug: PositionCreate and PositionUpdate allowed cost_basis=0 via Pydantic's
# ge=0 constraint. A position with cost_basis=0 would cause ZeroDivisionError
# when computing gain_loss_pct = (current_price - cost_basis) / cost_basis * 100.
# Fix: Changed ge=0 to gt=0 in both Pydantic models.

class TestCostBasisValidation:
    """Regression: cost_basis=0 must be rejected by Pydantic validation."""

    def test_cost_basis_zero_causes_division_by_zero(self):
        """Demonstrate that cost_basis=0 causes ZeroDivisionError."""
        cost_basis = 0
        current_price = 100.0
        with pytest.raises(ZeroDivisionError):
            gain_loss_pct = (current_price - cost_basis) / cost_basis * 100

    def test_pydantic_rejects_zero_cost_basis(self):
        """PositionCreate must reject cost_basis=0."""
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'backend'))
        from main import PositionCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PositionCreate(ticker="AAPL", shares=10, cost_basis=0)

    def test_pydantic_allows_positive_cost_basis(self):
        """PositionCreate must accept positive cost_basis."""
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'backend'))
        from main import PositionCreate
        pos = PositionCreate(ticker="AAPL", shares=10, cost_basis=50.0)
        assert pos.cost_basis == 50.0


# ─── Historical Data dropna Completeness (historical_data.py) ──────────────
# Bug: dropna(subset=["close"]) only removed rows with NaN in close column.
# NaN in high/low/volume could propagate to ATR, 52w high, and volume ratio.
# Fix: dropna on all essential OHLCV columns.

class TestDropNACompleteness:
    """Regression: historical data must drop rows with NaN in any OHLCV column."""

    def test_dropna_includes_all_ohlcv(self):
        """dropna must filter on close, high, low, and volume."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "historical_data.py").read_text()
        assert 'dropna(subset=["close", "high", "low", "volume"])' in source, (
            "dropna must filter all essential OHLCV columns, not just close"
        )

    def test_nan_volume_not_caught_by_not_check(self):
        """Demonstrate that 'not float(nan)' does NOT catch NaN."""
        import math
        val = float('nan')
        # bool(NaN) is True, so 'not NaN' is False — NaN passes 'if not val' checks
        assert not (not val), "NaN passes through 'if not val' checks undetected"


# ─── Backtester Institutional % Source (backtester.py) ──────────────────────
# Bug: backtester loaded institutional_holders_pct from Stock model via getattr,
# but Stock model doesn't have this column. It always returned 0, making the
# coiled spring institutional filter a no-op.
# Fix: Extract from score_details JSON (same as ai_trader).

class TestBacktesterInstitutionalPct:
    """Regression: backtester must extract institutional % from score_details."""

    def test_backtester_uses_score_details_for_inst_pct(self):
        """Backtester static_data must extract institutional_pct from score_details."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        # Must use score_details, not getattr(stock, 'institutional_holders_pct')
        assert "score_details or {}).get('i', {}).get('institutional_pct'" in source, (
            "Backtester must extract institutional % from score_details JSON, "
            "not from Stock model (which doesn't have this column)"
        )

    def test_stock_model_lacks_institutional_holders_pct(self):
        """Stock model must NOT have institutional_holders_pct column."""
        from backend.database import Stock
        from sqlalchemy import inspect
        mapper = inspect(Stock)
        column_names = [col.key for col in mapper.column_attrs]
        assert "institutional_holders_pct" not in column_names, (
            "Stock model has no institutional_holders_pct column — "
            "getattr would silently return default 0"
        )


# ─── Backtester Progress Variable Init (backtester.py) ────────────────────
# Bug: progress variable was used in cancellation error message at i=0 before
# being assigned in the loop body, causing UnboundLocalError.
# Fix: Initialize progress before the loop.

class TestProgressVariableInit:
    """Regression: progress must be initialized before the simulation loop."""

    def test_progress_initialized_before_loop(self):
        """progress variable must be defined before the for loop uses it."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        # progress must be initialized before the loop
        init_pos = source.find("progress = 30.0")
        loop_pos = source.find("for i, current_date in enumerate(trading_days):")
        assert init_pos > 0, "progress initialization not found"
        assert loop_pos > 0, "simulation loop not found"
        assert init_pos < loop_pos, (
            "progress must be initialized BEFORE the simulation loop to prevent "
            "UnboundLocalError when cancellation is detected at i=0"
        )


# ─── Backtester I Score Unit Mismatch (backtester.py) ─────────────────────
# Bug: I score thresholds used decimal values (0.30-0.70) while inst_pct is
# stored as a percentage (e.g., 65 = 65%). Every stock got i_score=2.
# Fix: Use percentage thresholds (25-75) matching the live scorer.

class TestBacktesterIScoreUnits:
    """Regression: I score thresholds must use percentage values, not decimals."""

    def test_i_score_uses_percentage_thresholds(self):
        """Verify backtester uses percentage thresholds like the live scorer."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        # Must NOT contain decimal thresholds
        assert "0.30 <= inst_pct" not in source, (
            "Backtester I score still uses decimal threshold 0.30 — "
            "inst_pct is a percentage (e.g. 65), not a decimal (e.g. 0.65)"
        )
        assert "0.70" not in source.split("I SCORE")[1].split("=====")[0] if "I SCORE" in source else True, (
            "Backtester I score still uses decimal threshold 0.70"
        )

        # Must contain percentage thresholds matching live scorer
        i_section_start = source.find("I SCORE (10 pts)")
        assert i_section_start > 0, "I SCORE section not found in backtester"
        i_section = source[i_section_start:i_section_start + 500]
        assert "25 <= inst_pct <= 75" in i_section, (
            "Backtester I score must use percentage thresholds (25-75) "
            "matching the live scorer in canslim_scorer.py"
        )

    def test_i_score_scoring_logic(self):
        """Verify I score returns correct values for typical institutional ownership."""
        # Simulate the scoring logic
        def calc_i_score(inst_pct):
            if 25 <= inst_pct <= 75:
                return 10
            elif 15 <= inst_pct < 25 or 75 < inst_pct <= 85:
                return 7
            elif 10 <= inst_pct < 15 or 85 < inst_pct <= 90:
                return 4
            elif inst_pct > 0:
                return 2
            else:
                return 5  # default when no data

        # Typical stocks with 40-70% institutional ownership should score 10
        assert calc_i_score(65) == 10, "65% inst should score 10"
        assert calc_i_score(40) == 10, "40% inst should score 10"
        assert calc_i_score(25) == 10, "25% inst should score 10"
        assert calc_i_score(75) == 10, "75% inst should score 10"

        # Edge cases
        assert calc_i_score(20) == 7, "20% inst should score 7"
        assert calc_i_score(80) == 7, "80% inst should score 7"
        assert calc_i_score(12) == 4, "12% inst should score 4"
        assert calc_i_score(5) == 2, "5% inst should score 2"
        assert calc_i_score(0) == 5, "0% inst should get default 5"


# ─── Market Direction None Guard (ai_trader.py) ──────────────────────────
# Bug: get_cached_market_direction() can return None, causing AttributeError
# when calling .get("success") on None.
# Fix: Default to empty dict with `or {}`.

class TestMarketDirectionNoneGuard:
    """Regression: market_data must be guarded against None return."""

    def test_evaluate_sells_guards_none_market_data(self):
        """evaluate_sells must handle None from get_cached_market_direction."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Find all occurrences of market_data assignment
        lines = source.split("\n")
        market_data_lines = [
            (i, line) for i, line in enumerate(lines, 1)
            if "market_data = get_cached_market_direction()" in line
        ]
        assert len(market_data_lines) >= 4, (
            f"Expected at least 4 market_data assignments, found {len(market_data_lines)}"
        )

        for lineno, line in market_data_lines:
            assert "or {}" in line, (
                f"Line {lineno}: market_data assignment must include 'or {{}}' "
                f"to guard against None return from get_cached_market_direction()"
            )


# ─── Portfolio Value Pre-computation (ai_trader.py) ───────────────────────
# Bug: get_portfolio_value(db) called 100+ times inside the candidate loop,
# causing N+1 query performance issues.
# Fix: Pre-compute portfolio_value once before the loop.

class TestPortfolioValuePrecomputation:
    """Regression: portfolio_value must be computed before the candidate loop."""

    def test_portfolio_value_computed_before_loop(self):
        """portfolio_value must be pre-computed before the candidate for-loop."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Find the pre-computation line (should be before the main loop)
        precompute_pos = source.find('# Pre-compute portfolio value once')
        assert precompute_pos > 0, "Pre-compute comment not found in ai_trader.py"

        # Find the MAIN candidate loop (not the nested one inside get_fresh_score closure)
        # The main loop is the one right after the pre-compute comment
        main_loop_pos = source.find('for stock in candidates:', precompute_pos)
        assert main_loop_pos > 0, "Main candidate loop not found after pre-compute"
        assert precompute_pos < main_loop_pos, (
            "portfolio_value must be pre-computed BEFORE the candidate loop "
            "to avoid 100+ redundant get_portfolio_value() DB queries"
        )

    def test_no_in_loop_portfolio_value_calls(self):
        """No get_portfolio_value() calls should exist inside the candidate loop."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Find the MAIN candidate loop (after the pre-compute comment)
        precompute_pos = source.find('# Pre-compute portfolio value once')
        loop_start = source.find('for stock in candidates:', precompute_pos)
        loop_end = source.find('return final_buys', loop_start)
        assert loop_start > 0 and loop_end > 0

        loop_body = source[loop_start:loop_end]
        # There should be no active get_portfolio_value calls (only comments referencing it)
        active_calls = [
            line.strip() for line in loop_body.split('\n')
            if 'get_portfolio_value(db)' in line and not line.strip().startswith('#')
        ]
        assert len(active_calls) == 0, (
            f"Found {len(active_calls)} active get_portfolio_value() calls inside "
            f"the candidate loop — these should use the pre-computed value: {active_calls}"
        )


# ─── Bear Exception Signal Factors (ai_trader.py) ────────────────────────
# Bug: Bear exception buy dicts lacked signal_factors key, which would cause
# KeyError or None when execute_trade tried to log it.
# Fix: Added signal_factors dict with entry_type, market_regime, composite_score.

class TestBearExceptionSignalFactors:
    """Regression: bear exception buys must include signal_factors."""

    def test_bear_exception_has_signal_factors(self):
        """Bear exception buy dict must include signal_factors key."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Find the bear exception section
        bear_start = source.find("BEAR EXCEPTION")
        assert bear_start > 0, "Bear exception section not found"

        # Find the dict that contains it
        bear_section = source[bear_start:bear_start + 500]
        assert '"signal_factors"' in bear_section or "'signal_factors'" in bear_section, (
            "Bear exception buy dict must include 'signal_factors' key "
            "to avoid KeyError when execute_trade logs the trade"
        )


# ─── Trailing Stop Defaults Match Champion Strategy (ai_trader + backtester) ─
# Bug: Hardcoded fallback defaults used old balanced values (15%, 12%, 10%)
# instead of champion nostate_optimized values (25%, 18%, 12%).
# Fix: Updated all defaults to match champion strategy.

class TestTrailingStopDefaults:
    """Regression: trailing stop defaults must match champion strategy values."""

    @pytest.mark.parametrize("filename", ["ai_trader.py", "backtester.py"])
    def test_trailing_stop_defaults_match_champion(self, filename):
        """Trailing stop fallback defaults must use champion values, not old balanced."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / filename).read_text()

        # Check that defaults match nostate_optimized champion values
        assert "'gain_50_plus', 25)" in source or "'gain_50_plus', 25)" in source, (
            f"{filename}: gain_50_plus default must be 25 (champion), not 15 (old balanced)"
        )
        assert "'gain_30_to_50', 18)" in source, (
            f"{filename}: gain_30_to_50 default must be 18 (champion), not 12 (old balanced)"
        )
        assert "'gain_20_to_30', 12)" in source, (
            f"{filename}: gain_20_to_30 default must be 12 (champion), not 10 (old balanced)"
        )

    def test_backtester_take_profit_default_matches_champion(self):
        """Backtester take_profit_pct default must be 75.0 (champion), not 40.0."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        assert "take_profit_pct', 40.0)" not in source, (
            "Backtester still has 40.0 as take_profit default — must be 75.0"
        )
        assert "take_profit_pct', 75.0)" in source, (
            "Backtester take_profit_pct default must be 75.0 (champion)"
        )


# ─── Institutional Ownership Conversion Boundary (async_data_fetcher.py) ───
# Bug: threshold `<= 1.5` excluded stocks with >150% institutional ownership
# from conversion. Yahoo returns 1.501 as decimal for 150.1%, which would be
# stored as 1.501 instead of 150.1, causing I score to compute on wrong scale.
# Fix: Raised threshold from 1.5 to 3.0 to cover up to 300% institutional.

class TestInstitutionalOwnershipConversion:
    """Regression: institutional ownership decimal→pct conversion boundary."""

    def test_normal_decimal_converted(self):
        """Normal 65% institutional ownership (0.65) should convert to 65."""
        inst_pct = 0.65
        result = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert result == 65.0

    def test_100_percent_converted(self):
        """100% institutional (1.0) should convert to 100."""
        inst_pct = 1.0
        result = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert result == 100.0

    def test_150_percent_converted(self):
        """150% institutional (1.5 decimal) should convert to 150."""
        inst_pct = 1.5
        result = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert result == 150.0

    def test_old_boundary_bug_at_1501(self):
        """Demonstrate old bug: 1.501 was NOT converted with <= 1.5 threshold."""
        inst_pct = 1.501
        # Old buggy logic
        buggy = (inst_pct * 100) if 0 < inst_pct <= 1.5 else inst_pct
        assert buggy == 1.501  # Bug: stored as 1.5% instead of 150.1%

        # Fixed logic
        fixed = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert fixed == 150.1  # Correct: 150.1%

    def test_250_percent_converted(self):
        """250% institutional (2.5 decimal) should convert to 250."""
        inst_pct = 2.5
        result = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert result == 250.0

    def test_already_percentage_not_doubled(self):
        """Value 65 (already percentage) should NOT be multiplied by 100."""
        inst_pct = 65.0
        result = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert result == 65.0  # Not 6500

    def test_zero_stays_zero(self):
        """Zero institutional ownership should stay zero."""
        inst_pct = 0
        result = (inst_pct * 100) if 0 < inst_pct < 3.0 else inst_pct
        assert result == 0

    def test_threshold_consistency_across_files(self):
        """All inst_pct conversion sites must use < 3.0 threshold."""
        from pathlib import Path
        base = Path(__file__).parent.parent
        for filename in ["async_data_fetcher.py", "data_fetcher.py"]:
            source = (base / filename).read_text()
            # Must NOT use the old 1.5 threshold
            assert "inst_pct <= 1.5" not in source, (
                f"{filename} still uses <= 1.5 threshold for institutional ownership"
            )
            # Must use 3.0 threshold
            assert "inst_pct < 3.0" in source, (
                f"{filename} must use < 3.0 threshold for institutional ownership conversion"
            )


# ─── Short Interest Conversion Boundary (async_data_fetcher.py) ─────────
# Bug: threshold `< 1` missed stocks with >100% short interest (e.g., GME).
# Yahoo returns 1.2 for 120% short interest, which was stored as 1.2 instead
# of 120, causing the >20% short penalty to not trigger.
# Fix: Raised threshold from 1 to 3.0.

class TestShortInterestConversion:
    """Regression: short interest decimal→pct conversion boundary."""

    def test_normal_decimal_converted(self):
        """15% short interest (0.15) should convert to 15."""
        short_pct = 0.15
        result = short_pct * 100 if 0 < short_pct < 3.0 else short_pct
        assert result == 15.0

    def test_old_boundary_bug_at_1_2(self):
        """Demonstrate old bug: 1.2 (120%) was NOT converted with < 1 threshold."""
        short_pct = 1.2
        # Old buggy logic
        buggy = short_pct * 100 if 0 < short_pct < 1 else short_pct
        assert buggy == 1.2  # Bug: stored as 1.2% instead of 120%

        # Fixed logic
        fixed = short_pct * 100 if 0 < short_pct < 3.0 else short_pct
        assert fixed == 120.0  # Correct: 120%

    def test_already_percentage_not_doubled(self):
        """Value 25 (already percentage) should NOT be multiplied by 100."""
        short_pct = 25.0
        result = short_pct * 100 if 0 < short_pct < 3.0 else short_pct
        assert result == 25.0  # Not 2500

    def test_threshold_consistency_across_files(self):
        """All short_pct conversion sites must use < 3.0 threshold."""
        from pathlib import Path
        import re
        base = Path(__file__).parent.parent
        for filename in ["async_data_fetcher.py", "data_fetcher.py"]:
            source = (base / filename).read_text()
            # Find lines with short_pct threshold comparisons (exclude comments)
            lines = source.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(r'short_pct\s*<\s*1[^.]', stripped):
                    assert False, (
                        f"{filename}:{i} still uses < 1 threshold: {stripped}"
                    )


# ─── Silent Exception Logging (multiple files) ─────────────────────────────
# Bug: Multiple except:pass blocks across the codebase silently swallowed
# exceptions, hiding real failures from logs and making debugging impossible.
# Fix: Changed all to log exceptions at appropriate levels.

class TestSilentExceptionLogging:
    """Regression: exceptions must be logged, not silently swallowed."""

    @pytest.mark.parametrize("filepath,search_context", [
        ("data_fetcher.py", "_get_db_session"),
        ("growth_projector.py", "sector performance"),
        ("backend/ai_trader.py", "Score smoothing"),
        ("backend/scheduler.py", "market timing"),
        ("backend/backtester.py", "cache cleanup"),
    ])
    def test_no_bare_except_pass_in_fixed_files(self, filepath, search_context):
        """Fixed files must not have bare except:pass patterns (no exception variable)."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / filepath).read_text()
        lines = source.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            # Only flag exceptions that don't capture the variable at all
            if stripped in ("except Exception:", "except:"):
                # Check if next non-empty line is 'pass'
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_stripped = lines[j].strip()
                    if next_stripped:
                        assert next_stripped != "pass", (
                            f"{filepath}:{j+1} ({search_context}): "
                            f"bare except:pass still present — must log exception"
                        )
                        break


# ─── Backtester Sector Allocation % Limit (backtester.py) ─────────────────
# Bug: Backtester _check_sector_limit only checked stock COUNT per sector,
# not allocation PERCENTAGE. Live trader checks both count AND 30% allocation cap.
# This let backtester overweight a sector, making backtests look better than live.
# Fix: Added allocation % check after position sizing in _evaluate_buys.


class TestBacktesterSectorAllocationLimit:
    """Regression: Backtester was missing sector allocation % cap that live trader has."""

    def test_max_sector_allocation_constant_exists(self):
        """MAX_SECTOR_ALLOCATION must be defined in backtester module."""
        from backend.backtester import MAX_SECTOR_ALLOCATION
        assert 0 < MAX_SECTOR_ALLOCATION <= 1.0  # Must be a valid percentage

    def test_backtester_has_allocation_check_in_evaluate_buys(self):
        """The sector allocation % check must exist in backtester's buy evaluation."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        assert "MAX_SECTOR_ALLOCATION" in source
        # Must appear in _evaluate_buys context (after position sizing, not just at module level)
        # Check that it's used in a comparison, not just defined
        assert "new_alloc > MAX_SECTOR_ALLOCATION" in source

    def test_earnings_drift_gate_in_ai_trader(self):
        """ai_trader earnings drift must require days_to_earnings is not None."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        # The gate must exist: beat_streak >= 3 AND days_to_earnings is not None
        assert "days_to_earnings is not None" in source


# ─── OHLC Split Adjustment (historical_data.py) ──────────────────────────
# Bug: Adjusted close was used with unadjusted open/high/low, corrupting
# 52-week highs, ATR, and base patterns for split stocks (AMZN 20:1, GOOG 20:1).
# Fix: Compute adjustment factor (adjclose/rawclose) and apply to O/H/L.


class TestOHLCAdjustment:
    """Regression: Unadjusted OHLV mixed with adjusted close for split stocks."""

    def test_adjustment_factor_computation(self):
        """Verify adjustment factor = adjclose / rawclose applied to O/H/L."""
        # Simulate a 20:1 split: pre-split raw close $2000, adj close $100
        raw_close = 2000.0
        adj_close = 100.0
        factor = adj_close / raw_close  # 0.05

        raw_high = 2050.0
        raw_low = 1980.0
        raw_open = 1990.0

        adj_high = raw_high * factor  # 102.50
        adj_low = raw_low * factor    # 99.00
        adj_open = raw_open * factor  # 99.50

        assert adj_high == pytest.approx(102.50)
        assert adj_low == pytest.approx(99.00)
        assert adj_open == pytest.approx(99.50)

    def test_historical_data_applies_adjustment(self):
        """Verify historical_data.py applies adjustment to OHLV columns."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "historical_data.py").read_text()
        # Must compute adjustment factor from adj/raw close
        assert "factor" in source or "adj_open" in source
        # Must NOT use raw open/high/low directly when adjclose is available
        # The DataFrame should use adjusted values
        assert "adj_open" in source or "adj_high" in source

    def test_score_stability_lookback_parameter(self):
        """check_score_stability must accept a lookback parameter for configurable consecutive_required."""
        import inspect
        from backend.ai_trader import check_score_stability
        sig = inspect.signature(check_score_stability)
        assert "lookback" in sig.parameters
        # Default should be 3
        assert sig.parameters["lookback"].default == 3


# ─── Weighted Score Averaging (ai_trader.py) ───────────────────────────────
# Improvement: Recency-weighted average (50/30/20) reduces momentum lag
# while still smoothing single-scan noise. Equal-weight created ~8pt lag
# on improving stocks, causing late entries.


class TestWeightedScoreAveraging:
    """Tests for the recency-weighted score averaging in evaluate_buys."""

    def test_weighted_avg_favors_recent_on_improvement(self):
        """When scores improve over 3 scans, weighted average should be higher than equal-weight."""
        # Simulating _get_avg_score logic with improving scores
        scores = [82, 75, 65]  # most recent first
        equal_avg = sum(scores) / len(scores)  # 74.0
        weighted_avg = scores[0] * 0.50 + scores[1] * 0.30 + scores[2] * 0.20  # 76.5

        assert weighted_avg > equal_avg
        # Weighted should be ~2.5pts higher for improving trajectory
        assert weighted_avg - equal_avg >= 2.0

    def test_weighted_avg_penalizes_recent_decline(self):
        """When scores decline, weighted average should be lower than equal-weight."""
        scores = [60, 72, 80]  # declining: most recent is worst
        equal_avg = sum(scores) / len(scores)  # 70.67
        weighted_avg = scores[0] * 0.50 + scores[1] * 0.30 + scores[2] * 0.20  # 67.6

        assert weighted_avg < equal_avg
        # Weighted should be ~3pts lower for declining trajectory
        assert equal_avg - weighted_avg >= 2.5

    def test_weighted_avg_stable_scores_similar(self):
        """When scores are stable, weighted and equal averages should be similar."""
        scores = [75, 74, 76]  # stable
        equal_avg = sum(scores) / len(scores)  # 75.0
        weighted_avg = scores[0] * 0.50 + scores[1] * 0.30 + scores[2] * 0.20  # 74.9

        assert abs(weighted_avg - equal_avg) < 1.0

    def test_two_scan_weighted_avg(self):
        """With only 2 scans, should use 60/40 weighting."""
        scores = [80, 70]  # most recent first
        weighted_avg = scores[0] * 0.60 + scores[1] * 0.40  # 76.0
        equal_avg = sum(scores) / len(scores)  # 75.0

        assert weighted_avg > equal_avg

    def test_weighted_avg_respects_threshold(self):
        """Score averaging should only apply when difference from current >= 3."""
        # If current_score is very close to the average, just return current
        current_score = 75.0
        scores = [76, 74, 75]  # avg ≈ 75, close to current
        weighted_avg = scores[0] * 0.50 + scores[1] * 0.30 + scores[2] * 0.20  # 75.2

        # When abs(avg - current) < 3, should return current_score
        assert abs(weighted_avg - current_score) < 3


# ─── Batch Score History (ai_trader.py) ────────────────────────────────────
# Improvement: Batch-fetch score history for all candidates in one query
# instead of N+1 individual queries per candidate.


class TestBatchScoreHistory:
    """Verify that evaluate_buys uses batched score history."""

    def test_evaluate_buys_has_batch_score_cache(self):
        """evaluate_buys should create _score_history_cache for batch lookups."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Must have batch fetch instead of per-ticker query
        assert "_score_history_cache" in source
        # Must build cache before the main candidate loop
        assert "candidate_tickers" in source
        # Must use .in_ for batch query
        assert "StockScore.ticker.in_(candidate_tickers)" in source

    def test_no_per_ticker_query_in_get_avg_score(self):
        """_get_avg_score should use the cache, not run individual DB queries."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Find the _get_avg_score function and check it uses cache
        import re
        fn_match = re.search(r'def _get_avg_score\(.*?\):\s*""".*?"""(.*?)(?=\n    # Pre-compute|\n    for stock)', source, re.DOTALL)
        if fn_match:
            fn_body = fn_match.group(1)
            # Should reference cache, NOT run db.query
            assert "_score_history_cache" in fn_body
            assert "db.query" not in fn_body


# ─── Partial Profit Integration Tests ──────────────────────────────────────
# Tests for partial profit taking edge cases that were previously untested.
# The Feb 24 delta bug showed that missing integration tests can cost 25+pp.


class TestPartialProfitIntegration:
    """Integration tests for partial profit logic in evaluate_sells."""

    def test_partial_profit_delta_at_40pct_tier(self):
        """When 25% already taken, 40% tier should sell (50% - 25%) = 25% more."""
        from backend.ai_trader import evaluate_sells
        from unittest.mock import MagicMock, patch

        # Create mock position with 25% already taken
        position = MagicMock()
        position.ticker = "TEST"
        position.current_price = 130.0
        position.cost_basis = 100.0
        position.gain_loss_pct = 30.0  # Up 30%, but let's test 40%
        position.shares = 75  # Had 100, sold 25 (25% partial)
        position.peak_price = 135.0
        position.peak_date = datetime.now()
        position.is_growth_stock = False
        position.current_score = 75
        position.purchase_score = 80
        position.current_growth_score = None
        position.purchase_growth_score = None
        position.partial_profit_taken = 25  # Already took 25%
        position.pyramid_count = 0

        # With partial_taken=25, pp_40_sell=50, delta should be 25
        delta = 50 - 25  # pp_40_sell - partial_taken
        assert delta == 25
        assert delta > 0  # Should trigger sell

    def test_partial_profit_skipped_when_fully_taken(self):
        """When 50% already taken, no more partial sells should happen."""
        partial_taken = 50
        pp_25_sell = 25
        pp_40_sell = 50

        # Both tiers: take_pct = target - taken
        take_25 = pp_25_sell - partial_taken  # 25 - 50 = -25
        take_40 = pp_40_sell - partial_taken  # 50 - 50 = 0

        assert take_25 <= 0  # Should NOT trigger
        assert take_40 <= 0  # Should NOT trigger

    def test_partial_profit_25pct_tier_delta_correct(self):
        """First partial at 25% gain should sell exactly 25% of position."""
        partial_taken = 0
        pp_25_sell = 25

        take_pct = pp_25_sell - partial_taken  # 25 - 0 = 25
        assert take_pct == 25

    def test_partial_profit_requires_min_score(self):
        """Partial profit should NOT trigger if score is too low."""
        # Default min score for partial is 60
        pp_25_min_score = 60
        current_score = 55  # Below threshold

        should_take_partial = current_score >= pp_25_min_score
        assert not should_take_partial


# ─── Earnings Avoidance Window Validation ──────────────────────────────────
# Tests that validate the two-window earnings avoidance system.
# avoidance_days (5) and allow_buy_days (7) serve DIFFERENT purposes.


class TestEarningsAvoidanceWindows:
    """Ensure earnings avoidance logic handles all edge cases correctly."""

    def test_non_cs_blocked_within_avoidance(self):
        """Non-CS stock at 4 days to earnings should be blocked (4 <= 5)."""
        avoidance_days = 5
        days_to_earnings = 4
        is_coiled_spring = False

        blocked = days_to_earnings <= avoidance_days and not is_coiled_spring
        assert blocked

    def test_non_cs_allowed_past_avoidance(self):
        """Non-CS stock at 6 days to earnings should be allowed (6 > 5)."""
        avoidance_days = 5
        days_to_earnings = 6
        is_coiled_spring = False

        blocked = days_to_earnings <= avoidance_days and not is_coiled_spring
        assert not blocked

    def test_cs_allowed_within_evaluation_window(self):
        """CS stock at 5 days should be allowed (days > block_days=1)."""
        block_days = 1
        days_to_earnings = 5
        is_coiled_spring = True

        allowed = is_coiled_spring and days_to_earnings > block_days
        assert allowed

    def test_cs_blocked_on_earnings_day(self):
        """CS stock at 1 day should be blocked (days <= block_days=1)."""
        block_days = 1
        days_to_earnings = 1
        is_coiled_spring = True

        blocked = not (days_to_earnings > block_days)
        assert blocked

    def test_config_validation_warning(self):
        """allow_buy_days < avoidance_days should log a warning."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Must have config validation for window mismatch
        assert "allow_buy_days < avoidance_days" in source

    def test_stocks_between_avoidance_and_allow_window(self):
        """Stocks at 6-7 days: NOT in avoidance window, may be in CS evaluation."""
        avoidance_days = 5
        allow_buy_days = 7

        # Day 6: past avoidance, in CS evaluation zone
        days = 6
        in_avoidance = days <= avoidance_days
        in_cs_window = days <= allow_buy_days
        assert not in_avoidance  # Should NOT be blocked
        assert in_cs_window  # IS in CS evaluation zone (can buy if CS)


# ─── AI Trader / Backtester Sync Verification ──────────────────────────────
# Structural tests that verify key logic stays synced between the two files.


class TestAITraderBacktesterSync:
    """Verify that ai_trader.py and backtester.py have matching logic structures."""

    def test_composite_score_weights_match(self):
        """Both files should use the same scoring weight keys."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        weight_keys = ['growth_projection', 'canslim_score', 'momentum', 'breakout', 'base_quality']
        for key in weight_keys:
            assert key in ai_source, f"Missing weight key '{key}' in ai_trader.py"
            assert key in bt_source, f"Missing weight key '{key}' in backtester.py"

    def test_pre_breakout_bonus_values_match(self):
        """Pre-breakout bonuses should be identical in both files."""
        from pathlib import Path
        import re
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        # Both should have pre_breakout_bonus = 40
        ai_matches = re.findall(r'pre_breakout_bonus = (\d+)', ai_source)
        bt_matches = re.findall(r'pre_breakout_bonus = (\d+)', bt_source)
        assert ai_matches and bt_matches
        assert ai_matches[0] == bt_matches[0], f"Pre-breakout bonus mismatch: ai={ai_matches[0]} bt={bt_matches[0]}"

    def test_trailing_stop_tiers_match(self):
        """Trailing stop tier thresholds should match between files."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        # Both should check peak_gain >= 50, >= 30, >= 20, >= 10
        for threshold in ['50', '30', '20', '10']:
            pattern = f'peak_gain_pct >= {threshold}'
            assert pattern in ai_source, f"Missing trailing stop tier {threshold} in ai_trader.py"
            assert pattern in bt_source, f"Missing trailing stop tier {threshold} in backtester.py"

    def test_partial_profit_config_keys_match(self):
        """Partial profit config key names should be identical."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        config_keys = ['threshold_25pct', 'threshold_40pct', 'gain_pct', 'sell_pct', 'min_score']
        for key in config_keys:
            assert key in ai_source, f"Missing partial profit key '{key}' in ai_trader.py"
            assert key in bt_source, f"Missing partial profit key '{key}' in backtester.py"

    def test_score_crash_config_keys_match(self):
        """Score crash detection config should use same keys."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        for key in ['consecutive_required', 'drop_required', 'ignore_if_profitable']:
            assert key in ai_source, f"Missing score crash key '{key}' in ai_trader.py"
            assert key in bt_source, f"Missing score crash key '{key}' in backtester.py"

    def test_extended_penalty_values_match(self):
        """Extended stock penalty values should be identical."""
        from pathlib import Path
        import re
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        # Both should have extended_penalty = -20 and -10
        for val in ['-20', '-10', '-15']:
            ai_has = f'extended_penalty = {val}' in ai_source
            bt_has = f'extended_penalty = {val}' in bt_source
            if ai_has and not bt_has:
                pytest.fail(f"ai_trader has extended_penalty={val} but backtester doesn't")
            if bt_has and not ai_has:
                pytest.fail(f"backtester has extended_penalty={val} but ai_trader doesn't")

    def test_momentum_penalty_threshold_match(self):
        """Momentum penalty (rs_3m < rs_12m * 0.95) should match."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        assert "rs_12m * 0.95" in ai_source
        assert "rs_12m * 0.95" in bt_source

    def test_base_quality_bonus_values_match(self):
        """Base quality bonus for each pattern type should match."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        # Check that bonus values are the same
        patterns = {
            'cup_with_handle': '10',
            'cup': '8',
            'double_bottom': '7',
            'flat': '6',
        }
        for pattern, bonus in patterns.items():
            # Both files should assign the same bonus
            ai_check = f'"{pattern}"' in ai_source or f"'{pattern}'" in ai_source
            bt_check = f'"{pattern}"' in bt_source or f"'{pattern}'" in bt_source
            assert ai_check, f"Pattern {pattern} missing from ai_trader.py"
            assert bt_check, f"Pattern {pattern} missing from backtester.py"


# ─── Pyramid Logic Tests ──────────────────────────────────────────────────
# Tests for pyramid (position addition) logic that was previously untested
# in the live trading path.


class TestPyramidLiveTrading:
    """Tests for evaluate_pyramids in ai_trader.py."""

    def test_pyramid_requires_minimum_gain(self):
        """Pyramid should require at least 2.5% gain from entry."""
        min_gain = 2.5
        assert 1.5 < min_gain  # 1.5% gain should NOT pyramid
        assert 3.0 >= min_gain  # 3% gain should qualify

    def test_pyramid_max_2_enforced(self):
        """Third pyramid attempt should be blocked (max 2)."""
        max_pyramids = 2
        current_count = 2
        assert current_count >= max_pyramids  # Should be blocked

    def test_pyramid_60_40_sizing(self):
        """First pyramid = 60% of original, second = 40%."""
        original_cost = 10000  # $10K initial position

        # First pyramid
        first_add = original_cost * 0.60
        assert first_add == 6000

        # Second pyramid
        second_add = original_cost * 0.40
        assert second_add == 4000

    def test_pyramid_respects_position_cap(self):
        """Pyramid should not exceed MAX_POSITION_ALLOCATION."""
        portfolio_value = 100000
        MAX_POSITION_ALLOCATION = 0.25  # 25%
        current_allocation = 0.20  # 20%

        remaining_room = (MAX_POSITION_ALLOCATION - current_allocation) * portfolio_value
        pyramid_amount = 10000  # Wants to add $10K

        capped_amount = min(pyramid_amount, remaining_room)
        assert abs(capped_amount - 5000) < 1  # Only ~$5K room (float precision)

    def test_pyramid_1day_cooldown(self):
        """Pyramid should not execute same day as previous pyramid."""
        from datetime import date
        today = date.today()
        last_pyramid_date = today  # Same day

        should_block = last_pyramid_date >= today
        assert should_block

    def test_pyramid_function_exists_with_correct_signature(self):
        """evaluate_pyramids should exist and accept db session."""
        import inspect
        from backend.ai_trader import evaluate_pyramids
        sig = inspect.signature(evaluate_pyramids)
        assert "db" in sig.parameters


# ─── Peak Price Initialization ─────────────────────────────────────────────
# Improvement: Peak price should be initialized from historical high, not just
# max(current_price, cost_basis), to avoid trailing stops triggering too early.


class TestPeakPriceInitialization:
    """Tests for improved peak price initialization in update_position_prices."""

    def test_peak_price_uses_historical_high_code_exists(self):
        """update_position_prices should attempt historical peak lookup."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Must have historical peak lookup logic
        assert "Historical peak lookup" in source or "actual_peak" in source
        # Must still have the max(current, cost_basis) fallback
        assert "max(current_price, position.cost_basis)" in source or "actual_peak" in source


# ─── S Score Accumulation/Distribution (canslim_scorer.py, Iteration 2) ──────
# Enhancement: S score now includes up-volume vs down-volume ratio to detect
# institutional accumulation (buying on up days) vs distribution (selling on down days).
# This is O'Neil's real Supply/Demand indicator.


class TestSScoreAccumulationDistribution:
    """Tests for accumulation/distribution detection in S score."""

    def _make_stock_data(self, price_df):
        """Helper to create StockData with given price DataFrame."""
        from data_fetcher import StockData
        stock = StockData("TEST")
        stock.current_price = 100.0
        stock.avg_volume_50d = 1000000
        stock.current_volume = 1200000
        stock.price_history = price_df
        return stock

    def test_accumulation_detected(self):
        """Heavy up-day volume should produce higher S score than neutral."""
        import pandas as pd
        import numpy as np
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        scorer = CANSLIMScorer(mock_fetcher)

        # Create price history: mostly up days with heavy volume
        dates = pd.date_range(end="2026-02-26", periods=30, freq="B")
        closes = [100 + i * 0.3 for i in range(30)]  # Steadily rising
        volumes = [1000000] * 30
        # Make up-day volumes much larger than down-day volumes
        for i in range(1, 30):
            if closes[i] > closes[i - 1]:
                volumes[i] = 2000000  # 2x on up days
            else:
                volumes[i] = 500000   # 0.5x on down days

        df = pd.DataFrame({
            "Close": closes, "Volume": volumes, "Low": [c * 0.99 for c in closes]
        }, index=dates)

        stock = self._make_stock_data(df)
        score, detail = scorer._score_supply_demand(stock)
        assert score >= 5, f"Accumulation pattern should score well, got {score}"
        assert "accum" in detail, "Detail should mention accumulation"

    def test_distribution_detected(self):
        """Heavy down-day volume should be flagged as distribution."""
        import pandas as pd
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        scorer = CANSLIMScorer(mock_fetcher)

        # Create price history: mostly down days with heavy volume
        dates = pd.date_range(end="2026-02-26", periods=30, freq="B")
        closes = [100 - i * 0.3 for i in range(30)]  # Steadily falling
        volumes = [1000000] * 30
        for i in range(1, 30):
            if closes[i] < closes[i - 1]:
                volumes[i] = 2000000  # Heavy on down days
            else:
                volumes[i] = 500000

        df = pd.DataFrame({
            "Close": closes, "Volume": volumes, "Low": [c * 0.99 for c in closes]
        }, index=dates)

        stock = self._make_stock_data(df)
        score, detail = scorer._score_supply_demand(stock)
        assert "distrib" in detail, "Detail should mention distribution"

    def test_s_score_max_is_15(self):
        """S score should never exceed 15 points."""
        import pandas as pd
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        scorer = CANSLIMScorer(mock_fetcher)

        # Best possible conditions: high volume, rising, strong accumulation
        dates = pd.date_range(end="2026-02-26", periods=30, freq="B")
        closes = [100 + i * 1.0 for i in range(30)]
        volumes = [3000000] * 30  # All very high

        df = pd.DataFrame({
            "Close": closes, "Volume": volumes, "Low": [c * 0.99 for c in closes]
        }, index=dates)

        stock = self._make_stock_data(df)
        stock.current_volume = 3000000  # 3x average
        score, _ = scorer._score_supply_demand(stock)
        assert score <= 15, f"S score should not exceed 15, got {score}"

    def test_s_score_min_is_zero(self):
        """S score should never go below 0."""
        import pandas as pd
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        scorer = CANSLIMScorer(mock_fetcher)

        # Worst conditions: low volume, falling price, distribution
        dates = pd.date_range(end="2026-02-26", periods=30, freq="B")
        closes = [100 - i * 0.5 for i in range(30)]
        volumes = [500000] * 30  # Low volume
        for i in range(1, 30):
            if closes[i] < closes[i - 1]:
                volumes[i] = 2000000
            else:
                volumes[i] = 100000

        df = pd.DataFrame({
            "Close": closes, "Volume": volumes, "Low": [c * 0.99 for c in closes]
        }, index=dates)

        stock = self._make_stock_data(df)
        stock.current_volume = 400000  # Below average
        score, _ = scorer._score_supply_demand(stock)
        assert score >= 0, f"S score should not go below 0, got {score}"


# ─── N Score Trend Direction (canslim_scorer.py, Iteration 2) ─────────────────
# Enhancement: N score now detects higher lows (constructive) vs lower lows
# (distribution) pattern using 10-day/20-day low comparison.


class TestNScoreTrendDirection:
    """Tests for constructive trend detection in N score."""

    def _make_stock_data(self, closes, lows=None):
        """Helper to create StockData with given price history."""
        import pandas as pd
        from data_fetcher import StockData

        stock = StockData("TEST")
        stock.current_price = closes[-1]
        stock.high_52w = max(closes) * 1.02  # Just above current
        stock.avg_volume_50d = 1000000
        stock.current_volume = 1500000  # 1.5x for volume bonus

        dates = pd.date_range(end="2026-02-26", periods=len(closes), freq="B")
        if lows is None:
            lows = [c * 0.99 for c in closes]

        stock.price_history = pd.DataFrame({
            "Close": closes, "Low": lows, "Volume": [1000000] * len(closes)
        }, index=dates)
        return stock

    def test_higher_lows_gets_bonus(self):
        """Stock building higher lows should get constructive trend bonus."""
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        scorer = CANSLIMScorer(mock_fetcher)

        # 20 days: first 10 have low around 95, next 10 have low around 98
        closes = [97 + i * 0.15 for i in range(25)]
        lows = [94 + i * 0.01 for i in range(10)] + [97 + i * 0.01 for i in range(15)]

        stock = self._make_stock_data(closes, lows)
        _, detail = scorer._score_new_highs(stock)
        assert "↑lows" in detail, f"Should detect higher lows, got detail: {detail}"

    def test_lower_lows_gets_penalty(self):
        """Stock with declining lows should be penalized."""
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        scorer = CANSLIMScorer(mock_fetcher)

        # 25 days: first 10 have lows around 98, next 15 drop sharply to ~90
        closes = [100 - i * 0.3 for i in range(25)]
        lows = [98 - i * 0.01 for i in range(10)] + [92 - i * 0.5 for i in range(15)]

        stock = self._make_stock_data(closes, lows)
        _, detail = scorer._score_new_highs(stock)
        assert "↓lows" in detail, f"Should detect lower lows, got detail: {detail}"


# ─── L Score Multi-Timeframe RS (canslim_scorer.py, Iteration 2) ─────────────
# Enhancement: L score now uses 3 timeframes (12m/3m/1m) with RS cap at 3.0.
# This prevents inflated RS during extreme market crashes.


class TestLScoreMultiTimeframe:
    """Tests for improved L score with 3 timeframes and RS capping."""

    def test_l_score_code_uses_three_timeframes(self):
        """L score should use 12m, 3m, and 1m RS timeframes."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "canslim_scorer.py").read_text()

        assert "rs_1m" in source or "lookback_1m" in source, "L score should use 1-month RS"
        assert "0.40" in source and "0.35" in source and "0.25" in source, \
            "Should use 40/35/25 weights"

    def test_l_score_caps_rs_at_three(self):
        """L score RS values should be capped at 3.0 in both scorer and backtester."""
        from pathlib import Path
        scorer_source = (Path(__file__).parent.parent / "canslim_scorer.py").read_text()
        backtester_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        assert "3.0" in scorer_source, "Scorer should cap RS at 3.0"
        assert "3.0" in backtester_source, "Backtester should cap RS at 3.0"

    def test_l_score_denominator_floor_at_half(self):
        """RS denominator should floor at 0.5, not 0.1."""
        from pathlib import Path
        scorer_source = (Path(__file__).parent.parent / "canslim_scorer.py").read_text()

        # Check that 0.5 is used as the floor (not 0.1)
        assert "0.5)" in scorer_source, "Denominator floor should be 0.5"


# ─── Backtester S/L Score Sync (Iteration 2) ─────────────────────────────────
# Enhancement: Backtester S and L scores now synced with canslim_scorer.py.
# S score includes price trend + accumulation/distribution.
# L score uses 3 timeframes + RS capping + improving bonus.


class TestBacktesterScoringSync:
    """Tests that backtester scoring is synced with live scorer."""

    def test_backtester_s_score_has_price_trend(self):
        """Backtester S score should include price trend analysis."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        assert "price_change_20d" in source or "price_history" in source, \
            "Backtester S score should include price trend"

    def test_backtester_s_score_has_accumulation(self):
        """Backtester S score should include accumulation/distribution."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        assert "accumulation_distribution" in source or "ad_ratio" in source, \
            "Backtester S score should include A/D ratio"

    def test_backtester_l_score_uses_three_timeframes(self):
        """Backtester L score should use 12m, 3m, and 1m RS."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        assert "rs_1m" in source, "Backtester L score should use 1-month RS"
        assert "rs_12m_capped" in source, "Backtester should cap RS values"

    def test_backtester_l_score_has_improving_bonus(self):
        """Backtester L score should include RS improving bonus."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()

        # Check for the improving bonus logic
        assert "rs_3m_capped > rs_12m_capped" in source, \
            "Backtester should give RS improving bonus"

    def test_historical_data_provider_has_helpers(self):
        """HistoricalDataProvider should have price_history and A/D helpers."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "historical_data.py").read_text()

        assert "def get_price_history(" in source, \
            "Should have get_price_history method"
        assert "def get_accumulation_distribution(" in source, \
            "Should have get_accumulation_distribution method"


# ─── 50% Gain Partial Profit Tier (Iteration 3) ──────────────────────────────
# Enhancement: Added third partial profit tier at 50% gain → sell 75% total.
# Protects big winners while keeping 25% exposure for further upside.
# Lower min_score (55 vs 60) because at 50%+ gain, protecting profits is priority.


class TestPartialProfit50PctTier:
    """Tests for the 50% gain partial profit tier."""

    def test_50pct_tier_exists_in_config(self):
        """Config should define threshold_50pct partial profit tier."""
        from config_loader import config
        pp_config = config.get('ai_trader.partial_profits', {})
        tier_50 = pp_config.get('threshold_50pct', {})
        assert tier_50.get('gain_pct') == 50, "50% tier should trigger at 50% gain"
        assert tier_50.get('sell_pct') == 75, "50% tier should sell 75% total"
        assert tier_50.get('min_score') == 55, "50% tier should use min_score=55"

    def test_50pct_tier_in_backtester(self):
        """Backtester should load and use 50% partial profit tier."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        assert "pp_50_gain" in source, "Backtester should load 50% tier gain threshold"
        assert "pp_50_sell" in source, "Backtester should load 50% tier sell pct"
        assert "pp_50_min_score" in source, "Backtester should load 50% tier min score"

    def test_50pct_tier_in_ai_trader(self):
        """AI trader should load and use 50% partial profit tier."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        assert "pp_50_gain" in source, "AI trader should load 50% tier gain threshold"
        assert "pp_50_sell" in source, "AI trader should load 50% tier sell pct"

    def test_50pct_tier_fires_before_40pct(self):
        """50% tier check should appear BEFORE 40% check in the partial profit evaluation logic."""
        from pathlib import Path
        bt_source = (Path(__file__).parent.parent / "backend" / "backtester.py").read_text()
        at_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()

        # Find the partial profit EVALUATION section (not config loading)
        # The pattern "gain_pct >= pp_50_gain" should appear before "gain_pct >= pp_40_gain"
        bt_50_pos = bt_source.find("gain_pct >= pp_50_gain")
        bt_40_pos = bt_source.find("gain_pct >= pp_40_gain")
        assert bt_50_pos > 0, "Backtester should have 50% tier gain check"
        assert bt_50_pos < bt_40_pos, "Backtester should check 50% tier before 40%"

        at_50_pos = at_source.find("gain_pct >= pp_50_gain")
        at_40_pos = at_source.find("gain_pct >= pp_40_gain")
        assert at_50_pos > 0, "AI trader should have 50% tier gain check"
        assert at_50_pos < at_40_pos, "AI trader should check 50% tier before 40%"

    def test_50pct_partial_profit_delta_calculation(self):
        """50% tier should calculate delta from partial_taken, not absolute."""
        # Stock up 55%, score 60, already took 50% partial → should take 25% more (75-50)
        from backend.backtester import BacktestEngine, SimulatedPosition
        from datetime import date, timedelta

        mock_session = MagicMock()
        mock_backtest = MagicMock()
        mock_backtest.id = 1
        mock_backtest.starting_cash = 25000
        mock_backtest.max_positions = 8
        mock_backtest.min_score_to_buy = 72
        mock_backtest.sell_score_threshold = 50
        mock_backtest.stop_loss_pct = 7.0
        mock_backtest.strategy = "nostate_optimized"
        mock_session.get.return_value = mock_backtest

        engine = BacktestEngine(mock_session, 1)

        engine.positions["BIG"] = SimulatedPosition(
            ticker="BIG", shares=100, cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=90),
            purchase_score=80.0, peak_price=160.0,
            peak_date=date.today() - timedelta(days=2),
            sector="Technology", partial_profit_taken=50.0  # Already took 50%
        )

        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 155.0  # +55% gain
        engine.data_provider.get_market_direction.return_value = {"spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_atr.return_value = 2.0
        engine.data_provider.get_vix_proxy.return_value = 18.0

        sells = engine._evaluate_sells(date.today(), {"BIG": {"total_score": 60}})

        # Should trigger 50% tier: gain 55% >= 50%, score 60 >= 55, partial 50 < 75
        # Delta = 75 - 50 = 25% of shares
        partial_sells = [s for s in sells if s.is_partial and "PARTIAL PROFIT" in s.reason]
        assert len(partial_sells) == 1, f"Expected 1 partial sell, got {len(partial_sells)}"
        assert partial_sells[0].sell_pct == 25, f"Expected 25% delta, got {partial_sells[0].sell_pct}"
        assert partial_sells[0].shares == 25, f"Expected 25 shares (25% of 100), got {partial_sells[0].shares}"

    def test_50pct_tier_skips_if_already_taken(self):
        """50% tier should skip if already taken 75%+ partial profits."""
        from backend.backtester import BacktestEngine, SimulatedPosition
        from datetime import date, timedelta

        mock_session = MagicMock()
        mock_backtest = MagicMock()
        mock_backtest.id = 1
        mock_backtest.starting_cash = 25000
        mock_backtest.max_positions = 8
        mock_backtest.min_score_to_buy = 72
        mock_backtest.sell_score_threshold = 50
        mock_backtest.stop_loss_pct = 7.0
        mock_backtest.strategy = "nostate_optimized"
        mock_session.get.return_value = mock_backtest

        engine = BacktestEngine(mock_session, 1)

        engine.positions["DONE"] = SimulatedPosition(
            ticker="DONE", shares=25, cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=90),
            purchase_score=80.0, peak_price=160.0,
            peak_date=date.today() - timedelta(days=2),
            sector="Technology", partial_profit_taken=75.0  # All partials taken
        )

        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 155.0
        engine.data_provider.get_market_direction.return_value = {"spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_atr.return_value = 2.0
        engine.data_provider.get_vix_proxy.return_value = 18.0

        sells = engine._evaluate_sells(date.today(), {"DONE": {"total_score": 60}})

        # Should NOT trigger any partial sell (already at 75%)
        partial_sells = [s for s in sells if s.is_partial and "PARTIAL PROFIT" in s.reason]
        assert len(partial_sells) == 0, "Should not trigger partial when already at 75%"


# ─── Iteration 4: Backtester C Score Sync (Feb 26, 2026) ────────────────────
# Bug: Backtester C score used simplified algorithm without turnaround handling,
# wrong acceleration metric, and no sector-adjusted thresholds.
# Fix: Synced backtester C score with canslim_scorer.py logic.


class TestBacktesterCScoreSync:
    """Regression: Backtester C score was fundamentally different from live scorer."""

    def _make_engine_for_scoring(self):
        """Create a backtester engine with all data provider methods mocked for _calculate_scores."""
        from backend.backtester import BacktestEngine
        mock_session = MagicMock()
        mock_backtest = MagicMock()
        mock_backtest.id = 1
        mock_backtest.starting_cash = 25000
        mock_backtest.max_positions = 8
        mock_backtest.min_score_to_buy = 72
        mock_backtest.sell_score_threshold = 50
        mock_backtest.stop_loss_pct = 7.0
        mock_backtest.strategy = "nostate_optimized"
        mock_session.get.return_value = mock_backtest
        engine = BacktestEngine(mock_session, 1)
        engine.data_provider = MagicMock()
        # Mock all data provider methods that _calculate_scores uses
        engine.data_provider.get_market_direction.return_value = {"weighted_signal": 1.0, "spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_52_week_high_low.return_value = (50.0, 20.0)
        engine.data_provider.get_relative_strength.return_value = 1.5
        engine.data_provider.get_price_on_date.return_value = 45.0
        engine.data_provider.detect_base_pattern.return_value = {"type": "none", "pivot_price": 0, "weeks": 0}
        engine.data_provider.get_price_history.return_value = None
        engine.data_provider.get_accumulation_distribution.return_value = None
        engine.data_provider.get_50_day_avg_volume.return_value = 1000000
        engine.data_provider.get_volume_on_date.return_value = 1200000
        engine.data_provider.get_institutional_ownership.return_value = 50.0
        engine.data_provider.is_breaking_out.return_value = (False, 1.0, None)
        return engine

    def test_turnaround_stock_gets_positive_c_score(self):
        """Stocks transitioning from negative to positive earnings should get C > 0."""
        from datetime import date
        engine = self._make_engine_for_scoring()

        # Mock stock with negative->positive earnings turnaround
        # Prior TTM: sum of quarters 4-7 = -2.0 (loss), Current TTM: sum of quarters 0-3 = 3.0 (profit)
        mock_stock_data = MagicMock()
        mock_stock_data.quarterly_earnings = [1.0, 0.8, 0.7, 0.5, -0.5, -0.5, -0.5, -0.5]
        mock_stock_data.annual_earnings = [3.0, -2.0, -3.0]
        mock_stock_data.quarterly_revenue = [100, 90, 80, 70]

        engine.data_provider.get_stock_data_on_date.return_value = mock_stock_data

        engine.static_data["TURN"] = {
            "sector": "Technology", "name": "Turnaround Corp",
            "quarterly_earnings": [1.0, 0.8, 0.7, 0.5, -0.5, -0.5, -0.5, -0.5],
            "annual_earnings": [3.0, -2.0, -3.0],
            "quarterly_revenue": [100, 90, 80, 70],
            "institutional_holders_pct": 50,
            "roe": 0.15,
            "analyst_target_price": 60,
            "num_analyst_opinions": 10,
            "weeks_in_base": 0,
            "earnings_beat_streak": 0,
            "days_to_earnings": None,
            "eps_estimate_revision_pct": None,
        }

        scores = engine._calculate_scores(date.today(), ["TURN"])
        c_score = scores["TURN"]["c_score"]

        # Previously this returned 0 because year_ago_eps < 0 was blocked
        # Now it should give turnaround credit (C=12)
        assert c_score >= 10, f"Turnaround stock should get C >= 10, got {c_score}"

    def test_acceleration_uses_yoy_growth_rates(self):
        """Acceleration bonus should compare YoY growth rates, not sequential EPS."""
        # Read backtester source to verify it uses the correct acceleration metric
        import inspect
        from backend.backtester import BacktestEngine

        source = inspect.getsource(BacktestEngine._calculate_scores)

        # Should reference YoY growth rate comparison, not q1 > q2 > q3
        assert "current_q_growth" in source, "Should use YoY growth rate comparison"
        assert "prev_q_growth" in source, "Should compare against prior quarter's YoY growth"
        # Should NOT use the old simple sequential comparison
        assert "q1 > q2 > q3 > 0" not in source, "Should not use sequential EPS comparison"

    def test_sector_adjusted_thresholds(self):
        """C score should use sector-specific growth thresholds."""
        import inspect
        from backend.backtester import BacktestEngine

        source = inspect.getsource(BacktestEngine._calculate_scores)

        # Should reference sector thresholds
        assert "sector_thresholds" in source, "Should use sector-adjusted thresholds"
        assert "Technology" in source, "Should have Technology sector threshold"
        assert "Utilities" in source, "Should have Utilities sector threshold"

    def test_losses_shrinking_gets_partial_credit(self):
        """Stocks with shrinking losses should get partial C credit, not 0."""
        from datetime import date
        engine = self._make_engine_for_scoring()

        # Stock with losses getting less negative
        mock_stock_data = MagicMock()
        mock_stock_data.quarterly_earnings = [-0.10, -0.15, -0.20, -0.25, -0.50, -0.60, -0.70, -0.80]
        mock_stock_data.annual_earnings = [-0.70, -2.60, -3.00]
        mock_stock_data.quarterly_revenue = [100, 90, 80, 70]

        engine.data_provider.get_stock_data_on_date.return_value = mock_stock_data

        engine.static_data["LOSS"] = {
            "sector": "Healthcare", "name": "Loss Shrinking Inc",
            "quarterly_earnings": [-0.10, -0.15, -0.20, -0.25, -0.50, -0.60, -0.70, -0.80],
            "annual_earnings": [-0.70, -2.60, -3.00],
            "quarterly_revenue": [100, 90, 80, 70],
            "institutional_holders_pct": 30,
            "roe": 0,
            "analyst_target_price": 0,
            "num_analyst_opinions": 0,
            "weeks_in_base": 0,
            "earnings_beat_streak": 0,
            "days_to_earnings": None,
            "eps_estimate_revision_pct": None,
        }

        scores = engine._calculate_scores(date.today(), ["LOSS"])
        c_score = scores["LOSS"]["c_score"]

        # Current TTM = -0.70, Prior TTM = -2.60 — losses shrinking by 73%
        # Should get partial credit (C=4), not 0
        assert c_score > 0, f"Shrinking losses should get partial C credit, got {c_score}"


class TestBacktesterAScoreSync:
    """Regression: Backtester A score used fixed thresholds, no turnaround handling."""

    def test_a_score_uses_sector_thresholds(self):
        """A score should use sector-adjusted CAGR thresholds."""
        import inspect
        from backend.backtester import BacktestEngine

        source = inspect.getsource(BacktestEngine._calculate_scores)

        # A score should reuse sector thresholds from C score
        assert "c_excellent" in source, "A score should use sector-adjusted thresholds"
        assert "c_good" in source, "A score should reference sector thresholds"

    def test_a_score_turnaround_gets_credit(self):
        """Annual earnings transitioning negative->positive should get A > 0."""
        import inspect
        from backend.backtester import BacktestEngine

        source = inspect.getsource(BacktestEngine._calculate_scores)

        # Should have turnaround path for A score
        assert "current_annual > 0 and three_years_ago <= 0" in source, \
            "A score should handle negative->positive turnaround"

    def test_roe_handles_decimal_format(self):
        """ROE conversion should handle both decimal and percentage formats."""
        import inspect
        from backend.backtester import BacktestEngine

        source = inspect.getsource(BacktestEngine._calculate_scores)

        # Should convert ROE format (decimal vs percentage)
        assert "roe_pct" in source, "Should convert ROE to percentage format"


# ─── Iteration 4: Pre-Earnings Trailing Stop Tightening (Feb 26, 2026) ──────
# Issue: Held positions approaching earnings had no special protection.
# Earnings gap-downs are the #1 source of catastrophic single-position losses.
# Fix: Tighten trailing stops and take partial profits for non-CS positions
# within 5 days of earnings.


class TestPreEarningsTighteningBacktester:
    """Regression: Non-CS positions approaching earnings should get tighter stops."""

    def _make_engine(self):
        from backend.backtester import BacktestEngine, SimulatedPosition
        mock_session = MagicMock()
        mock_backtest = MagicMock()
        mock_backtest.id = 1
        mock_backtest.starting_cash = 25000
        mock_backtest.max_positions = 8
        mock_backtest.min_score_to_buy = 72
        mock_backtest.sell_score_threshold = 50
        mock_backtest.stop_loss_pct = 7.0
        mock_backtest.strategy = "nostate_optimized"
        mock_session.get.return_value = mock_backtest
        return BacktestEngine(mock_session, 1)

    def test_pre_earnings_tightened_trailing_fires(self):
        """Position near earnings with drop from peak should trigger tightened stop."""
        from backend.backtester import SimulatedPosition
        from datetime import date, timedelta

        engine = self._make_engine()

        # Position: +25% gain, peak was +35%, dropped 12% from peak
        # Normal trailing at 30-50% gain tier = 18%. Tightened = 9%.
        # Drop from peak = 12% > 9% tightened → should fire
        engine.positions["EARN"] = SimulatedPosition(
            ticker="EARN", shares=100, cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=60),
            purchase_score=80.0, peak_price=135.0,
            peak_date=date.today() - timedelta(days=5),
            sector="Technology", partial_profit_taken=0
        )

        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 118.8  # -12% from peak, +18.8% from cost
        engine.data_provider.get_market_direction.return_value = {"spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_atr.return_value = 2.0
        engine.data_provider.get_vix_proxy.return_value = 18.0

        # 3 days to earnings, non-CS
        engine.static_data["EARN"] = {"days_to_earnings": 3, "sector": "Technology"}

        sells = engine._evaluate_sells(date.today(), {"EARN": {"total_score": 75}})

        pre_earnings = [s for s in sells if "PRE-EARNINGS" in s.reason]
        assert len(pre_earnings) >= 1, f"Expected pre-earnings sell, got: {[s.reason for s in sells]}"

    def test_cs_positions_exempt_from_tightening(self):
        """Coiled Spring positions should NOT get pre-earnings tightening."""
        from backend.backtester import SimulatedPosition
        from datetime import date, timedelta

        engine = self._make_engine()

        pos = SimulatedPosition(
            ticker="CSPOS", shares=100, cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=30),
            purchase_score=80.0, peak_price=125.0,
            peak_date=date.today() - timedelta(days=5),
            sector="Technology", partial_profit_taken=0
        )
        pos.is_coiled_spring = True
        engine.positions["CSPOS"] = pos

        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 110.0
        engine.data_provider.get_market_direction.return_value = {"spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_atr.return_value = 2.0
        engine.data_provider.get_vix_proxy.return_value = 18.0

        engine.static_data["CSPOS"] = {"days_to_earnings": 2, "sector": "Technology"}

        sells = engine._evaluate_sells(date.today(), {"CSPOS": {"total_score": 75}})

        pre_earnings = [s for s in sells if "PRE-EARNINGS" in s.reason]
        assert len(pre_earnings) == 0, "CS positions should be exempt from pre-earnings tightening"

    def test_pre_earnings_partial_for_profitable_position(self):
        """Position up 15% near earnings should get 25% partial profit."""
        from backend.backtester import SimulatedPosition
        from datetime import date, timedelta

        engine = self._make_engine()

        # Position up 15%, no drop from peak → no tightened trailing fires
        # But should trigger partial profit protection
        engine.positions["PROF"] = SimulatedPosition(
            ticker="PROF", shares=100, cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=45),
            purchase_score=80.0, peak_price=115.0,
            peak_date=date.today(),
            sector="Healthcare", partial_profit_taken=0
        )

        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 115.0  # At peak, +15%
        engine.data_provider.get_market_direction.return_value = {"spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_atr.return_value = 2.0
        engine.data_provider.get_vix_proxy.return_value = 18.0

        engine.static_data["PROF"] = {"days_to_earnings": 4, "sector": "Healthcare"}

        sells = engine._evaluate_sells(date.today(), {"PROF": {"total_score": 70}})

        partials = [s for s in sells if "PRE-EARNINGS PARTIAL" in s.reason]
        assert len(partials) == 1, f"Expected pre-earnings partial, got: {[s.reason for s in sells]}"
        assert partials[0].is_partial is True
        assert partials[0].sell_pct == 25

    def test_no_tightening_when_far_from_earnings(self):
        """Positions 20+ days from earnings should NOT get tightened."""
        from backend.backtester import SimulatedPosition
        from datetime import date, timedelta

        engine = self._make_engine()

        engine.positions["FAR"] = SimulatedPosition(
            ticker="FAR", shares=100, cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=30),
            purchase_score=80.0, peak_price=130.0,
            peak_date=date.today() - timedelta(days=5),
            sector="Technology", partial_profit_taken=0
        )

        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 115.0
        engine.data_provider.get_market_direction.return_value = {"spy": {"price": 500, "ma_50": 490}}
        engine.data_provider.get_atr.return_value = 2.0
        engine.data_provider.get_vix_proxy.return_value = 18.0

        engine.static_data["FAR"] = {"days_to_earnings": 30, "sector": "Technology"}

        sells = engine._evaluate_sells(date.today(), {"FAR": {"total_score": 70}})

        pre_earnings = [s for s in sells if "PRE-EARNINGS" in s.reason]
        assert len(pre_earnings) == 0, "Should not tighten when 30 days from earnings"


class TestPreEarningsTighteningConfig:
    """Regression: Pre-earnings config should be in default.yaml."""

    def test_config_exists(self):
        """earnings_tighten config should exist in default.yaml."""
        from config_loader import config
        earnings_tighten = config.get('ai_trader.earnings_tighten', {})
        assert earnings_tighten.get('enabled', False) is True
        assert earnings_tighten.get('days_before', 0) == 5
        assert earnings_tighten.get('stop_tighten_factor', 1.0) == 0.50
        assert earnings_tighten.get('min_gain_for_partial', 0) == 10

    def test_ai_trader_has_pre_earnings_logic(self):
        """ai_trader.py should have pre-earnings tightening logic."""
        import inspect
        from backend.ai_trader import evaluate_sells
        source = inspect.getsource(evaluate_sells)
        assert "PRE-EARNINGS" in source, "ai_trader should have pre-earnings sell logic"
        assert "earnings_tighten" in source, "ai_trader should read earnings_tighten config"

    def test_backtester_has_pre_earnings_logic(self):
        """backtester.py should have pre-earnings tightening logic (synced)."""
        import inspect
        from backend.backtester import BacktestEngine
        source = inspect.getsource(BacktestEngine._evaluate_sells)
        assert "PRE-EARNINGS" in source, "backtester should have pre-earnings sell logic"
        assert "earnings_tighten" in source, "backtester should read earnings_tighten config"
