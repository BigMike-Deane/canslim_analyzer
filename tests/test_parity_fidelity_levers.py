"""Tests for the two default-off parity-fidelity A/B levers (2026-06-10).

backend/backtester.py gained two profile-overridable, default-OFF levers from
the parity audit (docs/parity-audit-evaluate-sells.md D4/D5) that make the
backtester model the live trader's intraday behavior:

  - peak_from_daily_highs (D4): track peak_price from the daily HIGH instead
    of the close, mirroring live's intraday peak watermark.
  - stop_on_daily_low (D5): trigger stop-family exits when the daily LOW
    breached the level, filling at the stop level (or the open on a
    gap-through) — the pessimistic bound that brackets live's ~90min cadence.

Both must be exact no-ops when not enabled — the champion's close-only
behavior is pinned by the default-off tests. ai_trader.py intentionally has
NEITHER lever (they make the backtester more live-like, not vice versa).
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from tests.test_backtester import make_mock_db, _soft_zone_config_data


def _engine(mock_config, profile_extra=None, stop_loss_pct=10.0):
    """Engine with the standard soft-zone config and a 10% base stop."""
    from backend.backtester import BacktestEngine

    cfg = _soft_zone_config_data()
    def config_get(key, default=None):
        return cfg.get(key, default if default is not None else {})
    mock_config.get = config_get

    mock_session, _ = make_mock_db(stop_loss_pct=stop_loss_pct)
    engine = BacktestEngine(mock_session, 1)
    engine.profile['stop_loss_pct'] = stop_loss_pct
    if profile_extra:
        engine.profile.update(profile_extra)
    engine.data_provider = MagicMock()
    engine.data_provider.get_vix_proxy.return_value = 18.0
    engine.data_provider.get_atr.return_value = 2.0  # ATR stop 5% < base, so base rules
    engine.data_provider.get_market_direction.return_value = {"weighted_signal": 1.0}
    engine.cash = 20000
    return engine


def _add_position(engine, ticker, *, cost=100.0, peak=100.0, days_held=30,
                  pyramids=0, price=95.0, current_date=None):
    from backend.backtester import SimulatedPosition

    current_date = current_date or date.today()
    engine.static_data = {ticker: {"sector": "Technology"}}
    engine.positions[ticker] = SimulatedPosition(
        ticker=ticker, shares=100, cost_basis=cost,
        purchase_date=current_date - timedelta(days=days_held),
        purchase_score=75, peak_price=peak,
        peak_date=current_date - timedelta(days=5),
        sector="Technology", pyramid_count=pyramids,
    )
    engine.data_provider.get_price_on_date.return_value = price


PEAK_ON = {'peak_from_daily_highs': {'enabled': True}}


class TestPeakFromDailyHighs:
    @patch('backend.backtester.config')
    def test_default_off_tracks_close_only(self, mock_config):
        """No lever: a daily high above the peak is invisible (champion pinned)."""
        engine = _engine(mock_config)
        _add_position(engine, "HI", peak=110.0, price=105.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 104.0, "high": 120.0, "low": 103.0, "close": 105.0}

        engine._update_positions(date.today())

        assert engine.positions["HI"].peak_price == 110.0
        engine.data_provider.get_ohlc_on_date.assert_not_called()
        assert 'peak_from_high_only' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_enabled_raises_peak_from_high(self, mock_config):
        """High 120 > peak 110 > close 105: lever raises the watermark to 120."""
        engine = _engine(mock_config, PEAK_ON)
        _add_position(engine, "HI", peak=110.0, price=105.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 104.0, "high": 120.0, "low": 103.0, "close": 105.0}

        engine._update_positions(date.today())

        assert engine.positions["HI"].peak_price == 120.0
        assert engine.positions["HI"].peak_date == date.today()
        assert engine.overlay_stats.get('peak_from_high_only') == 1

    @patch('backend.backtester.config')
    def test_close_driven_update_not_counted_as_lever_effect(self, mock_config):
        """Close 105 already beats peak 100: the stat must NOT increment."""
        engine = _engine(mock_config, PEAK_ON)
        _add_position(engine, "UP", peak=100.0, price=105.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 101.0, "high": 108.0, "low": 100.0, "close": 105.0}

        engine._update_positions(date.today())

        assert engine.positions["UP"].peak_price == 108.0
        assert 'peak_from_high_only' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_missing_ohlc_falls_back_to_close(self, mock_config):
        """No bar for the date: behave exactly like the close-only baseline."""
        engine = _engine(mock_config, PEAK_ON)
        _add_position(engine, "GAP", peak=110.0, price=105.0)
        engine.data_provider.get_ohlc_on_date.return_value = None

        engine._update_positions(date.today())

        assert engine.positions["GAP"].peak_price == 110.0


LOW_ON = {'stop_on_daily_low': {'enabled': True, 'include_trailing': True}}


class TestStopOnDailyLow:
    @patch('backend.backtester.config')
    def test_default_off_ignores_intraday_breach(self, mock_config):
        """Close -5% (above 10% stop), low breached: close-only must NOT sell."""
        engine = _engine(mock_config)
        _add_position(engine, "DIP", price=95.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 98.0, "high": 99.0, "low": 88.0, "close": 95.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert sells == []
        engine.data_provider.get_ohlc_on_date.assert_not_called()
        assert 'low_only_stops' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_low_breach_fills_at_stop_level(self, mock_config):
        """Low 88 <= level 90, open above: trigger and fill AT the stop level."""
        engine = _engine(mock_config, LOW_ON)
        _add_position(engine, "DIP", price=95.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 98.0, "high": 99.0, "low": 88.0, "close": 95.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert len(sells) == 1
        assert sells[0].reason.startswith("STOP LOSS: Down 10.0%")
        assert sells[0].price == 90.0
        assert sells[0]._signal_factors["intraday_low_stop"] is True
        assert sells[0]._signal_factors["gain_pct"] == -10.0
        assert engine.overlay_stats.get('low_only_stops') == 1

    @patch('backend.backtester.config')
    def test_gap_through_open_fills_at_open(self, mock_config):
        """Open 85 already below level 90: a live stop fills at the open, not 90."""
        engine = _engine(mock_config, LOW_ON)
        _add_position(engine, "GAPDN", price=95.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 85.0, "high": 96.0, "low": 84.0, "close": 95.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert len(sells) == 1
        assert sells[0].price == 85.0
        assert sells[0]._signal_factors["gain_pct"] == -15.0

    @patch('backend.backtester.config')
    def test_close_triggered_stop_fills_at_level_not_close(self, mock_config):
        """Close 89 triggers anyway; lever models the intraday fill at 90 and
        does NOT count it as a lever-only stop."""
        engine = _engine(mock_config, LOW_ON)
        _add_position(engine, "THRU", price=89.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 98.0, "high": 99.0, "low": 88.0, "close": 89.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert len(sells) == 1
        assert sells[0].price == 90.0
        assert "intraday_low_stop" not in sells[0]._signal_factors
        assert 'low_only_stops' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_trailing_stop_triggers_on_low(self, mock_config):
        """Peak 115 (8% tier -> level 105.80): close 107 holds, low 105 trips."""
        engine = _engine(mock_config, LOW_ON)
        _add_position(engine, "TRL", peak=115.0, price=107.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 110.0, "high": 111.0, "low": 105.0, "close": 107.0}

        sells = engine._evaluate_sells(date.today(), {})

        trails = [s for s in sells if s.reason.startswith("TRAILING STOP")]
        assert len(trails) == 1
        assert trails[0].price == 115.0 * 0.92  # fill at the trail level
        assert trails[0]._signal_factors["intraday_low_stop"] is True
        assert trails[0]._signal_factors["drop_from_peak"] == 8.0
        assert engine.overlay_stats.get('low_only_trailing') == 1

    @patch('backend.backtester.config')
    def test_include_trailing_false_spares_trailing(self, mock_config):
        """Same intraday trail breach with include_trailing off: no sell."""
        engine = _engine(mock_config, {'stop_on_daily_low': {
            'enabled': True, 'include_trailing': False}})
        _add_position(engine, "TRL", peak=115.0, price=107.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 110.0, "high": 111.0, "low": 105.0, "close": 107.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert sells == []
        assert 'low_only_trailing' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_partial_trailing_on_low_resets_peak_to_fill(self, mock_config):
        """Pyramided high-scorer: 50% partial at the (pyramid-widened) trail
        level, peak reset to the fill price — not the close."""
        engine = _engine(mock_config, LOW_ON)
        # pyramids=2 widens the 8% tier to 12% -> level 115 * 0.88 = 101.20
        _add_position(engine, "CONV", peak=115.0, price=107.0, pyramids=2)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 110.0, "high": 111.0, "low": 101.0, "close": 107.0}

        sells = engine._evaluate_sells(date.today(), {"CONV": {"total_score": 70}})

        assert len(sells) == 1
        assert sells[0].is_partial is True
        assert sells[0].shares == 50.0
        assert abs(sells[0].price - 101.20) < 1e-9
        assert abs(engine.positions["CONV"].peak_price - 101.20) < 1e-9

    @patch('backend.backtester.config')
    def test_use_prior_peak_spares_same_day_whipsaw(self, mock_config):
        """Optimistic ordering: peak rose to 120 only today (prev 100), so the
        low is checked against yesterday's 100-peak and must NOT trigger."""
        engine = _engine(mock_config, {
            'stop_on_daily_low': {'enabled': True, 'include_trailing': True,
                                  'use_prior_peak': True},
            'trailing_stops': {'gain_20_to_30': 12},  # pin tier (profile varies)
        })
        # current peak 125 (e.g. D4 raised it from today's high), prev peak 100.
        # peak=120 would put peak_gain at 19.99999... (fp) — wrong tier.
        _add_position(engine, "WHIP", peak=125.0, price=113.0)
        engine.positions["WHIP"].prev_peak_price = 100.0
        # tier from current peak: peak_gain 25% -> 12% trail
        # pessimistic level: 125*0.88 = 110.0 (low 109.5 would trip)
        # optimistic level:  100*0.88 = 88.0  (low 109.5 must NOT trip)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 115.0, "high": 116.0, "low": 109.5, "close": 113.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert sells == []
        assert 'low_only_trailing' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_default_ordering_trips_same_day_whipsaw(self, mock_config):
        """Same scenario without use_prior_peak: low 109.5 <= 110.0 trips."""
        engine = _engine(mock_config, {
            **LOW_ON,
            'trailing_stops': {'gain_20_to_30': 12},  # pin tier (profile varies)
        })
        _add_position(engine, "WHIP", peak=125.0, price=113.0)
        engine.positions["WHIP"].prev_peak_price = 100.0
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 115.0, "high": 116.0, "low": 109.5, "close": 113.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert len(sells) == 1
        assert abs(sells[0].price - 125.0 * 0.88) < 1e-9

    @patch('backend.backtester.config')
    def test_use_prior_peak_unset_prev_falls_back_to_current(self, mock_config):
        """prev_peak_price never populated (0.0): fall back to the current peak
        so the knob degrades to the default ordering, not to no-trigger."""
        engine = _engine(mock_config, {'stop_on_daily_low': {
            'enabled': True, 'include_trailing': True, 'use_prior_peak': True}})
        _add_position(engine, "TRL", peak=115.0, price=107.0)
        assert engine.positions["TRL"].prev_peak_price == 0.0
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 110.0, "high": 111.0, "low": 105.0, "close": 107.0}

        sells = engine._evaluate_sells(date.today(), {})

        assert len(sells) == 1
        assert sells[0].price == 115.0 * 0.92

    @patch('backend.backtester.config')
    def test_update_positions_snapshots_prev_peak(self, mock_config):
        """_update_positions must record the pre-update peak each day."""
        engine = _engine(mock_config, PEAK_ON)
        _add_position(engine, "SNAP", peak=110.0, price=105.0)
        engine.data_provider.get_ohlc_on_date.return_value = {
            "open": 104.0, "high": 120.0, "low": 103.0, "close": 105.0}

        engine._update_positions(date.today())

        assert engine.positions["SNAP"].prev_peak_price == 110.0
        assert engine.positions["SNAP"].peak_price == 120.0

    @patch('backend.backtester.config')
    def test_missing_ohlc_falls_back_to_close_only(self, mock_config):
        """Lever on but no bar: behave exactly like the close-only baseline."""
        engine = _engine(mock_config, LOW_ON)
        _add_position(engine, "NOBAR", price=95.0)
        engine.data_provider.get_ohlc_on_date.return_value = None

        sells = engine._evaluate_sells(date.today(), {})

        assert sells == []


class TestGetOhlcOnDate:
    def _provider(self):
        import pandas as pd
        from backend.historical_data import HistoricalDataProvider

        provider = HistoricalDataProvider.__new__(HistoricalDataProvider)
        provider._price_cache = {
            "AAPL": pd.DataFrame({
                "date": [date(2024, 1, 2), date(2024, 1, 3)],
                "open": [100.0, 102.0],
                "high": [105.0, 106.0],
                "low": [99.0, 101.0],
                "close": [104.0, 103.0],
                "volume": [1e6, 1.1e6],
            })
        }
        return provider

    def test_exact_date_returns_full_bar(self):
        bar = self._provider().get_ohlc_on_date("AAPL", date(2024, 1, 3))
        assert bar == {"open": 102.0, "high": 106.0, "low": 101.0, "close": 103.0}

    def test_missing_date_returns_none_no_fallback(self):
        """No prior-date fallback: a stale high/low is wrong for watermarks."""
        assert self._provider().get_ohlc_on_date("AAPL", date(2024, 1, 4)) is None

    def test_unknown_ticker_returns_none(self):
        assert self._provider().get_ohlc_on_date("ZZZZ", date(2024, 1, 3)) is None


CLOCK_ON = {'score_crash_scan_clock': {'enabled': True, 'scans_per_day': 4}}

# Score 48 threads the gates: crash-eligible (48 < threshold 50, drop
# 75-48=27 > 20, gain -5% < ignore-if-profitable 10) without tripping the
# WEAK POSITION branch (48 > sell_score_threshold 45 in make_mock_db).
CRASH_SCORES = {"CRSH": {"total_score": 48.0}}


class TestScoreCrashScanClock:
    @patch('backend.backtester.config')
    def test_default_off_single_low_day_no_crash(self, mock_config):
        """Daily clock (champion pinned): one low day is 1 low scan — no sell,
        single append to history."""
        engine = _engine(mock_config)
        _add_position(engine, "CRSH", price=95.0)

        sells = engine._evaluate_sells(date.today(), CRASH_SCORES)

        assert sells == []
        assert engine.score_history["CRSH"] == [48.0]
        assert 'scan_clock_crash_sells' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_default_off_crash_confirms_on_second_day(self, mock_config):
        """Daily clock: the 2-of-last-3 gate confirms on the SECOND low day
        (the legacy 3-consecutive gate alone would take three)."""
        engine = _engine(mock_config)
        _add_position(engine, "CRSH", price=95.0)

        day1 = engine._evaluate_sells(date.today() - timedelta(days=1), CRASH_SCORES)
        day2 = engine._evaluate_sells(date.today(), CRASH_SCORES)

        assert day1 == []
        assert len(day2) == 1
        assert day2[0].reason.startswith("SCORE CRASH")
        assert 'scan_clock_crash_sells' not in engine.overlay_stats  # lever off

    @patch('backend.backtester.config')
    def test_enabled_single_low_day_crash_sells(self, mock_config):
        """Scan clock: one low day stands in for 4 low scans — the same
        afternoon-confirm live exhibits. Sell fires on day one."""
        engine = _engine(mock_config, CLOCK_ON)
        _add_position(engine, "CRSH", price=95.0)

        sells = engine._evaluate_sells(date.today(), CRASH_SCORES)

        assert len(sells) == 1
        assert sells[0].reason.startswith("SCORE CRASH")
        assert sells[0]._signal_factors["sell_reason"] == "SCORE CRASH"
        assert engine.overlay_stats.get('scan_clock_crash_sells') == 1

    @patch('backend.backtester.config')
    def test_enabled_history_stays_capped_at_five(self, mock_config):
        """Replication must not grow the stability window beyond 5 entries."""
        engine = _engine(mock_config, CLOCK_ON)
        _add_position(engine, "CRSH", price=95.0)

        engine._evaluate_sells(date.today() - timedelta(days=1), CRASH_SCORES)
        engine._evaluate_sells(date.today(), CRASH_SCORES)

        assert engine.score_history["CRSH"] == [48.0] * 5

    @patch('backend.backtester.config')
    def test_enabled_healthy_score_no_crash(self, mock_config):
        """The lever only accelerates the clock — it must not invent sells
        when the crash conditions (drop > 20, score < 50) don't hold."""
        engine = _engine(mock_config, CLOCK_ON)
        _add_position(engine, "OK", price=95.0)

        sells = engine._evaluate_sells(date.today(), {"OK": {"total_score": 60.0}})

        assert sells == []
        assert 'scan_clock_crash_sells' not in engine.overlay_stats


class TestLiveTraderUntouched:
    def test_ai_trader_has_no_lever(self):
        """These levers make the backtester MORE live-like; there is nothing to
        mirror into ai_trader.py. If this fails, someone copied them over —
        that must be a deliberate decision, not a drive-by sync."""
        import inspect
        import backend.ai_trader as ai_trader

        src = inspect.getsource(ai_trader)
        assert "peak_from_daily_highs" not in src
        assert "stop_on_daily_low" not in src
        assert "score_crash_scan_clock" not in src
