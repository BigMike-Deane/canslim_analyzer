"""Tests for the two default-off exit-redesign A/B levers (2026-06-09).

backend/backtester.py::_evaluate_sells gained two profile-overridable,
default-OFF levers for the post-June-18 exit redesign sweeps:

  - aging_loser_guard: clamp the ATR-widened stop for stale positions that
    never showed a real gain (peak < max_peak_gain after min_holding_days).
  - profit_giveback_floor: once peak gain arms the floor, exit if the
    position round-trips to/below floor_gain_pct.

Both must be exact no-ops when not enabled — the champion's behavior is
pinned by the default-off tests. ai_trader.py intentionally has NEITHER
lever (research only; nothing ships live until a sweep survives).
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


def _add_position(engine, ticker, *, cost=100.0, peak=100.0, days_held=60,
                  pyramids=0, price=91.0, current_date=None):
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


AGING_ON = {'aging_loser_guard': {
    'enabled': True, 'min_holding_days': 45, 'max_peak_gain': 5.0,
    'clamp_stop_pct': 8.0, 'skip_if_pyramided': True,
}}


class TestAgingLoserGuard:
    @patch('backend.backtester.config')
    def test_clamps_stop_for_stale_never_worked_position(self, mock_config):
        """Held 60d, never peaked +5%, down 9%: clamp (8%) fires where base (10%) wouldn't."""
        engine = _engine(mock_config, AGING_ON)
        _add_position(engine, "STALE", peak=102.0, days_held=60, price=91.0)

        sells = engine._evaluate_sells(date.today(), {})

        stops = [s for s in sells if s.reason.startswith("STOP LOSS")]
        assert len(stops) == 1
        assert stops[0]._signal_factors.get("aging_loser") is True
        assert engine.overlay_stats.get('aging_loser_stops') == 1

    @patch('backend.backtester.config')
    def test_default_off_is_noop(self, mock_config):
        """No profile key, no YAML key: -9% stale position must NOT be stopped (champion pinned)."""
        engine = _engine(mock_config)
        _add_position(engine, "STALE", peak=102.0, days_held=60, price=91.0)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s.reason.startswith("STOP LOSS")]
        assert 'aging_loser_stops' not in engine.overlay_stats

    @patch('backend.backtester.config')
    def test_position_that_peaked_is_spared(self, mock_config):
        """Peak gain past max_peak_gain means the thesis worked once — no clamp."""
        engine = _engine(mock_config, AGING_ON)
        _add_position(engine, "PEAKED", peak=106.0, days_held=60, price=91.0)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s._signal_factors.get("aging_loser")]
        assert engine.overlay_stats.get('aging_loser_stops', 0) == 0

    @patch('backend.backtester.config')
    def test_young_position_is_spared(self, mock_config):
        """Held under min_holding_days: outside the aging window, no clamp."""
        engine = _engine(mock_config, AGING_ON)
        _add_position(engine, "YOUNG", peak=100.0, days_held=30, price=91.0)

        sells = engine._evaluate_sells(date.today(), {})

        assert sells == []

    @patch('backend.backtester.config')
    def test_pyramided_position_is_spared(self, mock_config):
        engine = _engine(mock_config, AGING_ON)
        _add_position(engine, "PYR", peak=100.0, days_held=60, price=91.0, pyramids=1)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s.reason.startswith("STOP LOSS")]


FLOOR_ON = {'profit_giveback_floor': {
    'enabled': True, 'arm_at_peak_gain': 12.0, 'floor_gain_pct': 1.0,
    'skip_if_pyramided': False,
}}


class TestProfitGivebackFloor:
    @patch('backend.backtester.config')
    def test_armed_winner_round_tripping_exits_at_floor(self, mock_config):
        """Peaked +15%, back to +0.5%: floor exit fires before the trailing tier."""
        engine = _engine(mock_config, FLOOR_ON)
        _add_position(engine, "RT", peak=115.0, days_held=30, price=100.5)

        sells = engine._evaluate_sells(date.today(), {})

        floor = [s for s in sells if s.reason.startswith("GIVEBACK FLOOR")]
        assert len(floor) == 1
        assert floor[0].shares == 100  # full sell
        assert floor[0]._signal_factors["sell_reason"] == "GIVEBACK FLOOR"
        assert floor[0]._signal_factors["peak_gain_pct"] == 15.0
        assert engine.overlay_stats.get('giveback_floor_exits') == 1

    @patch('backend.backtester.config')
    def test_default_off_is_noop(self, mock_config):
        """Same round trip with the lever off: only the normal trailing stop fires."""
        engine = _engine(mock_config)
        _add_position(engine, "RT", peak=115.0, days_held=30, price=100.5)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s.reason.startswith("GIVEBACK FLOOR")]
        assert 'giveback_floor_exits' not in engine.overlay_stats
        # champion behavior: the 10-20 tier trailing stop handles it
        assert [s for s in sells if s.reason.startswith("TRAILING STOP")]

    @patch('backend.backtester.config')
    def test_unarmed_peak_does_not_floor(self, mock_config):
        """Peak below arm_at_peak_gain never arms the floor."""
        engine = _engine(mock_config, FLOOR_ON)
        _add_position(engine, "SMALLPK", peak=108.0, days_held=30, price=100.5)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s.reason.startswith("GIVEBACK FLOOR")]

    @patch('backend.backtester.config')
    def test_gain_above_floor_keeps_riding(self, mock_config):
        """Armed but still up +5%: floor not hit, normal trailing logic governs."""
        engine = _engine(mock_config, FLOOR_ON)
        _add_position(engine, "RIDING", peak=115.0, days_held=30, price=105.0)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s.reason.startswith("GIVEBACK FLOOR")]

    @patch('backend.backtester.config')
    def test_skip_if_pyramided_spares_conviction_position(self, mock_config):
        engine = _engine(mock_config, {'profit_giveback_floor': {
            'enabled': True, 'arm_at_peak_gain': 12.0, 'floor_gain_pct': 1.0,
            'skip_if_pyramided': True,
        }})
        _add_position(engine, "CONV", peak=115.0, days_held=30, price=100.5, pyramids=2)

        sells = engine._evaluate_sells(date.today(), {})

        assert not [s for s in sells if s.reason.startswith("GIVEBACK FLOOR")]


class TestLiveTraderUntouched:
    def test_ai_trader_has_neither_lever(self):
        """Research levers live ONLY in the backtester until a sweep survives.
        If this fails, someone mirrored them into ai_trader.py — that must be
        a deliberate post-sweep, post-freeze decision, not a drive-by sync."""
        import inspect
        import backend.ai_trader as ai_trader

        src = inspect.getsource(ai_trader)
        assert "aging_loser_guard" not in src
        assert "profit_giveback_floor" not in src
