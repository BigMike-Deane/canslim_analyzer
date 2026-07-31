"""Tests for the default-off pyramid bearish-market-gate lever (2026-07-31).

backend/backtester.py::_evaluate_pyramids gained a profile-overridable,
default-OFF lever:

  - pyramid_rules.bearish_market_gate: when SPY < 50MA (the same signal that
    blocks new buys), skip ALL adds — pyramids and scale-ins. Live behavior
    is asymmetric: the regime gate blocks a fresh position while allowing
    adds to existing ones (user-3 ITIC/SN pyramids fired Jul-28/29 2026
    inside a bearish tape).

Must be an exact no-op when not enabled — champion behavior is pinned by the
default-off test. ai_trader.py intentionally does NOT have the lever
(research only; nothing ships live without a sweep + owner call).
"""

from datetime import date
from unittest.mock import patch

from tests.test_exit_redesign_levers import _engine, _add_position

GATE_ON = {'pyramid_rules': {'bearish_market_gate': True}}

BULL_SPY = {'spy': {'price': 750.0, 'ma_50': 700.0}}
BEAR_SPY = {'spy': {'price': 690.0, 'ma_50': 700.0}}


def _pyramid_ready_engine(mock_config, profile_extra=None):
    """Engine with one position that qualifies for a standard pyramid:
    +3% gain, score 75, no prior pyramids, allocation well under max."""
    engine = _engine(mock_config, profile_extra)
    _add_position(engine, "AAPL", cost=100.0, peak=103.0, days_held=10, price=103.0)
    # Shrink to ~13% allocation so MAX_POSITION_ALLOCATION doesn't skip it
    engine.positions["AAPL"].shares = 30
    return engine


SCORES = {"AAPL": {"total_score": 75.0}}


class TestPyramidBearishGate:
    @patch('backend.backtester.config')
    def test_default_off_pyramids_fire_in_bearish_tape(self, mock_config):
        """Champion behavior pinned: no lever -> pyramids fire even bearish."""
        engine = _pyramid_ready_engine(mock_config)
        engine.data_provider.get_market_direction.return_value = BEAR_SPY
        pyramids = engine._evaluate_pyramids(date.today(), SCORES)
        assert len(pyramids) == 1
        assert pyramids[0].action == "PYRAMID"

    @patch('backend.backtester.config')
    def test_gate_on_blocks_pyramids_when_spy_below_50ma(self, mock_config):
        engine = _pyramid_ready_engine(mock_config, GATE_ON)
        engine.data_provider.get_market_direction.return_value = BEAR_SPY
        assert engine._evaluate_pyramids(date.today(), SCORES) == []

    @patch('backend.backtester.config')
    def test_gate_on_allows_pyramids_when_spy_above_50ma(self, mock_config):
        engine = _pyramid_ready_engine(mock_config, GATE_ON)
        engine.data_provider.get_market_direction.return_value = BULL_SPY
        pyramids = engine._evaluate_pyramids(date.today(), SCORES)
        assert len(pyramids) == 1

    @patch('backend.backtester.config')
    def test_gate_on_missing_spy_data_counts_as_bearish(self, mock_config):
        """Mirrors the buy gate's conservatism: no SPY data -> no adds."""
        engine = _pyramid_ready_engine(mock_config, GATE_ON)
        engine.data_provider.get_market_direction.return_value = {}
        assert engine._evaluate_pyramids(date.today(), SCORES) == []

    def test_ai_trader_does_not_have_the_lever(self):
        """Research lever is backtester-only until a sweep survives."""
        from pathlib import Path
        ai_source = (Path(__file__).parent.parent / "backend" / "ai_trader.py").read_text()
        assert "bearish_market_gate" not in ai_source
