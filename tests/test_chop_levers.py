"""Chop levers (2026-08-25 investigation): chop_entry_bar_blocks and
chop_trim_pct — profile-gated, default OFF, shadow-A/B-only.

Both share the chop-band definition with chop_damper_multiplier: SPY
sitting 0..band% above its 50MA. Below-MA days are the binary gate's job.
"""
from backend.trading_engine import chop_entry_bar_blocks, chop_trim_pct

BAR_ON = {'chop_entry_bar': {'enabled': True, 'band_pct': 1.5, 'require_breakout': True}}
TRIM_ON = {'chop_trim': {'enabled': True, 'band_pct': 1.5, 'min_ext_pct': 25, 'trim_pct': 30}}

# SPY 700 with 50MA 700 -> dist 0.7% (chop band); 715 -> +2.1% (trend)
CHOP_SPY, TREND_SPY, MA = 704.9, 715.0, 700.0


class TestChopEntryBar:
    def test_off_by_default(self):
        assert chop_entry_bar_blocks(CHOP_SPY, MA, {}, False) is False
        assert chop_entry_bar_blocks(CHOP_SPY, MA, {'chop_entry_bar': {}}, False) is False

    def test_blocks_unconfirmed_entry_in_chop_band(self):
        assert chop_entry_bar_blocks(CHOP_SPY, MA, BAR_ON, is_breaking_out=False) is True

    def test_confirmed_breakout_passes_in_chop_band(self):
        assert chop_entry_bar_blocks(CHOP_SPY, MA, BAR_ON, is_breaking_out=True) is False

    def test_trend_day_is_champion_identical(self):
        assert chop_entry_bar_blocks(TREND_SPY, MA, BAR_ON, is_breaking_out=False) is False

    def test_below_ma_is_not_chop(self):
        # Below the 50MA the binary SPY gate owns behavior — no extra block.
        assert chop_entry_bar_blocks(690.0, MA, BAR_ON, is_breaking_out=False) is False

    def test_missing_market_data_fails_open(self):
        assert chop_entry_bar_blocks(0, MA, BAR_ON, False) is False
        assert chop_entry_bar_blocks(CHOP_SPY, 0, BAR_ON, False) is False
        assert chop_entry_bar_blocks(None, None, BAR_ON, False) is False


class TestChopTrim:
    def test_off_by_default(self):
        assert chop_trim_pct(CHOP_SPY, MA, {}, 40.0) is None

    def test_trims_extended_holding_in_chop_band(self):
        assert chop_trim_pct(CHOP_SPY, MA, TRIM_ON, 30.0) == 30.0

    def test_min_extension_respected(self):
        assert chop_trim_pct(CHOP_SPY, MA, TRIM_ON, 24.9) is None
        assert chop_trim_pct(CHOP_SPY, MA, TRIM_ON, 25.0) == 30.0

    def test_trend_day_never_trims(self):
        assert chop_trim_pct(TREND_SPY, MA, TRIM_ON, 60.0) is None

    def test_below_ma_never_trims(self):
        assert chop_trim_pct(690.0, MA, TRIM_ON, 60.0) is None

    def test_missing_data_fails_open(self):
        assert chop_trim_pct(CHOP_SPY, MA, TRIM_ON, None) is None
        assert chop_trim_pct(CHOP_SPY, MA, TRIM_ON, float('nan')) is None
        assert chop_trim_pct(0, MA, TRIM_ON, 60.0) is None

    def test_custom_trim_pct(self):
        cfg = {'chop_trim': {'enabled': True, 'band_pct': 1.5,
                             'min_ext_pct': 20, 'trim_pct': 50}}
        assert chop_trim_pct(CHOP_SPY, MA, cfg, 22.0) == 50.0
