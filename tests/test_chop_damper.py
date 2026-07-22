"""chop_damper_multiplier — profile-gated position damper for chop days.

Default-OFF research lever evaluated via live shadow A/B (owner directive
2026-07-22: prove we beat SPY before real money; regime attribution showed
chop days near the 50MA cost −7.9pp live while trend days earned +20.4pp).
The champion profile never enables it — these tests pin that OFF means
exactly 1.0 in every case.
"""

import os

os.environ.setdefault("CANSLIM_ENV", "development")

from backend.trading_engine import chop_damper_multiplier


ON = {"chop_damper": {"enabled": True, "band_pct": 1.5, "position_multiplier": 0.5}}


class TestChopDamperMultiplier:
    def test_disabled_is_always_neutral(self):
        for spy, ma in ((750, 749), (750, 700), (700, 750), (0, 0)):
            assert chop_damper_multiplier(spy, ma, {}) == 1.0
            assert chop_damper_multiplier(
                spy, ma, {"chop_damper": {"enabled": False}}) == 1.0

    def test_chop_zone_damps(self):
        # SPY 0.7% above its 50MA — inside the 0..1.5% chop band
        assert chop_damper_multiplier(704.9, 700.0, ON) == 0.5

    def test_exactly_at_ma_damps(self):
        assert chop_damper_multiplier(700.0, 700.0, ON) == 0.5

    def test_strong_trend_is_neutral(self):
        # 2.9% above the MA — trend day, full size
        assert chop_damper_multiplier(720.0, 700.0, ON) == 1.0

    def test_below_ma_is_neutral(self):
        # Below the MA is the binary gate's job — damper must not double-punish
        assert chop_damper_multiplier(690.0, 700.0, ON) == 1.0

    def test_just_above_band_is_neutral(self):
        # Marginally past band_pct → out of the chop zone (0 <= d < band).
        # (Exact-boundary equality is float-hazardous; probe just above.)
        assert chop_damper_multiplier(700.0 * 1.0151, 700.0, ON) == 1.0

    def test_missing_market_data_fails_open(self):
        assert chop_damper_multiplier(0, 700.0, ON) == 1.0
        assert chop_damper_multiplier(700.0, 0, ON) == 1.0

    def test_custom_config_values(self):
        prof = {"chop_damper": {"enabled": True, "band_pct": 3.0,
                                "position_multiplier": 0.25}}
        assert chop_damper_multiplier(714.0, 700.0, prof) == 0.25  # 2% < 3%
        assert chop_damper_multiplier(722.0, 700.0, prof) == 1.0   # 3.1% > 3%
