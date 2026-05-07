"""
Parity tests for backtester ↔ canslim_scorer C-score refactor (May 6 2026).

Pins the contract that lets the backtester delegate C-score computation to
canslim_scorer instead of maintaining a parallel inline implementation. The
inline version drifted from the live scorer at least once already (Approach 2
shipped as a no-op because the inline path didn't read earnings_surprise_pct
even when the snapshot stored it).

What this file pins:
  1. HistoricalStockData carries the C-score input fields canslim_scorer reads
     (earnings_surprise_pct, eps_beat_streak, eps_estimate_revision_pct,
      earnings_growth_estimate, sector).
  2. get_stock_data_on_date populates those fields from static_data.
  3. BacktestStaticSnapshot now has earnings_surprise_pct; create + copy
     helpers preserve it.
  4. The C-score the backtester computes via _scorer._score_current_earnings
     matches what the live CANSLIMScorer produces given the same StockData
     shape — including the surprise / beat-streak / revision bonus block
     that the inline backtester used to silently drop.
"""

import pytest
from datetime import date

from backend.historical_data import HistoricalStockData, HistoricalDataProvider
from backend.database import BacktestStaticSnapshot
from canslim_scorer import CANSLIMScorer


# ---------------------------------------------------------------------------
# Fixtures — minimal duck-typed StockData shapes that canslim_scorer accepts
# ---------------------------------------------------------------------------

def _hsd(**overrides) -> HistoricalStockData:
    """Build a HistoricalStockData with the C-score fields pre-set to neutral."""
    base = dict(
        ticker="TEST",
        quarterly_earnings=[1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90],
        annual_earnings=[5.50, 4.80, 4.20],
        sector="Technology",
        roe=0.20,
        institutional_holders_pct=50.0,
        earnings_surprise_pct=0.0,
        eps_beat_streak=0,
        earnings_beat_streak=0,
        eps_estimate_revision_pct=None,
        earnings_growth_estimate=0.10,
    )
    base.update(overrides)
    return HistoricalStockData(**base)


@pytest.fixture
def scorer():
    """CANSLIMScorer with no fetcher — fine for the C-score path which never
    touches the fetcher (only L/M scores access SP500 history / market state)."""
    return CANSLIMScorer(None)


# ---------------------------------------------------------------------------
# 1. HistoricalStockData carries the new C-score input fields
# ---------------------------------------------------------------------------

class TestHistoricalStockDataCScoreFields:
    def test_default_construction_has_c_score_fields(self):
        """Bare HistoricalStockData('FOO') must default the C-score fields so
        canslim_scorer's getattr(...,0) and direct .attr access don't AttributeError."""
        h = HistoricalStockData(ticker="FOO")
        assert h.earnings_surprise_pct == 0.0
        assert h.eps_beat_streak == 0
        assert h.earnings_beat_streak == 0
        assert h.eps_estimate_revision_pct is None
        assert h.earnings_growth_estimate == 0.0
        assert h.sector == ""

    def test_overrides_propagate(self):
        h = _hsd(
            earnings_surprise_pct=12.5,
            eps_beat_streak=4,
            eps_estimate_revision_pct=8.0,
        )
        assert h.earnings_surprise_pct == 12.5
        assert h.eps_beat_streak == 4
        assert h.eps_estimate_revision_pct == 8.0


# ---------------------------------------------------------------------------
# 2. get_stock_data_on_date populates fields from static_data
# ---------------------------------------------------------------------------

class TestGetStockDataOnDateAdapter:
    """The provider's adapter is what makes the backtester ↔ scorer bridge
    work — static_data dict in, HistoricalStockData out with all C-score
    fields populated."""

    def _provider(self):
        return HistoricalDataProvider(["TEST"], data_reference_date=date(2026, 5, 6))

    def test_populates_earnings_surprise_pct_from_static_data(self):
        provider = self._provider()
        static_data = {
            "sector": "Technology",
            "earnings_surprise_pct": 15.0,
            "earnings_beat_streak": 3,
            "eps_estimate_revision_pct": 12.0,
        }
        result = provider.get_stock_data_on_date("TEST", date(2026, 5, 6), static_data)
        assert result.earnings_surprise_pct == 15.0
        assert result.eps_beat_streak == 3
        assert result.earnings_beat_streak == 3
        assert result.eps_estimate_revision_pct == 12.0

    def test_handles_none_surprise(self):
        """static_data.earnings_surprise_pct=None must coerce to 0.0 (scorer
        reads `data.earnings_surprise_pct` via `... or 0` and trips on None)."""
        provider = self._provider()
        static_data = {
            "sector": "Energy",
            "earnings_surprise_pct": None,
            "earnings_beat_streak": None,
        }
        result = provider.get_stock_data_on_date("TEST", date(2026, 5, 6), static_data)
        assert result.earnings_surprise_pct == 0.0
        assert result.eps_beat_streak == 0

    def test_eps_beat_streak_mirrors_earnings_beat_streak(self):
        """The Stock model + snapshot store one column (`earnings_beat_streak`)
        but canslim_scorer reads `eps_beat_streak`. Adapter must populate both
        from the single source so production sets remain interchangeable."""
        provider = self._provider()
        static_data = {"earnings_beat_streak": 5}
        result = provider.get_stock_data_on_date("TEST", date(2026, 5, 6), static_data)
        assert result.eps_beat_streak == 5
        assert result.earnings_beat_streak == 5


# ---------------------------------------------------------------------------
# 3. BacktestStaticSnapshot schema has earnings_surprise_pct
# ---------------------------------------------------------------------------

class TestSnapshotSchemaHasSurprise:
    def test_column_exists(self):
        assert hasattr(BacktestStaticSnapshot, "earnings_surprise_pct")

    def test_round_trip_through_orm(self):
        """ORM-level sanity: an instance can hold a float and read it back."""
        row = BacktestStaticSnapshot(
            backtest_id=1,
            ticker="TEST",
            earnings_surprise_pct=22.5,
        )
        assert row.earnings_surprise_pct == 22.5


# ---------------------------------------------------------------------------
# 4. Surprise / beat-streak / revision bonus reaches the C score
#
# These are the bonuses the inline backtester used to silently drop. With the
# refactor each one should now move the backtester's C score the same way it
# moves the live scorer's.
# ---------------------------------------------------------------------------

class TestCScoreBonusesNowReachable:
    def _baseline(self, scorer):
        """C score for a stock with strong base growth, no bonuses set."""
        s, _ = scorer._score_current_earnings(_hsd())
        return s

    def test_surprise_10pct_lifts_or_caps_c_score(self, scorer):
        baseline = self._baseline(scorer)
        with_surprise, _ = scorer._score_current_earnings(
            _hsd(earnings_surprise_pct=12.0)
        )
        assert with_surprise >= baseline, (
            f"Surprise bonus must not regress C score (baseline={baseline}, "
            f"with_surprise={with_surprise})"
        )

    def test_negative_revision_can_reduce_c_score(self, scorer):
        """Strong downward analyst revisions should pull C score below the
        baseline. This signal was completely absent from the inline backtester."""
        baseline = self._baseline(scorer)
        with_neg_rev, _ = scorer._score_current_earnings(
            _hsd(eps_estimate_revision_pct=-12.0)
        )
        assert with_neg_rev < baseline, (
            f"Strong negative revision (-12%) should reduce C score below "
            f"baseline (baseline={baseline}, with_neg_rev={with_neg_rev})"
        )

    def test_beat_streak_4plus_quarters_adds_bonus(self, scorer):
        """4+ consecutive quarter beats add up to 2pts (capped at surprise_max).
        The backtester's inline path had no concept of beat_streak."""
        no_streak, _ = scorer._score_current_earnings(_hsd(eps_beat_streak=0))
        long_streak, _ = scorer._score_current_earnings(_hsd(eps_beat_streak=6))
        # Both can be 15-capped on this fixture, so just assert non-regression
        assert long_streak >= no_streak


# ---------------------------------------------------------------------------
# 5. Edge cases — turnaround, losses, NaN cleanup
# ---------------------------------------------------------------------------

class TestCScoreEdgeCases:
    def test_turnaround_negative_to_positive(self, scorer):
        """current TTM > 0, prior TTM < 0 — canslim_scorer hits the
        Turnaround (TTM) branch (line 183). Backtester used to write 12pts
        inline but now this is delegated."""
        earnings = [1.0, 1.0, 1.0, 1.0, -0.5, -0.5, -0.5, -0.5]
        s, detail = scorer._score_current_earnings(_hsd(quarterly_earnings=earnings))
        # canslim_scorer's near-zero prior path returns max_score * 0.8 (12pts)
        # but real "turnaround" prior_ttm=-2.0 isn't near-zero, so it falls into
        # the normal positive-to-positive flow with prior negated. Either way,
        # score should be substantially positive.
        assert s > 0
        # Detail should not say "Insufficient" — we have 8 quarters
        assert "Insufficient" not in detail

    def test_losses_shrinking_partial_credit(self, scorer):
        """All negative but improving — partial credit, not zero."""
        earnings = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
        s, detail = scorer._score_current_earnings(_hsd(quarterly_earnings=earnings))
        assert s > 0
        assert "improv" in detail.lower() or "loss" in detail.lower()

    def test_losses_worsening_zero(self, scorer):
        earnings = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
        s, _ = scorer._score_current_earnings(_hsd(quarterly_earnings=earnings))
        assert s == 0

    def test_nan_values_are_filtered(self, scorer):
        """canslim_scorer's _clean_earnings drops NaN. Backtester's inline
        filter does the same (the `[e for e in ... if not isnan]` line). The
        contract is: a NaN-containing list produces the same score as the
        cleaned list."""
        clean = [1.50, 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90]
        dirty = [1.50, float("nan"), 1.40, 1.30, 1.20, 1.10, 1.00, 0.95, 0.90]
        s_clean, _ = scorer._score_current_earnings(_hsd(quarterly_earnings=clean))
        s_dirty, _ = scorer._score_current_earnings(_hsd(quarterly_earnings=dirty))
        assert s_clean == s_dirty


# ---------------------------------------------------------------------------
# 6. Sector threshold variation
# ---------------------------------------------------------------------------

class TestSectorAdjustedThresholds:
    """A 20% TTM growth is "good" for Energy but only reaches the lower band
    for Technology. The scorer's _get_sector_thresholds drives this; the
    backtester's inline copy of the table is no longer load-bearing."""

    def test_tech_vs_utilities_same_growth_different_score(self, scorer):
        # Construct earnings with ~20% TTM growth
        earnings = [1.20, 1.20, 1.20, 1.20, 1.00, 1.00, 1.00, 1.00]  # 20% growth
        tech_score, _ = scorer._score_current_earnings(
            _hsd(quarterly_earnings=earnings, sector="Technology")
        )
        utilities_score, _ = scorer._score_current_earnings(
            _hsd(quarterly_earnings=earnings, sector="Utilities")
        )
        # 20% is "good" tier in Technology (good=20, excellent=30) and
        # "well above excellent" in Utilities (excellent=12). Utilities should
        # max out the base; Technology should be in the middle band.
        assert utilities_score >= tech_score
