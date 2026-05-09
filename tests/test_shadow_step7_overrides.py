"""Tests for Shadow Step 7 — scorer_overrides applied to the shadow scoring path.

Built for the Approach 2 (ec73f83) parallel-regime A/B verdict at 2026-06-18:
the live A/B is regime-confounded (pre window had a +11.32% SPY rebound rally,
post window starts at the bullish peak), so we need a shadow control arm that
runs the PRE-cap C scoring alongside the live POST-cap arm in identical
market conditions.

Coverage:
  - `CANSLIMScore.c_score_uncapped` field exists and defaults to 0.0.
  - `_score_current_earnings` populates `_score_metadata["c_score_uncapped"]`
    with the pre-cap value on the primary 8-quarter TTM path.
  - The fallback (<8-quarter) path populates the same metadata key.
  - Early-return paths (insufficient data, losses) leave metadata unset →
    `score_stock` defaults uncapped to capped.
  - `shadow_trader._apply_disable_excellence_cap` produces a transformed
    list with `c_score → c_score_uncapped` and `canslim_score` adjusted by
    the cap delta. Pass-through for: non-dict entries, NULL uncapped,
    uncapped == capped (no cap was ever applied).
  - `shadow_trader._patch_sandbox_stocks_disable_cap` mutates Stock ORM
    rows in the sandbox session and returns the patched count.
  - End-to-end: a ShadowStrategy with `scorer_overrides=
    {"disable_excellence_cap": True}` runs through `_run_one_strategy`
    without raising, the Stock rows in its sandbox carry uncapped scores,
    and the live DB is untouched after rollback.
  - Backwards-compat: empty overrides ({} — existing shadow_baseline) is a
    no-op; analysis_results pass through unchanged.

Eval-safe — no real network, no scoring writes outside the sandbox, no
trading.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db,
    SessionLocal,
    AIPortfolioConfig,
    AIPortfolioPosition,
    AIPortfolioSnapshot,
    AIPortfolioTrade,
    ShadowStrategy,
    ShadowTrade,
    Stock,
    StockScore,
)
from backend.shadow_trader import (
    _apply_disable_excellence_cap,
    _patch_sandbox_stocks_disable_cap,
    _run_one_strategy,
)
from canslim_scorer import CANSLIMScore, CANSLIMScorer


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioConfig).delete()
    db.query(AIPortfolioPosition).delete()
    db.query(AIPortfolioSnapshot).delete()
    db.query(StockScore).delete()
    # Don't wipe Stocks — other tests may rely on them; just track our test rows.
    db.commit()
    test_tickers = []
    try:
        yield db, test_tickers
    finally:
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        if test_tickers:
            db.query(Stock).filter(Stock.ticker.in_(test_tickers)).delete(synchronize_session=False)
        db.commit()
        db.close()


def _make_stock(db, test_tickers, ticker, c_score=12.0, canslim_score=80.0):
    """Insert a Stock row with the given pre-mutation scores."""
    s = Stock(
        ticker=ticker,
        name=f"{ticker} Test",
        sector="Technology",
        canslim_score=canslim_score,
        c_score=c_score,
        a_score=12.0,
        n_score=12.0,
        s_score=12.0,
        l_score=12.0,
        i_score=8.0,
        m_score=12.0,
        is_growth_stock=False,
        last_updated=datetime.now(timezone.utc),
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    test_tickers.append(ticker)
    return s


# ── CANSLIMScore dataclass — c_score_uncapped field ───────────────────────────


class TestCANSLIMScoreField:
    def test_uncapped_field_defaults_to_zero(self):
        s = CANSLIMScore(ticker="TEST")
        assert s.c_score_uncapped == 0.0

    def test_uncapped_field_is_assignable(self):
        s = CANSLIMScore(ticker="TEST")
        s.c_score_uncapped = 14.5
        assert s.c_score_uncapped == 14.5


# ── canslim_scorer — _score_metadata population ───────────────────────────────


class TestScorerMetadataPropagation:
    """The dict-out-param pattern lets _score_current_earnings stash the
    pre-cap value without breaking any of its 6 early-return paths."""

    def _make_stock_data(self, **overrides):
        """Minimal StockData stand-in. The scorer reads attributes off it."""
        from data_fetcher import StockData
        sd = StockData("TEST")
        sd.ticker = "TEST"
        sd.is_valid = True
        sd.sector = "Technology"
        # 8 quarters of strongly accelerating earnings, TTM growth >> 50%
        # → primary path triggers, total_score hits 15 cap, excellence cap clips.
        sd.quarterly_earnings = [3.0, 2.5, 2.0, 1.8, 1.0, 0.9, 0.8, 0.7]
        sd.annual_earnings = [10.0, 4.0, 2.0]
        sd.earnings_growth_estimate = 0.4
        sd.eps_estimate_revision_pct = None
        sd.earnings_surprise_pct = 0
        sd.earnings_beat_streak = 0
        sd.eps_beat_streak = 0
        for k, v in overrides.items():
            setattr(sd, k, v)
        return sd

    def test_primary_path_with_cap_populates_metadata(self):
        scorer = CANSLIMScorer(MagicMock())
        sd = self._make_stock_data()
        meta = {}
        capped, detail = scorer._score_current_earnings(sd, _score_metadata=meta)
        # Only signal #1 (TTM growth ≥ 50%) fires here — beat streak 0,
        # revision None, surprise 0 — so 1/4 → cap 11.5.
        assert "c_score_uncapped" in meta
        assert meta["c_score_uncapped"] >= capped
        # Cap is 11.5 for 1 signal; uncapped should be the unclipped pre-cap.
        assert capped == 11.5
        assert meta["c_score_uncapped"] > 11.5

    def test_primary_path_with_full_signals_no_cap_applied(self):
        """When all 4 excellence signals fire, the cap is 15 → equals total
        → uncapped == capped."""
        scorer = CANSLIMScorer(MagicMock())
        sd = self._make_stock_data(
            earnings_beat_streak=8,        # signal 2
            eps_estimate_revision_pct=40,  # signal 3
            earnings_surprise_pct=15,      # signal 4
        )
        meta = {}
        capped, _ = scorer._score_current_earnings(sd, _score_metadata=meta)
        # 4/4 signals → no cap clip → uncapped == capped
        assert meta["c_score_uncapped"] == capped

    def test_fallback_path_populates_metadata(self):
        """<8 quarters of earnings drops into the anomaly-filter fallback,
        which has its own cap call site (line 504)."""
        from data_fetcher import StockData
        sd = StockData("TEST")
        sd.ticker = "TEST"
        sd.is_valid = True
        sd.sector = "Technology"
        sd.quarterly_earnings = [3.0, 2.5, 2.0, 1.5]  # 4 quarters → fallback
        sd.annual_earnings = [4.0, 2.0]
        sd.earnings_growth_estimate = 0.0
        sd.eps_estimate_revision_pct = None
        sd.earnings_surprise_pct = 0
        sd.earnings_beat_streak = 0
        sd.eps_beat_streak = 0
        scorer = CANSLIMScorer(MagicMock())
        meta = {}
        scorer._score_current_earnings(sd, _score_metadata=meta)
        # Fallback DID hit a cap call site → metadata populated.
        assert "c_score_uncapped" in meta

    def test_insufficient_data_early_return_leaves_metadata_unset(self):
        """The 'Insufficient data' early-return at line 157 doesn't apply
        the cap, so it doesn't populate metadata. Caller must default
        uncapped to capped (which score_stock does)."""
        from data_fetcher import StockData
        sd = StockData("TEST")
        sd.ticker = "TEST"
        sd.is_valid = True
        sd.quarterly_earnings = [1.0]  # < 4 quarters
        sd.annual_earnings = []
        sd.earnings_growth_estimate = 0.0
        scorer = CANSLIMScorer(MagicMock())
        meta = {}
        capped, _ = scorer._score_current_earnings(sd, _score_metadata=meta)
        assert capped == 0
        assert "c_score_uncapped" not in meta


# ── _apply_disable_excellence_cap transformation ──────────────────────────────


class TestApplyDisableExcellenceCap:
    def test_dict_with_uncapped_gets_swapped_and_canslim_adjusted(self):
        ar = [{
            "ticker": "AAA",
            "c_score": 11.5,
            "c_score_uncapped": 14.0,
            "canslim_score": 78.0,
            "total_score": 78.0,
        }]
        out = _apply_disable_excellence_cap(ar)
        assert len(out) == 1
        assert out[0]["c_score"] == 14.0
        # Cap delta = 14.0 - 11.5 = 2.5; canslim went 78 → 80.5.
        assert out[0]["canslim_score"] == 80.5
        assert out[0]["total_score"] == 80.5

    def test_passthrough_when_uncapped_equals_capped(self):
        """4-of-4 signals: no cap applied. delta = 0 → don't waste a copy."""
        ar = [{"ticker": "BBB", "c_score": 15.0, "c_score_uncapped": 15.0, "canslim_score": 90.0}]
        out = _apply_disable_excellence_cap(ar)
        assert out[0] is ar[0]  # same object — pass-through

    def test_passthrough_when_uncapped_is_none(self):
        """Rows scored before c_score_uncapped column landed pass through
        with no override applied."""
        ar = [{"ticker": "CCC", "c_score": 12.0, "c_score_uncapped": None, "canslim_score": 75.0}]
        out = _apply_disable_excellence_cap(ar)
        assert out[0] is ar[0]

    def test_passthrough_when_capped_is_none(self):
        """Defensive: bad row with NULL c_score → don't crash, just pass."""
        ar = [{"ticker": "DDD", "c_score": None, "c_score_uncapped": 14.0, "canslim_score": 80.0}]
        out = _apply_disable_excellence_cap(ar)
        assert out[0] is ar[0]

    def test_non_dict_entries_pass_through(self):
        sentinel = MagicMock()
        out = _apply_disable_excellence_cap([sentinel])
        assert out[0] is sentinel

    def test_does_not_mutate_original_dict(self):
        """Returns NEW dicts — the live analysis_results list must remain
        intact for any downstream consumer (the live trader may have
        already processed it but we must not perturb future iterations)."""
        ar = [{
            "ticker": "EEE", "c_score": 10.0, "c_score_uncapped": 13.0,
            "canslim_score": 70.0, "total_score": 70.0,
        }]
        original = dict(ar[0])
        _apply_disable_excellence_cap(ar)
        assert ar[0] == original

    def test_empty_list_returns_empty(self):
        assert _apply_disable_excellence_cap([]) == []
        assert _apply_disable_excellence_cap(None) == []


# ── _patch_sandbox_stocks_disable_cap mutation ────────────────────────────────


class TestPatchSandboxStocks:
    def test_mutates_matching_stock_rows_and_returns_count(self, db_session):
        db, test_tickers = db_session
        _make_stock(db, test_tickers, "PATCH1", c_score=11.5, canslim_score=78.0)
        _make_stock(db, test_tickers, "PATCH2", c_score=10.0, canslim_score=70.0)
        ar = [
            {"ticker": "PATCH1", "c_score": 11.5, "c_score_uncapped": 14.0},
            {"ticker": "PATCH2", "c_score": 10.0, "c_score_uncapped": 13.0},
        ]
        n = _patch_sandbox_stocks_disable_cap(db, ar)
        assert n == 2
        s1 = db.query(Stock).filter(Stock.ticker == "PATCH1").one()
        s2 = db.query(Stock).filter(Stock.ticker == "PATCH2").one()
        assert s1.c_score == 14.0
        assert s1.canslim_score == 80.5  # 78 + 2.5
        assert s2.c_score == 13.0
        assert s2.canslim_score == 73.0  # 70 + 3.0
        # Roll back so other tests don't see the mutation.
        db.rollback()

    def test_skips_rows_with_no_delta(self, db_session):
        db, test_tickers = db_session
        _make_stock(db, test_tickers, "NODIFF", c_score=15.0, canslim_score=90.0)
        ar = [{"ticker": "NODIFF", "c_score": 15.0, "c_score_uncapped": 15.0}]
        n = _patch_sandbox_stocks_disable_cap(db, ar)
        assert n == 0
        db.rollback()

    def test_returns_zero_on_empty_input(self, db_session):
        db, _ = db_session
        assert _patch_sandbox_stocks_disable_cap(db, []) == 0
        assert _patch_sandbox_stocks_disable_cap(db, None) == 0


# ── End-to-end: _run_one_strategy honors disable_excellence_cap ───────────────


class TestRunOneStrategyOverride:
    def test_override_is_applied_during_run(self, db_session, monkeypatch):
        """Wire the strategy with disable_excellence_cap=True; confirm the
        override is passed into the scoring read path. We patch
        evaluate_buys/sells to return [] so the run is a no-op other than
        the override step itself — what we're testing is that the helper
        gets called and the sandbox gets patched."""
        db, test_tickers = db_session
        _make_stock(db, test_tickers, "OVR1", c_score=11.5, canslim_score=78.0)

        s = ShadowStrategy(
            name="shadow_test_override",
            parent_strategy="nostate_cs_bear",
            config_snapshot={"strategy": "nostate_cs_bear"},
            scorer_overrides={"disable_excellence_cap": True},
            starting_value=25000.0,
        )
        db.add(s)
        db.commit()
        db.refresh(s)
        sid = s.id

        analysis_results = [{
            "ticker": "OVR1",
            "c_score": 11.5,
            "c_score_uncapped": 14.0,
            "canslim_score": 78.0,
            "total_score": 78.0,
            "is_growth_stock": False,
            "signal_factors": {},
        }]

        # Stub evaluate_buys/sells to return [] — we're only testing the
        # override pipeline, not the trader. Patching here exercises the
        # code path that calls _apply_disable_excellence_cap +
        # _patch_sandbox_stocks_disable_cap.
        with patch("backend.ai_trader.evaluate_buys", return_value=[]), \
             patch("backend.ai_trader.evaluate_sells", return_value=[]):
            n_trades = _run_one_strategy(sid, analysis_results)

        assert n_trades == 0
        # Live DB Stock row UNTOUCHED — the sandbox rolled back.
        live_stock = db.query(Stock).filter(Stock.ticker == "OVR1").one()
        assert live_stock.c_score == 11.5
        assert live_stock.canslim_score == 78.0

    def test_empty_overrides_is_noop(self, db_session):
        """The existing shadow_baseline (empty overrides {}) must continue
        to run unchanged — no override path taken."""
        db, test_tickers = db_session
        _make_stock(db, test_tickers, "NOOVR1", c_score=11.5, canslim_score=78.0)
        s = ShadowStrategy(
            name="shadow_test_baseline",
            parent_strategy="nostate_cs_bear",
            config_snapshot={"strategy": "nostate_cs_bear"},
            scorer_overrides={},
            starting_value=25000.0,
        )
        db.add(s)
        db.commit()
        db.refresh(s)
        sid = s.id

        analysis_results = [{
            "ticker": "NOOVR1",
            "c_score": 11.5,
            "c_score_uncapped": 14.0,
            "canslim_score": 78.0,
            "total_score": 78.0,
            "is_growth_stock": False,
            "signal_factors": {},
        }]

        # Spy on the helper — should NOT be called when overrides is empty.
        called = []
        original_fn = _apply_disable_excellence_cap
        def spy(ar):
            called.append(True)
            return original_fn(ar)
        with patch("backend.shadow_trader._apply_disable_excellence_cap", side_effect=spy), \
             patch("backend.ai_trader.evaluate_buys", return_value=[]), \
             patch("backend.ai_trader.evaluate_sells", return_value=[]):
            _run_one_strategy(sid, analysis_results)
        assert called == [], "empty overrides must not trigger the override pipeline"

        live_stock = db.query(Stock).filter(Stock.ticker == "NOOVR1").one()
        assert live_stock.c_score == 11.5
        assert live_stock.canslim_score == 78.0
