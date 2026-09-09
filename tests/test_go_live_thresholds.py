"""
Pre-registered go-live thresholds (2026-09-09).

The standing rule was "no real money until edge vs SPY is statistically
proven". At the measured effect size (d=0.054) that needs ~2,745 trading
days -- about 10.5 years -- so it was replaced with an explicit
multi-criteria threshold set, registered BEFORE the data that would tempt
anyone to relax it.

These tests exist to make the thresholds tamper-evident. Every one of them
fails loudly if a criterion is quietly loosened, and the "all five" test
fails if a criterion is dropped entirely.
"""

import os
import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import SessionLocal, init_db
from backend.routes import admin as A

init_db()

PASSING_EDGE = {
    # one-sided p = 0.02 (t>0, two-sided 0.04) -> clears 85% one-sided
    "alpha_significance": {"t_stat": 2.1, "p_value": 0.04},
    "regime_edge": {"trend": {"n_days": 70}, "chop": {"n_days": 30}},  # 30% chop
    "regime_mix": {"blended_daily_excess_bps": 4.2},
    "excess_return_pct": 20.0,
    "max_drawdown_pct": -18.0,
    "spy_max_drawdown_pct": -15.0,      # gap 3pp, inside the 5pp margin
    "closed_trades": 60,
}
NOISE_FLOOR = 8.13


@pytest.fixture
def db():
    s = SessionLocal()
    yield s
    s.close()


def _gate(db, edge=None, floor=NOISE_FLOOR, slips=None):
    """Evaluate with stop-slippage stubbed, so the pure criteria are testable
    without seeding trades."""
    from unittest.mock import patch
    edge = {**PASSING_EDGE, **(edge or {})}

    class _Q:
        def filter(self, *a, **k): return self
        def all(self): return slips if slips is not None else _default_slips()

    def _default_slips():
        class T:
            signal_factors = {"slippage_pp": 0.7}
        return [T(), T()]

    with patch.object(db, "query", return_value=_Q()):
        return A._go_live_gate(db, edge, floor)


class TestAllCriteriaMustHold:

    def test_passing_book_meets_all_five(self, db):
        g = _gate(db)
        assert g["n_total"] == 5, "a criterion was added or removed"
        assert g["all_met"] is True, g["blocking"]
        assert g["blocking"] == []

    def test_registration_date_and_confidence_are_pinned(self, db):
        g = _gate(db)
        assert g["registered_on"] == "2026-09-09"
        # 85% one-sided was the deliberate choice; 95% two-sided costs a
        # decade at this effect size.
        assert g["confidence_one_sided"] == 0.85


class TestEachCriterionBlocks:

    def test_weak_significance_blocks(self, db):
        # one-sided p = 0.295 -> above the 0.15 bar
        g = _gate(db, {"alpha_significance": {"t_stat": 0.54, "p_value": 0.589}})
        assert "blended_edge" in g["blocking"]

    def test_negative_t_stat_does_not_sneak_through(self, db):
        # A LOSING book with a tiny two-sided p must not read as significant:
        # one-sided p = 1 - 0.04/2 = 0.98.
        g = _gate(db, {"alpha_significance": {"t_stat": -2.1, "p_value": 0.04}})
        assert "blended_edge" in g["blocking"]

    def test_trend_only_window_blocks(self, db):
        # 10% chop is not a mixed-regime window; the trend-day edge alone is
        # exactly what this criterion exists to refuse.
        g = _gate(db, {"regime_edge": {"trend": {"n_days": 90},
                                       "chop": {"n_days": 10}}})
        assert "blended_edge" in g["blocking"]

    def test_edge_below_the_noise_floor_blocks(self, db):
        # 6pp of excess against an 8.13pp path-noise floor is not evidence.
        g = _gate(db, {"excess_return_pct": 6.0})
        assert "clears_noise_floor" in g["blocking"]

    def test_worse_drawdown_than_spy_blocks(self, db):
        # Beating SPY on return while doubling its drawdown is not a win.
        g = _gate(db, {"max_drawdown_pct": -32.0, "spy_max_drawdown_pct": -15.0})
        assert "drawdown" in g["blocking"]

    def test_high_stop_slippage_blocks(self, db):
        class T:
            signal_factors = {"slippage_pp": 3.0}
        g = _gate(db, slips=[T(), T()])
        assert "stop_slippage" in g["blocking"]

    def test_unmeasured_slippage_blocks(self, db):
        # No instrumented stops at all => cannot assert execution quality.
        # Absence of evidence must not read as evidence of safety.
        g = _gate(db, slips=[])
        assert "stop_slippage" in g["blocking"]

    def test_too_few_closed_trades_blocks(self, db):
        g = _gate(db, {"closed_trades": 49})
        assert "sample" in g["blocking"]

    def test_missing_noise_floor_blocks(self, db):
        # If the vintage clock cannot produce a floor, criterion 2 is
        # unevaluable and must NOT pass by default.
        g = _gate(db, floor=None)
        assert "clears_noise_floor" in g["blocking"]


class TestThresholdsAreNotSilentlyLoosened:
    """Tamper-evidence. Changing any of these is a deliberate act."""

    def test_pinned_constants(self):
        assert A.GO_LIVE_CONFIDENCE == 0.85
        assert A.GO_LIVE_MIN_CHOP_SHARE == 30.0
        assert A.GO_LIVE_MAX_DD_MARGIN_PP == 5.0
        assert A.GO_LIVE_MAX_SLIPPAGE_PP == 1.5
        assert A.GO_LIVE_MIN_CLOSED_TRADES == 50


class TestGateInputCarriesRegimeData:
    """criterion 1 needs regime_edge/regime_mix, which compute_edge_metrics
    does NOT produce -- the /edge endpoint layers them on from SPY's distance
    to its 50MA.

    Without them the gate sees chop_share=None and criterion 1 can never
    pass, no matter how good the edge gets. It fails CLOSED (safe), but it
    would be blocked by a missing input rather than a real shortfall. Caught
    in prod verification minutes after the gate first deployed, 2026-09-09.
    """

    def test_owner_edge_metrics_includes_regime_keys(self):
        from datetime import date, datetime, timedelta, timezone
        from backend.database import (
            SessionLocal, User, AIPortfolioConfig, AIPortfolioSnapshot,
            MarketSnapshot,
        )

        db = SessionLocal()
        try:
            db.query(AIPortfolioSnapshot).filter_by(user_id=1).delete()
            db.query(AIPortfolioConfig).filter_by(user_id=1).delete()
            if not db.query(User).filter_by(id=1).first():
                db.add(User(id=1, email="owner@acct.example.org",
                            display_name="Owner", is_active=True,
                            is_admin=True, hashed_password=""))
            db.add(AIPortfolioConfig(user_id=1, starting_cash=25000.0,
                                     current_cash=25000.0, is_active=True))
            # 40 days of book + SPY WITH a 50MA, so regime classification has
            # something to classify.
            #
            # Deliberately placed ~400 days back. MarketSnapshot is shared
            # app-wide and keyed by date; seeding up to TODAY makes these rows
            # the "latest market snapshot", which feeds market direction and
            # broke three unrelated breakout/stock tests when this was first
            # written. Old dates cannot become the latest row.
            for i in range(40):
                day = date.today() - timedelta(days=440 - i)
                db.add(AIPortfolioSnapshot(
                    user_id=1,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=40 - i),
                    date=day, total_value=25000.0 + i * 25,
                    cash=1000.0, positions_value=24000.0 + i * 25,
                    positions_count=1, total_return=i * 25.0,
                    total_return_pct=i * 0.1,
                ))
                row = db.query(MarketSnapshot).filter_by(date=day).first()
                price, ma = 500.0 + i, 495.0 + i * 0.9
                if row:
                    row.spy_price, row.spy_50_ma = price, ma
                else:
                    db.add(MarketSnapshot(date=day, spy_price=price, spy_50_ma=ma))
            db.commit()

            m = A._owner_edge_metrics(db)
            assert m, "no edge metrics computed"
            assert "regime_edge" in m, (
                "regime_edge missing -- go-live criterion 1 would be "
                "permanently blocked on a missing input")
            assert "regime_mix" in m, "regime_mix missing"
        finally:
            db.query(AIPortfolioSnapshot).filter_by(user_id=1).delete()
            db.query(AIPortfolioConfig).filter_by(user_id=1).delete()
            # Remove the MarketSnapshot rows this test created, so it leaves
            # the shared table exactly as it found it.
            db.query(MarketSnapshot).filter(
                MarketSnapshot.date >= date.today() - timedelta(days=440),
                MarketSnapshot.date <= date.today() - timedelta(days=401),
            ).delete(synchronize_session=False)
            db.commit()
            db.close()
