"""Tests for /api/admin/experiment-gates — the gate-progress readout.

The endpoint measures accrual toward the pre-registered promotion gates
(config/default.yaml comments). Direct-invocation pattern mirrors
tests/test_ml_demotion_cohort.py.
"""

import asyncio
import os
from datetime import date, datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, AIPortfolioTrade, MarketSnapshot,
    ShadowStrategy, ShadowTrade,
)

T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    for model in (ShadowTrade, ShadowStrategy, MarketSnapshot, AIPortfolioTrade):
        db.query(model).delete()
    db.commit()
    try:
        yield db
    finally:
        for model in (ShadowTrade, ShadowStrategy, MarketSnapshot, AIPortfolioTrade):
            db.query(model).delete()
        db.commit()
        db.close()


def _arm(db, name, *, activated_at=T0, parent="nostate_cs_bear"):
    s = ShadowStrategy(
        name=name, parent_strategy=parent, config_snapshot={},
        scorer_overrides={}, starting_value=25000.0,
        activated_at=activated_at,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _trade(db, arm_id, ticker, action, *, price=100.0, shares=5.0,
           reason=None, signal_factors=None, executed_at=None):
    t = ShadowTrade(
        shadow_strategy_id=arm_id, ticker=ticker, action=action,
        shares=shares, price=price, total_value=shares * price,
        reason=reason, signal_factors=signal_factors,
        executed_at=executed_at or (T0 + timedelta(days=1)),
    )
    db.add(t)
    db.commit()
    return t


def _snap(db, d, spy_price, spy_50_ma):
    db.add(MarketSnapshot(date=d, spy_price=spy_price, spy_50_ma=spy_50_ma))
    db.commit()


def _call(db):
    from backend.routes.admin import experiment_gates
    return asyncio.run(experiment_gates(current_user=None, db=db))


def _metric(arm_row, label):
    for m in arm_row["gate_metrics"]:
        if m["label"] == label:
            return m
    raise AssertionError(f"metric {label!r} missing: {arm_row['gate_metrics']}")


class TestShape:
    def test_arms_listed_with_counts_and_email_gate(self, db_session):
        a = _arm(db_session, "shadow_wide_trail")
        _trade(db_session, a.id, "AAPL", "BUY")
        _trade(db_session, a.id, "AAPL", "SELL", reason="TRAILING STOP: x")
        out = _call(db_session)
        assert out["program_clocks"]["stop_loss_recheck"]["target"] == 5
        row = out["arms"][0]
        assert row["name"] == "shadow_wide_trail"
        assert row["buys"] == 1 and row["sells"] == 1
        assert _metric(row, "closed sells (weekly-email gate)")["n"] == 1
        assert _metric(row, "exits")["n"] == 1

    def test_sweep_rows_excluded_from_counts(self, db_session):
        a = _arm(db_session, "shadow_chop_spy")
        _trade(db_session, a.id, "SPY", "BUY", reason="SPY SWEEP: park idle cash")
        _trade(db_session, a.id, "SPY", "SELL", reason="SPY SWEEP LIQUIDATION")
        out = _call(db_session)
        row = out["arms"][0]
        assert row["buys"] == 0 and row["sells"] == 0


class TestChopDays:
    def test_counts_only_days_in_band_since_activation(self, db_session):
        _arm(db_session, "shadow_chop_damper", activated_at=T0)
        # in band (+1.0%), out of band (+3%), below MA (-1%), pre-activation in band
        _snap(db_session, date(2026, 8, 2), 505.0, 500.0)
        _snap(db_session, date(2026, 8, 3), 515.0, 500.0)
        _snap(db_session, date(2026, 8, 4), 495.0, 500.0)
        _snap(db_session, date(2026, 7, 20), 505.0, 500.0)
        out = _call(db_session)
        assert _metric(out["arms"][0], "chop days") == {
            "label": "chop days", "n": 1, "target": 15}


class TestBaselineComparisons:
    def test_count_exempt_pyramids_are_arm_only_rows(self, db_session):
        base = _arm(db_session, "shadow_baseline")
        relief = _arm(db_session, "shadow_sector_relief")
        d = T0 + timedelta(days=2)
        # Mirrored live pyramid: appears in BOTH stacks -> not counted
        _trade(db_session, base.id, "ARGX", "PYRAMID", reason="PYRAMID: Winner", executed_at=d)
        _trade(db_session, relief.id, "ARGX", "PYRAMID", reason="PYRAMID: Winner", executed_at=d)
        # Exempt pyramid: only in the relief arm -> counted
        _trade(db_session, relief.id, "NVDA", "PYRAMID", reason="PYRAMID: Winner", executed_at=d)
        out = _call(db_session)
        relief_row = next(r for r in out["arms"] if r["name"] == "shadow_sector_relief")
        assert _metric(relief_row, "count-exempt pyramids")["n"] == 1

    def test_cs_exempt_suppressed_pre_earnings_exits(self, db_session):
        base = _arm(db_session, "shadow_baseline")
        exempt = _arm(db_session, "shadow_cs_exempt")
        d = T0 + timedelta(days=3)
        # Baseline trims a CS name pre-earnings; the exempt arm holds -> suppressed=1
        _trade(db_session, base.id, "GWRE", "SELL",
               reason="PRE-EARNINGS PARTIAL: 3d to earnings", executed_at=d)
        # Both stacks exit another name -> not suppressed
        _trade(db_session, base.id, "HEI", "SELL",
               reason="PRE-EARNINGS STOP: 2d to earnings", executed_at=d)
        _trade(db_session, exempt.id, "HEI", "SELL",
               reason="PRE-EARNINGS STOP: 2d to earnings", executed_at=d)
        _trade(db_session, exempt.id, "CSPX", "BUY",
               signal_factors={"coiled_spring": True})
        out = _call(db_session)
        row = next(r for r in out["arms"] if r["name"] == "shadow_cs_exempt")
        assert _metric(row, "pre-earnings exits suppressed")["n"] == 1
        assert _metric(row, "CS-cohort buys")["n"] == 1

    def test_ml_veto_off_counts_sub_threshold_buys(self, db_session):
        """The kill-or-bless arm's gate metric: buys the live 0.30 veto
        would have blocked. Missing ml_confidence must not count."""
        a = _arm(db_session, "shadow_ml_veto_off")
        _trade(db_session, a.id, "LOW", "BUY",
               signal_factors={"ml_confidence": 0.209})
        _trade(db_session, a.id, "HIGH", "BUY",
               signal_factors={"ml_confidence": 0.35})
        _trade(db_session, a.id, "NOML", "BUY",
               signal_factors={"breakout": True})
        out = _call(db_session)
        assert _metric(out["arms"][0], "sub-0.30-confidence buys taken") == {
            "label": "sub-0.30-confidence buys taken", "n": 1, "target": 5}

    def test_cs_window14_counts_widened_band_buys(self, db_session):
        a = _arm(db_session, "shadow_cs_window14")
        _trade(db_session, a.id, "A8", "BUY",
               signal_factors={"coiled_spring": True, "cs_days_to_earnings": 12})
        _trade(db_session, a.id, "A5", "BUY",
               signal_factors={"coiled_spring": True, "cs_days_to_earnings": 5})
        _trade(db_session, a.id, "NOCS", "BUY", signal_factors={"breakout": True})
        out = _call(db_session)
        assert _metric(out["arms"][0], "CS buys in 8-14d band")["n"] == 1


class TestStopLossClock:
    def test_owner_stops_since_cutoff_with_avg(self, db_session):
        def stop(user_id, cost, shares, realized, when):
            db_session.add(AIPortfolioTrade(
                ticker="X", action="SELL", shares=shares, price=cost,
                total_value=cost * shares, cost_basis=cost,
                realized_gain=realized, user_id=user_id,
                reason="STOP LOSS: Down 8.0%", executed_at=when))
            db_session.commit()
        stop(1, 100.0, 10.0, -80.0, datetime(2026, 7, 1, tzinfo=timezone.utc))   # -8%
        stop(2, 100.0, 10.0, -90.0, datetime(2026, 8, 1, tzinfo=timezone.utc))   # -9%
        stop(3, 100.0, 10.0, -80.0, datetime(2026, 8, 1, tzinfo=timezone.utc))   # wrong user
        stop(1, 100.0, 10.0, -80.0, datetime(2026, 6, 1, tzinfo=timezone.utc))   # pre-cutoff
        out = _call(db_session)
        clock = out["program_clocks"]["stop_loss_recheck"]
        assert clock["n"] == 2
        assert clock["avg_loss_pct"] == -8.5
        assert clock["bar_pct"] == -10.0


def _stop_sell(db, gain_pct, *, user_id=1, ticker="AAA", days_ago=10,
               split_artifact=False):
    """Owner-portfolio STOP LOSS sell landing at gain_pct vs cost."""
    cost, shares = 100.0, 10.0
    sf = {"sell_reason": "STOP LOSS", "gain_pct": gain_pct}
    if split_artifact:
        sf["split_artifact"] = True
    t = AIPortfolioTrade(
        user_id=user_id, ticker=ticker, action="SELL",
        shares=shares, price=cost * (1 + gain_pct / 100),
        total_value=shares * cost * (1 + gain_pct / 100),
        cost_basis=cost, realized_gain=shares * cost * gain_pct / 100,
        reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%",
        signal_factors=sf,
        executed_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago),
    )
    db.add(t)
    db.commit()
    return t


class TestStopClockVerdict:
    """PM program 2026-08-25: the stop-loss re-check verdict fires
    mechanically at n>=5 and split artifacts are excluded (same rule as
    the reconciliation endpoint)."""

    def test_no_verdict_below_target(self, db_session):
        for i, g in enumerate((-8.0, -8.5, -9.0)):
            _stop_sell(db_session, g, ticker=f"T{i}")
        clock = _call(db_session)["program_clocks"]["stop_loss_recheck"]
        assert clock["n"] == 3
        assert clock["verdict"] is None

    def test_pass_at_five_clean_stops_above_bar(self, db_session):
        for i, g in enumerate((-8.0, -8.5, -9.0, -7.5, -8.2)):
            _stop_sell(db_session, g, ticker=f"T{i}")
        clock = _call(db_session)["program_clocks"]["stop_loss_recheck"]
        assert clock["n"] == 5
        assert clock["verdict"] == "PASS"
        assert clock["avg_loss_pct"] > -10.0

    def test_fail_when_average_breaches_bar(self, db_session):
        for i, g in enumerate((-12.0, -14.0, -11.0, -13.0, -12.5)):
            _stop_sell(db_session, g, ticker=f"T{i}")
        clock = _call(db_session)["program_clocks"]["stop_loss_recheck"]
        assert clock["verdict"] == "FAIL"

    def test_split_artifacts_excluded(self, db_session):
        for i, g in enumerate((-8.0, -8.5, -9.0, -7.5)):
            _stop_sell(db_session, g, ticker=f"T{i}")
        # A -50% split artifact (the SFBS shape) must not reach the cohort:
        # n stays 4 (no verdict yet) and the average stays clean.
        _stop_sell(db_session, -50.2, ticker="SFBS", split_artifact=True)
        clock = _call(db_session)["program_clocks"]["stop_loss_recheck"]
        assert clock["n"] == 4
        assert clock["verdict"] is None
        assert clock["avg_loss_pct"] > -10.0

    def test_pure_function_matches_endpoint(self, db_session):
        from backend.routes.admin import compute_experiment_gates
        for i, g in enumerate((-8.0, -8.5)):
            _stop_sell(db_session, g, ticker=f"T{i}")
        assert compute_experiment_gates(db_session) == _call(db_session)


class TestChopArmGates:
    """Gate metrics for the aug-25 chop arms (entry bar + trim)."""

    def test_chop_trim_counts_trims(self, db_session):
        arm = _arm(db_session, "shadow_chop_trim")
        _arm(db_session, "shadow_baseline")
        _trade(db_session, arm.id, "AAA", "SELL",
               reason="CHOP TRIM (30%): +31% above 50MA in chop regime")
        _trade(db_session, arm.id, "BBB", "SELL", reason="TRAILING STOP: test")
        out = _call(db_session)
        row = next(a for a in out["arms"] if a["name"] == "shadow_chop_trim")
        trims = _metric(row, "chop trims fired")
        assert trims["n"] == 1
        assert trims["target"] == 5
        assert _metric(row, "chop days") is not None

    def test_chop_entry_bar_suppression_proxy(self, db_session):
        arm = _arm(db_session, "shadow_chop_entry_bar")
        base = _arm(db_session, "shadow_baseline")
        # Baseline bought two names; the arm only took one -> 1 suppressed
        _trade(db_session, base.id, "AAA", "BUY")
        _trade(db_session, base.id, "BBB", "BUY")
        _trade(db_session, arm.id, "AAA", "BUY")
        out = _call(db_session)
        row = next(a for a in out["arms"] if a["name"] == "shadow_chop_entry_bar")
        sup = _metric(row, "baseline buys not taken (suppression proxy)")
        assert sup["n"] == 1
