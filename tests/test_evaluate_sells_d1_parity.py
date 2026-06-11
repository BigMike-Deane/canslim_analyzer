"""D1 parity harness — `new_position_guard` clamp in the live sell path.

Seeds the June-19 exit-parity work (`docs/parity-audit-evaluate-sells.md`,
divergence **D1**) with a *behavioral* regression test, complementing the
source-string sentinel in `tests/test_stop_guard_monitor.py`.

The bug: `_check_and_execute_stop_losses_impl` (ai_trader.py ~1303-1314) and
the backtester (`_evaluate_sells`, backtester.py ~2621-2635) both clamp a new
position's ATR-widened stop down to `guard_stop_pct` (8%) for its first
`guard_days` (21) days. The automated live path — `evaluate_sells`, the only
sell path the scheduler ever runs — does **not**. So a fast-falling new buy
rides the ATR-widened stop (up to 20%) live while the backtest cuts it at 8%
(backtest-optimistic; cost a real −$967 FSLR stop-out on June 9).

Why behavioral, not just source-string: the existing sentinel asserts the
literal string ``new_position_guard`` is absent from `evaluate_sells`. A
post-freeze port that adds the guard but inverts the clamp (`max` instead of
`min`), uses the wrong threshold, or mis-computes the window would satisfy the
string check yet still leak money. This file pins the *decision outcome*.

The core assertion (`test_new_position_below_atr_stop_is_cut_by_guard`) is
marked ``xfail(strict=True)``: it FAILS today (guard missing → position not
cut at −12% under an 18% ATR stop) and will XPASS the moment the D1 fix lands,
which — because strict — turns into a hard failure that forces whoever ports
the guard to delete the marker. That is the intended "flip green on the fix"
trip-wire. When it trips on June 19, also delete `backend/stop_guard_monitor.py`
and `tests/test_stop_guard_monitor.py` per that file's own instructions.

Two un-xfailed control tests bracket the core one so a green xfail can only
mean "the guard now fires," never "the harness drifted":
  - a new position *beyond* the wide ATR stop is cut today (stop branch is
    wired and reachable — isolates the xfail to the guard clamp, not plumbing),
  - a *seasoned* (>guard_days) position at the same −12%/18% is NOT cut in
    either era (proves the −12%/18% scenario stops out solely because of the
    first-21-days guard window).

Frozen-file safe: adds a test only; touches neither `ai_trader.py` nor
`canslim_scorer.py`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import (
    Base,
    Stock,
    AIPortfolioConfig,
    AIPortfolioPosition,
)

# The ATR-widened stop the live path computes for a volatile new buy. Wider
# than the 8% guard and wider than the −12% loss, so the guard clamp is the
# *only* thing that decides whether the position is cut.
WIDE_ATR_STOP_PCT = 18.0


@pytest.fixture
def db_session():
    """Fresh in-memory SQLite per test (no global state leakage)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def wide_atr_stop(monkeypatch):
    """Pin `calculate_atr_stop` to a fixed wide stop — no Yahoo HTTP, and the
    guard clamp becomes the sole determinant of the stop-loss decision."""
    import backend.ai_trader as ai_trader
    monkeypatch.setattr(
        ai_trader, "calculate_atr_stop",
        lambda ticker, current_price, base_stop_pct: WIDE_ATR_STOP_PCT,
    )


@pytest.fixture
def stub_market_bullish(monkeypatch):
    """SPY > 50MA so the bearish-widening branch stays off and the stub
    avoids any market-data network call."""
    import data_fetcher
    monkeypatch.setattr(
        data_fetcher, "get_cached_market_direction",
        lambda: {
            "success": True,
            "weighted_signal": 2.0,
            "indexes": {"SPY": {"price": 500.0, "ma_50": 480.0,
                                "ma_200": 450.0, "ema_21": 495.0}},
        },
    )


@pytest.fixture
def no_historical_data(monkeypatch):
    """Short-circuit the VIX-proxy provider path (no real HistoricalDataProvider)."""
    import backend.ai_trader as ai_trader
    monkeypatch.setattr(ai_trader, "HistoricalDataProvider", None)


def _seed_config(db, **overrides):
    defaults = dict(
        user_id=1, starting_cash=25000.0, current_cash=10000.0,
        max_positions=8, max_position_pct=12.0, min_score_to_buy=72,
        sell_score_threshold=45, take_profit_pct=75.0, stop_loss_pct=7.0,
        is_active=True, strategy="nostate_optimized",
    )
    defaults.update(overrides)
    cfg = AIPortfolioConfig(**defaults)
    db.add(cfg)
    db.commit()
    return cfg


def _seed_stock(db, ticker, price):
    db.add(Stock(
        ticker=ticker, sector="Technology", current_price=price,
        canslim_score=75.0, is_growth_stock=False, is_breaking_out=False,
        volume_ratio=1.0, days_to_earnings=None, weeks_in_base=4,
        earnings_beat_streak=0, projected_growth=15.0,
    ))
    db.commit()


def _seed_position(db, ticker, *, gain_pct, days_held, price=88.0):
    """A position whose peak == current (drop_from_peak == 0) so the trailing
    branch can never pre-empt the stop-loss decision under test, and whose
    score is healthy so neither score-crash nor weak-position can fire. The
    only sell that can occur is the stop loss — making the assertion a clean
    read on the guard clamp."""
    cost_basis = price / (1 + gain_pct / 100)
    db.add(AIPortfolioPosition(
        user_id=1, ticker=ticker, shares=100.0, cost_basis=cost_basis,
        purchase_date=datetime.now(timezone.utc) - timedelta(days=days_held),
        purchase_score=80.0, current_price=price, current_value=price * 100,
        gain_loss=(price - cost_basis) * 100, gain_loss_pct=gain_pct,
        current_score=75.0, is_growth_stock=False,
        peak_price=price, peak_date=datetime.now(timezone.utc),
        pyramid_count=0, partial_profit_taken=0.0,
    ))
    db.commit()


def _stop_loss_sells(sells):
    return [s for s in sells if "STOP LOSS" in s["reason"]]


@pytest.mark.xfail(
    reason="D1: new_position_guard not yet ported into evaluate_sells "
           "(June-18 freeze; fix lands June 19). When this XPASSES, port is "
           "live — remove this marker and delete the stop_guard_monitor "
           "sentinel + its test per their instructions.",
    strict=True,
)
def test_new_position_below_atr_stop_is_cut_by_guard(
    db_session, wide_atr_stop, stub_market_bullish, no_historical_data,
):
    """A 5-day-old position down 12% under an 18% ATR stop MUST be cut: the
    new_position_guard clamps the stop to 8%, so −12% <= −8% triggers. Today
    `evaluate_sells` omits the guard, leaves the stop at 18%, and rides the
    loss — the D1 bug this test pins until the June-19 port lands."""
    from backend.ai_trader import evaluate_sells

    _seed_config(db_session)
    _seed_stock(db_session, "NEWFALL", price=88.0)
    _seed_position(db_session, "NEWFALL", gain_pct=-12.0, days_held=5)

    sells = evaluate_sells(db_session, user_id=1)

    assert _stop_loss_sells(sells), (
        "Expected the new_position_guard to clamp the stop to 8% and cut a "
        "−12% position held 5 days; evaluate_sells did not (D1 unfixed)."
    )


def test_new_position_beyond_atr_stop_is_cut_today(
    db_session, wide_atr_stop, stub_market_bullish, no_historical_data,
):
    """Positive control (no guard needed): a new position down 20% breaches
    even the un-clamped 18% ATR stop, so the stop-loss branch fires today.
    Proves the branch is wired and reachable — the xfail above is isolated to
    the guard clamp, not to broken plumbing."""
    from backend.ai_trader import evaluate_sells

    _seed_config(db_session)
    _seed_stock(db_session, "BIGFALL", price=80.0)
    _seed_position(db_session, "BIGFALL", gain_pct=-20.0, days_held=5, price=80.0)

    sells = evaluate_sells(db_session, user_id=1)

    assert _stop_loss_sells(sells), "−20% under an 18% stop must stop out today."


def test_seasoned_position_rides_wide_atr_stop(
    db_session, wide_atr_stop, stub_market_bullish, no_historical_data,
):
    """Control: the same −12%/18% position held 60 days (> guard_days=21) is
    NOT cut — in either era. The guard window has expired, so the only reason
    the 5-day version stops out is the first-21-days clamp. Pins that the
    xfail captures the guard, not just "any −12% position sells."""
    from backend.ai_trader import evaluate_sells

    _seed_config(db_session)
    _seed_stock(db_session, "SEASONED", price=88.0)
    _seed_position(db_session, "SEASONED", gain_pct=-12.0, days_held=60)

    sells = evaluate_sells(db_session, user_id=1)

    assert not _stop_loss_sells(sells), (
        "A seasoned (>21d) −12% position should ride the 18% ATR stop; the "
        "guard must not apply outside its window."
    )
