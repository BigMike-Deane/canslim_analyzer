"""IWM history backfill for market_snapshots (2026-09-22).

IWM became a tracked (zero-weight) index on Sep-22. Every book started months
earlier, so the scoreboard's alpha-vs-IWM and any small-cap read need IWM on
the old snapshot rows. backend.main.backfill_iwm_snapshots fills them once at
boot from Yahoo's daily history; these tests pin its contract with an
injected series (no network).
"""

import os
from datetime import date, timedelta

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import init_db, SessionLocal, MarketSnapshot
from backend.main import backfill_iwm_snapshots

D0 = date(2026, 3, 2)  # a Monday


@pytest.fixture
def db():
    init_db()
    s = SessionLocal()
    s.query(MarketSnapshot).delete()
    s.commit()
    try:
        yield s
    finally:
        s.query(MarketSnapshot).delete()
        s.commit()
        s.close()


def _series(n, start=100.0):
    """n consecutive calendar days of closes: 100, 101, 102, ..."""
    return {D0 + timedelta(days=i): start + i for i in range(n)}


def test_fills_price_and_mas_from_trailing_closes(db):
    closes = _series(250)
    day = D0 + timedelta(days=210)
    db.add(MarketSnapshot(date=day, spy_price=500.0))
    db.commit()

    assert backfill_iwm_snapshots(db, closes) == 1
    row = db.query(MarketSnapshot).filter_by(date=day).one()
    assert row.iwm_price == 310.0
    # 50 closes ending at index 210: 261..310 -> mean 285.5
    assert row.iwm_50_ma == pytest.approx(285.5)
    # 200 closes ending at index 210: 111..310 -> mean 210.5
    assert row.iwm_200_ma == pytest.approx(210.5)


def test_short_history_leaves_ma_null_not_guessed(db):
    closes = _series(30)
    day = D0 + timedelta(days=20)
    db.add(MarketSnapshot(date=day))
    db.commit()
    backfill_iwm_snapshots(db, closes)
    row = db.query(MarketSnapshot).filter_by(date=day).one()
    assert row.iwm_price == 120.0
    assert row.iwm_50_ma is None and row.iwm_200_ma is None


def test_non_trading_day_carries_back_to_prior_close(db):
    closes = {D0: 100.0, D0 + timedelta(days=4): 104.0}   # Mon, Fri
    sat = D0 + timedelta(days=5)
    db.add(MarketSnapshot(date=sat))
    db.add(MarketSnapshot(date=D0 - timedelta(days=3)))   # before the series
    db.commit()
    assert backfill_iwm_snapshots(db, closes) == 1
    assert db.query(MarketSnapshot).filter_by(date=sat).one().iwm_price == 104.0
    assert db.query(MarketSnapshot).filter_by(
        date=D0 - timedelta(days=3)).one().iwm_price is None


def test_idempotent_and_never_rewrites_live_values(db):
    closes = _series(60)
    day = D0 + timedelta(days=55)
    db.add(MarketSnapshot(date=day, iwm_price=999.0, iwm_50_ma=900.0))
    db.commit()
    assert backfill_iwm_snapshots(db, closes) == 0
    row = db.query(MarketSnapshot).filter_by(date=day).one()
    assert row.iwm_price == 999.0 and row.iwm_50_ma == 900.0


def test_empty_series_is_a_no_op(db):
    db.add(MarketSnapshot(date=D0))
    db.commit()
    assert backfill_iwm_snapshots(db, {}) == 0


def test_no_empty_rows_means_no_fetch(db, monkeypatch):
    """Every boot runs the backfill; once history is filled it must not
    touch the network."""
    import data_fetcher
    calls = []
    monkeypatch.setattr(data_fetcher, "fetch_price_from_chart_api",
                        lambda *a, **k: calls.append(a) or {})
    db.add(MarketSnapshot(date=D0, iwm_price=100.0))
    db.commit()
    assert backfill_iwm_snapshots(db) == 0
    assert calls == []
