"""Boot-time ATR-stop cache warm-up (Sep-28).

A restart empties trading_engine._atr_stop_cache; until the first market-hours
stop check refills it, the Exit Plan cards show the 7% base and the live
checker's fetch-failure fallback drops to base. warm_atr_stop_cache refills it
for every held ticker at boot, with the base the live checker would use.
"""

import pytest

import backend.ai_trader as at
import backend.broker_mirror as bm
from backend.database import SessionLocal, AIPortfolioConfig, AIPortfolioPosition, User
from backend.trading_engine import _atr_stop_cache, cache_atr_stop, get_cached_atr_stop

UID = 9471


def _bars(atr):
    """20 flat-close days whose true range is `atr` -> 14-day ATR == atr."""
    closes = [100.0] * 20
    return ([c + atr / 2 for c in closes], [c - atr / 2 for c in closes], closes)


def _wipe(db):
    db.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == UID).delete()
    db.query(AIPortfolioConfig).filter(AIPortfolioConfig.user_id == UID).delete()
    db.commit()


@pytest.fixture
def db(monkeypatch):
    _atr_stop_cache.clear()
    s = SessionLocal()
    _wipe(s)
    if not s.query(User).filter_by(id=UID).first():
        s.add(User(id=UID, email=f"u{UID}@acct.example.org", display_name="warmup",
                   is_active=True, is_admin=False, hashed_password=""))
    s.add(AIPortfolioConfig(user_id=UID, is_active=True, strategy="nostate_cs_bear"))
    s.commit()
    # Only the test user's book: other rows in the shared test DB are not ours.
    real_query = s.query

    def scoped_query(model, *a, **k):
        q = real_query(model, *a, **k)
        if model is AIPortfolioConfig:
            q = q.filter(AIPortfolioConfig.user_id == UID)
        return q
    monkeypatch.setattr(s, "query", scoped_query)
    monkeypatch.setattr(bm, "stop_context",
                        lambda db, uid: {"base_pct": 7.0, "use_atr": True, "guard_cfg": {}})
    yield s
    monkeypatch.undo()
    _wipe(s)
    s.close()
    _atr_stop_cache.clear()


def _hold(db, ticker, price=100.0):
    db.add(AIPortfolioPosition(user_id=UID, ticker=ticker, shares=10,
                               cost_basis=price, current_price=price))
    db.commit()


def test_warms_every_held_ticker_with_the_widened_stop(db, monkeypatch):
    _hold(db, "VOL")      # ATR 4 -> 4% * 2.5 = 10% stop (wider than base)
    _hold(db, "CALM")     # ATR 1 -> 2.5%, base 7% wins
    monkeypatch.setattr(at, "_daily_hlc_for_atr",
                        lambda t: _bars(4.0 if t == "VOL" else 1.0))
    r = at.warm_atr_stop_cache(db)
    assert sorted(r["warmed"]) == ["CALM", "VOL"] and r["failed"] == []
    assert get_cached_atr_stop("VOL") == pytest.approx(10.0)
    assert get_cached_atr_stop("CALM") == pytest.approx(7.0)


def test_failed_fetch_is_reported_and_leaves_no_entry(db, monkeypatch):
    _hold(db, "DEAD")
    monkeypatch.setattr(at, "_daily_hlc_for_atr", lambda t: None)
    r = at.warm_atr_stop_cache(db)
    assert r == {"warmed": [], "failed": ["DEAD"]}
    assert get_cached_atr_stop("DEAD") is None


def test_already_cached_ticker_is_not_refetched(db, monkeypatch):
    _hold(db, "LIVE")
    cache_atr_stop("LIVE", 14.0)          # the live cycle got there first
    calls = []
    monkeypatch.setattr(at, "_daily_hlc_for_atr", lambda t: calls.append(t) or _bars(1.0))
    r = at.warm_atr_stop_cache(db)
    assert calls == [] and r["warmed"] == []
    assert get_cached_atr_stop("LIVE") == 14.0


def test_priceless_position_is_skipped(db, monkeypatch):
    db.add(AIPortfolioPosition(user_id=UID, ticker="NOPX", shares=1, cost_basis=10.0,
                               current_price=None))
    db.commit()
    monkeypatch.setattr(at, "_daily_hlc_for_atr", lambda t: _bars(1.0))
    assert at.warm_atr_stop_cache(db) == {"warmed": [], "failed": []}
