"""Corporate-action sweep (backend.corporate_actions), 2026-09-24.

Payload shapes are the ones Alpaca returned for the three names found in the
Sep-24 audit: ATAI (stock_and_cash_merger, CVR acquirer), FBRX (cash_merger),
HLX (name_change -> HOS). No test touches the network: active symbols,
actions and quotes are injected.
"""
import os
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database import (
    SessionLocal, init_db, AIPortfolioConfig, AIPortfolioPosition, AIPortfolioSnapshot,
    AIPortfolioTrade, ShadowStrategy, ShadowTrade, ShadowEquityMark, ShadowPositionPeak,
    Stock, DelistedTicker,
)
from backend import corporate_actions as ca

TICKERS = ["ATAI", "FBRX", "HLX", "DEAD", "ALIVE", "SEAL-PB", "HELD"]

ATAI = {"corporate_actions": {"stock_and_cash_mergers": [{
    "acquiree_symbol": "ATAI", "acquiree_rate": 1, "acquirer_symbol": "046CVR015",
    "acquirer_rate": 1, "cash_rate": 6.75, "effective_date": "2026-09-11",
    "payable_date": "2026-09-14", "process_date": "2026-09-14"}]}}
FBRX = {"corporate_actions": {"cash_mergers": [{
    "acquiree_symbol": "FBRX", "effective_date": "2026-08-27", "rate": 77}]}}
HLX = {"corporate_actions": {"name_changes": [{
    "old_symbol": "HLX", "new_symbol": "HOS", "process_date": "2026-09-02"}]}}
TODAY = date(2026, 9, 24)
ACTIVE = {"AAPL", "HELD"}


def _clean(db):
    for m in (ShadowPositionPeak, ShadowEquityMark, ShadowTrade, ShadowStrategy,
              AIPortfolioTrade, AIPortfolioPosition, AIPortfolioSnapshot, AIPortfolioConfig):
        db.query(m).delete()
    db.query(Stock).filter(Stock.ticker.in_(TICKERS)).delete(synchronize_session=False)
    db.query(DelistedTicker).filter(DelistedTicker.ticker.in_(TICKERS)).delete(synchronize_session=False)
    db.commit()


@pytest.fixture
def db():
    init_db()
    s = SessionLocal()
    _clean(s)
    try:
        yield s
    finally:
        _clean(s)
        s.close()


def _quote(days_ago_by_ticker):
    def fn(tk):
        if tk not in days_ago_by_ticker:
            return {}
        ts = datetime(2026, 9, 24, 16, tzinfo=timezone.utc) - timedelta(days=days_ago_by_ticker[tk])
        return {"symbol": tk, "price": 1.0, "volume": 0, "timestamp": int(ts.timestamp())}
    return fn


def test_normalize_real_payloads():
    acts = ca.normalize_actions(ATAI) + ca.normalize_actions(FBRX) + ca.normalize_actions(HLX)
    by = {a["symbol"]: a for a in acts}
    assert by["ATAI"]["kind"] == "stock_and_cash_merger" and by["ATAI"]["cash"] == 6.75
    assert by["ATAI"]["effective"] == date(2026, 9, 11)
    assert by["FBRX"] == {"kind": "cash_merger", "symbol": "FBRX", "cash": 77.0,
                          "effective": date(2026, 8, 27)}
    assert by["HLX"]["kind"] == "name_change" and by["HLX"]["new_symbol"] == "HOS"


def test_cash_per_share_counts_a_cvr_as_zero_but_not_tradable_stock():
    atai = ca.normalize_actions(ATAI)[0]
    assert ca.cash_per_share(atai, ACTIVE) == 6.75
    assert ca.cash_per_share(dict(atai, acquirer="AAPL"), ACTIVE) is None
    assert ca.cash_per_share(ca.normalize_actions(HLX)[0], ACTIVE) is None


def test_effective_open_is_0930_et_in_utc():
    assert ca.effective_open_utc(date(2026, 9, 11)) == datetime(2026, 9, 11, 13, 30)
    assert ca.effective_open_utc(date(2026, 12, 1)) == datetime(2026, 12, 1, 14, 30)


def test_live_cash_merger_closes_at_deal_cash_and_restates_history(db):
    db.add(AIPortfolioConfig(user_id=3, starting_cash=25000.0, current_cash=1000.0,
                             is_active=True, strategy="nostate_cs_bear"))
    db.add(AIPortfolioPosition(user_id=3, ticker="ATAI", shares=100.0, cost_basis=7.415,
                               current_price=7.35, current_value=735.0,
                               purchase_date=datetime(2026, 8, 20, 15, 27)))
    before = AIPortfolioSnapshot(user_id=3, timestamp=datetime(2026, 9, 10, 19, 55),
                                 total_value=25100.0, cash=1000.0, positions_value=24100.0,
                                 positions_count=8, total_return=100.0, total_return_pct=0.4)
    after = AIPortfolioSnapshot(user_id=3, timestamp=datetime(2026, 9, 12, 19, 55),
                                total_value=25200.0, cash=1000.0, positions_value=24200.0,
                                positions_count=8, total_return=200.0, total_return_pct=0.8,
                                prev_value=25100.0, value_change=100.0)
    db.add_all([before, after])
    db.commit()

    s = ca.run_sweep(db, active=ACTIVE, actions=ca.normalize_actions(ATAI),
                     quote_fn=_quote({}), today=TODAY, universe=False)

    assert [c["ticker"] for c in s["closed"]] == ["ATAI"]
    trade = db.query(AIPortfolioTrade).one()
    assert (trade.action, trade.price, trade.shares) == ("SELL", 6.75, 100.0)
    assert trade.executed_at == datetime(2026, 9, 11, 13, 30)
    assert trade.realized_gain == pytest.approx((6.75 - 7.415) * 100)
    assert trade.reason.startswith("CASH MERGER")
    assert db.query(AIPortfolioPosition).count() == 0
    assert db.query(AIPortfolioConfig).one().current_cash == pytest.approx(1675.0)
    db.expire_all()
    b, a = db.get(AIPortfolioSnapshot, before.id), db.get(AIPortfolioSnapshot, after.id)
    assert b.total_value == pytest.approx(25100.0)            # before the deal: untouched
    assert a.total_value == pytest.approx(25200.0 - 60.0)     # 100 sh x ($7.35 - $6.75)
    assert a.cash == pytest.approx(1675.0) and a.positions_count == 7
    assert a.value_change == pytest.approx(40.0)


def test_live_lot_bought_after_the_deal_is_left_alone(db):
    db.add(AIPortfolioConfig(user_id=3, starting_cash=25000.0, current_cash=1000.0, is_active=True))
    db.add(AIPortfolioPosition(user_id=3, ticker="ATAI", shares=10.0, cost_basis=7.0,
                               current_price=7.35, purchase_date=datetime(2026, 9, 15, 15)))
    db.commit()
    s = ca.run_sweep(db, active=ACTIVE, actions=ca.normalize_actions(ATAI),
                     quote_fn=_quote({}), today=TODAY, universe=False)
    assert s["closed"] == [] and db.query(AIPortfolioPosition).count() == 1


def test_shadow_holding_closed_at_effective_date_and_marks_rebuilt(db):
    strat = ShadowStrategy(name="shadow_ca_test", parent_strategy="nostate_cs_bear",
                           config_snapshot={}, scorer_overrides={}, starting_value=25000.0,
                           activated_at=datetime(2026, 8, 18, tzinfo=timezone.utc))
    db.add(strat)
    db.commit()
    db.add(ShadowTrade(shadow_strategy_id=strat.id, ticker="FBRX", action="BUY", shares=10.0,
                       price=76.93, total_value=769.3,
                       executed_at=datetime(2026, 8, 20, 19, 11, tzinfo=timezone.utc)))
    for d in (date(2026, 8, 26), date(2026, 8, 27), date(2026, 9, 1)):
        db.add(ShadowEquityMark(shadow_strategy_id=strat.id, date=d, equity=25000.0, cash=0.0,
                                positions_value=0.0, sweep_value=0.0, n_positions=1,
                                unpriced_positions=0))
    db.commit()

    s = ca.run_sweep(db, active=ACTIVE, actions=ca.normalize_actions(FBRX),
                     quote_fn=_quote({}), today=TODAY, universe=False)

    assert [(c["book"], c["ticker"]) for c in s["closed"]] == [("shadow_ca_test", "FBRX")]
    sell = db.query(ShadowTrade).filter(ShadowTrade.action == "SELL").one()
    assert sell.price == 77.0 and sell.shares == pytest.approx(10.0)
    assert sell.executed_at.replace(tzinfo=None) == datetime(2026, 8, 27, 13, 30)
    assert [m.date for m in db.query(ShadowEquityMark).all()] == [date(2026, 8, 26)]


def test_rename_of_held_name_needs_a_human(db):
    db.add(AIPortfolioConfig(user_id=1, starting_cash=25000.0, current_cash=1000.0, is_active=True))
    db.add(AIPortfolioPosition(user_id=1, ticker="HLX", shares=5.0, cost_basis=10.0,
                               current_price=10.6, purchase_date=datetime(2026, 8, 20)))
    db.commit()
    s = ca.run_sweep(db, active=ACTIVE, actions=ca.normalize_actions(HLX),
                     quote_fn=_quote({}), today=TODAY, universe=False)
    assert s["closed"] == [] and s["manual"] == ["HLX: name_change eff 2026-09-02 -> HOS"]
    assert db.query(AIPortfolioPosition).count() == 1


def test_held_name_that_stopped_trading_is_flagged(db):
    db.add(AIPortfolioConfig(user_id=1, starting_cash=25000.0, current_cash=1000.0, is_active=True))
    db.add(AIPortfolioPosition(user_id=1, ticker="HELD", shares=5.0, cost_basis=10.0,
                               current_price=10.0, purchase_date=datetime(2026, 9, 1)))
    db.commit()
    s = ca.run_sweep(db, active=ACTIVE, actions=[], quote_fn=_quote({"HELD": 7}),
                     today=TODAY, universe=False)
    assert len(s["stale_held"]) == 1 and s["stale_held"][0].startswith("HELD: last trade 2026-09-17")
    fresh = ca.run_sweep(db, active=ACTIVE, actions=[], quote_fn=_quote({"HELD": 0}),
                         today=TODAY, universe=False)
    assert fresh["stale_held"] == []


def test_dead_universe_names_go_to_delisted_tickers(db):
    for tk in ("DEAD", "ALIVE", "SEAL-PB", "HELD"):
        db.add(Stock(ticker=tk))
    db.commit()
    marked = ca.mark_dead_universe(db, ACTIVE, held={"HELD"},
                                   quote_fn=_quote({"DEAD": 30, "ALIVE": 1}),
                                   today=TODAY, pause_s=0)
    db.commit()
    assert marked == ["DEAD"]          # ALIVE trades; SEAL-PB is a preferred; HELD is active
    row = db.query(DelistedTicker).filter(DelistedTicker.ticker == "DEAD").one()
    assert row.failure_count >= 3 and row.recheck_after > datetime.now(timezone.utc).replace(tzinfo=None)
    # Already excluded -> not re-queried tomorrow.
    asked = []
    ca.mark_dead_universe(db, ACTIVE, held={"HELD"},
                          quote_fn=lambda tk: asked.append(tk) or _quote({"ALIVE": 1})(tk),
                          today=TODAY, pause_s=0)
    assert asked == ["ALIVE"]


def test_no_asset_list_means_no_action(db):
    s = ca.run_sweep(db, active=set(), actions=ca.normalize_actions(ATAI),
                     quote_fn=_quote({}), today=TODAY, universe=False)
    assert s["error"] and s["closed"] == []
