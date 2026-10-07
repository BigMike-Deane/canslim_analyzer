# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""H7: the live rules (backend/backtester.py, the live trader's mirror) run on
point-in-time data (pre-registered in docs/phase2-pit-backtest-plan.md,
"H7 pre-registration").

Only data plumbing is replaced; every trading method is inherited unchanged:
  - PITDataProvider: stitched per-CIK prices incl. delisted names (FMP + Alpaca
    + ticker history), index ETFs from the PIT cache; no network, no disk cache.
  - PITBacktester._calculate_scores: the daily PIT score table (p7_daily_scores)
    served through the engine's own frozen-score path (_build_score_from_frozen,
    which adds fresh price-derived fields: bases, breakouts, RS, A/D).
    projected_growth uses the engine's formula (EPS growth x .30 + annual CAGR x
    .25 + RS momentum x .45). Held names missing from a day's table carry their
    last score forward (live always rescans holdings).
  - static_data (sector, days to earnings, beat streak, surprise, 13F inst %) is
    refreshed per date instead of frozen at today's values; estimate revision is
    None (no history: known gap).
  - 0.095% cost per side on every fill (the engine has none).
  - No start-date survivorship filter: names enter when eligible, delisted
    names exit at their last price via the engine's stale-price handling.

Runs on a scratch SQLite DB (DATABASE_URL set below) -- never production.

  python3 p7_backtest.py --offset 0 [--end 2026-10-05]
Output: META_DIR/h7/vintage_<offset>.json (equity curve, trades, summary)
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import date

import pandas as pd  # noqa: E402

import m2_adapter as m  # noqa: E402  (sets sys.path to the repo root)
from common import DATA_DIR, META_DIR, load_prices  # noqa: E402

SCRATCH = DATA_DIR / "h7_db"  # throwaway SQLite per vintage; survives restarts, never production

COST = 0.00095
START, END = "2016-01-04", "2026-10-05"
SCORE_FLOOR = 35  # lowest effective_min_score the engine can reach (score-floor decay / recovery)


def _db(offset):
    SCRATCH.mkdir(exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{SCRATCH}/h7_v{offset}.db"
    sys.path.insert(0, str(m.REPO / "backend"))
    from backend.database import Base, SessionLocal, engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return SessionLocal()


def load_daily():
    files = sorted(glob.glob(str(META_DIR / "daily" / "*.csv.gz")))
    d = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    d = d[d.date.notna()]
    # one trading key per company: its symbol, unless two CIKs share it
    sym = d.drop_duplicates("cik").set_index("cik").symbol
    dup = set(sym[sym.duplicated(keep=False)].index)
    d["key"] = [f"{s}.{c}" if c in dup else s for c, s in zip(d.cik, d.symbol)]
    # ~10M rows: keep only what the engine reads, in compact types
    d = d.drop(columns=["symbol", "mcap_m"])
    for c in ["total", "c", "a", "n", "s", "l", "i", "m", "eps_growth", "annual_cagr", "surprise_pct", "inst_pct",
              "days_to_earnings"]:
        d[c] = d[c].astype("float32")
    d["beat_streak"] = d.beat_streak.astype("int16")
    return d


def make_classes():
    from backend.backtester import BacktestEngine
    from backend.historical_data import MARKET_INDEXES, HistoricalDataProvider

    def _frame(px):
        f = px[["Open", "High", "Low", "Close", "Volume"]].copy()
        f.columns = ["open", "high", "low", "close", "volume"]
        f.insert(0, "date", [x.date() for x in f.index])
        return f.reset_index(drop=True)

    class PITDataProvider(HistoricalDataProvider):
        def __init__(self, key_to_cik, **kw):
            self.key_to_cik = key_to_cik
            super().__init__(list(key_to_cik), **kw)

        def _init_disk_cache(self):
            self._disk_cache, self._disk_cache_enabled = None, False

        def preload_data(self, start_date, end_date, progress_callback=None):
            lo = pd.Timestamp(start_date) - pd.Timedelta(days=400)
            for idx in MARKET_INDEXES:
                self._index_cache[idx] = _frame(load_prices(idx).loc[lo:pd.Timestamp(end_date)])
            for key, cik in self.key_to_cik.items():
                px = m.cik_prices(cik).loc[lo:pd.Timestamp(end_date)]
                if len(px):
                    self._price_cache[key] = _frame(px)
            spy = self._index_cache["SPY"]
            self._trading_days = sorted(spy.loc[(spy.date >= start_date) & (spy.date <= end_date), "date"])
            self._is_loaded = True
            return bool(self._price_cache)

    class PITBacktester(BacktestEngine):
        def __init__(self, db, backtest_id, daily):
            super().__init__(db, backtest_id)
            self.daily = {k: g.set_index("key") for k, g in daily.groupby("date")}
            self.key_to_cik = daily.drop_duplicates("key").set_index("key").cik.to_dict()
            self.last_row: dict = {}
            self.equity: list = []

        # ---- data plumbing -------------------------------------------------
        def _rs_momentum(self, t, d):
            rs = [min(self.data_provider.get_relative_strength(t, d, k) or 1.0, 3.0) for k in (12, 3, 1)]
            return ((rs[0] * 0.40 + rs[1] * 0.35 + rs[2] * 0.25) - 1.0) * 100

        def _calculate_scores(self, current_date, tickers=None):
            ds = str(current_date)
            if self._score_cache_date != ds:
                self._score_cache, self._score_cache_date = {}, ds
            day = self.daily.get(ds)
            if day is not None:
                for t, r in day.iterrows():
                    self.last_row[t] = r
            if tickers is None:
                want = [] if day is None else list(day.index[day.total >= SCORE_FLOOR])
                want += [t for t in self.positions if t not in want]
            else:
                want = tickers
            out = {}
            for t in want:
                if t in self._score_cache:
                    out[t] = self._score_cache[t]
                    continue
                r = day.loc[t] if (day is not None and t in day.index) else self.last_row.get(t)
                if r is None:
                    continue
                # python floats: float32 panel values would reach trade JSON (signal_factors) unserializable
                frozen = {"total_score": float(r.total), "c_score": float(r.c), "a_score": float(r.a),
                          "n_score": float(r.n), "s_score": float(r.s), "l_score": float(r.l),
                          "i_score": float(r.i), "m_score": float(r.m),
                          "projected_growth": float(r.eps_growth * 0.30 + r.annual_cagr * 0.25
                                                    + self._rs_momentum(t, current_date) * 0.45)}
                sd = self._build_score_from_frozen(t, current_date, frozen)
                if sd:
                    out[t] = self._score_cache[t] = sd
            return out

        def _save_persistent_scores(self, date_str, scores):
            pass

        def _refresh_static(self, current_date):
            day = self.daily.get(str(current_date))
            if day is None:
                return
            for t, r in day.iterrows():
                self.static_data[t] = {
                    "sector": r.sector if isinstance(r.sector, str) else "",
                    "days_to_earnings": None if pd.isna(r.days_to_earnings) else int(r.days_to_earnings),
                    "earnings_beat_streak": int(r.beat_streak), "earnings_surprise_pct": float(r.surprise_pct),
                    "institutional_holders_pct": float(r.inst_pct), "eps_estimate_revision_pct": None,
                    "weeks_in_base": 0, "name": t}

        def _simulate_day(self, current_date):
            self._refresh_static(current_date)
            super()._simulate_day(current_date)
            self.equity.append((str(current_date), self._get_portfolio_value(current_date)))

        # ---- costs ---------------------------------------------------------
        def _execute_buy(self, current_date, trade):
            trade.price *= 1 + COST
            return super()._execute_buy(current_date, trade)

        def _execute_pyramid(self, current_date, trade):
            trade.price *= 1 + COST
            return super()._execute_pyramid(current_date, trade)

        def _execute_sell(self, current_date, trade):
            trade.price *= 1 - COST
            return super()._execute_sell(current_date, trade)

        # ---- run: the engine's run() minus DB universe / snapshot / filter -------
        def run_pit(self):
            self.data_provider = PITDataProvider(self.key_to_cik, data_reference_date=self.backtest.end_date)
            self.data_provider.preload_data(self.backtest.start_date, self.backtest.end_date)
            self.data_provider.precompute_market_direction()
            days = self.data_provider.get_trading_days()
            self.spy_start_price = self.data_provider.get_spy_price_on_date(days[0])
            if self.spy_start_price > 0:
                self.spy_shares = self.backtest.starting_cash / self.spy_start_price
            if self.market_state_enabled:
                s = self.data_provider.get_spy_daily_data(days[0])
                self.market_state.initialize_state(spy_close=s["close"], spy_ma50=s["ma50"], spy_ema21=s["ema21"])
            self._refresh_static(days[0])
            self.is_seed_day = True
            self._seed_initial_positions(days[0])
            self._trading_days = days
            t0 = time.time()
            for i, d in enumerate(days):
                self._current_day_idx = i
                if i > 0:
                    self.is_seed_day = False
                self._simulate_day(d)
                if i % 50 == 0:
                    self.db.commit()
                    v = self.equity[-1][1] if self.equity else 0
                    print(f"  {d} day {i}/{len(days)} value ${v:,.0f} positions {len(self.positions)} "
                          f"{time.time() - t0:.0f}s", flush=True)
            self._calculate_final_metrics()
            self.db.commit()

    return PITDataProvider, PITBacktester


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--end", default=END)
    a = ap.parse_args()
    db = _db(a.offset)
    from backend.database import BacktestRun, BacktestTrade
    _, PITBacktester = make_classes()
    t0 = time.time()
    daily = load_daily()
    sess = sorted(daily.date.unique())
    start = sess[a.offset]
    print(f"daily table {len(daily):,} rows, {daily.key.nunique():,} names; vintage +{a.offset} starts {start}; "
          f"load {time.time() - t0:.0f}s", flush=True)
    bt = BacktestRun(name=f"h7-v{a.offset}", start_date=date.fromisoformat(start), end_date=date.fromisoformat(a.end),
                     starting_cash=25000, stock_universe="custom", strategy="nostate_cs_bear", status="running")
    db.add(bt)
    db.commit()
    eng = PITBacktester(db, bt.id, daily[daily.date >= str(pd.Timestamp(start) - pd.Timedelta(days=10))])
    del daily  # the engine keeps its own per-date split
    eng.run_pit()
    trades = [{"date": str(t.date), "ticker": t.ticker, "action": t.action, "shares": t.shares, "price": t.price,
               "gain": t.realized_gain, "reason": t.reason}
              for t in db.query(BacktestTrade).filter(BacktestTrade.backtest_id == bt.id).all()]
    out = META_DIR / "h7"
    out.mkdir(exist_ok=True)
    json.dump({"offset": a.offset, "start": start, "end": a.end, "equity": eng.equity, "trades": trades,
               "total_return_pct": bt.total_return_pct, "max_drawdown_pct": bt.max_drawdown_pct},
              open(out / f"vintage_{a.offset}.json", "w"))
    print(f"done: {len(trades)} trades, final ${eng.equity[-1][1]:,.0f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
