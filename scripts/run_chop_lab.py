"""Chop Lab — read-only decomposition of chop-day bleed on the live owner book.

Replays user-1's trade history into a daily lot-level book, marks it to
market with Yahoo daily closes, and attributes each day's portfolio-minus-SPY
excess to extension buckets and lot origins. Then sizes three counterfactual
levers over the SAME live window by re-running the replay with the lever on:

  trim30   arm-12 rule: first chop day a position sits >=25% above its own
           50MA, sell 30% to cash (once per position generation)
  pyrgate  candidate arm: suppress PYRAMID adds executed while SPY sat in
           the chop band (0..1.5% above its 50MA)
  damper   arm-3 rule: halve BUY lots executed while SPY sat in the chop band

Evidence for prioritization ONLY — forward shadow A/B stays the promotion
verdict (jun-22 rule). Same replay class as scripts/run_exit_lab.py: sizing
mechanics on historical marks, no scorer counterfactuals.

Run (read-only; local or anywhere with DB reach):
  DATABASE_URL=postgresql://canslim:<pw>@<host>:5432/canslim \\
      python3 scripts/run_chop_lab.py

Caveats: dividends ignored on both legs; SELL rows cap at available shares
when a lever shrank the book; chop flag prefers market_snapshots (the gate's
own source) and falls back to Yahoo-derived SPY/50MA when a snapshot is
missing.
"""
import os
import sys
from collections import defaultdict
from datetime import date

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

BAND_PCT = 1.5          # chop band: SPY 0..1.5% above its 50MA (gate parity)
TRIM_EXT_PCT = float(os.environ.get("TRIM_EXT_PCT", 25.0))  # arm-12 threshold
TRIM_PCT = 0.30         # arm-12 trim fraction
PRICE_START = "2026-01-02"  # warmup for per-stock 50MA before the April launch


def load_db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL (read-only queries only).")
    eng = create_engine(url)
    with eng.connect() as cx:
        trades = pd.read_sql(text(
            "SELECT ticker, action, shares, price, executed_at "
            "FROM ai_portfolio_trades WHERE user_id=1 "
            "ORDER BY executed_at, id"), cx)
        snaps = pd.read_sql(text(
            "SELECT date, spy_price, spy_50_ma FROM market_snapshots "
            "WHERE date >= '2026-03-01' ORDER BY date"), cx)
        port = pd.read_sql(text(
            "SELECT date, total_value, cash FROM ai_portfolio_snapshots "
            "WHERE user_id=1 ORDER BY date"), cx)
    trades["d"] = pd.to_datetime(trades["executed_at"]).dt.date
    snaps["date"] = pd.to_datetime(snaps["date"]).dt.date
    port["date"] = pd.to_datetime(port["date"]).dt.date
    return trades, snaps, port


def fetch_prices(tickers):
    data = yf.download(list(tickers) + ["SPY"], start=PRICE_START,
                       auto_adjust=False, progress=False)
    close = data["Close"].ffill()
    close.index = [ts.date() for ts in close.index]
    # Yahoo closes are retroactively split-adjusted even with
    # auto_adjust=False; DB share counts are raw. Un-adjust closes before
    # each in-window split so shares x price matches reality.
    for t in tickers:
        try:
            sp = yf.Ticker(t).splits
            sp = sp[sp.index >= PRICE_START]
            for ts, ratio in sp.items():
                sd = ts.date()
                mask = [d < sd for d in close.index]
                close.loc[mask, t] = close.loc[mask, t] * float(ratio)
                print(f"note: {t} {ratio:.0f}:1 split {sd} — closes before "
                      "it un-adjusted to match DB share counts")
        except Exception:
            pass
    return close


def chop_flags(days, snaps, spy, spy_ma):
    """date -> {'band','meter','gap'}; snapshot first, Yahoo fallback.

    band  — the ARMS' trigger: SPY 0..BAND_PCT% above its 50MA
    meter — the edge card's chop bucket: gap <= BAND_PCT INCLUDING
            below-MA days (edge_metrics.regime_conditional_edge)
    """
    snap_gap = {}
    for _, r in snaps.iterrows():
        if r.spy_price and r.spy_50_ma:
            snap_gap[r.date] = (r.spy_price - r.spy_50_ma) / r.spy_50_ma * 100
    out, fallbacks = {}, 0
    for d in days:
        gap = snap_gap.get(d)
        if gap is None:
            gap = (spy[d] - spy_ma[d]) / spy_ma[d] * 100
            fallbacks += 1
        out[d] = {"band": 0 <= gap <= BAND_PCT, "meter": gap <= BAND_PCT,
                  "gap": gap}
    if fallbacks:
        print(f"note: {fallbacks}/{len(days)} days used Yahoo-derived chop flag")
    return out


def replay(trades, days, close, ma50, flags, spy_ret,
           lever=None, start_cash=25000.0):
    """Daily lot-level replay. Returns per-day excess-$ series + attribution.

    lever: None | 'trim30' | 'pyrgate' | 'damper'
    Excess convention (vs an all-SPY benchmark of equal value):
      stock lot: v * (ret_stock - ret_spy);  cash: v * (0 - ret_spy)
    """
    by_day = trades.groupby("d")
    lots = defaultdict(list)      # ticker -> [{sh, origin, chop_open}]
    cash = start_cash
    trimmed_gen = set()           # tickers trimmed this generation (trim30)
    trims_fired, pyr_skipped, buys_halved = 0, [], 0
    daily = {}                    # d -> dict(excess, chop, buckets, origins)

    prev = None
    for d in days:
        # 1) mark yesterday's book over (prev -> d)
        if prev is not None:
            row = {"excess": 0.0, "band": flags[d]["band"],
                   "meter": flags[d]["meter"],
                   "buckets": defaultdict(float), "origins": defaultdict(float),
                   "pos_value": 0.0}
            for t, ls in lots.items():
                sh = sum(l["sh"] for l in ls)
                if sh <= 1e-9:
                    continue
                c0, c1 = close[t].get(prev), close[t].get(d)
                if c0 is None or c1 is None or pd.isna(c0) or pd.isna(c1):
                    continue
                ret = c1 / c0 - 1
                exc = sh * c0 * (ret - spy_ret[d])
                row["excess"] += exc
                row["pos_value"] += sh * c0
                m = ma50[t].get(prev)
                ext = (c0 / m - 1) * 100 if m and not pd.isna(m) else None
                bucket = ("na" if ext is None else
                          "below_ma" if ext < 0 else
                          "<10%" if ext < 10 else
                          "10-25%" if ext < TRIM_EXT_PCT else ">=25%")
                row["buckets"][bucket] += exc
                for l in ls:
                    if l["sh"] > 1e-9:
                        key = (l["origin"] +
                               ("_chop" if l["chop_open"] else "_trend"))
                        row["origins"][key] += l["sh"] * c0 * (ret - spy_ret[d])
            row["excess"] += cash * (0 - spy_ret[d])
            row["cash"] = cash
            daily[d] = row

        # 2) apply today's trades (end-of-day convention)
        is_chop = flags[d]["band"]
        for _, tr in (by_day.get_group(d).iterrows()
                      if d in by_day.groups else []):
            t, sh, px = tr.ticker, float(tr.shares), float(tr.price)
            if tr.action in ("BUY", "PYRAMID"):
                if lever == "pyrgate" and tr.action == "PYRAMID" and is_chop:
                    pyr_skipped.append((str(d), t, round(sh * px)))
                    continue
                if lever == "damper" and tr.action == "BUY" and is_chop:
                    sh *= 0.5
                    buys_halved += 1
                if sum(l["sh"] for l in lots[t]) <= 1e-9:
                    trimmed_gen.discard(t)   # new generation
                lots[t].append({"sh": sh, "chop_open": is_chop,
                                "origin": "pyr" if tr.action == "PYRAMID"
                                else "buy"})
                cash -= sh * px
            elif tr.action == "SELL":
                want = sh
                for l in lots[t]:
                    take = min(l["sh"], want)
                    l["sh"] -= take
                    want -= take
                    cash += take * px
                    if want <= 1e-9:
                        break

        # 3) arm-12 trim rule (after trades, at today's close);
        #    trim30spy parks proceeds in SPY instead of cash
        if lever in ("trim30", "trim30spy") and is_chop:
            for t, ls in list(lots.items()):
                if t == "SPY":
                    continue
                sh = sum(l["sh"] for l in ls)
                c, m = close[t].get(d), ma50[t].get(d)
                if (sh > 1e-9 and t not in trimmed_gen and c and m
                        and not pd.isna(c) and not pd.isna(m)
                        and (c / m - 1) * 100 >= TRIM_EXT_PCT):
                    want = sh * TRIM_PCT
                    proceeds = 0.0
                    for l in ls:
                        take = min(l["sh"], want)
                        l["sh"] -= take
                        want -= take
                        proceeds += take * c
                        if want <= 1e-9:
                            break
                    if lever == "trim30spy":
                        lots["SPY"].append(
                            {"sh": proceeds / close["SPY"][d],
                             "chop_open": True, "origin": "buy"})
                    else:
                        cash += proceeds
                    trimmed_gen.add(t)
                    trims_fired += 1
        prev = d

    meta = {"trims_fired": trims_fired, "pyr_skipped": pyr_skipped,
            "buys_halved": buys_halved, "end_cash": cash}
    return daily, meta


def summarize(daily, key="meter"):
    chop = [r for r in daily.values() if r[key]]
    trend = [r for r in daily.values() if not r[key]]
    def tot(rows):
        return sum(r["excess"] for r in rows)
    def bps(rows):
        vals = [(r["excess"] / (r["pos_value"] + r["cash"])) * 1e4
                for r in rows if r["pos_value"] + r["cash"] > 0]
        return sum(vals) / len(vals) if vals else 0.0
    return {"chop_n": len(chop), "trend_n": len(trend),
            "chop_$": tot(chop), "trend_$": tot(trend), "all_$": tot(daily.values()),
            "chop_bps": bps(chop), "trend_bps": bps(trend)}


def main():
    trades, snaps, port = load_db()
    tickers = sorted(trades.ticker.unique())
    close = fetch_prices(tickers)
    ma50 = close.rolling(50, min_periods=35).mean()
    spy = close["SPY"]
    spy_ma = ma50["SPY"]

    first = trades.d.min()
    days = [d for d in close.index if d >= first]
    spy_ret = {d1: spy[d1] / spy[d0] - 1
               for d0, d1 in zip(days, days[1:])}
    days_r = days  # replay from first trade date; excess starts day 2
    flags = chop_flags(days_r, snaps, spy, spy_ma)

    base, bmeta = replay(trades, days_r, close, ma50, flags, spy_ret)
    sm = summarize(base, "meter")   # edge-card bucket: gap <= 1.5 incl below-MA
    sb = summarize(base, "band")    # arms' trigger band: 0 <= gap <= 1.5

    print("=" * 72)
    print(f"CHOP LAB  window {first} → {days_r[-1]}")
    print(f"meter-chop days {sm['chop_n']} (edge-card def, anchor 29) | "
          f"band-chop days {sb['chop_n']} (arms' trigger)")
    print(f"validation: replay end cash ${bmeta['end_cash']:,.0f} "
          f"(DB cash ${port.iloc[-1].cash:,.0f}); "
          f"meter-chop bps/day {sm['chop_bps']:+.1f} "
          f"(bootstrap anchor -32.4)")
    print("-" * 72)
    print(f"BASE excess vs SPY (meter def): chop {sm['chop_$']:+,.0f}$ "
          f"({sm['chop_bps']:+.1f} bps/d) | trend {sm['trend_$']:+,.0f}$ "
          f"({sm['trend_bps']:+.1f} bps/d)")
    print(f"BASE excess vs SPY (band def):  chop {sb['chop_$']:+,.0f}$ "
          f"({sb['chop_bps']:+.1f} bps/d)")

    for key, label in (("meter", "meter-chop"), ("band", "band-chop")):
        bux, orx = defaultdict(float), defaultdict(float)
        for r in base.values():
            if r[key]:
                for k, v in r["buckets"].items():
                    bux[k] += v
                for k, v in r["origins"].items():
                    orx[k] += v
        print(f"\n{label}-day bleed by extension bucket (position vs own 50MA):")
        for k in (">=25%", "10-25%", "<10%", "below_ma", "na"):
            if k in bux:
                print(f"  {k:>9}: {bux[k]:+9,.0f}$")
        print(f"{label}-day bleed by lot origin (regime at add):")
        for k in ("buy_trend", "buy_chop", "pyr_trend", "pyr_chop"):
            if k in orx:
                print(f"  {k:>9}: {orx[k]:+9,.0f}$")

    print("-" * 72)
    for lever in ("trim30", "trim30spy", "pyrgate", "damper"):
        cf, meta = replay(trades, days_r, close, ma50, flags, spy_ret,
                          lever=lever)
        cs = summarize(cf, "meter")
        extra = (f"trims={meta['trims_fired']}"
                 if lever in ("trim30", "trim30spy") else
                 f"pyramids skipped={len(meta['pyr_skipped'])} "
                 f"{meta['pyr_skipped']}" if lever == "pyrgate" else
                 f"buys halved={meta['buys_halved']}")
        print(f"{lever:8} Δchop {cs['chop_$']-sm['chop_$']:+8,.0f}$  "
              f"Δtrend {cs['trend_$']-sm['trend_$']:+8,.0f}$  "
              f"Δtotal {cs['all_$']-sm['all_$']:+8,.0f}$   ({extra})")
    print("=" * 72)
    print("Deltas are counterfactual-minus-actual cumulative excess vs SPY.")
    print("Positive Δtotal = lever would have helped over this live window.")


if __name__ == "__main__":
    main()
