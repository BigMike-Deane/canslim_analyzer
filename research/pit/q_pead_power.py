# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Power check for a one-season forward test of post-earnings drift (Oct-9, EXPLORATORY).

Question: if we watch ONE earnings season live (fresh data), how often would the
beat-vs-miss drift show up clearly? Measured on every season 2016-2026 with the same
universe as Score v3 (close > $5, mcap >= $1B, 20d $ volume >= $5M at the nearest
panel date <= report date). Nothing here is a verdict; it only sizes the live test.

Event: FMP earnings row with epsActual and epsEstimated. Entry = close of the first
session AFTER the report date (safe for before-open and after-close reports; known
live by then). Excess = stock close-to-close return minus SPY total return, same dates.

  python3 q_pead_power.py [--workers W]
Output: META_DIR/q_pead_events.csv.gz + printed per-season table.
"""
import argparse
import json
import multiprocessing as mp

import numpy as np
import pandas as pd

from common import FMP_DIR, META_DIR, load_prices
from v3_assemble import spy_tr

HORIZONS = (20, 60)
SPY = None
UNIV = None


def _init():
    global SPY, UNIV
    SPY = spy_tr()
    UNIV = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date", "symbol", "sector", "mcap"],
                       parse_dates=["date"]).sort_values("date")


def events_for(sym):
    try:
        data = json.load(open(FMP_DIR / "earnings" / f"{sym}.json")) or []
    except (OSError, ValueError):
        return []
    rows = sorted((x for x in data if x.get("date") and x.get("epsActual") is not None
                   and x.get("epsEstimated") is not None), key=lambda x: x["date"])
    if not rows:
        return []
    u = UNIV[UNIV.symbol == sym]
    if u.empty:
        return []
    px = load_prices(sym)
    if px.empty:
        return []
    close = px.Close
    out, streak = [], 0
    for x in rows:
        a, e = float(x["epsActual"]), float(x["epsEstimated"])
        streak = streak + 1 if a > e else 0
        d = pd.Timestamp(x["date"])
        if d < pd.Timestamp("2016-01-15"):
            continue
        m = u[(u.date <= d) & (u.date >= d - pd.Timedelta(days=21))]
        if m.empty:
            continue
        r = m.iloc[-1]
        i = close.index.searchsorted(d, side="right")       # first session after the report date
        if i >= len(close):
            continue
        rec = {"symbol": sym, "cik": r.cik, "sector": r.sector, "mcap": r.mcap, "date": d,
               "entry": close.index[i], "beat": a > e, "streak": streak,
               "surprise": max(-200.0, min(200.0, (a - e) / abs(e) * 100)) if e != 0 else np.nan}
        for h in HORIZONS:
            if i + h < len(close):
                t0, t1 = close.index[i], close.index[i + h]
                s0, s1 = SPY.asof(t0), SPY.asof(t1)
                rec[f"x{h}"] = close.iloc[i + h] / close.iloc[i] - 1 - (s1 / s0 - 1)
        out.append(rec)
    return out


def season_table(ev, h):
    col = f"x{h}"
    d = ev[ev[col].notna()].copy()
    lo, hi = d[col].quantile(0.005), d[col].quantile(0.995)
    d[col] = d[col].clip(lo, hi)                                # a few glitch prints only
    d["season"] = d.date.dt.to_period("Q")
    rows = []
    for s, g in d.groupby("season"):
        if len(g) < 200:
            continue
        q = g.surprise.rank(pct=True)
        top, bot = g[q > 0.8][col], g[q <= 0.2][col]
        beat, miss = g[g.beat][col], g[~g.beat][col]
        streak, = (g[g.streak >= 4][col],)
        rows.append({"season": str(s), "n": len(g), "quint_spread": top.mean() - bot.mean(),
                     "beat_miss": beat.mean() - miss.mean(), "streak4_miss": streak.mean() - miss.mean(),
                     "naive_t": (top.mean() - bot.mean()) / np.sqrt(top.var() / len(top) + bot.var() / len(bot))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    _init()
    syms = sorted(UNIV.symbol.dropna().unique())
    with mp.Pool(a.workers, initializer=_init) as pool:
        ev = pd.DataFrame([r for rs in pool.imap_unordered(events_for, syms, chunksize=20) for r in rs])
    ev.to_csv(META_DIR / "q_pead_events.csv.gz", index=False)
    print(f"{len(ev):,} events, {ev.cik.nunique():,} companies, {ev.date.min().date()} .. {ev.date.max().date()}")
    for h in HORIZONS:
        t = season_table(ev, h)
        print(f"\n=== {h}-session drift after the report (excess vs SPY TR), per season ===")
        print(t.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        for c in ("quint_spread", "beat_miss", "streak4_miss"):
            m, s = t[c].mean(), t[c].std()
            print(f"{c:13s}: mean {m:+.2%} | sd across seasons {s:.2%} | single-season z {m / s:+.2f} | "
                  f"seasons > 0: {(t[c] > 0).mean():.0%} | > half the mean: {(t[c] > m / 2).mean():.0%} "
                  f"| 2016-20 {t[t.season < '2021'][c].mean():+.2%} vs 2021-26 {t[t.season >= '2021'][c].mean():+.2%}")


if __name__ == "__main__":
    main()
