# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""Calibration for the Lab stop rules (docs/exposure-plan.md "Lab stop rules").
Uses ONLY history the Lab never sees (fund prices to 2026-10-07, the 1928-2026 backtest);
run before the first Lab fill so the thresholds can't be fitted to forward results.

1. Real fund tracking vs the backtest's cost model: SSO (2006->) vs 2x S&P TR - (cash+0.5%)
   - 2x0.9% fee; SPY vs 1x; SGOV / BIL vs the 3-month T-bill rate.
2. Break-even costs: the extra trade cost (bps per unit of exposure changed) and the extra
   annual drag on the 2x leg that erase each rule's 1994-2026 edge over S&P TR.
3. Normal bad stretches: each rule's trailing excess vs S&P TR over 63/126/252/756
   sessions, 1928-2026 -- 1st percentile and worst ever.

  python3 lab_stop_calibration.py
"""
import gzip
import json

import numpy as np
import pandas as pd

import a_exposure as A
import a_exposure2 as A2
from common import fmp_get

T = A.T
HORIZONS = (63, 126, 252, 756)


def fund(symbol):
    path = T / f"{symbol}_adj.json.gz"
    if not path.exists():
        rows = fmp_get("historical-price-eod/dividend-adjusted", symbol=symbol, **{"from": "1993-01-01", "to": "2026-10-07"})
        with gzip.open(path, "wt") as fh:
            json.dump(rows, fh)
    rows = json.load(gzip.open(path))
    s = pd.Series({pd.Timestamp(r["date"]): float(r["adjClose"]) for r in rows}).sort_index()
    return s[s.index <= "2026-10-07"]


def ann(x):
    return float(x.mean() * 252)


def tracking(tr, rate):
    print("1. Real funds vs the backtest's cost model (annualized gap = actual - modeled; negative = fund worse)")
    sso, spy = fund("SSO").pct_change().dropna(), fund("SPY").pct_change()
    idx = sso.index.intersection(spy.index)
    r = rate.reindex(idx, method="ffill")
    model2 = 2 * spy[idx] - (r + A.LEV_SPREAD) / 252 - 2 * A.LEV_FEE / 252
    gap = sso[idx] - model2
    by_yr = gap.groupby(gap.index.year).apply(ann)
    print(f"  SSO {idx[0].date()}->{idx[-1].date()}: mean gap {ann(gap):+.2%}/yr | median year {by_yr.median():+.2%} | "
          f"worst year {by_yr.min():+.2%} ({by_yr.idxmin()}) | best {by_yr.max():+.2%}")
    print("  by year: " + " ".join(f"{y}:{v:+.1%}" for y, v in by_yr.items()))
    for lo in ("2009-01-01", "2021-01-01"):
        print(f"  SSO mean gap since {lo[:4]}: {ann(gap[gap.index >= lo]):+.2%}/yr")
    gap = gap[gap.index >= "2009-01-01"]  # 2006-08 partial years / crisis swap terms excluded from the charge
    roll = gap.rolling(126).mean() * 252
    print(f"  trailing-126-session gap: 5th pct {roll.quantile(0.05):+.2%}/yr, worst {roll.min():+.2%}/yr")
    # SPY (the A5 1x leg) vs S&P TR via the index series the backtest uses
    spy_gap = (spy[idx] - tr.reindex(idx).fillna(0) * 1 + A.ETF_FEE / 252)
    print(f"  SPY vs backtest 1x leg (S&P TR - 0.09%): mean gap {ann(spy_gap):+.2%}/yr")
    for sym in ("SGOV", "BIL"):
        c = fund(sym).pct_change().dropna()
        i = c.index[c.index >= "2021-01-01"] if sym == "SGOV" else c.index[c.index >= "2008-01-01"]
        g = c[i] - rate.reindex(i, method="ffill") / 252
        print(f"  {sym} {i[0].date()}->{i[-1].date()} vs 3-mo T-bill: mean gap {ann(g):+.2%}/yr")
    return float(ann(gap[gap.index >= "2021-01-01"]))  # the cost regime the Lab actually trades in


def edge(px, tr, rate, ex, cost=A.TRADE_COST, drag=0.0, period=A.P2):
    e = ex.shift(2).fillna(0).clip(lower=0)
    r, _ = A.run(px, tr, rate, ex)
    r = r - (e.diff().abs().fillna(0) * (cost - A.TRADE_COST)) - np.where(e > 1, drag / 252, 0.0)
    r = pd.Series(r, index=tr.index)
    return A.stats(r, rate, *period)["cagr"] - A.stats(tr, rate, *period)["cagr"]


def breakeven(px, tr, rate, rules):
    print("\n2. Break-even costs (1994-2026 edge over S&P TR -> 0)")
    out = {}
    for name, ex in rules.items():
        base = edge(px, tr, rate, ex)
        lo, hi = A.TRADE_COST, 0.02
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if edge(px, tr, rate, ex, cost=mid) > 0 else (lo, mid)
        cost_be = lo
        lo, hi = 0.0, 0.10
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if edge(px, tr, rate, ex, drag=mid) > 0 else (lo, mid)
        drag_be = lo
        e = ex.shift(2).fillna(0)
        p2 = e[(e.index >= A.P2[0]) & (e.index <= A.P2[1])]
        turn = float(p2.diff().abs().sum() / ((p2.index[-1] - p2.index[0]).days / 365.25))
        out[name] = dict(edge=base, cost_be_bps=cost_be * 1e4, drag_be=drag_be, turnover=turn)
        print(f"  {name}: edge {base:+.2%}/yr | exposure traded {turn:.1f}x/yr | break-even trade cost "
              f"{cost_be * 1e4:.0f} bps (model 5) | break-even extra 2x-leg drag {drag_be:+.2%}/yr")
    return out


def bad_stretches(px, tr, rate, rules):
    print("\n3. Trailing excess vs S&P TR (compounded), 1928-10 -> 2026-10")
    out = {}
    start = pd.Timestamp(A.P1[0])
    for name, ex in rules.items():
        r, _ = A.run(px, tr, rate, ex)
        lr, lb = np.log1p(r[r.index >= start]), np.log1p(tr[tr.index >= start])
        row = {}
        for h in HORIZONS:
            x = np.expm1(lr.rolling(h).sum()) - np.expm1(lb.rolling(h).sum())
            x = x.dropna()
            x2 = x[x.index >= A.P2[0]]
            row[h] = dict(p1=float(x.quantile(0.01)), worst=float(x.min()), when=str(x.idxmin().date()),
                          p2_worst=float(x2.min()), p2_when=str(x2.idxmin().date()), p2_p1=float(x2.quantile(0.01)))
        eq = (1 + r[r.index >= start]).cumprod()
        dd = 1 - eq / eq.cummax()
        row["max_dd"] = float(dd.max())
        row["max_dd_when"] = str(dd.idxmax().date())
        out[name] = row
        print(f"  {name} (max DD {row['max_dd']:.1%} {row['max_dd_when']}):")
        for h in HORIZONS:
            v = row[h]
            print(f"    {h:>3}d: 1928-2026 1st pct {v['p1']:+.1%}, worst {v['worst']:+.1%} ({v['when']}) | "
                  f"1994-2026 1st pct {v['p2_p1']:+.1%}, worst {v['p2_worst']:+.1%} ({v['p2_when']})")
    return out


def main():
    px, tr, rate = A.load()
    rules = {"A1": 2 * (px > px.rolling(200).mean()).astype(float), "A5": A2.a5_exposure(px, tr, rate)}
    gap = tracking(tr, rate)
    be = breakeven(px, tr, rate, rules)
    bs = bad_stretches(px, tr, rate, rules)
    print(f"\n4. Edge after charging the 2021-26 measured SSO gap ({gap:+.2%}/yr) to the 2x leg (1994-2026):")
    for name, ex in rules.items():
        print(f"  {name}: {edge(px, tr, rate, ex, drag=-gap):+.2%}/yr (was {be[name]['edge']:+.2%})")
    json.dump({"sso_gap": gap, "breakeven": be, "bad_stretches": bs},
              open(A.DATA_DIR / "meta" / "lab_stop_calibration.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
