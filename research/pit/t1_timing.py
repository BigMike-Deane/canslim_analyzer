# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Market-timing test (pre-registered in docs/phase2-pit-backtest-plan.md,
"Market-timing test pre-registration"). SPY <-> cash, daily.

T1 = live champion gate: in SPY when SPY close > 50-day SMA.
T2 = scorer's composite M (calculate_index_m_score x MARKET_INDEX_WEIGHTS on
     SPY/QQQ/DIA, renormalized over indexes with >= 200 sessions) >= 7.5/15.
Signal at close t -> position from close t+1. Cash = 3-month T-bill (DTB3).
5 bps per switch. Signals on split-adjusted closes, returns on dividend-adjusted.

Data: DATA_DIR/timing/{SPY,QQQ,DIA}.json.gz (FMP adj + raw), DTB3.csv (FRED).
  python3 t1_timing.py
"""
import gzip
import json

import numpy as np
import pandas as pd

from common import DATA_DIR
from m2_adapter import MARKET_INDEX_WEIGHTS, calculate_index_m_score

TD = DATA_DIR / "timing"
END = "2026-10-05"
COST = 0.0005
PERIODS = {"A 1994-2015": ("1994-01-03", "2015-12-31"),
           "B1 2016-2021": ("2016-01-01", "2021-12-31"),
           "B2 2022-2026": ("2022-01-03", END)}


def load(sym):
    d = json.load(gzip.open(TD / f"{sym}.json.gz", "rt"))
    raw = pd.Series({r["date"]: r["close"] for r in d["raw"]}, dtype=float)
    adj = pd.Series({r["date"]: r["adjClose"] for r in d["adj"]}, dtype=float)
    df = pd.DataFrame({"raw": raw, "adj": adj})
    df.index = pd.to_datetime(df.index)
    return df.sort_index().loc[:END]


def m_composite(px):
    """Composite M (0-15) per date, as data_fetcher/m2_adapter compute it."""
    parts, weights = [], []
    for sym, w in MARKET_INDEX_WEIGHTS.items():
        c = px[sym].raw.dropna()
        ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
        ok = ma200.notna()
        s = pd.Series([calculate_index_m_score(p, a, b) for p, a, b in zip(c[ok], ma50[ok], ma200[ok])],
                      index=c[ok].index)
        parts.append(s * w)
        weights.append(pd.Series(w, index=s.index))
    num = pd.concat(parts, axis=1).sum(axis=1, min_count=1)
    den = pd.concat(weights, axis=1).sum(axis=1, min_count=1)
    return (num / den * 15.0).round(1)


def run(sig, ret, cash, lag):
    pos = sig.shift(lag).fillna(0.0)
    r = pos * ret + (1 - pos) * cash
    switch = pos.diff().abs().fillna(0.0)
    return r - switch * COST, pos, switch


def stats(r, cash):
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (252 / n) - 1
    ex = r - cash
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
    dd = (eq / eq.cummax() - 1).min()
    return cagr, sharpe, dd


def memmel(r1, r2, cash):
    """Jobson-Korkie test of Sharpe(r1) - Sharpe(r2), Memmel (2003) correction."""
    a, b = r1 - cash, r2 - cash
    s1, s2 = a.mean() / a.std(), b.mean() / b.std()
    rho = np.corrcoef(a, b)[0, 1]
    T = len(a)
    var = (2 - 2 * rho + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / T
    return (s1 - s2) / np.sqrt(var)


def main():
    px = {s: load(s) for s in ["SPY", "QQQ", "DIA"]}
    spy = px["SPY"]
    ret = spy.adj.pct_change()
    tb = pd.read_csv(TD / "DTB3.csv", index_col=0, parse_dates=True).iloc[:, 0]
    tb = pd.to_numeric(tb, errors="coerce").reindex(spy.index).ffill() / 100 / 252

    sigs = {
        "T1 SPY>50MA (live gate)": (spy.raw > spy.raw.rolling(50).mean()).astype(float)
                                   .where(spy.raw.rolling(50).mean().notna()),
        "T2 composite M>=7.5": (m_composite(px) >= 7.5).astype(float).reindex(spy.index),
    }
    verdicts = {}
    for name, sig in sigs.items():
        print(f"\n{name}")
        r, pos, sw = run(sig, ret, tb, lag=2)
        r0, _, _ = run(sig, ret, tb, lag=1)
        beat_all, risk_all = True, True
        for pname, (lo, hi) in PERIODS.items():
            sl = slice(lo, hi)
            rr, bh, cc = r.loc[sl].dropna(), ret.loc[sl].dropna(), tb.loc[sl]
            idx = rr.index.intersection(bh.index)
            rr, bh, cc = rr[idx], bh[idx], cc.reindex(idx)
            tc, ts, td = stats(rr, cc)
            bc, bs, bd = stats(bh, cc)
            sc, _, _ = stats(r0.loc[idx], cc)
            yrs = len(idx) / 252
            mo = (rr.groupby(idx.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
                  - bh.groupby(idx.to_period("M")).apply(lambda x: (1 + x).prod() - 1))
            print(f"  {pname}: timed CAGR {tc:+.2%} vs SPY {bc:+.2%} | Sharpe {ts:.2f} vs {bs:.2f} | "
                  f"maxDD {td:.1%} vs {bd:.1%} | invested {pos.loc[idx].mean():.0%} | "
                  f"switches/yr {sw.loc[idx].sum() / yrs:.1f} | worst month vs SPY {mo.min():+.1%} | "
                  f"same-close CAGR {sc:+.2%}")
            beat_all &= tc > bc
            risk_all &= (ts > bs) and (td > bd)
        full = slice("1994-01-03", END)
        z = memmel(r.loc[full].dropna(), ret.loc[full].reindex(r.loc[full].dropna().index), tb.loc[full].reindex(r.loc[full].dropna().index))
        print(f"  Sharpe difference 1994-2026 (Memmel z): {z:+.2f}")
        v = "PASS (beats SPY)" if beat_all else "RISK TOOL (does not meet beat-SPY goal)" if risk_all else "FAIL"
        verdicts[name] = v
        print(f"  => {v}")
    print(f"\nPrimary (T1) verdict: {verdicts['T1 SPY>50MA (live gate)']}")


if __name__ == "__main__":
    main()
