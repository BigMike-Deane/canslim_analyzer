# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""Exposure family 2 (docs/exposure-plan.md): A4 Faber monthly 10-month SMA 2x, A5 graded dual
momentum. Same engine, costs, delay and gates as a_exposure.py.

  python3 a_exposure2.py
"""
import numpy as np
import pandas as pd

import a_exposure as A


def a4_exposure(px):
    me = px.resample("ME").last()
    sig = (me > me.rolling(10).mean()).astype(float) * 2
    # value set on each month's last session, carried through the next month
    last_days = px.groupby(px.index.to_period("M")).apply(lambda s: s.index[-1])
    e = pd.Series(np.nan, index=px.index)
    for per, day in last_days.items():
        v = sig.get(per.to_timestamp("M"), np.nan)
        e.loc[day] = v
    return e.ffill().fillna(0)


def a5_exposure(px, tr, rate):
    s1 = (px > px.rolling(200).mean()).astype(float)
    r12 = (1 + tr).rolling(252).apply(np.prod, raw=True) - 1
    c12 = (rate / 252).rolling(252).sum()
    s2 = (r12 > c12).astype(float)
    return (s1 + s2).fillna(0)


def main():
    px, tr, rate = A.load()
    rules = {"A4 Faber monthly 10-mo SMA 2x": a4_exposure(px), "A5 graded dual momentum": a5_exposure(px, tr, rate),
             "A1 200d 2x (reference)": 2 * (px > px.rolling(200).mean()).astype(float)}
    start = pd.Timestamp(A.P1[0])
    for name, ex in rules.items():
        r, e = A.run(px, tr, rate, ex)
        print(f"\n{name}:")
        ok = True
        for nm, (lo, hi) in (("P1 1928-93", A.P1), ("P2 1994-26", A.P2)):
            s, sb = A.stats(r, rate, lo, hi, e), A.stats(tr, rate, lo, hi)
            beat, dd_ok = s["cagr"] > sb["cagr"], s["dd"] <= sb["dd"]
            ok &= beat and dd_ok
            print(f"  {nm}: CAGR {s['cagr']:.2%} vs {sb['cagr']:.2%} ({'beat' if beat else 'trail'}) | max DD {s['dd']:.1%} vs "
                  f"{sb['dd']:.1%} | Sharpe {s['sharpe']:.2f} vs {sb['sharpe']:.2f} | invested {s['invested']:.0%} | "
                  f"switches/yr {s['switches_yr']:.1f} | worst yr {s['worst_yr']:+.1%}")
        frac, n = A.rolling_beat(r[r.index >= start], tr[tr.index >= start])
        ok &= frac >= 0.60
        print(f"  rolling 10-yr windows beating S&P TR: {frac:.0%} of {n}")
        sens = []
        for d in (1, 3, 5):
            r2, _ = A.run(px, tr, rate, ex, delay=d)
            sens.append(f"delay {d}d: P1 {A.stats(r2, rate, *A.P1)['cagr']:.2%} P2 {A.stats(r2, rate, *A.P2)['cagr']:.2%}")
        print("  sensitivity: " + " | ".join(sens))
        if "reference" not in name:
            print(f"  => {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
