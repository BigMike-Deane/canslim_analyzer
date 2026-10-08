# Diagnostic (2026-10-08): is the A1/A5 edge timing or just leverage? Compares each rule with constant leverage
# at the same average exposure and at the same volatility. Reporting only, no gate.  python3 a_constant_leverage.py
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import a_exposure as A, a_exposure2 as A2
px, tr, rate = A.load()
rules = {"A1": 2 * (px > px.rolling(200).mean()).astype(float), "A5": A2.a5_exposure(px, tr, rate)}
for per_name, per in (("P1 1928-93", A.P1), ("P2 1994-26", A.P2), ("2009-26", ("2009-01-02", "2026-10-05"))):
    print(f"\n{per_name}: S&P TR {A.stats(tr, rate, *per)['cagr']:.2%}  DD {A.stats(tr, rate, *per)['dd']:.1%}")
    for name, ex in rules.items():
        r, e = A.run(px, tr, rate, ex)
        s = A.stats(r, rate, *per)
        ee = e[(e.index >= per[0]) & (e.index <= per[1])]
        xr = r[(r.index >= per[0]) & (r.index <= per[1])]
        # constant leverage with the SAME average exposure, and with the SAME volatility
        avg = float(ee.mean())
        rc, _ = A.run(px, tr, rate, pd.Series(avg, index=px.index))
        sc = A.stats(rc, rate, *per)
        vol_t = xr.std()
        lo, hi = 0.1, 3.0
        for _ in range(30):
            m = (lo + hi) / 2
            rv, _ = A.run(px, tr, rate, pd.Series(m, index=px.index))
            v = rv[(rv.index >= per[0]) & (rv.index <= per[1])].std()
            lo, hi = (m, hi) if v < vol_t else (lo, m)
        rv, _ = A.run(px, tr, rate, pd.Series(lo, index=px.index)); sv = A.stats(rv, rate, *per)
        print(f"  {name}: {s['cagr']:.2%} DD {s['dd']:.1%} Sharpe {s['sharpe']:.2f} | constant {avg:.2f}x (same avg exposure): "
              f"{sc['cagr']:.2%} DD {sc['dd']:.1%} | constant {lo:.2f}x (same volatility): {sv['cagr']:.2%} DD {sv['dd']:.1%} Sharpe {sv['sharpe']:.2f}")
