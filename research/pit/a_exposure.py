# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""A-family exposure strategies (docs/exposure-plan.md). Daily S&P total return 1928->,
cash from NBER/TB3MS/DTB3; A1 200-day 2x, A1r 1x, A2 vol target (16%, cap 2x), A3 trend+vol.

  python3 a_exposure.py            # the single pre-registered run (+ reported sensitivities)
"""
import gzip
import json

import numpy as np
import pandas as pd

from common import DATA_DIR

T = DATA_DIR / "timing"
P1, P2 = ("1928-10-01", "1993-12-31"), ("1994-01-03", "2026-10-05")
LEV_SPREAD, LEV_FEE, ETF_FEE, TRADE_COST = 0.005, 0.009, 0.0009, 0.0005


def load():
    g = pd.DataFrame(json.load(gzip.open(T / "GSPC_long.json.gz")))
    px = pd.Series(g.price.to_numpy(), index=pd.to_datetime(g.date)).sort_index()
    # dividends: Shiller monthly trailing-12m D / P -> daily accrual until SPY exists
    sh = pd.read_excel(T / "shiller" / "ie_data.xls", sheet_name="Data", skiprows=7)
    sh = sh[pd.to_numeric(sh.Date, errors="coerce").notna()].copy()
    yr = sh.Date.astype(float).astype(int)
    mo = ((sh.Date.astype(float) - yr) * 100).round().astype(int)
    sh.index = pd.to_datetime(dict(year=yr, month=mo, day=1))
    dy = (pd.to_numeric(sh.D, errors="coerce") / pd.to_numeric(sh.P, errors="coerce")).dropna()
    daily_dy = dy.reindex(px.index, method="ffill")
    tr = px.pct_change() + daily_dy / 252
    spy = json.load(gzip.open(T / "SPY.json.gz"))["adj"]
    spy = pd.Series({pd.Timestamp(r["date"]): r["adjClose"] for r in spy}).sort_index().pct_change()
    cut = pd.Timestamp("1993-02-01")
    tr = pd.concat([tr[tr.index < cut], spy[spy.index >= cut].reindex(px.index[px.index >= cut])])
    # cash rate (annual decimal)
    def fred(name):
        f = pd.read_csv(T / "fred" / f"{name}.csv")
        s = pd.to_numeric(f.iloc[:, 1], errors="coerce")
        return pd.Series(s.to_numpy(), index=pd.to_datetime(f.iloc[:, 0])).dropna() / 100
    nber, tb3ms, dtb3 = fred("NBER_SHORT"), fred("TB3MS"), fred("DTB3")
    rate = pd.concat([nber[nber.index < "1934-01-01"], tb3ms[(tb3ms.index >= "1934-01-01") & (tb3ms.index < "1954-01-01")],
                      dtb3[dtb3.index >= "1954-01-01"]]).sort_index()
    rate = rate[~rate.index.duplicated()].reindex(px.index, method="ffill").fillna(0.03)
    return px, tr.fillna(0), rate


def run(px, tr, rate, exposure, delay=2):
    """exposure: Series of target exposure from the close of day t; it earns day t+delay's return."""
    e = exposure.shift(delay).fillna(0).clip(lower=0)
    cash_d = rate / 252
    fin = np.where(e > 1, (e - 1) * (rate + LEV_SPREAD) / 252 + e * LEV_FEE / 252, 0.0)
    fee = np.where(e <= 1, e * ETF_FEE / 252, 0.0)
    r = e * tr + np.clip(1 - e, 0, None) * cash_d - fin - fee - e.diff().abs().fillna(0) * TRADE_COST
    return pd.Series(r, index=tr.index), e


def stats(r, rate, lo, hi, e=None):
    x = r[(r.index >= lo) & (r.index <= hi)]
    eq = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    ex = x - (rate / 252).reindex(x.index)
    out = {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "dd": float((1 - eq / eq.cummax()).max()),
           "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)), "worst_yr": float(x.groupby(x.index.year).apply(lambda v: (1 + v).prod() - 1).min())}
    if e is not None:
        ee = e[(e.index >= lo) & (e.index <= hi)]
        out["invested"] = float((ee > 0).mean())
        out["switches_yr"] = float(((ee > 0).astype(int).diff().abs() > 0).sum() / yrs)
    return out


def rolling_beat(r, b, years=10):
    eq_r, eq_b = (1 + r).cumprod(), (1 + b).cumprod()
    m_r, m_b = eq_r.resample("ME").last(), eq_b.resample("ME").last()
    n = years * 12
    w = (m_r.shift(-n) / m_r).dropna(), (m_b.shift(-n) / m_b).dropna()
    return float((w[0] > w[1]).mean()), len(w[0])


def main():
    px, tr, rate = load()
    sma200 = px.rolling(200).mean()
    above = (px > sma200).astype(float)
    vol = tr.rolling(21).std() * np.sqrt(252)
    vt = (0.16 / vol).clip(upper=2.0).fillna(0)
    rules = {"A1 200d 2x": 2 * above, "A1r 200d 1x (ref)": above, "A2 vol target 16% (cap 2x)": vt,
             "A3 trend + vol target": vt * above}
    bench = tr
    start = pd.Timestamp(P1[0])
    print("S&P TR buy & hold:")
    for nm, (lo, hi) in (("P1 1928-93", P1), ("P2 1994-26", P2)):
        s = stats(bench, rate, lo, hi)
        print(f"  {nm}: CAGR {s['cagr']:.2%} | max DD {s['dd']:.1%} | Sharpe {s['sharpe']:.2f} | worst yr {s['worst_yr']:+.1%}")
    verdicts = {}
    for name, ex in rules.items():
        r, e = run(px, tr, rate, ex)
        print(f"\n{name}:")
        ok = True
        for nm, (lo, hi) in (("P1 1928-93", P1), ("P2 1994-26", P2)):
            s, sb = stats(r, rate, lo, hi, e), stats(bench, rate, lo, hi)
            beat, dd_ok = s["cagr"] > sb["cagr"], s["dd"] <= sb["dd"]
            ok &= beat and dd_ok
            print(f"  {nm}: CAGR {s['cagr']:.2%} vs {sb['cagr']:.2%} ({'beat' if beat else 'trail'}) | max DD {s['dd']:.1%} vs "
                  f"{sb['dd']:.1%} | Sharpe {s['sharpe']:.2f} vs {sb['sharpe']:.2f} | invested {s['invested']:.0%} | "
                  f"switches/yr {s['switches_yr']:.1f} | worst yr {s['worst_yr']:+.1%}")
        rr, bb = r[r.index >= start], bench[bench.index >= start]
        frac, n = rolling_beat(rr, bb)
        ok &= frac >= 0.60
        print(f"  rolling 10-yr windows beating S&P TR: {frac:.0%} of {n}")
        sens = []
        for d in (1, 3):
            r2, _ = run(px, tr, rate, ex, delay=d)
            sens.append(f"delay {d}d: P1 {stats(r2, rate, *P1)['cagr']:.2%} P2 {stats(r2, rate, *P2)['cagr']:.2%}")
        print("  sensitivity (reporting): " + " | ".join(sens))
        if name.startswith("A1 "):
            for lev in (1.5, 3.0):
                r3, _ = run(px, tr, rate, lev * above)
                print(f"  leverage {lev}x (reporting): P1 {stats(r3, rate, *P1)['cagr']:.2%} P2 {stats(r3, rate, *P2)['cagr']:.2%} "
                      f"| DD P1 {stats(r3, rate, *P1)['dd']:.1%} P2 {stats(r3, rate, *P2)['dd']:.1%}")
        if not name.startswith("A1r"):
            verdicts[name] = ok
    print("\n" + "\n".join(f"=> {k}: {'PASS' if v else 'FAIL'}" for k, v in verdicts.items()))


if __name__ == "__main__":
    main()
