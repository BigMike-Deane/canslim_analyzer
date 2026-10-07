# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false
"""C1 (docs/exposure-plan.md): A1's daily exposure applied to the v5b tilted book.

Daily book returns: each rebalance's v5b weights (every 20 sessions) drift with the
stocks' daily split-adjusted prices until the next rebalance; each name's dividends over
the period (v3_dividends div20) are credited on the period's last day. A1 exposure,
financing and costs exactly as a_exposure.run. Compared with A1 (on the S&P) and SPY TR
over the same 2019-01 -> 2026-09 dates.

  python3 c1_combo.py      # after v5_model.py has written scores_v5_real.csv.gz on the corrected data
"""
import numpy as np
import pandas as pd

import a_exposure as A
import m2_adapter as m
from common import META_DIR
from v3_model import OUT


def tilt_w(g):
    u = g.v5b.rank(pct=True)
    mult = (1 + (2 * u - 1)).fillna(1.0)
    w = g.mcap * mult
    return (w / w.sum()).to_numpy()


def book_daily(scores, sess):
    dates = sorted(scores.date.unique())[::2]
    div = pd.read_csv(META_DIR / "v3_dividends.csv.gz", parse_dates=["date"]).set_index(["cik", "date"]).div20
    px_cache, out, cost = {}, [], {}
    prev_w = None
    for j, d in enumerate(dates):
        end = dates[j + 1] if j + 1 < len(dates) else sess[min(sess.searchsorted(d) + 20, len(sess) - 1)]
        g = scores[scores.date == d]
        w = pd.Series(tilt_w(g), index=g.cik.to_numpy())
        days = sess[(sess > d) & (sess <= end)]
        rel = []
        for cik in w.index:
            if cik not in px_cache:
                p = m.cik_prices(cik)
                px_cache[cik] = p.Close if not p.empty else pd.Series(dtype=float)
            p = px_cache[cik]
            seg = p.reindex(days.insert(0, d)).ffill()
            if seg.isna().iloc[0]:
                seg = pd.Series(1.0, index=seg.index)
            rel.append((seg / seg.iloc[0]).fillna(1.0).to_numpy())
        rel = np.array(rel)                                   # names x (days+1)
        val = (w.to_numpy()[:, None] * rel).sum(axis=0)        # drifting book value, starts at 1
        r = val[1:] / val[:-1] - 1
        dv = sum(w.get(c, 0) * div.get((c, d), 0) for c in w.index)
        r[-1] += dv / val[-2] if len(r) else 0
        to = 0.5 * w.sub(prev_w, fill_value=0).abs().sum() if prev_w is not None else 0.0
        if len(r):
            r[0] -= to * 0.0019
        out.append(pd.Series(r, index=days))
        prev_w = w
        cost[d] = to
    return pd.concat(out)


def main():
    px, tr, rate = A.load()
    sess = tr.index
    scores = pd.read_csv(OUT / "scores_v5_real.csv.gz", parse_dates=["date"])
    lo, hi = scores.date.min(), pd.Timestamp("2026-09-30")
    book = book_daily(scores, sess)
    book = book[(book.index > lo) & (book.index <= hi)]
    above = (px > px.rolling(200).mean()).astype(float)
    res = {}
    for name, sleeve, lev in (("SPY TR (buy & hold)", tr, None), ("A1: 2x S&P / cash", tr, 2.0),
                              ("C1: 2x v5b book / cash", book, 2.0), ("C1 1x (reference)", book, 1.0),
                              ("v5b book (buy & hold)", book, None)):
        sl = sleeve.reindex(book.index).fillna(0)
        if lev is None:
            r = sl
        else:
            r, _ = A.run(px.reindex(book.index), sl, rate.reindex(book.index), (lev * above).reindex(book.index).fillna(0))
        eq = (1 + r).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        ex = r - rate.reindex(r.index) / 252
        res[name] = {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "dd": float((1 - eq / eq.cummax()).max()),
                     "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)),
                     "years": r.groupby(r.index.year).apply(lambda v: (1 + v).prod() - 1)}
        print(f"{name:26s} CAGR {res[name]['cagr']:.2%} | max DD {res[name]['dd']:.1%} | Sharpe {res[name]['sharpe']:.2f}")
    a1, c1 = res["A1: 2x S&P / cash"], res["C1: 2x v5b book / cash"]
    beat_years = int((c1["years"] > a1["years"]).sum())
    print("\nby year (C1 vs A1): " + "  ".join(f"{y}: {c1['years'][y]:+.1%} vs {a1['years'][y]:+.1%}" for y in c1["years"].index))
    gates = {"1 C1 CAGR > A1": c1["cagr"] > a1["cagr"], "2 C1 CAGR > SPY TR": c1["cagr"] > res["SPY TR (buy & hold)"]["cagr"],
             "3 C1 beats A1 >= 5/8 yrs": beat_years >= 5, "4 C1 DD <= A1 DD + 5pp": c1["dd"] <= a1["dd"] + 0.05}
    for k, v in gates.items():
        print(f"  gate {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n=> C1 {'PASS' if all(gates.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
