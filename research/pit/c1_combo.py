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
    import a_exposure2 as A2
    expos = {"A1": 2 * (px > px.rolling(200).mean()).astype(float), "A4": A2.a4_exposure(px),
             "A5": A2.a5_exposure(px, tr, rate)}
    idx = book.index
    def simulate(sleeve, ex):
        r, _ = A.run(px.reindex(idx), sleeve.reindex(idx).fillna(0), rate.reindex(idx), ex.reindex(idx).fillna(0))
        return r
    def summarize(name, r):
        eq = (1 + r).cumprod()
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        ex = r - rate.reindex(r.index) / 252
        out = {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "dd": float((1 - eq / eq.cummax()).max()),
               "sharpe": float(ex.mean() / ex.std() * np.sqrt(252)),
               "years": r.groupby(r.index.year).apply(lambda v: (1 + v).prod() - 1)}
        print(f"{name:34s} CAGR {out['cagr']:.2%} | max DD {out['dd']:.1%} | Sharpe {out['sharpe']:.2f}")
        return out
    spy = summarize("SPY TR (buy & hold)", tr.reindex(idx).fillna(0))
    summarize("v5b book (buy & hold)", book)
    res = {}
    for k, ex in expos.items():
        res[k] = (summarize(f"{k} on S&P", simulate(tr, ex)), summarize(f"{k} on v5b book (C-{k})", simulate(book, ex)))
    for k, (a, c) in res.items():
        beat = int((c["years"] > a["years"]).sum())
        g = {"C CAGR > exposure-on-S&P": c["cagr"] > a["cagr"], "C CAGR > SPY TR": c["cagr"] > spy["cagr"],
             "C beats exposure-on-S&P >= 5/8 yrs": beat >= 5, "C DD <= its S&P version + 5pp": c["dd"] <= a["dd"] + 0.05}
        print(f"\nC-{k} vs {k}-on-S&P: " + "  ".join(f"{y}: {c['years'][y]:+.1%} vs {a['years'][y]:+.1%}" for y in c["years"].index))
        print("  " + " | ".join(f"{n}: {'PASS' if v else 'FAIL'}" for n, v in g.items()) + f" => {'PASS' if all(g.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()
