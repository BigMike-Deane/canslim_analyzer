# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v3 walk-forward models + portfolio simulation (docs/score-v3-plan.md).

M1 ridge on per-date centered percentile ranks (penalty picked on the last training
year); M2 HistGradientBoosting (fixed settings) on ranks + market context. Test years
2019..2026: train only on panel dates whose 60-session target window ends before Jan 1
of the test year. Portfolio: every 20 sessions hold top 25 EW, <= 5 per sector, keep a
holding while it ranks in the top 50; 19 bps round trip on turnover; vs SPY TR.

  python3 v3_model.py --placebo   # shuffled target within date: pipeline check, must show ~0 IC
  python3 v3_model.py             # THE single pre-registered run
Output: META_DIR/v3/  (scores, yearly weights, equity, report)
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from common import META_DIR
from p4_tests import nw_t
from v3_assemble import CONTEXT, FEATURES

TEST_YEARS = range(2019, 2027)
ALPHAS = (1, 10, 100, 1000)
TOP, KEEP, SECTOR_MAX, RT_COST = 25, 50, 5, 0.0019
OUT = META_DIR / "v3"


def prep(t, placebo, seed=0):
    t = t[t.y.notna()].copy()
    lo, hi = t.groupby("date").y.transform(lambda v: v.quantile(0.01)), t.groupby("date").y.transform(lambda v: v.quantile(0.99))
    t["yw"] = t.y.clip(lo, hi)
    if placebo:
        rng = np.random.default_rng(seed)
        t["yw"] = t.groupby("date").yw.transform(lambda v: rng.permutation(v.to_numpy()))
    for f in FEATURES:
        t["R_" + f] = t.groupby("date")[f].rank(pct=True) - 0.5
    return t


def window_end(dates, sess):
    """Date on which each panel date's 60-session target window ends."""
    idx = sess.searchsorted(dates)
    return pd.Series(sess[np.minimum(idx + 60, len(sess) - 1)], index=dates)


def ic_by_date(df, col):
    return df.groupby("date").apply(lambda g: g[col].corr(g.yw, method="spearman"))


def fit_year(t, y, ends):
    start = pd.Timestamp(f"{y}-01-01")
    train = t[t.date.map(ends) < start]
    test = t[t.date.dt.year == y]
    X = ["R_" + f for f in FEATURES]
    # M1 ridge: penalty by validation on the last training year (its own embargo)
    vstart = start - pd.DateOffset(years=1)
    fit, val = train[train.date.map(ends) < vstart], train[train.date >= vstart]
    best = max(ALPHAS, key=lambda a: ic_by_date(
        val.assign(p=Ridge(alpha=a).fit(fit[X].fillna(0), fit.yw).predict(val[X].fillna(0))), "p").mean())
    ridge = Ridge(alpha=best).fit(train[X].fillna(0), train.yw)
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                        min_samples_leaf=200, random_state=0).fit(train[X + CONTEXT], train.yw)
    out = test[["cik", "date", "symbol", "sector", "yw", "y", "r20", "r60", "spy20_tr"]].copy()
    out["m1"] = ridge.predict(test[X].fillna(0))
    out["m2"] = gbm.predict(test[X + CONTEXT])
    w = pd.Series(ridge.coef_, index=FEATURES).rename(y)
    return out, w, best, len(train)


def portfolio(scores, col, sess):
    dates = sorted(scores.date.unique())[::2]  # every 20 sessions
    held, eq, rows, turn = [], 1.0, [], []
    spy = 1.0
    for d in dates:
        g = scores[scores.date == d].sort_values(col, ascending=False)
        g["rank"] = np.arange(1, len(g) + 1)
        keep = [c for c in held if c in set(g.cik[g["rank"] <= KEEP])]
        picks, per_sector = list(keep), g[g.cik.isin(keep)].sector.value_counts().to_dict()
        for r in g.itertuples():
            if len(picks) >= TOP:
                break
            if r.cik in picks or per_sector.get(r.sector, 0) >= SECTOR_MAX:
                continue
            picks.append(r.cik)
            per_sector[r.sector] = per_sector.get(r.sector, 0) + 1
        to = 1 - len(set(picks) & set(held)) / TOP if held else 1.0
        rets = g.set_index("cik").r20.reindex(picks)
        port = rets.fillna(0).mean() - to * RT_COST
        eq *= 1 + port
        spy *= 1 + g.spy20_tr.iloc[0]
        rows.append({"date": d, "equity": eq, "spy": spy, "ret": port, "spy_ret": g.spy20_tr.iloc[0],
                     "turnover": to, "missing_ret": int(rets.isna().sum())})
        turn.append(to)
        held = picks
    e = pd.DataFrame(rows).set_index("date")
    return e


def stats(e):
    yrs = (e.index[-1] - e.index[0]).days / 365.25 + 20 / 252
    cagr = e.equity.iloc[-1] ** (1 / yrs) - 1
    scagr = e.spy.iloc[-1] ** (1 / yrs) - 1
    dd = (1 - e.equity / e.equity.cummax()).max()
    sdd = (1 - e.spy / e.spy.cummax()).max()
    by_year = e.groupby(e.index.year).apply(lambda g: pd.Series({
        "port": (1 + g.ret).prod() - 1, "spy": (1 + g.spy_ret).prod() - 1}))
    return {"cagr": cagr, "spy_cagr": scagr, "max_dd": dd, "spy_max_dd": sdd,
            "years_beat": int((by_year.port > by_year.spy).sum()), "n_years": len(by_year),
            "turnover": e.turnover.mean(), "by_year": by_year}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luck-runs", type=int, default=200)
    a = ap.parse_args()
    tag = f"placebo{a.seed}" if a.placebo else "real"
    OUT.mkdir(exist_ok=True)
    t = prep(pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"]), a.placebo, a.seed)
    from v3_assemble import spy_tr
    sess = spy_tr().index
    ends = window_end(pd.DatetimeIndex(sorted(t.date.unique())), sess)
    parts, weights = [], []
    for y in TEST_YEARS:
        out, w, alpha, n = fit_year(t, y, ends)
        parts.append(out)
        weights.append(w)
        print(f"{y}: train {n:,} rows, ridge alpha {alpha}, test {len(out):,} rows", flush=True)
    s = pd.concat(parts)
    s.to_csv(OUT / f"scores_{tag}.csv.gz", index=False)
    W = pd.DataFrame(weights).T
    W.to_csv(OUT / f"ridge_weights_{tag}.csv")

    print(f"\n=== Score v3 {'PLACEBO (shuffled target)' if a.placebo else 'pre-registered run'} ===")
    res = {}
    for col, name in (("m1", "M1 ridge"), ("m2", "M2 boosted trees")):
        ic = ic_by_date(s, col)
        e = portfolio(s, col, sess)
        st = stats(e)
        e.to_csv(OUT / f"equity_{tag}_{col}.csv")
        h1, h2 = ic[ic.index.year <= 2022], ic[ic.index.year >= 2023]
        res[col] = {"ic": ic.mean(), "ic_t": nw_t(ic), "ic_h1": h1.mean(), "ic_h2": h2.mean(), **{k: v for k, v in st.items() if k != "by_year"}}
        print(f"\n{name}: mean IC {ic.mean():+.4f} (NW t {nw_t(ic):+.2f}) | 2019-22 {h1.mean():+.4f} | 2023-26 {h2.mean():+.4f}")
        print(f"  portfolio CAGR {st['cagr']:.1%} vs SPY TR {st['spy_cagr']:.1%} | max DD {st['max_dd']:.1%} vs {st['spy_max_dd']:.1%} | "
              f"beat SPY {st['years_beat']}/{st['n_years']} yrs | turnover/rebalance {st['turnover']:.0%} | "
              f"missing returns {int(e.missing_ret.sum())}")
        print("  " + "  ".join(f"{y}: {r.port:+.1%} vs {r.spy:+.1%}" for y, r in st["by_year"].iterrows()))
    # amendment 1: luck band -- the same portfolio rules on random scores
    rng = np.random.default_rng(12345)
    luck = []
    for _ in range(a.luck_runs if not a.placebo else 0):
        luck.append(stats(portfolio(s.assign(rnd=rng.random(len(s))), "rnd", sess))["cagr"])
    if luck:
        luck = np.array(luck)
        print(f"\nluck band ({len(luck)} random-score portfolios, same rules): CAGR 5th {np.percentile(luck, 5):.1%} | "
              f"median {np.median(luck):.1%} | 95th {np.percentile(luck, 95):.1%}")
        for k in res:
            res[k]["luck_pctile"] = float((luck < res[k]["cagr"]).mean() * 100)
            print(f"  {k} portfolio CAGR {res[k]['cagr']:.1%} = {res[k]['luck_pctile']:.0f}th percentile of random")
    cand = max(res, key=lambda k: res[k]["ic"])
    r = res[cand]
    gates = {"1 signal (t>=3, both halves >0)": r["ic_t"] >= 3 and r["ic_h1"] > 0 and r["ic_h2"] > 0,
             "2 CAGR > SPY TR": r["cagr"] > r["spy_cagr"],
             "3 beat SPY >= 5 of 8 yrs": r["years_beat"] >= 5,
             "4 max DD <= SPY + 10pp": r["max_dd"] <= r["spy_max_dd"] + 0.10}
    print(f"\ncandidate (higher mean IC): {cand}")
    for k, v in gates.items():
        print(f"  gate {k}: {'PASS' if v else 'FAIL'}")
    verdict = all(gates.values())
    print(f"\n=> Score v3 {'(placebo) ' if a.placebo else ''}{'PASS -> 3-month forward paper trading' if verdict else 'FAIL'}")
    print("\nridge weights by test year (on centered ranks; + = higher rank -> higher expected excess):")
    print((W * 100).round(2).to_string())
    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()} | {"candidate": cand, "pass": verdict},
              open(OUT / f"report_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
