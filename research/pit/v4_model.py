# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v4 (docs/score-v3-plan.md, "Score v4 pre-registration").

Per test year Y (2019..2026), on training dates only (60-session embargo): per-date
Spearman IC of each of the 38 v3 features vs winsorized 60-session excess, inside the
500 largest names. Keep a feature if NW t >= 2 and its mean IC has the same sign in
both halves of the training dates; score = mean of kept features' signed centered
ranks. Portfolio: top 30, weights ~ sqrt(mcap) capped 8%, <= 6 per sector, keep while
in top 60, 19 bps on weight turnover, total return; vs SPY TR.

  python3 v4_model.py --placebo [--seed N]   # shuffled target: pipeline check
  python3 v4_model.py                        # THE single pre-registered run
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import META_DIR
from p4_tests import nw_t
from v3_assemble import FEATURES, spy_tr
from v3_model import OUT, TEST_YEARS, ic_by_date, prep, stats, window_end

TOP_UNIV, TOP, KEEP, SECTOR_MAX, CAP_W, RT_COST = 500, 30, 60, 6, 0.08, 0.0019


def select(train):
    dates = sorted(train.date.unique())
    half = dates[len(dates) // 2]
    kept = {}
    for f in FEATURES:
        ic = ic_by_date(train[train[f].notna()], "R_" + f).dropna()
        if len(ic) < 10:
            continue
        t, a, b = nw_t(ic), ic[ic.index < half].mean(), ic[ic.index >= half].mean()
        if abs(t) >= 2.0 and np.sign(a) == np.sign(b) == np.sign(ic.mean()):
            kept[f] = (float(np.sign(ic.mean())), float(t))
    return kept


def score(df, kept):
    if not kept:
        return pd.Series(np.nan, index=df.index)
    return sum(s * df["R_" + f].fillna(0) for f, (s, _) in kept.items()) / len(kept)


def portfolio(scores, col):
    dates = sorted(scores.date.unique())[::2]
    held, eq, spy, rows = {}, 1.0, 1.0, []
    for d in dates:
        g = scores[scores.date == d].sort_values(col, ascending=False)
        if g[col].isna().all():  # no feature kept: cap-weighted universe
            w = (g.mcap / g.mcap.sum()).set_axis(g.cik)
        else:
            g = g.assign(rank=np.arange(1, len(g) + 1))
            keep = [c for c in held if c in set(g.cik[g["rank"] <= KEEP])]
            picks, per_sector = list(keep), g[g.cik.isin(keep)].sector.value_counts().to_dict()
            for r in g.itertuples():
                if len(picks) >= TOP:
                    break
                if r.cik in picks or per_sector.get(r.sector, 0) >= SECTOR_MAX:
                    continue
                picks.append(r.cik)
                per_sector[r.sector] = per_sector.get(r.sector, 0) + 1
            m = g.set_index("cik").mcap.reindex(picks)
            w = np.sqrt(m) / np.sqrt(m).sum()
            for _ in range(10):  # cap at CAP_W, redistribute
                over = w > CAP_W
                if not over.any():
                    break
                w[over] = CAP_W
                w[~over] = w[~over] / w[~over].sum() * (1 - CAP_W * over.sum())
        to = 0.5 * sum(abs(w.get(c, 0) - held.get(c, 0)) for c in set(w.index) | set(held)) if held else 1.0
        rets = g.set_index("cik").r20.reindex(w.index).fillna(0)
        port = float((w * rets).sum()) - to * RT_COST
        sr = g.spy20_tr.iloc[0]
        eq *= 1 + port
        spy *= 1 + sr
        rows.append({"date": d, "equity": eq, "spy": spy, "ret": port, "spy_ret": sr, "turnover": to,
                     "missing_ret": 0, "n": len(w)})
        held = w.to_dict()
    return pd.DataFrame(rows).set_index("date")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luck-runs", type=int, default=200)
    a = ap.parse_args()
    tag = f"v4_placebo{a.seed}" if a.placebo else "v4_real"
    OUT.mkdir(exist_ok=True)
    full = prep(pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"]), a.placebo, a.seed)
    t = full[full.groupby("date").mcap.rank(ascending=False, method="first") <= TOP_UNIV].copy()
    sess = spy_tr().index
    ends = window_end(pd.DatetimeIndex(sorted(t.date.unique())), sess)
    parts, kept_log = [], {}
    for y in TEST_YEARS:
        start = pd.Timestamp(f"{y}-01-01")
        kept = select(t[t.date.map(ends) < start])
        kept_log[y] = kept
        test = t[t.date.dt.year == y].copy()
        test["v4"] = score(test, kept)
        parts.append(test[["cik", "date", "symbol", "sector", "mcap", "yw", "y", "r20", "r60", "spy20_tr", "v4"]])
        print(f"{y}: kept {len(kept)}: " + ", ".join(f"{'+' if s > 0 else '-'}{f}({tt:+.1f})" for f, (s, tt) in
                                                      sorted(kept.items(), key=lambda kv: -abs(kv[1][1]))), flush=True)
    s = pd.concat(parts)
    s.to_csv(OUT / f"scores_{tag}.csv.gz", index=False)

    print(f"\n=== Score v4 {'PLACEBO (shuffled target)' if a.placebo else 'pre-registered run'} ===")
    ic = ic_by_date(s, "v4")
    h1, h2 = ic[ic.index.year <= 2022], ic[ic.index.year >= 2023]
    e = portfolio(s, "v4")
    e.to_csv(OUT / f"equity_{tag}.csv")
    st = stats(e)
    print(f"500 largest: mean OOS IC {ic.mean():+.4f} (NW t {nw_t(ic):+.2f}) | 2019-22 {h1.mean():+.4f} | 2023-26 {h2.mean():+.4f}")
    print(f"  portfolio CAGR {st['cagr']:.1%} vs SPY TR {st['spy_cagr']:.1%} | max DD {st['max_dd']:.1%} vs {st['spy_max_dd']:.1%} | "
          f"beat SPY {st['years_beat']}/{st['n_years']} yrs | turnover/rebalance {st['turnover']:.0%}")
    print("  " + "  ".join(f"{y}: {r.port:+.1%} vs {r.spy:+.1%}" for y, r in st["by_year"].iterrows()))
    res = {"ic": ic.mean(), "ic_t": nw_t(ic), "ic_h1": h1.mean(), "ic_h2": h2.mean(),
           **{k: v for k, v in st.items() if k != "by_year"}}
    if not a.placebo:
        rng = np.random.default_rng(12345)
        luck = np.array([stats(portfolio(s.assign(rnd=rng.random(len(s))), "rnd"))["cagr"] for _ in range(a.luck_runs)])
        res["luck_pctile"] = float((luck < st["cagr"]).mean() * 100)
        print(f"\nluck band ({len(luck)} random-score portfolios, same rules, 500 largest): 5th {np.percentile(luck, 5):.1%} | "
              f"median {np.median(luck):.1%} | 95th {np.percentile(luck, 95):.1%} -> v4 at {res['luck_pctile']:.0f}th percentile")
        # reporting only: the same yearly selections scored on the full v3 universe
        fparts = []
        for y in TEST_YEARS:
            ft = full[full.date.dt.year == y].copy()
            ft["v4"] = score(ft, kept_log[y])
            fparts.append(ft)
        fic = ic_by_date(pd.concat(fparts), "v4")
        print(f"reporting: full v3 universe IC {fic.mean():+.4f} (NW t {nw_t(fic):+.2f})")
    gates = {"1 signal (t>=3, both halves >0)": res["ic_t"] >= 3 and res["ic_h1"] > 0 and res["ic_h2"] > 0,
             "2 CAGR > SPY TR": res["cagr"] > res["spy_cagr"],
             "3 beat SPY >= 5 of 8 yrs": res["years_beat"] >= 5,
             "4 max DD <= SPY + 10pp": res["max_dd"] <= res["spy_max_dd"] + 0.10}
    for k, v in gates.items():
        print(f"  gate {k}: {'PASS' if v else 'FAIL'}")
    verdict = all(gates.values())
    print(f"\n=> Score v4 {'(placebo) ' if a.placebo else ''}{'PASS -> candidate for forward paper' if verdict else 'FAIL'}")
    json.dump({**{k: float(v) for k, v in res.items()}, "pass": verdict,
               "kept": {str(y): {f: v for f, v in k.items()} for y, k in kept_log.items()}},
              open(OUT / f"report_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
