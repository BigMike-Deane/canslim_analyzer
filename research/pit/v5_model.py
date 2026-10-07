# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v5 — SPY-relative tilt (docs/score-v3-plan.md, "Trial ledger + Score v5").

Weights w ~ cap x m, m = 1 + (2u - 1) with u = score percentile in the 500 largest
(0x .. 2x; missing score 1x). Rebalance every 20 sessions, 19 bps per unit one-way
turnover, total return. Scores: v5a (v4 selection in the 500 largest), v5b (selection
on the full universe, applied to the 500 largest), v5c (fixed in-sample composite).

  python3 v5_model.py --placebo [--seed N]
  python3 v5_model.py
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import META_DIR
from p4_tests import nw_t
from v3_assemble import spy_tr
from v3_model import OUT, TEST_YEARS, ic_by_date, prep, stats, window_end
from v4_model import score, select

TOP_UNIV, RT_COST, N_TRIALS = 500, 0.0019, 6
V5C = {"beat_streak": 1.0, "surprise_pct": 1.0, "s1": 1.0, "s3": 1.0, "roe": 1.0, "dtc": -1.0}


def tilt(scores, col):
    dates = sorted(scores.date.unique())[::2]
    prev, eq, spy, rows = None, 1.0, 1.0, []
    for d in dates:
        g = scores[scores.date == d]
        u = g[col].rank(pct=True)
        m = (1 + (2 * u - 1)).fillna(1.0)
        w = (g.mcap * m) / (g.mcap * m).sum()
        w.index = g.cik
        to = 0.5 * w.sub(prev, fill_value=0).abs().sum() if prev is not None else 0.0
        port = float((w * g.r20.fillna(0).to_numpy()).sum()) - to * RT_COST
        sr = g.spy20_tr.iloc[0]
        eq *= 1 + port
        spy *= 1 + sr
        rows.append({"date": d, "equity": eq, "spy": spy, "ret": port, "spy_ret": sr, "turnover": to, "missing_ret": 0})
        prev = w
    return pd.DataFrame(rows).set_index("date")


def report(name, s, col, e):
    st = stats(e)
    ic = ic_by_date(s[s[col].notna()], col)
    act = e.ret - e.spy_ret
    te = act.std() * np.sqrt(12.6)
    ir = act.mean() * 12.6 / te if te > 0 else np.nan
    print(f"\n{name}: OOS IC (500 largest) {ic.mean():+.4f} (NW t {nw_t(ic):+.2f}) | CAGR {st['cagr']:.2%} vs SPY TR "
          f"{st['spy_cagr']:.2%} (active {st['cagr'] - st['spy_cagr']:+.2%}) | TE {te:.1%} IR {ir:+.2f} | "
          f"DD {st['max_dd']:.1%} vs {st['spy_max_dd']:.1%} | beat {st['years_beat']}/{st['n_years']} yrs | turnover {st['turnover']:.0%}")
    print("  " + "  ".join(f"{y}: {r.port:+.1%} vs {r.spy:+.1%}" for y, r in st["by_year"].iterrows()))
    return {"ic": ic.mean(), "ic_t": nw_t(ic), "te": te, "ir": ir, "active": st["cagr"] - st["spy_cagr"],
            **{k: v for k, v in st.items() if k != "by_year"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luck-runs", type=int, default=300)
    a = ap.parse_args()
    tag = f"v5_placebo{a.seed}" if a.placebo else "v5_real"
    full = prep(pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"]), a.placebo, a.seed)
    full["big"] = full.groupby("date").mcap.rank(ascending=False, method="first") <= TOP_UNIV
    big = full[full.big].copy()
    sess = spy_tr().index
    ends_b = window_end(pd.DatetimeIndex(sorted(big.date.unique())), sess)
    ends_f = window_end(pd.DatetimeIndex(sorted(full.date.unique())), sess)
    parts, kept = [], {}
    for y in TEST_YEARS:
        start = pd.Timestamp(f"{y}-01-01")
        ka = select(big[big.date.map(ends_b) < start])
        kb = select(full[full.date.map(ends_f) < start])
        kept[y] = {"v5a": ka, "v5b": kb}
        test = big[big.date.dt.year == y].copy()
        test["v5a"], test["v5b"] = score(test, ka), score(test, kb)
        test["v5c"] = sum(sgn * test["R_" + f].fillna(0) for f, sgn in V5C.items()) / len(V5C)
        parts.append(test)
        print(f"{y}: v5a kept {sorted(ka)} | v5b kept {sorted(kb)}", flush=True)
    s = pd.concat(parts)
    s[["cik", "date", "symbol", "mcap", "yw", "r20", "spy20_tr", "v5a", "v5b", "v5c"]].to_csv(OUT / f"scores_{tag}.csv.gz", index=False)

    print(f"\n=== Score v5 tilt {'PLACEBO' if a.placebo else 'pre-registered run'} ===")
    bench = tilt(s.assign(none=np.nan), "none")
    bst = stats(bench)
    print(f"benchmark (cap-weighted 500 largest): CAGR {bst['cagr']:.2%} vs SPY TR {bst['spy_cagr']:.2%}")
    res = {}
    for col, name in (("v5a", "v5a selection in 500 largest"), ("v5b", "v5b selection on full universe"),
                      ("v5c", "v5c fixed composite (IN-SAMPLE, cannot pass)")):
        e = tilt(s, col)
        e.to_csv(OUT / f"equity_{tag}_{col}.csv")
        res[col] = report(name, s, col, e)
    if a.placebo:
        return
    rng = np.random.default_rng(777)
    luck = np.array([stats(tilt(s.assign(rnd=rng.random(len(s))), "rnd"))["cagr"] for _ in range(a.luck_runs)]) - bst["spy_cagr"]
    best_of = np.array([rng.choice(luck, N_TRIALS, replace=False).max() for _ in range(2000)])
    bar = float(np.percentile(best_of, 95))
    print(f"\nluck: random-score tilts active return 5th {np.percentile(luck, 5):+.2%} | median {np.median(luck):+.2%} | "
          f"95th {np.percentile(luck, 95):+.2%}; best-of-{N_TRIALS} 95th = {bar:+.2%} (gate 3 bar)")
    verdicts = {}
    for col in ("v5a", "v5b"):
        r = res[col]
        g = {"1 CAGR > SPY TR": r["active"] > 0, "2 beat >= 5/8 yrs": r["years_beat"] >= 5,
             "3 active > luck bar": r["active"] > bar, "4 DD <= SPY+10pp": r["max_dd"] <= r["spy_max_dd"] + 0.10}
        verdicts[col] = all(g.values())
        print(f"{col}: " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in g.items()) +
              f" => {'PASS' if verdicts[col] else 'FAIL'}")
    print(f"v5c (in-sample, reporting only): active {res['v5c']['active']:+.2%} vs bar {bar:+.2%}")
    json.dump({"res": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}, "luck_bar": bar,
               "pass": verdicts, "kept": {str(y): {k: list(v) for k, v in d.items()} for y, d in kept.items()}},
              open(OUT / f"report_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
