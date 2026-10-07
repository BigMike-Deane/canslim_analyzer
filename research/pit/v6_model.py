# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v6 (docs/score-v3-plan.md, "Score v6 pre-registration").

v6a = v5b with EAR in the candidate pool. v6b = v6a + recency weighting of earnings-event
features (rank x exp(-days_since/60)) + sector-neutral ranks. Same walk-forward selection
(full-universe IC, NW t >= 2, sign-consistent halves), same tilt on the 500 largest, luck bar
= 95th pct of the best of 8 random tilts; then one confirmation on ranks 501-1000.

  python3 v6_model.py --placebo [--seed N]
  python3 v6_model.py
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import META_DIR
from p4_tests import nw_t
from v3_assemble import FEATURES, spy_tr
from v3_model import OUT, TEST_YEARS, ic_by_date, stats, window_end
from v5_model import report, tilt

FEATS = FEATURES + ["ear"]
EVENT = ["beat_streak", "surprise_pct", "s1", "ear"]
N_TRIALS, TOP_UNIV = 8, 500


def prep(t, placebo, seed, variant):
    t = t[t.y.notna()].copy()
    lo, hi = t.groupby("date").y.transform(lambda v: v.quantile(0.01)), t.groupby("date").y.transform(lambda v: v.quantile(0.99))
    t["yw"] = t.y.clip(lo, hi)
    if placebo:
        rng = np.random.default_rng(seed)
        t["yw"] = t.groupby("date").yw.transform(lambda v: rng.permutation(v.to_numpy()))
    keys = ["date", "sector"] if variant == "v6b" else ["date"]
    if variant == "v6b":
        t["sector"] = t.sector.fillna("NA")
    for f in FEATS:
        r = t.groupby(keys)[f].rank(pct=True) - 0.5
        if variant == "v6b" and f in EVENT:
            r = r * np.exp(-t.days_since.fillna(1e9) / 60.0)
        t["R_" + f] = r
    return t


def select(train):
    dates = sorted(train.date.unique())
    half = dates[len(dates) // 2]
    kept = {}
    for f in FEATS:
        ic = ic_by_date(train[train[f].notna()], "R_" + f).dropna()
        if len(ic) < 10:
            continue
        tt, a, b = nw_t(ic), ic[ic.index < half].mean(), ic[ic.index >= half].mean()
        if abs(tt) >= 2.0 and np.sign(a) == np.sign(b) == np.sign(ic.mean()):
            kept[f] = (float(np.sign(ic.mean())), float(tt))
    return kept


def score(df, kept):
    if not kept:
        return pd.Series(np.nan, index=df.index)
    return sum(s * df["R_" + f].fillna(0) for f, (s, _) in kept.items()) / len(kept)


def active_vs_own(df, col, dates):
    """Tilt vs the universe's own cap-weighted benchmark (for the 501-1000 confirmation)."""
    eq_t = eq_b = 1.0
    for d in dates:
        g = df[df.date == d]
        r = g.r20.fillna(0).to_numpy()
        u = g[col].rank(pct=True)
        w = (g.mcap * (1 + (2 * u - 1)).fillna(1.0))
        eq_t *= 1 + float((w / w.sum() * r).sum())
        eq_b *= 1 + float((g.mcap / g.mcap.sum() * r).sum())
    yrs = (dates[-1] - dates[0]).days / 365.25 + 20 / 252
    return eq_t ** (1 / yrs) - eq_b ** (1 / yrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"])
    base = base.merge(pd.read_csv(META_DIR / "v6_features.csv.gz", parse_dates=["date"]), on=["cik", "date"], how="left")
    base["rk"] = base.groupby("date").mcap.rank(ascending=False, method="first")
    sess = spy_tr().index
    results, scored = {}, {}
    for variant in ("v6a", "v6b"):
        full = prep(base, a.placebo, a.seed, variant)
        ends = window_end(pd.DatetimeIndex(sorted(full.date.unique())), sess)
        parts, kept_log = [], {}
        for y in TEST_YEARS:
            k = select(full[full.date.map(ends) < pd.Timestamp(f"{y}-01-01")])
            kept_log[y] = k
            t = full[full.date.dt.year == y].copy()
            t[variant] = score(t, k)
            parts.append(t)
            print(f"{variant} {y}: kept " + ", ".join(f"{'+' if s > 0 else '-'}{f}({tt:+.1f})" for f, (s, tt) in
                                                      sorted(k.items(), key=lambda kv: -abs(kv[1][1]))), flush=True)
        s = pd.concat(parts)
        scored[variant] = s
        big = s[s.rk <= TOP_UNIV]
        e = tilt(big, variant)
        results[variant] = report(variant, big, variant, e)
        results[variant]["kept"] = {str(y): sorted(k) for y, k in kept_log.items()}
    if a.placebo:
        return
    big = scored["v6a"][scored["v6a"].rk <= TOP_UNIV]
    rng = np.random.default_rng(888)
    bst = stats(tilt(big.assign(none=np.nan), "none"))
    luck = np.array([stats(tilt(big.assign(rnd=rng.random(len(big))), "rnd"))["cagr"] for _ in range(300)]) - bst["spy_cagr"]
    bar = float(np.percentile([rng.choice(luck, N_TRIALS, replace=False).max() for _ in range(5000)], 95))
    print(f"\nluck: random tilts median {np.median(luck):+.2%}, 95th {np.percentile(luck, 95):+.2%}; best-of-{N_TRIALS} 95th = {bar:+.2%}")
    for v in ("v6a", "v6b"):
        r = results[v]
        g = {"1 CAGR > SPY TR": r["active"] > 0, "2 beat >= 5/8": r["years_beat"] >= 5,
             "3 active > luck bar": r["active"] > bar, "4 DD <= SPY+10pp": r["max_dd"] <= r["spy_max_dd"] + 0.10}
        mid = scored[v][(scored[v].rk > 500) & (scored[v].rk <= 1000)]
        dates = sorted(mid.date.unique())[::2]
        real = active_vs_own(mid, v, dates)
        mluck = np.array([active_vs_own(mid.assign(rnd=rng.random(len(mid))), "rnd", dates) for _ in range(300)])
        pct = float((mluck < real).mean() * 100)
        stage1 = all(g.values())
        print(f"{v}: " + " | ".join(f"{k}: {'PASS' if x else 'FAIL'}" for k, x in g.items()) +
              f" => stage 1 {'PASS' if stage1 else 'FAIL'}; ranks 501-1000 confirmation {real:+.2%}/yr = {pct:.0f}th pct "
              f"({'PASS' if pct > 90 else 'FAIL'})")
        r.update({"luck_bar": bar, "mid_active": real, "mid_pctile": pct, "pass": bool(stage1 and pct > 90)})
    json.dump({k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()}
               for k, v in results.items()}, open(OUT / "report_v6_real.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
