# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v9 — documented volume effects (docs/score-v3-plan.md, "Score v9 pre-registration",
committed 1f21069 before any number). Trial 11. Built from v8_model.py; only the feature set,
the trial count and the file names differ.

Stage 1 (reporting): per-feature IC scoreboard for the 4 volume features.
Stage 2 (gated): v5b's exact walk-forward rule with the candidate pool widened to the
38 v3 features + 4 volume features; S&P-relative tilt on the 500 largest; best-of-11 luck bar.
Reporting: v5b re-run in the same script (v9 - v5b), the in-sample CANSLIM 2.0 composite
`c2`, and ranks 501-1000 confirmation.

  python3 v9_model.py --placebo [--seed N]   # shuffled target: pipeline check only
  python3 v9_model.py                        # THE single pre-registered run
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import META_DIR
from p4_tests import nw_t
from v3_assemble import FEATURES, spy_tr
from v3_model import OUT, TEST_YEARS, ic_by_date, prep, stats, window_end
from v4_model import score
from v5_model import report, tilt

NEW = ["vspike", "turnover", "lowvol_mom", "ear_vol"]
LETTER = {"vspike": "S", "turnover": "S", "lowvol_mom": "L", "ear_vol": "C"}
TOP_UNIV, N_TRIALS = 500, 11
C2 = {"C": [("beat_streak", 1), ("surprise_pct", 1)], "A": [("roe", 1)], "S": [("s3", 1), ("dtc", -1)],
      "I": [("n_brokers", 1)]}


def select_pool(train, pool):
    """v4_model.select with an explicit candidate pool (identical rule)."""
    dates = sorted(train.date.unique())
    half = dates[len(dates) // 2]
    kept = {}
    for f in pool:
        ic = ic_by_date(train[train[f].notna()], "R_" + f).dropna()
        if len(ic) < 10:
            continue
        t, a, b = nw_t(ic), ic[ic.index < half].mean(), ic[ic.index >= half].mean()
        if abs(t) >= 2.0 and np.sign(a) == np.sign(b) == np.sign(ic.mean()):
            kept[f] = (float(np.sign(ic.mean())), float(t))
    return kept


def mid_check(s, col, dates, runs=300):
    """Same tilt on ranks 501-1000 vs that universe's cap-weighted benchmark (v5_diagnostics #3)."""
    mid = s[(s.rk > 500) & (s.rk <= 1000)]

    def act(df, c):
        eq_t = eq_b = 1.0
        for d in dates:
            g = df[df.date == d]
            u = g[c].rank(pct=True)
            m = (1 + (2 * u - 1)).fillna(1.0)
            w = (g.mcap * m) / (g.mcap * m).sum()
            r = g.r20.fillna(0).to_numpy()
            eq_t *= 1 + float((w.to_numpy() * r).sum())
            eq_b *= 1 + float(((g.mcap / g.mcap.sum()).to_numpy() * r).sum())
        yrs = (dates[-1] - dates[0]).days / 365.25 + 20 / 252
        return eq_t ** (1 / yrs) - eq_b ** (1 / yrs)
    real = act(mid, col)
    rng = np.random.default_rng(99)
    luck = np.array([act(mid.assign(rnd=rng.random(len(mid))), "rnd") for _ in range(runs)])
    ic = ic_by_date(mid[mid[col].notna()], col)
    return {"active": real, "pctile": float((luck < real).mean() * 100), "ic": float(ic.mean()), "ic_t": float(nw_t(ic))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luck-runs", type=int, default=300)
    a = ap.parse_args()
    tag = f"v9_placebo{a.seed}" if a.placebo else "v9_real"
    t = pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"])
    t = t.merge(pd.read_csv(META_DIR / "v9_features.csv.gz", parse_dates=["date"]), on=["cik", "date"], how="left")
    full = prep(t, a.placebo, a.seed)
    for f in ("vspike", "turnover", "ear_vol"):
        full["R_" + f] = full.groupby("date")[f].rank(pct=True) - 0.5
    # Lee-Swaminathan: winners with LOW turnover (centered ranks, same date), then ranked like the rest
    full["lowvol_mom"] = (full["R_s4"] - full["R_turnover"]).where(full.s4.notna() & full.turnover.notna())
    full["R_lowvol_mom"] = full.groupby("date")["lowvol_mom"].rank(pct=True) - 0.5
    full["rk"] = full.groupby("date").mcap.rank(ascending=False, method="first")

    # ---- stage 1: scoreboard for the new features (reporting) ----
    half = pd.Timestamp("2021-01-01")
    print(f"=== Stage 1: documented volume effects, full v3 universe, IC vs 60-session excess "
          f"{'(PLACEBO)' if a.placebo else ''} ===")
    board = {}
    for f in NEW:
        ic = ic_by_date(full[full[f].notna()], f).dropna()
        h1, h2 = ic[ic.index < half].mean(), ic[ic.index >= half].mean()
        board[f] = {"letter": LETTER[f], "coverage": float(full[f].notna().mean()), "ic": float(ic.mean()),
                    "t": float(nw_t(ic)), "2016-20": float(h1), "2021-26": float(h2),
                    "same_sign": bool(np.sign(h1) == np.sign(h2))}
        b = board[f]
        print(f"  {b['letter']} {f:12s} cov {b['coverage']:.0%} | IC {b['ic']:+.4f} t {b['t']:+.2f} | "
              f"2016-20 {h1:+.4f} 2021-26 {h2:+.4f} {'same sign' if b['same_sign'] else 'FLIPS'}")

    # ---- stage 2: v9 (gated) + v5b reference, identical machinery ----
    ends = window_end(pd.DatetimeIndex(sorted(full.date.unique())), spy_tr().index)
    parts, kept = [], {}
    for y in TEST_YEARS:
        train = full[full.date.map(ends) < pd.Timestamp(f"{y}-01-01")]
        k8, k5 = select_pool(train, FEATURES + NEW), select_pool(train, FEATURES)
        kept[y] = {"v9": k8, "v5b": k5}
        test = full[full.date.dt.year == y].copy()
        test["v9"], test["v5b"] = score(test, k8), score(test, k5)
        test["c2"] = sum(np.mean([sg * test["R_" + f].fillna(0) for f, sg in fs], axis=0) for fs in C2.values()) / len(C2)
        parts.append(test)
        print(f"{y}: v9 kept " + ", ".join(f"{'+' if s > 0 else '-'}{f}({tt:+.1f})" for f, (s, tt) in
                                         sorted(k8.items(), key=lambda kv: -abs(kv[1][1]))) +
              f" | new kept: {sorted(set(k8) & set(NEW)) or 'none'}", flush=True)
    s_all = pd.concat(parts)
    s = s_all[s_all.rk <= TOP_UNIV]
    s[["cik", "date", "symbol", "mcap", "r20", "spy20_tr", "v9", "v5b", "c2"]].to_csv(OUT / f"scores_{tag}.csv.gz", index=False)

    print(f"\n=== Score v9 tilt {'PLACEBO' if a.placebo else 'pre-registered run (trial 11)'} ===")
    bst = stats(tilt(s.assign(none=np.nan), "none"))
    print(f"benchmark (cap-weighted 500 largest): CAGR {bst['cagr']:.2%} vs SPY TR {bst['spy_cagr']:.2%}")
    res = {}
    for col, name in (("v9", "v9 (v5b rule, pool + 4 volume features)"), ("v5b", "v5b reference (same script)"),
                      ("c2", "c2 CANSLIM 2.0 composite (IN-SAMPLE, cannot pass)")):
        e = tilt(s, col)
        e.to_csv(OUT / f"equity_{tag}_{col}.csv")
        res[col] = report(name, s, col, e)
    dates = sorted(s.date.unique())[::2]
    mids = {col: mid_check(s_all, col, dates) for col in ("v9", "v5b", "c2")}
    for col, r in mids.items():
        print(f"ranks 501-1000 {col}: active vs own benchmark {r['active']:+.2%}/yr, {r['pctile']:.0f}th pct of random, "
              f"IC {r['ic']:+.4f} (t {r['ic_t']:+.2f})")
    print(f"v9 - v5b active: {res['v9']['active'] - res['v5b']['active']:+.2%}/yr")
    if a.placebo:
        return
    rng = np.random.default_rng(777)
    luck = np.array([stats(tilt(s.assign(rnd=rng.random(len(s))), "rnd"))["cagr"] for _ in range(a.luck_runs)]) - bst["spy_cagr"]
    best_of = np.array([rng.choice(luck, N_TRIALS, replace=False).max() for _ in range(2000)])
    bar = float(np.percentile(best_of, 95))
    print(f"\nluck: random tilts median {np.median(luck):+.2%}, 95th {np.percentile(luck, 95):+.2%}; "
          f"best-of-{N_TRIALS} 95th = {bar:+.2%} (gate 3 bar)")
    r = res["v9"]
    g = {"1 CAGR > SPY TR": r["active"] > 0, "2 beat >= 5/8 yrs": r["years_beat"] >= 5,
         "3 active > luck bar": r["active"] > bar, "4 DD <= SPY+10pp": r["max_dd"] <= r["spy_max_dd"] + 0.10}
    verdict = all(g.values())
    print("v9: " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in g.items()) + f" => {'PASS' if verdict else 'FAIL'}")
    new_2026 = sorted(set(kept[2026]["v9"]) & set(NEW))
    print(f"new features kept in the 2026 selection: {new_2026 or 'none'} -> "
          f"{'join CANSLIM 2.0' if verdict and new_2026 else 'CANSLIM 2.0 uses v5b 2026 selection'}")
    json.dump({"board": board, "res": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
               "mid": mids, "luck_bar": bar, "pass": verdict, "gates": {k: bool(v) for k, v in g.items()},
               "kept": {str(y): {k: {f: list(v) for f, v in d.items()} for k, d in kd.items()} for y, kd in kept.items()},
               "new_2026": new_2026}, open(OUT / f"report_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
