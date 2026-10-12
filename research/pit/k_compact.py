# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Compact portfolios, <= 14 positions (docs/compact-portfolio-plan.md, pre-registered ed95e6c
before any number). K1 SPY core + 13 best-scored large caps, K3 A5 core + K1's stocks, K4 11 sector
funds tilted by CANSLIM 2.0, K5 SPY core + fresh earnings-beat streaks. Trials 12-14.

  python3 k_compact.py --placebo --luck-runs 20   # stock returns shuffled within each date: mechanics only
  python3 k_compact.py            # THE single pre-registered run
"""
import argparse
import json

import numpy as np
import pandas as pd

import a_exposure as A
from a_exposure2 import a5_exposure
from common import META_DIR
from v3_model import OUT, stats

RT_COST, N_TRIALS, SLOTS, SLEEVE, SECTOR_MAX = 0.0019, 14, 13, 0.40, 3
K1_UNIV, K1_KEEP, K5_HOLD, K5_STREAK, FUND_FEE = 100, 26, 3, 4, 0.0009


def load():
    s = pd.read_csv(OUT / "scores_v8_real.csv.gz", usecols=["cik", "date", "symbol", "mcap", "r20", "spy20_tr", "v5b"],
                    parse_dates=["date"])
    t = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date", "sector", "beat_streak", "surprise_pct"],
                    parse_dates=["date"])
    s = s.merge(t, on=["cik", "date"], how="left")
    s["sector"] = s.sector.fillna("Unknown")
    dates = sorted(s.date.unique())[::2]
    s = s[s.date.isin(dates)].copy()
    prev = {d: p for p, d in zip(dates[:-1], dates[1:])}
    b0 = s[["cik", "date", "beat_streak"]].rename(columns={"beat_streak": "streak_prev"})
    b0["date"] = b0.date.map({p: d for d, p in prev.items()})
    s = s.merge(b0.dropna(subset=["date"]), on=["cik", "date"], how="left")
    s["fresh"] = (s.beat_streak >= K5_STREAK) & (s.beat_streak > s.streak_prev.fillna(0))
    return s, dates


def a5_core(dates):
    px, tr, rate = A.load()
    r, _ = A.run(px, tr, rate, a5_exposure(px, tr, rate))
    out = {}
    for d in dates:
        i = r.index.searchsorted(d, side="right")
        out[d] = float((1 + r.iloc[i:i + 20]).prod() - 1)
    return out


def frame(rows):
    e = pd.DataFrame(rows).set_index("date")
    e["equity"], e["spy"] = (1 + e.ret).cumprod(), (1 + e.spy_ret).cumprod()
    return e


def pick(order, sectors, held, held_sectors, n):
    """Fill up to n names from `order` (best first), max SECTOR_MAX per sector incl. kept holdings."""
    out, cnt = [], dict(held_sectors)
    for c in order:
        if len(out) >= n:
            break
        if c in held:
            continue
        sc = sectors[c]
        if cnt.get(sc, 0) >= SECTOR_MAX:
            continue
        out.append(c)
        cnt[sc] = cnt.get(sc, 0) + 1
    return out


def sleeve_run(panel, dates, mode, score_of=None, core=None, sleeve=SLEEVE, rng=None):
    """mode 'k1': rank-based with keep band; 'k5': fresh-beat events held K5_HOLD periods
    ('k5r' = same slots/holds filled at random)."""
    held, rows, prev_w = {}, [], None
    for k, d in enumerate(dates):
        g = panel[d]
        sectors = g.sector.to_dict()
        if mode == "k1":
            u = g[g.v5b.notna()].nlargest(K1_UNIV, "mcap")
            sc = score_of(u)
            rank = sc.rank(ascending=False, method="first")
            held = {c: v for c, v in held.items() if c in rank.index and rank[c] <= K1_KEEP}
            order = list(sc.sort_values(ascending=False).index)
        else:
            held = {c: v for c, v in held.items() if k - v < K5_HOLD and c in g.index}
            if mode == "k5":
                el = g[g.fresh]
                order = list(el.surprise_pct.sort_values(ascending=False).index)
            else:
                order = list(rng.permutation(g.index.to_numpy()))
        hs = {}
        for c in held:
            hs[sectors[c]] = hs.get(sectors[c], 0) + 1
        for c in pick(order, sectors, held, hs, SLOTS - len(held)):
            held[c] = k
        w = {c: sleeve / SLOTS for c in held}
        w["CORE"] = 1 - sum(w.values())
        cr = core[d] if core is not None else float(g.spy20_tr.iloc[0])
        r20 = g.r20.reindex([c for c in w if c != "CORE"]).fillna(0)
        ret = w["CORE"] * cr + float((r20 * (sleeve / SLOTS)).sum())
        ws = pd.Series(w)
        to = 0.5 * ws.sub(prev_w, fill_value=0).abs().sum() if prev_w is not None else 0.0
        rows.append({"date": d, "ret": ret - to * RT_COST, "spy_ret": float(g.spy20_tr.iloc[0]), "turnover": to,
                     "n_stocks": len(held)})
        prev_w = ws
    return frame(rows)


def sector_run(panel, dates, sector_score=None):
    rows, prev_w = [], None
    for d in dates:
        g = panel[d]
        cap = g.groupby("sector").mcap.sum()
        ret = g.assign(x=g.r20.fillna(0) * g.mcap).groupby("sector").x.sum() / cap
        if sector_score is None:
            gs = g[g.v5b.notna()].assign(u=lambda x: x.v5b.rank(pct=True))
            sscore = gs.assign(x=gs.u * gs.mcap).groupby("sector").x.sum() / gs.groupby("sector").mcap.sum()
        else:
            sscore = sector_score(cap.index)
        u = sscore.reindex(cap.index).rank(pct=True)
        m = (1 + (2 * u - 1)).fillna(1.0)
        w = cap * m / (cap * m).sum()
        to = 0.5 * w.sub(prev_w, fill_value=0).abs().sum() if prev_w is not None else 0.0
        rows.append({"date": d, "ret": float((w * ret).sum()) - to * RT_COST - FUND_FEE / 12.6,
                     "spy_ret": float(g.spy20_tr.iloc[0]), "turnover": to, "n_stocks": len(w)})
        prev_w = w
    return frame(rows)


def roll12(e):
    a, b = e.equity, e.spy
    n = 13
    return float(((a.shift(-n) / a) > (b.shift(-n) / b)).dropna().mean())


def summarize(name, e):
    st = stats(e)
    act = st["cagr"] - st["spy_cagr"]
    print(f"\n{name}: CAGR {st['cagr']:.2%} vs SPY TR {st['spy_cagr']:.2%} (active {act:+.2%}) | DD {st['max_dd']:.1%} vs "
          f"{st['spy_max_dd']:.1%} | beat {st['years_beat']}/{st['n_years']} yrs | rolling 12m beat {roll12(e):.0%} | "
          f"turnover {st['turnover']:.0%}/period | avg stocks {e.n_stocks.mean():.1f}")
    print("  " + "  ".join(f"{y}: {r.port:+.1%} vs {r.spy:+.1%}" for y, r in st["by_year"].iterrows()))
    return {"active": act, **{k: v for k, v in st.items() if k != "by_year"}, "roll12": roll12(e)}


def luck_bar(luck):
    rng = np.random.default_rng(4242)
    best = np.array([rng.choice(luck, N_TRIALS, replace=False).max() for _ in range(2000)])
    return float(np.percentile(best, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luck-runs", type=int, default=300)
    ap.add_argument("--placebo", action="store_true")
    a = ap.parse_args()
    s, dates = load()
    if a.placebo:
        prng = np.random.default_rng(5)
        s["r20"] = s.groupby("date").r20.transform(lambda v: prng.permutation(v.to_numpy()))
    panel = {d: g.set_index("cik") for d, g in s.groupby("date")}
    core5 = a5_core(dates)
    print(f"=== Compact portfolios (<= 14 positions), {dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d}, {len(dates)} periods ===")
    by_score = lambda u: u.v5b
    res, eq = {}, {}
    eq["K1"] = sleeve_run(panel, dates, "k1", by_score)
    eq["K3"] = sleeve_run(panel, dates, "k1", by_score, core=core5)
    eq["K4"] = sector_run(panel, dates)
    eq["K5"] = sleeve_run(panel, dates, "k5")
    names = {"K1": "K1 SPY 60% + 13 best-scored of the 100 largest", "K3": "K3 A5 core 60% + K1's 13 stocks",
             "K4": "K4 11 sector funds tilted by CANSLIM 2.0", "K5": "K5 SPY 60% + fresh earnings-beat streaks"}
    for k, e in eq.items():
        res[k] = summarize(names[k], e)
        if not a.placebo:
            e.to_csv(OUT / f"equity_compact_{k}.csv")

    print("\n--- reporting only ---")
    rep = {}
    rep["K1_size"] = summarize("K1 with the 13 LARGEST instead of best-scored", sleeve_run(panel, dates, "k1", lambda u: u.mcap))
    for w in (0.3, 1.0):
        rep[f"K1_sleeve{int(w * 100)}"] = summarize(f"K1 sleeve {w:.0%}", sleeve_run(panel, dates, "k1", by_score, sleeve=w))
    rep["A5_core_only"] = summarize("A5 core alone (100%, no stocks)", sleeve_run(panel, dates, "k1", by_score, core=core5, sleeve=0.0))

    rng = np.random.default_rng(1212)
    luck = {"K1": [], "K3": [], "K4": [], "K5": []}
    for i in range(a.luck_runs):
        rs = {c: v for c, v in zip(s.cik.unique(), rng.random(s.cik.nunique()))}   # persistent per company
        rnd = lambda u: pd.Series(u.index.map(rs).to_numpy(), index=u.index)
        luck["K1"].append(stats(sleeve_run(panel, dates, "k1", rnd))["cagr"])
        luck["K3"].append(stats(sleeve_run(panel, dates, "k1", rnd, core=core5))["cagr"])
        ss = dict(zip(sorted(s.sector.unique()), rng.random(s.sector.nunique())))
        luck["K4"].append(stats(sector_run(panel, dates, lambda idx: pd.Series([ss[x] for x in idx], index=idx)))["cagr"])
        luck["K5"].append(stats(sleeve_run(panel, dates, "k5r", rng=rng))["cagr"])
        if (i + 1) % 50 == 0:
            print(f"  luck runs {i + 1}/{a.luck_runs}", flush=True)
    spy_cagr = res["K1"]["spy_cagr"]
    verdict = {}
    print(f"\n=== Gates (luck bar = 95th pct of best of {N_TRIALS} random versions of the same design) ===")
    for k in ("K1", "K4", "K5", "K3"):
        l = np.array(luck[k]) - spy_cagr
        bar = luck_bar(l)
        r = res[k]
        g3 = (r["active"] > bar) if k != "K3" else verdict.get("K1_g3", False)
        g = {"1 CAGR > SPY": r["active"] > 0, "2 >= 5/8 yrs": r["years_beat"] >= 5,
             "3 > luck bar" if k != "K3" else "3 K1 passes gate 3": g3, "4 DD <= SPY+10pp": r["max_dd"] <= r["spy_max_dd"] + 0.10}
        if k == "K1":
            verdict["K1_g3"] = g3
        verdict[k] = all(g.values())
        pct = float((l < r["active"]).mean() * 100)
        print(f"{k}: active {r['active']:+.2%} | random median {np.median(l):+.2%}, 95th {np.percentile(l, 95):+.2%}, "
              f"bar {bar:+.2%} | {pct:.0f}th pct of random")
        print("    " + " | ".join(f"{n}: {'PASS' if v else 'FAIL'}" for n, v in g.items()) + f" => {'PASS' if verdict[k] else 'FAIL'}")
        r.update({"luck_median": float(np.median(l)), "luck_95": float(np.percentile(l, 95)), "luck_bar": bar, "pct": pct})
    if a.placebo:
        return
    json.dump({"res": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
               "reporting": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in rep.items()},
               "pass": {k: bool(v) for k, v in verdict.items() if k in ("K1", "K3", "K4", "K5")}},
              open(OUT / "report_compact.json", "w"), indent=1)


if __name__ == "__main__":
    main()
