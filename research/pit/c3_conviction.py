# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Conviction portfolio (docs/conviction-portfolio-plan.md, pre-registered 68e02ba before any number).
Part A: do the top 1-2% of CANSLIM 2.0 (within band) beat the top 5-10%? Part B (trial 15): 10 names,
conviction-weighted, core + small caps, no index.

  python3 c3_conviction.py --placebo --luck-runs 20   # r20 shuffled within each date: mechanics only
  python3 c3_conviction.py                            # THE single pre-registered run
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

import c2_smallcap as C2
from common import META_DIR
from p4_tests import nw_t
from v3_model import OUT, TEST_YEARS, prep, stats

N_TRIALS, N_HOLD, SECTOR_MAX, KEEP_PCT = 15, 10, 3, 0.95
COST_SMALL, COST_CORE = 0.0040, 0.0019
BUCKETS = [("top 1%", 0.99, 1.01), ("1-2%", 0.98, 0.99), ("2-5%", 0.95, 0.98), ("5-10%", 0.90, 0.95),
           ("10-20%", 0.80, 0.90), ("below 20%", -1, 0.80)]


def frozen_score(g, k):
    return (sum(sg * g["R_" + f].fillna(0) for f, (sg, _) in k.items()) / len(k)) if k else pd.Series(np.nan, index=g.index)


def build():
    kept = {int(y): v["v5b"] for y, v in json.load(open(OUT / "report_v8_real.json"))["kept"].items()}
    core = prep(pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"]), False)
    core = core[core.date.dt.year >= TEST_YEARS[0]].copy()
    small = C2.build()
    small = small[small.date.dt.year >= TEST_YEARS[0]].copy()
    dates = sorted(core.date.unique())[::2]
    sec = pd.concat((pd.read_csv(f, usecols=["cik", "date", "sector"], parse_dates=["date"])
                     for f in sorted(glob.glob(str(META_DIR / "daily" / "*.csv.gz")))), ignore_index=True)
    sec = sec[sec.date.isin(dates)].drop_duplicates(["cik", "date"])
    parts = []
    for band, t in (("core", core), ("small", small)):
        t = t[t.date.isin(dates)].copy()
        t["score"] = pd.concat([frozen_score(t[t.date.dt.year == y], kept[y]) for y in TEST_YEARS if (t.date.dt.year == y).any()])
        if "sector" not in t:
            t = t.merge(sec, on=["cik", "date"], how="left")
        t["band"] = band
        parts.append(t[["cik", "date", "symbol", "mcap", "sector", "r20", "score", "band"]])
    p = pd.concat(parts, ignore_index=True)
    p = p[p.score.notna()].copy()
    p["sector"] = p.sector.fillna("Unknown")
    spy = core.drop_duplicates("date").set_index("date").spy20_tr
    p["spy20_tr"] = p.date.map(spy)
    p["pct"] = p.groupby(["date", "band"]).score.rank(pct=True)
    return p, dates


def part_a(p):
    p = p.assign(x=p.r20.fillna(0) - p.spy20_tr)
    out = {}
    for scope, g in (("pooled", p), ("core", p[p.band == "core"]), ("small", p[p.band == "small"])):
        row = {}
        for name, lo, hi in BUCKETS + [("top 2%", 0.98, 1.01)]:
            per = g[(g.pct >= lo) & (g.pct < hi)].groupby("date").x.mean().dropna()
            row[name] = {"mean": float(per.mean()), "t": float(nw_t(per)), "n_dates": len(per),
                         "h1": float(per[per.index.year <= 2022].mean()), "h2": float(per[per.index.year >= 2023].mean())}
        out[scope] = row
    for scope, row in out.items():
        print(f"\n  {scope}: " + " | ".join(f"{k} {v['mean']:+.2%} (t {v['t']:+.2f})" for k, v in row.items() if k != "top 2%"))
        t2 = row["top 2%"]
        print(f"    top 2%: {t2['mean']:+.2%}/20d (t {t2['t']:+.2f}) | 2019-22 {t2['h1']:+.2%} | 2023-26 {t2['h2']:+.2%}")
    pr = out["pooled"]
    ok = pr["top 2%"]["mean"] > 0 and pr["top 2%"]["t"] >= 2.0 and pr["top 2%"]["mean"] > pr["5-10%"]["mean"] \
        and pr["top 2%"]["h1"] > 0 and pr["top 2%"]["h2"] > 0
    return ok, out


def portfolio(panel, dates, n=N_HOLD, weights="conviction", core_w=0.0, bands=("core", "small"), pct_of=None):
    held, rows, prev = [], [], None
    for d in dates:
        g = panel[d]
        g = g[g.band.isin(bands)]
        pct = pct_of(g) if pct_of is not None else g.pct
        held = [c for c in held if c in pct.index and pct[c] >= KEEP_PCT]
        cnt = g.sector.reindex(held).value_counts().to_dict()
        for c in pct.sort_values(ascending=False).index:
            if len(held) >= n:
                break
            if c in held:
                continue
            s = g.sector[c]
            if cnt.get(s, 0) >= SECTOR_MAX:
                continue
            held.append(c)
            cnt[s] = cnt.get(s, 0) + 1
        order = pct.reindex(held).sort_values(ascending=False).index
        raw = pd.Series(np.arange(len(order), 0, -1, dtype=float) if weights == "conviction" else np.ones(len(order)), index=order)
        w = raw / raw.sum() * (1 - core_w) if len(raw) else raw
        w["SPY"] = core_w + (1 - core_w if not len(raw) else 0.0)
        r = g.r20.reindex(order).fillna(0)
        spy = float(g.spy20_tr.iloc[0])
        ret = float((w.drop("SPY") * r).sum()) + w["SPY"] * spy
        dw = w.sub(prev, fill_value=0).abs() if prev is not None else w.abs()
        cost_rate = pd.Series(COST_CORE, index=dw.index)
        caps = g.mcap.reindex(dw.index)
        cost_rate[caps < 1e9] = COST_SMALL
        if prev is not None:
            pcap = prev_caps.reindex(dw.index)
            cost_rate[(caps.isna()) & (pcap < 1e9)] = COST_SMALL
        cost = float((dw * cost_rate).sum())
        rows.append({"date": d, "ret": ret - cost, "spy_ret": spy, "turnover": 0.5 * float(dw.sum()),
                     "small_w": float(w.drop("SPY")[caps.reindex(w.drop("SPY").index) < 1e9].sum()), "n": len(order)})
        prev, prev_caps = w, g.mcap
    e = pd.DataFrame(rows).set_index("date")
    e["equity"], e["spy"] = (1 + e.ret).cumprod(), (1 + e.spy_ret).cumprod()
    return e


def roll12(e):
    return float(((e.equity.shift(-13) / e.equity) > (e.spy.shift(-13) / e.spy)).dropna().mean())


def summarize(name, e):
    st = stats(e)
    act = st["cagr"] - st["spy_cagr"]
    print(f"\n{name}: CAGR {st['cagr']:.2%} vs SPY TR {st['spy_cagr']:.2%} (active {act:+.2%}) | DD {st['max_dd']:.1%} vs "
          f"{st['spy_max_dd']:.1%} | beat {st['years_beat']}/{st['n_years']} yrs | rolling 12m beat {roll12(e):.0%} | "
          f"one-way turnover {st['turnover']:.0%}/period | small caps {e.small_w.mean():.0%} of book | avg names {e.n.mean():.1f}")
    print("  " + "  ".join(f"{y}: {r.port:+.1%} vs {r.spy:+.1%}" for y, r in st["by_year"].iterrows()))
    return {"active": act, **{k: v for k, v in st.items() if k != "by_year"}, "roll12": roll12(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luck-runs", type=int, default=300)
    ap.add_argument("--placebo", action="store_true")
    a = ap.parse_args()
    p, dates = build()
    if a.placebo:
        prng = np.random.default_rng(7)
        p["r20"] = p.groupby("date").r20.transform(lambda v: prng.permutation(v.to_numpy()))
    p = p[p.date.isin(dates)]
    print(f"=== Conviction portfolio {'PLACEBO' if a.placebo else 'pre-registered run'}: {dates[0]:%Y-%m-%d} -> "
          f"{dates[-1]:%Y-%m-%d}, {len(dates)} periods; names/date core {p[p.band == 'core'].groupby('date').size().median():.0f}, "
          f"small {p[p.band == 'small'].groupby('date').size().median():.0f} ===")
    print("\nPART A — mean 20-session return minus SPY TR by score bucket (within band):")
    ok_a, a_res = part_a(p)
    print(f"\n=> Gate A (pooled top 2% > 0 with t >= 2, > 5-10% bucket, positive both halves): {'PASS' if ok_a else 'FAIL'}")

    panel = {d: g.set_index("cik") for d, g in p.groupby("date")}
    print("\nPART B — conviction portfolio")
    e = portfolio(panel, dates)
    res = summarize("B: 10 names, conviction-weighted, core + small, no index", e)
    rep = {}
    for nm, kw in (("6 names", {"n": 6}), ("14 names", {"n": 14}), ("10 equal-weight", {"weights": "equal"}),
                   ("10 core only", {"bands": ("core",)}), ("10 small only", {"bands": ("small",)}),
                   ("10 + 30% SPY core", {"core_w": 0.3})):
        rep[nm] = summarize(f"(reporting) {nm}", portfolio(panel, dates, **kw))
    rng = np.random.default_rng(1515)
    ciks = p.cik.unique()
    luck = []
    for i in range(a.luck_runs):
        rs = pd.Series(rng.random(len(ciks)), index=ciks)
        luck.append(stats(portfolio(panel, dates, pct_of=lambda g: g.index.to_series().map(rs).groupby(g.band).rank(pct=True)))["cagr"])
        if (i + 1) % 50 == 0:
            print(f"  luck runs {i + 1}/{a.luck_runs}", flush=True)
    l = np.array(luck) - res["spy_cagr"]
    best = np.array([np.random.default_rng(i).choice(l, N_TRIALS, replace=False).max() for i in range(2000)])
    bar = float(np.percentile(best, 95))
    pct = float((l < res["active"]).mean() * 100)
    g = {"1 CAGR > SPY": res["active"] > 0, "2 >= 5/8 yrs": res["years_beat"] >= 5, "3 > luck bar": res["active"] > bar,
         "4 DD <= SPY+20pp": res["max_dd"] <= res["spy_max_dd"] + 0.20}
    ok_b = all(g.values())
    print(f"\nluck: random 10-name books (same rules) active 5th {np.percentile(l, 5):+.2%} | median {np.median(l):+.2%} | "
          f"95th {np.percentile(l, 95):+.2%}; best-of-{N_TRIALS} 95th = {bar:+.2%}; B is at the {pct:.0f}th pct")
    print("B: " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in g.items()) + f" => {'PASS' if ok_b else 'FAIL'}")
    if a.placebo:
        return
    e.to_csv(OUT / "equity_conviction.csv")
    json.dump({"part_a": a_res, "gate_a": bool(ok_a), "b": {k: float(v) for k, v in res.items()},
               "reporting": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in rep.items()},
               "luck": {"median": float(np.median(l)), "p95": float(np.percentile(l, 95)), "bar": bar, "pct": pct},
               "gate_b": bool(ok_b)}, open(OUT / "report_conviction.json", "w"), indent=1)


if __name__ == "__main__":
    main()
