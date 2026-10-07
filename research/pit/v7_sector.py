# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v7 — industry momentum sector tilt (docs/score-v3-plan.md, "Score v7 pre-registration").

Every 20 sessions in the 500 largest: sector 6-1 momentum = cap-weighted mean of each stock's
(1 + mom6) / (1 + ret1m) - 1 (126-session return skipping the last 21); top 3 sectors x1.5,
bottom 3 x0.5 (sectors with >= 5 names), stocks cap-weighted inside sectors. 19 bps on one-way
turnover, total return, vs SPY TR. Luck: random sector rankings. Confirmation: ranks 501-1000.

  python3 v7_sector.py
"""
import numpy as np
import pandas as pd

from common import META_DIR
from v3_model import OUT, stats

RT_COST, N_TRIALS = 0.0019, 9


def sector_mult(g, rng=None):
    s61 = (1 + g.mom6) / (1 + g.ret1m) - 1
    x = g.assign(s61=s61, w=g.mcap)
    counts = x.groupby("sector").size()
    ok = counts[counts >= 5].index
    mom = x[x.sector.isin(ok)].groupby("sector").apply(lambda v: (v.s61 * v.w).sum() / v.w.sum(), include_groups=False)
    order = list(mom.sort_values(ascending=False).index)
    if rng is not None:
        order = list(rng.permutation(order))
    mult = {s: 1.0 for s in x.sector.unique()}
    for s in order[:3]:
        mult[s] = 1.5
    for s in order[-3:]:
        mult[s] = 0.5
    return x.sector.map(mult).fillna(1.0).to_numpy()


def run(df, rng=None, stock_col=None, vs_own=False):
    dates = sorted(df.date.unique())[::2]
    prev, eq, bench, rows = None, 1.0, 1.0, []
    for d in dates:
        g = df[df.date == d]
        m = sector_mult(g, rng)
        if stock_col is not None:
            u = g[stock_col].rank(pct=True)
            m = m * (1 + (2 * u - 1)).fillna(1.0).to_numpy()
        w = pd.Series((g.mcap.to_numpy() * m), index=g.cik.to_numpy())
        w = w / w.sum()
        to = 0.5 * w.sub(prev, fill_value=0).abs().sum() if prev is not None else 0.0
        r = g.r20.fillna(0).to_numpy()
        port = float((w.to_numpy() * r).sum()) - to * RT_COST
        b = float(((g.mcap / g.mcap.sum()).to_numpy() * r).sum()) if vs_own else g.spy20_tr.iloc[0]
        eq *= 1 + port
        bench *= 1 + b
        rows.append({"date": d, "equity": eq, "spy": bench, "ret": port, "spy_ret": b, "turnover": to, "missing_ret": 0})
        prev = w
    return pd.DataFrame(rows).set_index("date")


def main():
    t = pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"])
    t = t[t.date.dt.year >= 2019].copy()
    t["sector"] = t.sector.fillna("NA")
    t["rk"] = t.groupby("date").mcap.rank(ascending=False, method="first")
    v5 = pd.read_csv(OUT / "scores_v5_real.csv.gz", parse_dates=["date"])[["cik", "date", "v5b"]]
    t = t.merge(v5, on=["cik", "date"], how="left")
    big, mid = t[t.rk <= 500], t[(t.rk > 500) & (t.rk <= 1000)]

    st = stats(run(big))
    act = st["cagr"] - st["spy_cagr"]
    print(f"v7 sector momentum tilt (500 largest): CAGR {st['cagr']:.2%} vs SPY TR {st['spy_cagr']:.2%} (active {act:+.2%}) | "
          f"DD {st['max_dd']:.1%} vs {st['spy_max_dd']:.1%} | beat {st['years_beat']}/{st['n_years']} yrs | turnover {st['turnover']:.0%}")
    print("  " + "  ".join(f"{y}: {r.port:+.1%} vs {r.spy:+.1%}" for y, r in st["by_year"].iterrows()))
    rng = np.random.default_rng(999)
    luck = np.array([(lambda s: s["cagr"] - s["spy_cagr"])(stats(run(big, rng))) for _ in range(300)])
    bar = float(np.percentile([rng.choice(luck, N_TRIALS, replace=False).max() for _ in range(5000)], 95))
    print(f"luck (random sector rankings): median {np.median(luck):+.2%}, 95th {np.percentile(luck, 95):+.2%}; "
          f"best-of-{N_TRIALS} 95th = {bar:+.2%}")
    sm = stats(run(mid, vs_own=True))
    mid_act = sm["cagr"] - sm["spy_cagr"]
    mluck = np.array([(lambda s: s["cagr"] - s["spy_cagr"])(stats(run(mid, rng, vs_own=True))) for _ in range(300)])
    pct = float((mluck < mid_act).mean() * 100)
    gates = {"1 CAGR > SPY TR": act > 0, "2 beat >= 5/8": st["years_beat"] >= 5, "3 active > luck bar": act > bar,
             "4 DD <= SPY+10pp": st["max_dd"] <= st["spy_max_dd"] + 0.10, "5 ranks 501-1000 > 90th pct": pct > 90}
    for k, v in gates.items():
        print(f"  gate {k}: {'PASS' if v else 'FAIL'}")
    print(f"  (ranks 501-1000: {mid_act:+.2%}/yr vs own benchmark = {pct:.0f}th pct)")
    print(f"\n=> v7 {'PASS' if all(gates.values()) else 'FAIL'}")
    sc = stats(run(big, stock_col="v5b"))
    print(f"\nreporting: v7 sector tilt + v5b stock tilt: CAGR {sc['cagr']:.2%} vs SPY TR {sc['spy_cagr']:.2%} "
          f"(active {sc['cagr'] - sc['spy_cagr']:+.2%}) | beat {sc['years_beat']}/{sc['n_years']} | DD {sc['max_dd']:.1%}")


if __name__ == "__main__":
    main()
