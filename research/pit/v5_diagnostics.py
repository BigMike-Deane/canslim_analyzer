# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""v5b diagnostics (nothing is chosen here; v5b's yearly feature selections are reused as-is).

1. Factor attribution: v5b active return (vs the cap-weighted 500 largest) regressed on
   long-short style factors built in the same universe (cap-weighted top third - bottom
   third, 20-session returns): size, value (B/M), momentum (12-1), quality (ROE),
   low volatility (60d), beat streak.
2. Losing years: Brinson split of v5b's active return per year into sector allocation vs
   within-sector selection.
3. Generalization: the same tilt (same yearly selections) on companies ranked 501-1000 by
   market cap, vs that universe's own cap-weighted benchmark and 300 random tilts there.
"""
import json

import numpy as np
import pandas as pd

from common import META_DIR
from v3_model import OUT, TEST_YEARS, prep
from v4_model import score


def tilt_weights(g, col):
    u = g[col].rank(pct=True)
    m = (1 + (2 * u - 1)).fillna(1.0)
    return (g.mcap * m) / (g.mcap * m).sum()


def factor_series(big, dates):
    out = {}
    for d in dates:
        g = big[big.date == d]
        row = {}
        for name, col, sign in (("size", "logcap", -1), ("value", "bm", 1), ("momentum", "s4", 1),
                                ("quality", "roe", 1), ("lowvol", "vol60", -1), ("beat", "beat_streak", 1)):
            x = g[g[col].notna()]
            if len(x) < 30:
                continue
            q = (x[col] * sign).rank(pct=True)
            hi, lo = x[q > 2 / 3], x[q <= 1 / 3]
            row[name] = ((hi.r20 * hi.mcap).sum() / hi.mcap.sum()) - ((lo.r20 * lo.mcap).sum() / lo.mcap.sum())
        out[d] = row
    return pd.DataFrame(out).T


def main():
    rep = json.load(open(OUT / "report_v5_real.json"))
    kept = {int(y): {f: None for f in v["v5b"]} for y, v in rep["kept"].items()}
    full = prep(pd.read_csv(META_DIR / "v3_table.csv.gz", parse_dates=["date"]), False)
    full["rk"] = full.groupby("date").mcap.rank(ascending=False, method="first")
    # signs from the real run's training ICs are not stored -> recompute selection signs exactly as v4.select did
    from v4_model import select
    from v3_assemble import spy_tr
    from v3_model import window_end
    ends = window_end(pd.DatetimeIndex(sorted(full.date.unique())), spy_tr().index)
    parts = []
    for y in TEST_YEARS:
        k = select(full[full.date.map(ends) < pd.Timestamp(f"{y}-01-01")])
        assert sorted(k) == sorted(kept[y]), (y, sorted(k), sorted(kept[y]))
        t = full[full.date.dt.year == y].copy()
        t["v5b"] = score(t, k)
        parts.append(t)
    s = pd.concat(parts)
    dates = sorted(s.date.unique())[::2]

    # ---- 1. factor attribution -------------------------------------------------
    big = s[s.rk <= 500]
    act = {}
    for d in dates:
        g = big[big.date == d]
        w, b = tilt_weights(g, "v5b"), g.mcap / g.mcap.sum()
        act[d] = float(((w - b) * g.r20.fillna(0)).sum())
    act = pd.Series(act)
    F = factor_series(big, dates).reindex(act.index).fillna(0)
    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, act.to_numpy(), rcond=None)
    resid = act.to_numpy() - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=X.shape[1]))
    print("1. FACTOR ATTRIBUTION (v5b active vs cap-weighted 500 largest, per 20 sessions)")
    print(f"   raw active {act.mean() * 12.6:+.2%}/yr")
    for name, b_, s_ in zip(["alpha (annualized)"] + list(F.columns), beta, se):
        if name.startswith("alpha"):
            print(f"   {name:20s} {b_ * 12.6:+.2%}/yr (t {b_ / s_:+.2f})")
        else:
            print(f"   {name:20s} loading {b_:+.3f} (t {b_ / s_:+.2f}); factor mean {F[name].mean() * 12.6:+.2%}/yr")
    print(f"   R^2 {1 - resid.var() / act.var():.2f}")

    # ---- 2. Brinson by year ------------------------------------------------------
    print("\n2. LOSING-YEAR AUTOPSY (v5b active by year: sector allocation vs selection within sectors)")
    rows = []
    for d in dates:
        g = big[big.date == d].assign(w=lambda x: tilt_weights(x, "v5b"), b=lambda x: x.mcap / x.mcap.sum(),
                                      r=lambda x: x.r20.fillna(0))
        sec = g.groupby("sector").apply(lambda x: pd.Series({
            "w": x.w.sum(), "b": x.b.sum(),
            "rw": (x.w * x.r).sum() / x.w.sum() if x.w.sum() > 0 else 0,
            "rb": (x.b * x.r).sum() / x.b.sum() if x.b.sum() > 0 else 0}), include_groups=False)
        rb_tot = (sec.b * sec.rb).sum()
        rows.append({"date": d, "alloc": ((sec.w - sec.b) * (sec.rb - rb_tot)).sum(),
                     "select": (sec.w * (sec.rw - sec.rb)).sum()})
    bz = pd.DataFrame(rows).set_index("date")
    yr = bz.groupby(bz.index.year).sum()
    for y, r in yr.iterrows():
        print(f"   {y}: allocation {r.alloc:+.2%}  selection {r.select:+.2%}  total {r.alloc + r.select:+.2%}")

    # ---- 3. generalization to ranks 501-1000 --------------------------------------
    mid = s[(s.rk > 500) & (s.rk <= 1000)]
    def act_cagr(df, col):
        eq_t, eq_b = 1.0, 1.0
        for d in dates:
            g = df[df.date == d]
            r = g.r20.fillna(0).to_numpy()
            eq_t *= 1 + float((tilt_weights(g, col) * r).sum())
            eq_b *= 1 + float(((g.mcap / g.mcap.sum()) * r).sum())
        yrs = (dates[-1] - dates[0]).days / 365.25 + 20 / 252
        return eq_t ** (1 / yrs) - eq_b ** (1 / yrs)
    real = act_cagr(mid, "v5b")
    rng = np.random.default_rng(99)
    luck = np.array([act_cagr(mid.assign(rnd=rng.random(len(mid))), "rnd") for _ in range(300)])
    from p4_tests import nw_t
    ic = mid.groupby("date").apply(lambda g: g.v5b.corr(g.yw, method="spearman"), include_groups=False)
    print(f"\n3. GENERALIZATION: ranks 501-1000 (never traded by v5)")
    print(f"   IC {ic.mean():+.4f} (NW t {nw_t(ic):+.2f}) | tilt active vs own cap-weighted benchmark {real:+.2%}/yr | "
          f"random tilts median {np.median(luck):+.2%}, 95th {np.percentile(luck, 95):+.2%} -> "
          f"{(luck < real).mean() * 100:.0f}th percentile")


if __name__ == "__main__":
    main()
