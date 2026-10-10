# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""CANSLIM 2.0 small-cap confirmation (docs/score-v3-plan.md, pre-registered 1f21069 before any
number). Frozen v5b walk-forward formula (selected on >= $1B) applied to the $250M-$1B band.

  python3 c2_smallcap.py
"""
import glob
import json

import numpy as np
import pandas as pd

from common import META_DIR
from m3_signal_tests import load
from p4_tests import nw_t
from v3_assemble import add_insider, add_short_interest
from v3_model import OUT, TEST_YEARS

FEATS = ["surprise_pct", "beat_streak", "roe", "s3", "dtc", "n_brokers",      # 2026 formula
         "a", "am", "eps_growth", "ins_net90", "inst_pct", "si_pct"]            # also used 2019-2025
LO, HI, MIN_DVOL, RT_COST = 250e6, 1e9, 1e6, 0.0019


def build():
    t = load()[["cik", "date", "symbol", "close", "mcap", "a", "r20", "r60", "spy20", "spy60"]]
    t = t[t.date >= "2016-01-01"]
    pf = pd.read_csv(META_DIR / "v3_price_features.csv.gz", usecols=["cik", "date", "dvol20"], parse_dates=["date"])
    t = t.merge(pf, on=["cik", "date"], how="left")
    t = t[(t.close > 5) & (t.mcap >= LO) & (t.mcap < HI) & (t.dvol20 >= MIN_DVOL)].copy()
    dates = set(t.date)
    daily = pd.concat((pd.read_csv(f, usecols=["cik", "date", "surprise_pct", "beat_streak", "eps_growth", "inst_pct"], parse_dates=["date"])
                       for f in sorted(glob.glob(str(META_DIR / "daily" / "*.csv.gz")))), ignore_index=True)
    t = t.merge(daily[daily.date.isin(dates)], on=["cik", "date"], how="left")
    for f, keep in (("p3_signals.csv.gz", ["s3"]), ("p4_signals.csv.gz", ["roe"])):
        t = t.merge(pd.read_csv(META_DIR / f, usecols=["cik", "date"] + keep, parse_dates=["date"]), on=["cik", "date"], how="left")
    import p8_analyst as p8
    ev, _ = p8.load_events(set(t.cik))
    t = t.merge(p8.am_signal(t[["cik", "date"]], ev)[["cik", "date", "n_brokers", "am"]], on=["cik", "date"], how="left")
    t = add_insider(t)
    t = add_short_interest(t)
    t["y"] = t.r60 - t.spy60
    t = t[t.y.notna()].copy()
    lo, hi = t.groupby("date").y.transform(lambda v: v.quantile(0.01)), t.groupby("date").y.transform(lambda v: v.quantile(0.99))
    t["yw"] = t.y.clip(lo, hi)
    for f in FEATS:
        t["R_" + f] = t.groupby("date")[f].rank(pct=True) - 0.5
    return t


def tilt_active(s, col, dates):
    eq_t = eq_b = 1.0
    prev_t = prev_b = None
    for d in dates:
        g = s[s.date == d]
        u = g[col].rank(pct=True)
        m = (1 + (2 * u - 1)).fillna(1.0)
        wt = pd.Series(((g.mcap * m) / (g.mcap * m).sum()).to_numpy(), index=g.cik)
        wb = pd.Series((g.mcap / g.mcap.sum()).to_numpy(), index=g.cik)
        r = pd.Series(g.r20.fillna(0).to_numpy(), index=g.cik)
        ct = 0.5 * wt.sub(prev_t, fill_value=0).abs().sum() * RT_COST if prev_t is not None else 0.0
        cb = 0.5 * wb.sub(prev_b, fill_value=0).abs().sum() * RT_COST if prev_b is not None else 0.0
        eq_t *= 1 + float((wt * r).sum()) - ct
        eq_b *= 1 + float((wb * r).sum()) - cb
        prev_t, prev_b = wt, wb
    yrs = (dates[-1] - dates[0]).days / 365.25 + 20 / 252
    return eq_t ** (1 / yrs) - eq_b ** (1 / yrs)


def main():
    rep = json.load(open(OUT / "report_v8_real.json"))
    kept = {int(y): v["v5b"] for y, v in rep["kept"].items()}
    t = build()
    print(f"band: {len(t):,} rows, {t.cik.nunique():,} companies, median {t.groupby('date').size().median():.0f}/date; "
          "coverage " + " ".join(f"{f} {t[f].notna().mean():.0%}" for f in FEATS))
    parts = []
    for y in TEST_YEARS:
        k = kept[y]
        g = t[t.date.dt.year == y].copy()
        g["c2"] = (sum(sg * g["R_" + f].fillna(0) for f, (sg, _) in k.items()) / len(k)) if k else np.nan
        parts.append(g)
        print(f"{y}: frozen v5b set {sorted(k) or 'none (score missing -> 1x)'}")
    s = pd.concat(parts)
    s = s[s.c2.notna()]
    ic = s.groupby("date").apply(lambda g: g.c2.corr(g.yw, method="spearman"), include_groups=False).dropna()
    h1, h2 = ic[ic.index.year <= 2022], ic[ic.index.year >= 2023]
    dates = sorted(s.date.unique())[::2]
    act = tilt_active(s, "c2", dates)
    rng = np.random.default_rng(99)
    luck = np.array([tilt_active(s.assign(rnd=rng.random(len(s))), "rnd", dates) for _ in range(300)])
    pct = float((luck < act).mean() * 100)
    q = s.groupby("date").apply(lambda g: g[g.c2.rank(pct=True) > 0.8].y.mean() - g[g.c2.rank(pct=True) <= 0.2].y.mean(),
                                include_groups=False)
    verdict = nw_t(ic) >= 2.0 and act > 0 and pct >= 90
    print(f"\nOOS IC {ic.mean():+.4f} (NW t {nw_t(ic):+.2f}) | 2019-22 {h1.mean():+.4f} | 2023-26 {h2.mean():+.4f}")
    print(f"tilt vs band cap-weighted benchmark: {act:+.2%}/yr, {pct:.0f}th pct of 300 random tilts "
          f"(random median {np.median(luck):+.2%}, 95th {np.percentile(luck, 95):+.2%})")
    print(f"reporting: top-minus-bottom quintile 60-session excess {q.mean():+.2%} (positive on {(q > 0).mean():.0%} of dates)")
    print(f"\n=> {'CARRIES OVER to $250M-$1B' if verdict else 'DOES NOT carry over to $250M-$1B'} "
          f"(needs IC t >= 2 AND tilt > 0 at >= 90th pct)")
    json.dump({"ic": float(ic.mean()), "ic_t": float(nw_t(ic)), "ic_h1": float(h1.mean()), "ic_h2": float(h2.mean()),
               "tilt_active": float(act), "tilt_pct": pct, "quintile_spread": float(q.mean()), "carries_over": bool(verdict)},
              open(OUT / "report_c2_smallcap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
