# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false
"""H8 analyst rating momentum (pre-registered in docs/phase2-pit-backtest-plan.md,
"H8 pre-registration").

AM for company X on panel date D: FMP grade events dated strictly before D in the prior
90 calendar days, each broker's latest event; net = #upgrade - #downgrade. Missing unless
>= 3 distinct brokers had events in the prior 365 days. UP = net >= 1, DOWN = net <= -1.
Grades are per ticker; an event counts for a CIK only inside that ticker's dated segment
(symbol_segments), so a reused ticker never leaks another company's ratings.

Usage:
  python3 p8_analyst.py --coverage-only   # coverage + group sizes, NO returns (safe while fetching)
  python3 p8_analyst.py                   # the single pre-registered run (needs the fetch done)
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from common import FMP_DIR, META_DIR
from p4_tests import RT_COST, TOP_N, line, turnover

WIN, COVER, MIN_BROKERS = 90, 365, 3


def load_events(ciks):
    seg = pd.read_csv(META_DIR / "symbol_segments.csv.gz", parse_dates=["seg_from", "seg_to"])
    seg = seg[seg.cik.isin(ciks)]
    rows, missing = [], 0
    for sym, g in seg.groupby("symbol"):
        f = FMP_DIR / "grades" / f"{sym}.json"
        if not f.exists():
            missing += 1
            continue
        ev = json.load(open(f)) or []
        if not isinstance(ev, list) or not ev:
            continue
        e = pd.DataFrame(ev)
        if not {"date", "gradingCompany", "action"} <= set(e.columns):
            continue
        e["date"] = pd.to_datetime(e.date, errors="coerce")
        e = e.dropna(subset=["date"])
        for s in g.itertuples():
            m = e[(e.date >= s.seg_from) & (e.date <= s.seg_to)]
            if len(m):
                rows.append(pd.DataFrame({"cik": s.cik, "date": m.date.values,
                                          "broker": m.gradingCompany.str.strip().str.lower().values,
                                          "action": m.action.str.strip().str.lower().values}))
    ev = pd.concat(rows, ignore_index=True).drop_duplicates() if rows else pd.DataFrame(
        columns=["cik", "date", "broker", "action"])
    return ev, missing


def am_signal(panel, ev):
    """Per (cik, date) in panel: n_brokers_365, net_90 (NaN where not covered)."""
    ev = ev.sort_values("date")
    out = []
    for d, g in panel.groupby("date"):
        cover = ev[(ev.date < d) & (ev.date >= d - pd.Timedelta(days=COVER)) & ev.cik.isin(g.cik)]
        nb = cover.groupby("cik").broker.nunique()
        recent = cover[cover.date >= d - pd.Timedelta(days=WIN)]
        last = recent.groupby(["cik", "broker"]).action.last()
        net = (last == "upgrade").groupby("cik").sum() - (last == "downgrade").groupby("cik").sum()
        x = pd.DataFrame({"cik": g.cik.values, "date": d})
        x["n_brokers"] = x.cik.map(nb).fillna(0).astype(int)
        x["am"] = x.cik.map(net).fillna(0).astype(float).where(x.n_brokers >= MIN_BROKERS)
        out.append(x)
    return pd.concat(out, ignore_index=True)


def groups(d):
    g = d[d.am.notna()].copy()
    g["grp"] = np.select([g.am >= 1, g.am <= -1], ["UP", "DOWN"], "FLAT")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-only", action="store_true")
    a = ap.parse_args()
    from m3_signal_tests import load
    d = load()
    d = d[d.groupby("date").mcap.rank(ascending=False, method="first") <= TOP_N].copy()
    ev, missing = load_events(set(d.cik))
    print(f"universe: top {TOP_N}/date, {len(d):,} rows, {d.date.nunique()} dates; "
          f"grade events {len(ev):,} on {ev.cik.nunique():,} CIKs; symbols without a grades file: {missing}")
    print("actions:", ev.action.value_counts().head(8).to_dict())
    d = d.merge(am_signal(d[["cik", "date"]], ev), on=["cik", "date"], how="left")
    g = groups(d)

    cov = d.groupby(d.date.dt.year).am.apply(lambda s: s.notna().mean())
    print("\ncoverage (AM present, share of universe) by year:")
    print("  " + "  ".join(f"{y} {c:.0%}" for y, c in cov.items()))
    cov_ok = bool((cov >= 0.60).all())
    sizes = g.groupby(["date", "grp"]).size().unstack().mean()
    print(f"mean names per date: {sizes.round(0).to_dict()}  |  coverage gate (>=60% every year): "
          f"{'PASS' if cov_ok else 'FAIL -> exploratory only'}")
    if a.coverage_only:
        return

    if "done:" not in open(META_DIR / "m1_grades.log").read():
        sys.exit("grades fetch not finished -- the pre-registered run needs the full fetch")
    for h in (20, 60):
        d[f"x{h}"] = d[f"r{h}"] - d.groupby("date")[f"r{h}"].transform("mean")
    g = groups(d)
    per = {h: g[g[f"x{h}"].notna()].groupby(["date", "grp"])[f"x{h}"].mean().unstack() for h in (20, 60)}

    print("\nH8 AM analyst rating momentum")
    m, t, tr, te = line("(a) UP - DOWN 60d (NW t)", (per[60].UP - per[60].DOWN).dropna())
    a_ok = m > 0 and t >= 3.0 and tr > 0 and te > 0
    up = g[(g.grp == "UP") & g.r60.notna()]
    b1 = (up.groupby("date").r60.mean() - up.groupby("date").spy60.first()
          - turnover(up.groupby("date").cik.apply(set)) * RT_COST).dropna()
    keep = d[d.r60.notna() & ~d.index.isin(g[g.grp == "DOWN"].index)].copy()
    keep["w"] = keep.mcap / keep.groupby("date").mcap.transform("sum")
    wts = keep.groupby("date")[["cik", "w"]].apply(lambda x: dict(zip(x.cik, x.w)))
    b2 = ((keep.w * keep.r60).groupby(keep.date).sum() - keep.groupby("date").spy60.first()
          - turnover(keep.groupby("date").cik.apply(set), wts) * RT_COST).dropna()
    _, _, b1tr, b1te = line("(b1) UP EW - SPY, net", b1)
    _, _, b2tr, b2te = line("(b2) cap-wt ex-DOWN - SPY, net", b2)
    b_ok = (b1tr > 0 and b1te > 0) or (b2tr > 0 and b2te > 0)
    print("  reporting only:")
    line("UP - DOWN 20d (NW t)", (per[20].UP - per[20].DOWN).dropna())
    line("FLAT excess 60d", per[60].FLAT.dropna())
    line("UP excess 60d", per[60].UP.dropna())
    line("DOWN excess 60d", per[60].DOWN.dropna())
    d = d.merge(pd.read_csv(META_DIR / "p3_signals.csv.gz", parse_dates=["date"])[["cik", "date", "s1"]],
                on=["cik", "date"], how="left")
    rk = lambda c: d.groupby("date")[c].rank(pct=True)  # noqa: E731
    d["am_pead"] = pd.concat([rk("am"), rk("s1")], axis=1).mean(axis=1).where(d.am.notna() & d.s1.notna())
    from p4_tests import quint, spread
    line("AM + PEAD composite top-bot 60d (in-sample)", spread(quint(d, "am_pead"), 60))
    rc = d[["am", "mcap"]].assign(mcap=np.log(d.mcap)).corr("spearman").iloc[0, 1]
    print(f"  rank corr AM vs log mcap {rc:+.2f}")
    ok = cov_ok and a_ok and b_ok
    print(f"\n=> H8 {'PASS -> portfolio simulation + forward paper only' if ok else 'FAIL'} "
          f"(coverage {'ok' if cov_ok else 'no'}, (a) {'ok' if a_ok else 'no'}, (b) {'ok' if b_ok else 'no'})")


if __name__ == "__main__":
    main()
