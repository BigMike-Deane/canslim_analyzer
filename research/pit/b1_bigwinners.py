# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Big-winner study Part 1 (docs/big-winner-plan.md, pre-registered 2026-10-08):
can a signal known on day D pick the stocks that gain >= +50% over the next 126
sessions more often than it picks the ones that lose >= 25%?

Panel = the M3 universe (every 10th session from 2016-01, close > $3, mcap >= $300M,
delisted names included) + the daily PIT panel's earnings fields. Forward returns
and price features come from the same stitched per-company price series.

  python3 b1_bigwinners.py [--workers 8]
Output: META_DIR/bigwin/b1_panel.csv.gz, b1_results.json; printed table.
"""
import argparse
import glob
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import META_DIR

H, BW, BL = 126, 0.50, -0.25
LAST_ENTRY = "2026-03-31"
HALVES = (("2016-01-01", "2020-12-31"), ("2021-01-01", LAST_ENTRY))
CANDIDATES = ["total", "beat_streak", "surprise_pct", "earn_mom", "mom6", "near_high"]
CONTROLS = ["vol60", "max21"]
GATE = {"win_lift": 1.5, "skew_lift": 1.25}

PANEL: pd.DataFrame = pd.DataFrame()


def work(ciks):
    out = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            if px.empty:
                continue
            c, syms, idx = px.Close.to_numpy(float), px.symbol.to_numpy(), px.index
            ret = np.r_[np.nan, c[1:] / c[:-1] - 1]
            for d in PANEL.loc[PANEL.cik == cik, "date"]:
                k = idx.searchsorted(d, side="right") - 1
                if k < 0:
                    continue
                rec = {"cik": cik, "date": d}
                j = k + H
                if j < len(c) and len(set(syms[k:j + 1])) == 1:
                    rec["r126"] = c[j] / c[k] - 1
                elif j >= len(c) and idx[-1] < pd.Timestamp("2026-09-01") and len(set(syms[k:])) == 1:
                    rec["r126"] = c[-1] / c[k] - 1          # delisted inside the window: last price
                if k >= 126:
                    rec["mom6"] = c[k - 21] / c[k - 126] - 1
                if k >= 251:
                    rec["near_high"] = c[k] / np.nanmax(c[k - 251:k + 1])
                if k >= 60:
                    rec["vol60"] = np.nanstd(ret[k - 59:k + 1]) * np.sqrt(252)
                    rec["max21"] = np.nanmax(ret[k - 20:k + 1])
                out.append(rec)
        except Exception as e:  # one bad company must not kill the panel
            out.append({"cik": cik, "date": None, "error": repr(e)[:200]})
    return out


def build(workers):
    global PANEL
    p = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik", "date", "symbol", "close", "mcap", "total"],
                    parse_dates=["date"])
    p = p[p.date <= LAST_ENTRY]
    dates = set(p.date.dt.strftime("%Y-%m-%d"))
    parts = []
    for f in sorted(glob.glob(str(META_DIR / "daily" / "*.csv.gz"))):
        d = pd.read_csv(f, usecols=["cik", "date", "beat_streak", "surprise_pct", "eps_growth"])
        parts.append(d[d.date.isin(dates)])
    earn = pd.concat(parts)
    earn["date"] = pd.to_datetime(earn.date)
    PANEL = p.merge(earn, on=["cik", "date"], how="left")
    m.identity()
    from common import _segments
    _segments()
    ciks = sorted(PANEL.cik.unique())
    chunks = [ciks[i:i + 25] for i in range(0, len(ciks), 25)]
    rows = []
    with mp.get_context("fork").Pool(workers) as pool:
        for rs in pool.imap_unordered(work, chunks):
            rows += rs
    f = pd.DataFrame(rows)
    errs = int(f["error"].notna().sum()) if "error" in f.columns else 0
    f = f[f.date.notna()].drop(columns=["error"], errors="ignore")
    f["date"] = pd.to_datetime(f.date)
    out = PANEL.merge(f, on=["cik", "date"], how="left")
    print(f"panel {len(out):,} rows, {out.cik.nunique():,} companies, {out.date.nunique()} dates; "
          f"r126 known {out.r126.notna().mean():.1%}; errors {errs}", flush=True)
    return out


def evaluate(df):
    df = df[df.r126.notna()].copy()
    df["bw"], df["bl"] = df.r126 >= BW, df.r126 <= BL
    g = df.groupby("date")
    for col in ("beat_streak", "surprise_pct", "eps_growth"):
        df[f"_{col}"] = g[col].rank(pct=True)
    df["earn_mom"] = df[["_beat_streak", "_surprise_pct", "_eps_growth"]].mean(axis=1, skipna=False)
    res = {}
    for sig in CANDIDATES + CONTROLS:
        res[sig] = {}
        for lo, hi in HALVES:
            x = df[(df.date >= lo) & (df.date <= hi) & df[sig].notna()].copy()
            x["pr"] = x.groupby("date")[sig].rank(pct=True, method="average")
            top = x[x.pr >= 0.90]
            pbw, pbl, tbw, tbl = x.bw.mean(), x.bl.mean(), top.bw.mean(), top.bl.mean()
            win_lift = tbw / pbw if pbw else np.nan
            skew_lift = (tbw / tbl) / (pbw / pbl) if tbl and pbl and pbw else np.nan
            res[sig][lo[:4]] = dict(n_all=len(x), n_top=len(top), p_bw_all=pbw, p_bl_all=pbl, p_bw_top=tbw,
                                    p_bl_top=tbl, win_lift=win_lift, skew_lift=skew_lift,
                                    mean_top=top.r126.mean(), mean_all=x.r126.mean(),
                                    median_top=top.r126.median(), median_all=x.r126.median())
        h = res[sig].values()
        res[sig]["pass"] = bool(sig in CANDIDATES and all(
            v["win_lift"] >= GATE["win_lift"] and v["skew_lift"] >= GATE["skew_lift"] and v["mean_top"] > v["mean_all"]
            for v in h))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="skip the build if b1_panel.csv.gz exists")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = META_DIR / "bigwin"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "b1_panel.csv.gz"
    if a.reuse and path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = build(a.workers)
        df.to_csv(path, index=False)
    res = evaluate(df)
    json.dump(res, open(out_dir / "b1_results.json", "w"), indent=1, default=float)
    print(f"\nBig winner = +{BW:.0%} in {H} sessions; big loser = {BL:.0%}. Top group = top 10% per date.")
    for sig, r in res.items():
        tag = "PASS" if r["pass"] else ("control" if sig in CONTROLS else "fail")
        cells = " | ".join(f"{k}: BW {v['p_bw_top']:.1%} vs {v['p_bw_all']:.1%} (lift {v['win_lift']:.2f}), "
                           f"BL {v['p_bl_top']:.1%} vs {v['p_bl_all']:.1%}, skew {v['skew_lift']:.2f}, "
                           f"mean {v['mean_top']:+.1%} vs {v['mean_all']:+.1%}"
                           for k, v in r.items() if k != "pass")
        print(f"  {sig:<13} {tag:<7} {cells}")
    print(f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
