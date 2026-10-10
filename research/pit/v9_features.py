# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v9 volume features for every (cik, date) of the v3 table (docs/score-v3-plan.md,
"Score v9 pre-registration", committed 1f21069 before any number). Trial 11.

vspike    log(mean volume last 5 sessions / mean volume sessions 6-55 before D)
turnover  mean daily share volume, last 252 sessions / shares outstanding. Volume in the price
          files is split-adjusted to TODAY's basis, so shares are put on the same basis:
          market cap on D / split-adjusted close on D (= share volume / shares, one basis).
ear_vol   log(mean volume E-1..E+1 / mean volume E-60..E-11), E = latest FMP report whose next
          session is before D and within 63 sessions of D
(lowvol_mom = R(s4) - R(turnover) is built in v9_model.py from the ranks.)

  python3 v9_features.py [--workers 10] [--limit N]
Output: META_DIR/v9_features.csv.gz
"""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import FMP_DIR, META_DIR

FEATS = ["vspike", "turnover", "ear_vol"]
PANEL = None


def setup():
    global PANEL
    PANEL = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date", "mcap"], parse_dates=["date"])
    m.identity()
    from common import _segments
    _segments()


_EARN: dict = {}


def report_dates(sym):
    if sym not in _EARN:
        try:
            rows = json.load(open(FMP_DIR / "earnings" / f"{sym}.json")) or []
        except (OSError, ValueError):
            rows = []
        _EARN[sym] = sorted(pd.Timestamp(r["date"]) for r in rows if r.get("date") and r.get("epsActual") is not None)
    return _EARN[sym]


def feats(px, k, mcap):
    v, c = px.Volume.to_numpy(dtype=float), px.Close.to_numpy(dtype=float)
    out = {f: np.nan for f in FEATS}
    if k >= 55:
        recent, base = np.nanmean(v[k - 4:k + 1]), np.nanmean(v[k - 55:k - 5])
        if recent > 0 and base > 0:
            out["vspike"] = float(np.log(recent / base))
    if k >= 251 and mcap and c[k] > 0:
        shares_adj = mcap / c[k]
        if shares_adj > 0:
            out["turnover"] = float(np.nanmean(v[k - 251:k + 1]) / shares_adj)
    sym = px.symbol.iloc[k]
    dates = [e for e in report_dates(sym) if e <= px.index[k]]
    if dates:
        ie = px.index.searchsorted(dates[-1])          # first session on/after the report date
        if ie + 1 < k and k - ie <= 63 and ie >= 60:
            ev, bv = np.nanmean(v[ie - 1:ie + 2]), np.nanmean(v[ie - 60:ie - 10])
            if ev > 0 and bv > 0:
                out["ear_vol"] = float(np.log(ev / bv))
    return out


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            if px.empty:
                continue
            for r in PANEL[PANEL.cik == cik].itertuples():
                k = px.index.searchsorted(r.date, side="right") - 1
                if k < 0:
                    continue
                rows.append({"cik": cik, "date": r.date, **feats(px, k, r.mcap)})
        except Exception as e:
            rows.append({"cik": cik, "date": None, "error": repr(e)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    setup()
    ciks = sorted(PANEL.cik.unique())
    if a.limit:
        ciks = ciks[:: max(1, len(ciks) // a.limit)][: a.limit]
    chunks = [ciks[i:i + 20] for i in range(0, len(ciks), 20)]
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(work, chunks), 1):
            out += rows
            if i % 40 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    name = "v9_features_sample.csv.gz" if a.limit else "v9_features.csv.gz"
    df.to_csv(META_DIR / name, index=False)
    print(f"wrote {len(df):,} rows; errors {len(errs)}; {time.time() - t0:.0f}s", flush=True)
    print("coverage: " + "  ".join(f"{f} {df[f].notna().mean():.0%}" for f in FEATS))
    print(df[FEATS].describe().T[["mean", "std", "min", "50%", "max"]].to_string(float_format=lambda v: f"{v:.4f}"))
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
