# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Score v3 price/volume features for every (cik, date) row of the v3 panel
(docs/score-v3-plan.md). Stitched per-CIK series (m2_adapter.cik_prices),
everything as of D's close (the row the forward return starts from).

ret1m, mom3, mom6, dist52 (close / 252d high - 1), vol60, max21, beta252 (vs
SPY), dvol20 ($ volume, 20d mean; universe filter), voltrend (20d / 120d mean
volume).

  python3 v3_price_features.py [--workers 8]
Output: META_DIR/v3_price_features.csv.gz
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from m3_signal_tests import load

PANEL = None
SPYR = None


def setup():
    global PANEL, SPYR
    PANEL = load()[["cik", "date"]]
    SPYR = m.prices("SPY").Close.pct_change()
    from common import _segments
    _segments()


def feats(px, k):
    c, v = px.Close.to_numpy(), px.Volume.to_numpy()
    hi = px.High.to_numpy() if "High" in px else c
    r = {}
    for name, lb in (("ret1m", 21), ("mom3", 63), ("mom6", 126)):
        r[name] = c[k] / c[k - lb] - 1 if k >= lb and c[k - lb] > 0 else np.nan
    if k >= 200:
        r["dist52"] = c[k] / np.nanmax(hi[max(0, k - 251):k + 1]) - 1
    dr = np.diff(c[max(0, k - 252):k + 1]) / c[max(0, k - 252):k]
    if len(dr) >= 60:
        r["vol60"] = np.nanstd(dr[-60:])
        r["max21"] = np.nanmax(dr[-21:])
    if len(dr) >= 200:
        idx = px.index[max(0, k - 252) + 1:k + 1]
        s = SPYR.reindex(idx).to_numpy()
        ok = ~(np.isnan(dr) | np.isnan(s))
        if ok.sum() >= 150 and np.var(s[ok]) > 0:
            r["beta252"] = np.cov(dr[ok], s[ok])[0, 1] / np.var(s[ok])
    if k >= 120:
        r["dvol20"] = np.nanmean(c[k - 19:k + 1] * v[k - 19:k + 1])
        v120 = np.nanmean(v[k - 119:k + 1])
        r["voltrend"] = np.nanmean(v[k - 19:k + 1]) / v120 if v120 > 0 else np.nan
    return r


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            if px.empty:
                continue
            for d in PANEL.date[PANEL.cik == cik]:
                k = px.index.searchsorted(d, side="right") - 1
                if k < 21:
                    continue
                rows.append({"cik": cik, "date": d, **feats(px, k)})
        except Exception as e:
            rows.append({"cik": cik, "date": None, "error": repr(e)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    setup()
    ciks = sorted(PANEL.cik.unique())
    if a.limit:
        ciks = ciks[:: max(1, len(ciks) // a.limit)][: a.limit]
    chunks = [ciks[i:i + 25] for i in range(0, len(ciks), 25)]
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(work, chunks), 1):
            out += rows
            if i % 40 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    name = "v3_price_features_sample.csv.gz" if a.limit else "v3_price_features.csv.gz"
    df.to_csv(m.META_DIR / name, index=False)
    print(f"wrote {len(df):,} rows; errors {len(errs)}; {time.time() - t0:.0f}s", flush=True)
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
