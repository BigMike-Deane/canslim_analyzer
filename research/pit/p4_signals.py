# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""Batch 2 signals on the M3 panel (pre-registered in
docs/phase2-pit-backtest-plan.md, "Batch 2 / Score v2 pre-registration").

ep     sum of latest 4 quarterly diluted EPS as known on D / actual close
bm     latest StockholdersEquity (filed <= D) / market cap
lv     -stdev of daily returns, last 252 sessions (>= 200 returns)
roe    latest 10-K FY net income / equity at that FY end (equity > 0)
nacc   -(net income - operating cash flow) / assets, same 10-K FY

  python3 p4_signals.py [--workers W] [--limit N]
Output: META_DIR/p4_signals.csv.gz (cik, date, ep, bm, lv, roe, nacc)
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import META_DIR

PANEL = None
FUND: dict = {}


def _read(path, concepts):
    f = pd.read_csv(path, low_memory=False, usecols=["cik", "concept", "unit", "start", "end", "val", "filed"])
    f = f[f.concept.isin(concepts) & (f.unit == "USD")]
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c], errors="coerce")
    return f.dropna(subset=["end", "filed", "val"])


def setup():
    global PANEL, FUND
    PANEL = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik", "date", "close", "mcap"], parse_dates=["date"])
    f = pd.concat([_read(META_DIR / "sec_facts.csv.gz", ["NetIncomeLoss", "StockholdersEquity"]),
                   _read(META_DIR / "sec_facts_q.csv.gz", ["Assets", "NetCashProvidedByUsedInOperatingActivities"])])
    f["days"] = (f.end - f.start).dt.days
    FUND = {cik: g for cik, g in f.groupby("cik")}
    m._facts(); m.identity()
    from common import _segments
    _segments()


def latest_val(rows, d, end=None):
    """Value as known on d: for `end` if given, else the newest period end."""
    r = rows[rows.filed <= d]
    if end is not None:
        r = r[r.end == end]
    if r.empty:
        return np.nan, None
    r = r.sort_values(["end", "filed"])
    return r.val.iloc[-1], r.end.iloc[-1]


def ttm_eps(cik, d):
    t = m._eps_table(cik)
    if t.empty:
        return np.nan
    q = t[(t.kind == "Q") & (t.filed <= d)].sort_values("filed").drop_duplicates("end", keep="last")
    q = q.sort_values("end", ascending=False).head(4)
    if len(q) < 4 or q.end.iloc[0] < d - pd.Timedelta(days=456) or (q.end.iloc[0] - q.end.iloc[3]).days > 300:
        return np.nan
    return q.val.sum()


def fundamentals(cik, d, mcap):
    out = {"bm": np.nan, "roe": np.nan, "nacc": np.nan}
    f = FUND.get(cik)
    if f is None:
        return out
    eq = f[(f.concept == "StockholdersEquity") & f.start.isna()]
    e, _ = latest_val(eq, d)
    if not np.isnan(e) and mcap > 0:
        out["bm"] = e / mcap
    ni_fy = f[(f.concept == "NetIncomeLoss") & f.days.between(350, 380)]
    ni, end = latest_val(ni_fy, d)
    if end is None:
        return out
    e_fy, _ = latest_val(eq, d, end)
    if e_fy and e_fy > 0:
        out["roe"] = ni / e_fy
    cfo, _ = latest_val(f[(f.concept == "NetCashProvidedByUsedInOperatingActivities") & f.days.between(350, 380)], d, end)
    assets, _ = latest_val(f[(f.concept == "Assets") & f.start.isna()], d, end)
    if not np.isnan(cfo) and assets and assets > 0 and abs((ni - cfo) / assets) <= 2:
        out["nacc"] = -(ni - cfo) / assets  # |accruals| > 2x assets = unit error (REAL), dropped
    return out


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            for r in PANEL[PANEL.cik == cik].itertuples():
                d = r.date
                rec = {"cik": cik, "date": d, "ep": np.nan, "lv": np.nan, **fundamentals(cik, d, r.mcap)}
                eps = ttm_eps(cik, d)
                if not np.isnan(eps) and r.close > 0 and abs(eps / r.close) <= 5:
                    rec["ep"] = eps / r.close  # |E/P| > 5 = SEC unit error (NPO), dropped
                k = px.index.searchsorted(d, side="right") - 1
                if k >= 200:
                    ret = px.Close.iloc[max(0, k - 252):k + 1].pct_change().dropna()
                    if len(ret) >= 200:
                        rec["lv"] = -ret.std()
                rows.append(rec)
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
    print(f"{len(ciks):,} CIKs; setup {time.time() - t0:.0f}s", flush=True)
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, chunks), 1):
            out += r
            if i % 20 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    name = "p4_signals_sample.csv.gz" if a.limit else "p4_signals.csv.gz"
    df.to_csv(META_DIR / name, index=False)
    cols = ["ep", "bm", "lv", "roe", "nacc"]
    print(f"wrote {len(df):,} rows; errors {len(errs)}; coverage " +
          " ".join(f"{c} {df[c].notna().mean():.0%}" for c in cols), flush=True)
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
