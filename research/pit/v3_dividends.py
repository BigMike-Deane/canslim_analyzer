# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOperatorIssue=false
"""Score v3 amendment 1: cash dividends so stock returns are total return like SPY's.

FMP /stable/dividends per symbol (adjDividend = split-adjusted, same basis as the
stitched Close). For each v3-table row: div20 / div60 = sum of adjDividend with ex-date
in (D, D + h sessions] / Close at D, using the symbol the CIK traded under on each ex-date.

  python3 v3_dividends.py fetch     # resumable, scanner-aware FMP pacing
  python3 v3_dividends.py compute   # -> META_DIR/v3_dividends.csv.gz (cik, date, div20, div60)
"""
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import FMP_DIR, META_DIR, cached, fmp_get

DIV_DIR = FMP_DIR / "dividends"
TABLE = None
DIVS: dict = {}


def symbols():
    t = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik"])
    seg = pd.read_csv(META_DIR / "symbol_segments.csv.gz")
    return sorted(set(seg[seg.cik.isin(set(t.cik))].symbol))


def fetch():
    syms = symbols()
    print(f"dividends for {len(syms):,} symbols", flush=True)
    for i, s in enumerate(syms, 1):
        cached(DIV_DIR / f"{s}.json", lambda: fmp_get("dividends", symbol=s, limit=1000))
        if i % 500 == 0:
            print(f"  {i:,}/{len(syms):,}", flush=True)
    print("done: fetch", flush=True)


def load_divs(sym):
    import json
    f = DIV_DIR / f"{sym}.json"
    if not f.exists():
        return pd.Series(dtype=float)
    d = json.load(open(f)) or []
    if not isinstance(d, list) or not d:
        return pd.Series(dtype=float)
    x = pd.DataFrame(d)
    if "adjDividend" not in x:
        return pd.Series(dtype=float)
    x["date"] = pd.to_datetime(x.date, errors="coerce")
    return x.dropna(subset=["date"]).groupby("date").adjDividend.sum().sort_index()


def work(ciks):
    rows = []
    for cik in ciks:
        px = m.cik_prices(cik)
        if px.empty:
            continue
        divs = {s: load_divs(s) for s in px.symbol.unique()}
        idx = px.index
        for d in TABLE.date[TABLE.cik == cik]:
            k = idx.searchsorted(d, side="right") - 1
            if k < 0:
                continue
            c0 = px.Close.iloc[k]
            rec = {"cik": cik, "date": d}
            for h in (20, 60):
                j = min(k + h, len(idx) - 1)
                lo, hi = idx[k], idx[j]
                tot = 0.0
                for s, dv in divs.items():
                    if dv.empty:
                        continue
                    seg = px.symbol.iloc[k:j + 1]
                    if (seg == s).any():  # only ex-dates while the CIK traded as this symbol
                        days = idx[k:j + 1][seg.to_numpy() == s]
                        w = dv[(dv.index > lo) & (dv.index <= hi)]
                        w = w[(w.index >= days.min()) & (w.index <= days.max() + pd.Timedelta(days=4))]
                        tot += w.sum()
                rec[f"div{h}"] = tot / c0 if c0 > 0 else np.nan
            rows.append(rec)
    return rows


def compute():
    global TABLE
    TABLE = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date"], parse_dates=["date"])
    from common import _segments
    _segments()
    ciks = sorted(TABLE.cik.unique())
    chunks = [ciks[i:i + 25] for i in range(0, len(ciks), 25)]
    t0, out = time.time(), []
    with mp.get_context("fork").Pool(10) as pool:
        for i, r in enumerate(pool.imap_unordered(work, chunks), 1):
            out += r
            if i % 40 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(out)
    d.to_csv(META_DIR / "v3_dividends.csv.gz", index=False)
    print(f"done: {len(d):,} rows; median div60 {d.div60.median():.4f}, share >0 {(d.div60 > 0).mean():.0%}", flush=True)


if __name__ == "__main__":
    {"fetch": fetch, "compute": compute}[sys.argv[1]]()
