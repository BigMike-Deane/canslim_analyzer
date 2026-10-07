# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false
"""Score v6 features (docs/score-v3-plan.md, "Score v6 pre-registration") for every
(cik, date) row of the v3 table:

ear              stock return - SPY return, close of session E-2 -> close of E+1, for the
                 latest FMP report date E with E+1 < D (window spans before/after-bell timing)
days_since       D - E in calendar days (missing EAR if > 120)

  python3 v6_features.py [--workers 10]   -> META_DIR/v6_features.csv.gz
"""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import FMP_DIR, META_DIR

TABLE = None
SPY = None


def setup():
    global TABLE, SPY
    TABLE = pd.read_csv(META_DIR / "v3_table.csv.gz", usecols=["cik", "date"], parse_dates=["date"])
    SPY = m.prices("SPY").Close
    from common import _segments
    _segments()


def report_dates(symbols):
    out = set()
    for s in symbols:
        f = FMP_DIR / "earnings" / f"{s}.json"
        if f.exists():
            for x in json.load(open(f)) or []:
                if x.get("date") and x.get("epsActual") is not None:  # a report that actually happened
                    out.add(pd.Timestamp(x["date"]))
    return sorted(out)


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            if px.empty:
                continue
            idx, c = px.index, px.Close
            reps = report_dates(px.symbol.unique())
            # EAR per report, with the session after which it is known (E+1)
            ears = []
            for e in reps:
                k = idx.searchsorted(e)          # first session >= E
                if k < 2 or k + 1 >= len(idx):
                    continue
                a, b = idx[k - 2], idx[k + 1]
                if px.symbol.iloc[k - 2] != px.symbol.iloc[k + 1]:
                    continue
                spy_a, spy_b = SPY.asof(a), SPY.asof(b)
                ears.append((b, e, c.iloc[k + 1] / c.iloc[k - 2] - spy_b / spy_a))
            known = pd.DataFrame(ears, columns=["known", "e", "ear"]).sort_values("known") if ears else None
            for d in TABLE.date[TABLE.cik == cik]:
                rec = {"cik": cik, "date": d, "ear": np.nan, "days_since": np.nan}
                if known is not None:
                    k = known[known.known < d]
                    if len(k):
                        last = k.iloc[-1]
                        ds = (d - last.e).days
                        rec["days_since"] = ds
                        if ds <= 120:
                            rec["ear"] = last.ear
                rows.append(rec)
        except Exception as ex:
            rows.append({"cik": cik, "date": None, "error": repr(ex)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    setup()
    ciks = sorted(TABLE.cik.unique())
    chunks = [ciks[i:i + 25] for i in range(0, len(ciks), 25)]
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, chunks), 1):
            out += r
            if i % 40 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    df.to_csv(META_DIR / "v6_features.csv.gz", index=False)
    print(f"wrote {len(df):,} rows; EAR coverage {df.ear.notna().mean():.0%}; median |EAR| {df.ear.abs().median():.3f}; "
          f"errors {len(errs)}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
