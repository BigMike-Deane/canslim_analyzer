# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""Phase 3 signal lab: S1-S4 on the M3 panel (pre-registered in
docs/phase2-pit-backtest-plan.md, "Phase 3 signal-lab pre-registration").

S1 PEAD        (epsActual - epsEstimated) / split-adjusted close on D, latest
               FMP report dated strictly before D and within 63 sessions
S2 GP/A        latest 10-K FY (filed <= D) gross profit / assets at FY end
S3 issuance    -log(shares_now / (shares_then x splits between)), dei shares
               as known on D vs D-365d; splits across all the CIK's symbols;
               both counts filed within 150d of their date;
               missing if a bankruptcy (xxxxQ) ticker trades in the window
S4 mom 12-1    close(D-21) / close(D-252) - 1, stitched series

  python3 p3_signals.py [--workers W]
Output: META_DIR/p3_signals.csv.gz (cik, date, s1..s4)
"""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import FMP_DIR, META_DIR, split_factor

REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
COST = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"]

PANEL = None
FUND: dict = {}
SESS = None


def _read(path, concepts):
    f = pd.read_csv(path, low_memory=False, usecols=["cik", "concept", "unit", "start", "end", "val", "filed"])
    f = f[f.concept.isin(concepts) & (f.unit == "USD")]
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c], errors="coerce")
    return f.dropna(subset=["end", "filed", "val"])


def setup():
    global PANEL, FUND, SESS
    PANEL = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik", "date", "symbol"], parse_dates=["date"])
    f = pd.concat([_read(META_DIR / "sec_facts.csv.gz", REV),
                   _read(META_DIR / "sec_facts_q.csv.gz", ["GrossProfit", "Assets", *COST])])
    f["days"] = (f.end - f.start).dt.days
    FUND = {cik: g for cik, g in f.groupby("cik")}
    SESS = m.prices("SPY").index
    m._facts(); m.identity()
    from common import _segments
    _segments()


def asof(rows, end, d):
    """Value for period `end` as known on d (latest filing <= d)."""
    r = rows[(rows.end == end) & (rows.filed <= d)]
    return r.sort_values("filed").val.iloc[-1] if len(r) else np.nan


def gpa(cik, d):
    f = FUND.get(cik)
    if f is None:
        return np.nan
    known = f[f.filed <= d]
    fy = known[known.days.between(350, 380)]
    if fy.empty:
        return np.nan
    end = fy.end.max()
    per = fy[fy.end == end]
    gp = asof(per[per.concept == "GrossProfit"], end, d)
    if np.isnan(gp):
        rev = next((v for c in REV if not np.isnan(v := asof(per[per.concept == c], end, d))), np.nan)
        cost = next((v for c in COST if not np.isnan(v := asof(per[per.concept == c], end, d))), np.nan)
        gp = rev - cost
    assets = asof(known[(known.concept == "Assets") & known.start.isna()], end, d)
    return gp / assets if assets and assets > 0 and not np.isnan(gp) else np.nan


def pead(d, adj_close, reports):
    ds = d.strftime("%Y-%m-%d")
    past = [x for x in reports if x["date"] < ds]
    if not past or not adj_close or adj_close <= 0:
        return np.nan
    x = past[-1]
    if SESS.searchsorted(d) - SESS.searchsorted(pd.Timestamp(x["date"]), side="right") + 1 > 63:
        return np.nan
    return (x["epsActual"] - x["epsEstimated"]) / adj_close


STALE = pd.Timedelta(days=150)


def splits_between(cik, t0, t1):
    """Product of split ratios in (t0, t1] across EVERY symbol the company used
    (a reverse split is often filed under the later ticker); one count per date."""
    from common import _SPLITS, _segments
    seen = {}
    for sym in set(_segments().get(cik, pd.DataFrame(columns=["symbol"])).symbol):
        split_factor(sym, t1)  # loads _SPLITS[sym]
        for when, ratio in _SPLITS.get(sym, []):
            if t0 < when <= t1:
                seen.setdefault(when.normalize(), ratio)
    return float(np.prod(list(seen.values()))) if seen else 1.0


def issuance(cik, sh, d):
    """-log(shares now / split-adjusted shares a year ago); both counts must be
    fresh (filed within 150 days of the date they stand for)."""
    t0 = d - pd.Timedelta(days=365)
    from common import _segments
    seg = _segments().get(cik)
    if seg is not None and ((seg.symbol.str.fullmatch(r"[A-Z]{4}Q")) & (seg.seg_from <= d) & (seg.seg_to > t0)).any():
        return np.nan  # bankruptcy ticker in the window: old equity cancelled, not a buyback
    now, then = sh[sh.filed <= d], sh[sh.filed <= t0]
    if now.empty or then.empty or d - now.filed.iloc[-1] > STALE or t0 - then.filed.iloc[-1] > STALE:
        return np.nan
    sn, st = now.val.iloc[-1], then.val.iloc[-1] * splits_between(cik, t0, d)
    return -np.log(sn / st) if sn > 0 and st > 0 else np.nan


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            dates = PANEL[PANEL.cik == cik]
            sym = dates.symbol.iloc[0]
            px = m.cik_prices(cik)
            sh = m._shares_table(cik)
            p = FMP_DIR / "earnings" / f"{sym}.json"
            rep = sorted((x for x in (json.loads(p.read_text()) if p.exists() else [])
                          if x.get("date") and x.get("epsActual") is not None and x.get("epsEstimated") is not None),
                         key=lambda x: x["date"])
            for d in dates.date:
                k = px.index.searchsorted(d, side="right") - 1
                rec = {"cik": cik, "date": d, "s1": np.nan, "s2": gpa(cik, d), "s3": np.nan, "s4": np.nan}
                if k >= 0:
                    rec["s1"] = pead(d, px.Close.iloc[k], rep)
                    if k >= 252:
                        rec["s4"] = px.Close.iloc[k - 21] / px.Close.iloc[k - 252] - 1
                rec["s3"] = issuance(cik, sh, d)
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
    print(f"{len(ciks):,} CIKs, {len(PANEL):,} panel rows; setup {time.time() - t0:.0f}s", flush=True)
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, chunks), 1):
            out += r
            if i % 20 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    name = "p3_signals_sample.csv.gz" if a.limit else "p3_signals.csv.gz"
    df.to_csv(META_DIR / name, index=False)
    cov = df[["s1", "s2", "s3", "s4"]].notna().mean()
    print(f"wrote {len(df):,} rows; errors {len(errs)}; coverage " +
          " ".join(f"{c} {v:.0%}" for c, v in cov.items()), flush=True)
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
