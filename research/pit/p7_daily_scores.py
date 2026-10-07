# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""H7 input: point-in-time CANSLIM scores for every eligible company on every
session 2016-01-04 -> 2026-10-05 (pre-registered in
docs/phase2-pit-backtest-plan.md, "H7 pre-registration").

Eligible = inside the identity window, fresh price on D, actual close > $3,
market cap >= $300M as known (actual close x SEC shares). Scores come from the
live CANSLIMScorer via m2_adapter (same as the M3 panel); eps_growth and
annual_cagr replicate backtester.py's projected_growth inputs. Per-date values
of the engine's static_data inputs ride along: surprise %, beat streak, 13F
institutional %, and days to the next FMP-dated earnings report.

  python3 p7_daily_scores.py [--workers W] [--limit N]
Output: META_DIR/daily/<chunk>.csv.gz (one per 25-CIK chunk; resumable)
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import json

import m2_adapter as m
from canslim_scorer import CANSLIMScorer
from common import FMP_DIR, META_DIR, split_factor

START, END = "2016-01-04", "2026-10-05"
MIN_PRICE, MIN_CAP = 3.0, 300e6
OUT = META_DIR / "daily"
SESS: list = []
MSCORE: dict = {}


def setup():
    global SESS, MSCORE
    SESS = list(m.prices("SPY").loc[START:END].index)
    MSCORE = {d: m.market_score_asof(d) for d in SESS}
    m._facts(); m._inst(); m.identity()
    from common import _segments
    _segments()


def eps_growth(q):
    """backtester._calculate_scores eps_growth (quarterly list, newest first)."""
    q = [e for e in q if e is not None and e == e]
    if len(q) >= 5:
        cur, pri = (sum(q[0:4]), sum(q[4:8])) if len(q) >= 8 else (q[0], q[4])
        if cur > 0 and pri < 0:
            return (cur - pri) / abs(pri) * 100
        if cur < 0:
            return 0.0
        if abs(pri) < 0.01:
            return 100.0 if cur > 0 else 0.0
        return (cur - pri) / abs(pri) * 100
    if len(q) >= 2:
        if q[1] > 0 and q[0] > 0:
            return (q[0] - q[1]) / abs(q[1]) * 100
        if q[0] > 0 and q[1] <= 0:
            return 50.0
    return 0.0


def annual_cagr(a):
    a = [e for e in a if e is not None and e == e]
    if len(a) >= 3:
        if a[2] > 0 and a[0] > 0:
            return ((a[0] / a[2]) ** 0.5 - 1) * 100
        if a[0] > 0 and a[2] <= 0:
            return 50.0
    return 0.0


def work(chunk):
    idx, ciks = chunk
    path = OUT / f"{idx:04d}.csv.gz"
    if path.exists():
        return idx, -1
    rows = []
    ident = m.identity()
    for cik in ciks:
        try:
            row = ident.loc[cik]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            px = m.cik_prices(cik)
            if px.empty:
                continue
            sh = m._shares_table(cik)
            ep = FMP_DIR / "earnings" / f"{row.symbol}.json"
            edates = sorted({x["date"] for x in (json.loads(ep.read_text()) if ep.exists() else []) if x.get("date")})
            for d in SESS:
                if not (row.valid_from <= d <= row.valid_to):
                    continue
                k = px.index.searchsorted(d, side="right") - 1
                if k < 0 or px.index[k] < d - pd.Timedelta(days=5):
                    continue
                close = px.Close.iloc[k] * split_factor(px.symbol.iloc[k], d)
                known = sh[sh.filed <= d]
                if close <= MIN_PRICE or known.empty or close * known.val.iloc[-1] < MIN_CAP:
                    continue
                sd = m.stock_data_asof(cik, d)
                if sd is None:
                    continue
                scorer = CANSLIMScorer(m._AsOfFetcher(d))
                scorer._market_score, scorer._market_detail = MSCORE[d], "pit"
                sc = scorer.score_stock(sd)
                ds = d.strftime("%Y-%m-%d")
                nxt = next((e for e in edates if e > ds), None)  # next report date (live sees the scheduled date)
                rows.append((cik, ds, row.symbol, round(sc.total_score, 2),
                             sc.c_score, sc.a_score, sc.n_score, sc.s_score, sc.l_score, sc.i_score, sc.m_score,
                             round(eps_growth(sd.quarterly_earnings), 2), round(annual_cagr(sd.annual_earnings), 2),
                             sd.sector, round(close * known.val.iloc[-1] / 1e6, 1),
                             round(sd.earnings_surprise_pct or 0, 2), int(sd.eps_beat_streak or 0),
                             round(sd.institutional_holders_pct or 0, 2),
                             (pd.Timestamp(nxt) - d).days if nxt else None))
        except Exception as e:
            rows.append((cik, None, repr(e)[:150]) + (np.nan,) * 16)
    pd.DataFrame(rows, columns=["cik", "date", "symbol", "total", "c", "a", "n", "s", "l", "i", "m",
                                "eps_growth", "annual_cagr", "sector", "mcap_m", "surprise_pct", "beat_streak",
                                "inst_pct", "days_to_earnings"]).to_csv(tmp := path.with_suffix(".tmp.gz"), index=False)
    tmp.replace(path)  # atomic: a restart never leaves a half-written chunk that looks finished
    return idx, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    setup()
    panel = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik"])
    ciks = sorted(panel.cik.unique())  # companies that were ever eligible on a panel date
    if a.limit:
        ciks = ciks[:: max(1, len(ciks) // a.limit)][: a.limit]
    chunks = list(enumerate(ciks[i:i + 25] for i in range(0, len(ciks), 25)))
    print(f"{len(ciks):,} CIKs, {len(SESS)} sessions, {len(chunks)} chunks; setup {time.time() - t0:.0f}s", flush=True)
    done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, (idx, n) in enumerate(pool.imap_unordered(work, chunks), 1):
            done += max(n, 0)
            if i % 10 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {done:,} rows, {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
