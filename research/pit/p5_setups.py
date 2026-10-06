# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""H5/H6: live setup detection replayed point-in-time on the M3 panel
(pre-registered in docs/phase2-pit-backtest-plan.md, "Test it the way it
trades").

Per panel row, using only data through D:
  - weekly bars = last 26 W-FRI bars of the stitched daily series (live reads 6
    months of Yahoo weekly bars)
  - TechnicalAnalyzer.detect_base_pattern -> base type, weeks, pivot
  - TechnicalAnalyzer.is_breaking_out on the PIT StockData
  - trading_engine.calculate_entry_signals -> entry_type
  - forward 120-session return (ticker-change windows dropped, like M3)

  python3 p5_setups.py [--workers W] [--limit N]
Output: META_DIR/p5_setups.csv.gz
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from backend.trading_engine import calculate_entry_signals
from canslim_scorer import TechnicalAnalyzer as TA
from common import META_DIR

H = 120
PANEL = None


def setup():
    global PANEL
    PANEL = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik", "date", "total"], parse_dates=["date"])
    m._facts(); m._inst(); m.identity()
    from common import _segments
    _segments()


def weekly(px, d):
    w = px.loc[d - pd.Timedelta(days=200): d].resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    w = w.iloc[-26:]
    return [{"open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": r.Volume}
            for r in w.itertuples()]


def fwd(px, d):
    k = px.index.searchsorted(d, side="right") - 1
    if k < 0:
        return np.nan, False
    j = k + H
    if px.symbol.iloc[k:min(j, len(px) - 1) + 1].nunique() > 1:
        return np.nan, False
    if j < len(px):
        return px.Close.iloc[j] / px.Close.iloc[k] - 1, False
    return px.Close.iloc[-1] / px.Close.iloc[k] - 1, True


def work(ciks):
    rows = []
    for cik in ciks:
        try:
            px = m.cik_prices(cik)
            for r in PANEL[PANEL.cik == cik].itertuples():
                d = r.date
                sd = m.stock_data_asof(cik, d)
                if sd is None:
                    continue
                base = TA.detect_base_pattern(weekly(px, d))
                brk, brk_vol = TA.is_breaking_out(sd, base)
                es = calculate_entry_signals(
                    current_price=sd.current_price, week_52_high=sd.high_52w,
                    pivot_price=base.get("pivot_price", 0) or 0, base_type=base.get("type", "none"),
                    weeks_in_base=base.get("weeks", 0) or 0, is_breaking_out=brk,
                    breakout_volume_ratio=brk_vol, volume_ratio=TA.calculate_volume_ratio(sd),
                    effective_score=r.total)
                r120, cut = fwd(px, d)
                rows.append({"cik": cik, "date": d, "base_type": base.get("type", "none"),
                             "base_weeks": base.get("weeks", 0), "pct_from_pivot": es["pct_from_pivot"],
                             "entry_type": es["entry_type"], "breaking_out": brk, "r120": r120, "cut120": cut})
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
    name = "p5_setups_sample.csv.gz" if a.limit else "p5_setups.csv.gz"
    df.to_csv(META_DIR / name, index=False)
    print(f"wrote {len(df):,} rows; errors {len(errs)}; entry types "
          f"{df.entry_type.value_counts(normalize=True).round(3).to_dict()}; bases "
          f"{df.base_type.value_counts(normalize=True).round(3).to_dict()}; r120 coverage {df.r120.notna().mean():.0%}",
          flush=True)
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
