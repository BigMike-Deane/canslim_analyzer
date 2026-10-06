# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M3: the point-in-time score panel (pre-registered in
docs/phase2-pit-backtest-plan.md, "M3 pre-registration").

Every 10th NYSE session 2016-01-04 -> last date with 60 forward sessions.
Universe per date: fresh price, close > $3, market cap >= $300M (close x SEC
shares outstanding as known that day). Each member is scored by the real
scorer via m2_adapter.score_asof; forward 10/20/60-session returns come from
the same stitched price series.

Work is split by CIK (prices/filings load once per company), in forked
workers that share the parent's preloaded tables.

  python3 m3_panel.py [--limit N] [--workers W]
Output: META_DIR/m3_panel.csv.gz
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

import m2_adapter as m
from common import split_factor

STEP, HORIZONS = 10, (10, 20, 60)
MIN_PRICE, MIN_CAP = 3.0, 300e6
COMP = ["c", "a", "n", "s", "l", "i", "m"]

DATES: list = []
MSCORE: dict = {}
SPY = IWM = None


def setup():
    global DATES, MSCORE, SPY, IWM
    SPY, IWM = m.prices("SPY").Close, m.prices("IWM").Close
    sess = SPY.loc["2016-01-04":].index
    last_ok = len(sess) - 1 - max(HORIZONS)
    DATES = list(sess[: last_ok + 1 : STEP])
    MSCORE = {d: m.market_score_asof(d) for d in DATES}
    # preload shared tables before fork (copy-on-write in workers)
    m._facts(), m._inst(), m.identity()
    from common import _segments
    _segments()


def fwd(series, d, h):
    i = series.index.searchsorted(d)
    if i >= len(series) or series.index[i] != d:
        return np.nan, False
    j = i + h
    if j < len(series):
        return series.iloc[j] / series.iloc[i] - 1, False
    return series.iloc[-1] / series.iloc[i] - 1, True  # ran out: last price, flagged


def work(ciks):
    rows = []
    ident = m.identity()
    for cik in ciks:
        try:
            row = ident.loc[cik]
            px = m.cik_prices(cik)
            if px.empty:
                continue
            sh = m._shares_table(cik)
            for d in DATES:
                if not (row.valid_from <= d <= row.valid_to):
                    continue
                k = px.index.searchsorted(d, side="right") - 1
                if k < 0 or px.index[k] < d - pd.Timedelta(days=5):
                    continue
                sym_d = px.symbol.iloc[k]
                close = px.Close.iloc[k] * split_factor(sym_d, d)  # actual price then
                s_known = sh[sh.filed <= d]
                if close <= MIN_PRICE or s_known.empty:
                    continue
                mcap = close * s_known.val.iloc[-1]  # actual price x shares as known then
                if mcap < MIN_CAP:
                    continue
                sc = m.score_asof(cik, d, m_score=MSCORE[d])
                if sc is None:
                    continue
                rec = {"cik": cik, "date": d, "symbol": row.symbol, "close": close, "mcap": mcap,
                       "total": sc.total_score, **{c: getattr(sc, f"{c}_score") for c in COMP}}
                # forward returns from D's own close (px row k may be <= D if D had no print)
                series, syms = px.Close.iloc[k:], px.symbol.iloc[k:]
                for h in HORIZONS:
                    r, cut = fwd(series, series.index[0], h)
                    # a window spanning a ticker change may splice two securities
                    # (post-bankruptcy relisting): exclude it
                    if syms.iloc[: h + 1].nunique() > 1:
                        r = np.nan
                    rec[f"r{h}"], rec[f"cut{h}"] = r, cut
                    rec[f"spy{h}"] = fwd(SPY, d, h)[0]
                    rec[f"iwm{h}"] = fwd(IWM, d, h)[0]
                rows.append(rec)
        except Exception as e:  # one bad company must not kill the panel
            rows.append({"cik": cik, "date": None, "error": repr(e)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    setup()
    ident = m.identity()
    if "ambiguous" in ident:  # sibling filers left without a CUSIP of their own (m0_identity)
        ident = ident[~ident.ambiguous.astype(bool)]
    ciks = sorted(ident.index.unique())
    if a.limit:
        ciks = ciks[:: max(1, len(ciks) // a.limit)][: a.limit]
    chunks = [ciks[i:i + 25] for i in range(0, len(ciks), 25)]
    print(f"{len(DATES)} dates {DATES[0].date()} -> {DATES[-1].date()}; {len(ciks):,} CIKs; "
          f"{a.workers} workers; setup {time.time() - t0:.0f}s", flush=True)
    out = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(work, chunks), 1):
            out += rows
            if i % 20 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(out)
    errs = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    df = df[df.date.notna()].drop(columns=["error"], errors="ignore")
    name = "m3_panel_sample.csv.gz" if a.limit else "m3_panel.csv.gz"
    df.to_csv(m.META_DIR / name, index=False)
    print(f"wrote {len(df):,} rows ({df.cik.nunique():,} CIKs, {df.date.nunique()} dates); "
          f"errors: {len(errs)}; {time.time() - t0:.0f}s", flush=True)
    if len(errs):
        print(errs.error.value_counts().head(5).to_string())


if __name__ == "__main__":
    main()
