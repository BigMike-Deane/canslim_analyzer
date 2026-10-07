# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Market-cap repair (v5 diagnostics, 2026-10-07): ~6% of companies had market caps off
by > 2x (SEC per-class / ADR-ratio / scale share counts: AVGO 0.08x, KLAC 0.11x, MA 0.13x,
ONC 11.6x; Alphabet absent 2016-mid-2024). FMP /stable/historical-market-capitalization
gives a daily series per symbol; it becomes the primary market cap, ours the fallback.

Output: FMP_DIR/mcap/<SYMBOL>.json (resumable), then META_DIR/fmp_mcap.csv.gz (symbol, date, mcap)
"""
import json

import pandas as pd

from common import FMP_DIR, META_DIR, cached, fmp_get

OUT = FMP_DIR / "mcap"


def main():
    seg = pd.read_csv(META_DIR / "symbol_segments.csv.gz")
    panel = pd.read_csv(META_DIR / "m3_panel.csv.gz", usecols=["cik", "mcap"])
    size = panel.groupby("cik").mcap.max()  # largest first: partial data covers the most weight
    seg = seg[seg.cik.isin(size.index)].assign(sz=lambda x: x.cik.map(size)).sort_values("sz", ascending=False)
    syms = list(dict.fromkeys(seg.symbol.dropna()))
    print(f"historical market cap for {len(syms):,} symbols", flush=True)
    for i, s in enumerate(syms, 1):
        cached(OUT / f"{s}.json", lambda: fmp_get("historical-market-capitalization", symbol=s, limit=5000,
                                                  **{"from": "2015-06-01", "to": "2026-10-06"}))
        if i % 500 == 0:
            print(f"  {i:,}/{len(syms):,}", flush=True)
    rows = []
    for f in OUT.glob("*.json"):
        d = json.load(open(f)) or []
        if isinstance(d, list) and d:
            x = pd.DataFrame(d)[["symbol", "date", "marketCap"]]
            rows.append(x[x.marketCap > 0])
    m = pd.concat(rows, ignore_index=True).rename(columns={"marketCap": "mcap"})
    m.to_csv(META_DIR / "fmp_mcap.csv.gz", index=False)
    print(f"done: {len(m):,} rows, {m.symbol.nunique():,} symbols", flush=True)


if __name__ == "__main__":
    main()
