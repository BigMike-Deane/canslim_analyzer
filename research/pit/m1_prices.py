# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M1: split-adjusted daily OHLCV from FMP for every mapped symbol + index ETFs.

`historical-price-eod/full` is split-adjusted (checked: NVDA 10:1 on
2024-06-10 is continuous) and reaches 2009 in one call. Prices are stored
whole; clipping to each CIK's filing window (ticker reuse) happens at use time.
Resumable: one gzip JSON per symbol. Rate: PIT_FMP_PER_MIN.

Output: FMP_DIR/prices/<SYMBOL>.json.gz, META_DIR/price_coverage.csv.gz
"""
import gzip
import json

import pandas as pd

from common import FMP_DIR, META_DIR, fmp_get

INDEXES = ["SPY", "QQQ", "DIA", "IWM"]
OUT = FMP_DIR / "prices"
OUT.mkdir(parents=True, exist_ok=True)
KEEP = ("date", "open", "high", "low", "close", "volume")


def fetch(symbol):
    path = OUT / f"{symbol}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    data = fmp_get("historical-price-eod/full", symbol=symbol, **{"from": "2009-01-01"}) or []
    rows = [[r.get(k) for k in KEEP] for r in data if isinstance(r, dict)]
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(rows, fh)
    tmp.replace(path)
    return rows


def main():
    tickers = pd.read_csv(META_DIR / "cik_tickers.csv.gz").dropna(subset=["symbol"])
    symbols = INDEXES + sorted(set(tickers.symbol) - set(INDEXES))
    print(f"prices for {len(symbols):,} symbols", flush=True)
    cov = []
    for i, sym in enumerate(symbols, 1):
        rows = fetch(sym)
        cov.append((sym, len(rows), rows[-1][0] if rows else None, rows[0][0] if rows else None))
        if i % 500 == 0:
            print(f"  {i:,}/{len(symbols):,}", flush=True)
    df = pd.DataFrame(cov, columns=["symbol", "n_days", "first_date", "last_date"])
    df.to_csv(META_DIR / "price_coverage.csv.gz", index=False)
    print(f"with prices: {(df.n_days > 0).sum():,}/{len(df):,}")


if __name__ == "__main__":
    main()
