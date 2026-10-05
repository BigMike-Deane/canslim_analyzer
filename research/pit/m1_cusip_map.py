# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M1: CUSIP <-> ticker-at-the-time, from SEC fails-to-deliver files.

13F holdings identify securities only by CUSIP. FTD files list CUSIP, symbol
and settlement date twice a month, so each (cusip, symbol) pair gets the
date range it was observed. A CUSIP can change tickers and a ticker can be
reused by a new CUSIP; the date ranges keep both straight.

Output: META_DIR/cusip_symbol.csv.gz (cusip, symbol, first_seen, last_seen, n_obs, description)
"""
import io
import zipfile

import pandas as pd

from common import META_DIR, SEC_DIR


def main():
    parts = []
    for path in sorted((SEC_DIR / "ftd").glob("cnsfails*.zip")):
        with zipfile.ZipFile(path) as z:
            raw = z.read(z.namelist()[0]).decode("latin1")
        df = pd.read_csv(io.StringIO(raw), sep="|", dtype=str, on_bad_lines="skip",
                         usecols=[0, 1, 2, 4], names=["date", "cusip", "symbol", "desc"], header=0)
        parts.append(df.dropna(subset=["cusip", "symbol"]))
    d = pd.concat(parts, ignore_index=True)
    d = d[d.date.str.fullmatch(r"\d{8}", na=False)]
    d["cusip"] = d.cusip.str.strip().str.upper()
    d["symbol"] = d.symbol.str.strip().str.upper()
    d = d[d.cusip.str.len() == 9]
    m = (d.groupby(["cusip", "symbol"])
           .agg(first_seen=("date", "min"), last_seen=("date", "max"),
                n_obs=("date", "size"), description=("desc", "last"))
           .reset_index())
    m.to_csv(META_DIR / "cusip_symbol.csv.gz", index=False)
    print(f"{len(d):,} FTD rows -> {len(m):,} (cusip, symbol) pairs; "
          f"{m.cusip.nunique():,} CUSIPs, {m.symbol.nunique():,} symbols")
    print(f"CUSIPs seen under >1 symbol: {(m.groupby('cusip').size() > 1).sum():,}; "
          f"symbols on >1 CUSIP: {(m.groupby('symbol').size() > 1).sum():,}")


if __name__ == "__main__":
    main()
