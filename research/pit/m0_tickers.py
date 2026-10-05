"""M0 step 2: map each universe CIK to the ticker its common stock traded under.

FMP search-cik resolves delisted CIKs too (SVB -> SIVB) but also returns
preferreds, warrants, units and foreign listings; pick the US common stock.
Resumable: one cached JSON per CIK. Rate: PIT_FMP_PER_MIN (default 60).

Output: META_DIR/cik_tickers.csv.gz (cik, symbol, exchange, n_candidates, all_symbols)
"""
import re
import sys

import pandas as pd

from common import FMP_DIR, META_DIR, cached, fmp_get

US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSEARCA", "CBOE", "BATS"}
MIN_QUARTERS = 4


def pick_common(cands):
    """Choose the most likely common-stock symbol among FMP candidates."""
    us = [c for c in cands if c.get("exchange") in US_EXCHANGES and c.get("currency", "USD") == "USD"]
    pool = us or [c for c in cands if c.get("exchange") in {"OTC", "PNK"}]
    if not pool:
        return None
    def rank(c):
        sym, name = c["symbol"], c.get("companyName", "").lower()
        junk = bool(re.search(r"[-./^]", sym)) or any(
            w in name for w in ("preferred", "warrant", "unit", "notes", "debenture", "rights", "depositary"))
        return (junk, c.get("exchange") not in US_EXCHANGES, len(sym), sym)
    return min(pool, key=rank)


def main():
    ciks = pd.read_csv(META_DIR / "universe_ciks.csv.gz")
    ciks = ciks[ciks.n_quarters >= MIN_QUARTERS].sort_values("cik")
    print(f"mapping {len(ciks):,} CIKs (>= {MIN_QUARTERS} quarters)", flush=True)
    out = []
    for i, cik in enumerate(ciks.cik, 1):
        cands = cached(FMP_DIR / "search_cik" / f"{cik}.json",
                       lambda: fmp_get("search-cik", cik=str(cik).zfill(10))) or []
        best = pick_common(cands)
        out.append((cik, best and best["symbol"], best and best.get("exchange"), len(cands),
                    " ".join(c["symbol"] for c in cands)))
        if i % 500 == 0:
            print(f"  {i:,}/{len(ciks):,}", flush=True)
    df = pd.DataFrame(out, columns=["cik", "symbol", "exchange", "n_candidates", "all_symbols"])
    df.to_csv(META_DIR / "cik_tickers.csv.gz", index=False)
    hit = df.symbol.notna().mean()
    print(f"resolved {df.symbol.notna().sum():,}/{len(df):,} ({hit:.1%}); "
          f"no FMP match: {(df.n_candidates == 0).sum():,}")
    dup = df[df.symbol.notna()].symbol.value_counts()
    print(f"symbols claimed by >1 CIK (ticker reuse): {(dup > 1).sum():,}")
    sys.exit(0)


if __name__ == "__main__":
    main()
