# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M1: split history per symbol (FMP /stable/splits), to recover the ACTUAL
price on a past date from split-adjusted prices:
    actual_close(D) = adjusted_close(D) * prod(numerator/denominator for splits after D)
Needed for the M3 universe filter (price > $3, market cap = actual close x
SEC shares as of D). Scores are unaffected (the scorer uses price ratios).
Resumable; scanner-aware FMP pacing.

Output: FMP_DIR/splits/<SYMBOL>.json
"""
import pandas as pd

from common import FMP_DIR, META_DIR, cached, fmp_get


def main():
    syms = sorted(set(pd.read_csv(META_DIR / "symbol_segments.csv.gz").symbol))
    print(f"splits for {len(syms):,} symbols", flush=True)
    n = 0
    for i, s in enumerate(syms, 1):
        n += bool(cached(FMP_DIR / "splits" / f"{s}.json", lambda: fmp_get("splits", symbol=s)))
        if i % 1000 == 0:
            print(f"  {i:,}/{len(syms):,} (with splits: {n:,})", flush=True)
    print(f"done: {n:,} symbols have split history", flush=True)


if __name__ == "__main__":
    main()
