# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Analyst actions per symbol (FMP /stable/grades: dated upgrades/downgrades,
2012+; /stable/grades-historical: monthly rating-count snapshots, 2019+).
Dated events, so point-in-time by construction. Resumable; scanner-aware FMP
pacing. Candidate input for an analyst-momentum hypothesis (not yet
pre-registered; coverage is checked first).

Output: FMP_DIR/grades/<SYMBOL>.json, FMP_DIR/grades_hist/<SYMBOL>.json
"""
import pandas as pd

from common import FMP_DIR, META_DIR, cached, fmp_get


def main():
    syms = sorted(set(pd.read_csv(META_DIR / "symbol_segments.csv.gz").symbol))
    print(f"grades for {len(syms):,} symbols", flush=True)
    n = h = 0
    for i, s in enumerate(syms, 1):
        n += bool(cached(FMP_DIR / "grades" / f"{s}.json", lambda: fmp_get("grades", symbol=s, limit=5000)))
        h += bool(cached(FMP_DIR / "grades_hist" / f"{s}.json",
                         lambda: fmp_get("grades-historical", symbol=s, limit=1000)))
        if i % 500 == 0:
            print(f"  {i:,}/{len(syms):,} (with grades: {n:,}, with monthly counts: {h:,})", flush=True)
    print(f"done: grades {n:,}, monthly counts {h:,}", flush=True)


if __name__ == "__main__":
    main()
