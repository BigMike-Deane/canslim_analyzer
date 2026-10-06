# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Phase 3: extra SEC companyfacts concepts for the signal lab (gross
profitability, accruals). Same as-first-reported / filed-date semantics as
m1_sec_fundamentals; separate cache so the M1-M3 inputs stay untouched.

Output: SEC_DIR/facts_q/<cik>.json.gz and META_DIR/sec_facts_q.csv.gz
"""
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd

from common import META_DIR, SEC_DIR
from m1_sec_fundamentals import FIELDS, fetch_one

KEEP_Q = {
    ("us-gaap", "GrossProfit"), ("us-gaap", "CostOfRevenue"),
    ("us-gaap", "CostOfGoodsAndServicesSold"), ("us-gaap", "CostOfGoodsSold"),
    ("us-gaap", "Assets"), ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "OperatingIncomeLoss"),
}
OUT_Q = SEC_DIR / "facts_q"
OUT_Q.mkdir(parents=True, exist_ok=True)


def main():
    ciks = pd.read_csv(META_DIR / "universe_ciks.csv.gz")
    ciks = ciks[ciks.n_quarters >= 4].cik.tolist()
    print(f"companyfacts (quality concepts) for {len(ciks):,} CIKs", flush=True)
    counts = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, status in enumerate(pool.map(partial(fetch_one, keep=KEEP_Q, out=OUT_Q), ciks), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 1000 == 0:
                print(f"  {i:,}/{len(ciks):,} {counts}", flush=True)
    print("done", counts, flush=True)
    frames = []
    for cik in ciks:
        with gzip.open(OUT_Q / f"{cik}.json.gz", "rt") as fh:
            rows = json.load(fh)
        if rows:
            df = pd.DataFrame(rows, columns=["concept", "unit", *FIELDS])
            df.insert(0, "cik", cik)
            frames.append(df)
    facts = pd.concat(frames, ignore_index=True)
    facts.to_csv(META_DIR / "sec_facts_q.csv.gz", index=False)
    print(f"facts: {len(facts):,} rows, {facts.cik.nunique():,} CIKs")
    print(facts.groupby("concept").cik.nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
