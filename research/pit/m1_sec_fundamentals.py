# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M1: point-in-time fundamentals from SEC XBRL companyfacts.

Every fact carries the `filed` date of the filing that reported it, and a
restated period appears once per filing. So "what was known on date D" is
exact: use facts with filed <= D, and for each period take the earliest
filing (as first reported) or the latest filing <= D (as known then).

Only the concepts the scorer needs are kept; full companyfacts files are
several MB each. Resumable: one compact gzip JSON per CIK.

Output: SEC_DIR/facts/<cik>.json.gz and META_DIR/sec_facts.csv.gz
"""
import gzip
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import META_DIR, SEC_DIR, sec_get

KEEP = {
    ("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasic"),
    ("us-gaap", "NetIncomeLoss"), ("us-gaap", "StockholdersEquity"),
    ("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "SalesRevenueNet"),
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
}
FIELDS = ("start", "end", "val", "filed", "form", "fy", "fp", "frame")
OUT = SEC_DIR / "facts"
OUT.mkdir(parents=True, exist_ok=True)


def fetch_one(cik, keep=KEEP, out=OUT):
    path = out / f"{cik}.json.gz"
    if path.exists():
        return "cached"
    data = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(cik).zfill(10)}.json")
    rows = []
    for (tax, concept) in keep:
        units = ((data or {}).get("facts", {}).get(tax, {}).get(concept) or {}).get("units", {})
        for unit, facts in units.items():
            for f in facts:
                rows.append([concept, unit] + [f.get(k) for k in FIELDS])
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(rows, fh)
    tmp.replace(path)
    return "fetched" if data else "missing"


def main():
    ciks = pd.read_csv(META_DIR / "universe_ciks.csv.gz")
    ciks = ciks[ciks.n_quarters >= 4].cik.tolist()
    print(f"companyfacts for {len(ciks):,} CIKs", flush=True)
    counts = {}
    with ThreadPoolExecutor(max_workers=4) as pool:  # throttle in common caps at 8/s
        for i, status in enumerate(pool.map(fetch_one, ciks), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 1000 == 0:
                print(f"  {i:,}/{len(ciks):,} {counts}", flush=True)
    print("done", counts, flush=True)

    frames = []
    for cik in ciks:
        with gzip.open(OUT / f"{cik}.json.gz", "rt") as fh:
            rows = json.load(fh)
        if rows:
            df = pd.DataFrame(rows, columns=["concept", "unit", *FIELDS])
            df.insert(0, "cik", cik)
            frames.append(df)
    facts = pd.concat(frames, ignore_index=True)
    facts.to_csv(META_DIR / "sec_facts.csv.gz", index=False)
    print(f"facts: {len(facts):,} rows, {facts.cik.nunique():,} CIKs")
    print(facts.groupby("concept").cik.nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
