# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M0 step 1: the survivor-free company universe, from SEC XBRL frames.

A company is in the universe for a calendar quarter if it filed a quarterly
diluted or basic EPS fact for it. That is exactly the set the C/A scorer can
score. Delisted and acquired companies stay in because SEC never drops filers.
Keyed by CIK (tickers get reused).

Output: META_DIR/universe_quarters.csv.gz (cik, name, quarter, concept)
        META_DIR/universe_ciks.csv.gz (cik, name, first_q, last_q, n_quarters)
"""
import pandas as pd

from common import META_DIR, SEC_DIR, cached, sec_get

CONCEPTS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
# Filers that report EPS only per share class (Visa) have no undimensioned EPS
# frame; total net income catches them (2026-10-06 fix). Non-traded filers it
# adds fall out later (no ticker / no price).
EXTRA = [("NetIncomeLoss", "USD")]
QUARTERS = [f"CY{y}Q{q}" for y in range(2009, 2027) for q in range(1, 5)]
QUARTERS = [x for x in QUARTERS if x <= "CY2026Q2"]


def main():
    rows = []
    for concept in CONCEPTS:
        for q in QUARTERS:
            url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD-per-shares/{q}.json"
            data = cached(SEC_DIR / "frames" / f"{concept}_{q}.json", lambda: sec_get(url))
            for d in (data or {}).get("data", []):
                rows.append((d["cik"], d.get("entityName", ""), q, concept))
        print(f"{concept}: done")
    for concept, unit in EXTRA:
        for q in QUARTERS:
            url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/{unit}/{q}.json"
            data = cached(SEC_DIR / "frames" / f"{concept}_{q}.json", lambda: sec_get(url))
            for d in (data or {}).get("data", []):
                rows.append((d["cik"], d.get("entityName", ""), q, concept))
        print(f"{concept}: done")
    df = pd.DataFrame(rows, columns=["cik", "name", "quarter", "concept"])
    df.to_csv(META_DIR / "universe_quarters.csv.gz", index=False)

    per_q = df.drop_duplicates(["cik", "quarter"])
    ciks = per_q.groupby("cik").agg(
        name=("name", "last"), first_q=("quarter", "min"),
        last_q=("quarter", "max"), n_quarters=("quarter", "nunique")).reset_index()
    ciks.to_csv(META_DIR / "universe_ciks.csv.gz", index=False)

    by_year = per_q.assign(year=per_q.quarter.str[2:6]).groupby("year").cik.nunique()
    print(f"distinct CIKs: {len(ciks):,}")
    print("companies per year:", by_year.to_dict())
    gone = ciks[ciks.last_q < "CY2025Q3"]
    print(f"stopped filing before 2025Q3 (delisted/acquired/etc): {len(gone):,} "
          f"({len(gone) / len(ciks):.0%})")


if __name__ == "__main__":
    main()
