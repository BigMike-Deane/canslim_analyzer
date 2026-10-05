# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M0 gate: survivorship coverage. For each year, what share of SEC filers
have a usable price series (inside their identity window) that year?

"Liquid" can't be defined by market cap without prices (circular), so the
gate population is companies reporting >= $100M annual revenue that year
(SEC facts). Gate (docs/phase2-pit-backtest-plan.md): >= 90% covered in every
year. Also reported: all filers, and filers that later stopped filing
(delisted/acquired) -- the group survivorship bias would drop.
"""
import pandas as pd

from common import META_DIR, load_prices

REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]


def price_years(symbol, lo, hi):
    px = load_prices(symbol)
    if px.empty:
        return set()
    d = px.index[(px.index >= lo) & (px.index <= hi)]
    counts = pd.Series(1, index=d).groupby(d.year).size()
    return set(counts[counts >= 100].index)  # >= 100 sessions = "priced that year"


def main():
    uq = pd.read_csv(META_DIR / "universe_quarters.csv.gz")
    uq["year"] = uq.quarter.str[2:6].astype(int)
    filers = uq.drop_duplicates(["cik", "year"])[["cik", "year"]]

    f = pd.read_csv(META_DIR / "sec_facts.csv.gz", low_memory=False,
                    usecols=["cik", "concept", "start", "end", "val"])
    f = f[f.concept.isin(REV)]
    days = (pd.to_datetime(f.end, errors="coerce") - pd.to_datetime(f.start, errors="coerce")).dt.days
    fy = f[days.between(350, 380)].assign(year=pd.to_datetime(f.end, errors="coerce").dt.year)
    big = fy.groupby(["cik", "year"]).val.max().reset_index()
    big = big[big.val >= 100e6][["cik", "year"]]

    ident = pd.read_csv(META_DIR / "identity.csv.gz", parse_dates=["valid_from", "valid_to"])
    covered = set()
    for r in ident.itertuples():
        for y in price_years(r.symbol, r.valid_from, r.valid_to):
            covered.add((r.cik, y))

    last = uq.groupby("cik").quarter.max()
    gone = set(last[last < "CY2025Q3"].index)

    def rate(df):
        hit = df.apply(lambda r: (r.cik, r.year) in covered, axis=1)
        return df.assign(hit=hit).groupby("year").hit.mean()

    rep = pd.DataFrame({
        "all_filers": rate(filers),
        "rev_ge_100M": rate(big.merge(filers, on=["cik", "year"])),
        "later_delisted": rate(filers[filers.cik.isin(gone)]),
    }).loc[2010:2026]
    rep["n_rev_ge_100M"] = big.merge(filers, on=["cik", "year"]).groupby("year").size()
    print((rep[["all_filers", "rev_ge_100M", "later_delisted"]] * 100).round(1)
          .join(rep.n_rev_ge_100M).to_string())
    ok = (rep.rev_ge_100M.loc[2016:2025] >= 0.90).all()
    print(f"\nM0 GATE: {'PASS' if ok else 'FAIL'} (>= 90% of revenue>=$100M filers priced, every year 2016-2025; window amended 2026-10-05)")
    rep.to_csv(META_DIR / "m0_coverage.csv")


if __name__ == "__main__":
    main()
