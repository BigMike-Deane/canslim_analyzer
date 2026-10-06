# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M0 step 4: every ticker a company traded under, with dates.

Companies change tickers (Frontier FTR -> FYBR, Westar WR -> Evergy EVRG,
post-bankruptcy relistings). FMP and the name match give one ticker each,
usually the latest, so earlier years look unpriced. The company's CUSIPs
(identity.csv.gz) appear in fails-to-deliver under each ticker they used, with
first/last-seen dates. Those become dated segments; common.load_cik_prices
stitches the price series across them.

Output: META_DIR/symbol_segments.csv.gz (cik, symbol, seg_from, seg_to, n_obs)
"""
import pandas as pd

from common import META_DIR

PAD = pd.Timedelta(days=20)  # FTD sightings are sparse; pad each segment's edges


def main():
    ident = pd.read_csv(META_DIR / "identity.csv.gz", parse_dates=["valid_from", "valid_to"])
    ftd = pd.read_csv(META_DIR / "cusip_symbol.csv.gz", dtype=str)
    ftd = ftd[ftd.symbol.str.fullmatch(r"[A-Z]{1,5}", na=False)]
    ftd["first_seen"], ftd["last_seen"] = pd.to_datetime(ftd.first_seen), pd.to_datetime(ftd.last_seen)
    ftd["n_obs"] = ftd.n_obs.astype(int)
    by_cusip = {c: g for c, g in ftd.groupby("cusip")}

    rows = []
    for r in ident.itertuples():
        # the identity symbol over the whole valid window, lowest priority
        rows.append((r.cik, r.symbol, r.valid_from, r.valid_to, 0))
        if not isinstance(r.cusips, str):
            continue
        for c in r.cusips.split():
            for s in by_cusip.get(c, pd.DataFrame()).itertuples():
                lo = max(s.first_seen - PAD, r.valid_from)
                hi = min(s.last_seen + PAD, r.valid_to)
                if hi > lo:
                    rows.append((r.cik, s.symbol, lo, hi, s.n_obs))
    seg = pd.DataFrame(rows, columns=["cik", "symbol", "seg_from", "seg_to", "n_obs"])
    seg = seg.groupby(["cik", "symbol"], as_index=False).agg(
        seg_from=("seg_from", "min"), seg_to=("seg_to", "max"), n_obs=("n_obs", "max"))
    seg.to_csv(META_DIR / "symbol_segments.csv.gz", index=False)
    multi = seg.groupby("cik").symbol.nunique()
    print(f"{len(seg):,} segments for {seg.cik.nunique():,} CIKs; "
          f"CIKs with >1 historical ticker: {(multi > 1).sum():,}; "
          f"distinct symbols: {seg.symbol.nunique():,}")


if __name__ == "__main__":
    main()
