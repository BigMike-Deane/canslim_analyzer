# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M0 fix: pick the COMMON STOCK among FMP's search-cik candidates.

m0_tickers took FMP's first candidate. A CIK that issues several securities can
list a note/ETN/preferred first (JPMorgan 19617 -> AMJ, an MLP ETN, CUSIP
46625H365; Comcast -> CCZ, exchangeable notes). CUSIP convention: the issue
number (characters 7-8) of a company's common stock is normally "1x" (JPM
46625H100, CMCSA 20030N101); notes/ETNs/preferreds/units use other numbers.
Rule: if the picked symbol never traded under a "1x" CUSIP in fails-to-deliver
and another plain candidate did, switch to it (most "1x" FTD observations).
Candidates: 1-5 letters, not a 5-letter Nasdaq special-security code (4 letters
+ F/P/Q/R/U/V/W/Y/Z: foreign, preferred, bankrupt, rights, units, when-issued,
warrants, ADR, misc), and traded at least half as often as the original pick
(keeps MSTR over its recent thin preferreds STRC/STRK). Name matching was tried first and
rejected: FTD descriptions are truncated ("JPMORGAN CHASE & CO ALERIAN ML").
Idempotent (re-picks from `all_symbols`). Rows changed get source 'fmp_fixed'.

Output: META_DIR/cik_tickers.csv.gz (updated in place)
"""
import re

import pandas as pd

from common import META_DIR


def main():
    t = pd.read_csv(META_DIR / "cik_tickers.csv.gz")
    t.loc[t.source == "fmp_fixed", "source"] = "fmp"  # idempotent: re-decide every multi-candidate row
    ftd = pd.read_csv(META_DIR / "cusip_symbol.csv.gz", dtype=str)
    ftd["n_obs"] = ftd.n_obs.astype(int)
    common = ftd[ftd.cusip.str[6] == "1"].groupby("symbol").n_obs.sum()  # symbol -> FTD obs under a 1x CUSIP
    total = ftd.groupby("symbol").n_obs.sum()
    changed = 0
    for i, r in t[(t.n_candidates.fillna(1) > 1) & t.all_symbols.notna() & (t.source == "fmp")].iterrows():
        if r.symbol in common.index:
            continue
        alts = [s for s in str(r.all_symbols).split()
                if re.fullmatch(r"[A-Z]{1,5}", s) and not re.fullmatch(r"[A-Z]{4}[FPQRUVWYZ]", s)
                and s in common.index and common[s] >= 0.5 * total.get(r.symbol, 0)]
        if alts:
            t.at[i, "symbol"] = max(alts, key=lambda s: common[s])
            t.at[i, "source"] = "fmp_fixed"
            changed += 1
    t.to_csv(META_DIR / "cik_tickers.csv.gz", index=False)
    print(f"re-picked the common stock for {changed:,} CIKs")
    print(t[t.source == "fmp_fixed"][["cik", "symbol", "all_symbols"]].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
