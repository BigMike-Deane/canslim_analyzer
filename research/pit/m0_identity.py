# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M0 step 3: who a ticker belonged to, and when; plus each company's CUSIPs.

Tickers get reused (BBBY: Bed Bath & Beyond, later Beyond Inc.), so a CIK's
FMP symbol can carry another company's prices before or after its own life.
Fails-to-deliver files record which CUSIP traded under a symbol on each date.
A CUSIP's first 6 characters identify the issuer. For each CIK we take the
issuer whose span under the symbol best overlaps the CIK's SEC filing window:
  - valid_from/valid_to: the dates the symbol's prices belong to this CIK
  - cusips: that issuer's 9-char CUSIPs under the symbol (13F join key)
CIKs whose symbol never appears in FTD fall back to the filing window.

Output: META_DIR/identity.csv.gz
"""
import pandas as pd

from common import META_DIR

PAD_BEFORE = pd.Timedelta(days=730)   # price history needed before the first filing
PAD_AFTER = pd.Timedelta(days=120)


def filing_windows():
    f = pd.read_csv(META_DIR / "sec_facts.csv.gz", usecols=["cik", "filed", "form"], low_memory=False)
    f = f[f.form.isin(["10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F"])]
    f["filed"] = pd.to_datetime(f.filed, errors="coerce")
    return f.groupby("cik").filed.agg(first_filed="min", last_filed="max")


def main():
    tick = pd.read_csv(META_DIR / "cik_tickers.csv.gz").dropna(subset=["symbol"])
    win = filing_windows()
    ftd = pd.read_csv(META_DIR / "cusip_symbol.csv.gz", dtype=str)
    ftd["first_seen"] = pd.to_datetime(ftd.first_seen)
    ftd["last_seen"] = pd.to_datetime(ftd.last_seen)
    ftd["issuer"] = ftd.cusip.str[:6]
    spans = (ftd.groupby(["symbol", "issuer"])
                .agg(start=("first_seen", "min"), end=("last_seen", "max"),
                     cusips=("cusip", lambda s: " ".join(sorted(set(s)))))
                .reset_index())
    by_symbol = {s: g for s, g in spans.groupby("symbol")}

    rows = []
    for t in tick.itertuples():
        if t.cik not in win.index:
            continue
        lo, hi = win.loc[t.cik, "first_filed"], win.loc[t.cik, "last_filed"]
        w_lo, w_hi = lo - PAD_BEFORE, hi + PAD_AFTER
        g = by_symbol.get(t.symbol)
        if g is None:
            rows.append((t.cik, t.symbol, w_lo, w_hi, "", "filing_window", 0, len(set())))
            continue
        # Issuers whose span overlaps the CIK's own filing window are this
        # company (reorgs and redomiciles change the CUSIP issuer: ITT 2016,
        # PNR 2014, FRT 2022). Spans wholly outside it are ticker reuse.
        overlap = (g.end.clip(upper=hi + PAD_AFTER) - g.start.clip(lower=lo)).dt.days
        mine, others = g[overlap > 0], g[overlap <= 0]
        if mine.empty:
            rows.append((t.cik, t.symbol, w_lo, w_hi, "", "no_overlap", 0, len(g)))
            continue
        start, end = mine.start.min(), mine.end.max()
        # Extend into the padding only where no other issuer held the symbol.
        if others[others.end < start].empty:
            start = min(start, w_lo)
        if others[others.start > end].empty:
            end = max(end, w_hi)
        cusips = " ".join(sorted(set(" ".join(mine.cusips).split())))
        rows.append((t.cik, t.symbol, start, end, cusips, "ftd", int(overlap.clip(lower=0).sum()), len(g)))
    out = pd.DataFrame(rows, columns=["cik", "symbol", "valid_from", "valid_to", "cusips",
                                      "source", "overlap_days", "n_issuers"])
    out.to_csv(META_DIR / "identity.csv.gz", index=False)
    print(out.source.value_counts().to_string())
    print(f"symbols with >1 issuer in FTD (reuse handled): {(out.n_issuers > 1).sum():,}")


if __name__ == "__main__":
    main()
