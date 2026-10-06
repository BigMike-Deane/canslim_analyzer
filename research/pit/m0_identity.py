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

from common import META_DIR, SEC_DIR, cached, sec_get
from m0_name_match import norm

PAD_BEFORE = pd.Timedelta(days=730)   # price history needed before the first filing
PAD_AFTER = pd.Timedelta(days=120)


def filing_windows():
    f = pd.read_csv(META_DIR / "sec_facts.csv.gz", usecols=["cik", "filed", "form"], low_memory=False)
    f = f[f.form.isin(["10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F"])]
    f["filed"] = pd.to_datetime(f.filed, errors="coerce")
    return f.groupby("cik").filed.agg(first_filed="min", last_filed="max")


def sec_names(cik):
    sub = cached(SEC_DIR / "submissions" / f"{cik}.json",
                 lambda: sec_get(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")) or {}
    return [sub.get("name", "")] + [x.get("name", "") for x in sub.get("formerNames", [])]


def by_name(cik, mine):
    """Several issuers held this symbol inside the CIK's filing window. A reused
    ticker (FI: Frank's International 2013-21, Fiserv 2023-) would otherwise
    hand both companies both CUSIPs. Keep the issuers whose FTD description
    shares a name token with the company's SEC names; none share -> keep all."""
    toks = set().union(*(norm(n) for n in sec_names(cik)))
    keep = mine[mine.desc.map(lambda d: bool(set(norm(d)) & toks))]
    return keep if len(keep) else mine


ABBR = {"HLDG": "HOLDING", "HLDGS": "HOLDINGS", "HOLDINGS": "HOLDING", "INTL": "INTERNATIONAL",
        "TECH": "TECHNOLOGY", "TECHNOLOGIES": "TECHNOLOGY", "GRP": "GROUP", "FINL": "FINANCIAL",
        "BANCORPORATION": "BANCORP", "SVCS": "SERVICES", "MGMT": "MANAGEMENT", "PHARMS": "PHARMACEUTICALS"}


def _toks(s):
    return {ABBR.get(t, t) for t in norm(s)}


def fmp_cik(symbol):
    import json
    from common import FMP_DIR
    p = FMP_DIR / "profile" / f"{symbol}.json"
    try:
        rows = json.loads(p.read_text()) if p.exists() else []
        return int(rows[0]["cik"]) if rows and rows[0].get("cik") else None
    except (ValueError, KeyError, IndexError):
        return None


def resolve_shared(out, ftd):
    """After by_name, some CUSIPs are still claimed by several CIKs: a listed
    parent plus a sibling filer with a similar name (subsidiary, LP, private
    fund). Each shared CUSIP goes to the claimant whose SEC name is best
    explained by the FTD description (share of name tokens found, >= 0.6, strict
    winner); it is removed from everyone else. A CIK left with no CUSIP of its
    own is flagged `ambiguous` and kept out of the research panel."""
    desc = ftd.drop_duplicates("cusip").set_index("cusip").description
    x = out[out.cusips.fillna("") != ""].assign(c=lambda d: d.cusips.str.split()).explode("c")
    claims = x.groupby("c").cik.apply(list)
    drop = {}  # cik -> cusips to remove
    for cu, ciks in claims[claims.map(len) > 1].items():
        d = _toks(desc.get(cu, ""))
        score = {}
        for cik in ciks:
            best = 0.0
            for nm in sec_names(cik):
                t = _toks(nm)
                if t:
                    best = max(best, len(t & d) / len(t))
            score[cik] = best
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        winner = ranked[0][0] if ranked[0][1] >= 0.6 and ranked[0][1] > ranked[1][1] else None
        if winner is None and ranked[0][1] >= 0.6:
            # tie between similarly named filers (Disney 2019 holdco, Eaton plc):
            # FMP's profile names the CIK that owns the listing
            tied = [c for c, v in ranked if v == ranked[0][1]]
            owner = {fmp_cik(sym) for sym in out[out.cik.isin(tied)].symbol}
            hit = [c for c in tied if c in owner]
            winner = hit[0] if len(hit) == 1 else None
        for cik in ciks:
            if cik != winner:
                drop.setdefault(cik, set()).add(cu)
    out["ambiguous"] = False
    for i, r in out.iterrows():
        if r.cik in drop:
            left = [c for c in str(r.cusips).split() if c not in drop[r.cik]]
            out.at[i, "cusips"] = " ".join(left)
            out.at[i, "ambiguous"] = not left
    return out


def main():
    tick = pd.read_csv(META_DIR / "cik_tickers.csv.gz").dropna(subset=["symbol"])
    win = filing_windows()
    ftd = pd.read_csv(META_DIR / "cusip_symbol.csv.gz", dtype=str)
    ftd["first_seen"] = pd.to_datetime(ftd.first_seen)
    ftd["last_seen"] = pd.to_datetime(ftd.last_seen)
    ftd["issuer"] = ftd.cusip.str[:6]
    spans = (ftd.groupby(["symbol", "issuer"])
                .agg(start=("first_seen", "min"), end=("last_seen", "max"),
                     cusips=("cusip", lambda s: " ".join(sorted(set(s)))),
                     desc=("description", "first"))
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
        if len(mine) > 1:
            mine = by_name(t.cik, mine)
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
    out = resolve_shared(out, ftd)
    out.to_csv(META_DIR / "identity.csv.gz", index=False)
    print(f"ambiguous (no CUSIP of their own after resolving shared CUSIPs): {int(out.ambiguous.sum()):,}")
    print(out.source.value_counts().to_string())
    print(f"symbols with >1 issuer in FTD (reuse handled): {(out.n_issuers > 1).sum():,}")


if __name__ == "__main__":
    main()
