# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M0 step 2b: tickers for CIKs FMP can't resolve (mostly delisted/acquired:
Walgreen Co, Aetna, Safeway, Google Inc...). FMP's search-cik only knows
current entities.

Match the CIK's SEC names (current + formerNames from the submissions API) to
fails-to-deliver descriptions (truncated ~30 chars, share-class noise), within
the CIK's filing window. Accept only a single clear winner per CIK.

Output: rows appended to META_DIR/cik_tickers.csv.gz with source='ftd_name'
(FMP-resolved rows keep source='fmp'); review sample in m0_name_match.csv.
"""
import re

import pandas as pd

from common import META_DIR, SEC_DIR, cached, sec_get

STOP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC",
        "LP", "LLC", "THE", "NEW", "COM", "COMMON", "STOCK", "STK", "SHS", "SH", "CL", "CLASS",
        "ORD", "NPV", "PAR", "DEL", "HLDG", "HLDGS", "HOLDING", "HOLDINGS", "GROUP", "ST",
        "TR", "TRUST", "SA", "NV", "AG", "N", "V", "A", "B", "C", "VOTING", "NON", "ORDINARY",
        "SHARES", "SHARE", "REG", "REGISTERED", "ADR", "ADS", "SPONSORED", "SPON", "UNSPON", "VTG"}
MIN_OVERLAP_DAYS = 120


def norm(s: str) -> list[str]:
    s = (s or "").upper().replace("&", " AND ").replace("'", "")
    s = re.sub(r"\([^)]*\)", " ", s)             # FTD state tags: FLIR SYSTEMS, INC. (DE)
    s = re.sub(r"/[A-Z]{2,3}/", " ", s)               # SEC state tags: AETNA INC /PA/
    s = re.sub(r"USD[\d.]+|\$[\d.]+|\d+(\.\d+)?%?", " ", s)
    s = re.sub(r"[^A-Z ]", " ", s)
    toks = [t for t in s.split() if t not in STOP]
    return [t for i, t in enumerate(toks) if t not in toks[:i]]  # WALT DISNEY CO-DISNEY


def prefix_match(sec_toks, ftd_toks):
    """FTD text is truncated; require its tokens to be a prefix of the SEC
    name (last FTD token may itself be truncated) and >= 6 letters overall."""
    if not ftd_toks or not sec_toks:
        return False
    if ftd_toks == sec_toks and len("".join(ftd_toks)) >= 2:
        return True  # exact name (AETNA, CIGNA, DELL, CA) is safe even when short
    if len(sec_toks) >= 2 and ftd_toks[:len(sec_toks)] == sec_toks:
        return True  # FTD adds words after the full SEC name (... COMMON, ... NON-VTG)
    if len("".join(ftd_toks)) < 6:
        return False
    head, last = ftd_toks[:-1], ftd_toks[-1]
    if sec_toks[:len(head)] != head or len(sec_toks) <= len(head):
        return False
    return sec_toks[len(head)].startswith(last)


def main():
    tick = pd.read_csv(META_DIR / "cik_tickers.csv.gz")
    # Idempotent re-runs: earlier name matches go back to unresolved.
    if "source" in tick:
        tick.loc[tick.source == "ftd_name", ["symbol", "exchange"]] = None
    tick = tick.drop_duplicates("cik").assign(source="fmp")
    unresolved = tick[tick.symbol.isna()].cik.tolist()
    uq = pd.read_csv(META_DIR / "universe_quarters.csv.gz")
    win = uq.assign(y=uq.quarter.str[2:6].astype(int), q=uq.quarter.str[-1].astype(int))
    win["d"] = pd.to_datetime(win.y.astype(str) + "-" + (win.q * 3).astype(str) + "-28")
    win = win.groupby("cik").d.agg(lo="min", hi="max")

    names = {}
    for i, cik in enumerate(unresolved, 1):
        sub = cached(SEC_DIR / "submissions" / f"{cik}.json",
                     lambda: sec_get(f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json")) or {}
        names[cik] = [sub.get("name", "")] + [x.get("name", "") for x in sub.get("formerNames", [])]
        if i % 500 == 0:
            print(f"  submissions {i:,}/{len(unresolved):,}", flush=True)

    ftd = pd.read_csv(META_DIR / "cusip_symbol.csv.gz", dtype=str)
    ftd = ftd[ftd.symbol.str.fullmatch(r"[A-Z]{1,5}", na=False)]
    ftd["lo"], ftd["hi"] = pd.to_datetime(ftd.first_seen), pd.to_datetime(ftd.last_seen)
    ftd["toks"] = ftd.description.map(norm)
    ftd["key"] = ftd.toks.map(lambda t: t[0] if t else "")
    by_key = {k: g for k, g in ftd.groupby("key") if k}

    rows, review = [], []
    for cik in unresolved:
        if cik not in win.index:
            continue
        lo, hi = win.loc[cik, "lo"] - pd.Timedelta(days=120), win.loc[cik, "hi"] + pd.Timedelta(days=120)
        cands = []
        for nm in names.get(cik, []):
            st = norm(nm)
            if not st or st[0] not in by_key:
                continue
            g = by_key[st[0]]
            for r in g[g.toks.map(lambda ft: prefix_match(st, ft))].itertuples():
                ov = (min(r.hi, hi) - max(r.lo, lo)).days
                if ov >= MIN_OVERLAP_DAYS:
                    cands.append((ov, -len(r.symbol), r.symbol, r.description, nm))
        if not cands:
            continue
        exact = [c for c in cands if norm(c[3]) == norm(c[4])]
        if exact:
            cands = exact
        cands.sort(reverse=True)
        syms = {c[2] for c in cands}
        best = cands[0]
        # a second distinct symbol with comparable overlap = ambiguous (share classes ok if one dominates)
        rival = [c for c in cands if c[2] != best[2] and c[0] >= 0.8 * best[0]]
        if rival and not all(c[3] == best[3] for c in rival):
            review.append((cik, names[cik][0], " | ".join(sorted(syms)), "ambiguous"))
            continue
        rows.append((cik, best[2], "FTD", 0, " ".join(sorted(syms)), "ftd_name"))
        review.append((cik, names[cik][0], best[2], best[3]))

    extra = pd.DataFrame(rows, columns=["cik", "symbol", "exchange", "n_candidates", "all_symbols", "source"])
    out = pd.concat([tick[tick.symbol.notna()], extra], ignore_index=True)
    out = pd.concat([out, tick[tick.symbol.isna() & ~tick.cik.isin(extra.cik)]], ignore_index=True)
    out.to_csv(META_DIR / "cik_tickers.csv.gz", index=False)
    pd.DataFrame(review, columns=["cik", "sec_name", "symbol", "ftd_description"]).to_csv(
        META_DIR / "m0_name_match.csv", index=False)
    amb = sum(r[3] == "ambiguous" for r in review)
    print(f"unresolved {len(unresolved):,} -> matched {len(extra):,}, ambiguous {amb:,}, "
          f"still none {len(unresolved) - len(extra) - amb:,}")


if __name__ == "__main__":
    main()
