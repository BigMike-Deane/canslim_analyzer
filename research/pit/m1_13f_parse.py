# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""M1: aggregate SEC 13F holdings into institutional shares per (quarter, CUSIP).

Point-in-time rule: for report period P, count only filings received within
KNOWN_LAG_DAYS of P's end; the aggregate is "known" from P + KNOWN_LAG_DAYS.
Late originals and later restatements are ignored (they weren't known then).

Per (filer, period), use the latest original/RESTATEMENT accession filed in
the window, plus any NEW HOLDINGS amendments filed in the window. Share rows
only (SSHPRNAMTTYPE == 'SH'), options excluded (PUTCALL empty).

Output: META_DIR/inst_13f.csv.gz (period, known_date, cusip, inst_shares, n_filers)
"""
import zipfile

import pandas as pd

from common import META_DIR, SEC_DIR

KNOWN_LAG_DAYS = 60
F13_DIR = SEC_DIR / "13f"


def read_tsv(z, name, usecols):
    # Some data sets nest the tables in a folder (e.g. 01jun2025-31aug2025).
    member = next(n for n in z.namelist() if n.rsplit("/", 1)[-1].upper() == name.upper())
    with z.open(member) as fh:
        return pd.read_csv(fh, sep="\t", usecols=usecols, dtype=str, quoting=3,
                           on_bad_lines="skip", encoding="latin1")


def choose_accessions(zips):
    """Pass 1 over the small SUBMISSION/COVERPAGE tables."""
    subs = []
    for path in zips:
        with zipfile.ZipFile(path) as z:
            s = read_tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
            c = read_tsv(z, "COVERPAGE.tsv", ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE"])
        subs.append(s.merge(c, on="ACCESSION_NUMBER", how="left").assign(zip=path.name))
    s = pd.concat(subs, ignore_index=True)
    s = s[s.SUBMISSIONTYPE.isin(["13F-HR", "13F-HR/A"])].copy()
    s["filed"] = pd.to_datetime(s.FILING_DATE, format="%d-%b-%Y", errors="coerce")
    s["period"] = pd.to_datetime(s.PERIODOFREPORT, format="%d-%b-%Y", errors="coerce")
    s = s[s.filed.notna() & s.period.notna()]
    s = s[s.filed <= s.period + pd.Timedelta(days=KNOWN_LAG_DAYS)]
    is_new = (s.ISAMENDMENT == "Y") & s.AMENDMENTTYPE.fillna("").str.upper().str.contains("NEW HOLDINGS")
    base = s[~is_new].sort_values("filed").groupby(["CIK", "period"]).tail(1)
    keep = pd.concat([base, s[is_new]])
    # A period whose filings mostly fall outside the downloaded data sets
    # (2013-03-31: only filings from the first window) would read as near-zero
    # ownership. Drop periods with < half the median filer count.
    filers = keep.groupby("period").CIK.nunique()
    thin = filers[filers < 0.5 * filers.median()].index
    if len(thin):
        print(f"dropping thin periods: {[str(p.date()) for p in thin]}", flush=True)
    keep = keep[~keep.period.isin(thin)]
    return keep[["ACCESSION_NUMBER", "period", "CIK", "zip"]].drop_duplicates("ACCESSION_NUMBER")


def main(zips=None):
    zips = sorted(zips or F13_DIR.glob("*_form13f.zip"))
    keep = choose_accessions(zips)
    print(f"accessions kept: {len(keep):,} across {keep.period.nunique()} periods", flush=True)
    acc_period = keep.set_index("ACCESSION_NUMBER").period
    parts = []
    for path in zips:
        accs = set(keep.loc[keep.zip == path.name, "ACCESSION_NUMBER"])
        if not accs:
            continue
        with zipfile.ZipFile(path) as z:
            it = read_tsv(z, "INFOTABLE.tsv", ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"])
        it = it[it.ACCESSION_NUMBER.isin(accs) & (it.SSHPRNAMTTYPE == "SH") & it.PUTCALL.isna()]
        it["shares"] = pd.to_numeric(it.SSHPRNAMT, errors="coerce")
        it["cusip"] = it.CUSIP.str.strip().str.upper().str[:9]
        it["period"] = it.ACCESSION_NUMBER.map(acc_period)
        agg = (it.groupby(["period", "cusip"])
                 .agg(inst_shares=("shares", "sum"), n_filers=("ACCESSION_NUMBER", "nunique"))
                 .reset_index())
        parts.append(agg)
        print(f"  {path.name}: {len(it):,} holdings -> {len(agg):,} (period, cusip)", flush=True)
    out = (pd.concat(parts).groupby(["period", "cusip"], as_index=False)
             .agg(inst_shares=("inst_shares", "sum"), n_filers=("n_filers", "sum")))
    out["known_date"] = out.period + pd.Timedelta(days=KNOWN_LAG_DAYS)
    out = out[out.cusip.str.len() == 9]
    out.to_csv(META_DIR / "inst_13f.csv.gz", index=False)
    print(f"wrote {len(out):,} rows; periods {out.period.min().date()} -> {out.period.max().date()}")
    print(out.groupby("period").cusip.nunique().tail(8).to_string())


if __name__ == "__main__":
    main()
