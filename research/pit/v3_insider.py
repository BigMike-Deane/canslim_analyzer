# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false
"""Score v3 input: SEC insider transactions data sets (Forms 3/4/5, quarterly, 2006->).

Downloads 2015Q3 -> latest quarter (resumable, SEC UA from .env, one consumer at a
time), keeps non-derivative open-market trades (codes P = purchase, S = sale) from
Form 4 filings, dated by FILING_DATE (when the market could know).

Output: META_DIR/insider_trades.csv.gz  (issuer_cik, filed, owner_cik, code, shares, price, value)
"""
import io
import zipfile

import pandas as pd

from common import META_DIR, SEC_DIR
from m1_13f_download import download

BASE = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
OUT_DIR = SEC_DIR / "form345"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def quarters():
    for y in range(2015, 2027):
        for q in range(1, 5):
            if (y, q) >= (2015, 3) and (y, q) <= (2026, 3):
                yield f"{y}q{q}"


def parse(path):
    z = zipfile.ZipFile(path)
    names = {n.split("/")[-1].upper(): n for n in z.namelist()}
    sub =pd.read_csv(io.BytesIO(z.read(names["SUBMISSION.TSV"])), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
    own = pd.read_csv(io.BytesIO(z.read(names["REPORTINGOWNER.TSV"])), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "RPTOWNERCIK"]).drop_duplicates("ACCESSION_NUMBER")
    tr = pd.read_csv(io.BytesIO(z.read(names["NONDERIV_TRANS.TSV"])), sep="\t", dtype=str,
                     usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    tr = tr[tr.TRANS_CODE.isin(["P", "S"])]
    d = tr.merge(sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])], on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    d["shares"] = pd.to_numeric(d.TRANS_SHARES, errors="coerce")
    d["price"] = pd.to_numeric(d.TRANS_PRICEPERSHARE, errors="coerce")
    d["filed"] = pd.to_datetime(d.FILING_DATE, format="mixed", errors="coerce").dt.date
    d = d.dropna(subset=["shares", "filed"])
    return pd.DataFrame({"issuer_cik": pd.to_numeric(d.ISSUERCIK, errors="coerce"), "filed": d.filed,
                         "owner_cik": pd.to_numeric(d.RPTOWNERCIK, errors="coerce"), "code": d.TRANS_CODE,
                         "shares": d.shares, "price": d.price, "value": d.shares * d.price.fillna(0)})


def main():
    parts = []
    for q in quarters():
        dest = OUT_DIR / f"{q}_form345.zip"
        st = download(f"{BASE}/{q}_form345.zip", dest)
        if st not in ("ok", "cached"):
            print(f"{q}: {st}", flush=True)
            continue
        p = parse(dest)
        print(f"{q}: {st}, {len(p):,} P/S trades", flush=True)
        parts.append(p)
    d = pd.concat(parts, ignore_index=True).dropna(subset=["issuer_cik"]).drop_duplicates()
    d["issuer_cik"] = d.issuer_cik.astype(int)
    d.to_csv(META_DIR / "insider_trades.csv.gz", index=False)
    print(f"done: {len(d):,} trades, {d.issuer_cik.nunique():,} issuers, "
          f"P {int((d.code == 'P').sum()):,} / S {int((d.code == 'S').sum()):,}", flush=True)


if __name__ == "__main__":
    main()
