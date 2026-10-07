# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportPossiblyUnboundVariable=false
"""Score v3 input: FINRA consolidated short interest (exchange-listed + OTC, twice
monthly; free API, history from mid-2018). Published ~8 business days after the
settlement date, so a row is usable from settlement + 12 calendar days.

Per month -> DATA_DIR/finra_si/<YYYY-MM>.csv.gz (resumable); then
META_DIR/short_interest.csv.gz (symbol, settle, known, si, adv, dtc).
"""
import time

import pandas as pd
import requests

from common import DATA_DIR, META_DIR

URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
OUT = DATA_DIR / "finra_si"
OUT.mkdir(parents=True, exist_ok=True)
COLS = ["symbolCode", "settlementDate", "currentShortPositionQuantity",
        "averageDailyVolumeQuantity", "daysToCoverQuantity", "marketClassCode"]


def fetch_month(start, end):
    rows, offset = [], 0
    while True:
        body = {"limit": 5000, "offset": offset, "fields": COLS,
                "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": start, "endDate": end}]}
        for attempt in range(5):
            r = requests.post(URL, json=body, headers={"Accept": "application/json"}, timeout=120)
            if r.ok:
                break
            time.sleep(5 * 2 ** attempt)
        r.raise_for_status()
        page = r.json() if r.text.strip() else []
        rows += page
        if len(page) < 5000:
            return pd.DataFrame(rows, columns=COLS)
        offset += 5000
        time.sleep(0.3)


def main():
    months = pd.period_range("2018-01", "2026-10", freq="M")
    for m in months:
        f = OUT / f"{m}.csv.gz"
        if f.exists():
            continue
        d = fetch_month(str(m.start_time.date()), str(m.end_time.date()))
        tmp = f.with_suffix(".part")
        d.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(f)
        print(f"{m}: {len(d):,} rows", flush=True)
    d = pd.concat((pd.read_csv(f) for f in sorted(OUT.glob("*.csv.gz"))), ignore_index=True)
    d = d.rename(columns={"symbolCode": "symbol", "settlementDate": "settle", "currentShortPositionQuantity": "si",
                          "averageDailyVolumeQuantity": "adv", "daysToCoverQuantity": "dtc"})
    d["settle"] = pd.to_datetime(d.settle)
    d["known"] = d.settle + pd.Timedelta(days=12)
    d = d.dropna(subset=["symbol", "si"]).drop_duplicates(["symbol", "settle"])
    d[["symbol", "settle", "known", "si", "adv", "dtc", "marketClassCode"]].to_csv(
        META_DIR / "short_interest.csv.gz", index=False)
    print(f"done: {len(d):,} rows, {d.symbol.nunique():,} symbols, {d.settle.min().date()} -> {d.settle.max().date()}",
          flush=True)


if __name__ == "__main__":
    main()
