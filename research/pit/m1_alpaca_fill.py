# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalIterable=false
"""M1: fill FMP's delisted-stock holes with Alpaca SIP daily bars (2016+).

FMP drops many old delisted names (or its series for a reused ticker starts
with the new owner). Alpaca SIP bars keep delisted stocks from 2016 on (AET,
MON, CELG verified). Alpaca keys live only on the VPS, so the fetch runs
there via ssh (read-only market-data calls) and streams JSON lines back.

  python3 m1_alpaca_fill.py   # build the need-list, fetch, save
Output: DATA_DIR/alpaca/<SYMBOL>.json.gz  (same row layout as FMP prices)
Merging (FMP first, Alpaca for missing dates) happens in common.load_prices.
"""
import gzip
import json
import subprocess

import pandas as pd

from common import DATA_DIR, META_DIR, load_fmp_prices

START = pd.Timestamp("2016-01-04")
OUT = DATA_DIR / "alpaca"
OUT.mkdir(parents=True, exist_ok=True)

REMOTE = r'''
import os, sys, json, time, requests
H = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
     "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
for line in sys.stdin:
    sym, start, end = line.split()
    bars, token = [], None
    for _ in range(10):
        p = {"timeframe": "1Day", "start": start, "end": end, "limit": 10000,
             "adjustment": "split", "feed": "sip"}
        if token: p["page_token"] = token
        for attempt in range(4):
            r = requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars", headers=H, params=p, timeout=60)
            if r.status_code == 429: time.sleep(10 * (attempt + 1)); continue
            break
        j = r.json() if r.ok else {}
        bars += [[b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"]] for b in (j.get("bars") or [])]
        token = j.get("next_page_token")
        time.sleep(0.6)  # ~100 req/min, well under the 200/min data limit
        if not token: break
    print(json.dumps([sym, bars]), flush=True)
'''


def need_list():
    seg = pd.read_csv(META_DIR / "symbol_segments.csv.gz", parse_dates=["seg_from", "seg_to"])
    today = pd.Timestamp.today().normalize()
    need = {}
    for r in seg.itertuples():
        lo, hi = max(r.seg_from, START), min(r.seg_to, today)
        if hi - lo < pd.Timedelta(days=30):
            continue
        px = load_fmp_prices(r.symbol)
        px = px.loc[lo:hi] if len(px) else px
        if len(px) == 0 or px.index[0] > lo + pd.Timedelta(days=10) or px.index[-1] < hi - pd.Timedelta(days=10):
            a, b = need.get(r.symbol, (lo, hi))
            need[r.symbol] = (min(a, lo), max(b, hi))
    return need


def main():
    need = {s: w for s, w in need_list().items() if not (OUT / f"{s}.json.gz").exists()}
    print(f"symbols needing Alpaca fill: {len(need):,}", flush=True)
    if not need:
        return
    payload = "".join(f"{s} {a.date()} {b.date()}\n" for s, (a, b) in need.items())
    ssh = ["ssh", "-o", "ConnectTimeout=10", "root@100.104.189.36"]
    # Ship the script as a file (inline -c quoting breaks through ssh's shell).
    subprocess.run(ssh + ["docker exec -i canslim-analyzer sh -c 'cat > /tmp/pit_alpaca_fill.py'"],
                   input=REMOTE, text=True, check=True)
    cmd = ssh + ["docker exec -i canslim-analyzer python3 /tmp/pit_alpaca_fill.py"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    proc.stdin.write(payload)
    proc.stdin.close()
    got = hit = 0
    for line in proc.stdout:
        sym, bars = json.loads(line)
        with gzip.open(OUT / f"{sym}.json.gz", "wt") as fh:
            json.dump(bars, fh)
        got += 1
        hit += bool(bars)
        if got % 200 == 0:
            print(f"  {got:,}/{len(need):,} (with bars: {hit:,})", flush=True)
    proc.wait()
    print(f"done: {got:,} fetched, {hit:,} with bars", flush=True)


if __name__ == "__main__":
    main()
