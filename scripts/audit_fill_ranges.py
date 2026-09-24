# Read-only audit (Sep-24): every live + shadow fill vs its session raw SIP high/low.
# Run in the container the same way as audit_shadow_offhours_edge.py.
# Read-only: every live + shadow fill since Aug-18 vs that ET session's raw SIP high/low.
import sys, requests
sys.path.insert(0, "/app")
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from sqlalchemy import text
from backend.database import SessionLocal
from backend import alpaca_data as ad
ET = ZoneInfo("America/New_York")
db = SessionLocal()
rows = []
for src, sql in [
  ("live", "SELECT 'u'||user_id, id, ticker, action, price, executed_at, left(reason,60) FROM ai_portfolio_trades WHERE executed_at >= '2026-08-18'"),
  ("shadow", "SELECT 's'||shadow_strategy_id, id, ticker, action, price, executed_at, left(reason,60) FROM shadow_trades WHERE executed_at >= '2026-08-18'"),
]:
    for r in db.execute(text(sql)).fetchall():
        rows.append((src,) + tuple(r))
tickers = sorted({r[3] for r in rows})
hdr = ad._headers()
bars = {}
for i in range(0, len(tickers), 50):
    batch = [ad.alpaca_symbol(t) for t in tickers[i:i+50]]
    back = dict(zip(batch, tickers[i:i+50]))
    params = {"symbols": ",".join(batch), "timeframe": "1Day", "feed": "sip", "adjustment": "raw",
              "start": "2026-08-17", "end": (datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat(), "limit": 10000}
    tok = None
    while True:
        if tok: params["page_token"] = tok
        b = requests.get(f"{ad.DATA_BASE_URL}/v2/stocks/bars", headers=hdr, params=params, timeout=30).json()
        for sym, bl in (b.get("bars") or {}).items():
            for x in bl:
                d = ad.parse_ts(x["t"]).astimezone(ET).date()
                bars[(back.get(sym, sym), d)] = (x["l"], x["h"], x["o"], x["c"])
        tok = b.get("next_page_token")
        if not tok: break
out, nobar, n = [], [], 0
for src, who, tid, tk, act, px, ts, why in rows:
    et = ts.replace(tzinfo=timezone.utc).astimezone(ET)
    k = (tk, et.date())
    if act == "SPLIT": continue
    if k not in bars:
        nobar.append((src, who, tid, tk, act, px, et.strftime("%m-%d %H:%M")))
        continue
    n += 1
    lo, hi, o, c = bars[k]
    if px < lo * 0.998 or px > hi * 1.002:
        dev = (px/lo - 1) if px < lo else (px/hi - 1)
        out.append((src, who, tid, tk, act, px, lo, hi, f"{dev*100:+.2f}%", et.strftime("%m-%d %H:%M ET"), why))
print(f"checked {n} fills; {len(out)} OUTSIDE day range; {len(nobar)} with no bar")
for o in sorted(out, key=lambda x: x[9]): print(o)
print("--- no bar:")
for o in nobar: print(o)
