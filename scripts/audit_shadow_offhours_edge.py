# Read-only audit (Sep-24). Run in the container:
#   scp scripts/audit_shadow_offhours_edge.py root@VPS:/tmp/ && ssh root@VPS "docker cp /tmp/audit_shadow_offhours_edge.py canslim-analyzer:/tmp/ && docker exec -w /app canslim-analyzer python /tmp/audit_shadow_offhours_edge.py"
# Oct-21 readout Amendment 2026-09-24 uses this to net each arm's off-hours edge.
# Read-only: off-hours shadow fills vs the next achievable price (next session open).
import sys, requests
sys.path.insert(0, "/app")
from datetime import datetime, timezone, timedelta, time
from zoneinfo import ZoneInfo
from collections import defaultdict
from sqlalchemy import text
from backend.database import SessionLocal
from backend import alpaca_data as ad
from backend.ai_trader import is_trading_day
ET = ZoneInfo("America/New_York")
db = SessionLocal()
rows = db.execute(text("""SELECT s.name, t.ticker, t.action, t.price, t.total_value, t.executed_at
  FROM shadow_trades t JOIN shadow_strategies s ON s.id=t.shadow_strategy_id
  WHERE t.executed_at >= '2026-08-18' AND t.action <> 'SPLIT'""")).fetchall()
def off(et):
    return (not is_trading_day(et)) or et.time() < time(9,30) or et.time() > time(16,0)
def next_session(et):
    d = et.date() if (is_trading_day(et) and et.time() < time(9,30)) else et.date() + timedelta(days=1)
    while not is_trading_day(datetime(d.year, d.month, d.day, 12, tzinfo=ET)):
        d += timedelta(days=1)
    return d
offs = []
for name, tk, act, px, tv, ts in rows:
    et = ts.replace(tzinfo=timezone.utc).astimezone(ET)
    if off(et): offs.append((name, tk, act, px, tv, et, next_session(et)))
tks = sorted({o[1] for o in offs})
hdr = ad._headers(); opens = {}
for i in range(0, len(tks), 50):
    batch = [ad.alpaca_symbol(t) for t in tks[i:i+50]]; back = dict(zip(batch, tks[i:i+50]))
    params = {"symbols": ",".join(batch), "timeframe": "1Day", "feed": "sip", "adjustment": "raw",
              "start": "2026-08-18", "end": (datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat(), "limit": 10000}
    tok = None
    while True:
        if tok: params["page_token"] = tok
        b = requests.get(f"{ad.DATA_BASE_URL}/v2/stocks/bars", headers=hdr, params=params, timeout=30).json()
        for sym, bl in (b.get("bars") or {}).items():
            for x in bl: opens[(back.get(sym, sym), ad.parse_ts(x["t"]).astimezone(ET).date())] = x["o"]
        tok = b.get("next_page_token")
        if not tok: break
agg = defaultdict(lambda: [0, 0.0, 0.0, 0]); allb = []; missing = 0
for name, tk, act, px, tv, et, nd in offs:
    o = opens.get((tk, nd))
    if o is None: missing += 1; continue
    # advantage to the shadow vs the next achievable price, in bps and $
    adv = (o/px - 1) if act in ("BUY", "PYRAMID") else (px/o - 1)
    a = agg[name]; a[0] += 1; a[1] += adv*1e4; a[2] += adv*tv; a[3] += 1 if adv > 0 else 0
    allb.append((adv*1e4, name, tk, act, et.strftime("%a %m-%d %H:%M"), px, o))
print(f"off-hours fills: {len(offs)}  priced: {len(allb)}  no next-open bar yet: {missing}")
print(f"{'arm':26} {'n':>3} {'mean bps':>9} {'$ adv':>8} {'%fav':>5}")
for k, (n, sb, sd, fav) in sorted(agg.items()):
    print(f"{k:26} {n:3d} {sb/n:9.1f} {sd:8.2f} {100*fav/n:5.0f}")
import statistics as st
v = [x[0] for x in allb]
print(f"ALL: n={len(v)} mean={st.mean(v):.1f} bps median={st.median(v):.1f} sd={st.pstdev(v):.1f}")
for side in ("BUY", "PYRAMID", "SELL"):
    s = [x[0] for x in allb if x[3] == side]
    if s: print(f"  {side:8} n={len(s):3d} mean={st.mean(s):7.1f} bps")
print("largest |adv|:")
for x in sorted(allb, key=lambda x: -abs(x[0]))[:8]: print("  ", f"{x[0]:+.0f}bps", *x[1:])
