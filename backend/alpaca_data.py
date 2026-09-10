"""Alpaca market data (2026-09-10): daily bars for the ATR stop, and a
last-resort live price.

Why: the ATR stop -- which decides how far below cost every position's HARD
stop sits -- was computed from Yahoo's unofficial chart API, fetched on every
stop check. Yahoo throttles this box (YFRateLimitError in the scan logs), and
a failed fetch silently fell back to a cached or BASE stop. Alpaca's data API
is an official, keyed endpoint, and its bars come from the consolidated tape
(SIP): the same tape the broker's stop orders elect on.

What the free data plan actually allows (probed live 2026-09-10):
  * Historical SIP bars: yes, up to 15 minutes ago -> daily bars, including
    today's partial bar, with ``end = now - 16 min``. This is the ATR source.
  * Real-time SIP quotes: NO ("subscription does not permit querying recent
    SIP data").
  * Real-time IEX: yes, but IEX is one venue. CARE quoted $31.15 x $35.54 on
    it. So live prices stay on FMP (real-time consolidated); IEX's last
    trade is used only when every other source has failed, and only if it
    printed in the last 15 minutes.

Keys: the same paper keys as the broker mirror (market data is not a
trading endpoint -- nothing here can place an order). Without keys every
call returns None and callers keep their old path.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DATA_BASE_URL = "https://data.alpaca.markets"
# The free plan refuses SIP data newer than 15 minutes; ask for 16 to be safe.
SIP_DELAY = timedelta(minutes=16)
IEX_MAX_AGE = timedelta(minutes=15)
TIMEOUT = 5.0


def _headers() -> Optional[dict]:
    from backend.broker_mirror import credentials
    creds = credentials()
    if not creds:
        return None
    return {"APCA-API-KEY-ID": creds[0], "APCA-API-SECRET-KEY": creds[1],
            "Accept": "application/json"}


def parse_ts(value: str) -> Optional[datetime]:
    """Alpaca stamps are RFC-3339 with NANOsecond fractions
    ("2026-09-10T15:24:00.929936936Z"); datetime takes at most micro."""
    try:
        head, _, frac = value.rstrip("Z").partition(".")
        ts = datetime.fromisoformat(head).replace(tzinfo=timezone.utc)
        return ts + timedelta(microseconds=int((frac + "000000")[:6])) if frac else ts
    except (ValueError, AttributeError):
        return None


def alpaca_symbol(ticker: str) -> str:
    """The app spells class shares BRK-B; Alpaca spells them BRK.B."""
    return ticker.replace("-", ".")


def daily_bars(ticker: str, days: int = 45, now: datetime = None) -> Optional[list]:
    """Split-adjusted daily bars from the consolidated tape, oldest first,
    today's partial bar included (to 16 minutes ago). None on any failure --
    the caller falls back, so this must never raise."""
    headers = _headers()
    if headers is None:
        return None
    now = now or datetime.now(timezone.utc)
    symbol = alpaca_symbol(ticker)
    params = {
        "symbols": symbol, "timeframe": "1Day", "feed": "sip",
        # Split-adjusted, so a split inside the window can't masquerade as a
        # giant true range and blow the stop out to its cap.
        "adjustment": "split",
        "start": (now - timedelta(days=days)).strftime("%Y-%m-%d"),
        "end": (now - SIP_DELAY).isoformat(),
        "limit": 10000,
    }
    try:
        r = requests.get(f"{DATA_BASE_URL}/v2/stocks/bars", headers=headers,
                         params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            logger.debug(f"Alpaca bars {ticker}: HTTP {r.status_code} {r.text[:120]}")
            return None
        bars = (r.json().get("bars") or {}).get(symbol) or []
        return bars or None
    except Exception as e:   # network, JSON, anything: fall back quietly
        logger.debug(f"Alpaca bars {ticker} failed: {e}")
        return None


def daily_hlc(ticker: str, days: int = 45) -> Optional[tuple]:
    """(highs, lows, closes), oldest first -- the shape calculate_atr_stop
    consumes -- or None."""
    bars = daily_bars(ticker, days=days)
    if not bars:
        return None
    try:
        return ([float(b["h"]) for b in bars], [float(b["l"]) for b in bars],
                [float(b["c"]) for b in bars])
    except (KeyError, TypeError, ValueError):
        return None


def last_trade_price(ticker: str, now: datetime = None) -> Optional[float]:
    """Latest IEX trade, if it printed within IEX_MAX_AGE; else None.
    Last-resort only -- see the module docstring for why."""
    headers = _headers()
    if headers is None:
        return None
    now = now or datetime.now(timezone.utc)
    symbol = alpaca_symbol(ticker)
    try:
        r = requests.get(f"{DATA_BASE_URL}/v2/stocks/{symbol}/trades/latest",
                         headers=headers, params={"feed": "iex"}, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        trade = r.json().get("trade") or {}
        price, ts = trade.get("p"), trade.get("t")
        if not price or not ts:
            return None
        printed = parse_ts(ts)
        if printed is None or now - printed > IEX_MAX_AGE:
            return None
        return float(price)
    except Exception as e:
        logger.debug(f"Alpaca last trade {ticker} failed: {e}")
        return None
