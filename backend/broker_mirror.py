"""Broker mirror: copy one live book's trades into an Alpaca PAPER account.

Why (2026-09-10): the go-live gate needs execution realism the internal
book cannot supply. The app books every trade at the price IT saw; nothing
measures what a real broker would have filled. The Sep-10 twin-account
finding sharpened this -- two books on identical signals drifted ~7pp apart
from sizing/fill path dependence alone -- so the gap between a real account
and its paper twin is not a rounding error, it is a number the go-live
decision needs.

Design rules, in priority order:

1. THE INTERNAL BOOK STAYS THE SOURCE OF TRUTH. The mirror never feeds back
   into a strategy decision. Every running clock and arm depends on that.
2. NEVER TOUCH THE TRADE PATH. The mirror tails COMMITTED rows of
   ai_portfolio_trades from a scheduler job. A hook inside execute_trade
   would run before the caller's commit: a rolled-back trade would leave an
   orphan broker order, and a broker failure could break a live trade.
3. PAPER ONLY, STRUCTURALLY. The base URL is a constant with no override.
   Pointing this at a live account is a code change and an owner decision,
   not a config flip.
4. SHARE-FOR-SHARE. Broker qty == internal qty, so every fill is a direct
   price comparison and reconciliation is a quantity diff. The paper
   account's surplus cash simply sits idle.
5. IDEMPOTENT. client_order_id = "cs-u<user>-t<trade>" (or -s<n> for seeds);
   a retry after a timeout adopts the existing broker order instead of
   placing a second one.

Activation is an explicit act (``seed``): it copies the book held at that
moment (SEED rows) and records a trade-id watermark; only trades above the
watermark are mirrored. Until then the job is inert.
"""

import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ⚑ Paper, and only paper. Deliberately NOT read from the environment.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"

ACTIVATION_KEY = "broker_mirror_activation"
OPEN_STATUSES = ("submitted", "partially_filled")
FAILED_STATUSES = ("rejected", "canceled", "expired", "error")
# Alpaca statuses that still resolve on their own.
_BROKER_WORKING = {"new", "accepted", "pending_new", "accepted_for_bidding",
                   "held", "calculated", "pending_replace", "pending_cancel",
                   "replaced", "done_for_day", "partially_filled"}
PENDING_ALERT_MINUTES = 30


class AlpacaError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"Alpaca {status}: {message}")
        self.status = status
        self.message = message


def credentials() -> Optional[tuple]:
    key = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    secret = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()
    return (key, secret) if key and secret else None


def mirror_config() -> dict:
    try:
        from config_loader import config
        return {
            "enabled": bool(config.get("broker_mirror.enabled", True)),
            "user_id": int(config.get("broker_mirror.user_id", 1)),
        }
    except Exception:
        return {"enabled": True, "user_id": 1}


class AlpacaPaperClient:
    """Minimal REST client for the handful of endpoints the mirror needs."""

    def __init__(self, key: str, secret: str, timeout: float = 10.0):
        self.base = PAPER_BASE_URL
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        })

    def _req(self, method: str, path: str, **kw):
        r = self.session.request(method, self.base + path, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            try:
                msg = r.json().get("message") or r.text
            except ValueError:
                msg = r.text
            raise AlpacaError(r.status_code, str(msg)[:300])
        return r.json() if r.content else None

    def account(self) -> dict:
        return self._req("GET", "/v2/account")

    def clock(self) -> dict:
        return self._req("GET", "/v2/clock")

    def asset(self, symbol: str) -> dict:
        return self._req("GET", f"/v2/assets/{symbol}")

    def positions(self) -> list:
        return self._req("GET", "/v2/positions") or []

    def submit_order(self, symbol: str, qty: float, side: str, client_order_id: str) -> dict:
        return self._req("POST", "/v2/orders", json={
            "symbol": symbol,
            "qty": _qty_str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",   # the only TIF Alpaca allows for fractional qty
            "client_order_id": client_order_id,
        })

    def order_by_client_id(self, client_order_id: str) -> dict:
        return self._req("GET", "/v2/orders:by_client_order_id",
                         params={"client_order_id": client_order_id})


def get_client() -> Optional[AlpacaPaperClient]:
    creds = credentials()
    return AlpacaPaperClient(*creds) if creds else None


# ---------------------------------------------------------------- pure helpers

def _qty_str(qty: float) -> str:
    """Alpaca accepts up to 9 decimals; trim float noise and trailing zeros."""
    return f"{qty:.9f}".rstrip("0").rstrip(".")


def slippage_bps(side: str, internal_price: Optional[float], fill_price: Optional[float]) -> Optional[float]:
    """Positive = worse for us (paid more on a buy, received less on a sell)."""
    if not internal_price or not fill_price or internal_price <= 0:
        return None
    if side == "buy":
        return round((fill_price - internal_price) / internal_price * 1e4, 1)
    return round((internal_price - fill_price) / internal_price * 1e4, 1)


def broker_qty(internal_qty: float, fractionable: bool, side: str,
               held: float, close_all: bool) -> tuple:
    """Quantity to send and a note explaining any change from internal_qty.

    Sells never exceed what the broker holds; a full internal exit sells the
    broker's whole position so no dust is left behind. Non-fractionable
    assets round DOWN to whole shares (never buy more than the book did).
    """
    note = None
    if side == "sell":
        if held <= 0:
            return 0.0, "broker holds none"
        if close_all:
            return held, None
        qty = min(internal_qty, held)
        if qty < internal_qty:
            note = f"capped at broker holding {held:g}"
    else:
        qty = internal_qty
    if not fractionable and qty != math.floor(qty):
        whole = float(math.floor(qty))
        note = f"not fractionable: {qty:g} -> {whole:g}"
        qty = whole
    if qty <= 0:
        return 0.0, note or "zero quantity"
    return round(qty, 9), note


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------- activation

def get_activation() -> Optional[dict]:
    from backend.database import get_system_setting
    return get_system_setting(ACTIVATION_KEY, None)


def seed(db, client: AlpacaPaperClient = None) -> dict:
    """Activate the mirror: copy the currently-held book as SEED rows and set
    the trade-id watermark. Refuses to run twice."""
    from sqlalchemy import func
    from backend.database import (
        AIPortfolioPosition, AIPortfolioTrade, BrokerMirrorOrder, set_system_setting,
    )

    if get_activation():
        return {"ok": False, "error": "already activated"}
    client = client or get_client()
    if client is None:
        return {"ok": False, "error": "no Alpaca credentials (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)"}
    account = client.account()  # fail fast on bad keys before writing anything
    uid = mirror_config()["user_id"]

    watermark = db.query(func.max(AIPortfolioTrade.id)).filter(
        AIPortfolioTrade.user_id == uid).scalar() or 0
    held_at_broker = {p["symbol"]: float(p.get("qty") or 0) for p in client.positions()}
    positions = db.query(AIPortfolioPosition).filter(
        AIPortfolioPosition.user_id == uid).all()

    rows = []
    for i, p in enumerate(sorted(positions, key=lambda x: x.ticker)):
        qty = (p.shares or 0) - held_at_broker.get(p.ticker, 0.0)
        if qty <= 0:
            continue
        rows.append(BrokerMirrorOrder(
            user_id=uid, trade_id=None, ticker=p.ticker, action="SEED", side="buy",
            internal_qty=round(qty, 9), internal_price=p.current_price,
            internal_at=_utcnow(), reason="SEED: copy of the book held at activation",
            client_order_id=f"cs-u{uid}-s{watermark}-{i}", status="pending",
        ))

    # Optimistic race guard: a trade landing between the watermark read and
    # here would be copied twice (in the seed AND as a mirrored trade).
    after = db.query(func.max(AIPortfolioTrade.id)).filter(
        AIPortfolioTrade.user_id == uid).scalar() or 0
    if after != watermark:
        db.rollback()
        return {"ok": False, "error": "a trade executed during seeding -- retry"}

    for r in rows:
        db.add(r)
    db.commit()
    activation = {
        "user_id": uid,
        "watermark_trade_id": watermark,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "account_number": account.get("account_number"),
    }
    set_system_setting(ACTIVATION_KEY, activation)
    logger.info(f"Broker mirror ACTIVATED for user {uid}: {len(rows)} seed orders, "
                f"watermark trade #{watermark}")
    return {"ok": True, "seeded": len(rows), **activation}


# ---------------------------------------------------------------- the job

def enqueue_new_trades(db, activation: dict) -> int:
    """Create a pending mirror row for every committed trade above the
    watermark that does not have one yet."""
    from backend.database import AIPortfolioTrade, AIPortfolioPosition, BrokerMirrorOrder

    uid = activation["user_id"]
    trades = db.query(AIPortfolioTrade).outerjoin(
        BrokerMirrorOrder, BrokerMirrorOrder.trade_id == AIPortfolioTrade.id
    ).filter(
        AIPortfolioTrade.user_id == uid,
        AIPortfolioTrade.id > activation["watermark_trade_id"],
        BrokerMirrorOrder.id.is_(None),
    ).order_by(AIPortfolioTrade.id).all()

    held = {t for (t,) in db.query(AIPortfolioPosition.ticker).filter(
        AIPortfolioPosition.user_id == uid).all()}
    n = 0
    for t in trades:
        action = (t.action or "").upper()
        side = {"BUY": "buy", "PYRAMID": "buy", "SELL": "sell"}.get(action)
        row = BrokerMirrorOrder(
            user_id=uid, trade_id=t.id, ticker=t.ticker, action=action or "?",
            side=side or "buy", internal_qty=t.shares or 0.0, internal_price=t.price,
            internal_at=t.executed_at, reason=(t.reason or "")[:250],
            # Full internal exit (position row gone) -> sell the broker's whole
            # holding, not the booked share count, so no dust is left.
            close_all=(side == "sell" and t.ticker not in held),
            client_order_id=f"cs-u{uid}-t{t.id}", status="pending",
        )
        if (t.reason or "").startswith("PAPER:"):
            row.status, row.note = "skipped", "internal dry-run (paper_mode) trade"
        elif side is None:
            row.status, row.note = "skipped", f"unmirrorable action {action!r}"
        db.add(row)
        n += 1
    if n:
        db.commit()
    return n


def _apply_broker_order(row, order: dict):
    row.broker_order_id = order.get("id") or row.broker_order_id
    status = order.get("status")
    filled = float(order.get("filled_qty") or 0)
    if order.get("filled_avg_price"):
        row.filled_avg_price = float(order["filled_avg_price"])
    row.filled_qty = filled
    if status == "filled":
        row.status = "filled"
        row.filled_at = _parse_ts(order.get("filled_at")) or _utcnow()
        if row.action != "SEED":   # a seed has no booked trade to compare against
            row.slippage_bps = slippage_bps(row.side, row.internal_price, row.filled_avg_price)
    elif status in ("canceled", "expired", "rejected"):
        # A partial fill that then expired still carries a usable price.
        row.status = status
        if filled > 0 and row.action != "SEED":
            row.slippage_bps = slippage_bps(row.side, row.internal_price, row.filled_avg_price)
    elif status == "partially_filled":
        row.status = "partially_filled"
    elif status in _BROKER_WORKING:
        row.status = "submitted"


def _resolve_asset(client: AlpacaPaperClient, ticker: str) -> dict:
    """Asset record, trying Alpaca's class-share spelling (BRK.B) when the
    app's (BRK-B) is unknown to the broker."""
    try:
        return client.asset(ticker)
    except AlpacaError as e:
        if e.status == 404 and "-" in ticker:
            return client.asset(ticker.replace("-", "."))
        raise


def _buy_in_flight(db, sell_row) -> bool:
    from backend.database import BrokerMirrorOrder
    return db.query(BrokerMirrorOrder.id).filter(
        BrokerMirrorOrder.user_id == sell_row.user_id,
        BrokerMirrorOrder.ticker == sell_row.ticker,
        BrokerMirrorOrder.side == "buy",
        BrokerMirrorOrder.id < sell_row.id,
        BrokerMirrorOrder.status.in_(("pending",) + OPEN_STATUSES),
    ).first() is not None


def submit_pending(db, client: AlpacaPaperClient) -> dict:
    from backend.database import BrokerMirrorOrder

    pending = db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.status == "pending").order_by(BrokerMirrorOrder.id).all()
    if not pending:
        return {"submitted": 0, "skipped": 0, "errors": 0}

    held = {}
    for p in client.positions():
        held[p["symbol"]] = float(p.get("qty_available") or p.get("qty") or 0)
    assets = {}
    out = {"submitted": 0, "skipped": 0, "errors": 0}

    for row in pending:
        try:
            if row.ticker not in assets:
                assets[row.ticker] = _resolve_asset(client, row.ticker)
            asset = assets[row.ticker]
            symbol = asset.get("symbol") or row.ticker
            if not asset.get("tradable", False):
                row.status, row.note = "skipped", "asset not tradable at broker"
                out["skipped"] += 1
                continue
            if row.side == "sell" and held.get(symbol, 0.0) <= 0 and _buy_in_flight(db, row):
                # The buy this sell unwinds has not filled yet (e.g. a seed
                # queued for the open). Wait for it instead of skipping.
                row.note = "waiting for the matching buy to fill"
                continue
            qty, note = broker_qty(row.internal_qty, bool(asset.get("fractionable")),
                                   row.side, held.get(symbol, 0.0), bool(row.close_all))
            row.note = note
            if qty <= 0:
                row.status = "skipped"
                out["skipped"] += 1
                continue
            try:
                order = client.submit_order(symbol, qty, row.side, row.client_order_id)
            except AlpacaError as e:
                if e.status == 422 and "unique" in e.message.lower():
                    # Timed out earlier but the broker did accept it: adopt.
                    order = client.order_by_client_id(row.client_order_id)
                else:
                    raise
            row.submitted_qty = qty
            row.submitted_at = _utcnow()
            row.status = "submitted"
            _apply_broker_order(row, order)
            if row.side == "sell":
                held[symbol] = max(held.get(symbol, 0.0) - qty, 0.0)
            out["submitted"] += 1
        except AlpacaError as e:
            if e.status >= 500 or e.status == 429:
                row.note = f"transient: {e}"   # stays pending, retried next run
            else:
                row.status, row.note = "error", str(e)
                out["errors"] += 1
        except requests.RequestException as e:
            row.note = f"transient: {e.__class__.__name__}"
        finally:
            db.commit()
    return out


def poll_open(db, client: AlpacaPaperClient) -> int:
    from backend.database import BrokerMirrorOrder

    rows = db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.status.in_(OPEN_STATUSES)).all()
    for row in rows:
        try:
            _apply_broker_order(row, client.order_by_client_id(row.client_order_id))
        except (AlpacaError, requests.RequestException) as e:
            logger.warning(f"broker mirror: poll {row.client_order_id} failed: {e}")
    if rows:
        db.commit()
    return len(rows)


def alert_failures(db) -> int:
    """One ops alert per failed order, plus orders stuck pending. A mirror
    that fails silently reads exactly like one with nothing to do."""
    from datetime import timedelta
    from backend.database import BrokerMirrorOrder

    stale = _utcnow() - timedelta(minutes=PENDING_ALERT_MINUTES)
    rows = db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.alerted.isnot(True),
        (BrokerMirrorOrder.status.in_(FAILED_STATUSES))
        | ((BrokerMirrorOrder.status == "pending") & (BrokerMirrorOrder.created_at < stale)),
    ).all()
    if not rows:
        return 0
    lines = [f"{r.action} {r.ticker} {r.internal_qty:g} -> {r.status}"
             f"{f' ({r.note})' if r.note else ''}" for r in rows[:10]]
    try:
        from backend.email_utils import send_ops_alert
        send_ops_alert(
            f"Broker mirror: {len(rows)} order(s) need attention",
            "Alpaca paper mirror could not complete: " + "; ".join(lines),
            priority="default", tags=["broker_mirror"],
            data={"client_order_ids": [r.client_order_id for r in rows]},
        )
    except Exception as e:
        logger.warning(f"broker mirror: alert failed: {e}")
    for r in rows:
        r.alerted = True
    db.commit()
    return len(rows)


def run_mirror_cycle(db, client: AlpacaPaperClient = None) -> dict:
    """Scheduler entry point. Cheap no-op until configured AND activated."""
    if not mirror_config()["enabled"]:
        return {"state": "disabled"}
    activation = get_activation()
    if not activation:
        return {"state": "not_activated"}
    client = client or get_client()
    if client is None:
        return {"state": "no_credentials"}
    enqueued = enqueue_new_trades(db, activation)
    sub = submit_pending(db, client)
    polled = poll_open(db, client)
    alerted = alert_failures(db)
    if enqueued or sub["submitted"] or sub["errors"]:
        logger.info(f"Broker mirror: enqueued {enqueued}, submitted {sub['submitted']}, "
                    f"skipped {sub['skipped']}, errors {sub['errors']}, polled {polled}")
    return {"state": "active", "enqueued": enqueued, **sub, "polled": polled, "alerted": alerted}


# ---------------------------------------------------------------- read side

def reconcile(internal: dict, broker: dict, tol: float = 1e-6) -> list:
    """Per-ticker quantity diff between the internal book and the broker."""
    out = []
    for t in sorted(set(internal) | set(broker)):
        i, b = internal.get(t, 0.0), broker.get(t, 0.0)
        out.append({"ticker": t, "internal_qty": round(i, 6), "broker_qty": round(b, 6),
                    "diff": round(b - i, 6), "match": abs(b - i) <= max(tol, 1e-6 * max(i, b))})
    return out


def summarize(rows: list) -> dict:
    """Fill-quality stats over mirrored TRADES (seeds excluded: no booked price)."""
    filled = [r for r in rows if r.action != "SEED" and r.slippage_bps is not None]

    def _avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    by_side = {}
    for side in ("buy", "sell"):
        xs = [r.slippage_bps for r in filled if r.side == side]
        by_side[side] = {"n": len(xs), "avg_bps": _avg(xs)}
    stops = [r.slippage_bps for r in filled
             if r.side == "sell" and ("STOP LOSS" in (r.reason or "") or "TRAILING STOP" in (r.reason or ""))]
    return {
        "n_mirrored": sum(1 for r in rows if r.action != "SEED"),
        "n_filled": len(filled),
        "avg_slippage_bps": _avg([r.slippage_bps for r in filled]),
        "by_side": by_side,
        "stop_sells": {"n": len(stops), "avg_bps": _avg(stops)},
        "n_failed": sum(1 for r in rows if r.status in FAILED_STATUSES),
        "n_skipped": sum(1 for r in rows if r.status == "skipped"),
        "n_open": sum(1 for r in rows if r.status in OPEN_STATUSES + ("pending",)),
    }
