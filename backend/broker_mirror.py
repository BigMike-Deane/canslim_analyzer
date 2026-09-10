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

PHASE 2 -- RESTING STOPS (2026-09-10). Phase 1 sells on a stop only after
the app's 15-minute checker has booked it, so it measures fill realism, not
stop behaviour. Go-live criterion 4 is about the latter: how far past the
stop the exit actually lands. So every broker-held position also carries a
sell-stop order AT THE BROKER, at the app's effective HARD stop (base,
ATR-widened, new-position guard -- the same helpers the checker calls).
DAY orders, because that is the only TIF Alpaca allows for fractional qty;
one placed after the close queues for the next open, so an overnight gap
fills at the opening print exactly as a real resting stop would -- the
residual the 15-minute checker cannot see.

  * HARD STOPS ONLY. Production fires trailing stops once a day in the
    close window (trailing_cadence.daily_only); a resting trailing order
    would fire intraday at levels the strategy never acts on.
  * The book stays the source of truth. A resting stop that fires while
    the app still holds leaves the broker FLAT (owner's call, Sep-10: a
    real account running resting stops would be flat). Later adds on that
    ticker are skipped; the app's own exit, when it comes, is "covered" by
    the stop's fill and paired with it -- same_exit when the app also
    hard-stopped that day, whipsaw otherwise.
  * Shares a resting stop reserves are not available to another sell, so
    any mirrored sell cancels the ticker's stop first and waits for it.
"""

import logging
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
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
# A resting stop's lifecycle. "lapsed" = expired at the close or canceled by
# us (replaced / released for a sell) -- routine, never an alert.
STOP_OPEN = ("pending",) + OPEN_STATUSES
# An ATR fetch the mirror had to make itself (cache cold after a restart) is
# not retried more often than this -- Yahoo throttles, and a stop left
# unplaced for a while costs coverage, where a wrong stop corrupts the read.
ATR_FETCH_RETRY_MINUTES = 30
_atr_fetch_attempts: dict = {}   # ticker -> last attempt (naive UTC)


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


_RESTING_STOP_DEFAULTS = {
    "enabled": True,
    # Re-place a working stop only when its level moves more than this. The
    # ATR stop drifts with every price update; re-placing each wiggle would
    # churn orders for no change in protection.
    "replace_tolerance_pct": 0.25,
    # Don't place a stop within this distance of the market: the broker
    # rejects a sell stop at/above the price, and the app's own checker is
    # about to exit that position anyway.
    "through_margin_pct": 0.2,
    # After a refused/failed placement, wait this long before trying again.
    "retry_minutes": 15,
}


def resting_stop_config() -> dict:
    try:
        from config_loader import config
        cfg = config.get("broker_mirror.resting_stops", {}) or {}
    except Exception:
        cfg = {}
    return {k: type(v)(cfg.get(k, v)) for k, v in _RESTING_STOP_DEFAULTS.items()}


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

    def submit_stop_order(self, symbol: str, qty: float, stop_price: float,
                          client_order_id: str) -> dict:
        """Sell-stop (stop-market). DAY: the only TIF allowed for fractional
        qty; placed after the close it queues for the next session."""
        return self._req("POST", "/v2/orders", json={
            "symbol": symbol,
            "qty": _qty_str(qty),
            "side": "sell",
            "type": "stop",
            "stop_price": _price_str(stop_price),
            "time_in_force": "day",
            "client_order_id": client_order_id,
        })

    def cancel_order(self, order_id: str):
        return self._req("DELETE", f"/v2/orders/{order_id}")

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


def round_stop_price(price: float) -> float:
    """Floor to the broker's tick: 2 decimals at/above $1, 4 below. DOWN,
    never up -- a stop rounded up would fire before the app's would."""
    if price >= 1:
        return math.floor(price * 100 + 1e-6) / 100
    return math.floor(price * 10_000 + 1e-6) / 10_000


def _price_str(price: float) -> str:
    return f"{price:.2f}" if price >= 1 else f"{price:.4f}"


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
    if row.action == "STOP":
        _apply_stop_status(row, status, order)
        return
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


def _apply_stop_status(row, status: str, order: dict):
    """Resting stops end routinely -- expired at the close, canceled by us
    for a replacement or a sell -- so those map to "lapsed", which never
    alerts. A stop left alerting every evening would train the owner to
    ignore the one alert that matters."""
    if status == "filled":
        row.status = "filled"
        row.filled_at = _parse_ts(order.get("filled_at")) or _utcnow()
        # vs the STOP price: how far past the trigger the exit landed.
        row.slippage_bps = slippage_bps("sell", row.stop_price, row.filled_avg_price)
    elif status in ("canceled", "expired"):
        row.status = "lapsed"
        if not row.note:
            row.note = "expired at the close" if status == "expired" else "canceled"
    elif status == "rejected":
        row.status = "rejected"
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


def _open_stop(db, user_id: int, ticker: str):
    """The ticker's working resting stop, if any."""
    from backend.database import BrokerMirrorOrder
    return db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.user_id == user_id,
        BrokerMirrorOrder.ticker == ticker,
        BrokerMirrorOrder.action == "STOP",
        BrokerMirrorOrder.status.in_(STOP_OPEN),
    ).order_by(BrokerMirrorOrder.id.desc()).first()


def open_stop_exit(db, user_id: int, ticker: str):
    """A resting stop that FIRED and has not yet been matched by the book's
    own full exit -- i.e. the broker is flat on ``ticker`` because of it.
    Closed by a "covered" full-exit sell linked to it."""
    from backend.database import BrokerMirrorOrder
    stop = db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.user_id == user_id,
        BrokerMirrorOrder.ticker == ticker,
        BrokerMirrorOrder.action == "STOP",
        BrokerMirrorOrder.status == "filled",
    ).order_by(BrokerMirrorOrder.id.desc()).first()
    if stop is None:
        return None
    closed = db.query(BrokerMirrorOrder.id).filter(
        BrokerMirrorOrder.user_id == user_id,
        BrokerMirrorOrder.linked_order_id == stop.id,
        BrokerMirrorOrder.status == "covered",
        BrokerMirrorOrder.close_all.is_(True),
    ).first()
    return None if closed else stop


def submit_pending(db, client: AlpacaPaperClient) -> dict:
    from backend.database import BrokerMirrorOrder

    pending = db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.status == "pending",
        BrokerMirrorOrder.action != "STOP",   # stops have their own lifecycle
    ).order_by(BrokerMirrorOrder.id).all()
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
            if row.side == "sell" and _open_stop(db, row.user_id, row.ticker) is not None:
                # The stop still reserves the shares; settle_against_stops
                # has asked the broker to cancel it.
                row.note = "waiting for the resting stop to cancel"
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


def poll_open(db, client: AlpacaPaperClient, include_stops: bool = True) -> int:
    from backend.database import BrokerMirrorOrder

    q = db.query(BrokerMirrorOrder).filter(BrokerMirrorOrder.status.in_(OPEN_STATUSES))
    if not include_stops:
        q = q.filter(BrokerMirrorOrder.action != "STOP")
    rows = q.all()
    for row in rows:
        try:
            _apply_broker_order(row, client.order_by_client_id(row.client_order_id))
        except (AlpacaError, requests.RequestException) as e:
            logger.warning(f"broker mirror: poll {row.client_order_id} failed: {e}")
    if rows:
        db.commit()
    return len(rows)


# ---------------------------------------------------------------- resting stops

def stop_context(db, user_id: int) -> dict:
    """Per-cycle inputs to the hard stop, resolved the way the live checker
    resolves them (ai_trader._check_and_execute_stop_losses_impl)."""
    from backend.database import AIPortfolioConfig
    from backend.trading_utils import get_strategy_profile, select_effective_stop_loss_pct
    from config_loader import config as yaml_config

    pcfg = db.query(AIPortfolioConfig).filter(AIPortfolioConfig.user_id == user_id).first()
    profile = get_strategy_profile(getattr(pcfg, "strategy", None) or "balanced")
    stops_cfg = yaml_config.get("ai_trader.stops", {}) or {}
    bearish = False
    try:
        from data_fetcher import get_cached_market_direction
        md = get_cached_market_direction() or {}
        spy = md.get("indexes", {}).get("SPY", {}) if md.get("success") else {}
        price, ma50 = spy.get("price", 0), spy.get("ma_50", 0)
        bearish = bool(price and ma50 and price < ma50)
    except Exception as e:
        logger.debug(f"broker mirror: market direction unavailable ({e})")
    base = select_effective_stop_loss_pct(
        profile=profile,
        stop_loss_config=stops_cfg,
        default_normal_pct=(pcfg.stop_loss_pct if pcfg and pcfg.stop_loss_pct else 15.0),
        is_bearish_market=bearish,
        # vix_stops is disabled in every environment; the checker's VIX
        # lookup preloads a month of history, too heavy for a 1-min job.
        vix_proxy=None, vix_config=None,
    )
    return {
        "base_pct": base,
        "use_atr": bool(stops_cfg.get("use_atr_stops", True)),
        "guard_cfg": yaml_config.get("ai_trader.new_position_guard", {}) or {},
    }


def hard_stop_level(position, ctx: dict, today=None) -> Optional[dict]:
    """The app's effective HARD stop for ``position``: ATR-widened base,
    then the new-position guard -- the chain both live stop paths run.

    The ATR stop is read from the cache the live checker writes each cycle
    (same process), so the broker stop follows the checker's latest value.
    A cold cache (fresh restart) gets one throttled fetch; if that fails the
    position gets NO stop rather than the bare base -- a stop tighter than
    the app's would fire where the app would not, and read as a whipsaw that
    never was.
    """
    from backend.trading_engine import get_cached_atr_stop, apply_new_position_guard

    cost, price = position.cost_basis, position.current_price
    if not cost or cost <= 0 or not price or price <= 0:
        return None
    ticker = position.ticker
    if ctx["use_atr"]:
        pct = get_cached_atr_stop(ticker)
        if pct is None:
            now = _utcnow()
            last = _atr_fetch_attempts.get(ticker)
            if last and now - last < timedelta(minutes=ATR_FETCH_RETRY_MINUTES):
                return None
            _atr_fetch_attempts[ticker] = now
            from backend.ai_trader import calculate_atr_stop
            calculate_atr_stop(ticker, price, ctx["base_pct"])   # caches on success
            pct = get_cached_atr_stop(ticker)
            if pct is None:
                return None
    else:
        pct = ctx["base_pct"]
    if position.purchase_date:
        bought = position.purchase_date.date() if hasattr(position.purchase_date, "date") else position.purchase_date
        held_days = ((today or datetime.now(timezone.utc).date()) - bought).days
        pct = apply_new_position_guard(
            pct, guard_config=ctx["guard_cfg"], holding_days=held_days,
            pyramid_count=getattr(position, "pyramid_count", 0) or 0,
        )
    return {"stop_pct": round(pct, 4), "cost_basis": cost,
            "stop_price": round_stop_price(cost * (1 - pct / 100.0))}


def _cancel_stop(client, stop, why: str):
    """Ask the broker to cancel a resting stop and record where it ended up.
    The stop may have fired in the meantime -- the caller must check
    ``stop.status`` afterwards (filled / lapsed / still open)."""
    try:
        order = client.order_by_client_id(stop.client_order_id)
    except AlpacaError as e:
        if e.status == 404:   # never reached the broker
            stop.status, stop.note = "lapsed", f"{why} (never reached the broker)"
            return
        raise
    _apply_broker_order(stop, order)
    if stop.status in STOP_OPEN:
        try:
            client.cancel_order(stop.broker_order_id)
        except AlpacaError as e:
            if e.status != 422:   # 422 = no longer cancelable (it just filled)
                raise
        _apply_broker_order(stop, client.order_by_client_id(stop.client_order_id))
    if stop.status == "lapsed":
        stop.note = why


def _cover(row, stop):
    """The book's sell arrives after a resting stop already exited the
    broker: nothing to send. Pair the two fills instead."""
    same_day = bool(stop.filled_at and row.internal_at
                    and stop.filled_at.date() == row.internal_at.date())
    row.status = "covered"
    row.linked_order_id = stop.id
    row.pairing = ("same_exit" if (row.reason or "").startswith("STOP LOSS") and same_day
                   else "whipsaw")
    row.filled_avg_price = stop.filled_avg_price
    row.filled_at = stop.filled_at
    # Broker's exit vs the price the app booked. For same_exit this is the
    # resting stop vs the 15-min checker; for a whipsaw, the cost (positive)
    # or saving of having been stopped out early.
    row.slippage_bps = slippage_bps("sell", row.internal_price, stop.filled_avg_price)
    when = f" on {stop.filled_at:%b-%d}" if stop.filled_at else ""
    row.note = (f"broker already exited via resting stop at "
                f"${stop.filled_avg_price or 0:.2f}{when}")


def settle_against_stops(db, client, user_id: int) -> int:
    """Run before any mirrored order goes out.

    * A sell whose ticker has a working stop: cancel the stop first (it
      reserves the shares). If the stop fired in the meantime, the sell is
      covered by it.
    * A sell or buy on a ticker the broker is flat on because a stop fired
      while the book held: covered (sell) / skipped (buy -- the broker stays
      flat until the book exits too).
    """
    from backend.database import BrokerMirrorOrder

    rows = db.query(BrokerMirrorOrder).filter(
        BrokerMirrorOrder.user_id == user_id,
        BrokerMirrorOrder.status == "pending",
        BrokerMirrorOrder.action.notin_(("SEED", "STOP")),
    ).order_by(BrokerMirrorOrder.id).all()
    n = 0
    for row in rows:
        try:
            exited = open_stop_exit(db, user_id, row.ticker)
            if row.side == "buy":
                if exited is not None:
                    row.status, row.linked_order_id = "skipped", exited.id
                    row.note = "broker flat: resting stop fired while the book held"
                    n += 1
                continue
            stop = _open_stop(db, user_id, row.ticker)
            if stop is not None:
                row.linked_order_id = stop.id   # a stop was working at this exit
                _cancel_stop(client, stop, "canceled: the book is selling")
                if stop.status == "filled":
                    exited = stop               # raced: it fired first
                elif stop.status in STOP_OPEN:
                    continue                    # cancel still settling
            if exited is not None:
                _cover(row, exited)
                n += 1
        except (AlpacaError, requests.RequestException) as e:
            logger.warning(f"broker mirror: settling {row.client_order_id} failed: {e}")
        finally:
            db.commit()
    return n


def _submit_stop(client, row, symbol: str):
    try:
        try:
            order = client.submit_stop_order(symbol, row.submitted_qty, row.stop_price,
                                             row.client_order_id)
        except AlpacaError as e:
            if e.status == 422 and "unique" in e.message.lower():
                order = client.order_by_client_id(row.client_order_id)
            else:
                raise
        row.submitted_at = _utcnow()
        row.status = "submitted"
        _apply_broker_order(row, order)
        return True
    except AlpacaError as e:
        if e.status >= 500 or e.status == 429:
            row.note = f"transient: {e}"
        elif "stop price" in e.message.lower():
            # Price moved through the level between our check and the
            # broker's: the app's checker owns this exit. Routine, quiet.
            row.status, row.note = "skipped", f"broker refused the level: {e.message[:120]}"
        else:
            row.status, row.note = "error", str(e)
    except requests.RequestException as e:
        row.note = f"transient: {e.__class__.__name__}"
    return False


def manage_resting_stops(db, client, user_id: int) -> dict:
    """Keep exactly one working DAY sell-stop per broker-held position, at
    the app's effective hard stop. Placed, re-placed when the level or the
    quantity changes, and canceled when the book no longer holds."""
    from backend.database import AIPortfolioPosition, BrokerMirrorOrder

    cfg = resting_stop_config()
    out = {"placed": 0, "replaced": 0, "canceled": 0, "through": 0}
    open_stops = {}
    for r in db.query(BrokerMirrorOrder).filter(
            BrokerMirrorOrder.user_id == user_id,
            BrokerMirrorOrder.action == "STOP",
            BrokerMirrorOrder.status.in_(STOP_OPEN)).order_by(BrokerMirrorOrder.id).all():
        open_stops[r.ticker] = r

    positions = {p.ticker: p for p in db.query(AIPortfolioPosition).filter(
        AIPortfolioPosition.user_id == user_id).all()}

    def _release(stop, why):
        try:
            _cancel_stop(client, stop, why)
            out["canceled"] += 1
        except (AlpacaError, requests.RequestException) as e:
            logger.warning(f"broker mirror: cancel {stop.client_order_id} failed: {e}")
        db.commit()

    if not cfg["enabled"]:
        for stop in open_stops.values():
            _release(stop, "canceled: resting stops disabled")
        return out
    for t, stop in list(open_stops.items()):
        if t not in positions:
            _release(stop, "canceled: the book no longer holds")
            del open_stops[t]

    # Tickers with a mirrored order still in flight: their quantity is about
    # to change, so wait for it rather than place a stop on the old size.
    busy = {t for (t,) in db.query(BrokerMirrorOrder.ticker).filter(
        BrokerMirrorOrder.user_id == user_id,
        BrokerMirrorOrder.action != "STOP",
        BrokerMirrorOrder.status.in_(STOP_OPEN)).all()}
    retry_after = _utcnow() - timedelta(minutes=cfg["retry_minutes"])
    cooling = {t for (t,) in db.query(BrokerMirrorOrder.ticker).filter(
        BrokerMirrorOrder.user_id == user_id,
        BrokerMirrorOrder.action == "STOP",
        BrokerMirrorOrder.status.in_(("skipped", "rejected", "error")),
        BrokerMirrorOrder.created_at >= retry_after).all()}

    broker = {p["symbol"]: p for p in client.positions()}
    ctx = None
    for t in sorted(positions):
        if t in busy or t in cooling:
            continue
        bp = broker.get(t) or broker.get(t.replace("-", "."))
        held = float((bp or {}).get("qty") or 0)
        if held <= 0:
            continue    # nothing to protect (incl. flat after a fired stop)
        existing = open_stops.get(t)
        if existing is not None and existing.status == "pending":
            _submit_stop(client, existing, bp["symbol"])   # retry a transient failure
            db.commit()
            continue
        if ctx is None:
            ctx = stop_context(db, user_id)
        level = hard_stop_level(positions[t], ctx)
        if level is None:
            continue
        stop_price = level["stop_price"]
        if existing is not None:
            same_qty = abs((existing.submitted_qty or 0) - held) <= 1e-6 * max(held, 1.0)
            moved = (abs(stop_price - existing.stop_price) / existing.stop_price * 100
                     if existing.stop_price else 100.0)
            if same_qty and moved <= cfg["replace_tolerance_pct"]:
                continue
        market = float((bp or {}).get("current_price") or 0)
        if market and market <= stop_price * (1 + cfg["through_margin_pct"] / 100):
            out["through"] += 1   # at/through the stop: the checker's exit
            continue
        if existing is not None:
            _release(existing, "replaced: level or quantity changed")
            if existing.status != "lapsed":
                continue          # still canceling, or it just fired
            out["replaced"] += 1
        row = BrokerMirrorOrder(
            user_id=user_id, trade_id=None, ticker=t, action="STOP", side="sell",
            internal_qty=held, internal_price=positions[t].current_price,
            internal_at=_utcnow(),
            reason=(f"RESTING STOP: {level['stop_pct']:.2f}% below "
                    f"cost ${level['cost_basis']:.2f}"),
            client_order_id=f"cs-u{user_id}-x{t}-{_utcnow():%y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}",
            status="pending", submitted_qty=held,
            stop_price=stop_price, stop_pct=level["stop_pct"], cost_basis=level["cost_basis"],
        )
        db.add(row)
        db.commit()   # row first: a crash mid-submit is adopted, never doubled
        if _submit_stop(client, row, bp["symbol"]):
            out["placed"] += 1
        db.commit()
    return out


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
    uid = activation["user_id"]
    enqueued = enqueue_new_trades(db, activation)
    # Poll first so a stop that fired since the last tick is known before a
    # sell is matched against it.
    polled = poll_open(db, client)
    settled = settle_against_stops(db, client, uid)
    sub = submit_pending(db, client)
    # Market orders just sent: a filled buy can get its stop this same tick.
    polled += poll_open(db, client, include_stops=False)
    stops = manage_resting_stops(db, client, uid)
    alerted = alert_failures(db)
    if enqueued or sub["submitted"] or sub["errors"] or settled or stops["placed"] or stops["canceled"]:
        logger.info(f"Broker mirror: enqueued {enqueued}, settled {settled}, submitted "
                    f"{sub['submitted']}, skipped {sub['skipped']}, errors {sub['errors']}, "
                    f"polled {polled}, stops {stops}")
    return {"state": "active", "enqueued": enqueued, "settled": settled, **sub,
            "polled": polled, "stops": stops, "alerted": alerted}


# ---------------------------------------------------------------- read side

def reconcile(internal: dict, broker: dict, tol: float = 1e-6,
              whole_share_only: set = frozenset(), stopped_out: set = frozenset()) -> list:
    """Per-ticker quantity diff between the internal book and the broker.

    ``whole_share_only`` names assets the broker won't trade fractionally.
    Those are bought rounded DOWN (broker_qty), so a sub-share shortfall is
    the designed outcome, not drift -- reported as a match with a note.
    First seen live 2026-09-10: HWBK 108 at the broker vs 108.33 booked. A
    reconciliation that warns permanently on expected rounding trains the
    reader to ignore it.

    ``stopped_out`` names tickers the broker is flat on because a resting
    stop fired while the book held -- a designed divergence, flagged as such
    rather than as drift.
    """
    out = []
    for t in sorted(set(internal) | set(broker)):
        i, b = internal.get(t, 0.0), broker.get(t, 0.0)
        exact = abs(b - i) <= max(tol, 1e-6 * max(i, b))
        rounded = (not exact and t in whole_share_only and b <= i and (i - b) < 1.0)
        row = {"ticker": t, "internal_qty": round(i, 6), "broker_qty": round(b, 6),
               "diff": round(b - i, 6), "match": exact or rounded}
        if rounded:
            row["note"] = "rounded down to whole shares (not fractionable at broker)"
        elif not exact and t in stopped_out and b == 0:
            row["divergence"] = "resting_stop"
            row["note"] = "flat at broker: resting stop fired, the book still holds"
        out.append(row)
    return out


def stop_slippage_pp(stop) -> Optional[float]:
    """How far past its stop a fired resting stop filled, in % of cost
    basis -- the unit of go-live criterion 4 (the app's signal_factors
    slippage_pp is -gain_pct - stop_pct, i.e. (stop - fill) / cost)."""
    if not stop.stop_price or not stop.filled_avg_price or not stop.cost_basis:
        return None
    return round((stop.stop_price - stop.filled_avg_price) / stop.cost_basis * 100, 2)


def summarize(rows: list, internal_slippage_pp: Optional[dict] = None) -> dict:
    """Fill-quality stats over mirrored TRADES (seeds excluded: no booked
    price; resting stops reported separately). ``internal_slippage_pp`` maps
    trade_id -> the app's own recorded slippage_pp, for pairing."""
    trades = [r for r in rows if r.action not in ("SEED", "STOP")]
    # A whipsaw's "slippage" compares fills days apart -- not a fill-quality number.
    filled = [r for r in trades if r.slippage_bps is not None and r.pairing != "whipsaw"]

    def _avg(xs, nd=1):
        return round(sum(xs) / len(xs), nd) if xs else None

    by_side = {}
    for side in ("buy", "sell"):
        xs = [r.slippage_bps for r in filled if r.side == side]
        by_side[side] = {"n": len(xs), "avg_bps": _avg(xs)}
    stops = [r.slippage_bps for r in filled
             if r.side == "sell" and ("STOP LOSS" in (r.reason or "") or "TRAILING STOP" in (r.reason or ""))]

    stop_rows = {r.id: r for r in rows if r.action == "STOP"}
    covered = [r for r in trades if r.status == "covered"]
    same = [r for r in covered if r.pairing == "same_exit"]
    whip = [r for r in covered if r.pairing == "whipsaw"]
    broker_pp = [stop_slippage_pp(stop_rows[r.linked_order_id]) for r in same
                 if r.linked_order_id in stop_rows]
    ipp = internal_slippage_pp or {}
    app_pp = [ipp[r.trade_id] for r in same if ipp.get(r.trade_id) is not None]
    # The app hard-stopped while a resting stop was working and it did NOT
    # fire: the two stop levels disagreed. Should stay ~0.
    missed = [r for r in trades
              if r.side == "sell" and (r.reason or "").startswith("STOP LOSS")
              and r.status != "covered" and r.linked_order_id in stop_rows
              and stop_rows[r.linked_order_id].status != "filled"]
    return {
        "n_mirrored": len(trades),
        "n_filled": len(filled),
        "avg_slippage_bps": _avg([r.slippage_bps for r in filled]),
        "by_side": by_side,
        "stop_sells": {"n": len(stops), "avg_bps": _avg(stops)},
        "n_failed": sum(1 for r in rows if r.status in FAILED_STATUSES),
        "n_skipped": sum(1 for r in trades if r.status == "skipped"),
        "n_open": sum(1 for r in trades if r.status in STOP_OPEN),
        "resting_stops": {
            "working": sum(1 for r in stop_rows.values() if r.status in OPEN_STATUSES),
            "fired": sum(1 for r in stop_rows.values() if r.status == "filled"),
            "same_exit": {
                "n": len(same),
                "avg_bps_vs_booked": _avg([r.slippage_bps for r in same if r.slippage_bps is not None]),
                "broker_slippage_pp": _avg([x for x in broker_pp if x is not None], 2),
                "app_slippage_pp": _avg(app_pp, 2),
                "n_app_measured": len(app_pp),
            },
            "whipsaw": {"n": len(whip),
                        "avg_bps": _avg([r.slippage_bps for r in whip if r.slippage_bps is not None])},
            "missed": len(missed),
        },
    }
