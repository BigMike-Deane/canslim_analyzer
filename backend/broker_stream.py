"""Alpaca trade-update stream for the broker mirror (2026-09-10).

The mirror polls every open order once a minute. That stays -- it is the
source of consistency -- but a fill no longer waits for the next poll: this
thread holds one websocket to Alpaca's PAPER ``trade_updates`` stream and
applies fills / cancels / rejections to their ``broker_mirror_orders`` row
the moment the broker reports them. Resting stops matter most here: a stop
that fires is recorded within a second, with the broker's own fill time.

Safety properties:

  * PAPER ONLY -- the URL derives from broker_mirror.PAPER_BASE_URL, the
    same no-override constant the order client uses.
  * ACCELERATOR, NOT AUTHORITY -- the stream only moves rows that are still
    OPEN; it never touches a finished row. If it races the minute job on a
    row (the fill lands before the submit is committed), the next poll
    corrects it -- worst case is exactly the pre-stream behaviour.
  * INERT UNTIL ACTIVATED -- no keys or no activation: the thread sleeps and
    re-checks; it never holds a connection for nothing.
  * ONE CONNECTION -- a module lock; uvicorn runs a single worker.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

IDLE_RECHECK_SECONDS = 60
RECV_TIMEOUT_SECONDS = 30
MAX_BACKOFF_SECONDS = 60
UNAUTHORIZED_BACKOFF_SECONDS = 300

_state = {"running": False, "connected": False, "connected_since": None,
          "last_event_at": None, "events": 0, "applied": 0, "reconnects": 0,
          "last_error": None}
_lock = threading.Lock()
_stop = threading.Event()
_thread = None


class StreamAuthError(Exception):
    pass


def stream_url() -> str:
    from backend.broker_mirror import PAPER_BASE_URL
    return PAPER_BASE_URL.replace("https://", "wss://") + "/stream"


def status() -> dict:
    return dict(_state)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- applying events

def handle_trade_update(db, data: dict) -> bool:
    """Apply one trade_updates event to its mirror row. True if a row moved."""
    from backend import broker_mirror as bm
    from backend.database import BrokerMirrorOrder

    order = (data or {}).get("order") or {}
    cid = order.get("client_order_id")
    if not cid:
        return False
    row = db.query(BrokerMirrorOrder).filter(BrokerMirrorOrder.client_order_id == cid).first()
    # Not ours (manual order in the account), or the submit hasn't been
    # committed yet -- the minute poll picks the latter up.
    if row is None or row.status not in bm.STOP_OPEN:
        return False
    before = (row.status, row.filled_qty)
    bm._apply_broker_order(row, order)
    moved = (row.status, row.filled_qty) != before
    if moved:
        db.commit()
        logger.info(f"Broker stream: {data.get('event')} {row.ticker} {row.action} -> {row.status}")
    else:
        db.rollback()
    return moved


def handle_message(raw, db_factory) -> None:
    """One websocket frame (Alpaca sends trade updates as binary JSON)."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    msg = json.loads(raw)
    if msg.get("stream") != "trade_updates":
        return
    _state["events"] += 1
    _state["last_event_at"] = _now_iso()
    db = db_factory()
    try:
        if handle_trade_update(db, msg.get("data") or {}):
            _state["applied"] += 1
    finally:
        db.close()


# ---------------------------------------------------------------- the connection

def _ready() -> bool:
    from backend import broker_mirror as bm
    return (bm.mirror_config()["enabled"] and bm.credentials() is not None
            and bm.get_activation() is not None)


def _expect(ws, stream: str, timeout: float = 10):
    raw = ws.recv(timeout=timeout)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    msg = json.loads(raw)
    if msg.get("stream") != stream:
        raise RuntimeError(f"expected {stream}, got {str(msg)[:120]}")
    return msg.get("data") or {}


def run_session(connect, db_factory, stop: threading.Event = _stop) -> None:
    """One connection: authenticate, subscribe, pump events until the
    socket drops or ``stop`` is set. Raises on failure (caller backs off)."""
    from backend.broker_mirror import credentials
    key, secret = credentials()
    with connect(stream_url(), open_timeout=10, close_timeout=5) as ws:
        ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
        auth = _expect(ws, "authorization")
        if auth.get("status") != "authorized":
            raise StreamAuthError(f"stream auth refused: {auth}")
        ws.send(json.dumps({"action": "listen", "data": {"streams": ["trade_updates"]}}))
        listening = _expect(ws, "listening")
        if "trade_updates" not in (listening.get("streams") or []):
            raise RuntimeError(f"not subscribed: {listening}")
        _state.update(connected=True, connected_since=_now_iso(), last_error=None)
        logger.info("Broker stream: connected to Alpaca paper trade_updates")
        while not stop.is_set():
            try:
                raw = ws.recv(timeout=RECV_TIMEOUT_SECONDS)
            except TimeoutError:
                # Quiet market (websockets' own pings keep it alive). Let go
                # of the connection if the mirror was disabled meanwhile.
                if not _ready():
                    return
                continue
            try:
                handle_message(raw, db_factory)
            except Exception as e:  # one bad frame must not drop the stream
                logger.warning(f"Broker stream: bad frame ({e})")


def _loop(connect, db_factory, sleep=time.sleep, stop: threading.Event = _stop):
    backoff = 1
    while not stop.is_set():
        if not _ready():
            sleep(IDLE_RECHECK_SECONDS)
            continue
        try:
            run_session(connect, db_factory, stop)
            backoff = 1
        except StreamAuthError as e:
            _state["last_error"] = str(e)[:300]
            logger.error(f"Broker stream: {e}")
            sleep(UNAUTHORIZED_BACKOFF_SECONDS)
        except Exception as e:
            _state["last_error"] = f"{e.__class__.__name__}: {e}"[:300]
            logger.warning(f"Broker stream dropped ({_state['last_error']}); retry in {backoff}s")
            sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
        finally:
            if _state["connected"]:
                _state["reconnects"] += 1
            _state["connected"] = False


def start_stream() -> bool:
    """Start the background thread once per process. False if already up."""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        from websockets.sync.client import connect
        from backend.database import SessionLocal
        _stop.clear()
        _thread = threading.Thread(target=_loop, args=(connect, SessionLocal),
                                   name="broker-stream", daemon=True)
        _thread.start()
        _state["running"] = True
        return True


def stop_stream():
    _stop.set()
    _state["running"] = False
