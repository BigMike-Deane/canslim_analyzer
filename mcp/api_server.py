#!/usr/bin/env python3
"""CANSLIM API MCP server (2026-09-10) -- read-only view of the LIVE app.

The db MCP reads tables; this one reads the app's COMPUTED answers -- the
go-live gate, the per-user alpha scoreboard, the broker mirror with its live
Alpaca reconciliation, system health -- through the same endpoints the Admin
page uses. Before it, every check-in either re-derived those numbers from
raw SQL or needed a fresh 30-minute JWT minted inside the container.

Auth: a READ-ONLY token (backend/auth.py READONLY_*): the backend refuses
anything but GET for it, and only the most recently minted one is valid.
It is read from $CANSLIM_API_TOKEN or ~/.config/canslim/readonly_token.
Mint once (the token goes straight into the file, never onto a screen):

    mkdir -p ~/.config/canslim && umask 077 && ssh root@100.104.189.36 \\
      'docker exec canslim-analyzer python3 -m backend.mint_readonly_token' \\
      > ~/.config/canslim/readonly_token

Responses are COMPACTED (long lists cut, long strings trimmed) -- the point
of the server is fewer tokens per check-in, not raw dumps.
"""

import json
import os

import requests
from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("CANSLIM_API_BASE", "http://100.104.189.36:8001").rstrip("/")
TOKEN_FILE = os.path.expanduser(
    os.environ.get("CANSLIM_API_TOKEN_FILE", "~/.config/canslim/readonly_token"))
MINT_HINT = ("mint one: mkdir -p ~/.config/canslim && umask 077 && ssh root@100.104.189.36 "
             "'docker exec canslim-analyzer python3 -m backend.mint_readonly_token' "
             "> ~/.config/canslim/readonly_token")
MAX_LIST = 15
MAX_STR = 400
MAX_CHARS = 20_000

mcp = FastMCP("canslim-api")


class ApiError(Exception):
    pass


def _token() -> str:
    tok = os.environ.get("CANSLIM_API_TOKEN", "").strip()
    if not tok and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            tok = f.read().strip()
    if not tok:
        raise ApiError(f"No read-only API token. {MINT_HINT}")
    return tok


def api_get(path: str, params: dict = None, http=requests) -> dict:
    if not path.startswith("/api/"):
        raise ApiError("path must start with /api/")
    r = http.get(BASE + path, params=params or {}, timeout=60,
                 headers={"Authorization": f"Bearer {_token()}", "Accept": "application/json"})
    if r.status_code == 401:
        raise ApiError(f"401 -- token expired, revoked or missing. {MINT_HINT}")
    if r.status_code >= 400:
        raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def compact(obj, max_list: int = MAX_LIST, max_str: int = MAX_STR):
    """Cut lists and strings, recursively; say what was cut."""
    if isinstance(obj, dict):
        return {k: compact(v, max_list, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        head = [compact(v, max_list, max_str) for v in obj[:max_list]]
        if len(obj) > max_list:
            head.append(f"... {len(obj) - max_list} more")
        return head
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "..."
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _dump(obj) -> str:
    text = json.dumps(obj, indent=1, default=str)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n... (truncated)"


def _safe(fn):
    try:
        return _dump(fn())
    except ApiError as e:
        return f"Error: {e}"
    except requests.RequestException as e:
        return f"Error: API unreachable at {BASE} ({e.__class__.__name__}) -- Tailscale up?"


# ---------------------------------------------------------------- shaping (pure)

def shape_program(gates: dict) -> dict:
    """experiment-gates -> the go-live gate, program clocks, and each arm's
    progress, without descriptions and notes the reader already knows."""
    clocks = gates.get("program_clocks") or {}
    go = clocks.get("go_live") or {}
    arms = []
    for a in gates.get("arms") or []:
        arms.append({
            "name": a.get("name"), "days": a.get("days_accrued"),
            "buys": a.get("buys"), "sells": a.get("sells"),
            "gates": [{k: m.get(k) for k in ("label", "n", "target", "value", "met", "kind")
                       if k in m} for m in a.get("gate_metrics") or []],
        })
    return {
        "go_live": {
            "n_met": go.get("n_met"), "n_total": go.get("n_total"),
            "all_met": go.get("all_met"), "blocking": go.get("blocking"),
            "criteria": [{"key": c.get("key"), "met": c.get("met"), "value": c.get("value"),
                          "target": c.get("target")} for c in go.get("criteria") or []],
        },
        "stop_loss_recheck": {k: (clocks.get("stop_loss_recheck") or {}).get(k)
                              for k in ("n", "target", "avg_loss_pct", "bar_pct", "verdict")},
        "vintage_spread": compact(clocks.get("vintage_spread"), max_list=6),
        "calendar": [c for c in clocks.get("calendar") or [] if (c.get("days_until") or 0) <= 60],
        "arms": arms,
    }


def shape_broker(bm: dict) -> dict:
    s = bm.get("summary") or {}
    recon = bm.get("reconciliation") or []
    return {
        "configured": bm.get("configured"), "activated": bool(bm.get("activation")),
        "account": bm.get("account"), "account_error": bm.get("account_error"),
        "market_open": bm.get("market_open"),
        "summary": s,
        "reconciliation": {
            "n_positions": len(recon),
            "mismatches": [r for r in recon if not r.get("match") and not r.get("divergence")],
            "designed_divergences": [r for r in recon if r.get("divergence")],
        },
        "resting_stops": bm.get("resting_stops"),
        "stop_events": (bm.get("stop_events") or [])[:10],
        "recent_orders": [{k: o.get(k) for k in ("internal_at", "action", "ticker", "status",
                                                 "internal_price", "filled_avg_price",
                                                 "slippage_bps", "pairing", "note")}
                          for o in (bm.get("orders") or [])[:10]],
    }


# ---------------------------------------------------------------- tools

@mcp.tool()
def program_status() -> str:
    """The go-live gate (5 pre-registered criteria: which pass, which block,
    current values vs targets), the stop-loss re-check, vintage path-noise
    spread, calendar clocks due within 60 days, and every experiment arm's
    accrual. Start here for "how's the program doing"."""
    return _safe(lambda: shape_program(api_get("/api/admin/experiment-gates")))


@mcp.tool()
def broker_mirror() -> str:
    """Alpaca PAPER mirror of the owner's book: account, fill slippage stats,
    resting stops (working + fired, broker vs app pp past the stop),
    position reconciliation against the live broker, recent mirrored orders."""
    return _safe(lambda: shape_broker(api_get("/api/admin/broker-mirror", {"limit": 20})))


@mcp.tool()
def portfolios() -> str:
    """Per-user scoreboard: each live book's alpha vs SPY over its own window
    (never raw return), plus the twin-account path-noise context."""
    return _safe(lambda: compact(api_get("/api/admin/user-portfolios")))


@mcp.tool()
def system_health() -> str:
    """App health: scan freshness, scheduler jobs, process memory/RSS,
    data gaps -- the System Health page's payload."""
    return _safe(lambda: compact(api_get("/api/system-health")))


@mcp.tool()
def owner_portfolio() -> str:
    """The owner's AI portfolio right now: cash, value, positions with
    exit plans (stop / trailing / target levels)."""
    return _safe(lambda: compact(api_get("/api/ai-portfolio")))


@mcp.tool()
def api_get_json(path: str, params_json: str = "{}", max_list: int = MAX_LIST) -> str:
    """Any other READ endpoint, e.g. path="/api/ai-portfolio/edge" or
    "/api/admin/buy-funnel". params_json is a JSON object of query params.
    The token is read-only: only GETs are possible."""
    try:
        params = json.loads(params_json or "{}")
    except ValueError:
        return "Error: params_json must be a JSON object"
    return _safe(lambda: compact(api_get(path, params), max_list=max(1, min(max_list, 200))))


if __name__ == "__main__":
    mcp.run()
