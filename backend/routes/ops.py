"""Ops alert intake (2026-09-10): Alertmanager -> owner web push.

The Prometheus alert rules in monitoring/alerts.yml existed since Sep-9, but
nothing delivered them -- there was no Alertmanager, so a firing alert went
nowhere. Alertmanager now POSTs every alert group here, and this turns it
into the same owner push every other ops alarm uses (send_ops_alert).
Critical alerts ALSO go out by email straight from Alertmanager, because
the alert that matters most -- "canslim-analyzer is gone" -- is exactly the
one this endpoint cannot deliver.

Auth: a shared bearer token (OPS_WEBHOOK_TOKEN in the VPS .env, handed to
Alertmanager as a file). The route is reachable through the public Caddy
proxy, so without a token anyone could push text to the owner's phone.
Unset token -> 503: fail closed, never open.
"""

import hmac
import logging
import os

from fastapi import APIRouter, Body, Header, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ops", tags=["ops"])

MAX_ALERTS_LISTED = 5


def _check_token(authorization: str):
    token = os.environ.get("OPS_WEBHOOK_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="OPS_WEBHOOK_TOKEN not configured")
    if not hmac.compare_digest((authorization or "").encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="bad token")


def format_alert_group(payload: dict) -> dict:
    """Alertmanager webhook payload (v4) -> one notification. Pure."""
    status = payload.get("status", "firing")
    alerts = payload.get("alerts") or []
    labels = payload.get("commonLabels") or (alerts[0].get("labels") if alerts else {}) or {}
    name = labels.get("alertname", "alert")
    severity = labels.get("severity", "warning")
    resolved = status == "resolved"

    lines = []
    for a in alerts[:MAX_ALERTS_LISTED]:
        ann = a.get("annotations") or {}
        text = ann.get("summary") or name
        if ann.get("description") and not resolved:
            text += f" -- {ann['description']}"
        lines.append(text)
    if len(alerts) > MAX_ALERTS_LISTED:
        lines.append(f"(+{len(alerts) - MAX_ALERTS_LISTED} more)")

    if resolved:
        title, priority = f"Resolved: {name}", "low"
    else:
        title = f"{'CRITICAL' if severity == 'critical' else 'Warning'}: {name}"
        # Critical must reach the owner at 3 AM (urgent bypasses mute and
        # quiet hours); a warning can wait for morning.
        priority = "urgent" if severity == "critical" else "default"
    return {
        "title": title,
        "message": "\n".join(lines) or name,
        "priority": priority,
        "tags": ["alertmanager", severity, status],
        "data": {"alertname": name, "severity": severity, "status": status,
                 "n_alerts": len(alerts)},
    }


@router.post("/alerts")
def alertmanager_webhook(payload: dict = Body(...), authorization: str = Header(None)):
    """Alertmanager webhook receiver. Sync `def`: send_ops_alert writes the
    DB and fans out web pushes -- blocking work for the threadpool."""
    _check_token(authorization)
    note = format_alert_group(payload)
    from backend.email_utils import send_ops_alert
    ok = send_ops_alert(note["title"], note["message"], priority=note["priority"],
                        tags=note["tags"], data=note["data"])
    logger.info(f"Alertmanager -> push: {note['title']} (delivered={ok})")
    return {"ok": bool(ok)}
