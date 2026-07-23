"""
Email utilities for CANSLIM Analyzer
Handles watchlist alerts and other email notifications
"""

import html
import ipaddress
import smtplib
import os
import logging
import socket
import requests
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_safe_webhook_url(url: str) -> bool:
    """Reject user-supplied webhook URLs that resolve to private / loopback /
    link-local / metadata ranges — a blind-SSRF guard. Any authenticated user
    can set their own webhook_url, and the server POSTs to it from the scheduler
    and the test endpoint; without this a user could point it at the cloud
    metadata endpoint (169.254.169.254), internal Redis, or the app's own
    internal port. DNS is resolved here so a public hostname can't A-record into
    a private range. Callers must also POST with allow_redirects=False so a
    302 can't hop into the private range after this check passes.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname
    if not host:
        return False
    try:
        addrs = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except (socket.gaierror, UnicodeError):
        return False  # unresolvable → treat as unsafe
    if not addrs:
        return False
    # RFC 6598 shared address space (100.64.0.0/10) is NOT covered by
    # is_private, but it's exactly where this deploy's internal services live
    # (the VPS Tailscale IP is 100.104.189.36) — block it explicitly.
    cgnat = ipaddress.ip_network("100.64.0.0/10")
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                or ip in cgnat):
            return False
    return True

# Load .env file if it exists
def _load_env():
    """Load environment variables from .env file if it exists"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists() and env_path.is_file():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

_load_env()

# Email configuration
GMAIL_ADDRESS = os.environ.get('CANSLIM_EMAIL', 'your-email@gmail.com')
GMAIL_APP_PASSWORD = os.environ.get('CANSLIM_APP_PASSWORD', 'your-app-password')
RECIPIENT_EMAIL = os.environ.get('CANSLIM_RECIPIENT', GMAIL_ADDRESS)


def send_email(subject: str, html_content: str, text_content: str,
               recipient: str = None) -> bool:
    """Send email via Gmail SMTP

    Args:
        subject: Email subject line
        html_content: HTML version of the email body
        text_content: Plain text version of the email body
        recipient: Override RECIPIENT_EMAIL for one-off sends (e.g. admin
            test-snapshot endpoint). When None, falls back to the module-level
            default — preserves the contract for all pre-existing callers.

    Returns:
        True if email sent successfully, False otherwise
    """
    to_addr = recipient or RECIPIENT_EMAIL
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = to_addr

    part1 = MIMEText(text_content, 'plain')
    part2 = MIMEText(html_content, 'html')
    msg.attach(part1)
    msg.attach(part2)

    # Port 465 (SMTPS) is blocked from this VPS's network egress; use port 587
    # (STARTTLS) instead. Discovered May 7 2026 — the original 465 path was
    # silently failing, which is why no watchlist-alert emails had been
    # delivered. Verified port 587 reaches smtp.gmail.com from the VPS host
    # AND from inside the Docker container (the previous "Network is unreachable"
    # error was specifically port 465).
    try:
        import ssl
        context = ssl.create_default_context()
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as server:
            server.starttls(context=context)
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, to_addr, msg.as_string())
        logger.info(f"Email sent successfully to {to_addr}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def send_watchlist_alert_email(item, stock, reasons: list,
                               recipient: str = None) -> bool:
    """Send email when watchlist alert triggers

    Args:
        item: Watchlist model instance
        stock: Stock model instance
        reasons: List of reason strings
        recipient: The watchlist OWNER's email. Watchlist rows are per-user;
            without this, every user's alert (including private notes) landed
            in the global RECIPIENT_EMAIL inbox — a cross-user leak.

    Returns:
        True if email sent successfully, False otherwise
    """
    subject = f"CANSLIM Alert: {item.ticker}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .stock-info {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
            .reason {{ background: #e7f5ff; padding: 10px; border-left: 4px solid #228be6; margin: 5px 0; }}
            .metric {{ display: inline-block; margin-right: 20px; }}
            .metric-value {{ font-size: 1.2em; font-weight: bold; }}
            .footer {{ color: #666; font-size: 0.9em; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2 style="margin: 0;">Watchlist Alert Triggered</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">{item.ticker} has met your alert criteria</p>
        </div>

        <div class="stock-info">
            <h3 style="margin-top: 0;">{html.escape(item.ticker)} - {html.escape(stock.name) if stock.name else 'Unknown'}</h3>
            <div class="metric">
                <div style="color: #666;">Current Price</div>
                <div class="metric-value">${(stock.current_price or 0):.2f}</div>
            </div>
            <div class="metric">
                <div style="color: #666;">CANSLIM Score</div>
                <div class="metric-value">{(stock.canslim_score or 0):.0f}</div>
            </div>
        </div>

        <h3>Alert Reasons:</h3>
        {''.join(f'<div class="reason">{html.escape(r)}</div>' for r in reasons)}

        {f'<p><strong>Your Notes:</strong> {html.escape(item.notes)}</p>' if item.notes else ''}

        <div class="footer">
            <p>Generated by CANSLIM Analyzer</p>
        </div>
    </body>
    </html>
    """

    text_content = f"""Watchlist Alert: {item.ticker}

{item.ticker} - {stock.name if stock.name else 'Unknown'}

Reasons:
{chr(10).join(f'- {r}' for r in reasons)}

Current Price: ${(stock.current_price or 0):.2f}
CANSLIM Score: {(stock.canslim_score or 0):.0f}
{f'Your Notes: {item.notes}' if item.notes else ''}
"""

    return send_email(subject, html_content, text_content, recipient=recipient)


# Webhook configuration
WEBHOOK_URL = os.environ.get('CANSLIM_WEBHOOK_URL', '')


def send_webhook_notification(title: str, message: str, priority: str = "default",
                              data: dict = None, tags: list = None,
                              click: str = None, markdown: bool = False,
                              url: str = None, kind: str = None) -> bool:
    """Send push notification via webhook (e.g., ntfy.sh, Pushover, or custom).

    Args:
        title: Notification title
        message: Notification body text
        priority: Priority level ("urgent", "high", "default", "low", "min")
        data: Optional extra data to include in payload
        tags: Optional list of ntfy emoji tags (e.g. ["moneybag", "chart_with_upwards_trend"])
        click: Optional URL to open when notification is tapped
        markdown: If True, enable markdown formatting in ntfy
        url: Override webhook URL. Falls back to global CANSLIM_WEBHOOK_URL env var.
             Pass an empty string explicitly to silence (per-user routing uses this).

    Returns:
        True if notification sent successfully, False otherwise
    """
    target_url = url if url is not None else WEBHOOK_URL
    if not target_url:
        logger.debug("Webhook URL not configured, skipping notification")
        return False

    # Owner-mute gate for the GLOBAL legacy webhook (2026-07-22 owner
    # report): the global CANSLIM_WEBHOOK_URL is de-facto the owner's phone,
    # but this path predates per-user prefs and bypassed mute_kinds — a
    # muted kind still pinged via ntfy even though the per-user broadcast
    # path correctly suppressed it. When a `kind` is supplied AND we're
    # firing the global URL (not an explicit per-user `url` override), run
    # the same _should_deliver gate the broadcast path uses for user 1.
    # Urgent priority bypasses by design; kind=None (ops/backup messages)
    # keeps legacy behavior.
    if kind and url is None and priority != "urgent":
        try:
            from backend.database import SessionLocal, User
            _db = SessionLocal()
            try:
                owner = _db.query(User).filter(User.id == 1).first()
            finally:
                _db.close()
            if owner is not None and not _should_deliver(owner, kind, priority, data):
                logger.info(f"Global webhook suppressed by owner prefs (kind={kind}): {title}")
                return False
        except Exception:
            pass  # pref lookup failure must never block a notification

    payload = {
        "title": title,
        "message": message,
        "priority": priority,
    }
    if data:
        payload["data"] = data

    # Map word priorities to ntfy numeric levels
    _ntfy_priority = {
        "urgent": "5", "high": "4", "default": "3", "low": "2", "min": "1",
    }

    # Handle different webhook formats
    try:
        # Check for ntfy.sh style (topic-based URL)
        if "ntfy" in target_url:
            headers = {
                "Title": title,
                "Priority": _ntfy_priority.get(priority, "3"),
            }
            if tags:
                headers["Tags"] = ",".join(tags)
            if click:
                headers["Click"] = click
            if markdown:
                headers["Markdown"] = "yes"
            response = requests.post(target_url, data=message.encode('utf-8'),
                                     headers=headers, timeout=10, allow_redirects=False)
        else:
            # Standard JSON webhook (Pushover, Discord, custom)
            response = requests.post(target_url, json=payload, timeout=10, allow_redirects=False)

        if response.status_code in (200, 201, 204):
            logger.info(f"Webhook notification sent: {title}")
            return True
        else:
            logger.warning(f"Webhook returned status {response.status_code}: {response.text[:200]}")
            return False

    except requests.exceptions.Timeout:
        logger.error("Webhook request timed out")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Webhook request failed: {e}")
        return False


def get_user_webhook_url(user_id: int) -> str:
    """Look up a user's per-user webhook URL. Returns empty string if not set
    or on lookup error — caller treats empty string as 'skip notification'."""
    if not user_id:
        return ""
    try:
        from backend.database import SessionLocal, User
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return (user.webhook_url or "").strip() if user else ""
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to load webhook_url for user {user_id}: {e}")
        return ""


def _should_deliver(user, kind: str, priority: str, data: dict = None) -> bool:
    """Apply the user's mute_kinds + quiet_hours + score-threshold filters
    to OUTBOUND delivery.

    Urgent items always pass — stop losses + circuit breakers are non-mutable
    by design. Returns True if the kind is permitted at the current time.

    If the data payload includes a 'score' field AND the user has set
    score_alert_threshold, alerts with score < threshold are suppressed.
    Alerts without a score field are unaffected — the threshold is a
    per-stock filter, not a global mute.
    """
    if priority == "urgent":
        return True
    if not user:
        return True  # be permissive on missing user row

    from backend.database import coerce_json_list
    mute = coerce_json_list(getattr(user, "mute_kinds", None))
    if kind in mute:
        return False

    qs, qe = user.quiet_hours_start, user.quiet_hours_end
    if qs is not None and qe is not None and qs != qe:
        # Quiet hours stored as America/Chicago local hour. If ZoneInfo isn't
        # available (older runtimes, stripped images), fail OPEN — skip the
        # gate rather than apply a wrong window. UTC-hour fallback would
        # silently shift quiet hours by 5-6 hours and suppress alerts during
        # the wrong part of the day.
        try:
            from datetime import datetime as _dt
            from zoneinfo import ZoneInfo
            hour = _dt.now(ZoneInfo("America/Chicago")).hour
            if qs < qe:
                in_quiet = qs <= hour < qe
            else:
                in_quiet = hour >= qs or hour < qe  # window crosses midnight
            if in_quiet:
                return False
        except Exception:
            pass  # tz unavailable / clock error — skip the gate (fail-open)

    threshold = getattr(user, "score_alert_threshold", None)
    if threshold is not None and data and "score" in data:
        try:
            if float(data["score"]) < float(threshold):
                return False
        except (TypeError, ValueError):
            pass  # malformed score field — don't suppress
    return True


def create_notification(user_id: int, kind: str, title: str, body: str,
                        priority: str = "default", tags: list = None,
                        data: dict = None) -> bool:
    """Persist an in-app notification for a user AND fan out a Web Push to
    every device they've registered. Fail-soft on every step — a DB error
    or push failure never blocks the trade pipeline or the parallel ntfy
    POST. Returns True on DB insert success.

    In-app DB rows are ALWAYS written so the bell + Notifications page show
    full history; only outbound push (and ntfy) are gated by the user's
    mute_kinds / quiet_hours preferences. Urgent items bypass the gate.
    """
    if not user_id:
        return False
    inserted = False
    user = None
    try:
        from backend.database import SessionLocal, Notification, User
        db = SessionLocal()
        try:
            note = Notification(
                user_id=user_id,
                kind=kind,
                title=title,
                body=body or "",
                priority=priority,
                tags=tags,
                data=data,
            )
            db.add(note)
            db.commit()
            inserted = True
            user = db.query(User).filter(User.id == user_id).first()
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to create notification for user {user_id} ({kind}): {e}")

    if not _should_deliver(user, kind, priority, data):
        return inserted

    # Push delivery is independent — fire even if DB insert failed; that way
    # users still get phone alerts during a degraded-DB window.
    try:
        urgency = "high" if priority in ("high", "urgent") else "normal"
        push_data = {"kind": kind, **(data or {})}
        if "url" not in push_data:
            ticker = (data or {}).get("ticker")
            push_data["url"] = f"/stock/{ticker}" if ticker else "/notifications"
        send_web_push_to_user(user_id, title=title, body=body, data=push_data, urgency=urgency)
    except Exception as e:
        logger.warning(f"Web push for user {user_id} ({kind}) failed: {e}")
    return inserted


def _send_web_push_to_subscriptions(subscriptions, title: str, body: str,
                                    data: dict = None, urgency: str = "normal") -> int:
    """Internal: deliver a Web Push payload to a set of PushSubscription rows.

    Returns count of successful deliveries. Subscriptions that the browser
    vendor reports as gone (HTTP 410, 404) are auto-pruned — phones that
    cleared site data, uninstalled the PWA, or revoked permission.

    Failures other than 410/404 are logged and counted as "not sent" but
    don't raise — the in-app DB row is the canonical record.
    """
    if not subscriptions:
        return 0
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        logger.warning("pywebpush not installed — skipping web push delivery")
        return 0

    private_key = os.environ.get("VAPID_PRIVATE_KEY", "").strip()
    subject = os.environ.get("VAPID_SUBJECT", "mailto:admin@canslim.local").strip()
    if not private_key:
        return 0

    import json as _json
    from backend.database import SessionLocal, PushSubscription

    payload = _json.dumps({
        "title": title, "body": body or "",
        "data": data or {},
    })
    sent = 0
    expired_ids = []
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {"p256dh": sub.p256dh_key, "auth": sub.auth_key},
                },
                data=payload,
                vapid_private_key=private_key,
                vapid_claims={"sub": subject},
                headers={"Urgency": urgency},
            )
            sent += 1
        except WebPushException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (404, 410):
                expired_ids.append(sub.id)
            else:
                logger.warning(f"WebPush failed for sub={sub.id}: {e}")
        except Exception as e:
            logger.warning(f"WebPush error for sub={sub.id}: {e}")

    if expired_ids:
        db = SessionLocal()
        try:
            db.query(PushSubscription).filter(
                PushSubscription.id.in_(expired_ids)
            ).delete(synchronize_session=False)
            db.commit()
            logger.info(f"Pruned {len(expired_ids)} expired push subscriptions")
        except Exception as e:
            logger.warning(f"Failed to prune expired subscriptions: {e}")
        finally:
            db.close()
    return sent


def send_web_push_to_user(user_id: int, title: str, body: str,
                          data: dict = None, urgency: str = "normal") -> int:
    """Deliver a Web Push to every device the user has registered."""
    if not user_id:
        return 0
    try:
        from backend.database import SessionLocal, PushSubscription
        db = SessionLocal()
        try:
            subs = db.query(PushSubscription).filter(
                PushSubscription.user_id == user_id
            ).all()
            return _send_web_push_to_subscriptions(subs, title, body, data, urgency)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to load push subscriptions for user {user_id}: {e}")
        return 0


def send_web_push_broadcast(title: str, body: str, data: dict = None,
                            urgency: str = "normal") -> int:
    """Deliver a Web Push to every device of every active user."""
    try:
        from backend.database import SessionLocal, PushSubscription, User
        db = SessionLocal()
        try:
            subs = (db.query(PushSubscription)
                    .join(User, User.id == PushSubscription.user_id)
                    .filter(User.is_active == True)
                    .all())
            return _send_web_push_to_subscriptions(subs, title, body, data, urgency)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to broadcast web push: {e}")
        return 0


def broadcast_notification(kind: str, title: str, body: str,
                           priority: str = "default", tags: list = None,
                           data: dict = None) -> int:
    """Persist an in-app notification for every active user AND fan out a
    Web Push to every active user's devices.

    Used for system-wide events (breakouts, market regime changes, etc.)
    that aren't tied to any single user's portfolio. Fail-soft at every
    step. Returns the number of DB rows inserted.
    """
    inserted = 0
    try:
        from backend.database import SessionLocal, Notification, User
        db = SessionLocal()
        try:
            user_ids = [uid for (uid,) in
                        db.query(User.id).filter(User.is_active == True).all()]
            if user_ids:
                now = datetime.now(timezone.utc)
                db.bulk_save_objects([
                    Notification(
                        user_id=uid, kind=kind, title=title, body=body or "",
                        priority=priority, tags=tags, data=data, created_at=now,
                    ) for uid in user_ids
                ])
                db.commit()
                inserted = len(user_ids)
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Failed to broadcast notification ({kind}): {e}")

    # Independent push fan-out — fire even if the DB insert failed.
    # Per-user, gated by _should_deliver: the blanket send_web_push_broadcast
    # bypassed mute_kinds/quiet_hours/score-threshold, making every mute
    # checkbox for broadcast-only kinds (breakout, coiled_spring, ...) a
    # placebo. In-app DB rows above stay unconditional (documented contract);
    # only outbound push honors the user's filters.
    try:
        urgency = "high" if priority in ("high", "urgent") else "normal"
        push_data = {"kind": kind, **(data or {})}
        if "url" not in push_data:
            ticker = (data or {}).get("ticker")
            push_data["url"] = f"/stock/{ticker}" if ticker else "/notifications"

        from backend.database import SessionLocal, User
        db = SessionLocal()
        try:
            # Column tuples (not ORM instances): usable after close, and
            # attribute access matches what _should_deliver reads.
            users = db.query(
                User.id, User.mute_kinds,
                User.quiet_hours_start, User.quiet_hours_end,
                User.score_alert_threshold,
            ).filter(User.is_active == True).all()
        finally:
            db.close()

        for user in users:
            if _should_deliver(user, kind, priority, data):
                send_web_push_to_user(user.id, title=title, body=body,
                                      data=push_data, urgency=urgency)
    except Exception as e:
        logger.warning(f"Web push broadcast ({kind}) failed: {e}")
    return inserted


def send_coiled_spring_alert_webhook(stock, cs_result: dict) -> bool:
    """Send webhook notification for Coiled Spring alerts

    Args:
        stock: Stock model instance
        cs_result: Coiled Spring calculation result dict

    Returns:
        True if notification sent successfully, False otherwise
    """
    days_to_earnings = cs_result.get('days_to_earnings', 'N/A')
    base_type = cs_result.get('base_type', 'unknown')
    weeks = cs_result.get('weeks_in_base', 0)
    beat_streak = cs_result.get('beat_streak', 0)

    title = f"🌀 Coiled Spring: {stock.ticker}"
    message = (
        f"{stock.ticker} - {getattr(stock, 'name', 'Unknown')}\n"
        f"Price: ${(stock.current_price or 0):.2f}\n"
        f"Score: {(stock.canslim_score or 0):.0f}\n"
        f"Earnings in {days_to_earnings} days\n"
        f"Base: {base_type} ({weeks}w), {beat_streak} beat streak"
    )

    data = {
        "ticker": stock.ticker,
        "price": stock.current_price,
        "score": stock.canslim_score,
        "days_to_earnings": days_to_earnings,
        "base_type": base_type,
        "weeks_in_base": weeks,
        "beat_streak": beat_streak,
    }

    tags = ["cyclone", "chart_with_upwards_trend"]
    broadcast_notification(kind="coiled_spring", title=title, body=message,
                           priority="high", tags=tags, data=data)
    return send_webhook_notification(title, message, priority="high", data=data, tags=tags,
                                     kind="coiled_spring")


def send_trade_webhook(ticker: str, action: str, shares: float, price: float,
                       reason: str, gain_pct: float = None,
                       user_id: int = None) -> bool:
    """Send webhook notification when AI trader executes a trade.

    Args:
        ticker: Stock ticker
        action: BUY or SELL
        shares: Number of shares
        price: Execution price
        reason: Trade reason
        gain_pct: Realized gain % (for sells)
        user_id: Owner of this trade. Notification routes to that user's
            webhook_url only. If user has no URL set, no notification fires.
            None = legacy behavior (use global CANSLIM_WEBHOOK_URL).

    Returns:
        True if sent successfully
    """
    title = f"{'BUY' if action == 'BUY' else 'SELL'}: {ticker}"
    message = f"{ticker}: {shares:.2f} shares @ ${price:.2f}\n{reason}"
    if gain_pct is not None:
        message = f"{ticker}: {shares:.2f} shares @ ${price:.2f} ({gain_pct:+.1f}%)\n{reason}"

    if action == "BUY":
        tags = ["moneybag", "chart_with_upwards_trend"]
    elif gain_pct is not None and gain_pct >= 0:
        tags = ["money_with_wings", "white_check_mark"]
    else:
        tags = ["money_with_wings", "chart_with_downwards_trend"]

    create_notification(
        user_id, kind="trade", title=title, body=message, priority="high", tags=tags,
        data={"ticker": ticker, "action": action, "shares": shares, "price": price,
              "gain_pct": gain_pct, "reason": reason},
    )

    url = get_user_webhook_url(user_id) if user_id is not None else None
    return send_webhook_notification(title, message, priority="high", tags=tags, url=url,
                                     kind="trade")


def send_stop_loss_webhook(ticker: str, shares: float, price: float,
                           stop_type: str, loss_pct: float,
                           user_id: int = None, is_partial: bool = False,
                           shares_kept: float = None) -> bool:
    """Send urgent webhook when stop loss triggers.

    Args:
        ticker: Stock ticker
        shares: Shares sold
        price: Sell price
        stop_type: STOP LOSS or TRAILING STOP
        loss_pct: Loss percentage
        user_id: Owner of this position. See send_trade_webhook for routing.
        is_partial: True for a partial trailing stop — the position is NOT
            closed. Makes the notification say so, instead of looking like a
            full liquidation (N1: live IESC partial read as a full exit
            2026-06-10). Default False → byte-identical to the legacy message.
        shares_kept: Remaining shares after the partial sell, surfaced when known.

    Returns:
        True if sent successfully
    """
    tags = ["rotating_light", "chart_with_downwards_trend"]
    if is_partial:
        title = f"PARTIAL {stop_type}: {ticker}"
        kept = f" — {shares_kept:.2f} shares kept" if shares_kept is not None else ""
        message = (f"{ticker}: sold {shares:.2f} shares @ ${price:.2f} ({loss_pct:+.1f}%)\n"
                   f"PARTIAL exit — position is STILL OPEN{kept}.")
    else:
        title = f"{stop_type}: {ticker}"
        message = f"{ticker}: {shares:.2f} shares @ ${price:.2f} ({loss_pct:+.1f}%)\nAutomatic stop triggered"

    create_notification(
        user_id, kind="stop_loss", title=title, body=message, priority="urgent", tags=tags,
        data={"ticker": ticker, "shares": shares, "price": price,
              "stop_type": stop_type, "loss_pct": loss_pct,
              "is_partial": is_partial, "shares_kept": shares_kept},
    )

    url = get_user_webhook_url(user_id) if user_id is not None else None
    return send_webhook_notification(title, message, priority="urgent",
                                     tags=tags, url=url)


def send_risk_alert_webhook(alert_type: str, details: str) -> bool:
    """Send webhook alert for portfolio risk conditions.

    Args:
        alert_type: Type of risk (heat, sector_concentration, position_size)
        details: Description of the risk condition

    Returns:
        True if sent successfully
    """
    titles = {
        "heat": "Portfolio Heat Warning",
        "sector_concentration": "Sector Concentration Alert",
        "position_size": "Position Size Alert",
        "drawdown": "Drawdown Warning",
    }
    tag_map = {
        "heat": ["fire", "warning"],
        "sector_concentration": ["warning", "pie"],
        "position_size": ["warning", "heavy_dollar_sign"],
        "drawdown": ["rotating_light", "chart_with_downwards_trend"],
    }
    title = titles.get(alert_type, f"Risk Alert: {alert_type}")
    tags = tag_map.get(alert_type, ["warning"])
    broadcast_notification(kind="risk_alert", title=title, body=details,
                           priority="high", tags=tags,
                           data={"alert_type": alert_type})
    return send_webhook_notification(title, details, priority="high", tags=tags,
                                     kind="risk_alert")


def send_scan_completion_push(stocks_scanned: int, total: int, scan_time: float,
                              buys: list = None, sells: list = None) -> bool:
    """Send push notification when a scan cycle completes.

    Args:
        stocks_scanned: Number of stocks successfully scanned
        total: Total stocks attempted
        scan_time: Scan duration in seconds
        buys: List of buy trade dicts (optional)
        sells: List of sell trade dicts (optional)

    Returns:
        True if sent successfully
    """
    buys = buys or []
    sells = sells or []

    title = f"Scan Complete: {stocks_scanned}/{total}"
    lines = [f"Scanned in {scan_time:.0f}s"]

    if buys or sells:
        trade_parts = []
        if buys:
            trade_parts.append(f"{len(buys)} buy{'s' if len(buys) != 1 else ''}")
        if sells:
            trade_parts.append(f"{len(sells)} sell{'s' if len(sells) != 1 else ''}")
        lines.append(f"Trades: {', '.join(trade_parts)}")

    message = "\n".join(lines)
    tags = ["mag"]
    if buys or sells:
        tags.append("bell")

    return send_webhook_notification(title, message, priority="low", tags=tags)


def send_spy_gate_change_push(new_state: str, spy_price: float, spy_ma50: float) -> bool:
    """Send push notification when SPY gate flips bullish/bearish.

    Args:
        new_state: "bullish" or "bearish"
        spy_price: Current SPY price
        spy_ma50: SPY 50-day moving average

    Returns:
        True if sent successfully
    """
    is_bullish = new_state == "bullish"
    emoji = "green_circle" if is_bullish else "red_circle"
    direction = "ABOVE" if is_bullish else "BELOW"
    action = "Buys ENABLED" if is_bullish else "Buys BLOCKED"
    diff = spy_price - spy_ma50
    diff_pct = (diff / spy_ma50) * 100 if spy_ma50 else 0

    title = f"SPY Gate: {new_state.upper()}"
    message = (
        f"SPY ${spy_price:.2f} crossed {direction} 50MA ${spy_ma50:.2f} ({diff_pct:+.2f}%)\n"
        f"{action} — nostate_optimized binary gate flipped"
    )
    tags = [emoji, "rotating_light"]
    broadcast_notification(kind="spy_gate_change", title=title, body=message,
                           priority="high", tags=tags,
                           data={"new_state": new_state, "spy_price": spy_price,
                                 "spy_ma50": spy_ma50, "diff_pct": diff_pct})
    return send_webhook_notification(title, message, priority="high", tags=tags,
                                     kind="spy_gate_change")


def send_score_crash_warning_push(ticker: str, purchase_score: float, current_score: float,
                                  gain_pct: float, consecutive_low: int,
                                  consecutive_required: int,
                                  user_id: int = None) -> bool:
    """Send early warning when a held position's score is dropping toward auto-sell.

    Args:
        ticker: Stock ticker
        purchase_score: Score when position was bought
        current_score: Current CANSLIM score
        gain_pct: Current gain/loss percentage
        consecutive_low: How many consecutive low scans so far
        consecutive_required: How many are needed to trigger auto-sell
        user_id: Owner of the position. See send_trade_webhook for routing.

    Returns:
        True if sent successfully
    """
    remaining = consecutive_required - consecutive_low
    title = f"Score Warning: {ticker}"
    message = (
        f"{ticker}: {purchase_score:.0f} -> {current_score:.0f} (position {gain_pct:+.1f}%)\n"
        f"{consecutive_low}/{consecutive_required} low scans - "
        f"{remaining} more before auto-sell"
    )
    tags = ["warning", "chart_with_downwards_trend"]

    create_notification(
        user_id, kind="score_crash", title=title, body=message, priority="high", tags=tags,
        data={"ticker": ticker, "purchase_score": purchase_score, "current_score": current_score,
              "gain_pct": gain_pct, "consecutive_low": consecutive_low,
              "consecutive_required": consecutive_required},
    )

    url = get_user_webhook_url(user_id) if user_id is not None else None
    return send_webhook_notification(title, message, priority="high",
                                     tags=tags, url=url, kind="score_crash")


def send_bear_base_update_push(total: int, top_candidates: list) -> bool:
    """Send push notification with bear market watchlist update.

    Args:
        total: Total candidates on the watchlist
        top_candidates: List of top candidate dicts with ticker, readiness_score

    Returns:
        True if sent successfully
    """
    title = f"Bear Base Watchlist: {total} candidates"
    top_str = ", ".join(
        f"{c['ticker']}({c.get('readiness_score', 0):.0f})"
        for c in top_candidates[:5]
    )
    message = f"Top: {top_str}\nStocks building quality bases during bear market"
    tags = ["bear", "mag"]
    broadcast_notification(kind="bear_base_update", title=title, body=message,
                           priority="default", tags=tags,
                           data={"total": total, "top": top_candidates[:5]})
    return send_webhook_notification(title, message, priority="default", tags=tags, kind="bear_base_update")


def send_market_turn_ready_push(candidates: list, spy_price: float, spy_ma50: float) -> bool:
    """Send push notification when SPY crosses back above 50MA with ready-to-buy list.

    Args:
        candidates: Top bear base candidates ready to buy
        spy_price: Current SPY price
        spy_ma50: SPY 50-day MA

    Returns:
        True if sent successfully
    """
    title = f"MARKET TURN: {len(candidates)} stocks ready"
    lines = [f"SPY ${spy_price:.2f} crossed ABOVE 50MA ${spy_ma50:.2f}"]
    for c in candidates[:5]:
        lines.append(f"{c['ticker']}: score {c.get('readiness_score', 0):.0f}, "
                     f"{c.get('base_type', 'base')} {c.get('weeks_in_base', 0)}w")
    message = "\n".join(lines)
    tags = ["green_circle", "rocket", "moneybag"]
    broadcast_notification(kind="market_turn", title=title, body=message,
                           priority="urgent", tags=tags,
                           data={"spy_price": spy_price, "spy_ma50": spy_ma50,
                                 "candidate_count": len(candidates),
                                 "top": candidates[:5]})
    return send_webhook_notification(title, message, priority="high",
                                     tags=tags, markdown=True, kind="market_turn")


def send_bear_market_report_push(report_data: dict) -> bool:
    """Send weekly bear market report as push notification.

    Args:
        report_data: Dict with bases_forming, rotation summary, buy_list

    Returns:
        True if sent successfully
    """
    bases = report_data.get("bases_forming", 0)
    improving = report_data.get("improving_groups", [])
    top_ready = report_data.get("top_ready", [])

    title = f"Weekly Bear Report: {bases} bases forming"
    lines = []
    if top_ready:
        top_str = ", ".join(f"{c['ticker']}({c.get('readiness_score', 0):.0f})" for c in top_ready[:3])
        lines.append(f"Ready: {top_str}")
    if improving:
        imp_str = ", ".join(g.get("industry", "?")[:20] for g in improving[:3])
        lines.append(f"Improving: {imp_str}")
    message = "\n".join(lines) if lines else "No notable changes this week"
    tags = ["bear", "clipboard"]
    broadcast_notification(kind="bear_market_report", title=title, body=message,
                           priority="default", tags=tags,
                           data={"bases_forming": bases})
    return send_webhook_notification(title, message, priority="default", tags=tags, kind="bear_market_report")
