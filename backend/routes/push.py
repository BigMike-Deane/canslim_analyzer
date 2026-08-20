"""Web Push subscription routes — per-user, per-device.

Companion to the in-app notification surface. The Settings page calls
GET /vapid-public-key (public, no auth — the public key is meant to be
exposed), then asks the browser for a push subscription, then POSTs it to
/subscribe. When email_utils.create_notification or broadcast_notification
fires, send_web_push_to_user fans out to every subscription for that user
and prunes any that the browser vendor reports as gone (HTTP 410).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import get_db, PushSubscription
from backend.auth import get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/push", tags=["push"])


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class SubscribeRequest(BaseModel):
    endpoint: str
    keys: SubscriptionKeys


class SubscriptionOut(BaseModel):
    id: int
    user_agent: Optional[str] = None
    created_at: datetime
    last_used_at: Optional[datetime] = None


@router.get("/vapid-public-key")
async def get_vapid_public_key():
    """Public endpoint — the VAPID public key is meant to be embedded in the
    frontend so the browser can encrypt the subscription request. The private
    key stays on the server."""
    pub = os.environ.get("VAPID_PUBLIC_KEY", "").strip()
    if not pub:
        raise HTTPException(status_code=503, detail="Web Push not configured on this server")
    return {"public_key": pub}


@router.post("/subscribe")
async def subscribe(
    req: SubscribeRequest,
    request: Request,
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Register a Web Push subscription for the current user's device.

    If the same endpoint already exists (re-subscribe after permission churn),
    we update the keys/user_id rather than creating a duplicate row.
    """
    ua = request.headers.get("user-agent", "")[:255] or None

    existing = db.query(PushSubscription).filter(
        PushSubscription.endpoint == req.endpoint
    ).first()

    if existing:
        if existing.user_id != current_user.id:
            # A push endpoint URL is a per-device credential. Re-binding one
            # user's endpoint to another account would silently reroute the
            # original owner's device into the caller's notification stream
            # (misdelivery/DoS). Legit device-handoff recovers by the old
            # account unsubscribing (or an admin pruning the row), which
            # issues a fresh endpoint on the next subscribe.
            logger.warning(
                "Push subscribe rejected: endpoint owned by user %s, "
                "requested by user %s", existing.user_id, current_user.id)
            raise HTTPException(
                status_code=409,
                detail="This device's push subscription belongs to another account")
        existing.p256dh_key = req.keys.p256dh
        existing.auth_key = req.keys.auth
        existing.user_agent = ua
        existing.last_used_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        sub = existing
    else:
        sub = PushSubscription(
            user_id=current_user.id,
            endpoint=req.endpoint,
            p256dh_key=req.keys.p256dh,
            auth_key=req.keys.auth,
            user_agent=ua,
        )
        db.add(sub)
        db.commit()
        db.refresh(sub)

    return {"id": sub.id, "user_agent": sub.user_agent, "created_at": sub.created_at}


@router.get("/subscriptions", response_model=list[SubscriptionOut])
async def list_subscriptions(
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List the current user's push subscriptions for the device-management UI."""
    rows = (db.query(PushSubscription)
            .filter(PushSubscription.user_id == current_user.id)
            .order_by(PushSubscription.created_at.desc())
            .all())
    return [SubscriptionOut.model_validate({
        "id": r.id, "user_agent": r.user_agent,
        "created_at": r.created_at, "last_used_at": r.last_used_at,
    }) for r in rows]


@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
    subscription_id: int,
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    sub = (db.query(PushSubscription)
           .filter(PushSubscription.id == subscription_id,
                   PushSubscription.user_id == current_user.id)
           .first())
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    db.delete(sub)
    db.commit()
    return {"deleted": True}


@router.post("/test")
async def send_test_push(
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Fire a test push to every subscription on the current user's account.
    Returns count of devices that accepted the push."""
    from backend.email_utils import send_web_push_to_user
    sent = send_web_push_to_user(
        current_user.id,
        title="CANSLIM Test",
        body=f"Push notifications working for {current_user.email}",
        data={"url": "/notifications"},
    )
    return {"sent": sent}
