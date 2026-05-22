"""In-app notification routes — per-user, scoped strictly to current_user.id.

Companion to the ntfy webhook path: trade/stop/score-crash events dual-write
to the Notification table via email_utils.create_notification() so users can
read recent activity inside the app even when their phone push fails.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict

from backend.database import get_db, Notification
from backend.auth import get_current_active_user

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class NotificationOut(BaseModel):
    id: int
    kind: str
    title: str
    body: str
    priority: str
    tags: Optional[list] = None
    data: Optional[dict] = None
    read_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class NotificationList(BaseModel):
    items: list[NotificationOut]
    total: int
    unread_count: int


@router.get("", response_model=NotificationList)
async def list_notifications(
    unread_only: bool = Query(False),
    kind: Optional[str] = Query(None, max_length=64),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List the current user's notifications, newest first.

    With unread_only=true, returns only items where read_at IS NULL. Otherwise
    returns all (read or unread). With kind=<kind>, filters to that single
    notification kind (e.g. "trade", "stop_loss"). `total` is the count under
    the same filters; `unread_count` is always the user's overall unread count
    (drives the header bell badge regardless of which view is open).
    """
    base = db.query(Notification).filter(Notification.user_id == current_user.id)
    if unread_only:
        base = base.filter(Notification.read_at.is_(None))
    if kind:
        base = base.filter(Notification.kind == kind)

    total = base.with_entities(func.count(Notification.id)).scalar() or 0

    items = (base
             .order_by(Notification.created_at.desc())
             .offset(offset).limit(limit)
             .all())

    unread_count = (db.query(func.count(Notification.id))
                    .filter(Notification.user_id == current_user.id,
                            Notification.read_at.is_(None))
                    .scalar() or 0)

    return NotificationList(
        items=[NotificationOut.model_validate(n) for n in items],
        total=total,
        unread_count=unread_count,
    )


@router.get("/unread-count")
async def unread_count(
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Lightweight endpoint for the header bell badge — meant to be polled."""
    count = (db.query(func.count(Notification.id))
             .filter(Notification.user_id == current_user.id,
                     Notification.read_at.is_(None))
             .scalar() or 0)
    return {"count": count}


@router.post("/{notification_id}/read", response_model=NotificationOut)
async def mark_read(
    notification_id: int,
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    note = (db.query(Notification)
            .filter(Notification.id == notification_id,
                    Notification.user_id == current_user.id)
            .first())
    if not note:
        raise HTTPException(status_code=404, detail="Notification not found")
    if note.read_at is None:
        note.read_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(note)
    return NotificationOut.model_validate(note)


@router.post("/read-all")
async def mark_all_read(
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Mark every unread notification for the current user as read."""
    now = datetime.now(timezone.utc)
    updated = (db.query(Notification)
               .filter(Notification.user_id == current_user.id,
                       Notification.read_at.is_(None))
               .update({Notification.read_at: now}, synchronize_session=False))
    db.commit()
    return {"updated": updated}


class ThresholdPreviewOut(BaseModel):
    """Sample of recent per-stock alert scores so the Settings page can show
    "would surface X of N alerts at this threshold" without re-querying the
    backend on every slider tick. Frontend filters the array client-side.
    """
    lookback_days: int
    total_with_score: int
    scores: list[float]
    current_threshold: Optional[float] = None


@router.get("/threshold-preview", response_model=ThresholdPreviewOut)
async def threshold_preview(
    lookback_days: int = Query(7, ge=1, le=30),
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Return recent per-stock alert scores for the current user — supports
    the Settings page's score-threshold preview UI.

    Only alerts where `data['score']` is a finite number are returned. System-
    wide alerts (SPY gate, market turns, morning briefing) carry no score and
    are excluded here because the threshold gate ignores them anyway. Urgent-
    priority alerts (stop losses, circuit breakers) are also excluded because
    `_should_deliver` bypasses the threshold for them — including them would
    misleadingly inflate the "would pass" count.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = (db.query(Notification)
            .filter(Notification.user_id == current_user.id,
                    Notification.created_at >= cutoff,
                    Notification.data.isnot(None),
                    # Mirror _should_deliver's urgent bypass — urgent alerts
                    # always pass regardless of threshold, so don't pollute
                    # the preview pool.
                    Notification.priority != "urgent")
            .all())
    scores: list[float] = []
    for n in rows:
        if not isinstance(n.data, dict):
            continue
        raw = n.data.get("score")
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        # Guard against NaN / inf so the frontend filter math stays sane
        if val != val or val in (float("inf"), float("-inf")):
            continue
        scores.append(val)

    return ThresholdPreviewOut(
        lookback_days=lookback_days,
        total_with_score=len(scores),
        scores=scores,
        current_threshold=getattr(current_user, "score_alert_threshold", None),
    )


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    note = (db.query(Notification)
            .filter(Notification.id == notification_id,
                    Notification.user_id == current_user.id)
            .first())
    if not note:
        raise HTTPException(status_code=404, detail="Notification not found")
    db.delete(note)
    db.commit()
    return {"deleted": True}
