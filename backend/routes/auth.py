"""Authentication routes: Google Sign-In, token refresh, user profile."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from backend.database import get_db, User
from backend.auth import (
    verify_google_token, create_access_token,
    create_refresh_token, Token, UserResponse,
    get_current_active_user, SECRET_KEY, ALGORITHM,
    GOOGLE_CLIENT_ID
)
from backend.audit_log import record_event
from backend.rate_limiter import limiter
from jose import JWTError, jwt

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _client_ip(request: Request | None) -> str:
    if request is None or request.client is None:
        return "unknown"
    return request.client.host or "unknown"


def _record_auth_fail(request: Request | None, reason: str) -> None:
    """PII-free audit: record only the failure reason, never the email/token."""
    try:
        record_event(
            "auth_fail",
            source_ip=_client_ip(request),
            route="/api/auth/google",
            detail=reason,
        )
    except Exception:
        # Audit-log must never mask the auth response.
        pass


class GoogleLoginRequest(BaseModel):
    credential: str


@router.post("/google", response_model=Token)
@limiter.limit("10/minute")
async def google_login(req: GoogleLoginRequest, request: Request = None, db: Session = Depends(get_db)):
    """Authenticate via Google Sign-In and return JWT tokens."""
    # Verify the Google ID token
    try:
        google_payload = verify_google_token(req.credential)
    except HTTPException:
        _record_auth_fail(request, "invalid_token")
        raise
    email = google_payload.get("email", "").lower()
    if not email:
        _record_auth_fail(request, "missing_email")
        raise HTTPException(status_code=401, detail="No email in Google token")

    # Look up user by email (invite-only: must already exist)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        _record_auth_fail(request, "unknown_user")
        raise HTTPException(
            status_code=403,
            detail="No account found for this email. Contact the admin for an invite.",
        )
    if not user.is_active:
        _record_auth_fail(request, "account_disabled")
        raise HTTPException(status_code=403, detail="Account is disabled")

    # Update display name from Google if not set
    google_name = google_payload.get("name")
    if google_name and not user.display_name:
        user.display_name = google_name
        db.commit()

    return Token(
        access_token=create_access_token(data={"sub": str(user.id)}),
        refresh_token=create_refresh_token(data={"sub": str(user.id)}),
    )


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=Token)
@limiter.limit("20/minute")
async def refresh_token(req: RefreshRequest, request: Request = None, db: Session = Depends(get_db)):
    """Exchange a refresh token for new access + refresh tokens."""
    try:
        payload = jwt.decode(req.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return Token(
        access_token=create_access_token(data={"sub": str(user.id)}),
        refresh_token=create_refresh_token(data={"sub": str(user.id)}),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_active_user)):
    """Get current user profile."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        is_admin=current_user.is_admin,
        is_active=current_user.is_active,
        webhook_url=current_user.webhook_url,
        mute_kinds=current_user.mute_kinds or [],
        quiet_hours_start=current_user.quiet_hours_start,
        quiet_hours_end=current_user.quiet_hours_end,
        score_alert_threshold=current_user.score_alert_threshold,
    )


class WebhookUpdateRequest(BaseModel):
    webhook_url: str  # Empty string to disable; otherwise must be http(s) URL


@router.patch("/me/webhook", response_model=UserResponse)
async def update_my_webhook(
    req: WebhookUpdateRequest,
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update the authenticated user's notification webhook URL.

    Pass an empty string to silence notifications for this user.
    """
    url = (req.webhook_url or "").strip()
    if url and not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="webhook_url must start with http:// or https://")

    current_user.webhook_url = url or None
    db.commit()
    db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        is_admin=current_user.is_admin,
        is_active=current_user.is_active,
        webhook_url=current_user.webhook_url,
        mute_kinds=current_user.mute_kinds or [],
        quiet_hours_start=current_user.quiet_hours_start,
        quiet_hours_end=current_user.quiet_hours_end,
        score_alert_threshold=current_user.score_alert_threshold,
    )


@router.post("/me/webhook/test")
async def test_my_webhook(current_user=Depends(get_current_active_user)):
    """Send a test notification to the authenticated user's webhook URL.
    Returns 400 if no URL configured, 200 with sent=true/false otherwise."""
    from backend.email_utils import send_webhook_notification
    url = (current_user.webhook_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="No webhook URL configured")
    sent = send_webhook_notification(
        title="CANSLIM Test",
        message=f"Test notification for {current_user.email}\nIf you see this, routing works.",
        priority="default",
        tags=["white_check_mark"],
        url=url,
    )
    return {"sent": sent, "url_configured": True}


class NotificationPrefsRequest(BaseModel):
    # Pass any subset; unset fields are left untouched. Pass [] to mute_kinds
    # to clear the mute list; pass null to quiet_hours_* to disable.
    mute_kinds: Optional[list[str]] = None
    quiet_hours_start: Optional[int] = None
    quiet_hours_end: Optional[int] = None
    clear_quiet_hours: bool = False
    # Min CANSLIM score for score-bearing alerts. None = no change. Pass 0
    # alongside clear_score_alert_threshold=true to disable the gate.
    score_alert_threshold: Optional[int] = None
    clear_score_alert_threshold: bool = False


# Whitelist of valid notification kinds for mute_kinds. Matches frontend
# KIND_META in pages/Notifications.jsx. Reject unknown kinds so a typo
# doesn't silently mute everything.
_VALID_KINDS = {
    "trade", "stop_loss", "score_crash", "breakout", "coiled_spring",
    "risk_alert", "spy_gate_change", "market_turn",
    "bear_base_update", "bear_market_report", "watchlist",
}


@router.patch("/me/notification-prefs", response_model=UserResponse)
async def update_notification_prefs(
    req: NotificationPrefsRequest,
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update the authenticated user's per-kind mute list and quiet hours.

    Quiet hours are interpreted in America/Chicago local time (single-owner
    app convention). Passing clear_quiet_hours=true sets both bounds to null.
    Urgent-priority notifications (stop losses, circuit breakers) always
    bypass these filters by design — see _should_deliver in email_utils.
    """
    if req.mute_kinds is not None:
        bad = [k for k in req.mute_kinds if k not in _VALID_KINDS]
        if bad:
            raise HTTPException(status_code=400, detail=f"Unknown kinds: {bad}")
        current_user.mute_kinds = list(req.mute_kinds)

    if req.clear_quiet_hours:
        current_user.quiet_hours_start = None
        current_user.quiet_hours_end = None
    else:
        if req.quiet_hours_start is not None:
            if not 0 <= req.quiet_hours_start <= 23:
                raise HTTPException(status_code=400, detail="quiet_hours_start must be 0-23")
            current_user.quiet_hours_start = req.quiet_hours_start
        if req.quiet_hours_end is not None:
            if not 0 <= req.quiet_hours_end <= 23:
                raise HTTPException(status_code=400, detail="quiet_hours_end must be 0-23")
            current_user.quiet_hours_end = req.quiet_hours_end

    if req.clear_score_alert_threshold:
        current_user.score_alert_threshold = None
    elif req.score_alert_threshold is not None:
        if not 0 <= req.score_alert_threshold <= 100:
            raise HTTPException(status_code=400, detail="score_alert_threshold must be 0-100")
        current_user.score_alert_threshold = req.score_alert_threshold

    db.commit()
    db.refresh(current_user)
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        is_admin=current_user.is_admin,
        is_active=current_user.is_active,
        webhook_url=current_user.webhook_url,
        mute_kinds=current_user.mute_kinds or [],
        quiet_hours_start=current_user.quiet_hours_start,
        quiet_hours_end=current_user.quiet_hours_end,
        score_alert_threshold=current_user.score_alert_threshold,
    )


@router.get("/config")
async def get_auth_config():
    """Return auth configuration for frontend (public, no auth required)."""
    return {
        "google_client_id": GOOGLE_CLIENT_ID,
        "require_auth": GOOGLE_CLIENT_ID != "",
    }
