"""Authentication routes: Google Sign-In, token refresh, user profile."""

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
from backend.rate_limiter import limiter
from jose import JWTError, jwt

router = APIRouter(prefix="/api/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    credential: str


@router.post("/google", response_model=Token)
@limiter.limit("10/minute")
async def google_login(req: GoogleLoginRequest, request: Request = None, db: Session = Depends(get_db)):
    """Authenticate via Google Sign-In and return JWT tokens."""
    # Verify the Google ID token
    google_payload = verify_google_token(req.credential)
    email = google_payload.get("email", "").lower()
    if not email:
        raise HTTPException(status_code=401, detail="No email in Google token")

    # Look up user by email (invite-only: must already exist)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=403,
            detail="No account found for this email. Contact the admin for an invite.",
        )
    if not user.is_active:
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


@router.get("/config")
async def get_auth_config():
    """Return auth configuration for frontend (public, no auth required)."""
    return {
        "google_client_id": GOOGLE_CLIENT_ID,
        "require_auth": GOOGLE_CLIENT_ID != "",
    }
