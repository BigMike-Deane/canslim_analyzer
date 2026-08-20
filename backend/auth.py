"""
Authentication module: Google Sign-In verification, JWT tokens, user dependencies.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db, User

import os
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
_DEV_SECRET_DEFAULT = "dev-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "false").lower() == "true"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# `... or DEFAULT` (not os.environ.get's default arg): docker-compose.yml wires
# `JWT_SECRET_KEY=${JWT_SECRET_KEY}`, which interpolates to an EMPTY STRING when
# the host/.env doesn't set it (.env.template ships it blank). os.environ.get
# would return "" — a present-but-empty key — and HS256 would happily sign and
# verify with it, so anyone could forge {"sub":"1"} and take over the owner.
# Treat blank as unset, and in an auth-required deploy refuse to run on the
# empty/dev key rather than silently accepting a full auth bypass.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or _DEV_SECRET_DEFAULT
if REQUIRE_AUTH and (not os.environ.get("JWT_SECRET_KEY") or SECRET_KEY == _DEV_SECRET_DEFAULT):
    raise RuntimeError(
        "REQUIRE_AUTH=true but JWT_SECRET_KEY is empty or the dev default. "
        "Set a strong JWT_SECRET_KEY in the environment before enabling auth."
    )


# --- Google Sign-In verification ---
def verify_google_token(id_token_str: str) -> dict:
    """
    Verify a Google ID token and return the payload.

    Returns dict with 'email', 'name', 'picture', etc.
    Raises HTTPException on invalid/expired tokens.
    """
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests

    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="Google Sign-In not configured (GOOGLE_CLIENT_ID missing)",
        )

    try:
        payload = google_id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
        if payload.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
            raise HTTPException(status_code=401, detail="Invalid token issuer")
        # The email claim is our account identity, so it must be VERIFIED.
        # Consumer Google accounts always are; some Workspace/federated
        # identities can carry email_verified=false, and an unverified email
        # matching a pre-created invite row would be an account takeover.
        if not payload.get("email_verified"):
            raise HTTPException(status_code=401, detail="Google account email not verified")
        return payload
    except ValueError as e:
        logger.warning(f"Google token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid Google token")


# --- JWT tokens ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """LEGACY refresh token — no jti, not server-recorded, not revocable.

    Kept only so tokens issued before the rotation deploy keep working for
    their remaining lifetime (the refresh endpoint accepts jti-less tokens
    until natural expiry). All NEW issuance goes through
    issue_refresh_token(); do not call this from login/refresh paths.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Concurrency grace: two tabs refreshing at once both present the same jti;
# the loser must not be treated as a thief. A revoked jti re-presented
# within this window gets a fresh pair without alarm.
REFRESH_REUSE_GRACE_SECONDS = 60


def issue_refresh_token(db, user_id: int) -> str:
    """Create a SINGLE-USE refresh token and record its jti for rotation.

    The DB row is what makes the token revocable; a syntactically valid
    refresh JWT whose jti has no row is rejected. Caller owns the commit
    boundary — this only db.add()s.
    """
    import secrets
    from backend.database import RefreshTokenRecord

    jti = secrets.token_hex(16)
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    token = jwt.encode(
        {"sub": str(user_id), "exp": expire, "type": "refresh", "jti": jti},
        SECRET_KEY, algorithm=ALGORITHM,
    )
    db.add(RefreshTokenRecord(
        jti=jti, user_id=user_id,
        issued_at=now.replace(tzinfo=None),
        expires_at=expire.replace(tzinfo=None),
    ))
    return token


# --- Pydantic schemas ---
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: str
    display_name: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    display_name: Optional[str]
    is_admin: bool
    is_active: bool
    webhook_url: Optional[str] = None
    mute_kinds: Optional[list[str]] = None
    quiet_hours_start: Optional[int] = None
    quiet_hours_end: Optional[int] = None
    score_alert_threshold: Optional[int] = None


# --- FastAPI Dependencies ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/google", auto_error=False)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get current authenticated user from JWT token.

    When REQUIRE_AUTH=false (default/dev mode):
      - No token -> returns User with id=1 (the owner)
      - Valid token -> returns the user
    When REQUIRE_AUTH=true:
      - No token -> raises 401
      - Valid token -> returns the user
    """

    if token is None:
        if not REQUIRE_AUTH:
            user = db.query(User).filter(User.id == 1).first()
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # int() is INSIDE the guard: a validly-signed token with a non-numeric
        # `sub` would otherwise raise ValueError → unhandled 500 instead of 401.
        user_id = int(user_id)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def get_current_active_user(current_user=Depends(get_current_user)):
    """Require an active user (raises if None)."""
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return current_user


def get_admin_user(current_user=Depends(get_current_active_user)):
    """Require admin privileges."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
