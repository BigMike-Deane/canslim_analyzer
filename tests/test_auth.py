"""Tests for authentication module (Google Sign-In)."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from backend.auth import (
    create_access_token, create_refresh_token,
    verify_google_token, ALGORITHM, GOOGLE_CLIENT_ID
)
from jose import jwt


class TestJWTTokens:
    def test_create_access_token(self):
        token = create_access_token(data={"sub": "1"})
        assert isinstance(token, str)
        payload = jwt.decode(token, "dev-secret-key-change-in-production", algorithms=[ALGORITHM])
        assert payload["sub"] == "1"
        assert payload["type"] == "access"

    def test_create_refresh_token(self):
        token = create_refresh_token(data={"sub": "1"})
        assert isinstance(token, str)
        payload = jwt.decode(token, "dev-secret-key-change-in-production", algorithms=[ALGORITHM])
        assert payload["sub"] == "1"
        assert payload["type"] == "refresh"

    def test_access_token_has_expiry(self):
        token = create_access_token(data={"sub": "1"})
        payload = jwt.decode(token, "dev-secret-key-change-in-production", algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_refresh_token_has_expiry(self):
        token = create_refresh_token(data={"sub": "1"})
        payload = jwt.decode(token, "dev-secret-key-change-in-production", algorithms=[ALGORITHM])
        assert "exp" in payload


class TestGoogleTokenVerification:
    def test_verify_google_token_no_client_id(self):
        """Should raise 500 when GOOGLE_CLIENT_ID is not set."""
        from fastapi import HTTPException
        with patch("backend.auth.GOOGLE_CLIENT_ID", ""):
            with pytest.raises(HTTPException) as exc_info:
                verify_google_token("fake-token")
            assert exc_info.value.status_code == 500
            assert "not configured" in exc_info.value.detail

    @patch("backend.auth.GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    def test_verify_google_token_invalid_token(self):
        """Should raise 401 on invalid Google token."""
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            verify_google_token("obviously-invalid-token")
        assert exc_info.value.status_code == 401

    @patch("backend.auth.GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_verify_google_token_success(self, mock_verify):
        """Should return payload on valid Google token."""
        mock_verify.return_value = {
            "email": "user@gmail.com",
            "name": "Test User",
            "iss": "accounts.google.com",
            "email_verified": True,
        }
        result = verify_google_token("valid-google-token")
        assert result["email"] == "user@gmail.com"
        assert result["name"] == "Test User"
        mock_verify.assert_called_once()


class TestUserModel:
    def test_user_model_exists(self):
        from backend.database import User
        assert User.__tablename__ == "users"

    def test_user_model_columns(self):
        from backend.database import User
        column_names = {c.name for c in User.__table__.columns}
        assert "id" in column_names
        assert "email" in column_names
        assert "hashed_password" in column_names
        assert "display_name" in column_names
        assert "is_active" in column_names
        assert "is_admin" in column_names
        assert "created_at" in column_names

    def test_user_hashed_password_nullable(self):
        """Google Sign-In mode: hashed_password should be nullable."""
        from backend.database import User
        col = User.__table__.columns["hashed_password"]
        assert col.nullable is True


class TestAuthSchemas:
    def test_token_schema(self):
        from backend.auth import Token
        t = Token(access_token="abc", refresh_token="def")
        assert t.token_type == "bearer"
        assert t.access_token == "abc"

    def test_user_response_schema(self):
        from backend.auth import UserResponse
        u = UserResponse(id=1, email="test@test.com", display_name="Test", is_admin=True, is_active=True)
        assert u.id == 1
        assert u.email == "test@test.com"

    def test_user_create_schema_no_password(self):
        """UserCreate should only require email (no password for Google Sign-In)."""
        from backend.auth import UserCreate
        u = UserCreate(email="test@test.com")
        assert u.email == "test@test.com"
        assert u.display_name is None

    def test_user_create_with_display_name(self):
        from backend.auth import UserCreate
        u = UserCreate(email="test@test.com", display_name="Tester")
        assert u.display_name == "Tester"


# ============================================================================
# Coverage close-out (post-b348b60): get_current_user, get_current_active_user,
# get_admin_user, plus the Google-issuer check at line 53.
#
# These are FastAPI dependencies but they are plain Python functions — calling
# them directly with a MagicMock db session is simpler than spinning up a
# TestClient and exercises every branch of the JWT decode + user lookup logic.
# ============================================================================


def _make_mock_db(user=None):
    """Build a MagicMock standing in for a SQLAlchemy Session.
    `db.query(User).filter(...).first()` returns the supplied user."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = user
    return db


def _make_mock_user(user_id=1, is_active=True, is_admin=False):
    """A plain object with the User attributes get_current_*_user reads.
    Using a MagicMock-ish stand-in (rather than a real DB row) keeps the
    test tree free of SQLAlchemy session lifecycle for these unit tests."""
    user = MagicMock()
    user.id = user_id
    user.is_active = is_active
    user.is_admin = is_admin
    return user


class TestGoogleIssuerCheck:
    """Closes line 53 — the issuer-validation branch in verify_google_token."""

    @patch("backend.auth.GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_rejects_payload_with_wrong_issuer(self, mock_verify):
        from fastapi import HTTPException
        # Issuer that isn't in the accepted set → 401
        mock_verify.return_value = {
            "email": "user@gmail.com", "name": "U",
            "iss": "evil.example.com",
        }
        with pytest.raises(HTTPException) as exc_info:
            verify_google_token("any-token")
        assert exc_info.value.status_code == 401
        assert "issuer" in exc_info.value.detail.lower()

    @patch("backend.auth.GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_accepts_https_accounts_google_com(self, mock_verify):
        # The other accepted issuer form
        mock_verify.return_value = {
            "email": "user@gmail.com",
            "iss": "https://accounts.google.com",
            "email_verified": True,
        }
        result = verify_google_token("any-token")
        assert result["email"] == "user@gmail.com"

    @patch("backend.auth.GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_rejects_unverified_email(self, mock_verify):
        """email is the account identity — a Google-signed token whose email
        is UNVERIFIED (possible on Workspace/federated identities) must not
        authenticate, or it could claim a pre-created invite row."""
        from fastapi import HTTPException
        for claim in ({"email_verified": False}, {}):  # explicit false + absent
            mock_verify.return_value = {
                "email": "user@gmail.com",
                "iss": "accounts.google.com",
                **claim,
            }
            with pytest.raises(HTTPException) as exc_info:
                verify_google_token("any-token")
            assert exc_info.value.status_code == 401
            assert "verified" in exc_info.value.detail.lower()


class TestGetCurrentUser:
    """Covers backend/auth.py:115-142 — the full JWT-decode + user-lookup body."""

    def test_returns_owner_when_token_missing_and_auth_disabled(self):
        # REQUIRE_AUTH defaults to False in dev; no token → return User id=1
        from backend.auth import get_current_user
        owner = _make_mock_user(user_id=1, is_active=True)
        db = _make_mock_db(user=owner)
        with patch("backend.auth.REQUIRE_AUTH", False):
            result = get_current_user(token=None, db=db)
        assert result is owner

    def test_raises_401_when_token_missing_and_auth_required(self):
        from fastapi import HTTPException
        from backend.auth import get_current_user
        with patch("backend.auth.REQUIRE_AUTH", True):
            with pytest.raises(HTTPException) as exc:
                get_current_user(token=None, db=_make_mock_db())
        assert exc.value.status_code == 401
        assert exc.value.headers.get("WWW-Authenticate") == "Bearer"

    def test_returns_user_for_valid_access_token(self):
        from backend.auth import get_current_user, create_access_token
        user = _make_mock_user(user_id=42, is_active=True)
        db = _make_mock_db(user=user)
        token = create_access_token(data={"sub": "42"})
        result = get_current_user(token=token, db=db)
        assert result is user

    def test_rejects_refresh_token_used_as_access(self):
        from fastapi import HTTPException
        from backend.auth import get_current_user, create_refresh_token
        token = create_refresh_token(data={"sub": "1"})
        with pytest.raises(HTTPException) as exc:
            get_current_user(token=token, db=_make_mock_db())
        assert exc.value.status_code == 401
        assert "Invalid token type" in exc.value.detail

    def test_rejects_token_with_no_subject_claim(self):
        from fastapi import HTTPException
        from backend.auth import get_current_user, SECRET_KEY, ALGORITHM
        # Hand-craft an access token with no 'sub' claim
        from datetime import datetime, timezone, timedelta
        payload = {
            "type": "access",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            get_current_user(token=token, db=_make_mock_db())
        assert exc.value.status_code == 401
        assert exc.value.detail == "Invalid token"

    def test_rejects_malformed_token(self):
        # Garbage string → JWTError → 401 "Invalid or expired token"
        from fastapi import HTTPException
        from backend.auth import get_current_user
        with pytest.raises(HTTPException) as exc:
            get_current_user(token="not.a.jwt", db=_make_mock_db())
        assert exc.value.status_code == 401
        assert "Invalid or expired token" in exc.value.detail
        assert exc.value.headers.get("WWW-Authenticate") == "Bearer"

    def test_non_numeric_sub_yields_401_not_500(self):
        # A validly-signed token with a non-numeric `sub` must 401, not raise
        # ValueError from int() → unhandled 500 (2026-07-03 audit).
        from fastapi import HTTPException
        from backend.auth import get_current_user, create_access_token
        token = create_access_token(data={"sub": "not-a-number"})
        with pytest.raises(HTTPException) as exc:
            get_current_user(token=token, db=_make_mock_db())
        assert exc.value.status_code == 401
        assert "Invalid or expired token" in exc.value.detail

    def test_rejects_expired_token(self):
        # Sign a token with an exp in the past → jose raises JWTError
        from fastapi import HTTPException
        from backend.auth import get_current_user, SECRET_KEY, ALGORITHM
        from datetime import datetime, timezone, timedelta
        payload = {
            "sub": "1",
            "type": "access",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            get_current_user(token=token, db=_make_mock_db())
        assert exc.value.status_code == 401

    def test_rejects_when_user_not_found_in_db(self):
        from fastapi import HTTPException
        from backend.auth import get_current_user, create_access_token
        token = create_access_token(data={"sub": "999"})
        # DB returns None for the user lookup → 401
        db = _make_mock_db(user=None)
        with pytest.raises(HTTPException) as exc:
            get_current_user(token=token, db=db)
        assert exc.value.status_code == 401
        assert "User not found or inactive" in exc.value.detail


class TestGetCurrentActiveUser:
    """Covers backend/auth.py:147-151."""

    def test_passes_through_active_user(self):
        from backend.auth import get_current_active_user
        user = _make_mock_user(user_id=1, is_active=True)
        assert get_current_active_user(current_user=user) is user

    def test_raises_401_when_current_user_is_none(self):
        from fastapi import HTTPException
        from backend.auth import get_current_active_user
        with pytest.raises(HTTPException) as exc:
            get_current_active_user(current_user=None)
        assert exc.value.status_code == 401
        assert exc.value.detail == "Not authenticated"

    def test_raises_403_when_user_disabled(self):
        from fastapi import HTTPException
        from backend.auth import get_current_active_user
        disabled = _make_mock_user(user_id=1, is_active=False)
        with pytest.raises(HTTPException) as exc:
            get_current_active_user(current_user=disabled)
        assert exc.value.status_code == 403
        assert "disabled" in exc.value.detail.lower()


class TestGetAdminUser:
    """Covers backend/auth.py:156-158."""

    def test_passes_through_admin(self):
        from backend.auth import get_admin_user
        admin = _make_mock_user(user_id=1, is_active=True, is_admin=True)
        assert get_admin_user(current_user=admin) is admin

    def test_raises_403_for_non_admin(self):
        from fastapi import HTTPException
        from backend.auth import get_admin_user
        regular = _make_mock_user(user_id=2, is_active=True, is_admin=False)
        with pytest.raises(HTTPException) as exc:
            get_admin_user(current_user=regular)
        assert exc.value.status_code == 403
        assert "Admin access required" in exc.value.detail
