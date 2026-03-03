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
