"""Refresh-token rotation tests (aug-20 security item).

Every refresh token carries a server-recorded jti; /api/auth/refresh spends
it (rotation), replays outside the concurrency grace revoke the user's whole
token family, and legacy jti-less tokens ride a bounded migration tail.

Real-sqlite harness (rotation is a DB behavior) + direct handler invocation,
mirroring tests/test_experiment_gates.py.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from jose import jwt

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import init_db, SessionLocal, User, RefreshTokenRecord
from backend.auth import (
    issue_refresh_token, create_refresh_token, SECRET_KEY, ALGORITHM,
    REFRESH_REUSE_GRACE_SECONDS,
)
from backend.routes.auth import refresh_token, RefreshRequest

ROT_USER_ID = 99010  # canonical 99000+ test-id range (see conftest note)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(RefreshTokenRecord).delete()
    db.query(User).filter(User.id == ROT_USER_ID).delete()
    db.commit()
    u = User(id=ROT_USER_ID, email="rotation@test.local",
             hashed_password="", is_active=True, is_admin=False)
    db.add(u)
    db.commit()
    try:
        yield db
    finally:
        db.query(RefreshTokenRecord).delete()
        db.query(User).filter(User.id == ROT_USER_ID).delete()
        db.commit()
        db.close()


def _refresh(db, token):
    return asyncio.run(refresh_token(
        RefreshRequest(refresh_token=token), request=None, db=db))


def _rec(db, token):
    jti = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]).get("jti")
    return db.query(RefreshTokenRecord).filter_by(jti=jti).first()


class TestIssuance:
    def test_issue_records_jti_row(self, db_session):
        t = issue_refresh_token(db_session, ROT_USER_ID)
        db_session.commit()
        payload = jwt.decode(t, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["type"] == "refresh" and payload["jti"]
        rec = _rec(db_session, t)
        assert rec is not None and rec.user_id == ROT_USER_ID
        assert rec.revoked_at is None


class TestRotation:
    def test_refresh_spends_old_and_records_new(self, db_session):
        t1 = issue_refresh_token(db_session, ROT_USER_ID)
        db_session.commit()
        pair = _refresh(db_session, t1)
        old = _rec(db_session, t1)
        new = _rec(db_session, pair.refresh_token)
        assert old.revoked_at is not None
        assert old.replaced_by_jti == jwt.decode(
            pair.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])["jti"]
        assert new is not None and new.revoked_at is None
        # The new token itself refreshes fine (chain continues)
        _refresh(db_session, pair.refresh_token)

    def test_replay_outside_grace_revokes_family(self, db_session):
        t1 = issue_refresh_token(db_session, ROT_USER_ID)
        db_session.commit()
        pair = _refresh(db_session, t1)
        # Backdate the spend beyond the grace window, then replay t1
        old = _rec(db_session, t1)
        old.revoked_at = (datetime.now(timezone.utc).replace(tzinfo=None)
                          - timedelta(seconds=REFRESH_REUSE_GRACE_SECONDS + 30))
        db_session.commit()
        with pytest.raises(HTTPException) as exc:
            _refresh(db_session, t1)
        assert exc.value.status_code == 401
        # The whole family is dead — including the legitimately rotated token
        assert _rec(db_session, pair.refresh_token).revoked_at is not None
        with pytest.raises(HTTPException):
            _refresh(db_session, pair.refresh_token)

    def test_replay_inside_grace_gets_fresh_pair_without_alarm(self, db_session):
        """Two tabs race: the loser presents an just-spent jti and must get
        a working pair, and the family must survive."""
        t1 = issue_refresh_token(db_session, ROT_USER_ID)
        db_session.commit()
        pair_winner = _refresh(db_session, t1)
        pair_loser = _refresh(db_session, t1)  # immediate replay, inside grace
        assert pair_loser.refresh_token
        assert _rec(db_session, pair_winner.refresh_token).revoked_at is None

    def test_forged_jti_with_valid_signature_rejected(self, db_session):
        """Signed but never recorded server-side — not exchangeable."""
        expire = datetime.now(timezone.utc) + timedelta(days=7)
        forged = jwt.encode(
            {"sub": str(ROT_USER_ID), "exp": expire, "type": "refresh",
             "jti": "deadbeef" * 4},
            SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            _refresh(db_session, forged)
        assert exc.value.status_code == 401

    def test_jti_recorded_for_other_user_rejected(self, db_session):
        t_other = issue_refresh_token(db_session, ROT_USER_ID)
        db_session.commit()
        # Same jti claim, different sub — signature valid, record mismatched
        jti = jwt.decode(t_other, SECRET_KEY, algorithms=[ALGORITHM])["jti"]
        expire = datetime.now(timezone.utc) + timedelta(days=7)
        crossed = jwt.encode(
            {"sub": "1", "exp": expire, "type": "refresh", "jti": jti},
            SECRET_KEY, algorithm=ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            _refresh(db_session, crossed)
        assert exc.value.status_code == 401


class TestLegacyMigrationTail:
    def test_legacy_jtiless_token_still_exchanges_and_upgrades(self, db_session):
        legacy = create_refresh_token(data={"sub": str(ROT_USER_ID)})
        pair = _refresh(db_session, legacy)
        # The exchanged-in token is recorded (upgraded to rotation)
        assert _rec(db_session, pair.refresh_token) is not None
