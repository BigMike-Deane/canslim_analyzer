"""
Read-only tooling token + the canslim-api MCP server (2026-09-10).

The token lets Claude's check-ins read the live app's computed answers
without minting a full-power JWT. What must hold, in the order it would hurt:

  * IT CAN ONLY READ -- every non-GET is 403, whatever the route.
  * ONLY THE LATEST ONE WORKS -- re-minting revokes the previous token;
    deleting the record revokes it outright; a DB miss fails closed.
  * ORDINARY LOGINS ARE UNTOUCHED -- tokens without a scope behave as before.
  * UNKNOWN SCOPES ARE REFUSED -- a future scope can't fall through to full
    access by accident.
"""

import importlib.util
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from jose import jwt

from backend import auth
from backend.database import SessionLocal, SystemSetting, User, init_db
from backend.main import app

UID = 99070
init_db()


@pytest.fixture(autouse=True)
def _user():
    db = SessionLocal()
    try:
        db.query(SystemSetting).filter(SystemSetting.key == auth.READONLY_JTI_KEY).delete()
        if not db.query(User).filter_by(id=UID).first():
            db.add(User(id=UID, email=f"u{UID}@acct.example.org", display_name="ro",
                        is_active=True, is_admin=True, hashed_password=""))
        db.commit()
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        db.query(SystemSetting).filter(SystemSetting.key == auth.READONLY_JTI_KEY).delete()
        db.commit()
    finally:
        db.close()


def _h(token):
    return {"Authorization": f"Bearer {token}"}


client = TestClient(app)


class TestReadOnlyScope:

    def test_reads_work(self):
        tok = auth.create_readonly_token(user_id=UID)
        r = client.get("/api/auth/me", headers=_h(tok))
        assert r.status_code == 200 and r.json()["id"] == UID

    @pytest.mark.parametrize("method,path", [
        ("delete", "/api/admin/program-milestones/999999"),
        ("post", "/api/admin/broker-mirror/seed"),
        ("post", "/api/ai-portfolio/run-cycle"),
    ])
    def test_writes_are_refused(self, method, path):
        tok = auth.create_readonly_token(user_id=UID)
        r = getattr(client, method)(path, headers=_h(tok))
        assert r.status_code == 403, (path, r.status_code, r.text[:200])

    def test_reminting_revokes_the_previous_token(self):
        old = auth.create_readonly_token(user_id=UID)
        new = auth.create_readonly_token(user_id=UID)
        assert client.get("/api/auth/me", headers=_h(old)).status_code == 401
        assert client.get("/api/auth/me", headers=_h(new)).status_code == 200

    def test_deleting_the_record_revokes(self):
        tok = auth.create_readonly_token(user_id=UID)
        db = SessionLocal()
        db.query(SystemSetting).filter(SystemSetting.key == auth.READONLY_JTI_KEY).delete()
        db.commit()
        db.close()
        assert client.get("/api/auth/me", headers=_h(tok)).status_code == 401

    def test_unknown_scope_is_refused(self):
        # Carries the CURRENT read token's jti, so only the scope check can
        # stop it -- a future scope must never fall through to access.
        read = auth.create_readonly_token(user_id=UID)
        jti = jwt.decode(read, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])["jti"]
        tok = auth.create_access_token({"sub": str(UID), "scope": "write", "jti": jti},
                                        expires_delta=timedelta(minutes=5))
        r = client.get("/api/auth/me", headers=_h(tok))
        assert r.status_code == 401 and "scope" in r.json()["detail"]

    def test_ordinary_login_tokens_are_untouched(self):
        tok = auth.create_access_token({"sub": str(UID)}, expires_delta=timedelta(minutes=5))
        assert client.get("/api/auth/me", headers=_h(tok)).status_code == 200
        r = client.delete("/api/admin/program-milestones/999999", headers=_h(tok))
        assert r.status_code != 403

    def test_the_token_says_what_it_is(self):
        claims = jwt.decode(auth.create_readonly_token(user_id=UID, days=30),
                            auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        assert claims["scope"] == "read" and claims["type"] == "access" and claims["jti"]

    def test_mint_script_prints_only_the_token(self):
        from backend import mint_readonly_token
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            assert mint_readonly_token.main(["--user-id", str(UID), "--days", "7"]) == 0
        lines = out.getvalue().strip().splitlines()
        assert len(lines) == 1 and lines[0].count(".") == 2       # a bare JWT
        assert "revoked" in err.getvalue()
        assert client.get("/api/auth/me", headers=_h(lines[0])).status_code == 200


# ---------------------------------------------------------------- the MCP server

@pytest.fixture
def srv(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "mcp", "api_server.py")
    spec = importlib.util.spec_from_file_location("canslim_api_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("CANSLIM_API_TOKEN", "tok123")
    return mod


class TestMcpServer:

    def _http(self, status=200, payload=None):
        calls = []

        def get(url, params=None, timeout=None, headers=None):
            calls.append((url, params, headers))
            r = MagicMock()
            r.status_code, r.text = status, "nope"
            r.json.return_value = payload or {}
            return r
        return MagicMock(get=get), calls

    def test_sends_the_token_and_only_reads_api_paths(self, srv):
        http, calls = self._http(payload={"ok": 1})
        assert srv.api_get("/api/admin/broker-mirror", {"limit": 5}, http=http) == {"ok": 1}
        url, params, headers = calls[0]
        assert url.endswith("/api/admin/broker-mirror") and params == {"limit": 5}
        assert headers["Authorization"] == "Bearer tok123"
        with pytest.raises(srv.ApiError):
            srv.api_get("/login", http=http)

    def test_401_says_how_to_fix_it(self, srv):
        http, _ = self._http(status=401)
        with pytest.raises(srv.ApiError, match="mint_readonly_token"):
            srv.api_get("/api/auth/me", http=http)

    def test_no_token_anywhere_says_how_to_fix_it(self, srv, monkeypatch, tmp_path):
        monkeypatch.delenv("CANSLIM_API_TOKEN")
        monkeypatch.setattr(srv, "TOKEN_FILE", str(tmp_path / "missing"))
        with pytest.raises(srv.ApiError, match="mint_readonly_token"):
            srv._token()

    def test_compact_cuts_and_says_so(self, srv):
        out = srv.compact({"xs": list(range(40)), "s": "a" * 1000, "f": 1.234567891})
        assert len(out["xs"]) == srv.MAX_LIST + 1 and out["xs"][-1] == "... 25 more"
        assert out["s"].endswith("...") and len(out["s"]) == srv.MAX_STR + 3
        assert out["f"] == 1.2346

    def test_program_shape_keeps_the_gate_and_drops_prose(self, srv):
        gates = {"program_clocks": {
            "go_live": {"n_met": 2, "n_total": 5, "all_met": False, "blocking": ["blended_edge"],
                        "note": "long prose " * 50,
                        "criteria": [{"key": "blended_edge", "met": False, "label": "x",
                                      "value": {"p": 0.29}, "target": {"p": 0.15}}]},
            "calendar": [{"label": "soon", "days_until": 10}, {"label": "far", "days_until": 200}],
            "stop_loss_recheck": {"n": 2, "target": 5, "label": "drop me"}},
            "arms": [{"name": "arm10", "description": "prose", "days_accrued": 30, "buys": 4,
                      "sells": 4, "gate_metrics": [{"label": "closed", "n": 4, "target": 5}]}]}
        s = srv.shape_program(gates)
        assert s["go_live"]["blocking"] == ["blended_edge"] and "note" not in s["go_live"]
        assert [c["label"] for c in s["calendar"]] == ["soon"]
        assert s["arms"][0] == {"name": "arm10", "days": 30, "buys": 4, "sells": 4,
                                "gates": [{"label": "closed", "n": 4, "target": 5}]}

    def test_broker_shape_separates_drift_from_design(self, srv):
        s = srv.shape_broker({"reconciliation": [
            {"ticker": "A", "match": True},
            {"ticker": "B", "match": False},
            {"ticker": "C", "match": False, "divergence": "resting_stop"}]})
        assert [r["ticker"] for r in s["reconciliation"]["mismatches"]] == ["B"]
        assert [r["ticker"] for r in s["reconciliation"]["designed_divergences"]] == ["C"]
