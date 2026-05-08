"""Cross-user authorization (IDOR) regression suite.

Companion to docs/security/authorization-audit.md. Every per-user resource type
in the catalog gets at least one parametrized "Bob tries to reach Alice" test
here. The point is to lock the audit's findings into CI so a future refactor
can't silently regress an ownership filter.

Threat model: alice (user_id=10) creates a resource. bob (user_id=11) presents
a valid auth credential and tries to read/mutate alice's row by id. Bob must
NEVER receive 200 with alice's data; expected outcomes are 404 (resource not
found within bob's scope) or 403 (forbidden). For collection endpoints, bob's
list must be empty even when alice's collection is populated.

Implementation note: rather than minting real JWTs and exercising the full
JOSE path (already covered by tests/test_auth.py), we override
`get_current_active_user` to dispatch on a synthetic header. Authorization
correctness is per-route enforcement, which is what we want to lock down.
"""

import os
from datetime import date, datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import (
    init_db, SessionLocal, User,
    PortfolioPosition, Watchlist,
    AIPortfolioPosition, AIPortfolioConfig, AIPortfolioTrade, AIPortfolioSnapshot,
    BacktestRun,
    FidelitySnapshot, FidelityPosition, FidelityTrade,
    Notification, PushSubscription,
)
from backend.auth import get_current_active_user, get_admin_user


# ── Per-user fixture (alice + bob) ───────────────────────────────────

ALICE_ID = 10
BOB_ID = 11


def _user_for_request(request: Request):
    """Auth dispatcher: maps `X-Test-User: alice|bob` to the corresponding row.

    No header → 401, matching REQUIRE_AUTH=true semantics. We deliberately do
    NOT fall through to id=1 (the dev-mode owner) — that's the foot-gun the
    audit warned about and a real test must avoid silent owner-impersonation.
    """
    who = request.headers.get("x-test-user", "").lower()
    if who not in ("alice", "bob"):
        raise HTTPException(status_code=401, detail="Test fixture: missing X-Test-User header")
    user_id = ALICE_ID if who == "alice" else BOB_ID
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Test fixture: user not in DB")
        # Detach so the closed session doesn't bite downstream handlers when
        # they read attributes (id/email/is_active/is_admin only — no relations).
        db.expunge(user)
        return user
    finally:
        db.close()


def _admin_for_request(request: Request):
    """Mirror of get_admin_user, but routed through our header dispatcher.
    Other test modules install a module-level override that returns an
    is_admin=True fixture user; without re-overriding here, /api/analyze/jobs
    would silently bypass the admin check during a full-suite run."""
    user = _user_for_request(request)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@pytest.fixture(scope="module", autouse=True)
def _setup_users_and_app():
    init_db()
    db = SessionLocal()
    try:
        for uid, email in [(ALICE_ID, "alice@test.com"), (BOB_ID, "bob@test.com")]:
            if not db.query(User).filter(User.id == uid).first():
                db.add(User(
                    id=uid, email=email, display_name=email.split("@")[0],
                    is_active=True, is_admin=False, hashed_password="",
                ))
        db.commit()
    finally:
        db.close()

    prev_user = app.dependency_overrides.get(get_current_active_user)
    prev_admin = app.dependency_overrides.get(get_admin_user)
    app.dependency_overrides[get_current_active_user] = _user_for_request
    app.dependency_overrides[get_admin_user] = _admin_for_request
    yield
    if prev_user is not None:
        app.dependency_overrides[get_current_active_user] = prev_user
    else:
        app.dependency_overrides.pop(get_current_active_user, None)
    if prev_admin is not None:
        app.dependency_overrides[get_admin_user] = prev_admin
    else:
        app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture
def client():
    return TestClient(app)


def _h(who: str) -> dict:
    return {"x-test-user": who}


# ── DB seeding helpers ───────────────────────────────────────────────

def _wipe_user_rows():
    """Per-test isolation. Strips both alice + bob owned rows from every model
    we exercise here, plus orphan PortfolioPosition rows (user_id IS NULL) so a
    failed sync-to-portfolio can't pollute the next test."""
    db = SessionLocal()
    try:
        for uid in (ALICE_ID, BOB_ID):
            db.query(PortfolioPosition).filter(PortfolioPosition.user_id == uid).delete()
            db.query(Watchlist).filter(Watchlist.user_id == uid).delete()
            db.query(AIPortfolioPosition).filter(AIPortfolioPosition.user_id == uid).delete()
            db.query(AIPortfolioConfig).filter(AIPortfolioConfig.user_id == uid).delete()
            db.query(AIPortfolioTrade).filter(AIPortfolioTrade.user_id == uid).delete()
            db.query(AIPortfolioSnapshot).filter(AIPortfolioSnapshot.user_id == uid).delete()
            db.query(BacktestRun).filter(BacktestRun.user_id == uid).delete()
            db.query(FidelityTrade).filter(FidelityTrade.user_id == uid).delete()
            db.query(Notification).filter(Notification.user_id == uid).delete()
            db.query(PushSubscription).filter(PushSubscription.user_id == uid).delete()
        # Snapshot/positions cascade via snapshot_id; clear by user_id directly.
        db.query(FidelitySnapshot).filter(FidelitySnapshot.user_id.in_([ALICE_ID, BOB_ID])).delete(synchronize_session=False)
        db.query(PortfolioPosition).filter(PortfolioPosition.user_id.is_(None)).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def _isolate_each_test():
    _wipe_user_rows()
    yield
    _wipe_user_rows()


# ── F-1: TestCrossUserAuthorization ───────────────────────────────────

class TestCrossUserAuthorization:
    """Bob must never reach a resource owned by alice."""

    def test_portfolio_list_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(PortfolioPosition(ticker="AAPL", shares=10, cost_basis=100, user_id=ALICE_ID))
            db.commit()
        finally:
            db.close()

        r_alice = client.get("/api/portfolio", headers=_h("alice"))
        r_bob = client.get("/api/portfolio", headers=_h("bob"))
        assert r_alice.status_code == 200
        assert any(p["ticker"] == "AAPL" for p in r_alice.json()["positions"])
        assert r_bob.status_code == 200
        assert r_bob.json()["positions"] == []

    def test_portfolio_delete_other_user_position(self, client):
        db = SessionLocal()
        try:
            pos = PortfolioPosition(ticker="MSFT", shares=5, cost_basis=200, user_id=ALICE_ID)
            db.add(pos)
            db.commit()
            pid = pos.id
        finally:
            db.close()

        r = client.delete(f"/api/portfolio/{pid}", headers=_h("bob"))
        assert r.status_code == 404, "Bob must not delete alice's position"

        # Confirm alice's row survives.
        db = SessionLocal()
        try:
            assert db.query(PortfolioPosition).filter_by(id=pid).first() is not None
        finally:
            db.close()

    def test_portfolio_update_other_user_position(self, client):
        db = SessionLocal()
        try:
            pos = PortfolioPosition(ticker="GOOG", shares=3, cost_basis=150, user_id=ALICE_ID)
            db.add(pos)
            db.commit()
            pid = pos.id
        finally:
            db.close()

        r = client.put(f"/api/portfolio/{pid}", json={"shares": 999}, headers=_h("bob"))
        assert r.status_code == 404
        db = SessionLocal()
        try:
            row = db.query(PortfolioPosition).filter_by(id=pid).first()
            assert row.shares == 3, "shares must remain unchanged after blocked PUT"
        finally:
            db.close()

    def test_watchlist_list_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(Watchlist(ticker="NVDA", user_id=ALICE_ID))
            db.commit()
        finally:
            db.close()

        r_alice = client.get("/api/watchlist", headers=_h("alice"))
        r_bob = client.get("/api/watchlist", headers=_h("bob"))
        assert r_alice.status_code == 200 and len(r_alice.json()["items"]) == 1
        assert r_bob.status_code == 200 and r_bob.json()["items"] == []

    def test_watchlist_delete_other_user_item(self, client):
        db = SessionLocal()
        try:
            w = Watchlist(ticker="TSLA", user_id=ALICE_ID)
            db.add(w)
            db.commit()
            wid = w.id
        finally:
            db.close()

        r = client.delete(f"/api/watchlist/{wid}", headers=_h("bob"))
        assert r.status_code == 404
        db = SessionLocal()
        try:
            assert db.query(Watchlist).filter_by(id=wid).first() is not None
        finally:
            db.close()

    def test_ai_portfolio_list_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(AIPortfolioConfig(user_id=ALICE_ID, starting_cash=25000))
            db.add(AIPortfolioPosition(user_id=ALICE_ID, ticker="META", shares=5, cost_basis=300))
            db.commit()
        finally:
            db.close()

        r_alice = client.get("/api/ai-portfolio", headers=_h("alice"))
        r_bob = client.get("/api/ai-portfolio", headers=_h("bob"))
        assert r_alice.status_code == 200
        assert any(p["ticker"] == "META" for p in r_alice.json()["positions"])
        assert r_bob.status_code == 200
        assert r_bob.json()["positions"] == []

    def test_ai_portfolio_trades_are_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(AIPortfolioTrade(
                user_id=ALICE_ID, ticker="AMD", action="BUY",
                shares=10, price=100, total_value=1000,
                executed_at=datetime.now(timezone.utc),
            ))
            db.commit()
        finally:
            db.close()

        r_bob = client.get("/api/ai-portfolio/trades", headers=_h("bob"))
        assert r_bob.status_code == 200
        assert r_bob.json() == []

    def test_ai_portfolio_history_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(AIPortfolioSnapshot(
                user_id=ALICE_ID, total_value=30000, cash=5000,
                positions_value=25000, positions_count=5,
                timestamp=datetime.now(timezone.utc),
            ))
            db.commit()
        finally:
            db.close()

        r_bob = client.get("/api/ai-portfolio/history", headers=_h("bob"))
        assert r_bob.status_code == 200
        assert r_bob.json() == []

    def test_trade_journal_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(AIPortfolioTrade(
                user_id=ALICE_ID, ticker="ORCL", action="BUY",
                shares=10, price=100, total_value=1000,
                executed_at=datetime.now(timezone.utc),
            ))
            db.commit()
        finally:
            db.close()

        r = client.get("/api/trade-journal", headers=_h("bob"))
        assert r.status_code == 200
        assert r.json()["entries"] == []

    def test_analytics_trades_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(AIPortfolioTrade(
                user_id=ALICE_ID, ticker="CRM", action="SELL",
                shares=10, price=120, total_value=1200,
                cost_basis=100, realized_gain=200,
                executed_at=datetime.now(timezone.utc),
            ))
            db.commit()
        finally:
            db.close()

        r = client.get("/api/analytics/trades", headers=_h("bob"))
        assert r.status_code == 200
        assert r.json()["summary"] == {}, "bob's analytics must reflect his own (empty) trades"

    def test_backtests_list_is_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(BacktestRun(
                user_id=ALICE_ID, name="alice-bt", status="completed",
                start_date=date(2024, 1, 1), end_date=date(2024, 6, 1),
                starting_cash=25000,
            ))
            db.commit()
        finally:
            db.close()

        r = client.get("/api/backtests", headers=_h("bob"))
        assert r.status_code == 200 and r.json() == []

    def test_backtest_detail_other_user(self, client):
        db = SessionLocal()
        try:
            bt = BacktestRun(
                user_id=ALICE_ID, name="alice-bt", status="completed",
                start_date=date(2024, 1, 1), end_date=date(2024, 6, 1),
                starting_cash=25000,
            )
            db.add(bt)
            db.commit()
            bt_id = bt.id
        finally:
            db.close()

        for path in (f"/api/backtests/{bt_id}",
                     f"/api/backtests/{bt_id}/status"):
            r = client.get(path, headers=_h("bob"))
            assert r.status_code == 404, f"bob must not GET {path}"

        r = client.delete(f"/api/backtests/{bt_id}", headers=_h("bob"))
        assert r.status_code == 404, "bob must not delete alice's backtest"

        r = client.post(f"/api/backtests/{bt_id}/cancel", headers=_h("bob"))
        assert r.status_code == 404
        r = client.post(f"/api/backtests/{bt_id}/rerun", headers=_h("bob"))
        assert r.status_code == 404

        # alice's row survives
        db = SessionLocal()
        try:
            assert db.query(BacktestRun).filter_by(id=bt_id).first() is not None
        finally:
            db.close()

    def test_fidelity_snapshots_are_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(FidelitySnapshot(
                user_id=ALICE_ID, snapshot_date=date.today(),
                cash_balance=1000, total_value=20000, positions_count=2,
            ))
            db.commit()
        finally:
            db.close()

        r = client.get("/api/fidelity/snapshots", headers=_h("bob"))
        assert r.status_code == 200 and r.json()["snapshots"] == []

        r = client.get("/api/fidelity/latest", headers=_h("bob"))
        assert r.status_code == 200 and r.json()["snapshot"] is None

    def test_fidelity_trades_are_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            db.add(FidelityTrade(
                user_id=ALICE_ID, run_date=date.today(), action="BUY",
                symbol="AAPL", quantity=10, price=150,
            ))
            db.commit()
        finally:
            db.close()

        r = client.get("/api/fidelity/trades", headers=_h("bob"))
        assert r.status_code == 200 and r.json()["trades"] == []

    def test_fidelity_sync_to_portfolio_is_tenant_isolated(self, client):
        """F-1 regression. Pre-fix, bob's sync would (a) overwrite or (b) mass-
        delete alice's PortfolioPosition rows. Post-fix, alice's manual
        portfolio is untouched regardless of what bob's Fidelity snapshot says."""
        db = SessionLocal()
        try:
            # Alice has a manual position.
            alice_pos = PortfolioPosition(
                ticker="AAPL", shares=100, cost_basis=150,
                current_price=180, current_value=18000,
                user_id=ALICE_ID,
            )
            db.add(alice_pos)

            # Bob has a Fidelity snapshot that ALSO contains AAPL but with very
            # different numbers, plus a ticker alice does not own. The bug
            # would have made bob's call overwrite alice's AAPL row AND delete
            # alice's row entirely (since bob's snapshot may not include every
            # ticker alice holds).
            snap = FidelitySnapshot(
                user_id=BOB_ID, snapshot_date=date.today(),
                cash_balance=500, total_value=50000, positions_count=1,
            )
            db.add(snap)
            db.flush()
            db.add(FidelityPosition(
                snapshot_id=snap.id, symbol="AAPL",
                description="Apple", quantity=999, last_price=10,
                current_value=9990, average_cost_basis=10,
                cost_basis_total=9990, total_gain_loss=0,
                total_gain_loss_pct=0, percent_of_account=100, position_type="Cash",
            ))
            db.commit()
            alice_pos_id = alice_pos.id
        finally:
            db.close()

        r = client.post("/api/fidelity/sync-to-portfolio", headers=_h("bob"))
        assert r.status_code == 200, r.text

        # Alice's row must be byte-identical to what she stored.
        db = SessionLocal()
        try:
            row = db.query(PortfolioPosition).filter_by(id=alice_pos_id).first()
            assert row is not None, "alice's manual AAPL was deleted by bob's sync (F-1 regression)"
            assert row.shares == 100, "alice's shares were overwritten (F-1 regression)"
            assert row.cost_basis == 150
            assert row.user_id == ALICE_ID

            # Bob's new row should exist and be tagged with bob's user_id.
            bob_rows = db.query(PortfolioPosition).filter_by(user_id=BOB_ID).all()
            assert len(bob_rows) == 1
            assert bob_rows[0].ticker == "AAPL"
            assert bob_rows[0].shares == 999
        finally:
            db.close()

    def test_notifications_are_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            n = Notification(
                user_id=ALICE_ID, kind="trade", title="alice-only",
                body="alice trade body", priority="default",
            )
            db.add(n)
            db.commit()
            nid = n.id
        finally:
            db.close()

        r = client.get("/api/notifications", headers=_h("bob"))
        assert r.status_code == 200 and r.json()["items"] == []

        r = client.get("/api/notifications/unread-count", headers=_h("bob"))
        assert r.status_code == 200 and r.json()["count"] == 0

        r = client.post(f"/api/notifications/{nid}/read", headers=_h("bob"))
        assert r.status_code == 404

        r = client.delete(f"/api/notifications/{nid}", headers=_h("bob"))
        assert r.status_code == 404

        db = SessionLocal()
        try:
            row = db.query(Notification).filter_by(id=nid).first()
            assert row is not None
            assert row.read_at is None
        finally:
            db.close()

    def test_push_subscriptions_are_scoped_per_user(self, client):
        db = SessionLocal()
        try:
            sub = PushSubscription(
                user_id=ALICE_ID,
                endpoint="https://example.invalid/alice-endpoint",
                p256dh_key="x", auth_key="y", user_agent="alice-agent",
            )
            db.add(sub)
            db.commit()
            sid = sub.id
        finally:
            db.close()

        r = client.get("/api/push/subscriptions", headers=_h("bob"))
        assert r.status_code == 200 and r.json() == []

        r = client.delete(f"/api/push/subscriptions/{sid}", headers=_h("bob"))
        assert r.status_code == 404

        db = SessionLocal()
        try:
            assert db.query(PushSubscription).filter_by(id=sid).first() is not None
        finally:
            db.close()


# ── F-2: privilege scope for /api/analyze/jobs/{job_id} ───────────────

class TestAnalyzeJobPrivilegeScope:
    """Producer (POST /api/analyze/scan) is admin-only; reader was active-only.
    Post-fix, reader is admin-only too."""

    def test_non_admin_cannot_read_analysis_job(self, client):
        # alice + bob are both is_admin=False per fixture.
        r = client.get("/api/analyze/jobs/1", headers=_h("alice"))
        assert r.status_code == 403, "non-admin must be rejected by /api/analyze/jobs/{id}"
