"""Coverage close-out tests for backend/routes/admin.py (90% → 96%+).

Targets the 47 remaining uncovered lines as of 2026-05-10:
  POST /api/admin/users (create_user)         — lines 48-63
  PATCH /api/admin/users/{user_id} (update_user) — lines 86-106
  Helper exception/edge branches               — 144-145, 161, 302,
    307-308, 329-330, 358, 364-365, 768-769, 828-829
  _resolve_ab_window post_window_days=None      — line 609
  cap_delta_diagnostics_endpoint Query coercion — 1195-1205
  compute_cap_delta_diagnostics defensive       — line 1095

The user-management routes use TestClient + auth bypass (matches the
pattern in tests/test_fidelity_routes.py + tests/test_main_coverage.py).
The helper functions are pure Python and tested directly with mock
trade objects — no DB roundtrip needed for those.
"""

import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient

from backend.main import app
from backend.database import init_db, SessionLocal, User, StockScore, Stock
from backend.auth import get_admin_user
from backend.routes import admin as admin_routes


# ── Auth bypass (admin) ────────────────────────────────────────────────


_fake_admin = User(
    id=99001, email="admin@test.com", display_name="Admin",
    is_active=True, is_admin=True, hashed_password="",
)


def _override_admin():
    return _fake_admin


# Module-level override removed; the per-test `_seed_admin_and_clean_users`
# fixture below applies + restores the override with proper isolation.
init_db()
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _module_cleanup():
    """When this file's tests finish, fully release the auth bypass +
    purge the fake admin row from the shared SessionLocal so other test
    files don't see leaked state.

    Without this, the id=99001 user persists in the dev SQLite DB across
    files; the dependency_overrides[get_admin_user] entry stays in
    backend.main.app and silently bypasses admin auth for any
    subsequent test that hits /api/admin/*.
    """
    yield
    # Teardown: wipe our fake admin + restore real auth resolver
    app.dependency_overrides.pop(get_admin_user, None)
    db = _db()
    try:
        db.query(User).filter(User.id == 99001).delete()
        db.commit()
    finally:
        db.close()


# ── DB setup / teardown ────────────────────────────────────────────────


def _db():
    return SessionLocal()


@pytest.fixture(autouse=True)
def _seed_admin_and_clean_users():
    """Each test starts with:
      1. The fake admin row (id=99001) re-seeded in the shared DB.
      2. `app.dependency_overrides[get_admin_user]` pointing at OUR
         _override_admin (other test files set their own overrides at
         module-import time; the last-loaded one wins, so we re-apply
         per-test rather than rely on module-level setup).

    Teardown wipes non-99001 user rows we created; module-scoped
    teardown will purge id=99001 itself when the file is done.
    """
    # (Re-)apply OUR admin override; remember the prior so we can
    # restore it at the end of this test.
    prev_override = app.dependency_overrides.get(get_admin_user)
    app.dependency_overrides[get_admin_user] = _override_admin

    db = _db()
    try:
        db.query(User).filter(User.id != 99001).delete()
        if not db.query(User).filter_by(id=99001).first():
            db.add(User(
                id=99001, email="admin@test.com", display_name="Admin",
                is_active=True, is_admin=True, hashed_password="",
            ))
        db.commit()
    finally:
        db.close()
    yield
    db = _db()
    try:
        db.query(User).filter(User.id != 99001).delete()
        db.commit()
    finally:
        db.close()
    # Restore prior override so subsequent files' tests see what they
    # set at their own module-import time.
    if prev_override is not None:
        app.dependency_overrides[get_admin_user] = prev_override
    else:
        app.dependency_overrides.pop(get_admin_user, None)


# ════════════════════════════════════════════════════════════════════════
# POST /api/admin/users — create_user (lines 41-69)
# ════════════════════════════════════════════════════════════════════════


class TestCreateUser:
    """Covers backend/routes/admin.py:42-69."""

    def test_create_user_happy_path_returns_user_response(self):
        """Branch: new email → 200, User.email lowercased, default
        display_name from local-part of the ORIGINAL email (case-preserved
        because .split() runs before .lower())."""
        r = client.post(
            "/api/admin/users",
            json={"email": "NewUser@Example.COM"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["email"] == "newuser@example.com"  # User.email lowercased
        # display_name comes from user_data.email.split("@")[0] — pre-lower
        # (route line 55 splits on the original, then line 53 lowercases for storage)
        assert body["display_name"] == "NewUser"
        assert body["is_active"] is True
        assert body["is_admin"] is False

    def test_create_user_with_explicit_display_name(self):
        """Branch: display_name provided → used directly."""
        r = client.post(
            "/api/admin/users",
            json={"email": "named@example.com", "display_name": "Named User"},
        )
        assert r.status_code == 200
        assert r.json()["display_name"] == "Named User"

    def test_duplicate_email_returns_400(self):
        """Branch: existing User with this email → 400 (line 49-50)."""
        # First creation succeeds
        r1 = client.post(
            "/api/admin/users", json={"email": "dup@example.com"},
        )
        assert r1.status_code == 200

        # Second creation with same email (any case) → 400
        r2 = client.post(
            "/api/admin/users", json={"email": "DUP@EXAMPLE.COM"},
        )
        assert r2.status_code == 400
        assert "already registered" in r2.json()["detail"]


# ════════════════════════════════════════════════════════════════════════
# PATCH /api/admin/users/{user_id} — update_user (lines 78-112)
# ════════════════════════════════════════════════════════════════════════


class TestUpdateUser:
    """Covers backend/routes/admin.py:79-112."""

    def _create_test_user(self, email="target@example.com", **kw):
        """Insert a non-admin User and return its id."""
        db = _db()
        try:
            u = User(
                email=email, display_name="Target",
                is_active=True, is_admin=False, hashed_password="",
            )
            for k, v in kw.items():
                setattr(u, k, v)
            db.add(u)
            db.commit()
            db.refresh(u)
            return u.id
        finally:
            db.close()

    def test_update_user_404_when_not_found(self):
        """Branch: user_id doesn't exist → 404 (line 87-88)."""
        r = client.patch("/api/admin/users/99999", json={"display_name": "X"})
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_update_display_name(self):
        """Branch: display_name set → field updated."""
        uid = self._create_test_user(email="rename@example.com")
        r = client.patch(
            f"/api/admin/users/{uid}",
            json={"display_name": "Renamed"},
        )
        assert r.status_code == 200
        assert r.json()["display_name"] == "Renamed"

    def test_update_is_active_toggle(self):
        """Branch: is_active=False on another user → applied."""
        uid = self._create_test_user(email="deactivate@example.com")
        r = client.patch(
            f"/api/admin/users/{uid}",
            json={"is_active": False},
        )
        assert r.status_code == 200
        assert r.json()["is_active"] is False

    def test_update_is_admin_promotion(self):
        """Branch: promote a regular user to admin."""
        uid = self._create_test_user(email="promote@example.com")
        r = client.patch(
            f"/api/admin/users/{uid}",
            json={"is_admin": True},
        )
        assert r.status_code == 200
        assert r.json()["is_admin"] is True

    def test_cannot_disable_self(self):
        """Branch: target.id == current_user.id with is_active=False → 400
        (line 91-92)."""
        # 99001 is the fake admin from the auth bypass
        r = client.patch(
            "/api/admin/users/99001",
            json={"is_active": False},
        )
        assert r.status_code == 400
        assert "Cannot disable your own account" in r.json()["detail"]

    def test_cannot_remove_own_admin(self):
        """Branch: target.id == current_user.id with is_admin=False → 400
        (line 93-94)."""
        r = client.patch(
            "/api/admin/users/99001",
            json={"is_admin": False},
        )
        assert r.status_code == 400
        assert "Cannot remove your own admin privileges" in r.json()["detail"]


# ════════════════════════════════════════════════════════════════════════
# Helper-function edge branches (no DB roundtrip needed)
# ════════════════════════════════════════════════════════════════════════


def _mock_trade(**kw):
    """Build a MagicMock standing in for a Trade ORM row.

    All trade-row helpers read attributes only, not relationships, so a
    MagicMock with explicit attributes is sufficient.
    """
    t = MagicMock()
    for k, v in kw.items():
        setattr(t, k, v)
    return t


class TestSummarizeTradesEdgeBranches:
    """Covers backend/routes/admin.py:144-145 (exception swallow during
    realized_pct math) and line 161 (SCORE_CRASH exit-reason bucket)."""

    def test_swallows_exception_during_realized_pct_math(self):
        """Line 144-145: float(t.cost_basis) raises → exception swallowed,
        realized_pcts unaffected.

        The SELL row also has reason 'STOP LOSS', so exit_reasons
        categorization still fires — exception is local to the
        realized_pct sub-block only.
        """
        weird_cost = MagicMock()
        weird_cost.__float__ = MagicMock(side_effect=TypeError("not a number"))
        trades = [
            _mock_trade(
                action='SELL',
                signal_factors=None,
                reason='STOP LOSS triggered',
                realized_gain=100.0,
                cost_basis=weird_cost,
                shares=10,
            ),
        ]
        result = admin_routes._summarize_trades(trades, days=30)
        # Categorization survived the exception
        assert result['exit_reasons'].get('STOP_LOSS', 0) == 1
        # No realized_pct accumulated — realized_sell_pct stayed None
        assert result['realized_sell_pct'] is None
        # The SELL row was still counted in by_action
        assert result['by_action'].get('SELL') == 1

    def test_score_crash_exit_reason_bucket(self):
        """Line 161: SCORE CRASH keyword in SELL reason → categorized as
        'SCORE_CRASH'."""
        trades = [
            _mock_trade(
                action='SELL',
                signal_factors=None,
                reason='SCORE CRASH detected after 25pt drop',
                realized_gain=-200.0,
                cost_basis=100.0,
                shares=10,
            ),
        ]
        result = admin_routes._summarize_trades(trades, days=30)
        assert result['exit_reasons'].get('SCORE_CRASH', 0) == 1

    def test_signal_factors_json_string_decoded(self):
        """Line 129-131: signal_factors stored as JSON string → parsed +
        ml_confidence extracted into the ml_confidence dict."""
        import json
        trades = [
            _mock_trade(
                action='BUY',
                signal_factors=json.dumps({"ml_confidence": 0.78}),
                reason='Composite score',
                realized_gain=None,
                cost_basis=None, shares=None,
            ),
        ]
        result = admin_routes._summarize_trades(trades, days=30)
        # ml_confidence dict populated from the JSON-string source
        assert result['ml_confidence']['n_with'] == 1
        assert result['ml_confidence']['mean'] == 0.78
        # 1 BUY counted in by_action
        assert result['by_action'].get('BUY') == 1


class TestPerTradeReturnsEdgeBranches:
    """Covers backend/routes/admin.py:302, 307-308."""

    def test_skips_sell_without_cost_basis(self):
        """Line 301-302: cost_basis missing → continue (skip row)."""
        trades = [
            _mock_trade(
                action='SELL',
                realized_gain=50.0,
                cost_basis=None,  # missing
                shares=10,
            ),
        ]
        assert admin_routes._per_trade_returns(trades) == []

    def test_skips_sell_without_shares(self):
        """Line 301-302: shares missing → continue."""
        trades = [
            _mock_trade(
                action='SELL',
                realized_gain=50.0,
                cost_basis=10.0,
                shares=None,  # missing
            ),
        ]
        assert admin_routes._per_trade_returns(trades) == []

    def test_skips_non_sell_rows(self):
        """Line 299-300: non-SELL action → continue early."""
        trades = [
            _mock_trade(action='BUY', realized_gain=None, cost_basis=10.0, shares=10),
            _mock_trade(action='PYRAMID', realized_gain=None, cost_basis=10.0, shares=10),
        ]
        assert admin_routes._per_trade_returns(trades) == []

    def test_swallows_exception_on_bad_numeric(self):
        """Line 307-308: float() raises mid-math → exception swallowed."""
        weird = MagicMock()
        weird.__float__ = MagicMock(side_effect=ValueError("not numeric"))
        trades = [
            _mock_trade(
                action='SELL',
                realized_gain=weird,  # float() raises
                cost_basis=10.0,
                shares=10,
            ),
        ]
        # No row produced, but no exception raised either
        assert admin_routes._per_trade_returns(trades) == []

    def test_happy_path_returns_pct_list(self):
        """Sanity: well-formed SELL → pct in output."""
        trades = [
            _mock_trade(
                action='SELL',
                realized_gain=200.0,  # +$200 on $1000 cost = +20%
                cost_basis=100.0,
                shares=10,
            ),
        ]
        result = admin_routes._per_trade_returns(trades)
        assert result == [20.0]


class TestRealizedMaxDrawdownEdgeBranches:
    """Covers backend/routes/admin.py:329-330 (exception during float)."""

    def test_returns_none_when_fewer_than_two_sells(self):
        """Line 320-321: <2 SELL rows → no curve, return None."""
        trades = [_mock_trade(action='SELL', realized_gain=10.0,
                              executed_at=datetime.now(timezone.utc))]
        assert admin_routes._realized_max_drawdown_pct(trades, 10000.0) is None

    def test_swallows_exception_in_cum_calc(self):
        """Line 329-330: float() on realized_gain raises → continue.

        Force the second SELL's realized_gain to raise on float(); the
        first SELL still increments `cum`, but the second is skipped.
        With only one valid SELL we still need >=2 SELLs to compute,
        and the second one fails the float() — so drawdown stays 0.
        """
        weird = MagicMock()
        weird.__float__ = MagicMock(side_effect=TypeError("bad"))
        now = datetime.now(timezone.utc)
        trades = [
            _mock_trade(
                action='SELL', realized_gain=100.0, executed_at=now,
            ),
            _mock_trade(
                action='SELL', realized_gain=weird,
                executed_at=now + timedelta(days=1),
            ),
        ]
        result = admin_routes._realized_max_drawdown_pct(trades, 10000.0)
        # No drawdown observed → 0.0 returned (not None — len(sells)>=2)
        assert result == 0.0

    def test_computes_drawdown_with_peak_then_trough(self):
        """Happy path: peak then trough → positive drawdown pct."""
        now = datetime.now(timezone.utc)
        trades = [
            _mock_trade(action='SELL', realized_gain=500.0, executed_at=now),
            _mock_trade(
                action='SELL', realized_gain=-300.0,
                executed_at=now + timedelta(days=1),
            ),
        ]
        result = admin_routes._realized_max_drawdown_pct(trades, 10000.0)
        # Peak=500, trough=200 → drawdown = (500-200)/(10000+500) = 2.86%
        assert result is not None
        assert result > 0


class TestSummarizeWindowEdgeBranches:
    """Covers backend/routes/admin.py:358, 364-365 — the cost_total
    computation in _summarize_window's capital-efficiency loop."""

    def test_skips_sell_without_cost_basis(self):
        """Line 357-358: cost_basis missing → continue."""
        trades = [
            _mock_trade(
                action='SELL', realized_gain=100.0,
                cost_basis=None, shares=10,
                signal_factors=None,
                reason='STOP LOSS',
                executed_at=datetime.now(timezone.utc),
            ),
        ]
        result = admin_routes._summarize_window(trades, days=30, starting_value=10000.0)
        # capital_efficiency_pct stays None (no valid cost totals)
        assert result.get('capital_efficiency_pct') is None

    def test_swallows_exception_in_cost_total(self):
        """Line 364-365: float(cost_basis) raises → continue."""
        weird = MagicMock()
        weird.__float__ = MagicMock(side_effect=TypeError("nope"))
        trades = [
            _mock_trade(
                action='SELL', realized_gain=100.0,
                cost_basis=weird, shares=10,
                signal_factors=None,
                reason='STOP LOSS',
                executed_at=datetime.now(timezone.utc),
            ),
        ]
        result = admin_routes._summarize_window(trades, days=30, starting_value=10000.0)
        # No crash; capital efficiency cleanly None
        assert result.get('capital_efficiency_pct') is None


# NOTE: backend/routes/admin.py:768-769 (the realized_pct exception
# swallow inside _serialize_trade_row) is *technically* unreachable in
# isolation: the same function's return-dict at lines 795-799 calls
# float() unguarded on shares/price/cost_basis/realized_gain. So any
# trade row crafted to trigger the exception inside the try-block also
# crashes the unguarded float() calls in the response dict. Those 2
# lines stay uncovered without a source-level refactor (e.g., guarding
# the return-dict float()s too). Trade-off: 2 stmts vs no source touch
# during the 2026-06-18 eval window. Accepted.


class TestBuildCumulativeCurveEdgeBranches:
    """Covers backend/routes/admin.py:828-829 — exception swallow inside
    _build_cumulative_curve when float(realized_gain) raises."""

    def test_swallows_exception_in_cum_gain(self):
        """Line 826-829: float(realized_gain) raises → continue."""
        weird = MagicMock()
        weird.__float__ = MagicMock(side_effect=TypeError("bad"))
        now = datetime.now(timezone.utc)
        trades = [
            _mock_trade(
                action='SELL', realized_gain=100.0, executed_at=now,
            ),
            _mock_trade(
                action='SELL', realized_gain=weird,
                executed_at=now + timedelta(days=1),
            ),
        ]
        start = now - timedelta(days=1)
        end = now + timedelta(days=2)
        curve = admin_routes._build_cumulative_curve(
            trades, start, end, starting_value=10000.0,
        )
        # First SELL produced a point; second was skipped silently.
        # Curve has at least one data point + terminal entry.
        assert len(curve) >= 1
        # cum_pct from first sell = 100/10000 * 100 = 1.0%
        assert any(pt['cum_pct'] == 1.0 for pt in curve)


# ════════════════════════════════════════════════════════════════════════
# _resolve_ab_window — post_window_days=None defaults to days-since-cutoff
# (line 608-609)
# ════════════════════════════════════════════════════════════════════════


class TestResolveAbWindowDefault:
    """Covers backend/routes/admin.py:608-609 — post_window_days=None
    auto-defaults to min(days_since_cutoff, 90)."""

    def test_none_defaults_to_days_since_cutoff_capped_at_90(self):
        """Branch: caller passes post_window_days=None → endpoint auto-
        derives it from cutoff_date. Requires an AIPortfolioConfig row
        on the strategy so _build_trade_query doesn't 404."""
        from backend.database import AIPortfolioConfig
        # Cutoff 7 days ago → days_since_cutoff=7, capped at 90 → 7
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()

        db = _db()
        try:
            # Clean & seed an AIPortfolioConfig on the strategy so
            # _build_trade_query returns a valid query
            db.query(AIPortfolioConfig).filter_by(user_id=99001).delete()
            db.add(AIPortfolioConfig(
                user_id=99001, starting_cash=25000, current_cash=20000,
                max_positions=8, max_position_pct=12,
                min_score_to_buy=72, sell_score_threshold=45,
                take_profit_pct=75, stop_loss_pct=8,
                is_active=True, strategy="nostate_optimized",
            ))
            db.commit()

            try:
                win = admin_routes._resolve_ab_window(
                    strategy="nostate_optimized",
                    cutoff_date=cutoff,
                    pre_window_days=30,
                    post_window_days=None,  # → auto-derive
                    exclude_pyramids=True,
                    db=db,
                    source="live",
                )
                assert win['post_window_days'] >= 1
                assert win['post_window_days'] <= 90
                # For a 7-day-old cutoff, expect ~7 days
                assert win['post_window_days'] <= 7
            finally:
                db.query(AIPortfolioConfig).filter_by(user_id=99001).delete()
                db.commit()
        finally:
            db.close()


# ════════════════════════════════════════════════════════════════════════
# cap_delta_diagnostics_endpoint — Query() coercion (lines 1195-1205)
# ════════════════════════════════════════════════════════════════════════


class TestCapDeltaDiagnosticsCoercion:
    """Covers backend/routes/admin.py:1194-1205. When tests bypass
    FastAPI's dependency resolver, Query() defaults arrive as Query
    objects rather than their underlying values — the endpoint coerces
    each one back to its expected type."""

    @pytest.fixture
    def db_session(self):
        """Scoped DB session — only deletes the rows this test seeds, not
        the entire Stock/StockScore tables (the existing test_cap_delta_
        diagnostics.py uses a broader wipe; we don't here because other
        tests in the suite rely on persistent Stock state)."""
        init_db()
        db = SessionLocal()
        # Track ticker prefixes seeded by these tests so cleanup is scoped.
        _our_tickers = {"ANOMALY"}
        # Pre-clean any prior leftovers under our prefixes
        stale = db.query(Stock).filter(Stock.ticker.in_(_our_tickers)).all()
        for s in stale:
            db.query(StockScore).filter(StockScore.stock_id == s.id).delete()
            db.delete(s)
        db.commit()
        try:
            yield db
        finally:
            # Targeted cleanup: only our tickers
            ours = db.query(Stock).filter(Stock.ticker.in_(_our_tickers)).all()
            for s in ours:
                db.query(StockScore).filter(StockScore.stock_id == s.id).delete()
                db.delete(s)
            db.commit()
            db.close()

    def test_query_object_days_coerced_to_default(self, db_session):
        """Lines 1194-1195: non-int `days` (Query sentinel) → coerced to 1."""
        from fastapi import Query as QueryFn
        import asyncio
        # Pass Query() sentinels for every coercion-checked arg
        result = asyncio.run(admin_routes.cap_delta_diagnostics_endpoint(
            days=QueryFn(default=1),
            min_delta=QueryFn(default=0.5),
            strategy=QueryFn(default="nostate_cs_bear"),
            buy_threshold=QueryFn(default=None),
            current_user=None,
            db=db_session,
            request=None,
        ))
        # All four coercions fired; endpoint returned a well-formed payload.
        assert 'window' in result
        assert result['window']['days'] == 1

    def test_explicit_args_skip_coercion(self, db_session):
        """Sanity: when args are already the right types, coercion is a
        no-op and the endpoint behaves identically."""
        import asyncio
        result = asyncio.run(admin_routes.cap_delta_diagnostics_endpoint(
            days=3,
            min_delta=1.0,
            strategy="nostate_cs_bear",
            buy_threshold=70.0,
            current_user=None,
            db=db_session,
            request=None,
        ))
        assert 'window' in result
        assert result['window']['days'] == 3
        assert result['threshold_crossings']['buy_threshold'] == 70.0


# ════════════════════════════════════════════════════════════════════════
# compute_cap_delta_diagnostics — defensive uncapped<threshold<=capped
# branch (line 1094-1095)
# ════════════════════════════════════════════════════════════════════════


class TestCapDeltaDefensiveBranch:
    """Covers backend/routes/admin.py:1094-1095 — the defensive branch
    where uncapped_total < buy_threshold <= total. This should never
    happen in practice (the cap can only lower the score, so uncapped >=
    capped), but the endpoint counts it so data anomalies are visible.

    To trigger it we need a StockScore row whose c_score_uncapped is
    LESS than c_score (i.e., delta < 0). The endpoint requires
    abs(delta) >= min_delta but the bucket logic only triggers the
    crossing check when delta is the raw value (not abs). Setting
    min_delta=0 lets a negative-delta row reach the threshold-crossing
    block.
    """

    @pytest.fixture
    def db_session(self):
        """Scoped DB session — only deletes the rows this test seeds, not
        the entire Stock/StockScore tables (the existing test_cap_delta_
        diagnostics.py uses a broader wipe; we don't here because other
        tests in the suite rely on persistent Stock state)."""
        init_db()
        db = SessionLocal()
        # Track ticker prefixes seeded by these tests so cleanup is scoped.
        _our_tickers = {"ANOMALY"}
        # Pre-clean any prior leftovers under our prefixes
        stale = db.query(Stock).filter(Stock.ticker.in_(_our_tickers)).all()
        for s in stale:
            db.query(StockScore).filter(StockScore.stock_id == s.id).delete()
            db.delete(s)
        db.commit()
        try:
            yield db
        finally:
            # Targeted cleanup: only our tickers
            ours = db.query(Stock).filter(Stock.ticker.in_(_our_tickers)).all()
            for s in ours:
                db.query(StockScore).filter(StockScore.stock_id == s.id).delete()
                db.delete(s)
            db.commit()
            db.close()

    def test_defensive_capped_above_uncapped_below_threshold(self, db_session):
        """Line 1094-1095: anomaly where uncapped_total < threshold <=
        total → cross_capped_above counter increments.

        StockScore has no `ticker` column — it joins via stock_id. The
        internal function `compute_cap_delta_diagnostics` accepts any
        min_delta (the endpoint's Query() adds a ge=0.0 constraint, but
        the function itself doesn't); pass a negative min_delta so the
        anomalous negative-delta row passes the `delta < min_delta`
        filter at line 1069 and reaches the defensive branch."""
        # Seed one stock with the inverted relationship (c_score_uncapped
        # < c_score). This is the data anomaly the defensive branch
        # exists for.
        s = Stock(ticker="ANOMALY", name="Anomaly",
                  sector="Tech", industry="Software")
        db_session.add(s)
        db_session.flush()

        # total_score=75, delta = c_score_uncapped - c_score = -5
        # → uncapped_total = 75 + (-5) = 70
        # buy_threshold = 72 → 70 < 72 <= 75 (defensive branch fires)
        now = datetime.now(timezone.utc)
        db_session.add(StockScore(
            stock_id=s.id,
            timestamp=now,
            date=now.date(),
            total_score=75.0, c_score=15.0, c_score_uncapped=10.0,
            a_score=10, n_score=10, s_score=10,
            l_score=10, i_score=10, m_score=10,
        ))
        db_session.commit()

        result = admin_routes.compute_cap_delta_diagnostics(
            db=db_session,
            days=1,
            min_delta=-10.0,  # bypass filter so negative-delta row reaches branch
            buy_threshold=72.0,
        )
        assert result['threshold_crossings']['rows_baseline_above_threshold_uncapped_below'] == 1
