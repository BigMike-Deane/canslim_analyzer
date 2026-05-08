"""Tests for the weekly A/B-eval snapshot email.

Covers backend/ab_eval_email.py (HTML render + send wrapper) and the
/api/admin/strategy-ab-eval/email-test endpoint. Reuses the seed helpers
from test_strategy_ab_eval — keep them aligned: a divergence here would
mean the email is computed off a different fixture than the dashboard.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi import HTTPException

from backend.database import (
    init_db, SessionLocal, AIPortfolioTrade, AIPortfolioConfig, User,
)

# Reuse the AB-eval test seed helpers verbatim — single source of truth for
# fixtures keeps email tests from drifting away from the dashboard tests.
from tests.test_strategy_ab_eval import _make_trade, _seed_user_strategy


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioConfig).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioConfig).delete()
        db.commit()
        db.close()


def _seed_keep_scenario(db, user_id=1, strategy="nostate_optimized"):
    """Pre window has flat returns; post window has strong winners — produces
    a 'keep' verdict with non-zero post SELL count for top-N rendering."""
    _seed_user_strategy(db, user_id=user_id, strategy=strategy)
    # 6 pre-cutoff SELLs, modest mixed returns
    pre_gains = [5.0, -3.0, 4.0, 6.0, -2.0, 3.0]
    for i, g in enumerate(pre_gains):
        db.add(_make_trade(
            ticker=f"PRE{i}",
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
            executed_at=datetime(2026, 4, 1 + i, tzinfo=timezone.utc),
            user_id=user_id,
        ))
    # 6 post-cutoff SELLs, much stronger
    post_gains = [25.0, 30.0, -5.0, 20.0, 18.0, 22.0]
    for i, g in enumerate(post_gains):
        db.add(_make_trade(
            ticker=f"POST{i}",
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
            executed_at=datetime(2026, 4, 16 + i, tzinfo=timezone.utc),
            reason="Take profit (40% trail)",
            user_id=user_id,
        ))
    db.commit()


def _seed_revert_scenario(db, user_id=1, strategy="nostate_optimized"):
    """Pre has strong winners; post regresses hard — produces 'revert'.

    Realized gains scaled up to $250+ so the return-pct delta on a $25k
    starting value crosses the -5pp threshold (otherwise both metrics regress
    but absolute return delta stays tiny → verdict reads marginal)."""
    _seed_user_strategy(db, user_id=user_id, strategy=strategy)
    pre_gains = [250.0, 300.0, 200.0, 180.0, 220.0, 280.0]
    for i, g in enumerate(pre_gains):
        db.add(_make_trade(
            ticker=f"PRE{i}",
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
            executed_at=datetime(2026, 4, 1 + i, tzinfo=timezone.utc),
            user_id=user_id,
        ))
    post_gains = [-200.0, -250.0, -150.0, -300.0, -100.0, -160.0]
    for i, g in enumerate(post_gains):
        db.add(_make_trade(
            ticker=f"POST{i}",
            action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
            executed_at=datetime(2026, 4, 16 + i, tzinfo=timezone.utc),
            user_id=user_id,
        ))
    db.commit()


# ============================================================================
# build_ab_eval_snapshot_html
# ============================================================================
class TestBuildSnapshotHTML:
    def test_keep_decision_label_in_html(self, db_session):
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert snap['decision'] == 'keep'
        assert 'KEEP' in snap['html']
        assert 'KEEP' in snap['subject']

    def test_revert_decision_label_in_html(self, db_session):
        _seed_revert_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert snap['decision'] == 'revert'
        assert 'REVERT' in snap['html']

    def test_window_dates_appear_in_html(self, db_session):
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        # Both window-boundary dates should be present somewhere in the body.
        assert "2026-04-01" in snap['html']  # pre start (cutoff - 14d)
        assert "2026-04-15" in snap['html']  # cutoff
        assert "2026-04-29" in snap['html']  # post end (cutoff + 14d)

    def test_starting_value_reference_appears(self, db_session):
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert snap['starting_value_reference'] == 25000.0
        # Rendered into HTML next to the side-by-side header.
        assert "starting_value_reference" in snap['html']
        assert "25,000" in snap['html']

    def test_top_5_post_cutoff_sells_sorted_desc(self, db_session):
        """The 'best' table sorts post SELLs by realized_pct desc, capped at 5.
        With our 6-SELL post window (winners 30,25,22,20,18 and one -5 loser),
        the best block must list +30% first, then descending."""
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        html = snap['html']
        # Header label is locked — operator searches for it visually.
        assert "Top 5 post-cutoff SELLs (best)" in html
        # Best section sorts desc: +30% appears before +25% in the document.
        idx_best = html.index("Top 5 post-cutoff SELLs (best)")
        idx_worst = html.index("Top 5 post-cutoff SELLs (worst)")
        best_block = html[idx_best:idx_worst]
        idx_30 = best_block.index("+30.00%")
        idx_25 = best_block.index("+25.00%")
        assert idx_30 < idx_25, "best block must be sorted realized_pct desc"

    def test_worst_block_renders_loser(self, db_session):
        """Worst-block sorts asc, so the -5% loser shows up first there even
        though the keep verdict makes the email overall positive."""
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert "Top 5 post-cutoff SELLs (worst)" in snap['html']
        # Worst block contains the loser; absent best block it would be missed.
        assert "-5.00%" in snap['html']

    def test_empty_post_window_renders_empty_state(self, db_session):
        """No post-cutoff SELLs → top-N tables fall back to italic empty-state
        copy. Subject still labels the decision (insufficient_data here)."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        # Only pre-cutoff SELLs.
        for i in range(6):
            db_session.add(_make_trade(
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=10.0,
                executed_at=datetime(2026, 4, 1 + i, tzinfo=timezone.utc),
            ))
        db_session.commit()
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert snap['post_sell_count'] == 0
        assert "No closed SELLs in the post-cutoff window" in snap['html']
        assert snap['decision'] == 'insufficient_data'
        assert "INSUFFICIENT DATA" in snap['html']

    def test_subject_includes_strategy_and_cutoff(self, db_session):
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert "nostate_optimized" in snap['subject']
        assert "2026-04-15" in snap['subject']

    def test_warnings_render_when_post_window_short(self, db_session):
        """Post window <21d triggers a 'minimum recommended' warning. The
        operator must see this caveat next to the verdict — silently emailing
        a 'keep' off a 14-day window would be misleading."""
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,  # both <21
        )
        assert "minimum recommended" in snap['html']

    def test_text_body_contains_decision_summary(self, db_session):
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        text = snap['text']
        assert "Decision: KEEP" in text
        assert "nostate_optimized" in text
        assert "2026-04-15" in text


# ============================================================================
# send_ab_eval_snapshot wrapper
# ============================================================================
class TestSendSnapshot:
    def test_calls_send_email_with_subject_html_text_recipient(self, db_session):
        _seed_keep_scenario(db_session)
        from backend import ab_eval_email
        with patch.object(ab_eval_email, 'send_email', return_value=True) as mock_send:
            result = ab_eval_email.send_ab_eval_snapshot(
                strategy="nostate_optimized",
                cutoff_date="2026-04-15",
                db=db_session,
                recipient="bayern.mikedeane@gmail.com",
                pre_window_days=14,
                post_window_days=14,
            )
        # send_email called once with (subject, html, text, recipient=...).
        assert mock_send.call_count == 1
        args, kwargs = mock_send.call_args
        assert "KEEP" in args[0]  # subject
        assert "<!DOCTYPE html>" in args[1]  # html body
        assert "Decision: KEEP" in args[2]  # text body
        assert kwargs.get('recipient') == "bayern.mikedeane@gmail.com"
        assert result['sent'] is True
        assert result['recipient'] == "bayern.mikedeane@gmail.com"
        assert result['decision'] == 'keep'

    def test_default_recipient_resolves_to_module_default(self, db_session):
        _seed_keep_scenario(db_session)
        from backend import ab_eval_email
        from backend import email_utils
        with patch.object(ab_eval_email, 'send_email', return_value=True):
            result = ab_eval_email.send_ab_eval_snapshot(
                strategy="nostate_optimized",
                cutoff_date="2026-04-15",
                db=db_session,
                pre_window_days=14,
                post_window_days=14,
            )
        # No recipient passed → resolved address is the module-level default.
        assert result['recipient'] == email_utils.RECIPIENT_EMAIL


# ============================================================================
# /api/admin/strategy-ab-eval/email-test endpoint
# ============================================================================
class TestEmailTestEndpoint:
    def _admin_user(self, is_admin=True):
        u = MagicMock(spec=User)
        u.id = 1
        u.email = "admin@test"
        u.is_active = True
        u.is_admin = is_admin
        return u

    def test_admin_guard_rejects_non_admin(self):
        """Non-admin user → 403 before the snapshot is ever built. Verified
        directly against get_admin_user since the endpoint is wired through
        Depends(get_admin_user)."""
        from backend.auth import get_admin_user
        non_admin = self._admin_user(is_admin=False)
        with pytest.raises(HTTPException) as exc:
            get_admin_user(current_user=non_admin)
        assert exc.value.status_code == 403
        assert "Admin access required" in exc.value.detail

    def test_400_on_bad_cutoff_date(self, db_session):
        """Bad ISO date propagates verbatim from _resolve_ab_window — same
        wording as the dashboard endpoint, locked by the existing 32 tests."""
        _seed_keep_scenario(db_session)
        from backend.routes.admin import (
            trigger_ab_eval_snapshot_email,
            ABEvalEmailTestRequest,
        )
        payload = ABEvalEmailTestRequest(
            strategy="nostate_optimized",
            cutoff_date="not-a-date",
        )
        with pytest.raises(HTTPException) as exc:
            asyncio.run(trigger_ab_eval_snapshot_email(
                payload=payload, current_user=self._admin_user(), db=db_session,
            ))
        assert exc.value.status_code == 400
        assert "valid ISO date" in exc.value.detail

    def test_happy_path_returns_sent_true_and_recipient(self, db_session):
        _seed_keep_scenario(db_session)
        from backend.routes.admin import (
            trigger_ab_eval_snapshot_email,
            ABEvalEmailTestRequest,
        )
        from backend import ab_eval_email
        payload = ABEvalEmailTestRequest(
            strategy="nostate_optimized",
            cutoff_date="2026-04-15",
            recipient_override="ops@example.com",
            pre_window_days=14,
            post_window_days=14,
        )
        with patch.object(ab_eval_email, 'send_email', return_value=True):
            result = asyncio.run(trigger_ab_eval_snapshot_email(
                payload=payload, current_user=self._admin_user(), db=db_session,
            ))
        assert result['sent'] is True
        assert result['recipient'] == "ops@example.com"
        assert result['decision'] == 'keep'
        assert "KEEP" in result['subject']
