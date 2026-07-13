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
# Defensive branches — small-tail close-out
# ============================================================================
class TestFormatHelpersDefensiveBranches:
    """The _fmt_* helpers swallow None and emit the dash glyph; the non-None
    happy path of _fmt_int needs an explicit test to lock comma formatting."""

    def test_fmt_int_non_none_uses_thousands_separator(self):
        from backend.ab_eval_email import _fmt_int
        assert _fmt_int(0) == "0"
        assert _fmt_int(1234) == "1,234"
        assert _fmt_int(1234567) == "1,234,567"

    def test_fmt_int_none_renders_dash(self):
        from backend.ab_eval_email import _fmt_int
        assert _fmt_int(None) == "–"


class TestPriorBuyMapBranches:
    """Buys in the pre-window seed the prior_buy_map; buys in the post-window
    update it as the loop walks chronologically (so subsequent post-window
    SELLs can pair against the latest BUY for that ticker). Without a BUY in
    either window neither branch fires."""

    def test_buy_in_pre_window_seeds_prior_buy_map(self, db_session):
        """Add a BUY in the pre-window, a paired SELL in the post-window —
        the SELL's hold_days should resolve via the pre-window BUY through
        prior_buy_map, which means line 167 fired during render."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        # 5 baseline SELLs in pre-window (decision-threshold floor).
        for i, g in enumerate([3.0, -2.0, 4.0, 5.0, -1.0, 2.0]):
            db_session.add(_make_trade(
                ticker=f"BASE{i}",
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
                executed_at=datetime(2026, 4, 1 + i, tzinfo=timezone.utc),
                user_id=1,
            ))
        # The BUY-in-pre / SELL-in-post pair the prior_buy_map should connect.
        db_session.add(_make_trade(
            ticker="PAIR",
            action="BUY", shares=10.0, cost_basis=100.0, realized_gain=None,
            executed_at=datetime(2026, 4, 10, tzinfo=timezone.utc),
            user_id=1,
        ))
        # 5 baseline SELLs in post-window so verdict isn't insufficient_data.
        for i, g in enumerate([20.0, 25.0, 18.0, 22.0, 15.0]):
            db_session.add(_make_trade(
                ticker=f"POSTB{i}",
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
                executed_at=datetime(2026, 4, 16 + i, tzinfo=timezone.utc),
                user_id=1,
            ))
        db_session.add(_make_trade(
            ticker="PAIR",
            action="SELL", shares=10.0, cost_basis=100.0, realized_gain=50.0,
            executed_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
            user_id=1,
        ))
        db_session.commit()
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        # Render succeeded — the BUY-walk path (line 167) executed without raising.
        assert "<!DOCTYPE html>" in snap['html']

    def test_buy_in_post_window_updates_prior_buy_map(self, db_session):
        """A BUY in the post-window followed by a paired SELL exercises the
        same accumulator on the post side (line 172)."""
        _seed_user_strategy(db_session, user_id=1, strategy="nostate_optimized")
        for i, g in enumerate([3.0, -2.0, 4.0, 5.0, -1.0, 2.0]):
            db_session.add(_make_trade(
                ticker=f"BASE{i}",
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
                executed_at=datetime(2026, 4, 1 + i, tzinfo=timezone.utc),
                user_id=1,
            ))
        # Post-window BUY at day 16, SELL at day 22 — same ticker, same user.
        db_session.add(_make_trade(
            ticker="POSTPAIR",
            action="BUY", shares=10.0, cost_basis=100.0, realized_gain=None,
            executed_at=datetime(2026, 4, 16, tzinfo=timezone.utc),
            user_id=1,
        ))
        for i, g in enumerate([20.0, 25.0, 18.0, 22.0]):
            db_session.add(_make_trade(
                ticker=f"POSTC{i}",
                action="SELL", shares=10.0, cost_basis=10.0, realized_gain=g,
                executed_at=datetime(2026, 4, 17 + i, tzinfo=timezone.utc),
                user_id=1,
            ))
        db_session.add(_make_trade(
            ticker="POSTPAIR",
            action="SELL", shares=10.0, cost_basis=100.0, realized_gain=50.0,
            executed_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
            user_id=1,
        ))
        db_session.commit()
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
        )
        assert "<!DOCTYPE html>" in snap['html']


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


# ============================================================================
# Step 6: source='shadow' plumbing through the snapshot builder + endpoint
# ============================================================================
class TestShadowSourceSnapshot:
    """Shadow source must flow strategy + cutoff through _resolve_ab_window
    and surface a [Shadow] prefix everywhere the operator can scan visually
    (subject, plain-text body, HTML badge). Reuses the helpers from
    test_strategy_ab_eval_shadow so the email path stays aligned with the
    dashboard path — divergence here would email a body computed off a
    different fixture than the dashboard renders."""

    def _seed_active_shadow_with_post_winners(self, db, name="shadow_baseline"):
        from backend.database import ShadowStrategy, ShadowTrade
        from tests.test_strategy_ab_eval_shadow import (
            _make_shadow_strategy, _make_shadow_trade,
        )
        ss = _make_shadow_strategy(name=name, starting_value=25000.0)
        db.add(ss)
        db.commit()
        db.refresh(ss)
        # Each SELL gets a same-day prior BUY, like a real shadow stack.
        # ShadowTrade has no user_id column, so any BUY in the window
        # exercises the _trade_scope_id path in the prior-BUY map —
        # regression for the 2026-07-13 scheduler crash (AttributeError:
        # 'ShadowTrade' object has no attribute 'user_id'), which a
        # SELL-only fixture could never hit.
        # 6 pre-cutoff modest mixed SELLs
        pre_gains = [5.0, -3.0, 4.0, 6.0, -2.0, 3.0]
        for i, g in enumerate(pre_gains):
            db.add(_make_shadow_trade(
                strategy_id=ss.id, action="BUY", shares=10.0,
                ticker=f"PRE{i}",
                executed_at=datetime(2026, 4, 1 + i, 10, tzinfo=timezone.utc),
            ))
            db.add(_make_shadow_trade(
                strategy_id=ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=g, ticker=f"PRE{i}",
                executed_at=datetime(2026, 4, 1 + i, 15, tzinfo=timezone.utc),
            ))
        # 6 post-cutoff strong winners
        post_gains = [25.0, 30.0, -5.0, 20.0, 18.0, 22.0]
        for i, g in enumerate(post_gains):
            db.add(_make_shadow_trade(
                strategy_id=ss.id, action="BUY", shares=10.0,
                ticker=f"POST{i}",
                executed_at=datetime(2026, 4, 16 + i, 10, tzinfo=timezone.utc),
            ))
            db.add(_make_shadow_trade(
                strategy_id=ss.id, action="SELL", shares=10.0, cost_basis=10.0,
                realized_gain=g, ticker=f"POST{i}",
                executed_at=datetime(2026, 4, 16 + i, 15, tzinfo=timezone.utc),
                reason="Take profit (40% trail)",
            ))
        db.commit()
        return ss

    def _shadow_db_session(self):
        from backend.database import (
            init_db, SessionLocal, AIPortfolioTrade, AIPortfolioConfig,
            ShadowStrategy, ShadowTrade,
        )
        init_db()
        db = SessionLocal()
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioConfig).delete()
        db.commit()
        return db

    def test_subject_carries_shadow_prefix(self):
        db = self._shadow_db_session()
        try:
            self._seed_active_shadow_with_post_winners(db)
            from backend.ab_eval_email import build_ab_eval_snapshot_html
            snap = build_ab_eval_snapshot_html(
                "shadow_baseline", "2026-04-15", db,
                pre_window_days=14, post_window_days=14, source="shadow",
            )
            assert snap['subject'].startswith("[Shadow] ")
            assert "shadow_baseline" in snap['subject']
            assert snap['source'] == 'shadow'
        finally:
            db.close()

    def test_html_body_renders_shadow_badge(self):
        db = self._shadow_db_session()
        try:
            self._seed_active_shadow_with_post_winners(db)
            from backend.ab_eval_email import build_ab_eval_snapshot_html
            snap = build_ab_eval_snapshot_html(
                "shadow_baseline", "2026-04-15", db,
                pre_window_days=14, post_window_days=14, source="shadow",
            )
            # The header strip surfaces a Shadow chip so the operator can
            # tell at a glance which delivery is the live verdict.
            assert ">Shadow</span>" in snap['html']
        finally:
            db.close()

    def test_text_body_carries_shadow_prefix(self):
        db = self._shadow_db_session()
        try:
            self._seed_active_shadow_with_post_winners(db)
            from backend.ab_eval_email import build_ab_eval_snapshot_html
            snap = build_ab_eval_snapshot_html(
                "shadow_baseline", "2026-04-15", db,
                pre_window_days=14, post_window_days=14, source="shadow",
            )
            assert snap['text'].startswith("[Shadow] ")
        finally:
            db.close()

    def test_shadow_buys_key_prior_map_by_strategy_id(self):
        """Regression for the 2026-07-13 weekly-email crash: the snapshot
        builder's prior-BUY map keyed on t.user_id, which ShadowTrade rows
        don't have. With BUYs in the fixture the builder must not raise,
        and a post SELL with no persisted holding_days must pair with its
        prior BUY via (ticker, shadow_strategy_id) for hold_days."""
        db = self._shadow_db_session()
        try:
            self._seed_active_shadow_with_post_winners(db)
            from backend.ab_eval_email import build_ab_eval_snapshot_html
            snap = build_ab_eval_snapshot_html(
                "shadow_baseline", "2026-04-15", db,
                pre_window_days=14, post_window_days=14, source="shadow",
            )
            # Fixture SELLs land 5h after their BUY, so the fallback diff
            # is 0 days — the pairing worked iff hold_days rendered as 0d
            # (an unpaired SELL renders the placeholder instead).
            assert snap['post_sell_count'] == 6
            assert '>0d</td>' in snap['html']
        finally:
            db.close()

    def test_live_source_does_not_render_shadow_markers(self, db_session):
        """Regression guard: live emails must never carry shadow visual
        markers. Tightens the contract that shadow markers fire only when
        explicitly opted in."""
        _seed_keep_scenario(db_session)
        from backend.ab_eval_email import build_ab_eval_snapshot_html
        snap = build_ab_eval_snapshot_html(
            "nostate_optimized", "2026-04-15", db_session,
            pre_window_days=14, post_window_days=14,
            # source defaults to 'live'
        )
        assert "[Shadow]" not in snap['subject']
        assert "[Shadow] " not in snap['text']
        assert ">Shadow</span>" not in snap['html']
        assert snap['source'] == 'live'

    def test_send_wrapper_passes_source_through(self):
        db = self._shadow_db_session()
        try:
            self._seed_active_shadow_with_post_winners(db)
            from backend import ab_eval_email
            with patch.object(ab_eval_email, 'send_email', return_value=True) as mock_send:
                result = ab_eval_email.send_ab_eval_snapshot(
                    strategy="shadow_baseline",
                    cutoff_date="2026-04-15",
                    db=db,
                    pre_window_days=14,
                    post_window_days=14,
                    source="shadow",
                )
            args, _ = mock_send.call_args
            assert args[0].startswith("[Shadow] ")
            assert result['source'] == 'shadow'
            assert result['decision'] == 'keep'
        finally:
            db.close()

    def test_endpoint_accepts_source_shadow(self):
        db = self._shadow_db_session()
        try:
            self._seed_active_shadow_with_post_winners(db)
            from backend.routes.admin import (
                trigger_ab_eval_snapshot_email,
                ABEvalEmailTestRequest,
            )
            from backend import ab_eval_email
            payload = ABEvalEmailTestRequest(
                strategy="shadow_baseline",
                cutoff_date="2026-04-15",
                pre_window_days=14,
                post_window_days=14,
                source="shadow",
            )
            mock_admin = MagicMock(spec=User)
            mock_admin.id = 1
            mock_admin.is_admin = True
            mock_admin.is_active = True
            with patch.object(ab_eval_email, 'send_email', return_value=True):
                result = asyncio.run(trigger_ab_eval_snapshot_email(
                    payload=payload, current_user=mock_admin, db=db,
                ))
            assert result['sent'] is True
            assert result['source'] == 'shadow'
            assert result['subject'].startswith("[Shadow] ")
        finally:
            db.close()

    def test_endpoint_unknown_shadow_404(self):
        """Unknown shadow name propagates verbatim from _build_trade_query."""
        db = self._shadow_db_session()
        try:
            from backend.routes.admin import (
                trigger_ab_eval_snapshot_email,
                ABEvalEmailTestRequest,
            )
            payload = ABEvalEmailTestRequest(
                strategy="does_not_exist",
                cutoff_date="2026-04-15",
                pre_window_days=14,
                post_window_days=14,
                source="shadow",
            )
            mock_admin = MagicMock(spec=User)
            mock_admin.id = 1
            mock_admin.is_admin = True
            mock_admin.is_active = True
            with pytest.raises(HTTPException) as exc:
                asyncio.run(trigger_ab_eval_snapshot_email(
                    payload=payload, current_user=mock_admin, db=db,
                ))
            assert exc.value.status_code == 404
            assert "Shadow strategy 'does_not_exist' not found" in exc.value.detail
        finally:
            db.close()
