"""Tests for backend.scheduler._run_weekly_ab_eval_email — Step 6 fan-out.

Step 6 parametrizes the Mon 9 AM UTC cron to deliver one snapshot per
delivery target: live first, then one per active ShadowStrategy. The
critical contracts:

  1. Live always runs first — it's the verdict-driver for the June 18 read.
  2. Each shadow stack is independently failure-isolated. A single shadow
     blowing up must not kill the live email or the other shadows.
  3. Archived shadows (archived_at != NULL) are skipped.
  4. CANSLIM_ENV=test continues to short-circuit the entire job — guards
     the test suite from accidentally hitting real SMTP.

We patch `send_ab_eval_snapshot` directly rather than running the full
email build path; the snapshot-render side is covered exhaustively by
test_ab_eval_email.py. Here we only validate scheduler-side ordering and
isolation.
"""

import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, ShadowStrategy, ShadowTrade,
    AIPortfolioTrade, AIPortfolioConfig,
)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.query(AIPortfolioTrade).delete()
    db.query(AIPortfolioConfig).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.query(AIPortfolioTrade).delete()
        db.query(AIPortfolioConfig).delete()
        db.commit()
        db.close()


def _make_shadow(name, archived=False):
    archived_at = datetime.now(timezone.utc) if archived else None
    return ShadowStrategy(
        name=name,
        parent_strategy="nostate_cs_bear",
        config_snapshot={},
        scorer_overrides={},
        starting_value=25000.0,
        archived_at=archived_at,
    )


def _ok_result(decision='keep', sent=True, post_sells=6):
    return {
        'sent': sent,
        'recipient': 'bayern.mikedeane@gmail.com',
        'subject': f'CANSLIM A/B [{decision.upper()}] x',
        'decision': decision,
        'post_sell_count': post_sells,
        'source': 'live',
    }


class TestWeeklyABEvalEmailFanOut:
    def _patch_env(self, monkeypatch):
        # Force the cron to actually run — CANSLIM_ENV=test is the suite
        # default and would short-circuit before any DB read.
        monkeypatch.setenv("CANSLIM_ENV", "development")

    def test_test_env_short_circuits(self, monkeypatch):
        """CANSLIM_ENV=test must skip without ever touching the DB or
        send_email — the test suite relies on this so it doesn't hit SMTP."""
        monkeypatch.setenv("CANSLIM_ENV", "test")
        from backend import scheduler as scheduler_mod
        with patch.object(scheduler_mod, 'logger') as mock_logger:
            # Patch SessionLocal to error if accessed — proves no DB touch.
            with patch('backend.database.SessionLocal', side_effect=AssertionError("DB touched in test env")):
                scheduler_mod._run_weekly_ab_eval_email()
            mock_logger.debug.assert_called_once()

    def test_live_only_when_no_shadows(self, db_session, monkeypatch):
        """Zero active shadows → exactly one call (live). Verifies the
        baseline behavior matches the pre-Step-6 contract."""
        self._patch_env(monkeypatch)
        from backend import scheduler as scheduler_mod
        with patch('backend.ab_eval_email.send_ab_eval_snapshot',
                   return_value=_ok_result()) as mock_send:
            scheduler_mod._run_weekly_ab_eval_email()
        assert mock_send.call_count == 1
        kwargs = mock_send.call_args.kwargs
        assert kwargs['strategy'] == 'nostate_cs_bear'
        assert kwargs['cutoff_date'] == '2026-05-07'
        assert 'source' not in kwargs or kwargs.get('source') == 'live'

    def test_live_then_one_per_active_shadow(self, db_session, monkeypatch):
        """Two active shadows → 3 calls: live + shadow_a + shadow_b. Live
        comes first; shadows iterate in id order."""
        self._patch_env(monkeypatch)
        db_session.add(_make_shadow("shadow_a"))
        db_session.add(_make_shadow("shadow_b"))
        db_session.commit()

        from backend import scheduler as scheduler_mod
        with patch('backend.ab_eval_email.send_ab_eval_snapshot',
                   return_value=_ok_result()) as mock_send:
            scheduler_mod._run_weekly_ab_eval_email()

        assert mock_send.call_count == 3
        calls = mock_send.call_args_list
        # Live first
        assert calls[0].kwargs['strategy'] == 'nostate_cs_bear'
        assert calls[0].kwargs.get('source', 'live') == 'live'
        # Shadows after, source='shadow', cutoff matches live
        assert calls[1].kwargs['strategy'] == 'shadow_a'
        assert calls[1].kwargs['source'] == 'shadow'
        assert calls[1].kwargs['cutoff_date'] == '2026-05-07'
        assert calls[2].kwargs['strategy'] == 'shadow_b'
        assert calls[2].kwargs['source'] == 'shadow'

    def test_archived_shadows_skipped(self, db_session, monkeypatch):
        """archived_at != NULL → not in the fan-out. Active sibling still
        gets its email. Mirrors the listing route's default filter."""
        self._patch_env(monkeypatch)
        db_session.add(_make_shadow("shadow_active"))
        db_session.add(_make_shadow("shadow_legacy", archived=True))
        db_session.commit()

        from backend import scheduler as scheduler_mod
        with patch('backend.ab_eval_email.send_ab_eval_snapshot',
                   return_value=_ok_result()) as mock_send:
            scheduler_mod._run_weekly_ab_eval_email()

        assert mock_send.call_count == 2  # live + active only
        called_strategies = [c.kwargs['strategy'] for c in mock_send.call_args_list]
        assert 'shadow_legacy' not in called_strategies
        assert 'shadow_active' in called_strategies

    def test_shadow_failure_does_not_abort_live(self, db_session, monkeypatch):
        """Live ran first AND succeeded; shadow exception is caught. Critical
        contract — June 18 verdict can't go missing because a shadow stack
        had a stale fixture."""
        self._patch_env(monkeypatch)
        db_session.add(_make_shadow("shadow_broken"))
        db_session.commit()

        from backend import scheduler as scheduler_mod
        # First call (live) succeeds; second (shadow) raises.
        side_effects = [_ok_result(), Exception("shadow boom")]
        with patch('backend.ab_eval_email.send_ab_eval_snapshot',
                   side_effect=side_effects) as mock_send, \
             patch.object(scheduler_mod, 'logger') as mock_logger:
            scheduler_mod._run_weekly_ab_eval_email()

        assert mock_send.call_count == 2
        # Live success was logged, shadow failure was logged at error level.
        assert any("(live)" in str(c) for c in mock_logger.info.call_args_list)
        assert any("Shadow A/B eval email failed for 'shadow_broken'" in str(c)
                   for c in mock_logger.error.call_args_list)

    def test_one_shadow_failing_does_not_abort_other_shadows(self, db_session, monkeypatch):
        """Two shadows, first one raises. The second shadow must still
        receive its snapshot — failure is isolated per-stack, not per-loop."""
        self._patch_env(monkeypatch)
        db_session.add(_make_shadow("shadow_first"))
        db_session.add(_make_shadow("shadow_second"))
        db_session.commit()

        from backend import scheduler as scheduler_mod
        side_effects = [
            _ok_result(),                    # live
            Exception("shadow_first boom"),  # first shadow raises
            _ok_result(decision='revert'),   # second shadow still runs
        ]
        with patch('backend.ab_eval_email.send_ab_eval_snapshot',
                   side_effect=side_effects) as mock_send:
            scheduler_mod._run_weekly_ab_eval_email()

        assert mock_send.call_count == 3
        called_strategies = [c.kwargs['strategy'] for c in mock_send.call_args_list]
        assert called_strategies == ['nostate_cs_bear', 'shadow_first', 'shadow_second']

    def test_live_failure_does_not_abort_shadow_fan_out(self, db_session, monkeypatch):
        """Live raises; shadow still runs. Inverse contract — operator
        loses the live verdict but still gets shadow visibility."""
        self._patch_env(monkeypatch)
        db_session.add(_make_shadow("shadow_keeps_running"))
        db_session.commit()

        from backend import scheduler as scheduler_mod
        side_effects = [Exception("live boom"), _ok_result()]
        with patch('backend.ab_eval_email.send_ab_eval_snapshot',
                   side_effect=side_effects) as mock_send, \
             patch.object(scheduler_mod, 'logger') as mock_logger:
            scheduler_mod._run_weekly_ab_eval_email()

        assert mock_send.call_count == 2
        assert any("Live A/B eval email failed" in str(c)
                   for c in mock_logger.error.call_args_list)

    def test_outer_db_session_failure_logs_and_returns(self, monkeypatch):
        """SessionLocal blowing up at the outer level is caught — the cron
        cannot let an exception propagate into APScheduler and kill the
        whole scheduler thread."""
        monkeypatch.setenv("CANSLIM_ENV", "development")
        from backend import scheduler as scheduler_mod
        with patch('backend.database.SessionLocal',
                   side_effect=RuntimeError("DB unavailable")), \
             patch.object(scheduler_mod, 'logger') as mock_logger:
            # Should not raise.
            scheduler_mod._run_weekly_ab_eval_email()
        assert any("Weekly A/B eval email outer failure" in str(c)
                   for c in mock_logger.error.call_args_list)
