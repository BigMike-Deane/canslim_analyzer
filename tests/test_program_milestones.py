"""Tests for the Program Ledger — backend/milestones.py writers + the
/api/admin/program-milestones routes.

The auto-writer tests exercise the REAL compute_experiment_gates against
fabricated DB rows (mirroring tests/test_experiment_gates.py) rather than
monkeypatching the gates payload, so a gates-shape change that would break
the writer breaks these tests too.
"""

import asyncio
import os
from datetime import date, datetime, timezone

import pytest
from fastapi import HTTPException

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, AIPortfolioTrade, MarketSnapshot,
    ProgramMilestone, ShadowStrategy, ShadowTrade,
)
from backend.milestones import (
    _SEEDS, add_milestone, record_auto_milestones, seed_history,
)

T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)

_MODELS = (ProgramMilestone, ShadowTrade, ShadowStrategy,
           MarketSnapshot, AIPortfolioTrade)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    for model in _MODELS:
        db.query(model).delete()
    db.commit()
    try:
        yield db
    finally:
        for model in _MODELS:
            db.query(model).delete()
        db.commit()
        db.close()


def _arm(db, name, *, activated_at=T0):
    s = ShadowStrategy(
        name=name, parent_strategy="nostate_cs_bear", config_snapshot={},
        scorer_overrides={}, starting_value=25000.0, activated_at=activated_at,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _keys(db):
    return {r.dedupe_key for r in db.query(ProgramMilestone).all()}


class TestAddMilestone:
    def test_dedupe_key_blocks_second_insert(self, db_session):
        assert add_milestone(db_session, title="one", dedupe_key="k1") is not None
        assert add_milestone(db_session, title="one again", dedupe_key="k1") is None
        assert db_session.query(ProgramMilestone).count() == 1

    def test_null_dedupe_rows_can_repeat(self, db_session):
        add_milestone(db_session, title="note", source="owner")
        add_milestone(db_session, title="note", source="owner")
        assert db_session.query(ProgramMilestone).count() == 2

    def test_unknown_category_falls_back_to_research(self, db_session):
        row = add_milestone(db_session, title="x", category="bogus")
        assert row.category == "research"


class TestSeedHistory:
    def test_seeds_once_then_noop(self, db_session):
        assert seed_history(db_session) == len(_SEEDS)
        assert seed_history(db_session) == 0
        assert db_session.query(ProgramMilestone).count() == len(_SEEDS)

    def test_seeds_preclaim_auto_namespace_with_true_dates(self, db_session):
        seed_history(db_session)
        row = db_session.query(ProgramMilestone).filter(
            ProgramMilestone.dedupe_key ==
            "gate:shadow_ml_veto_off:sub-0.30-confidence buys taken:target"
        ).first()
        assert row is not None
        assert row.occurred_at.date() == date(2026, 8, 20)
        # And the auto writer must therefore skip it even when the
        # condition holds (arm with 5 sub-0.30 buys).
        a = _arm(db_session, "shadow_ml_veto_off")
        for i in range(5):
            db_session.add(ShadowTrade(
                shadow_strategy_id=a.id, ticker=f"T{i}", action="BUY",
                shares=1.0, price=10.0, total_value=10.0,
                signal_factors={"ml_confidence": 0.1}, executed_at=T0))
        db_session.commit()
        record_auto_milestones(db_session)
        rows = db_session.query(ProgramMilestone).filter(
            ProgramMilestone.dedupe_key.like("gate:shadow_ml_veto_off%")).all()
        assert len(rows) == 1 and rows[0].occurred_at.date() == date(2026, 8, 20)


def _noncal_keys(db):
    """Ledger keys minus calendar-clock rows: the calendar clocks compare
    REAL pre-registered dates against wall-clock today, so once a date
    passes, every pass legitimately records it. Excluding them keeps these
    tests from expiring when a program date arrives."""
    return {k for k in _keys(db) if k and not k.startswith("calendar:")}


class TestAutoWriter:
    def test_empty_state_records_no_gate_events(self, db_session):
        record_auto_milestones(db_session)
        assert _noncal_keys(db_session) == set()

    def test_first_chop_day_fires_once(self, db_session):
        _arm(db_session, "shadow_chop_damper", activated_at=T0)
        db_session.add(MarketSnapshot(
            date=date(2026, 8, 2), spy_price=505.0, spy_50_ma=500.0))
        db_session.commit()
        record_auto_milestones(db_session)
        assert _noncal_keys(db_session) == {"gate:shadow_chop_damper:chop days:first"}
        # Idempotent — second pass records nothing new.
        record_auto_milestones(db_session)
        assert _noncal_keys(db_session) == {"gate:shadow_chop_damper:chop days:first"}

    def test_target_met_fires_with_verdict_disclaimer(self, db_session):
        a = _arm(db_session, "shadow_ml_veto_off")
        for i in range(5):
            db_session.add(ShadowTrade(
                shadow_strategy_id=a.id, ticker=f"T{i}", action="BUY",
                shares=1.0, price=10.0, total_value=10.0,
                signal_factors={"ml_confidence": 0.1}, executed_at=T0))
        db_session.commit()
        record_auto_milestones(db_session)
        row = db_session.query(ProgramMilestone).filter(
            ProgramMilestone.dedupe_key ==
            "gate:shadow_ml_veto_off:sub-0.30-confidence buys taken:target"
        ).first()
        assert row is not None and row.source == "auto"
        assert "5/5" in row.title

    def test_stop_verdict_records_as_verdict(self, db_session):
        for i in range(5):
            db_session.add(AIPortfolioTrade(
                ticker=f"S{i}", action="SELL", shares=10.0, price=92.0,
                total_value=920.0, cost_basis=100.0, realized_gain=-80.0,
                user_id=1, reason="STOP LOSS: Down 8.0%",
                executed_at=datetime(2026, 7, 1, tzinfo=timezone.utc)))
        db_session.commit()
        record_auto_milestones(db_session)
        keys = _keys(db_session)
        assert "clock:stop_loss_recheck:target" in keys
        assert "clock:stop_loss_recheck:verdict" in keys
        verdict = db_session.query(ProgramMilestone).filter(
            ProgramMilestone.dedupe_key == "clock:stop_loss_recheck:verdict").first()
        assert verdict.category == "verdict"
        assert "PASS" in verdict.title  # avg -8% clears the -10% bar


class TestRoutes:
    def test_create_list_delete_roundtrip(self, db_session):
        from backend.routes.admin import (
            MilestoneCreate, create_program_milestone,
            delete_program_milestone, list_program_milestones,
        )
        created = asyncio.run(create_program_milestone(
            body=MilestoneCreate(title="Owner note", detail="ctx"),
            current_user=None, db=db_session))
        assert created["source"] == "owner"
        # Direct invocation bypasses FastAPI's Query-default resolution, so
        # pass category/limit explicitly.
        listed = asyncio.run(list_program_milestones(
            current_user=None, db=db_session, category=None, limit=200))
        assert [r["title"] for r in listed] == ["Owner note"]
        asyncio.run(delete_program_milestone(
            milestone_id=created["id"], current_user=None, db=db_session))
        assert db_session.query(ProgramMilestone).count() == 0

    def test_list_newest_first_and_category_filter(self, db_session):
        add_milestone(db_session, title="old", category="fix",
                      occurred_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
        add_milestone(db_session, title="new", category="infra",
                      occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc))
        from backend.routes.admin import list_program_milestones
        listed = asyncio.run(list_program_milestones(
            current_user=None, db=db_session, category=None, limit=200))
        assert [r["title"] for r in listed] == ["new", "old"]
        only_fix = asyncio.run(list_program_milestones(
            current_user=None, db=db_session, category="fix", limit=200))
        assert [r["title"] for r in only_fix] == ["old"]

    def test_create_rejects_blank_title(self, db_session):
        from backend.routes.admin import MilestoneCreate, create_program_milestone
        with pytest.raises(HTTPException) as exc:
            asyncio.run(create_program_milestone(
                body=MilestoneCreate(title="   "),
                current_user=None, db=db_session))
        assert exc.value.status_code == 400

    def test_delete_missing_404s(self, db_session):
        from backend.routes.admin import delete_program_milestone
        with pytest.raises(HTTPException) as exc:
            asyncio.run(delete_program_milestone(
                milestone_id=999999, current_user=None, db=db_session))
        assert exc.value.status_code == 404


class TestMilestonePing:
    """2026-09-01 email demotion: fresh auto milestones fire ONE owner push
    (create_notification, kind=program_milestone) — the exception ping that
    replaced the weekly A/B email ritual. Empty pass = silence."""

    def _rows(self, db, *titles):
        return [add_milestone(db, title=t, source="auto",
                              dedupe_key=f"ping-test:{t}") for t in titles]

    def test_batch_ping_fires_once_with_all_titles(self, db_session):
        from unittest.mock import patch
        from backend.milestones import _notify_new_milestones
        rows = self._rows(db_session, "Gate A met", "Verdict B fired")
        with patch("backend.email_utils.create_notification") as ping:
            _notify_new_milestones(rows)
        ping.assert_called_once()
        args, kwargs = ping.call_args
        assert args[0] == 1  # owner
        assert kwargs["kind"] == "program_milestone"
        assert "Gate A met" in kwargs["body"]
        assert "Verdict B fired" in kwargs["body"]
        assert "2 new" in kwargs["title"]

    def test_single_row_ping_names_the_milestone(self, db_session):
        from unittest.mock import patch
        from backend.milestones import _notify_new_milestones
        rows = self._rows(db_session, "Stop-loss verdict: PASS")
        with patch("backend.email_utils.create_notification") as ping:
            _notify_new_milestones(rows)
        assert "Stop-loss verdict: PASS" in ping.call_args.kwargs["title"]

    def test_empty_pass_is_silent(self, db_session):
        from unittest.mock import patch
        from backend.milestones import _notify_new_milestones
        with patch("backend.email_utils.create_notification") as ping:
            _notify_new_milestones([])
        ping.assert_not_called()

    def test_ping_failure_never_raises(self, db_session):
        from unittest.mock import patch
        from backend.milestones import _notify_new_milestones
        rows = self._rows(db_session, "boom target")
        with patch("backend.email_utils.create_notification",
                   side_effect=RuntimeError("push down")):
            _notify_new_milestones(rows)  # must not raise


class TestSufficiencyMetric:
    """2026-09-02: the generic >=5-closed-sells metric appended to every arm
    is data sufficiency for the weekly A/B rule, NOT a promotion gate. Its
    2026-09-01 rows ("'closed sells' accrual met") read as gate events and
    triggered a false verdict read. The row still lands in the ledger (with
    an honest title, same dedupe key so history never re-fires) but never
    pings."""

    def _five_sells(self, db, arm):
        for i in range(5):
            db.add(ShadowTrade(
                shadow_strategy_id=arm.id, ticker=f"S{i}", action="SELL",
                shares=1.0, price=10.0, total_value=10.0, realized_gain=0.0,
                executed_at=T0))
        db.commit()

    def test_gates_payload_tags_sufficiency_kind(self, db_session):
        from backend.routes.admin import compute_experiment_gates
        from backend.milestones import SUFFICIENCY_LABEL
        _arm(db_session, "shadow_cs_window14")
        arm = next(a for a in compute_experiment_gates(db_session)["arms"]
                   if a["name"] == "shadow_cs_window14")
        kinds = {m["label"]: m.get("kind") for m in arm["gate_metrics"]}
        assert kinds[SUFFICIENCY_LABEL] == "sufficiency"
        # The arm's own pre-registered metric carries no sufficiency tag.
        assert kinds["CS buys in 8-14d band"] is None

    def test_sufficiency_row_keeps_key_but_says_not_a_gate(self, db_session):
        from backend.milestones import SUFFICIENCY_LABEL, is_sufficiency_row
        a = _arm(db_session, "shadow_cs_window14")
        self._five_sells(db_session, a)
        record_auto_milestones(db_session)
        key = f"gate:shadow_cs_window14:{SUFFICIENCY_LABEL}:target"
        row = db_session.query(ProgramMilestone).filter(
            ProgramMilestone.dedupe_key == key).first()
        assert row is not None, "sufficiency threshold still lands in the ledger"
        assert "not a promotion gate" in row.title
        assert "accrual met" not in row.title
        assert is_sufficiency_row(row)
        # The arm's real gate (0/5 band buys) did NOT fire.
        assert not any("8-14d" in (k or "") for k in _noncal_keys(db_session))

    def test_sufficiency_row_never_pings(self, db_session, monkeypatch):
        from unittest.mock import patch
        from backend import milestones as ms
        a = _arm(db_session, "shadow_cs_window14")
        self._five_sells(db_session, a)
        # run_milestone_pass opens its own SessionLocal — same test DB.
        with patch("backend.email_utils.create_notification") as ping:
            ms.run_milestone_pass()
        auto = db_session.query(ProgramMilestone).filter(
            ProgramMilestone.source == "auto").all()
        assert any(ms.is_sufficiency_row(r) for r in auto)
        # Only calendar rows (if any are due today) may have pinged — never
        # the sufficiency row.
        for call in ping.call_args_list:
            assert "not a promotion gate" not in call.kwargs.get("body", "")
            assert "not a promotion gate" not in call.kwargs.get("title", "")
