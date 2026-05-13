"""Tests for backend.shadow_strategy_sync.

Step 4 of the shadow paper-trading harness (design in
docs/shadow-paper-trading-design.md). Reconciles the
``shadow_strategy_profiles:`` YAML section against the ``shadow_strategies``
table at scheduler boot.

Pattern mirrors tests/test_strategy_ab_eval_shadow.py: in-memory SQLite via
init_db + SessionLocal, with delete-cycle fixture so each test starts clean.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, ShadowStrategy, ShadowTrade,
)
from backend import shadow_strategy_sync as sss


# Realistic minimal parent profile — the sync helper freezes whatever dict
# is associated with parent_strategy in strategy_profiles. Two parents so
# we can exercise multi-stack reconcile.
_PARENT_PROFILES = {
    "nostate_cs_bear": {
        "label": "NoState + CS Bear",
        "min_score": 73,
        "max_positions": 8,
        "stop_loss_pct": 7.0,
    },
    "nostate_optimized": {
        "label": "NoState Optimized",
        "min_score": 72,
        "max_positions": 8,
        "stop_loss_pct": 7.0,
    },
}


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(ShadowTrade).delete()
    db.query(ShadowStrategy).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(ShadowTrade).delete()
        db.query(ShadowStrategy).delete()
        db.commit()
        db.close()


@pytest.fixture
def patch_yaml(monkeypatch):
    """Helper to drive the sync from a per-test YAML state.

    Returns a function ``set_yaml(shadow_section, parent_section=None)`` that
    rewires the module's two loader helpers so the next sync call sees the
    desired YAML state. parent_section defaults to _PARENT_PROFILES.
    """

    def _set_yaml(shadow_section, parent_section=None):
        parents = parent_section if parent_section is not None else _PARENT_PROFILES
        monkeypatch.setattr(sss, "_load_yaml_section", lambda: shadow_section or {})
        monkeypatch.setattr(sss, "_load_strategy_profiles", lambda: parents)

    return _set_yaml


def _baseline_yaml(parent="nostate_cs_bear", **overrides):
    entry = {
        "parent_strategy": parent,
        "starting_value": 25000,
        "description": "Forward-only parity check against live cs_bear baseline",
        "scorer_overrides": {},
    }
    entry.update(overrides)
    return {"shadow_baseline": entry}


# ── Phase 1: insert ────────────────────────────────────────────────────────

class TestInsert:
    def test_empty_yaml_empty_db_is_noop(self, db_session, patch_yaml):
        patch_yaml({})
        result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert result == {"inserted": [], "updated": [], "archived": [], "skipped": []}
        assert db_session.query(ShadowStrategy).count() == 0

    def test_one_yaml_entry_empty_db_inserts_one(self, db_session, patch_yaml):
        patch_yaml(_baseline_yaml())
        result = sss.sync_shadow_strategies_from_yaml(db_session)

        assert result["inserted"] == ["shadow_baseline"]
        assert result["updated"] == []
        assert result["archived"] == []

        rows = db_session.query(ShadowStrategy).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.name == "shadow_baseline"
        assert row.parent_strategy == "nostate_cs_bear"
        assert row.starting_value == 25000.0
        assert row.scorer_overrides == {}
        assert row.archived_at is None
        assert row.activated_at is not None
        # config_snapshot must be a frozen copy of the parent profile
        assert row.config_snapshot["min_score"] == 73
        assert row.config_snapshot["label"] == "NoState + CS Bear"

    def test_multiple_yaml_entries_all_inserted(self, db_session, patch_yaml):
        yaml = _baseline_yaml()
        yaml["candidate_a"] = {
            "parent_strategy": "nostate_optimized",
            "starting_value": 50000,
            "description": "Higher allocation candidate",
            "scorer_overrides": {"l_weight": 1.2},
        }
        patch_yaml(yaml)
        result = sss.sync_shadow_strategies_from_yaml(db_session)

        assert sorted(result["inserted"]) == ["candidate_a", "shadow_baseline"]
        candidate = db_session.query(ShadowStrategy).filter_by(name="candidate_a").one()
        assert candidate.parent_strategy == "nostate_optimized"
        assert candidate.starting_value == 50000.0
        assert candidate.scorer_overrides == {"l_weight": 1.2}
        assert candidate.config_snapshot["label"] == "NoState Optimized"


# ── Phase 2: update / no-op ─────────────────────────────────────────────────

class TestUpdate:
    def test_no_changes_is_skip(self, db_session, patch_yaml):
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)

        # Second sync with identical YAML → all entries skipped, no churn.
        result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert result == {
            "inserted": [], "updated": [],
            "archived": [], "skipped": ["shadow_baseline"],
        }
        assert db_session.query(ShadowStrategy).count() == 1

    def test_mutable_field_update(self, db_session, patch_yaml):
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)

        # Edit the description + scorer_overrides + starting_value.
        patch_yaml(_baseline_yaml(
            description="Updated copy",
            scorer_overrides={"experimental_weight": 0.5},
            starting_value=30000,
        ))
        result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert result["updated"] == ["shadow_baseline"]
        assert result["skipped"] == []

        row = db_session.query(ShadowStrategy).one()
        assert row.description == "Updated copy"
        assert row.scorer_overrides == {"experimental_weight": 0.5}
        assert row.starting_value == 30000.0

    def test_immutable_fields_do_not_change(self, db_session, patch_yaml, caplog):
        """parent_strategy + config_snapshot are frozen at first INSERT.

        Even if YAML drifts the parent reference, the DB row keeps the
        original snapshot and we log a WARNING. (Operator can archive +
        re-add to start a fresh run with the new parent.)
        """
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)
        first_snapshot = dict(db_session.query(ShadowStrategy).one().config_snapshot)

        # Drift the YAML to a different parent.
        patch_yaml(_baseline_yaml(parent="nostate_optimized"))
        with caplog.at_level("WARNING"):
            sss.sync_shadow_strategies_from_yaml(db_session)

        row = db_session.query(ShadowStrategy).one()
        assert row.parent_strategy == "nostate_cs_bear"  # unchanged
        assert row.config_snapshot == first_snapshot      # unchanged
        assert any("parent_strategy in YAML" in m for m in caplog.messages)


# ── Phase 3: archive ────────────────────────────────────────────────────────

class TestArchive:
    def test_yaml_drops_a_name_archives_row(self, db_session, patch_yaml):
        # First seed two entries.
        yaml = _baseline_yaml()
        yaml["candidate_a"] = {
            "parent_strategy": "nostate_optimized",
            "starting_value": 25000,
            "scorer_overrides": {},
        }
        patch_yaml(yaml)
        sss.sync_shadow_strategies_from_yaml(db_session)
        assert db_session.query(ShadowStrategy).filter_by(archived_at=None).count() == 2

        # Now drop candidate_a from YAML.
        patch_yaml(_baseline_yaml())
        result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert result["archived"] == ["candidate_a"]

        archived = db_session.query(ShadowStrategy).filter_by(name="candidate_a").one()
        assert archived.archived_at is not None
        active = db_session.query(ShadowStrategy).filter_by(name="shadow_baseline").one()
        assert active.archived_at is None

    def test_archive_preserves_existing_trades(self, db_session, patch_yaml):
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)
        row = db_session.query(ShadowStrategy).one()

        # Drop a couple of trades against it.
        for i in range(3):
            db_session.add(ShadowTrade(
                shadow_strategy_id=row.id,
                ticker="TEST",
                action="BUY",
                shares=10.0,
                price=10.0 + i,
                total_value=100.0,
                executed_at=datetime.now(timezone.utc),
            ))
        db_session.commit()
        assert db_session.query(ShadowTrade).count() == 3

        # Drop the stack from YAML — sync should archive without touching trades.
        patch_yaml({})
        sss.sync_shadow_strategies_from_yaml(db_session)
        assert db_session.query(ShadowTrade).count() == 3
        archived = db_session.query(ShadowStrategy).one()
        assert archived.archived_at is not None

    def test_remove_then_readd_reactivates_same_row(self, db_session, patch_yaml):
        """Re-adding a previously-archived name reactivates the same row.

        The DB schema enforces UNIQUE(name) and admin.py readers do
        name-based lookups, so multiple rows per name are unsafe.
        Reactivate-in-place keeps existing trades attached (FK to id) and
        flips archived_at→None. Tracked as ``inserted`` in the result so
        operators see the re-add in the boot log.
        """
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)
        first_id = db_session.query(ShadowStrategy).one().id

        # Drop → archive
        patch_yaml({})
        sss.sync_shadow_strategies_from_yaml(db_session)
        archived_row = db_session.query(ShadowStrategy).one()
        assert archived_row.archived_at is not None
        archived_at_first_archive = archived_row.archived_at

        # Re-add same name
        patch_yaml(_baseline_yaml(description="post-archive resume"))
        result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert result["inserted"] == ["shadow_baseline"]

        rows = db_session.query(ShadowStrategy).filter_by(name="shadow_baseline").all()
        assert len(rows) == 1
        row = rows[0]
        assert row.id == first_id
        assert row.archived_at is None
        # activated_at re-stamped to the re-add moment (after the prior archive).
        assert row.activated_at >= archived_at_first_archive
        assert row.description == "post-archive resume"

    def test_readd_preserves_trades_attached_to_archived_row(self, db_session, patch_yaml):
        """Trades that accumulated under the archived window survive reactivation.

        They FK to id, not to archived_at, so reactivating-in-place keeps
        the full trade history visible to /strategy-ab-eval-trades.
        """
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)
        row = db_session.query(ShadowStrategy).one()

        for i in range(2):
            db_session.add(ShadowTrade(
                shadow_strategy_id=row.id,
                ticker="TEST",
                action="BUY",
                shares=10.0,
                price=10.0 + i,
                total_value=100.0,
                executed_at=datetime.now(timezone.utc) - timedelta(days=5),
            ))
        db_session.commit()

        # Archive then re-add
        patch_yaml({})
        sss.sync_shadow_strategies_from_yaml(db_session)
        patch_yaml(_baseline_yaml())
        sss.sync_shadow_strategies_from_yaml(db_session)

        # Trades must still be there, attached to the (now active) row.
        assert db_session.query(ShadowTrade).count() == 2


# ── Phase 4: validation + transaction safety ────────────────────────────────

class TestValidation:
    def test_unknown_parent_strategy_raises(self, db_session, patch_yaml):
        patch_yaml(_baseline_yaml(parent="does_not_exist"))
        with pytest.raises(ValueError, match="does_not_exist"):
            sss.sync_shadow_strategies_from_yaml(db_session)
        # No partial inserts.
        assert db_session.query(ShadowStrategy).count() == 0

    def test_missing_parent_strategy_raises(self, db_session, patch_yaml):
        patch_yaml({"shadow_baseline": {"starting_value": 25000}})
        with pytest.raises(ValueError, match="parent_strategy is required"):
            sss.sync_shadow_strategies_from_yaml(db_session)
        assert db_session.query(ShadowStrategy).count() == 0

    def test_partial_failure_rolls_back(self, db_session, patch_yaml):
        """If validation fails on the SECOND entry, the first must not persist.

        Pre-flight validation runs before any INSERT, so even though the
        first entry would have been valid, the bad second entry triggers a
        ValueError and the table stays empty.
        """
        yaml = _baseline_yaml()
        yaml["bad_one"] = {"parent_strategy": "ghost_strategy", "starting_value": 10000}
        patch_yaml(yaml)
        with pytest.raises(ValueError, match="ghost_strategy"):
            sss.sync_shadow_strategies_from_yaml(db_session)
        assert db_session.query(ShadowStrategy).count() == 0

    def test_already_archived_row_skipped_on_subsequent_sync(self, db_session, patch_yaml):
        """Idempotency: when a name has been dropped from YAML and the matching
        row is already archived, a subsequent sync must NOT bump archived_at.
        Pins the line-181 continue branch — once-archived stays archived with
        its original timestamp."""
        # Seed two entries, drop one, archive it.
        yaml = _baseline_yaml()
        yaml["candidate_a"] = {
            "parent_strategy": "nostate_optimized",
            "starting_value": 25000,
            "scorer_overrides": {},
        }
        patch_yaml(yaml)
        sss.sync_shadow_strategies_from_yaml(db_session)
        patch_yaml(_baseline_yaml())  # drop candidate_a
        first_result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert first_result["archived"] == ["candidate_a"]
        archived_row = db_session.query(ShadowStrategy).filter_by(name="candidate_a").one()
        original_archived_at = archived_row.archived_at
        assert original_archived_at is not None

        # Run sync AGAIN with the same dropped YAML. The already-archived row
        # must hit the early-continue at line 181 and NOT be re-archived.
        second_result = sss.sync_shadow_strategies_from_yaml(db_session)
        assert second_result["archived"] == []
        db_session.refresh(archived_row)
        assert archived_row.archived_at == original_archived_at

    def test_commit_failure_rolls_back_and_reraises(self, db_session, patch_yaml, monkeypatch):
        """When `db.commit()` raises mid-sync, the except branch rolls back
        and re-raises. Pins the lines-186-188 rollback contract — any
        inserts/mutations in the current transaction must not persist."""
        patch_yaml(_baseline_yaml())
        rollback_called = {"count": 0}
        original_rollback = db_session.rollback

        def tracking_rollback():
            rollback_called["count"] += 1
            return original_rollback()

        monkeypatch.setattr(db_session, "rollback", tracking_rollback)
        monkeypatch.setattr(db_session, "commit",
                            lambda: (_ for _ in ()).throw(RuntimeError("commit boom")))

        with pytest.raises(RuntimeError, match="commit boom"):
            sss.sync_shadow_strategies_from_yaml(db_session)
        assert rollback_called["count"] >= 1
        # Pre-flight INSERTs were rolled back — no rows persisted.
        assert db_session.query(ShadowStrategy).count() == 0
