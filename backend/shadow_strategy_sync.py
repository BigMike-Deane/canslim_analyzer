"""Shadow strategy provisioning sync.

Step 4 of the shadow paper-trading harness (design in
docs/shadow-paper-trading-design.md). Reconciles the
``shadow_strategy_profiles:`` YAML section against the ``shadow_strategies``
table at scheduler boot.

Operator workflow: edit ``config/default.yaml`` (or environment override),
deploy, and the candidate stack lights up in ``/admin/shadow-strategies`` on
next boot — no raw SQL required.

Reconcile semantics:
  - YAML name not in DB → INSERT (config_snapshot frozen from CURRENT YAML)
  - YAML name in active DB row → UPDATE mutable fields only (description,
    scorer_overrides, starting_value). config_snapshot and parent_strategy
    are immutable: changing them mid-stream would silently invalidate the
    forward-only A/B telemetry by altering the comparison baseline.
  - YAML name in DB but archived (archived_at != None) → REACTIVATE that
    same row (archived_at = None, activated_at = now, mutable fields
    refreshed). The row keeps its id, so existing shadow_trades stay
    attached. We reactivate-in-place rather than insert-fresh because the
    DB schema enforces UNIQUE(name) and admin.py readers do name-based
    lookups that would be ambiguous with multiple rows per name. Operators
    who want a clean forward-only window after a long archive should rename
    the stack in YAML.
  - DB name not in YAML and currently active → SOFT-ARCHIVE (archived_at = now).
    Existing shadow_trades rows are NOT deleted.

Validation: every parent_strategy must resolve in the running
``strategy_profiles:`` config. A bad parent raises ValueError and the entire
reconcile transaction is rolled back — partial inserts are never persisted.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from backend.database import ShadowStrategy

logger = logging.getLogger(__name__)


# Mutable fields that can change between deploys without invalidating the
# forward-only A/B telemetry. config_snapshot and parent_strategy are
# deliberately excluded.
_MUTABLE_FIELDS = ("description", "scorer_overrides", "starting_value")


def _load_yaml_section() -> Dict[str, Any]:
    """Read ``shadow_strategy_profiles:`` from the merged YAML config."""
    from config_loader import config as yaml_config
    return yaml_config.get("shadow_strategy_profiles", default={}) or {}


def _load_strategy_profiles() -> Dict[str, Any]:
    """Read ``strategy_profiles:`` for parent validation + snapshot freezing."""
    from config_loader import config as yaml_config
    return yaml_config.get("strategy_profiles", default={}) or {}


def sync_shadow_strategies_from_yaml(db: Session) -> Dict[str, List[str]]:
    """Reconcile shadow_strategy_profiles YAML against shadow_strategies DB.

    Single transaction: all reconcile actions commit together. On any error
    (e.g. unknown parent_strategy) the whole transaction is rolled back so
    we never leave the table in a half-applied state.

    Returns a dict with keys ``inserted``, ``updated``, ``archived``,
    ``skipped`` — each a list of shadow-strategy names. Caller logs the
    counts; the returned dict is also useful for tests.
    """
    yaml_entries = _load_yaml_section()
    parent_profiles = _load_strategy_profiles()

    # Validate parent_strategy refs up-front so we can fail before mutating.
    for name, entry in yaml_entries.items():
        parent = entry.get("parent_strategy") if isinstance(entry, dict) else None
        if not parent:
            raise ValueError(
                f"shadow_strategy_profiles.{name}: parent_strategy is required"
            )
        if parent not in parent_profiles:
            raise ValueError(
                f"shadow_strategy_profiles.{name}: parent_strategy '{parent}' "
                f"not found in strategy_profiles. Add the parent first or fix "
                f"the typo before re-deploying."
            )

    result: Dict[str, List[str]] = {
        "inserted": [],
        "updated": [],
        "archived": [],
        "skipped": [],
    }

    try:
        # Index ALL existing rows by name (active + archived). The schema
        # enforces UNIQUE(name) so there is at most one row per name.
        existing_by_name = {
            row.name: row
            for row in db.query(ShadowStrategy).all()
        }

        now = datetime.now(timezone.utc)

        # Phase 1 — INSERT/REACTIVATE/UPDATE for every YAML entry.
        for name, entry in yaml_entries.items():
            parent = entry["parent_strategy"]
            description = entry.get("description")
            scorer_overrides = entry.get("scorer_overrides") or {}
            starting_value = float(entry.get("starting_value", 25000.0))

            row = existing_by_name.get(name)
            if row is None:
                # Fresh insert. Snapshot the parent at this moment.
                snapshot = dict(parent_profiles.get(parent, {}))
                db.add(ShadowStrategy(
                    name=name,
                    parent_strategy=parent,
                    config_snapshot=snapshot,
                    scorer_overrides=scorer_overrides,
                    description=description,
                    starting_value=starting_value,
                    created_at=now,
                    activated_at=now,
                    archived_at=None,
                ))
                result["inserted"].append(name)
                continue

            if row.archived_at is not None:
                # Reactivate-in-place. Refresh mutable fields + re-stamp
                # activated_at so the listing endpoint shows a sensible date.
                # config_snapshot/parent_strategy stay frozen — that's the
                # whole point of the snapshot (forward-only telemetry).
                row.archived_at = None
                row.activated_at = now
                row.description = description
                row.scorer_overrides = scorer_overrides
                row.starting_value = starting_value
                result["inserted"].append(name)
                continue

            # Active row exists — drift-check on immutable fields, update mutables.
            if row.parent_strategy != parent:
                logger.warning(
                    "shadow_strategy_profiles.%s: parent_strategy in YAML (%s) "
                    "differs from frozen DB row (%s). Ignoring drift to keep "
                    "forward-only telemetry comparable; archive + re-add the "
                    "stack to start a fresh run.",
                    name, parent, row.parent_strategy,
                )

            changed = False
            if row.description != description:
                row.description = description
                changed = True
            # scorer_overrides is JSON-typed; compare structurally.
            if (row.scorer_overrides or {}) != (scorer_overrides or {}):
                row.scorer_overrides = scorer_overrides
                changed = True
            if float(row.starting_value or 0.0) != starting_value:
                row.starting_value = starting_value
                changed = True

            if changed:
                result["updated"].append(name)
            else:
                result["skipped"].append(name)

        # Phase 2 — ARCHIVE active rows whose name is no longer in YAML.
        yaml_names = set(yaml_entries.keys())
        for name, row in existing_by_name.items():
            if name in yaml_names:
                continue
            if row.archived_at is not None:
                continue  # Already archived, no-op.
            row.archived_at = now
            result["archived"].append(name)

        db.commit()
    except Exception:
        db.rollback()
        raise

    return result
