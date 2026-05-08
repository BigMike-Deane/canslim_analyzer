"""Register the shadow_baseline ShadowStrategy.

Idempotent registration of a no-op shadow stack that mirrors nostate_cs_bear
exactly (zero scorer_overrides). The forward-only ShadowTrade rows it emits
serve as the parity-test anchor for future non-baseline shadow stacks.

Usage (locally):
    CANSLIM_ENV=development python3 scripts/register_shadow_baseline.py

Usage (VPS):
    docker exec canslim-analyzer python3 -c "
    import subprocess; subprocess.run(['python3', '/app/scripts/register_shadow_baseline.py'])
    "
    # Note: scripts/ is NOT copied into the Docker image (CLAUDE.md). Use docker
    # cp first to ship this file in, OR run an inline equivalent:
    #     docker exec canslim-analyzer python3 -c "
    #     from backend.database import SessionLocal, ShadowStrategy
    #     from datetime import datetime, timezone
    #     db = SessionLocal()
    #     existing = db.query(ShadowStrategy).filter_by(name='shadow_baseline').first()
    #     if existing is None:
    #         db.add(ShadowStrategy(
    #             name='shadow_baseline',
    #             parent_strategy='nostate_cs_bear',
    #             config_snapshot={'strategy': 'nostate_cs_bear'},
    #             scorer_overrides={},
    #             starting_value=25000.0,
    #             description='No-op baseline; mirrors nostate_cs_bear for parity testing.',
    #         ))
    #         db.commit()
    #         print('shadow_baseline registered')
    #     else:
    #         print(f'shadow_baseline already exists (id={existing.id})')
    #     db.close()
    #     "
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make backend importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("CANSLIM_ENV", os.environ.get("CANSLIM_ENV", "development"))

from backend.database import SessionLocal, ShadowStrategy, init_db


SHADOW_BASELINE_CONFIG = {
    "name": "shadow_baseline",
    "parent_strategy": "nostate_cs_bear",
    "config_snapshot": {"strategy": "nostate_cs_bear"},
    "scorer_overrides": {},
    "starting_value": 25000.0,
    "description": (
        "No-op baseline. Mirrors nostate_cs_bear with zero scorer_overrides. "
        "Forward-only ShadowTrade rows it emits anchor parity comparison "
        "for future non-baseline shadow stacks. See "
        "docs/shadow-paper-trading-design.md."
    ),
}


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(ShadowStrategy).filter(
            ShadowStrategy.name == SHADOW_BASELINE_CONFIG["name"]
        ).first()
        if existing is not None:
            archived_note = " (archived)" if existing.archived_at else ""
            print(
                f"shadow_baseline already exists "
                f"(id={existing.id}, parent={existing.parent_strategy}{archived_note})"
            )
            return 0

        ss = ShadowStrategy(**SHADOW_BASELINE_CONFIG)
        db.add(ss)
        db.commit()
        db.refresh(ss)
        print(f"shadow_baseline registered (id={ss.id}, parent={ss.parent_strategy})")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
