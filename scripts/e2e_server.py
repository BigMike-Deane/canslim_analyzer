#!/usr/bin/env python3
"""Seed an isolated book and serve the app for the Playwright e2e suite.

Run by playwright.config.ts as its `webServer`. Never touches the real dev
or production database: it points DATABASE_URL at a throwaway SQLite file
BEFORE importing anything from backend/, because backend.database reads that
env var at import time.

The seeded book is the whole point. The three number-consistency bugs found
on 2026-09-09 were invisible to the existing tests because those seeded the
live cash EQUAL to the last snapshot's value, so the two ends of every window
matched by construction. Here they are deliberately different, and there is
no snapshot for today -- the pre-market condition under which the bugs
actually showed up.

`scripts/` is not copied into the Docker image, so this is dev-only by
construction.
"""

import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PORT = int(os.environ.get("E2E_PORT", "8011"))
DB_PATH = os.environ.get("E2E_DB") or os.path.join(
    tempfile.gettempdir(), "canslim_e2e.db"
)

# MUST precede any backend import.
os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("REQUIRE_AUTH", "true")
os.environ.setdefault("DISABLE_SCHEDULER", "true")
os.environ.setdefault("JWT_SECRET_KEY", "e2e-only-not-a-real-secret")

# Deterministic fixture. current_cash is the LIVE book (no positions, so
# get_portfolio_value() == current_cash exactly); the newest snapshot is
# deliberately elsewhere.
STARTING_CASH = 10000.0
LIVE_CASH = 10500.0
ANCHOR_VALUE = 10000.0      # 30 days back: the window anchor
STALE_SNAPSHOT = 11000.0    # yesterday: the last WRITTEN snapshot
USER_ID = 1


def seed():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    from backend.database import (
        init_db, SessionLocal, User,
        AIPortfolioConfig, AIPortfolioSnapshot, MarketSnapshot,
    )
    from backend.auth import create_access_token

    init_db()
    db = SessionLocal()
    try:
        db.add(User(
            id=USER_ID, email="e2e@test.local", display_name="E2E",
            is_active=True, is_admin=True, hashed_password="",
        ))
        db.add(AIPortfolioConfig(
            user_id=USER_ID, starting_cash=STARTING_CASH,
            current_cash=LIVE_CASH, is_active=True,
        ))

        def snap(days_ago, value):
            d = date.today() - timedelta(days=days_ago)
            db.add(AIPortfolioSnapshot(
                user_id=USER_ID,
                timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
                date=d, total_value=value, cash=value, positions_value=0.0,
                positions_count=0, total_return=value - STARTING_CASH,
                total_return_pct=((value - STARTING_CASH) / STARTING_CASH) * 100,
            ))
            # SPY series so the chart's benchmark normalizes instead of
            # rendering nulls.
            if not db.query(MarketSnapshot).filter_by(date=d).first():
                db.add(MarketSnapshot(date=d, spy_price=500.0 + days_ago))

        snap(30, ANCHOR_VALUE)
        snap(1, STALE_SNAPSHOT)
        # No snapshot today: the pre-market condition.
        db.commit()
    finally:
        db.close()

    token = create_access_token({"sub": str(USER_ID)})
    out = REPO / "frontend" / "e2e" / ".auth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "token": token,
        "live_cash": LIVE_CASH,
        "anchor_value": ANCHOR_VALUE,
        "stale_snapshot": STALE_SNAPSHOT,
    }))
    print(f"[e2e] seeded {DB_PATH}; auth written to {out}", flush=True)


if __name__ == "__main__":
    seed()
    import uvicorn
    from backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
