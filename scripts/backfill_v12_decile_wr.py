#!/usr/bin/env python3
"""Backfill ml_models.eval_decile_wr for the active v12 model (one-off).

Run on the VPS after deploy so the decile-WR pre-gate has an incumbent
baseline to compare candidates against. Without this, the first candidate
trained after deploy will hit the bootstrap path (auto-pass) and the gate
won't actually filter anything until the next training run.

Usage:
    docker exec canslim-analyzer python3 /app/scripts/backfill_v12_decile_wr.py
    # or, with a custom strategy/cutoff:
    docker exec canslim-analyzer python3 /app/scripts/backfill_v12_decile_wr.py \\
        --strategy nostate_optimized --cutoff 2026-05-01
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from datetime import datetime, timezone

from sqlalchemy import desc

from backend.database import SessionLocal, MLModel
from backend.routes.ml import (
    _compute_candidate_decile_wr,
    DECILE_WR_HOLDOUT_CUTOFF_ISO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="nostate_optimized")
    parser.add_argument(
        "--cutoff", default=DECILE_WR_HOLDOUT_CUTOFF_ISO,
        help=f"ISO date; backtests on/after are the OOS holdout (default {DECILE_WR_HOLDOUT_CUTOFF_ISO})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if eval_decile_wr is already set",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        active = db.query(MLModel).filter(
            MLModel.strategy == args.strategy,
            MLModel.status == "active",
        ).order_by(desc(MLModel.activated_at)).first()
        if active is None:
            print(f"No active model for strategy={args.strategy}", file=sys.stderr)
            sys.exit(1)
        if active.model_path is None:
            print(f"Active v{active.version} has no model_path on disk", file=sys.stderr)
            sys.exit(2)
        if active.eval_decile_wr is not None and not args.force:
            print(
                f"v{active.version} already has eval_decile_wr={active.eval_decile_wr:.4f}; "
                f"pass --force to recompute"
            )
            sys.exit(0)

        print(
            f"Computing decile WR for v{active.version} "
            f"({active.model_path}) on holdout after {args.cutoff}..."
        )
        wr = _compute_candidate_decile_wr(
            db, active.model_path, args.strategy, cutoff_iso=args.cutoff,
        )
        if wr is None:
            print(
                "Computation returned None (empty holdout / feature mismatch / load error). "
                "See logs for details.",
                file=sys.stderr,
            )
            sys.exit(3)

        active.eval_decile_wr = wr
        db.commit()
        print(f"Backfilled v{active.version}.eval_decile_wr = {wr:.4f}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
