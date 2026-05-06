"""
Pyramid action label sync tests.

Pins the contract that BOTH the live AI trader (`backend/ai_trader.py`) and
the backtester (`backend/backtester.py`) write pyramid additions with
`action="PYRAMID"` — the canonical value documented in the
`AIPortfolioTrade.action` enum at `backend/database.py:1036`.

Background (May 5, 2026): A pre-existing sync drift had the live trader
writing pyramids as `action="BUY" + reason="PYRAMID:..."` while the
backtester wrote `action="PYRAMID"`. The cs_bear health-check audit
flagged BUYs missing ml_confidence as a possible ML-gate leak; they were
actually un-relabeled pyramid rows. We chose Option A (flip live to match
backtester + DB schema) because:

  1. TradeJournal frontend already filters by `action === 'PYRAMID'` —
     live trades were silently invisible under that filter.
  2. ml_backfill could mistakenly anchor an ml_predictions row to a
     pyramid trade (pyramids aren't ML-gated).
  3. The DB schema docstring already lists PYRAMID as canonical.

Historical AIPortfolioTrade rows (pre-fix) intentionally retain the old
shape (`action="BUY" + reason="PYRAMID:..."`). Forward-only fix.

If a future maintainer wants to switch back to Option B (keep `action="BUY"`
for live pyramids), revert the call site in evaluate_pyramids and update
this test to assert `action="BUY"` + reason starts with "PYRAMID:".
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from unittest.mock import patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fresh_session():
    """In-memory SQLite session with backend.database schema applied."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.database import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


# ── 1. Source-level static checks (cheap, env-independent) ────────────────────


class TestPyramidActionInSource:
    """Static source-level checks. These catch reverts even before the
    function runs, and don't depend on DB/test fixtures."""

    def test_ai_trader_evaluate_pyramids_writes_pyramid_action(self):
        """The execute_trade call inside ai_trader's evaluate_pyramids
        section must use action="PYRAMID", not action="BUY"."""
        from backend import ai_trader

        src = inspect.getsource(ai_trader)
        # Find the pyramid block. The execute_trade call lives between
        # `evaluate_pyramids(db, user_id=user_id)` and the matching commit.
        marker = "pyramids = evaluate_pyramids(db, user_id=user_id)"
        assert marker in src, (
            "Couldn't locate the pyramid evaluation block in ai_trader.py — "
            "test needs updating if the structure changed."
        )
        idx = src.index(marker)
        # Look at the next ~2k chars (the pyramid execution loop). Strip
        # comments so the assertion isn't fooled by documentation that
        # mentions the old shape.
        block_lines = src[idx : idx + 2000].splitlines()
        block_code = "\n".join(
            line for line in block_lines if not line.lstrip().startswith("#")
        )
        assert 'action="PYRAMID"' in block_code, (
            "Live AI trader must write action=\"PYRAMID\" for pyramid additions "
            "(SYNC with backtester.py). If you intentionally changed this, "
            "update tests/test_pyramid_action_sync.py."
        )
        assert 'action="BUY"' not in block_code, (
            "Pyramid block in ai_trader.py contains action=\"BUY\" — that's "
            "the pre-2026-05-05 drifted shape. Use action=\"PYRAMID\"."
        )

    def test_backtester_pyramid_simulated_trade_uses_pyramid_action(self):
        """Backtester's _evaluate_pyramids and scale-in branches must
        produce SimulatedTrade(action="PYRAMID")."""
        from backend.backtester import BacktestEngine

        src = inspect.getsource(BacktestEngine._evaluate_pyramids)
        assert 'action="PYRAMID"' in src, (
            "Backtester._evaluate_pyramids must emit action=\"PYRAMID\" trades."
        )
        # Both code paths inside (scale-in pullback + standard O'Neil add)
        # must use PYRAMID. Two occurrences are expected in current source.
        assert src.count('action="PYRAMID"') >= 2, (
            "Backtester._evaluate_pyramids has multiple branches (scale-in "
            "pullback + standard O'Neil add). All must emit PYRAMID."
        )


# ── 2. Behavioral check: execute_trade writes the right label ─────────────────


class TestExecuteTradeWritesPyramidAction:
    """Round-trip: pass action="PYRAMID" through execute_trade and verify
    the persisted AIPortfolioTrade row carries that label end-to-end.
    Guards against any future re-mapping inside execute_trade itself."""

    def test_execute_trade_persists_pyramid_action_label(self):
        from backend.ai_trader import execute_trade
        from backend.database import AIPortfolioTrade

        db = _fresh_session()
        try:
            # Patch the webhook helper so this test doesn't try to make a
            # network call. We don't care about its arguments here — that's
            # covered by test_pyramid_webhook_relabels_to_buy below.
            with patch("backend.email_utils.send_trade_webhook", return_value=True):
                execute_trade(
                    db=db,
                    ticker="TEST",
                    action="PYRAMID",
                    shares=10.0,
                    price=50.0,
                    reason="PYRAMID: breakout +3%",
                    score=80.0,
                    growth_score=None,
                    is_growth_stock=False,
                    is_paper=False,
                    user_id=1,
                )
            db.commit()

            row = db.query(AIPortfolioTrade).filter_by(ticker="TEST").one()
            assert row.action == "PYRAMID", (
                f"Persisted action was {row.action!r}, expected 'PYRAMID'. "
                "execute_trade must NOT remap pyramid additions to BUY."
            )
            assert row.reason.startswith("PYRAMID:"), (
                "Pyramid trades must keep the reason prefix 'PYRAMID:' for "
                "downstream consumers that legacy-filter by reason."
            )
            # Cost basis should be unset (pyramids don't realize anything).
            assert row.realized_gain is None
        finally:
            db.close()

    def test_pyramid_webhook_relabels_to_buy_for_wire(self):
        """The ntfy template only knows BUY/SELL. execute_trade must
        re-label PYRAMID -> BUY on the webhook call so the existing
        notification template renders. The reason string ('PYRAMID: ...')
        carries the human-visible distinction."""
        from backend.ai_trader import execute_trade

        db = _fresh_session()
        try:
            with patch("backend.email_utils.send_trade_webhook") as mock_webhook:
                execute_trade(
                    db=db,
                    ticker="TEST2",
                    action="PYRAMID",
                    shares=5.0,
                    price=100.0,
                    reason="PYRAMID: vol spike",
                    score=75.0,
                    is_paper=False,
                    user_id=1,
                )
                db.commit()

            assert mock_webhook.called, (
                "execute_trade should fire send_trade_webhook for PYRAMID "
                "(via the non-SELL branch)."
            )
            args, kwargs = mock_webhook.call_args
            # signature: send_trade_webhook(ticker, action, shares, price, reason, ...)
            wire_action = args[1] if len(args) >= 2 else kwargs.get("action")
            assert wire_action == "BUY", (
                f"Webhook wire_action was {wire_action!r}; expected 'BUY' "
                "(PYRAMID is re-labeled for the ntfy template)."
            )
        finally:
            db.close()


# ── 3. ml_backfill defends against legacy pyramid rows anchoring outcomes ─────


class TestMLBackfillExcludesPyramidAnchors:
    """ml_backfill.backfill_actual_outcomes pairs ml_predictions with the
    matching BUY trade. Pyramids are not ML-gated — they must NEVER be
    matched. With Option A, live pyramids carry action="PYRAMID" so they
    auto-skip the action filter. Legacy rows (action="BUY" + reason
    starting with "PYRAMID:") are excluded by the SQL reason filter we
    added defensively."""

    def _make_model(self, db):
        """Insert a minimal MLModel row to satisfy the FK on ml_predictions."""
        from backend.database import MLModel

        model = MLModel(version=1, strategy="nostate_optimized", status="active")
        db.add(model)
        db.flush()
        return model.id

    def test_legacy_pyramid_row_is_not_anchored_as_buy(self):
        """A pre-fix pyramid row (action='BUY', reason='PYRAMID:...') in
        the same window as an ml_predictions row must NOT be matched."""
        from datetime import date as dt_date, datetime as dt_datetime, timedelta
        from backend.database import AIPortfolioTrade, MLPrediction
        from backend.ml_backfill import backfill_actual_outcomes

        db = _fresh_session()
        try:
            model_id = self._make_model(db)
            pred_date = dt_date.today() - timedelta(days=3)
            # ml_predictions row needing an outcome
            pred = MLPrediction(
                model_id=model_id,
                ticker="LEGACY",
                ml_confidence=0.42,
                features={},
                prediction_date=pred_date,
                created_at=dt_datetime.combine(pred_date, dt_datetime.min.time()),
            )
            db.add(pred)

            # Legacy-shape pyramid trade in the matching 2-day window
            day_start = dt_datetime.combine(pred_date, dt_datetime.min.time())
            db.add(
                AIPortfolioTrade(
                    ticker="LEGACY",
                    action="BUY",  # legacy pre-fix shape
                    shares=10.0,
                    price=100.0,
                    total_value=1000.0,
                    reason="PYRAMID: legacy add",
                    user_id=1,
                    executed_at=day_start + timedelta(hours=2),
                )
            )
            db.commit()

            counts = backfill_actual_outcomes(db=db, lookback_days=30)
            assert counts["no_buy"] >= 1, (
                "A legacy pyramid row was incorrectly anchored as the "
                "ML-prediction BUY. backfill_actual_outcomes must filter "
                "out reason LIKE 'PYRAMID:%'."
            )
        finally:
            db.close()

    def test_real_buy_in_same_window_is_still_anchored(self):
        """Sanity: filter must not break the normal case — a genuine BUY
        without PYRAMID reason should still be paired."""
        from datetime import date as dt_date, datetime as dt_datetime, timedelta
        from backend.database import AIPortfolioTrade, MLPrediction
        from backend.ml_backfill import backfill_actual_outcomes

        db = _fresh_session()
        try:
            model_id = self._make_model(db)
            pred_date = dt_date.today() - timedelta(days=3)
            db.add(
                MLPrediction(
                    model_id=model_id,
                    ticker="REAL",
                    ml_confidence=0.55,
                    features={},
                    prediction_date=pred_date,
                    created_at=dt_datetime.combine(pred_date, dt_datetime.min.time()),
                )
            )
            day_start = dt_datetime.combine(pred_date, dt_datetime.min.time())
            db.add(
                AIPortfolioTrade(
                    ticker="REAL",
                    action="BUY",
                    shares=10.0,
                    price=100.0,
                    total_value=1000.0,
                    reason="Pre-breakout entry, score=82",
                    user_id=1,
                    executed_at=day_start + timedelta(hours=2),
                )
            )
            db.commit()

            counts = backfill_actual_outcomes(db=db, lookback_days=30)
            # No SELL means still_open, but the buy was anchored (no_buy=0)
            assert counts["no_buy"] == 0, (
                "Filter regression: a genuine BUY was incorrectly excluded. "
                f"counts={counts}"
            )
            assert counts["still_open"] == 1
        finally:
            db.close()
