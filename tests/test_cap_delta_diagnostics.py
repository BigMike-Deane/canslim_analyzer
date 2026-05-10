"""Tests for /api/admin/cap-delta-diagnostics endpoint + helper.

Closes the Scenario C.3 blind spot from the Shadow Step 7 verification
(`d496b09`): population-level read of c_score vs c_score_uncapped over a
rolling stock_scores window, so a shadow_no_excellence_cap arm that emits
trades byte-identical to baseline (because today's universe happens to be
all 4-of-4 excellence stocks) is detectable from observability rather than
from waiting on divergent trade rows that may never materialize.

Direct-invocation pattern mirrors tests/test_strategy_ab_eval.py — the
endpoint takes JWT auth via `get_admin_user`, and tests bypass FastAPI's
dependency resolver by passing `current_user=None` and `db=db` explicitly.
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import (
    init_db, SessionLocal, Stock, StockScore,
)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(StockScore).delete()
    db.query(Stock).delete()
    db.commit()
    try:
        yield db
    finally:
        db.query(StockScore).delete()
        db.query(Stock).delete()
        db.commit()
        db.close()


def _make_stock(db, ticker, **kwargs):
    s = Stock(ticker=ticker, name=ticker, sector="Tech", industry="Software", **kwargs)
    db.add(s)
    db.flush()  # populate s.id without committing
    return s


def _make_score(db, stock_id, *, total_score, c_score, c_score_uncapped,
                executed_at=None):
    """Seed one StockScore row.

    `executed_at` controls the .timestamp column (the indexed column the
    diagnostics endpoint windows on). Defaults to "now" so rows land in
    the default 1-day window.
    """
    when = executed_at or datetime.now(timezone.utc)
    row = StockScore(
        stock_id=stock_id,
        timestamp=when,
        date=when.date(),
        total_score=total_score,
        c_score=c_score,
        c_score_uncapped=c_score_uncapped,
        a_score=0.0, n_score=0.0, s_score=0.0, l_score=0.0, i_score=0.0, m_score=0.0,
    )
    db.add(row)
    return row


def _call(db, **kwargs):
    """Call the endpoint with all Query-defaulted params explicitly resolved."""
    from backend.routes.admin import cap_delta_diagnostics_endpoint
    defaults = dict(
        days=1,
        min_delta=0.5,
        strategy="nostate_cs_bear",
        buy_threshold=72.0,
        current_user=None,
        db=db,
    )
    defaults.update(kwargs)
    return asyncio.run(cap_delta_diagnostics_endpoint(**defaults))


# ============================================================================
# Helper-level: response shape under empty + uniform conditions
# ============================================================================
class TestEmptyAndUniform:
    def test_empty_window_returns_200_shape(self, db_session):
        # No StockScore rows in window — endpoint must return zeroed shape, not 404.
        body = _call(db_session)
        assert body["population"]["rows_total"] == 0
        assert body["population"]["rows_with_delta"] == 0
        assert body["population"]["rows_with_delta_pct"] == 0.0
        assert body["population"]["delta_buckets"] == []
        assert body["population"]["avg_delta_when_present"] == 0.0
        assert body["population"]["max_delta"] == 0.0
        assert body["top_divergent_tickers"] == []
        assert body["window"]["n_score_writes"] == 0
        assert body["window"]["n_distinct_stocks"] == 0
        # Window is structurally present even when empty.
        assert body["window"]["days"] == 1
        assert body["window"]["start_utc"].endswith("Z")
        assert body["window"]["end_utc"].endswith("Z")

    def test_all_rows_uncapped_equals_capped(self, db_session):
        # Cap never bit — every row has delta == 0.
        s = _make_stock(db_session, "AAAA")
        for _ in range(10):
            _make_score(db_session, s.id, total_score=70.0, c_score=12.0, c_score_uncapped=12.0)
        db_session.commit()

        body = _call(db_session)
        assert body["population"]["rows_total"] == 10
        assert body["population"]["rows_with_uncapped"] == 10
        assert body["population"]["rows_with_delta"] == 0
        assert body["top_divergent_tickers"] == []
        # All ten land in the zero bucket; the other three buckets should
        # exist with n=0 so the frontend bar can render the full histogram.
        zero_bucket = next(b for b in body["population"]["delta_buckets"] if b["bucket"] == "0")
        assert zero_bucket["n"] == 10
        assert zero_bucket["pct"] == 100.0
        assert all(b["n"] == 0 for b in body["population"]["delta_buckets"] if b["bucket"] != "0")


# ============================================================================
# Mixed population — bucket counts + averages
# ============================================================================
class TestMixedPopulation:
    def test_bucket_counts_match_brief_example(self, db_session):
        # Brief's canonical seeding: 90 rows delta=0, 8 rows delta=1, 0 rows
        # delta in [1.5,3.0), 2 rows delta=3.5. Only AAAA used so the
        # divergent-ticker map is deterministic.
        s = _make_stock(db_session, "AAAA")
        for _ in range(90):
            _make_score(db_session, s.id, total_score=70.0, c_score=12.0, c_score_uncapped=12.0)
        for _ in range(8):
            _make_score(db_session, s.id, total_score=71.0, c_score=12.0, c_score_uncapped=13.0)
        for _ in range(2):
            _make_score(db_session, s.id, total_score=73.5, c_score=13.0, c_score_uncapped=16.5)
        db_session.commit()

        body = _call(db_session)
        bs = {b["bucket"]: b["n"] for b in body["population"]["delta_buckets"]}
        assert bs == {"0": 90, "0.5-1.5": 8, "1.5-3.0": 0, "3.0+": 2}
        # avg_delta_when_present excludes the zero-bucket rows.
        # (8*1.0 + 2*3.5) / 10 = 1.5
        assert body["population"]["avg_delta_when_present"] == 1.5
        assert body["population"]["max_delta"] == 3.5
        assert body["population"]["rows_with_delta"] == 10
        assert body["population"]["rows_with_delta_pct"] == 10.0

    def test_uncapped_null_counts_toward_zero_bucket(self, db_session):
        # Pre-Shadow-Step-7 rows — c_score_uncapped is NULL. They populate
        # the window but contribute zero divergence.
        s = _make_stock(db_session, "AAAA")
        for _ in range(5):
            _make_score(db_session, s.id, total_score=70.0, c_score=12.0, c_score_uncapped=None)
        for _ in range(3):
            _make_score(db_session, s.id, total_score=71.0, c_score=12.0, c_score_uncapped=14.0)
        db_session.commit()

        body = _call(db_session)
        assert body["population"]["rows_total"] == 8
        assert body["population"]["rows_with_uncapped"] == 3
        assert body["population"]["rows_with_delta"] == 3
        bs = {b["bucket"]: b["n"] for b in body["population"]["delta_buckets"]}
        assert bs["0"] == 5  # the 5 null-uncapped rows
        assert bs["1.5-3.0"] == 3


# ============================================================================
# Threshold crossings
# ============================================================================
class TestThresholdCrossings:
    def test_crossed_capped_below_uncapped_above(self, db_session):
        # Brief edge case: capped=71.5, uncapped=73.0, threshold=72.0.
        # delta=1.5 (in 1.5-3.0 bucket), total=71.5 < 72 ≤ 73.0 → crosses.
        s = _make_stock(db_session, "AAAA")
        _make_score(db_session, s.id, total_score=71.5, c_score=12.0, c_score_uncapped=13.5)
        db_session.commit()

        body = _call(db_session, buy_threshold=72.0)
        assert body["threshold_crossings"]["rows_baseline_below_threshold_uncapped_above"] == 1
        assert body["threshold_crossings"]["rows_baseline_above_threshold_uncapped_below"] == 0
        # Per-ticker flag matches.
        ticker_row = body["top_divergent_tickers"][0]
        assert ticker_row["ticker"] == "AAAA"
        assert ticker_row["crossed_buy_threshold_at_least_once"] is True
        assert ticker_row["max_capped_score"] == 71.5
        assert ticker_row["max_uncapped_score"] == 73.0

    def test_not_crossed_when_both_above(self, db_session):
        # capped=72.5, uncapped=73.0, threshold=72.0 — already qualifies
        # under both arms, so it does NOT count as a crossing.
        s = _make_stock(db_session, "BBBB")
        _make_score(db_session, s.id, total_score=72.5, c_score=12.0, c_score_uncapped=13.0)
        db_session.commit()

        body = _call(db_session, buy_threshold=72.0)
        assert body["threshold_crossings"]["rows_baseline_below_threshold_uncapped_above"] == 0
        assert body["top_divergent_tickers"][0]["crossed_buy_threshold_at_least_once"] is False


# ============================================================================
# Param sensitivity: min_delta + days
# ============================================================================
class TestParamSensitivity:
    def test_min_delta_filter_excludes_low_bucket_rows(self, db_session):
        # 8 rows with delta=1.0 should be divergent under default min_delta=0.5
        # but excluded entirely under min_delta=1.5.
        s = _make_stock(db_session, "CCCC")
        for _ in range(8):
            _make_score(db_session, s.id, total_score=70.0, c_score=12.0, c_score_uncapped=13.0)
        db_session.commit()

        loose = _call(db_session, min_delta=0.5)
        strict = _call(db_session, min_delta=1.5)

        assert loose["population"]["rows_with_delta"] == 8
        assert any(t["ticker"] == "CCCC" for t in loose["top_divergent_tickers"])

        assert strict["population"]["rows_with_delta"] == 0
        assert strict["top_divergent_tickers"] == []
        # All 8 rows now land in the zero bucket.
        zero_bucket = next(b for b in strict["population"]["delta_buckets"] if b["bucket"] == "0")
        assert zero_bucket["n"] == 8

    def test_days_param_widens_window(self, db_session):
        # Two divergent rows: one 12 hours ago (in 1d window) and one 5
        # days ago (only in 7d+ windows).
        s = _make_stock(db_session, "DDDD")
        now = datetime.now(timezone.utc)
        _make_score(db_session, s.id, total_score=71.0, c_score=12.0, c_score_uncapped=14.0,
                    executed_at=now - timedelta(hours=12))
        _make_score(db_session, s.id, total_score=70.0, c_score=12.0, c_score_uncapped=14.0,
                    executed_at=now - timedelta(days=5))
        db_session.commit()

        body_1d = _call(db_session, days=1)
        body_7d = _call(db_session, days=7)

        assert body_1d["population"]["rows_total"] == 1
        assert body_7d["population"]["rows_total"] == 2
        assert body_7d["population"]["rows_with_delta"] == 2


# ============================================================================
# Top-divergent-tickers ordering + idempotency
# ============================================================================
class TestTopDivergentOrdering:
    def test_sorted_by_max_delta_desc_then_ticker_asc(self, db_session):
        # Three tickers with different max deltas — verify desc ordering.
        # Two tickers tied on max_delta verify ticker-asc tie-breaker.
        ax = _make_stock(db_session, "AX")
        bx = _make_stock(db_session, "BX")
        cx = _make_stock(db_session, "CX")
        # AX: max delta 4.0
        _make_score(db_session, ax.id, total_score=71.0, c_score=12.0, c_score_uncapped=16.0)
        # BX: max delta 2.0
        _make_score(db_session, bx.id, total_score=71.0, c_score=12.0, c_score_uncapped=14.0)
        # CX: max delta 2.0 — tie with BX; should sort after BX alphabetically.
        _make_score(db_session, cx.id, total_score=71.0, c_score=12.0, c_score_uncapped=14.0)
        db_session.commit()

        body = _call(db_session)
        tickers = [t["ticker"] for t in body["top_divergent_tickers"]]
        assert tickers == ["AX", "BX", "CX"]

    def test_idempotent_same_inputs(self, db_session):
        # Call twice — bodies must be byte-identical (population shape +
        # ticker order). Window times will differ across the two calls so
        # we ignore those fields and compare population/threshold/tickers.
        s = _make_stock(db_session, "EEEE")
        _make_score(db_session, s.id, total_score=70.0, c_score=12.0, c_score_uncapped=14.0)
        db_session.commit()

        a = _call(db_session)
        b = _call(db_session)
        assert a["population"] == b["population"]
        assert a["threshold_crossings"] == b["threshold_crossings"]
        assert a["top_divergent_tickers"] == b["top_divergent_tickers"]


# ============================================================================
# Auth: non-admin users hit the admin guard. Mirrors the rejection path
# every /api/admin/* endpoint relies on — see backend/auth.py:get_admin_user.
# ============================================================================
class TestAuthRejection:
    def test_non_admin_user_blocked_by_admin_dep(self):
        from fastapi import HTTPException
        from backend.auth import get_admin_user
        from backend.database import User

        non_admin = User(
            id=999, email="not-admin@test.com", display_name="NoAdmin",
            is_active=True, is_admin=False, hashed_password="",
        )
        with pytest.raises(HTTPException) as excinfo:
            get_admin_user(current_user=non_admin)
        assert excinfo.value.status_code == 403
        assert "admin" in str(excinfo.value.detail).lower()
