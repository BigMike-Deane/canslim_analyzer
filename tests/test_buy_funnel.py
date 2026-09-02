"""Tests for the buy-candidate funnel ledger (backend/buy_funnel.py +
/api/admin/buy-funnel). The evaluate_buys instrumentation itself is covered
in tests/test_ai_trader_coverage.py::TestBuyFunnelInstrumentation, which
owns the market/config fixtures that function needs."""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.database import init_db, SessionLocal, BuyFunnelRow
from backend import buy_funnel as bf
from backend.buy_funnel import (
    FunnelCollector, persist_funnel, latest_cycle, ticker_history,
    list_strategies, purge_now, CYCLE_TICKER, STAGE_ORDER, DEFAULT_CAP,
)

T0 = datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()
    db.query(BuyFunnelRow).delete()
    db.commit()
    bf._last_purge_at = None
    try:
        yield db
    finally:
        db.query(BuyFunnelRow).delete()
        db.commit()
        db.close()


def _names(f, cap=DEFAULT_CAP):
    """Per-name rows only (drops cycle notes + the histogram note)."""
    return [r for r in f.to_rows(cap=cap) if r["ticker"] != CYCLE_TICKER]


class TestCollector:
    def test_first_rejection_wins(self):
        f = FunnelCollector()
        f.reject("AAA", "score_floor", "60 < 65", 60)
        f.reject("AAA", "volume_gate", "later gate must not overwrite", 60)
        assert _names(f)[0]["stage"] == "score_floor"

    def test_bought_upgrades_ranked_in_place(self):
        f = FunnelCollector()
        f.ranked("AAA", 1, composite=88.2, score=76)
        f.bought("AAA")
        row = _names(f)[0]
        assert row["stage"] == "bought" and row["rank"] == 1 and row["composite"] == 88.2

    def test_exec_skip_only_downgrades_ranked(self):
        f = FunnelCollector()
        f.reject("BBB", "ml_veto")
        f.exec_skip("BBB", "must not touch a rejection")
        f.ranked("AAA", 2)
        f.exec_skip("AAA", "no live price")
        stages = {r["ticker"]: r["stage"] for r in _names(f)}
        assert stages == {"AAA": "exec_skipped", "BBB": "ml_veto"}

    def test_nan_scores_are_cleaned(self):
        f = FunnelCollector()
        f.reject("AAA", "no_score", None, float("nan"))
        assert _names(f)[0]["score"] is None

    def test_cap_keeps_late_stages_and_all_notes(self):
        f = FunnelCollector()
        f.note("portfolio_full", "8/8")
        for i in range(30):
            f.reject(f"E{i}", "score_floor", None, 50 + i)
        f.reject("VETO", "ml_veto")
        f.ranked("R1", 1, 90)
        f.ranked("R2", 2, 85)
        rows = f.to_rows(cap=5)
        assert rows[0]["ticker"] == CYCLE_TICKER and rows[0]["stage"] == "portfolio_full"
        assert rows[1]["stage"] == "histogram"
        kept = [r["ticker"] for r in rows[2:]]
        assert kept[:3] == ["R1", "R2", "VETO"], kept
        # The two remaining slots go to the HIGHEST-scoring floor rejects.
        assert kept[3:] == ["E29", "E28"]

    def test_histogram_note_is_uncapped(self):
        import json
        f = FunnelCollector()
        for i in range(30):
            f.reject(f"E{i}", "score_floor", None, 50 + i)
        f.reject("VETO", "ml_veto")
        rows = f.to_rows(cap=3)
        hist = [r for r in rows if r["stage"] == "histogram"]
        assert len(hist) == 1 and hist[0]["ticker"] == CYCLE_TICKER
        assert json.loads(hist[0]["detail"]) == {"score_floor": 30, "ml_veto": 1}
        assert len([r for r in rows if r["ticker"] != CYCLE_TICKER]) == 3

    def test_empty_collector_has_no_histogram(self):
        assert FunnelCollector().to_rows() == []

    def test_every_stage_has_an_order(self):
        for st in ("score_floor", "ml_veto", "ranked", "exec_skipped", "bought"):
            assert st in STAGE_ORDER


class TestPersistAndRead:
    def _cycle(self, db, when, *, name="nostate_cs_bear", user_id=1, sid=None):
        f = FunnelCollector()
        f.reject("LOW", "score_floor", "60 < 65", 60)
        f.reject("VET", "ml_veto", "0.21 < 0.30", 74)
        f.ranked("TOP", 1, 91.0, 78)
        f.ranked("NXT", 2, 84.0, 75)
        f.bought("TOP")
        return persist_funnel(db, f, strategy_name=name, user_id=user_id,
                              shadow_strategy_id=sid, cycle_at=when)

    def test_persist_writes_rows_and_latest_cycle_reads_them(self, db_session):
        assert self._cycle(db_session, T0) == 5  # 4 names + histogram note
        out = latest_cycle(db_session, key="user:1")
        assert out["strategy"] == "nostate_cs_bear"
        assert out["stage_counts"] == {"score_floor": 1, "ml_veto": 1, "ranked": 1, "bought": 1}
        assert out["n_candidates"] == 4 and out["rows_capped"] is False
        assert out["notes"] == []  # histogram is folded into stage_counts, not listed
        # Funnel order: bought/ranked first (by rank), then late gates.
        assert [r["ticker"] for r in out["rows"]] == ["TOP", "NXT", "VET", "LOW"]
        assert out["rows"][0]["rank"] == 1 and out["rows"][0]["composite"] == 91.0

    def test_latest_cycle_picks_newest_per_key(self, db_session):
        self._cycle(db_session, T0)
        self._cycle(db_session, T0 + timedelta(hours=2))
        self._cycle(db_session, T0 + timedelta(hours=1), name="nostate_cs_window14",
                    user_id=None, sid=8)
        newest_user = latest_cycle(db_session, key="user:1")
        assert newest_user["cycle_at"].startswith((T0 + timedelta(hours=2)).strftime("%Y-%m-%dT%H"))
        arm = latest_cycle(db_session, key="shadow:8")
        assert arm["strategy"] == "nostate_cs_window14"
        # Default key = newest cycle overall (the user:1 one at +2h).
        assert latest_cycle(db_session)["strategy"] == "nostate_cs_bear"

    def test_empty_key_returns_empty_shape(self, db_session):
        out = latest_cycle(db_session, key="shadow:99")
        assert out["cycle_at"] is None and out["rows"] == [] and out["stage_counts"] == {}

    def test_ticker_history_spans_strategies_newest_first(self, db_session):
        self._cycle(db_session, datetime.now(timezone.utc) - timedelta(hours=3))
        self._cycle(db_session, datetime.now(timezone.utc) - timedelta(hours=1),
                    name="nostate_cs_window14", user_id=None, sid=8)
        hist = ticker_history(db_session, "vet", days=7)
        assert [h["strategy"] for h in hist] == ["nostate_cs_window14", "nostate_cs_bear"]
        assert all(h["stage"] == "ml_veto" for h in hist)

    def test_list_strategies_groups_live_and_shadow(self, db_session):
        self._cycle(db_session, datetime.now(timezone.utc) - timedelta(hours=1))
        self._cycle(db_session, datetime.now(timezone.utc),
                    name="nostate_cs_window14", user_id=None, sid=8)
        keys = [s["key"] for s in list_strategies(db_session)]
        assert keys == ["shadow:8", "user:1"]

    def test_notes_only_cycle_persists(self, db_session):
        f = FunnelCollector()
        f.note("portfolio_full", "8/8 positions")
        assert persist_funnel(db_session, f, strategy_name="nostate_cs_bear", user_id=1) == 1
        out = latest_cycle(db_session, key="user:1")
        assert out["rows"] == [] and out["notes"][0]["stage"] == "portfolio_full"

    def test_empty_collector_writes_nothing(self, db_session):
        assert persist_funnel(db_session, FunnelCollector(), strategy_name="x") == 0
        assert db_session.query(BuyFunnelRow).count() == 0

    def test_purge_drops_rows_past_retention(self, db_session):
        old = datetime.now(timezone.utc) - timedelta(days=bf.RETENTION_DAYS + 1)
        # persist_funnel purges opportunistically; pin the clock so the stale
        # rows survive their own insert and the explicit purge is what drops them.
        bf._last_purge_at = datetime.now(timezone.utc)
        self._cycle(db_session, old)
        self._cycle(db_session, datetime.now(timezone.utc))
        assert db_session.query(BuyFunnelRow).count() == 10
        assert purge_now(db_session) == 5
        assert db_session.query(BuyFunnelRow).count() == 5

    def test_purge_is_rate_limited_per_process(self, db_session):
        old = datetime.now(timezone.utc) - timedelta(days=bf.RETENTION_DAYS + 1)
        bf._last_purge_at = datetime.now(timezone.utc)
        self._cycle(db_session, old)
        bf._last_purge_at = None
        assert bf._maybe_purge(db_session) == 5  # first call runs
        self._cycle(db_session, old)  # inside the window: persist does not purge
        assert bf._maybe_purge(db_session) == 0  # and neither does a direct call
        assert db_session.query(BuyFunnelRow).count() == 5

    def test_persist_time_purge_runs_on_first_call(self, db_session):
        old = datetime.now(timezone.utc) - timedelta(days=bf.RETENTION_DAYS + 1)
        bf._last_purge_at = None
        self._cycle(db_session, old)
        # The first persist of the process purged the stale rows it just wrote.
        assert db_session.query(BuyFunnelRow).count() == 0


class TestRoute:
    def test_latest_cycle_mode(self, db_session):
        from backend.routes.admin import get_buy_funnel
        f = FunnelCollector()
        f.reject("LOW", "score_floor", "60 < 65", 60)
        f.ranked("TOP", 1, 90, 77)
        persist_funnel(db_session, f, strategy_name="nostate_cs_bear", user_id=1)
        out = asyncio.run(get_buy_funnel(
            current_user=None, db=db_session, request=None,
            key="user:1", ticker=None, days=7, limit=400))
        assert out["stage_order"] == STAGE_ORDER
        assert out["strategies"][0]["key"] == "user:1"
        assert out["cycle"]["stage_counts"] == {"score_floor": 1, "ranked": 1}

    def test_ticker_mode_uppercases(self, db_session):
        from backend.routes.admin import get_buy_funnel
        f = FunnelCollector()
        f.reject("PDEX", "score_floor", "73 but book full", 73)
        persist_funnel(db_session, f, strategy_name="nostate_cs_window14",
                       shadow_strategy_id=8)
        out = asyncio.run(get_buy_funnel(
            current_user=None, db=db_session, request=None,
            key=None, ticker="pdex", days=7, limit=400))
        assert out["ticker"] == "PDEX"
        assert out["rows"][0]["shadow_strategy_id"] == 8
        assert "cycle" not in out

    def test_capped_cycle_reports_true_counts(self, db_session):
        f = FunnelCollector()
        for i in range(60):
            f.reject(f"E{i}", "score_floor", None, 40 + i)
        f.ranked("TOP", 1, 90, 77)
        persist_funnel(db_session, f, strategy_name="nostate_cs_bear", user_id=1, cap=10)
        out = latest_cycle(db_session, key="user:1")
        assert out["stage_counts"] == {"score_floor": 60, "ranked": 1}
        assert out["n_candidates"] == 61 and out["rows_capped"] is True
        assert len(out["rows"]) == 10
