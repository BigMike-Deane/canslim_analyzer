"""
Tests for core API routes in backend/main.py.

Covers: health, scanner status, stocks, portfolio, ai-portfolio,
backtests, strategies, system-health, dashboard, watchlist, search.

Follows the existing pattern from test_api_endpoints.py:
- Uses FastAPI dependency_overrides to bypass auth
- Uses the default SQLite DB via init_db() + SessionLocal
- Seeds data into the real DB session for tests that need it
"""

import os
import pytest
from datetime import datetime, date, timezone, timedelta
from unittest.mock import patch, MagicMock

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import (
    init_db, get_db, SessionLocal, User, Stock, StockScore,
    PortfolioPosition, AIPortfolioConfig, AIPortfolioPosition,
    AIPortfolioTrade, BacktestRun, MarketSnapshot, Watchlist, MLModel,
    MLPrediction,
)
from backend.auth import get_current_active_user, get_admin_user
from tests.conftest import override_dependency


# ── Auth bypass ──────────────────────────────────────────────────────

_fake_user = User(
    id=1, email="test@test.com", display_name="Test User",
    is_active=True, is_admin=True, hashed_password="",
)


@pytest.fixture(autouse=True, scope="module")
def _auth_override():
    """Install this file's auth bypass for the duration of the module
    only, restoring whatever was there before. Replaces the older
    module-level `app.dependency_overrides[...] = lambda: _fake_user`
    pattern, which silently stomped overrides from other test files
    because pytest collection imports every test file before any test
    runs (last-loaded wins for the session). See tests/conftest.py."""
    with override_dependency(get_current_active_user, _fake_user), \
         override_dependency(get_admin_user, _fake_user):
        yield


init_db()
client = TestClient(app)


# ── Seed helpers ─────────────────────────────────────────────────────

def _get_db():
    return SessionLocal()


def _ensure_user():
    db = _get_db()
    try:
        if not db.query(User).filter_by(id=1).first():
            db.add(User(id=1, email="test@test.com", display_name="Test User",
                        is_active=True, is_admin=True, hashed_password=""))
            db.commit()
    finally:
        db.close()


def _ensure_stock(ticker="AAPL", score=85.0):
    db = _get_db()
    try:
        if not db.query(Stock).filter_by(ticker=ticker).first():
            db.add(Stock(
                ticker=ticker, name=f"{ticker} Inc.", sector="Technology",
                industry="Consumer Electronics", current_price=150.0,
                market_cap=2_500_000_000_000, week_52_high=180.0,
                week_52_low=120.0, canslim_score=score,
                c_score=12, a_score=11, n_score=10, s_score=13,
                l_score=12, i_score=8, m_score=9,
                projected_growth=15.0, growth_confidence="high",
                last_updated=datetime.now(timezone.utc),
            ))
            db.commit()
    finally:
        db.close()


def _ensure_market_snapshot():
    db = _get_db()
    try:
        if not db.query(MarketSnapshot).filter_by(date=date.today()).first():
            db.add(MarketSnapshot(
                date=date.today(), spy_price=520.0,
                market_score=10.0, market_trend="bullish",
            ))
            db.commit()
    finally:
        db.close()


def _ensure_portfolio_position(ticker="AAPL"):
    _ensure_user()
    _ensure_stock(ticker)
    db = _get_db()
    try:
        if not db.query(PortfolioPosition).filter_by(ticker=ticker, user_id=1).first():
            db.add(PortfolioPosition(
                user_id=1, ticker=ticker, shares=10, cost_basis=140.0,
                current_price=150.0, current_value=1500.0,
                gain_loss=100.0, gain_loss_pct=7.14,
                recommendation="hold", canslim_score=85.0,
            ))
            db.commit()
    finally:
        db.close()


def _ensure_ai_config():
    _ensure_user()
    db = _get_db()
    try:
        if not db.query(AIPortfolioConfig).filter_by(user_id=1).first():
            db.add(AIPortfolioConfig(
                user_id=1, starting_cash=25000.0, current_cash=20000.0,
                max_positions=8, max_position_pct=12.0,
                min_score_to_buy=72, sell_score_threshold=45,
                take_profit_pct=75.0, stop_loss_pct=8.0,
                is_active=True, strategy="nostate_optimized",
            ))
            db.commit()
    finally:
        db.close()


def _ensure_ai_position(ticker="NVDA"):
    _ensure_stock(ticker, score=88.0)
    _ensure_ai_config()
    db = _get_db()
    try:
        if not db.query(AIPortfolioPosition).filter_by(ticker=ticker, user_id=1).first():
            db.add(AIPortfolioPosition(
                user_id=1, ticker=ticker, shares=20, cost_basis=140.0,
                current_price=155.0, current_value=3100.0,
                gain_loss=300.0, gain_loss_pct=10.71,
                purchase_score=82.0, current_score=85.0,
                peak_price=160.0, peak_date=datetime.now(timezone.utc),
            ))
            db.commit()
    finally:
        db.close()


def _ensure_backtest():
    _ensure_user()
    db = _get_db()
    try:
        if not db.query(BacktestRun).filter_by(user_id=1, name="Test Backtest").first():
            db.add(BacktestRun(
                user_id=1, name="Test Backtest", status="completed",
                start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
                starting_cash=25000.0, final_value=35000.0,
                total_return_pct=40.0, spy_return_pct=20.0,
                max_drawdown_pct=10.0, sharpe_ratio=1.5,
                win_rate=65.0, total_trades=50,
                strategy="nostate_optimized", progress_pct=100.0,
                created_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            ))
            db.commit()
    finally:
        db.close()


# ── Seed once at module level ────────────────────────────────────────

_ensure_user()
_ensure_market_snapshot()
_ensure_stock("AAPL", score=90.0)
_ensure_stock("MSFT", score=85.0)


# ── Tests ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_shape(self):
        r = client.get("/health")
        d = r.json()
        assert "status" in d
        assert "database" in d
        assert "stocks_cached" in d
        assert "version" in d

    def test_health_db_healthy(self):
        d = client.get("/health").json()
        assert d["database"] == "healthy"


class TestScannerStatus:
    def test_scanner_status_returns_200(self):
        r = client.get("/api/scanner/status")
        assert r.status_code == 200

    def test_scanner_status_has_expected_keys(self):
        d = client.get("/api/scanner/status").json()
        assert "enabled" in d
        assert "is_scanning" in d
        assert "stocks_scanned" in d
        assert "total_stocks" in d


class TestStocksList:
    def test_stocks_returns_200(self):
        r = client.get("/api/stocks")
        assert r.status_code == 200

    def test_stocks_response_shape(self):
        d = client.get("/api/stocks").json()
        assert "stocks" in d
        assert "total" in d
        assert "limit" in d
        assert "offset" in d
        assert isinstance(d["stocks"], list)

    def test_stocks_contain_score_fields(self):
        stocks = client.get("/api/stocks").json()["stocks"]
        if stocks:
            s = stocks[0]
            for key in ("ticker", "name", "canslim_score", "c_score", "current_price"):
                assert key in s

    def test_stocks_pagination(self):
        d = client.get("/api/stocks?limit=1&offset=0").json()
        assert len(d["stocks"]) <= 1

    def test_stocks_filter_min_score(self):
        r = client.get("/api/stocks?min_score=88")
        assert r.status_code == 200

    def test_stocks_sort_by_price(self):
        r = client.get("/api/stocks?sort_by=current_price&sort_dir=asc")
        assert r.status_code == 200


class TestSingleStock:
    def test_existing_stock_returns_200(self):
        _ensure_stock("TSLA", score=78.0)
        r = client.get("/api/stocks/TSLA")
        assert r.status_code == 200

    def test_existing_stock_shape(self):
        _ensure_stock("TSLA", score=78.0)
        d = client.get("/api/stocks/TSLA").json()
        assert d["ticker"] == "TSLA"
        assert "canslim_score" in d
        assert "scores" in d
        assert "score_history" in d

    def test_invalid_ticker_format(self):
        r = client.get("/api/stocks/!!!INVALID!!!")
        assert r.status_code == 400

    def test_missing_stock_triggers_404(self):
        """A stock not in DB triggers live analysis; mock it to return None -> 404."""
        with patch("backend.main.analyze_stock", return_value=None):
            r = client.get("/api/stocks/ZZZZ")
            assert r.status_code == 404

    def _seed_multi_scan_history(self, ticker="MULTI"):
        """Seed a stock with 3 scans on the same day so we can tell daily-dedup
        and per-scan resolution apart."""
        _ensure_stock(ticker, score=70.0)
        db = _get_db()
        try:
            stock = db.query(Stock).filter_by(ticker=ticker).first()
            db.query(StockScore).filter_by(stock_id=stock.id).delete()
            today = date.today()
            base = datetime.combine(today, datetime.min.time())
            for hour, total in [(9, 71.0), (12, 73.0), (15, 75.0)]:
                db.add(StockScore(
                    stock_id=stock.id, date=today,
                    timestamp=base + timedelta(hours=hour),
                    total_score=total, c_score=12, a_score=11, n_score=10,
                    s_score=12, l_score=11, i_score=8, m_score=11,
                    current_price=150.0,
                ))
            db.commit()
        finally:
            db.close()

    def test_resolution_daily_dedups_to_one_per_day(self):
        self._seed_multi_scan_history("MULTI")
        d = client.get("/api/stocks/MULTI?resolution=daily").json()
        # 3 scans on the same day collapse to 1 entry.
        same_day = [h for h in d["score_history"] if h["date"] == date.today().isoformat()]
        assert len(same_day) == 1
        assert d["score_history_resolution"] == "daily"

    def test_resolution_all_returns_every_scan(self):
        self._seed_multi_scan_history("MULTI")
        d = client.get("/api/stocks/MULTI?resolution=all").json()
        same_day = [h for h in d["score_history"] if h["date"] == date.today().isoformat()]
        assert len(same_day) == 3
        assert d["score_history_resolution"] == "all"
        # Every entry must carry a timestamp so the chart can plot per-scan.
        assert all(h.get("timestamp") for h in same_day)

    def test_resolution_invalid_value_rejected(self):
        r = client.get("/api/stocks/TSLA?resolution=hourly")
        assert r.status_code == 422  # FastAPI regex validation


class TestPortfolio:
    @classmethod
    def setup_class(cls):
        _ensure_portfolio_position("GOOG")

    def test_portfolio_returns_200(self):
        r = client.get("/api/portfolio")
        assert r.status_code == 200

    def test_portfolio_response_shape(self):
        d = client.get("/api/portfolio").json()
        assert "positions" in d
        assert "summary" in d
        assert isinstance(d["positions"], list)
        summary = d["summary"]
        assert "total_value" in summary
        assert "positions_count" in summary

    def test_portfolio_position_fields(self):
        positions = client.get("/api/portfolio").json()["positions"]
        if positions:
            p = positions[0]
            for key in ("ticker", "shares", "cost_basis", "current_price", "gain_loss_pct"):
                assert key in p


class TestAIPortfolio:
    @classmethod
    def setup_class(cls):
        _ensure_ai_position("NVDA")

    def test_ai_portfolio_returns_200(self):
        with patch("backend.ai_trader.fetch_live_price", return_value=520.0):
            r = client.get("/api/ai-portfolio")
            assert r.status_code == 200

    def test_ai_portfolio_shape(self):
        with patch("backend.ai_trader.fetch_live_price", return_value=520.0):
            d = client.get("/api/ai-portfolio").json()
            assert "config" in d
            assert "summary" in d
            assert "positions" in d
            assert isinstance(d["positions"], list)

    def test_ai_portfolio_config_fields(self):
        with patch("backend.ai_trader.fetch_live_price", return_value=520.0):
            cfg = client.get("/api/ai-portfolio").json()["config"]
            for key in ("starting_cash", "max_positions", "is_active", "strategy"):
                assert key in cfg

    def test_ai_portfolio_trades_returns_200(self):
        r = client.get("/api/ai-portfolio/trades")
        assert r.status_code == 200

    def test_ai_portfolio_history_returns_200(self):
        r = client.get("/api/ai-portfolio/history?days=7")
        assert r.status_code == 200


class TestBacktests:
    @classmethod
    def setup_class(cls):
        _ensure_backtest()

    def test_list_backtests_returns_200(self):
        r = client.get("/api/backtests")
        assert r.status_code == 200

    def test_list_backtests_is_list(self):
        d = client.get("/api/backtests").json()
        assert isinstance(d, list)

    def test_backtest_entry_fields(self):
        backtests = client.get("/api/backtests").json()
        if backtests:
            bt = backtests[0]
            for key in ("id", "name", "status", "total_return_pct", "strategy", "created_at"):
                assert key in bt

    def test_backtest_presets_returns_200(self):
        r = client.get("/api/backtests/presets")
        assert r.status_code == 200
        presets = r.json()
        assert isinstance(presets, list)
        if presets:
            assert "name" in presets[0]

    def test_backtest_queue_returns_200(self):
        r = client.get("/api/backtests/queue")
        assert r.status_code == 200


class TestMLCompareMatrix:
    """The 4-way ML A/B/C/D matrix endpoint creates 4 BacktestRun records
    with distinct profile_overrides so we can attribute any lift to the
    bonus path vs the veto path independently."""

    def test_matrix_creates_four_distinct_runs(self):
        # Patch the queue so we don't actually run 4 backtests
        with patch("backend.backtest_queue.backtest_queue.enqueue") as mock_enqueue:
            r = client.post(
                "/api/ml/compare-matrix?"
                "start_date=2024-01-01&end_date=2024-06-01&"
                "strategy=nostate_optimized&min_confidence=0.30"
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["runs"]) == 4
        labels = [r["label"] for r in body["runs"]]
        assert labels == ["A baseline", "B bonus-only", "C veto-only", "D both"]
        # 4 enqueue calls (one per backtest)
        assert mock_enqueue.call_count == 4

    def test_matrix_profile_overrides_are_correct(self):
        with patch("backend.backtest_queue.backtest_queue.enqueue"):
            r = client.post(
                "/api/ml/compare-matrix?"
                "start_date=2024-01-01&end_date=2024-06-01&min_confidence=0.30"
            )
        runs = r.json()["runs"]
        cfg = {row["label"]: row["ml_signal"] for row in runs}
        # A: log-only, no veto — current production
        assert cfg["A baseline"]["log_only"] is True
        assert cfg["A baseline"]["min_confidence"] == 0.0
        # B: bonus path active, no veto — pure score modulation
        assert cfg["B bonus-only"]["log_only"] is False
        assert cfg["B bonus-only"]["min_confidence"] == 0.0
        # C: log-only but veto-on — gating without bonus
        assert cfg["C veto-only"]["log_only"] is True
        assert cfg["C veto-only"]["min_confidence"] == 0.30
        # D: both — full activation
        assert cfg["D both"]["log_only"] is False
        assert cfg["D both"]["min_confidence"] == 0.30

    def test_matrix_persists_records_with_overrides(self):
        with patch("backend.backtest_queue.backtest_queue.enqueue"):
            r = client.post(
                "/api/ml/compare-matrix?"
                "start_date=2024-01-01&end_date=2024-06-01"
            )
        run_ids = [row["id"] for row in r.json()["runs"]]
        db = _get_db()
        try:
            for rid in run_ids:
                rec = db.get(BacktestRun, rid)
                assert rec is not None
                assert rec.profile_overrides is not None
                assert "ml_signal" in rec.profile_overrides
        finally:
            db.close()


class TestMLMatricesEndpoint:
    """The /api/ml/matrices endpoint groups BacktestRun rows whose names
    follow the [ML A baseline] / B bonus-only / C veto-only / D both pattern.
    Incomplete sets (missing one or more siblings) must be filtered out."""

    _SUFFIX_FULL = "nostate_optimized 2024-01-01→2024-06-01"
    _SUFFIX_PARTIAL = "nostate_optimized 2025-01-01→2025-06-01"

    @classmethod
    def setup_class(cls):
        _ensure_user()
        db = _get_db()
        try:
            # Clean any prior test data so name lookups are deterministic
            db.query(BacktestRun).filter(
                BacktestRun.name.like("[ML %"),
                BacktestRun.name.like(f"%{cls._SUFFIX_FULL}"),
            ).delete(synchronize_session=False)
            db.query(BacktestRun).filter(
                BacktestRun.name.like("[ML %"),
                BacktestRun.name.like(f"%{cls._SUFFIX_PARTIAL}"),
            ).delete(synchronize_session=False)
            db.commit()

            # Full 4-variant matrix
            for label, ret, sharpe in [
                ("A baseline", 100.0, 1.5),
                ("B bonus-only", 110.0, 1.6),
                ("C veto-only", 105.0, 1.7),
                ("D both", 120.0, 1.4),
            ]:
                db.add(BacktestRun(
                    user_id=1,
                    name=f"[ML {label}] {cls._SUFFIX_FULL}",
                    status="completed",
                    start_date=date(2024, 1, 1), end_date=date(2024, 6, 1),
                    starting_cash=25000.0, total_return_pct=ret,
                    sharpe_ratio=sharpe, max_drawdown_pct=12.0,
                    total_trades=80, win_rate=60.0,
                    strategy="nostate_optimized",
                    profile_overrides={"ml_signal": {
                        "enabled": True,
                        "log_only": label in ("A baseline", "C veto-only"),
                        "min_confidence": 0.30 if "veto" in label or label == "D both" else 0.0,
                    }},
                    created_at=datetime.now(timezone.utc),
                ))
            # Partial set — only A and B exist, must be excluded
            for label in ("A baseline", "B bonus-only"):
                db.add(BacktestRun(
                    user_id=1,
                    name=f"[ML {label}] {cls._SUFFIX_PARTIAL}",
                    status="completed",
                    start_date=date(2025, 1, 1), end_date=date(2025, 6, 1),
                    starting_cash=25000.0, total_return_pct=50.0,
                    strategy="nostate_optimized",
                    profile_overrides={"ml_signal": {"enabled": True, "log_only": True}},
                    created_at=datetime.now(timezone.utc),
                ))
            db.commit()
        finally:
            db.close()

    def test_matrices_returns_200(self):
        r = client.get("/api/ml/matrices")
        assert r.status_code == 200, r.text

    def test_matrices_groups_complete_set(self):
        body = client.get("/api/ml/matrices").json()
        full = [m for m in body["matrices"] if m["suffix"] == self._SUFFIX_FULL]
        assert len(full) == 1
        m = full[0]
        assert m["strategy"] == "nostate_optimized"
        assert m["start_date"] == "2024-01-01"
        assert m["end_date"] == "2024-06-01"
        assert set(m["variants"].keys()) == {"A baseline", "B bonus-only", "C veto-only", "D both"}
        # Variant payload includes id + metrics + ml_signal
        a = m["variants"]["A baseline"]
        for key in ("id", "return_pct", "sharpe", "max_drawdown_pct", "total_trades", "ml_signal"):
            assert key in a
        # D both should have full activation config
        d = m["variants"]["D both"]
        assert d["ml_signal"]["log_only"] is False
        assert d["ml_signal"]["min_confidence"] == 0.30
        assert d["return_pct"] == 120.0

    def test_matrices_excludes_incomplete_sets(self):
        body = client.get("/api/ml/matrices").json()
        # The partial 2025 suffix has only A+B, so it must NOT be returned
        partial = [m for m in body["matrices"] if m["suffix"] == self._SUFFIX_PARTIAL]
        assert partial == []


class TestStrategies:
    def test_strategies_returns_200(self):
        r = client.get("/api/strategies")
        assert r.status_code == 200

    def test_strategies_is_list_of_profiles(self):
        d = client.get("/api/strategies").json()
        assert isinstance(d, list)
        if d:
            s = d[0]
            assert "name" in s
            assert "min_score" in s

    def test_strategies_hides_flagged_profiles(self):
        names = {s["name"] for s in client.get("/api/strategies").json()}
        # These two were retired May 1 2026 after the rolling-window matrix
        assert "nostate_correction_zone" not in names
        assert "nostate_high_conviction" not in names
        # But the live winners must still be visible
        assert "nostate_optimized" in names
        assert "nostate_cs_bear" in names

    def test_strategies_include_hidden_returns_all(self):
        names = {s["name"] for s in client.get("/api/strategies?include_hidden=true").json()}
        assert "nostate_correction_zone" in names
        assert "nostate_high_conviction" in names


class TestSystemHealth:
    """The /api/system-health endpoint imports scheduler, backup, redis modules.
    Some of these require PostgreSQL-specific SQL or Redis. We mock the heavy parts."""

    def test_system_health_returns_200(self):
        with patch("backend.scheduler.get_system_health", return_value={
            "last_successful_scan": None, "consecutive_scan_failures": 0,
        }), patch("backend.scheduler.get_scan_status", return_value={
            "enabled": False, "is_scanning": False, "stocks_scanned": 0, "total_stocks": 0,
        }), patch("backend.backup.get_backup_status", return_value={
            "last_backup": None, "backup_count": 0,
        }):
            r = client.get("/api/system-health")
            # SQLite doesn't support pg_database_size, so db query may fail
            # but the endpoint should still return 200 with degraded status
            assert r.status_code == 200

    def test_system_health_shape(self):
        with patch("backend.scheduler.get_system_health", return_value={}), \
             patch("backend.scheduler.get_scan_status", return_value={}), \
             patch("backend.backup.get_backup_status", return_value={}):
            r = client.get("/api/system-health")
            if r.status_code == 200:
                d = r.json()
                assert "database" in d
                assert "version" in d


class TestDashboard:
    def test_dashboard_returns_200(self):
        _ensure_market_snapshot()
        r = client.get("/api/dashboard")
        assert r.status_code == 200

    def test_dashboard_response_is_dict(self):
        _ensure_market_snapshot()
        d = client.get("/api/dashboard").json()
        assert isinstance(d, dict)


class TestMarketDirection:
    def test_market_direction_returns_200(self):
        _ensure_market_snapshot()
        r = client.get("/api/market-direction")
        assert r.status_code == 200


class TestWatchlist:
    def test_watchlist_returns_200(self):
        r = client.get("/api/watchlist")
        assert r.status_code == 200

    def test_watchlist_response_shape(self):
        d = client.get("/api/watchlist").json()
        assert "items" in d
        assert isinstance(d["items"], list)


class TestStockSearch:
    def test_search_returns_200(self):
        _ensure_stock("AAPL", score=90.0)
        r = client.get("/api/stocks/search?q=AAPL")
        assert r.status_code == 200

    def test_search_returns_list(self):
        d = client.get("/api/stocks/search?q=AAPL").json()
        assert isinstance(d, list)


class TestRateLimitStats:
    def test_rate_limit_stats_returns_200(self):
        r = client.get("/api/rate-limit-stats")
        assert r.status_code == 200


class TestMLValidation:
    """Verifies /api/ml/validation surfaces per_regime_auc on each fold."""

    def _seed_active_model(self, strategy="nostate_optimized"):
        db = _get_db()
        try:
            db.query(MLModel).filter(MLModel.strategy == strategy).delete()
            db.add(MLModel(
                version=99, strategy=strategy, status="active",
                model_type="classifier", training_samples=300, feature_count=22,
                roc_auc=0.58, accuracy=0.62, precision_score=0.55,
                recall_score=0.60, f1=0.57, brier_score=0.22,
                cv_results=[
                    {"fold": 0, "roc_auc": 0.57, "accuracy": 0.61, "precision": 0.54,
                     "recall": 0.58, "f1": 0.56, "train_size": 120, "test_size": 60,
                     "per_regime_auc": {"bullish": 0.62, "neutral": 0.51, "bearish": None}},
                    {"fold": 1, "roc_auc": 0.59, "accuracy": 0.63, "precision": 0.56,
                     "recall": 0.61, "f1": 0.58, "train_size": 180, "test_size": 60,
                     "per_regime_auc": {"bullish": 0.60, "neutral": 0.53, "bearish": 0.48}},
                    {"fold": 2, "roc_auc": 0.58, "accuracy": 0.62, "precision": 0.55,
                     "recall": 0.60, "f1": 0.57, "train_size": 240, "test_size": 60,
                     "per_regime_auc": {"bullish": 0.61, "neutral": None, "bearish": 0.50}},
                ],
                feature_importance={},
                created_at=datetime.now(timezone.utc),
                activated_at=datetime.now(timezone.utc),
            ))
            db.commit()
        finally:
            db.close()

    def _cleanup(self, strategy="nostate_optimized"):
        db = _get_db()
        try:
            db.query(MLModel).filter(MLModel.strategy == strategy, MLModel.version == 99).delete()
            db.commit()
        finally:
            db.close()

    def test_validation_returns_per_regime_auc_per_fold(self):
        self._seed_active_model()
        try:
            r = client.get("/api/ml/validation?strategy=nostate_optimized")
            assert r.status_code == 200
            d = r.json()
            assert "cv_results" in d
            assert len(d["cv_results"]) == 3
            for fold in d["cv_results"]:
                assert "per_regime_auc" in fold
                pra = fold["per_regime_auc"]
                assert set(pra.keys()) == {"bullish", "neutral", "bearish"}
                for v in pra.values():
                    assert v is None or isinstance(v, (int, float))
        finally:
            self._cleanup()

    def test_validation_404_when_no_active_model(self):
        self._cleanup(strategy="nonexistent_strategy_xyz")
        r = client.get("/api/ml/validation?strategy=nonexistent_strategy_xyz")
        assert r.status_code == 404


class TestTradeJournal:
    """Journal endpoint surfaces ML confidence and (optionally) vetoed candidates."""

    def _cleanup(self):
        db = _get_db()
        try:
            db.query(AIPortfolioTrade).filter_by(user_id=1).delete()
            db.query(MLPrediction).delete()
            db.query(MLModel).filter(MLModel.version >= 900).delete()
            db.commit()
        finally:
            db.close()

    def _seed_buy_trade(self, ticker="AAPL", ml_confidence=0.45, ml_bonus=0.0):
        _ensure_user()
        db = _get_db()
        try:
            db.add(AIPortfolioTrade(
                user_id=1, ticker=ticker, action="BUY",
                shares=10.0, price=100.0, total_value=1000.0,
                reason="PRE-BREAKOUT", canslim_score=78.0,
                signal_factors={
                    "entry_type": "pre-breakout",
                    "market_regime": "bullish",
                    "composite_score": 60.0,
                    "ml_confidence": ml_confidence,
                    "ml_bonus": ml_bonus,
                },
                executed_at=datetime.now(timezone.utc) - timedelta(days=1),
            ))
            db.commit()
        finally:
            db.close()

    def _seed_veto(self, ticker="ZZZ", ml_confidence=0.18, days_ago=1):
        db = _get_db()
        try:
            # Need an active model so MLPrediction.model_id has a target.
            m = db.query(MLModel).filter_by(version=999, strategy="nostate_optimized").first()
            if not m:
                m = MLModel(
                    version=999, strategy="nostate_optimized", status="active",
                    feature_count=24, roc_auc=0.6,
                    activated_at=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                )
                db.add(m)
                db.commit()
                db.refresh(m)
            db.add(MLPrediction(
                model_id=m.id, ticker=ticker,
                prediction_date=date.today() - timedelta(days=days_ago),
                ml_confidence=ml_confidence,
                features={"total_score": 73.0, "composite_score": 38.0, "c_score": 12.0},
                created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
            ))
            db.commit()
        finally:
            db.close()

    def test_journal_surfaces_ml_confidence(self):
        self._cleanup()
        self._seed_buy_trade(ticker="AAPL", ml_confidence=0.42, ml_bonus=2.4)
        d = client.get("/api/trade-journal?days=30").json()
        aapl = next((e for e in d["entries"] if e["ticker"] == "AAPL"), None)
        # Pre-fix this was always null because main.py read 'ml_prediction'.
        assert aapl is not None
        assert aapl["ml_confidence"] == pytest.approx(0.42)
        assert aapl["ml_bonus"] == pytest.approx(2.4)

    def test_vetoes_excluded_by_default(self):
        self._cleanup()
        self._seed_veto(ticker="ZZZ", ml_confidence=0.12)
        d = client.get("/api/trade-journal?days=30").json()
        assert all(e["action"] != "VETOED" for e in d["entries"])
        assert d["vetoed_count"] == 0

    def test_include_vetoed_surfaces_them(self):
        self._cleanup()
        self._seed_veto(ticker="ZZZ", ml_confidence=0.12)
        d = client.get("/api/trade-journal?days=30&include_vetoed=true").json()
        veto = next((e for e in d["entries"] if e["action"] == "VETOED"), None)
        assert veto is not None
        assert veto["ticker"] == "ZZZ"
        assert veto["ml_confidence"] == pytest.approx(0.12)
        assert d["vetoed_count"] >= 1

    def test_buy_suppresses_matching_veto_entry(self):
        # If a prediction ended in an actual buy on the same day, we should NOT
        # also list it as VETOED — that would double-count the decision.
        self._cleanup()
        self._seed_buy_trade(ticker="AAPL", ml_confidence=0.42)
        # Seed a veto-style ml_predictions row for the same ticker+day.
        db = _get_db()
        try:
            m = MLModel(version=998, strategy="nostate_optimized", status="active",
                        feature_count=24, roc_auc=0.6,
                        activated_at=datetime.now(timezone.utc),
                        created_at=datetime.now(timezone.utc))
            db.add(m); db.commit(); db.refresh(m)
            buy_at = datetime.now(timezone.utc) - timedelta(days=1)
            db.add(MLPrediction(
                model_id=m.id, ticker="AAPL", prediction_date=buy_at.date(),
                ml_confidence=0.42, features={"total_score": 78.0},
                created_at=buy_at,
            ))
            db.commit()
        finally:
            db.close()
        d = client.get("/api/trade-journal?days=30&include_vetoed=true").json()
        aapl_vetoes = [e for e in d["entries"] if e["ticker"] == "AAPL" and e["action"] == "VETOED"]
        assert aapl_vetoes == []  # Suppressed because a real BUY exists.

    def test_filter_action_vetoed_implies_include_vetoed(self):
        self._cleanup()
        self._seed_buy_trade(ticker="AAPL", ml_confidence=0.42)
        self._seed_veto(ticker="ZZZ", ml_confidence=0.12)
        d = client.get("/api/trade-journal?days=30&action=VETOED").json()
        # action=VETOED filters BUYs out, even without include_vetoed=true.
        assert all(e["action"] == "VETOED" for e in d["entries"])
        assert any(e["ticker"] == "ZZZ" for e in d["entries"])
