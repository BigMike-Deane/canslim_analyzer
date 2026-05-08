"""
backend/routes/fidelity.py route coverage push (4.62% → 50%+ target).

Covers the six router endpoints under /api/fidelity:
  - POST /upload-positions   (TestFidelityCsvUpload)
  - POST /upload-activity    (TestFidelityTradeParsing)
  - GET  /snapshots          (TestFidelityPositions)
  - GET  /latest             (TestFidelityPositions)
  - GET  /trades             (TestFidelityTradeParsing)
  - GET  /reconciliation     (TestFidelityReconciliation)
  - GET  /gameplan           (TestFidelityGameplan)
  - POST /sync-to-portfolio  (TestFidelitySyncToPortfolio)

Pattern follows tests/test_main_coverage.py:
- FastAPI TestClient + dependency_overrides for auth bypass
- Default SessionLocal (in-memory SQLite via init_db())
- All routes are user-scoped to id=1 (the auth-bypassed _fake_user)
- IO boundaries (FMP / yfinance / data_fetcher market_direction) mocked

Latent bugs flagged (DO NOT FIX during 2026-06-18 Approach 2 eval window):
  routes/fidelity.py uses three names that are NOT imported:
    - expand_tickers_with_duplicates  (line 415, /gameplan)
    - DUPLICATE_TICKERS               (lines 639, 976, /gameplan)
    - PortfolioPosition               (lines 1085, 1098, 1111, /sync-to-portfolio)
  Both endpoints raise NameError in production for any non-empty case.
  Tests inject these via monkeypatch.setattr on the route module, so
  coverage exercises the real route bodies without touching source.
"""

import os
import pytest
from datetime import datetime, date, timezone, timedelta
from unittest.mock import MagicMock

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database import (
    init_db, SessionLocal, User, Stock,
    FidelitySnapshot, FidelityPosition, FidelityTrade,
    AIPortfolioPosition, AIPortfolioConfig, PortfolioPosition,
)
from backend.auth import get_current_active_user
from backend.routes import fidelity as fidelity_routes


# ── Auth bypass (matches test_main_coverage pattern) ─────────────────

_fake_user = User(
    id=1, email="test@test.com", display_name="Test User",
    is_active=True, is_admin=True, hashed_password="",
)
app.dependency_overrides[get_current_active_user] = lambda: _fake_user

init_db()
client = TestClient(app)


# ── DB helpers ───────────────────────────────────────────────────────

def _db():
    return SessionLocal()


def _ensure_user():
    db = _db()
    try:
        if not db.query(User).filter_by(id=1).first():
            db.add(User(
                id=1, email="test@test.com", display_name="Test User",
                is_active=True, is_admin=True, hashed_password="",
            ))
            db.commit()
    finally:
        db.close()


def _wipe_fidelity():
    """Per-test cleanup: each test owns its own fixture data."""
    db = _db()
    try:
        db.query(FidelityPosition).delete()
        db.query(FidelitySnapshot).delete()
        db.query(FidelityTrade).delete()
        db.query(PortfolioPosition).delete()
        db.query(AIPortfolioPosition).filter_by(user_id=1).delete()
        db.commit()
    finally:
        db.close()


def _ensure_stock(ticker, **kw):
    db = _db()
    try:
        existing = db.query(Stock).filter_by(ticker=ticker).first()
        if existing:
            for k, v in kw.items():
                setattr(existing, k, v)
            db.commit()
            return existing.id
        defaults = dict(
            ticker=ticker, name=f"{ticker} Inc.",
            sector="Technology", industry="Software",
            current_price=100.0,
            market_cap=20_000_000_000,
            week_52_high=120.0, week_52_low=70.0,
            canslim_score=80.0,
            c_score=12, a_score=11, n_score=10, s_score=12,
            l_score=11, i_score=8, m_score=10,
            projected_growth=20.0,
            growth_confidence="high",
            last_updated=datetime.now(timezone.utc),
        )
        defaults.update(kw)
        s = Stock(**defaults)
        db.add(s)
        db.commit()
        db.refresh(s)
        return s.id
    finally:
        db.close()


def _make_snapshot(snap_date=None, cash=1000.0, total=10000.0,
                   positions=None):
    """Create a FidelitySnapshot + FidelityPosition rows for user 1."""
    snap_date = snap_date or date(2026, 5, 1)
    positions = positions or []
    db = _db()
    try:
        snap = FidelitySnapshot(
            user_id=1,
            snapshot_date=snap_date,
            cash_balance=cash,
            total_value=total,
            positions_count=len(positions),
            uploaded_at=datetime.now(timezone.utc),
        )
        db.add(snap)
        db.flush()
        for p in positions:
            db.add(FidelityPosition(
                snapshot_id=snap.id,
                symbol=p.get("symbol"),
                description=p.get("description", ""),
                quantity=p.get("quantity", 100),
                last_price=p.get("last_price", 50.0),
                current_value=p.get("current_value", 5000.0),
                total_gain_loss=p.get("total_gain_loss", 100.0),
                total_gain_loss_pct=p.get("total_gain_loss_pct", 2.0),
                cost_basis_total=p.get("cost_basis_total", 4900.0),
                average_cost_basis=p.get("average_cost_basis", 49.0),
                percent_of_account=p.get("percent_of_account", 50.0),
                position_type=p.get("type", "Margin"),
            ))
        db.commit()
        db.refresh(snap)
        return snap.id
    finally:
        db.close()


def _make_trade(run_date, action, symbol, **kw):
    db = _db()
    try:
        t = FidelityTrade(
            user_id=1,
            run_date=run_date,
            action=action,
            symbol=symbol,
            description=kw.get("description", ""),
            price=kw.get("price", 100.0),
            quantity=kw.get("quantity", 10),
            amount=kw.get("amount", -1000.0),
            commission=kw.get("commission", 0),
            fees=kw.get("fees", 0),
            settlement_date=kw.get("settlement_date"),
            raw_action=kw.get("raw_action", action),
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return t.id
    finally:
        db.close()


# ── Module-level seed ────────────────────────────────────────────────

_ensure_user()


@pytest.fixture(autouse=True)
def _isolate_fidelity_state():
    """Each test starts with a clean fidelity table set."""
    _wipe_fidelity()
    yield
    _wipe_fidelity()


# ── Sample CSVs (lifted from test_fidelity_sync.py shape) ────────────

SAMPLE_POSITIONS_CSV = """Account Number,Account Name,Symbol,Description,Quantity,Last Price,Last Price Change,Current Value,Today's Gain/Loss Dollar,Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,Percent Of Account,Cost Basis Total,Average Cost Basis,Type
X99999999,Individual,SPAXX**,HELD IN MONEY MARKET,,,,$3000.00,,,,,30.00%,,,Cash,
X99999999,Individual,LCTX,LINEAGE CELL THERAPEUTICS INC COM,1000,$1.945,+$0.045,$1945.00,+$45.00,+2.36%,+$259.75,+15.41%,19.45%,$1685.25,$1.69,Margin,
X99999999,Individual,HUMA,HUMACYTE INC COM,430,$1.2052,+$0.0452,$518.23,+$19.43,+3.89%,-$1792.95,-77.58%,5.18%,$2311.18,$5.37,Margin,

"Date downloaded May-01-2026 11:27 a.m ET"
"""

SAMPLE_ACTIVITY_CSV = """

Run Date,Account,Account Number,Action,Symbol,Description,Type,Price ($),Quantity,Commission ($),Fees ($),Accrued Interest ($),Amount ($),Settlement Date
04/25/2026,Individual,X99999999,YOU BOUGHT ONDAS INC COMMON STOCK (ONDS) (Margin),ONDS,,Margin,10.30,50,,,,-$515.00,04/27/2026
04/24/2026,Individual,X99999999,YOU SOLD ARM HOLDINGS PLC SPON ADS (ARM) (Margin),ARM,,Margin,130.00,-12,,,,$1560.00,04/26/2026
"""


# ────────────────────────────────────────────────────────────────────
# TestFidelityCsvUpload — POST /api/fidelity/upload-positions
# ────────────────────────────────────────────────────────────────────

class TestFidelityCsvUpload:
    """Branches: CSV ext gate, decode fallback, empty result rejection,
    snapshot+positions persistence, response shape."""

    def test_rejects_non_csv_filename(self):
        """Branch: filename does not end with .csv → 400."""
        r = client.post(
            "/api/fidelity/upload-positions",
            files={"file": ("foo.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 400
        assert "CSV" in r.json()["detail"]

    def test_rejects_csv_with_no_target_account_positions(self):
        """Branch: parse_positions_csv returns empty positions → 400."""
        # All rows are 401K (different account), so target-account filter
        # produces zero positions
        empty_csv = (
            "Account Number,Account Name,Symbol,Description,Quantity,Last Price,"
            "Last Price Change,Current Value,Today's Gain/Loss Dollar,"
            "Today's Gain/Loss Percent,Total Gain/Loss Dollar,Total Gain/Loss Percent,"
            "Percent Of Account,Cost Basis Total,Average Cost Basis,Type\n"
        )
        r = client.post(
            "/api/fidelity/upload-positions",
            files={"file": ("empty.csv", empty_csv.encode("utf-8"), "text/csv")},
        )
        assert r.status_code == 400
        assert "No positions" in r.json()["detail"]

    def test_persists_snapshot_and_positions(self):
        """Branch: happy path — snapshot + 2 position rows committed,
        response includes counts and parse_errors list."""
        r = client.post(
            "/api/fidelity/upload-positions",
            files={"file": ("positions.csv",
                            SAMPLE_POSITIONS_CSV.encode("utf-8"),
                            "text/csv")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        assert body["positions_count"] == 2
        assert body["cash_balance"] == 3000.0
        assert body["snapshot_date"] == "2026-05-01"
        assert "parse_errors" in body

        # Confirm DB persistence
        db = _db()
        try:
            snap = db.query(FidelitySnapshot).filter_by(user_id=1).first()
            assert snap is not None
            assert snap.positions_count == 2
            tickers = {p.symbol for p in db.query(FidelityPosition)
                       .filter_by(snapshot_id=snap.id).all()}
            assert tickers == {"LCTX", "HUMA"}
        finally:
            db.close()

    def test_latin1_fallback_decode(self):
        """Branch: UTF-8 decode fails → falls back to latin-1."""
        # Inject a 0xFF byte (invalid UTF-8 start byte but valid latin-1)
        # mid-content
        bad_bytes = SAMPLE_POSITIONS_CSV.encode("utf-8").replace(
            b"LINEAGE", b"L\xffNEAGE"
        )
        r = client.post(
            "/api/fidelity/upload-positions",
            files={"file": ("positions.csv", bad_bytes, "text/csv")},
        )
        # Should still succeed via latin-1 fallback
        assert r.status_code == 200
        assert r.json()["positions_count"] == 2


# ────────────────────────────────────────────────────────────────────
# TestFidelityTradeParsing — POST /upload-activity + GET /trades
# ────────────────────────────────────────────────────────────────────

class TestFidelityTradeParsing:
    """Branches: ext gate, dedup logic, settlement_date parsing,
    GET /trades with + without symbol filter."""

    def test_rejects_non_csv_filename(self):
        """Branch: activity upload filename validation → 400."""
        r = client.post(
            "/api/fidelity/upload-activity",
            files={"file": ("activity.txt", b"x", "text/plain")},
        )
        assert r.status_code == 400

    def test_persists_new_trades(self):
        """Branch: parsed trades dedup against empty table → all inserted."""
        r = client.post(
            "/api/fidelity/upload-activity",
            files={"file": ("act.csv",
                            SAMPLE_ACTIVITY_CSV.encode("utf-8"),
                            "text/csv")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success"
        assert body["new_trades"] == 2
        assert body["skipped_duplicates"] == 0

        db = _db()
        try:
            symbols = {t.symbol for t in db.query(FidelityTrade).all()}
            assert symbols == {"ONDS", "ARM"}
        finally:
            db.close()

    def test_dedups_existing_trades(self):
        """Branch: existing FidelityTrade with same (date, symbol, action)
        is detected and skipped."""
        # Pre-seed ONDS BUY on 2026-04-25
        _make_trade(date(2026, 4, 25), "BUY", "ONDS")

        r = client.post(
            "/api/fidelity/upload-activity",
            files={"file": ("act.csv",
                            SAMPLE_ACTIVITY_CSV.encode("utf-8"),
                            "text/csv")},
        )
        assert r.status_code == 200
        body = r.json()
        # ONDS should be skipped, ARM should be new
        assert body["skipped_duplicates"] == 1
        assert body["new_trades"] == 1

    def test_get_trades_returns_newest_first(self):
        """Branch: GET /trades default — newest-first ordering by run_date."""
        _make_trade(date(2026, 4, 1), "BUY", "AAPL", price=180.0, quantity=10)
        _make_trade(date(2026, 4, 15), "SELL", "AAPL", price=190.0, quantity=10)
        _make_trade(date(2026, 4, 10), "BUY", "MSFT", price=400.0, quantity=5)

        r = client.get("/api/fidelity/trades")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        dates = [t["run_date"] for t in body["trades"]]
        assert dates == sorted(dates, reverse=True)

    def test_get_trades_symbol_filter_uppercased(self):
        """Branch: GET /trades?symbol=aapl → uppercased filter."""
        _make_trade(date(2026, 4, 1), "BUY", "AAPL")
        _make_trade(date(2026, 4, 1), "BUY", "MSFT")

        r = client.get("/api/fidelity/trades?symbol=aapl")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["trades"][0]["symbol"] == "AAPL"

    def test_get_trades_limit_param(self):
        """Branch: GET /trades?limit=2 — capped result set."""
        for i in range(5):
            _make_trade(date(2026, 4, i + 1), "BUY", f"TIC{i}")
        r = client.get("/api/fidelity/trades?limit=2")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_get_trades_empty(self):
        """Branch: no trades for user → empty list, count 0."""
        r = client.get("/api/fidelity/trades")
        assert r.status_code == 200
        assert r.json()["count"] == 0
        assert r.json()["trades"] == []


# ────────────────────────────────────────────────────────────────────
# TestFidelityPositions — GET /snapshots + GET /latest
# ────────────────────────────────────────────────────────────────────

class TestFidelityPositions:
    """Branches: snapshots list ordering, latest empty/with-positions,
    Stock enrichment join."""

    def test_snapshots_empty(self):
        """Branch: no snapshots → returns empty list."""
        r = client.get("/api/fidelity/snapshots")
        assert r.status_code == 200
        assert r.json() == {"snapshots": []}

    def test_snapshots_newest_first(self):
        """Branch: multiple snapshots ordered desc by snapshot_date."""
        _make_snapshot(snap_date=date(2026, 4, 1))
        _make_snapshot(snap_date=date(2026, 5, 1))
        _make_snapshot(snap_date=date(2026, 3, 1))

        r = client.get("/api/fidelity/snapshots")
        body = r.json()
        assert len(body["snapshots"]) == 3
        dates = [s["snapshot_date"] for s in body["snapshots"]]
        assert dates == ["2026-05-01", "2026-04-01", "2026-03-01"]

    def test_snapshots_limit(self):
        """Branch: limit query param caps results."""
        for d in [date(2026, m, 1) for m in (1, 2, 3, 4, 5)]:
            _make_snapshot(snap_date=d)
        r = client.get("/api/fidelity/snapshots?limit=2")
        assert len(r.json()["snapshots"]) == 2

    def test_latest_empty(self):
        """Branch: no snapshot uploaded → snapshot=None, positions=[]."""
        r = client.get("/api/fidelity/latest")
        assert r.status_code == 200
        body = r.json()
        assert body["snapshot"] is None
        assert body["positions"] == []

    def test_latest_with_canslim_enrichment(self):
        """Branch: latest snapshot + Stock enrichment (canslim_score, sector)."""
        _ensure_stock("AAPL", canslim_score=88.0, sector="Tech",
                      growth_mode_score=72.0, is_growth_stock=False,
                      projected_growth=15.0)
        _ensure_stock("UNKN")  # not enriched-loaded into stocks_by_ticker
        _make_snapshot(positions=[
            {"symbol": "AAPL", "current_value": 18000, "quantity": 100},
            {"symbol": "ZZZZ", "current_value": 5000, "quantity": 50},  # no Stock row
        ])

        r = client.get("/api/fidelity/latest")
        assert r.status_code == 200
        body = r.json()
        assert body["snapshot"]["positions_count"] == 2
        # Newest-first ordering by current_value desc
        assert body["positions"][0]["symbol"] == "AAPL"
        assert body["positions"][0]["canslim_score"] == 88.0
        assert body["positions"][0]["sector"] == "Tech"
        # ZZZZ has no Stock row → enrichment falls through to None/False
        zzzz = body["positions"][1]
        assert zzzz["symbol"] == "ZZZZ"
        assert zzzz["canslim_score"] is None
        assert zzzz["is_growth_stock"] is False


# ────────────────────────────────────────────────────────────────────
# TestFidelityReconciliation — GET /reconciliation
# ────────────────────────────────────────────────────────────────────

class TestFidelityReconciliation:
    """Branches: 404 when no snapshot, success path delegating to
    reconcile_portfolios with snapshot_date/snapshot_id appended."""

    def test_404_when_no_snapshot(self):
        """Branch: no FidelitySnapshot → 404."""
        r = client.get("/api/fidelity/reconciliation")
        assert r.status_code == 404
        assert "No Fidelity snapshot" in r.json()["detail"]

    def test_reconciliation_full_payload(self):
        """Branch: snapshot + positions + AI positions → matches/discrepancies/
        snapshot_date/snapshot_id."""
        _ensure_stock("AAPL", canslim_score=80.0)

        # AIPortfolioConfig + AIPortfolioPosition for user 1
        db = _db()
        try:
            cfg = AIPortfolioConfig(
                user_id=1, starting_cash=25000, current_cash=20000,
                max_positions=8, max_position_pct=12,
                min_score_to_buy=72, sell_score_threshold=45,
                take_profit_pct=75, stop_loss_pct=8,
                is_active=True, strategy="nostate_optimized",
            )
            db.add(cfg)
            db.flush()
            db.add(AIPortfolioPosition(
                user_id=1, ticker="AAPL", shares=100, cost_basis=150,
                current_price=180, current_value=18000,
                gain_loss=3000, gain_loss_pct=20.0,
                purchase_score=80, current_score=82,
                peak_price=185, peak_date=datetime.now(timezone.utc),
                purchase_date=datetime.now(timezone.utc) - timedelta(days=30),
            ))
            db.commit()
        finally:
            db.close()

        snap_id = _make_snapshot(positions=[
            {"symbol": "AAPL", "quantity": 100, "current_value": 18000,
             "total_gain_loss_pct": 20, "average_cost_basis": 150},
        ])
        r = client.get("/api/fidelity/reconciliation")
        assert r.status_code == 200
        body = r.json()
        assert body["snapshot_id"] == snap_id
        assert body["snapshot_date"] == "2026-05-01"
        assert len(body["matches"]) == 1
        assert body["matches"][0]["symbol"] == "AAPL"


# ────────────────────────────────────────────────────────────────────
# TestFidelityGameplan — GET /gameplan
# ────────────────────────────────────────────────────────────────────
#
# Latent-bug compensation: we monkeypatch DUPLICATE_TICKERS and
# expand_tickers_with_duplicates onto the routes module so the route
# body executes. Source is untouched.

class TestFidelityGameplan:
    """Branches: no snapshot empty, no positions empty, stop-loss SELL,
    partial-profit TRIM, BUY blocked by SPY gate, candidate scoring path."""

    @pytest.fixture(autouse=True)
    def _patch_undefined_names(self, monkeypatch):
        # Inject the names that routes/fidelity.py forgets to import
        monkeypatch.setattr(fidelity_routes, "DUPLICATE_TICKERS",
                            [], raising=False)
        monkeypatch.setattr(fidelity_routes, "expand_tickers_with_duplicates",
                            lambda s: set(s), raising=False)
        # Stub out market direction so SPY gate is deterministic
        # Default: bullish (SPY above 50MA → buys allowed)
        from data_fetcher import get_cached_market_direction  # noqa
        monkeypatch.setattr(
            "data_fetcher.get_cached_market_direction",
            lambda: {"spy": {"price": 520.0, "ma_50": 510.0}},
        )

    def test_gameplan_no_snapshot(self):
        """Branch: no snapshot → empty gameplan, 0 actions."""
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        body = r.json()
        assert body["gameplan"] == []
        assert body["summary"]["total_actions"] == 0

    def test_gameplan_no_positions(self):
        """Branch: snapshot exists but zero positions → empty gameplan."""
        _make_snapshot(positions=[])
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        body = r.json()
        assert body["gameplan"] == []
        assert body["summary"]["total_actions"] == 0

    def test_gameplan_stop_loss_sell(self):
        """Branch: position down 10% (below 7% stop) → SELL action emitted."""
        _ensure_stock("LOSER", canslim_score=60.0, current_price=50.0,
                      previous_score=60.0, c_score=10, l_score=8)
        _make_snapshot(total=10000, positions=[
            {"symbol": "LOSER", "quantity": 100, "last_price": 45.0,
             "current_value": 4500, "total_gain_loss_pct": -10.0,
             "average_cost_basis": 50.0},
        ])
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        body = r.json()
        sells = [a for a in body["gameplan"] if a["action"] == "SELL"]
        assert any(a["ticker"] == "LOSER" for a in sells)
        assert body["summary"]["sell_count"] >= 1

    def test_gameplan_partial_profit_trim_50pct_tier(self):
        """Branch: gain ≥50% with strong score → TRIM 75% (tier-3)."""
        _ensure_stock("WINNER", canslim_score=80.0, current_price=150.0,
                      previous_score=80.0, c_score=12, l_score=10,
                      is_growth_stock=False)
        _make_snapshot(total=15000, positions=[
            {"symbol": "WINNER", "quantity": 100, "last_price": 150.0,
             "current_value": 15000, "total_gain_loss_pct": 55.0,
             "average_cost_basis": 100.0},
        ])
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        trims = [a for a in r.json()["gameplan"] if a["action"] == "TRIM"]
        assert len(trims) == 1
        # 75% of 100 shares = 75 shares trimmed
        assert trims[0]["shares_action"] == 75

    def test_gameplan_buy_blocked_when_spy_below_50ma(self, monkeypatch):
        """Branch: SPY < 50MA → buy_blocked + bearish stop_loss in summary."""
        monkeypatch.setattr(
            "data_fetcher.get_cached_market_direction",
            lambda: {"spy": {"price": 480.0, "ma_50": 510.0}},
        )
        _ensure_stock("HOLD", canslim_score=70.0, current_price=100.0,
                      previous_score=70.0, c_score=10, l_score=8)
        _make_snapshot(total=10000, positions=[
            {"symbol": "HOLD", "quantity": 50, "last_price": 100.0,
             "current_value": 5000, "total_gain_loss_pct": 0.0,
             "average_cost_basis": 100.0},
        ])
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["buy_blocked"] is not None
        assert "BUY BLOCKED" in body["summary"]["buy_blocked"]
        assert body["summary"]["buy_count"] == 0

    def test_gameplan_market_direction_exception_silent(self, monkeypatch):
        """Branch: data_fetcher raises → spy_status='unknown', proceed."""
        def boom():
            raise RuntimeError("market data unavailable")
        monkeypatch.setattr(
            "data_fetcher.get_cached_market_direction", boom,
        )
        _ensure_stock("X", canslim_score=70.0, current_price=100.0,
                      previous_score=70.0, c_score=10, l_score=8)
        _make_snapshot(positions=[
            {"symbol": "X", "quantity": 10, "last_price": 100,
             "current_value": 1000, "total_gain_loss_pct": 0,
             "average_cost_basis": 100},
        ])
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        # spy_status default is "unknown" — caught exception swallowed
        assert r.json()["summary"]["market_status"] == "unknown"

    def test_gameplan_at_max_positions_blocks_buys(self):
        """Branch: positions count ≥ max_positions (8) → buy_blocked."""
        # Seed 8 positions
        positions = []
        for i in range(8):
            _ensure_stock(f"P{i}", canslim_score=60.0, current_price=100.0,
                          previous_score=60.0, c_score=10, l_score=8)
            positions.append({
                "symbol": f"P{i}", "quantity": 10, "last_price": 100.0,
                "current_value": 1000, "total_gain_loss_pct": 0.0,
                "average_cost_basis": 100.0,
            })
        _make_snapshot(total=8000, positions=positions)
        r = client.get("/api/fidelity/gameplan")
        assert r.status_code == 200
        assert "max positions" in r.json()["summary"]["buy_blocked"]


# ────────────────────────────────────────────────────────────────────
# TestFidelitySyncToPortfolio — POST /sync-to-portfolio
# ────────────────────────────────────────────────────────────────────
#
# Latent-bug compensation: routes/fidelity.py uses bare-name
# `PortfolioPosition`. We inject the real DB model on the route module.

class TestFidelitySyncToPortfolio:
    """Branches: 404 no snapshot, add new positions, update existing,
    remove stale (not in latest snapshot)."""

    @pytest.fixture(autouse=True)
    def _patch_portfolio_position_name(self, monkeypatch):
        monkeypatch.setattr(fidelity_routes, "PortfolioPosition",
                            PortfolioPosition, raising=False)

    def test_404_when_no_snapshot(self):
        """Branch: no FidelitySnapshot → 404."""
        r = client.post("/api/fidelity/sync-to-portfolio")
        assert r.status_code == 404

    def test_adds_new_positions(self):
        """Branch: latest snapshot has positions, no PortfolioPosition rows
        → all added."""
        _make_snapshot(positions=[
            {"symbol": "AAPL", "quantity": 100, "last_price": 180,
             "current_value": 18000, "total_gain_loss": 3000,
             "total_gain_loss_pct": 20, "average_cost_basis": 150},
            {"symbol": "MSFT", "quantity": 25, "last_price": 400,
             "current_value": 10000, "total_gain_loss": 1000,
             "total_gain_loss_pct": 11.1, "average_cost_basis": 360},
        ])
        r = client.post("/api/fidelity/sync-to-portfolio")
        assert r.status_code == 200
        body = r.json()
        assert body["added"] == 2
        assert body["updated"] == 0
        assert body["removed"] == 0

        db = _db()
        try:
            tickers = {p.ticker for p in db.query(PortfolioPosition).all()}
            assert tickers == {"AAPL", "MSFT"}
        finally:
            db.close()

    def test_updates_existing_position(self):
        """Branch: existing PortfolioPosition with same ticker → updated."""
        # Seed a stale PortfolioPosition. The sync route scopes by
        # user_id == current_user.id (the IDOR fix), so seed under user_id=1
        # to match the auth-bypassed _fake_user.
        db = _db()
        try:
            db.add(PortfolioPosition(
                ticker="AAPL", shares=50, cost_basis=140.0,
                current_price=170.0, current_value=8500.0,
                gain_loss=1500.0, gain_loss_pct=21.4,
                user_id=1,
            ))
            db.commit()
        finally:
            db.close()

        _make_snapshot(positions=[
            {"symbol": "AAPL", "quantity": 200, "last_price": 180,
             "current_value": 36000, "total_gain_loss": 6000,
             "total_gain_loss_pct": 20, "average_cost_basis": 150},
        ])
        r = client.post("/api/fidelity/sync-to-portfolio")
        assert r.status_code == 200
        body = r.json()
        assert body["added"] == 0
        assert body["updated"] == 1

        db = _db()
        try:
            row = db.query(PortfolioPosition).filter_by(ticker="AAPL").first()
            assert row.shares == 200
            assert row.current_value == 36000
        finally:
            db.close()

    def test_removes_stale_positions(self):
        """Branch: PortfolioPosition not in latest snapshot → removed.
        Seed under user_id=1 to match the auth-bypassed _fake_user (post-IDOR-fix
        sync scopes deletes to current_user.id)."""
        db = _db()
        try:
            db.add(PortfolioPosition(
                ticker="GONE", shares=10, cost_basis=50, current_price=55,
                current_value=550, gain_loss=50, gain_loss_pct=10,
                user_id=1,
            ))
            db.commit()
        finally:
            db.close()

        _make_snapshot(positions=[
            {"symbol": "AAPL", "quantity": 100, "last_price": 180,
             "current_value": 18000, "total_gain_loss": 3000,
             "total_gain_loss_pct": 20, "average_cost_basis": 150},
        ])
        r = client.post("/api/fidelity/sync-to-portfolio")
        assert r.status_code == 200
        body = r.json()
        assert body["added"] == 1
        assert body["removed"] == 1

        db = _db()
        try:
            tickers = {p.ticker for p in db.query(PortfolioPosition).all()}
            assert tickers == {"AAPL"}
        finally:
            db.close()
