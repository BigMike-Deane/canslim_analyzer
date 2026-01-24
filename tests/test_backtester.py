"""
Tests for the CANSLIM Backtesting System
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Test HistoricalDataProvider
class TestHistoricalDataProvider:
    """Tests for the historical data provider"""

    def test_init(self):
        """Test provider initialization"""
        from backend.historical_data import HistoricalDataProvider

        tickers = ["AAPL", "MSFT", "GOOGL"]
        provider = HistoricalDataProvider(tickers)

        assert provider.tickers == tickers
        assert provider._is_loaded is False
        assert len(provider._price_cache) == 0

    def test_calculate_index_signal_bullish(self):
        """Test bullish signal calculation"""
        from backend.historical_data import HistoricalDataProvider

        provider = HistoricalDataProvider([])

        # Price above both MAs = strong bullish
        signal = provider._calculate_index_signal(price=100, ma_50=95, ma_200=90)
        assert signal == 2

    def test_calculate_index_signal_bearish(self):
        """Test bearish signal calculation"""
        from backend.historical_data import HistoricalDataProvider

        provider = HistoricalDataProvider([])

        # Price below both MAs = bearish
        signal = provider._calculate_index_signal(price=80, ma_50=90, ma_200=95)
        assert signal == -1

    def test_calculate_index_signal_neutral(self):
        """Test neutral signal calculation"""
        from backend.historical_data import HistoricalDataProvider

        provider = HistoricalDataProvider([])

        # Price above 50 but below 200 = neutral
        signal = provider._calculate_index_signal(price=92, ma_50=90, ma_200=95)
        assert signal == 0


# Test BacktestEngine
class TestBacktestEngine:
    """Tests for the backtesting engine"""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        from backend.database import BacktestRun

        mock_session = MagicMock()

        # Create a mock backtest
        mock_backtest = BacktestRun(
            id=1,
            name="Test Backtest",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            starting_cash=25000.0,
            stock_universe="sp500",
            max_positions=20,
            min_score_to_buy=65,
            sell_score_threshold=45,
            stop_loss_pct=10.0,
            status="pending"
        )

        mock_session.query.return_value.get.return_value = mock_backtest
        mock_session.query.return_value.all.return_value = []

        return mock_session, mock_backtest

    def test_engine_init(self, mock_db):
        """Test engine initialization"""
        from backend.backtester import BacktestEngine

        mock_session, mock_backtest = mock_db

        engine = BacktestEngine(mock_session, 1)

        assert engine.cash == 25000.0
        assert len(engine.positions) == 0
        assert engine.peak_portfolio_value == 25000.0

    def test_get_portfolio_value(self, mock_db):
        """Test portfolio value calculation"""
        from backend.backtester import BacktestEngine, SimulatedPosition

        mock_session, mock_backtest = mock_db

        engine = BacktestEngine(mock_session, 1)
        engine.cash = 10000.0

        # Add a position
        engine.positions["AAPL"] = SimulatedPosition(
            ticker="AAPL",
            shares=100,
            cost_basis=150.0,
            purchase_date=date.today(),
            purchase_score=75.0,
            peak_price=160.0,
            peak_date=date.today(),
            sector="Technology"
        )

        # Portfolio value = cash + position value (using peak price as approximation)
        value = engine._get_portfolio_value()
        assert value == 10000.0 + (100 * 160.0)  # $26,000

    def test_check_sector_limit(self, mock_db):
        """Test sector limit checking"""
        from backend.backtester import BacktestEngine, SimulatedPosition, MAX_STOCKS_PER_SECTOR

        mock_session, mock_backtest = mock_db

        engine = BacktestEngine(mock_session, 1)

        # Add positions in Technology sector
        for i in range(MAX_STOCKS_PER_SECTOR):
            engine.positions[f"TECH{i}"] = SimulatedPosition(
                ticker=f"TECH{i}",
                shares=10,
                cost_basis=100.0,
                purchase_date=date.today(),
                purchase_score=70.0,
                peak_price=100.0,
                peak_date=date.today(),
                sector="Technology"
            )

        # Should not allow another Technology stock
        assert engine._check_sector_limit("Technology") is False

        # Should allow a different sector
        assert engine._check_sector_limit("Healthcare") is True


class TestSimulatedPosition:
    """Tests for SimulatedPosition dataclass"""

    def test_position_creation(self):
        """Test creating a simulated position"""
        from backend.backtester import SimulatedPosition

        position = SimulatedPosition(
            ticker="AAPL",
            shares=100,
            cost_basis=150.0,
            purchase_date=date.today(),
            purchase_score=80.0,
            peak_price=160.0,
            peak_date=date.today(),
            is_growth_stock=False,
            sector="Technology"
        )

        assert position.ticker == "AAPL"
        assert position.shares == 100
        assert position.cost_basis == 150.0
        assert position.peak_price == 160.0


class TestTrailingStopLogic:
    """Tests for trailing stop loss logic"""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        from backend.database import BacktestRun

        mock_session = MagicMock()
        mock_backtest = BacktestRun(
            id=1,
            name="Test Backtest",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            starting_cash=25000.0,
            stock_universe="sp500",
            max_positions=20,
            min_score_to_buy=65,
            sell_score_threshold=45,
            stop_loss_pct=10.0,
            status="pending"
        )

        mock_session.query.return_value.get.return_value = mock_backtest
        mock_session.query.return_value.all.return_value = []

        return mock_session, mock_backtest

    def test_trailing_stop_50_pct_winner(self, mock_db):
        """Test 15% trailing stop for 50%+ winners"""
        from backend.backtester import BacktestEngine, SimulatedPosition

        mock_session, mock_backtest = mock_db
        engine = BacktestEngine(mock_session, 1)

        # Position bought at $100, peaked at $160 (60% gain)
        position = SimulatedPosition(
            ticker="AAPL",
            shares=100,
            cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=30),
            purchase_score=80.0,
            peak_price=160.0,
            peak_date=date.today() - timedelta(days=5),
            sector="Technology"
        )
        engine.positions["AAPL"] = position

        # Mock data provider
        engine.data_provider = MagicMock()

        # Current price at $136 = 15% drop from peak
        # Should trigger trailing stop (15% threshold for 50%+ gains)
        engine.data_provider.get_price_on_date.return_value = 136.0

        scores = {"AAPL": {"total_score": 70}}
        sells = engine._evaluate_sells(date.today(), scores)

        assert len(sells) == 1
        assert "TRAILING STOP" in sells[0].reason

    def test_stop_loss_triggers(self, mock_db):
        """Test hard stop loss at 10%"""
        from backend.backtester import BacktestEngine, SimulatedPosition

        mock_session, mock_backtest = mock_db
        engine = BacktestEngine(mock_session, 1)

        # Position bought at $100, now at $89 (11% loss)
        position = SimulatedPosition(
            ticker="AAPL",
            shares=100,
            cost_basis=100.0,
            purchase_date=date.today() - timedelta(days=10),
            purchase_score=80.0,
            peak_price=105.0,
            peak_date=date.today() - timedelta(days=7),
            sector="Technology"
        )
        engine.positions["AAPL"] = position

        # Mock data provider
        engine.data_provider = MagicMock()
        engine.data_provider.get_price_on_date.return_value = 89.0  # 11% loss

        scores = {"AAPL": {"total_score": 60}}
        sells = engine._evaluate_sells(date.today(), scores)

        assert len(sells) == 1
        assert "STOP LOSS" in sells[0].reason


class TestDatabaseModels:
    """Tests for backtest database models"""

    def test_backtest_run_model(self):
        """Test BacktestRun model creation"""
        from backend.database import BacktestRun

        backtest = BacktestRun(
            name="Test Backtest",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            starting_cash=25000.0,
            stock_universe="sp500",
            status="pending"  # Explicit status for testing
        )

        assert backtest.name == "Test Backtest"
        assert backtest.status == "pending"
        assert backtest.starting_cash == 25000.0

    def test_backtest_snapshot_model(self):
        """Test BacktestSnapshot model creation"""
        from backend.database import BacktestSnapshot

        snapshot = BacktestSnapshot(
            backtest_id=1,
            date=date.today(),
            total_value=26000.0,
            cash=5000.0,
            positions_value=21000.0,
            positions_count=5,
            cumulative_return_pct=4.0
        )

        assert snapshot.total_value == 26000.0
        assert snapshot.positions_count == 5

    def test_backtest_trade_model(self):
        """Test BacktestTrade model creation"""
        from backend.database import BacktestTrade

        trade = BacktestTrade(
            backtest_id=1,
            date=date.today(),
            ticker="AAPL",
            action="BUY",
            shares=100,
            price=150.0,
            total_value=15000.0,
            reason="Score 85 | 5% from high"
        )

        assert trade.ticker == "AAPL"
        assert trade.action == "BUY"
        assert trade.total_value == 15000.0
