"""
Database models for CANSLIM Analyzer Web App
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, Text, ForeignKey, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, date
from pathlib import Path

# Database setup
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR}/canslim.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI endpoints"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and run migrations"""
    Base.metadata.create_all(bind=engine)
    run_migrations()


def run_migrations():
    """Add any missing columns to existing tables and fix constraints"""
    import sqlite3
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    db_path = DATA_DIR / "canslim.db"
    if not db_path.exists():
        logger.info("No database yet, skipping migrations")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Define column migrations: (table, column, type)
    migrations = [
        ("stocks", "previous_score", "FLOAT"),
        ("stocks", "score_change", "FLOAT"),
        ("ai_portfolio_snapshots", "timestamp", "DATETIME"),
        ("ai_portfolio_snapshots", "prev_value", "FLOAT"),
        ("ai_portfolio_snapshots", "value_change", "FLOAT"),
        ("ai_portfolio_snapshots", "value_change_pct", "FLOAT"),
        # StockScore enhancements for backtesting
        ("stock_scores", "timestamp", "DATETIME"),
        ("stock_scores", "week_52_high", "FLOAT"),
        # Growth Mode Scoring
        ("stocks", "growth_mode_score", "FLOAT"),
        ("stocks", "growth_mode_details", "TEXT"),  # JSON stored as TEXT in SQLite
        ("stocks", "is_growth_stock", "BOOLEAN"),
        # Enhanced Earnings Analysis
        ("stocks", "eps_acceleration", "BOOLEAN"),
        ("stocks", "earnings_surprise_pct", "FLOAT"),
        ("stocks", "revenue_growth_pct", "FLOAT"),
        # Technical Analysis
        ("stocks", "volume_ratio", "FLOAT"),
        ("stocks", "weeks_in_base", "INTEGER"),
        ("stocks", "base_type", "TEXT"),
        ("stocks", "is_breaking_out", "BOOLEAN"),
        ("stocks", "breakout_volume_ratio", "FLOAT"),
        # AI Portfolio Growth Mode support
        ("ai_portfolio_positions", "is_growth_stock", "BOOLEAN DEFAULT 0"),
        ("ai_portfolio_positions", "purchase_growth_score", "FLOAT"),
        ("ai_portfolio_positions", "current_growth_score", "FLOAT"),
        ("ai_portfolio_trades", "growth_mode_score", "FLOAT"),
        ("ai_portfolio_trades", "is_growth_stock", "BOOLEAN DEFAULT 0"),
        # Multi-index market direction
        ("market_snapshots", "timestamp", "DATETIME"),
        ("market_snapshots", "spy_signal", "INTEGER"),
        ("market_snapshots", "qqq_price", "FLOAT"),
        ("market_snapshots", "qqq_50_ma", "FLOAT"),
        ("market_snapshots", "qqq_200_ma", "FLOAT"),
        ("market_snapshots", "qqq_signal", "INTEGER"),
        ("market_snapshots", "dia_price", "FLOAT"),
        ("market_snapshots", "dia_50_ma", "FLOAT"),
        ("market_snapshots", "dia_200_ma", "FLOAT"),
        ("market_snapshots", "dia_signal", "INTEGER"),
        ("market_snapshots", "weighted_signal", "FLOAT"),
        # Trailing stop loss tracking (AI Portfolio)
        ("ai_portfolio_positions", "peak_price", "FLOAT"),
        ("ai_portfolio_positions", "peak_date", "DATETIME"),
        # Insider trading signals
        ("stocks", "insider_buy_count", "INTEGER"),
        ("stocks", "insider_sell_count", "INTEGER"),
        ("stocks", "insider_net_shares", "FLOAT"),
        ("stocks", "insider_sentiment", "TEXT"),
        ("stocks", "insider_updated_at", "DATETIME"),
        # Short interest tracking
        ("stocks", "short_interest_pct", "FLOAT"),
        ("stocks", "short_ratio", "FLOAT"),
        ("stocks", "short_updated_at", "DATETIME"),
        # Score details for clickable breakdown
        ("stocks", "score_details", "TEXT"),  # JSON stored as TEXT in SQLite
        # Earnings/Revenue JSON columns
        ("stocks", "quarterly_earnings", "TEXT"),  # JSON stored as TEXT
        ("stocks", "annual_earnings", "TEXT"),  # JSON stored as TEXT
        ("stocks", "quarterly_revenue", "TEXT"),  # JSON stored as TEXT
        # Backtest cancellation support
        ("backtest_runs", "cancel_requested", "BOOLEAN DEFAULT 0"),
    ]

    for table, column, col_type in migrations:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            logger.info(f"Migration: Added {table}.{column}")
        except sqlite3.OperationalError:
            pass

    # Fix: Remove unique constraint on ai_portfolio_snapshots.date
    # Check if the old unique index exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='ix_ai_portfolio_snapshots_date' AND sql LIKE '%UNIQUE%'")
    has_unique = cursor.fetchone()

    # Also check for implicit unique constraint in table definition
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='ai_portfolio_snapshots'")
    table_sql = cursor.fetchone()
    needs_rebuild = has_unique or (table_sql and 'UNIQUE' in table_sql[0] and 'date' in table_sql[0].lower())

    if needs_rebuild:
        logger.info("Rebuilding ai_portfolio_snapshots table to remove unique constraint on date")
        try:
            # Create new table without unique constraint
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_portfolio_snapshots_new (
                    id INTEGER PRIMARY KEY,
                    timestamp DATETIME,
                    total_value FLOAT NOT NULL,
                    cash FLOAT NOT NULL,
                    positions_value FLOAT NOT NULL,
                    positions_count INTEGER NOT NULL,
                    total_return FLOAT,
                    total_return_pct FLOAT,
                    prev_value FLOAT,
                    value_change FLOAT,
                    value_change_pct FLOAT,
                    date DATE
                )
            ''')

            # Get existing columns from old table
            cursor.execute('PRAGMA table_info(ai_portfolio_snapshots)')
            old_cols = [row[1] for row in cursor.fetchall()]

            # Only copy columns that exist in both tables
            new_cols = ['id', 'timestamp', 'total_value', 'cash', 'positions_value', 'positions_count',
                        'total_return', 'total_return_pct', 'prev_value', 'value_change', 'value_change_pct', 'date']
            common_cols = [c for c in new_cols if c in old_cols]
            cols_str = ', '.join(common_cols)

            cursor.execute(f'''
                INSERT INTO ai_portfolio_snapshots_new ({cols_str})
                SELECT {cols_str} FROM ai_portfolio_snapshots
            ''')

            # Drop old table and rename new
            cursor.execute('DROP TABLE ai_portfolio_snapshots')
            cursor.execute('ALTER TABLE ai_portfolio_snapshots_new RENAME TO ai_portfolio_snapshots')

            # Recreate indexes (non-unique)
            cursor.execute('CREATE INDEX IF NOT EXISTS ix_ai_portfolio_snapshots_timestamp ON ai_portfolio_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS ix_ai_portfolio_snapshots_date ON ai_portfolio_snapshots(date)')

            logger.info("Successfully rebuilt ai_portfolio_snapshots table")
        except Exception as e:
            logger.error(f"Failed to rebuild ai_portfolio_snapshots: {e}")

    # Create indexes that may not exist on older databases
    index_migrations = [
        ('ix_stocks_sector', 'stocks', 'sector'),
        ('ix_stocks_canslim', 'stocks', 'canslim_score'),
        ('ix_stocks_price', 'stocks', 'current_price'),
        ('ix_stocks_score_price', 'stocks', 'canslim_score, current_price'),
        ('ix_stock_scores_stock_date', 'stock_scores', 'stock_id, date'),
        ('ix_stock_scores_stock_timestamp', 'stock_scores', 'stock_id, timestamp'),
        ('ix_stocks_growth_mode', 'stocks', 'growth_mode_score'),
        ('ix_stocks_breaking_out', 'stocks', 'is_breaking_out'),
        # Backtest indexes
        ('ix_backtest_runs_status', 'backtest_runs', 'status'),
        ('ix_backtest_snapshots_backtest_date', 'backtest_snapshots', 'backtest_id, date'),
        ('ix_backtest_trades_backtest_date', 'backtest_trades', 'backtest_id, date'),
        ('ix_backtest_positions_backtest', 'backtest_positions', 'backtest_id'),
    ]
    for idx_name, table, columns in index_migrations:
        try:
            cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})')
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()
    logger.info("Database migrations complete")


# ============== Models ==============

class Stock(Base):
    """Cached stock information"""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    market_cap = Column(Float)
    current_price = Column(Float)
    week_52_high = Column(Float)
    week_52_low = Column(Float)

    # Latest CANSLIM data
    canslim_score = Column(Float)
    previous_score = Column(Float)  # Score from previous scan
    score_change = Column(Float)  # Change from previous scan
    c_score = Column(Float)  # Current earnings
    a_score = Column(Float)  # Annual earnings
    n_score = Column(Float)  # New highs
    s_score = Column(Float)  # Supply/demand
    l_score = Column(Float)  # Leader/laggard
    i_score = Column(Float)  # Institutional
    m_score = Column(Float)  # Market direction
    score_details = Column(JSON)  # Detailed breakdown for each component {c: "...", a: "...", etc}

    # Growth projection
    projected_growth = Column(Float)
    growth_confidence = Column(String)  # low, medium, high

    # Growth Mode Scoring (alternative scoring for pre-revenue companies)
    growth_mode_score = Column(Float)  # 0-100 score using growth mode criteria
    growth_mode_details = Column(JSON)  # Component breakdown: R, F, N, S, L, I, M
    is_growth_stock = Column(Boolean, default=False)  # True if pre-revenue or high-growth

    # Enhanced Earnings Analysis
    eps_acceleration = Column(Boolean)  # True if EPS accelerating quarter over quarter
    earnings_surprise_pct = Column(Float)  # Latest earnings surprise %
    revenue_growth_pct = Column(Float)  # YoY revenue growth %
    quarterly_earnings = Column(JSON)  # List of quarterly EPS values (last 4-8 quarters)
    annual_earnings = Column(JSON)  # List of annual EPS values (last 3-5 years)
    quarterly_revenue = Column(JSON)  # List of quarterly revenue values

    # Technical Analysis
    volume_ratio = Column(Float)  # Current volume vs 50-day average
    weeks_in_base = Column(Integer)  # Weeks of consolidation
    base_type = Column(String)  # 'flat', 'cup', 'none'
    is_breaking_out = Column(Boolean, default=False)  # Price breaking out with volume
    breakout_volume_ratio = Column(Float)  # Volume surge on breakout day

    # Insider Trading Signals
    insider_buy_count = Column(Integer)  # Insider buys in last 3 months
    insider_sell_count = Column(Integer)  # Insider sells in last 3 months
    insider_net_shares = Column(Float)  # Net shares bought/sold
    insider_sentiment = Column(String)  # 'bullish', 'bearish', 'neutral'
    insider_updated_at = Column(DateTime)

    # Short Interest
    short_interest_pct = Column(Float)  # Short interest as % of float
    short_ratio = Column(Float)  # Days to cover
    short_updated_at = Column(DateTime)

    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Relationships
    scores = relationship("StockScore", back_populates="stock", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_stocks_sector', 'sector'),
        Index('ix_stocks_canslim', 'canslim_score'),
        Index('ix_stocks_price', 'current_price'),
        Index('ix_stocks_score_price', 'canslim_score', 'current_price'),  # Composite for filtered queries
    )


class StockScore(Base):
    """Historical CANSLIM scores for tracking changes - one record per scan"""
    __tablename__ = "stock_scores"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    date = Column(Date, nullable=False, index=True)  # Kept for easy daily grouping

    # CANSLIM breakdown
    total_score = Column(Float)
    c_score = Column(Float)
    a_score = Column(Float)
    n_score = Column(Float)
    s_score = Column(Float)
    l_score = Column(Float)
    i_score = Column(Float)
    m_score = Column(Float)

    # Growth projection and price at this point
    projected_growth = Column(Float)
    current_price = Column(Float)
    week_52_high = Column(Float)  # Track breakout proximity over time

    # Relationships
    stock = relationship("Stock", back_populates="scores")

    __table_args__ = (
        Index('ix_stock_scores_stock_date', 'stock_id', 'date'),
        Index('ix_stock_scores_stock_timestamp', 'stock_id', 'timestamp'),
    )


class PortfolioPosition(Base):
    """User's portfolio positions"""
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    shares = Column(Float, nullable=False)
    cost_basis = Column(Float)  # Average cost per share
    purchase_date = Column(Date)
    notes = Column(Text)

    # Cached current data
    current_price = Column(Float)
    current_value = Column(Float)
    gain_loss = Column(Float)
    gain_loss_pct = Column(Float)

    # CANSLIM recommendation
    recommendation = Column(String)  # buy, hold, sell
    canslim_score = Column(Float)
    score_change = Column(Float)  # vs last check

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Watchlist(Base):
    """Stocks user is watching"""
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    target_price = Column(Float)  # Alert when reaches this price
    alert_score = Column(Float)  # Alert when CANSLIM score reaches this


class AnalysisJob(Base):
    """Track background analysis jobs"""
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String)  # full_scan, portfolio_update, single_stock
    status = Column(String, default="pending")  # pending, running, completed, failed
    tickers_total = Column(Integer)
    tickers_processed = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketSnapshot(Base):
    """Daily market direction snapshot with multi-index support"""
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # S&P 500 data (SPY) - 50% weight
    spy_price = Column(Float)
    spy_50_ma = Column(Float)
    spy_200_ma = Column(Float)
    spy_signal = Column(Integer)  # -1 bearish, 0 neutral, 1 bullish, 2 strong bullish

    # NASDAQ 100 data (QQQ) - 30% weight
    qqq_price = Column(Float)
    qqq_50_ma = Column(Float)
    qqq_200_ma = Column(Float)
    qqq_signal = Column(Integer)

    # Dow Jones data (DIA) - 20% weight
    dia_price = Column(Float)
    dia_50_ma = Column(Float)
    dia_200_ma = Column(Float)
    dia_signal = Column(Integer)

    # Combined market score (M in CANSLIM)
    market_score = Column(Float)  # 0-15 CANSLIM score
    market_trend = Column(String)  # bullish, neutral, bearish
    weighted_signal = Column(Float)  # Combined weighted signal

    created_at = Column(DateTime, default=datetime.utcnow)


# ============== AI Portfolio Models ==============

class AIPortfolioConfig(Base):
    """AI Portfolio configuration"""
    __tablename__ = "ai_portfolio_config"

    id = Column(Integer, primary_key=True, index=True)
    starting_cash = Column(Float, default=25000.0)
    current_cash = Column(Float, default=25000.0)
    max_positions = Column(Integer, default=15)
    max_position_pct = Column(Float, default=10.0)  # Max % of portfolio per position
    min_score_to_buy = Column(Integer, default=75)
    sell_score_threshold = Column(Integer, default=50)  # Sell if score drops below
    take_profit_pct = Column(Float, default=25.0)  # Take profits at this gain %
    stop_loss_pct = Column(Float, default=15.0)  # Stop loss at this loss %
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AIPortfolioPosition(Base):
    """AI Portfolio current positions"""
    __tablename__ = "ai_portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    shares = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False)  # Price per share when bought
    purchase_date = Column(DateTime, default=datetime.utcnow)
    purchase_score = Column(Float)  # CANSLIM score when purchased

    # Current values (updated on each scan)
    current_price = Column(Float)
    current_value = Column(Float)
    gain_loss = Column(Float)
    gain_loss_pct = Column(Float)
    current_score = Column(Float)  # CANSLIM score

    # Growth Mode scoring (for pre-revenue/high-growth stocks)
    is_growth_stock = Column(Boolean, default=False)
    purchase_growth_score = Column(Float)  # Growth Mode score when purchased
    current_growth_score = Column(Float)  # Current Growth Mode score

    # Trailing stop loss tracking
    peak_price = Column(Float)  # Highest price since purchase (for trailing stop)
    peak_date = Column(DateTime)  # When peak was reached

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AIPortfolioTrade(Base):
    """AI Portfolio trade history"""
    __tablename__ = "ai_portfolio_trades"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)  # BUY, SELL
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    reason = Column(String)  # Why the trade was made
    canslim_score = Column(Float)  # CANSLIM score at time of trade
    growth_mode_score = Column(Float)  # Growth Mode score at time of trade
    is_growth_stock = Column(Boolean, default=False)

    # For sells, track the gain/loss
    cost_basis = Column(Float)  # Original cost basis for sells
    realized_gain = Column(Float)  # Profit/loss on the trade

    executed_at = Column(DateTime, default=datetime.utcnow, index=True)


class AIPortfolioSnapshot(Base):
    """AI Portfolio snapshots for performance chart - taken after each scan"""
    __tablename__ = "ai_portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, nullable=False, default=datetime.utcnow)

    total_value = Column(Float, nullable=False)  # Cash + positions
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    positions_count = Column(Integer, nullable=False)

    # Performance metrics
    total_return = Column(Float)  # Total return since inception
    total_return_pct = Column(Float)
    prev_value = Column(Float)  # Previous snapshot value for change calc
    value_change = Column(Float)  # Change from previous snapshot
    value_change_pct = Column(Float)

    # Keep date for backwards compatibility with existing chart
    date = Column(Date, index=True)  # Date portion for grouping


# ============== Backtesting Models ==============

class BacktestRun(Base):
    """A single backtest execution with configuration and results"""
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)  # User-friendly name
    status = Column(String, default="pending")  # pending, running, completed, failed

    # Configuration
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    starting_cash = Column(Float, default=25000.0)
    stock_universe = Column(String, default="sp500")  # sp500, all, custom
    custom_tickers = Column(JSON)  # If universe is custom

    # AI Config snapshot (frozen at backtest start)
    max_positions = Column(Integer, default=20)
    max_position_pct = Column(Float, default=12.0)
    min_score_to_buy = Column(Integer, default=65)
    sell_score_threshold = Column(Integer, default=45)
    stop_loss_pct = Column(Float, default=10.0)

    # Results summary (populated on completion)
    final_value = Column(Float)
    total_return_pct = Column(Float)
    max_drawdown_pct = Column(Float)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)  # % of profitable trades
    total_trades = Column(Integer)

    # Benchmark comparison
    spy_final_value = Column(Float)
    spy_return_pct = Column(Float)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    progress_pct = Column(Float, default=0.0)  # 0-100 progress during run
    cancel_requested = Column(Boolean, default=False)  # Flag to request cancellation

    # Relationships (cascade delete when backtest is deleted)
    daily_snapshots = relationship("BacktestSnapshot", back_populates="backtest_run", cascade="all, delete-orphan")
    trades = relationship("BacktestTrade", back_populates="backtest_run", cascade="all, delete-orphan")
    positions = relationship("BacktestPosition", back_populates="backtest_run", cascade="all, delete-orphan")


class BacktestSnapshot(Base):
    """Daily portfolio snapshot during backtest for performance chart"""
    __tablename__ = "backtest_snapshots"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Portfolio state
    total_value = Column(Float)
    cash = Column(Float)
    positions_value = Column(Float)
    positions_count = Column(Integer)

    # Performance metrics
    daily_return_pct = Column(Float)
    cumulative_return_pct = Column(Float)

    # Benchmark comparison (SPY buy-and-hold)
    spy_price = Column(Float)
    spy_value = Column(Float)  # Value if had bought SPY at start
    spy_return_pct = Column(Float)

    backtest_run = relationship("BacktestRun", back_populates="daily_snapshots")

    __table_args__ = (
        Index('ix_backtest_snapshots_backtest_date', 'backtest_id', 'date'),
    )


class BacktestTrade(Base):
    """Individual trade during backtest"""
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)

    date = Column(Date, nullable=False, index=True)
    ticker = Column(String, nullable=False)
    action = Column(String, nullable=False)  # BUY, SELL, PYRAMID
    shares = Column(Float)
    price = Column(Float)
    total_value = Column(Float)
    reason = Column(String)

    # Score at time of trade
    canslim_score = Column(Float)
    growth_mode_score = Column(Float)
    is_growth_stock = Column(Boolean, default=False)

    # For sells - realized P&L
    cost_basis = Column(Float)
    realized_gain = Column(Float)
    realized_gain_pct = Column(Float)
    holding_days = Column(Integer)

    backtest_run = relationship("BacktestRun", back_populates="trades")

    __table_args__ = (
        Index('ix_backtest_trades_backtest_date', 'backtest_id', 'date'),
    )


class BacktestPosition(Base):
    """Current positions during backtest simulation (cleared between runs)"""
    __tablename__ = "backtest_positions"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)

    ticker = Column(String, nullable=False)
    shares = Column(Float)
    cost_basis = Column(Float)
    purchase_date = Column(Date)
    purchase_score = Column(Float)

    # For trailing stop calculation
    peak_price = Column(Float)
    peak_date = Column(Date)

    # Growth mode
    is_growth_stock = Column(Boolean, default=False)
    purchase_growth_score = Column(Float)

    # Sector for allocation tracking
    sector = Column(String)

    backtest_run = relationship("BacktestRun", back_populates="positions")


# ============== Data Caching Models ==============

class StockDataCache(Base):
    """
    Persistent cache for raw stock data fetched from APIs.
    Survives container restarts and enables delta checking.
    """
    __tablename__ = "stock_data_cache"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)

    # Earnings data (refreshed daily)
    quarterly_earnings = Column(JSON)  # List of quarterly EPS values
    annual_earnings = Column(JSON)  # List of annual EPS values
    earnings_updated_at = Column(DateTime)

    # Revenue data (refreshed daily)
    quarterly_revenue = Column(JSON)  # List of quarterly revenue values
    annual_revenue = Column(JSON)  # List of annual revenue values
    revenue_updated_at = Column(DateTime)

    # Balance sheet data (refreshed daily)
    total_cash = Column(Float)
    total_debt = Column(Float)
    shares_outstanding = Column(Float)
    balance_updated_at = Column(DateTime)

    # Analyst data (refreshed daily)
    analyst_target_price = Column(Float)
    analyst_count = Column(Integer)
    analyst_updated_at = Column(DateTime)

    # Institutional data (refreshed weekly)
    institutional_holders_pct = Column(Float)
    institutional_updated_at = Column(DateTime)

    # Key metrics (refreshed daily)
    roe = Column(Float)
    trailing_pe = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    metrics_updated_at = Column(DateTime)

    # Hash of critical data for delta detection
    # If this hasn't changed, we can skip re-scoring
    data_hash = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
