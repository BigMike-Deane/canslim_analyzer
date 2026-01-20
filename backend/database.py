"""
Database models for CANSLIM Analyzer Web App
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, Text, ForeignKey, Index
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

    # Growth projection
    projected_growth = Column(Float)
    growth_confidence = Column(String)  # low, medium, high

    # Metadata
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Relationships
    scores = relationship("StockScore", back_populates="stock", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_stocks_sector', 'sector'),
        Index('ix_stocks_canslim', 'canslim_score'),
    )


class StockScore(Base):
    """Historical CANSLIM scores for tracking changes"""
    __tablename__ = "stock_scores"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)

    # CANSLIM breakdown
    total_score = Column(Float)
    c_score = Column(Float)
    a_score = Column(Float)
    n_score = Column(Float)
    s_score = Column(Float)
    l_score = Column(Float)
    i_score = Column(Float)
    m_score = Column(Float)

    # Growth projection at this point
    projected_growth = Column(Float)
    current_price = Column(Float)

    # Relationships
    stock = relationship("Stock", back_populates="scores")

    __table_args__ = (
        Index('ix_stock_scores_stock_date', 'stock_id', 'date'),
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
    """Daily market direction snapshot"""
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True, nullable=False)

    # S&P 500 data
    spy_price = Column(Float)
    spy_50_ma = Column(Float)
    spy_200_ma = Column(Float)

    # Market score (M in CANSLIM)
    market_score = Column(Float)
    market_trend = Column(String)  # bullish, neutral, bearish

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
    current_score = Column(Float)

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
    canslim_score = Column(Float)  # Score at time of trade

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
