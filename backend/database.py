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
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


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
