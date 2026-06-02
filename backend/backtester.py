"""
CANSLIM Backtesting Engine

Simulates historical trading using the AI Portfolio logic.
Runs day-by-day simulation over a historical period.
"""

import copy
import logging
import math
import sys
import os
import sqlite3
import json
from collections import deque
from datetime import date, datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

# Add parent directory to path for config_loader import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config

# Shared utilities (also used by ai_trader.py)
from backend.trading_utils import (
    _nan_safe,
    get_strategy_profile,
    ENTRY_TYPE_MAP_ML,
    REGIME_MAP_ML,
    MAX_SECTOR_ALLOCATION,
    MAX_STOCKS_PER_SECTOR,
    MAX_POSITION_ALLOCATION,
    should_take_partial_on_trailing_stop,
    apply_sector_allocation_cap,
    select_effective_stop_loss_pct,
)

# Shared trading engine logic (also used by ai_trader.py)
from backend.trading_engine import (
    get_trailing_stop_pct,
    apply_pyramid_widening,
    check_score_stability_from_history,
    get_partial_profit_action,
    calculate_base_quality_bonus,
    calculate_entry_signals,
    calculate_composite_score,
    calculate_position_size_pct,
    apply_position_size_multipliers,
    get_tightened_trailing_stop,
    evaluate_score_crash,
    sanitize_signal_factors,
)

from backend.database import (
    BacktestRun, BacktestSnapshot, BacktestTrade, Stock, StockScore
)
from sqlalchemy import func
from backend.historical_data import HistoricalDataProvider
from backend.market_state import MarketStateManager, MarketState
from backend.bear_base import compute_readiness_score
from canslim_scorer import CANSLIMScorer, calculate_coiled_spring_score

logger = logging.getLogger(__name__)

# Bump this version when scoring logic changes AND you've ensured Layer 4
# (fresh compute) is the desired source of truth. Be careful: this cache is
# load-bearing. The backtester depends on it for the entire 2022-Mar 2026
# replay window because StockScore (Layer 2) only retains the last ~30 days.
# Bumping to a fresh version means falling through to Layer 4 for every
# (ticker, date) the cache used to serve — and Layer 4 is semantically wrong
# for historical replay (see _calculate_scores docstring).
#
# v1 = production-validated cache (~2 GB, accumulated from past backtests,
#       served the +136.84% baseline). KEEP using this until BacktestStaticSnapshot
#       is extended to capture historically-correct point-in-time scalars,
#       at which point Layer 4 becomes valid and the cache can be retired.
SCORE_CACHE_VERSION = 1
SCORE_CACHE_DIR = "/tmp/backtest_cache"

# Lazy import for graceful fallback when ML dependencies not installed
try:
    from ml.model import get_ml_prediction, clear_prediction_cache
except ImportError:
    get_ml_prediction = lambda **kwargs: None
    clear_prediction_cache = lambda: None


@dataclass
class SimulatedPosition:
    """In-memory position for simulation"""
    ticker: str
    shares: float
    cost_basis: float
    purchase_date: date
    purchase_score: float
    peak_price: float
    peak_date: date
    is_growth_stock: bool = False
    sector: str = ""
    partial_profit_taken: float = 0.0  # Cumulative % of position sold as partial profits
    pyramid_count: int = 0  # Number of times pyramided into
    signal_factors: dict = field(default_factory=dict)  # Trade journal: factors that drove the buy
    is_experimental: bool = False  # True for nibble/V-bottom positions (isolated from circuit breaker)


@dataclass
class SimulatedTrade:
    """Trade to be executed"""
    ticker: str
    action: str  # BUY, SELL, PYRAMID
    shares: float
    price: float
    reason: str
    score: float = 0.0
    priority: int = 0
    is_growth_stock: bool = False
    is_partial: bool = False  # Whether this is a partial sell
    sell_pct: float = 100.0  # Percentage to sell (for partial sells)


def create_backtest_static_snapshot(db: Session, backtest_id: int) -> int:
    """Snapshot every static_data input for every stock at backtest creation.
    Backtester._load_static_data prefers these snapshot rows over the live
    cache, so the run is reproducible regardless of when it executes vs
    when it was created.

    Captures all fields that _load_static_data reads:
      - P1 fields (days_to_earnings, beat_streak, revision_pct, rank, weeks_in_base)
      - Static metadata (sector)
      - Cache scalars (roe, analyst_target_price, num_analyst_opinions)
      - Earnings/revenue arrays (JSON-encoded)
      - score_details JSON (used for institutional_holders_pct)

    Idempotent — returns 0 if a snapshot already exists for this
    backtest_id (avoids duplicate-key errors on re-trigger).
    """
    from backend.database import BacktestStaticSnapshot, Stock, StockDataCache

    existing = db.query(BacktestStaticSnapshot.id).filter(
        BacktestStaticSnapshot.backtest_id == backtest_id
    ).first()
    if existing is not None:
        logger.debug(f"Snapshot already exists for backtest {backtest_id}, skipping")
        return 0

    stocks = db.query(Stock).all()
    cache_by_ticker = {c.ticker: c for c in db.query(StockDataCache).all()}

    def _json_or_none(value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return value if isinstance(value, str) else value.decode('utf-8', errors='ignore')
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return None

    snapshots = []
    for s in stocks:
        cache = cache_by_ticker.get(s.ticker)
        snapshots.append(BacktestStaticSnapshot(
            backtest_id=backtest_id,
            ticker=s.ticker,
            # P1
            days_to_earnings=getattr(s, 'days_to_earnings', None),
            earnings_beat_streak=getattr(s, 'earnings_beat_streak', None),
            eps_estimate_revision_pct=getattr(s, 'eps_estimate_revision_pct', None),
            industry_group_rank=getattr(s, 'industry_group_rank', None),
            weeks_in_base=getattr(s, 'weeks_in_base', None),
            # Extended
            sector=getattr(s, 'sector', None),
            roe=(cache.roe if cache else None),
            analyst_target_price=(getattr(cache, 'analyst_target_price', None) if cache else None),
            # StockDataCache column is `analyst_count`; the historical_data
            # / static_data dict key is `num_analyst_opinions`. Snapshot the
            # real value (was previously hard-coded None to preserve a latent
            # bug — fixed in this commit, see comment in _load_static_data).
            num_analyst_opinions=(getattr(cache, 'analyst_count', None) if cache else None),
            quarterly_earnings=_json_or_none(getattr(s, 'quarterly_earnings', None)),
            annual_earnings=_json_or_none(getattr(s, 'annual_earnings', None)),
            quarterly_revenue=_json_or_none(getattr(s, 'quarterly_revenue', None)),
            score_details=_json_or_none(getattr(s, 'score_details', None)),
            # C-score signal (May 6 — needed by canslim_scorer's surprise bonus
            # path. Pulled from Stock to match the earnings_beat_streak source.)
            earnings_surprise_pct=getattr(s, 'earnings_surprise_pct', None),
        ))
    db.bulk_save_objects(snapshots)
    db.commit()
    logger.info(f"Snapshotted {len(snapshots)} tickers for backtest {backtest_id}")
    return len(snapshots)


def copy_backtest_static_snapshot(db: Session, source_backtest_id: int, target_backtest_id: int) -> int:
    """Copy every BacktestStaticSnapshot row from source to target backtest.

    Used by the model-graduation eval gate to give two consecutive backtests
    (incumbent vs. candidate) byte-identical inputs. Without this, two
    backtests run minutes apart can drift by 12pp simply because the
    stock_data_cache has been refreshed between them — the original drift
    mechanism documented in canslim-livescan-churn-investigation.md.

    Behavior:
    - Idempotent: if target already has any snapshot rows, return 0 without
      copying. Same pattern as create_backtest_static_snapshot. Lets the
      caller invoke this safely after BacktestEngine.run() (which itself
      idempotently creates rows).
    - Refuses to operate when source has no snapshot. The eval gate must
      capture from a real snapshotted run; sourceless copy is a misuse and
      would silently produce an empty target.

    Returns the number of rows inserted.
    """
    from backend.database import BacktestStaticSnapshot

    if source_backtest_id == target_backtest_id:
        raise ValueError(f"copy_backtest_static_snapshot: source == target ({source_backtest_id})")

    existing_target = db.query(BacktestStaticSnapshot.id).filter(
        BacktestStaticSnapshot.backtest_id == target_backtest_id
    ).first()
    if existing_target is not None:
        logger.debug(f"Target backtest {target_backtest_id} already has snapshot rows; skipping copy")
        return 0

    source_rows = db.query(BacktestStaticSnapshot).filter(
        BacktestStaticSnapshot.backtest_id == source_backtest_id
    ).all()
    if not source_rows:
        raise ValueError(
            f"Source backtest {source_backtest_id} has no static snapshot rows to copy. "
            f"Run the source backtest first (or call create_backtest_static_snapshot) before copying."
        )

    copies = []
    for src in source_rows:
        copies.append(BacktestStaticSnapshot(
            backtest_id=target_backtest_id,
            ticker=src.ticker,
            days_to_earnings=src.days_to_earnings,
            earnings_beat_streak=src.earnings_beat_streak,
            eps_estimate_revision_pct=src.eps_estimate_revision_pct,
            industry_group_rank=src.industry_group_rank,
            weeks_in_base=src.weeks_in_base,
            sector=src.sector,
            roe=src.roe,
            analyst_target_price=src.analyst_target_price,
            num_analyst_opinions=src.num_analyst_opinions,
            quarterly_earnings=src.quarterly_earnings,
            annual_earnings=src.annual_earnings,
            quarterly_revenue=src.quarterly_revenue,
            score_details=src.score_details,
            earnings_surprise_pct=getattr(src, 'earnings_surprise_pct', None),
        ))
    db.bulk_save_objects(copies)
    db.commit()
    logger.info(
        f"Copied {len(copies)} snapshot rows from backtest {source_backtest_id} "
        f"to backtest {target_backtest_id}"
    )
    return len(copies)


class BacktestEngine:
    """
    Runs a historical simulation of the CANSLIM AI trading strategy.
    """

    def __init__(self, db: Session, backtest_id: int, profile_overrides: dict = None):
        self.db = db
        self.backtest = db.get(BacktestRun, backtest_id)

        if not self.backtest:
            raise ValueError(f"Backtest {backtest_id} not found")

        # Data provider (initialized in run())
        self.data_provider: Optional[HistoricalDataProvider] = None

        # In-memory portfolio state
        self.cash: float = self.backtest.starting_cash
        self.positions: Dict[str, SimulatedPosition] = {}

        # Score history for stability checks (prevent selling on single score blips)
        self.score_history: Dict[str, List[float]] = {}

        # Static data cache (sector, earnings, etc. from database)
        self.static_data: Dict[str, dict] = {}

        # CANSLIM scorer for delegating score computation. Pass `None` for the
        # data_fetcher because we only call _score_current_earnings here, which
        # never accesses the fetcher (it operates purely on the StockData
        # passed in). Other scoring paths (L, M) that DO use the fetcher are
        # still computed inline by the backtester's day-loop.
        self._scorer = CANSLIMScorer(None)

        # Bear-base watchlist: ticker -> {readiness, first_seen, last_seen}.
        # Mirrors live BearBaseCandidate behavior — entries persist after a
        # stock's last qualifying scan for a 30-day grace period so the bonus
        # carries into the post-bear bull start (the period when the bonus
        # matters most). See _bear_base_bonus_for_candidate.
        self._bear_base_watchlist: Dict[str, dict] = {}

        # Track metrics
        self.peak_portfolio_value: float = self.backtest.starting_cash
        self.max_drawdown_pct: float = 0.0
        self.daily_returns: List[float] = []
        self.trades_executed: int = 0
        self.sells_executed: int = 0  # Track sells separately for accurate win rate
        self.profitable_trades: int = 0
        self.drawdown_halt: bool = False  # Circuit breaker: no new buys when True

        # Re-entry cooldown tracking: ticker -> (date, reason)
        self.recently_sold: Dict[str, tuple] = {}

        # Pyramid cooldown tracking: ticker -> last pyramid date (1-day min between pyramids)
        self.last_pyramid_date: Dict[str, date] = {}

        # Strategy profile (balanced or growth) — load BEFORE market state
        # so profile-level market_state overrides (e.g. FTD thresholds) apply
        self.strategy = self.backtest.strategy or "balanced"
        # Deep-copy: get_strategy_profile() returns a live reference into the
        # module-level config singleton's strategy_profiles dict. _deep_merge
        # below mutates self.profile in place, so without this copy a backtest
        # with profile_overrides permanently poisons the in-memory profile for
        # EVERY subsequent backtest in the same worker process (single-worker
        # queue). Symptom: sequential A/B pairs read identical because the
        # "control" inherited the prior run's overrides. Copy makes each run's
        # profile independent. Backtester-only — trading_utils/ai_trader untouched.
        self.profile = copy.deepcopy(get_strategy_profile(self.strategy))

        # Apply profile overrides for A/B testing (e.g., ML signal on/off)
        if profile_overrides:
            self._deep_merge(self.profile, profile_overrides)

        # Market state machine (replaces binary regime gate + advisory FTD)
        # Deep-merge profile-level market_state overrides with global config
        market_state_config = config.get('market_state', {})
        profile_ms_overrides = self.profile.get('market_state', {})
        if profile_ms_overrides:
            merged_ms_config = {**market_state_config}
            for key, val in profile_ms_overrides.items():
                if isinstance(val, dict) and key in merged_ms_config and isinstance(merged_ms_config[key], dict):
                    merged_ms_config[key] = {**merged_ms_config[key], **val}
                else:
                    merged_ms_config[key] = val
            market_state_config = merged_ms_config
        self.market_state = MarketStateManager(market_state_config)
        self.market_state_enabled = market_state_config.get('enabled', True)
        self.heat_penalty_active: bool = False
        self.ftd_penalty_active: bool = False

        # Track consecutive days with no positions (for re-seeding)
        self.idle_days: int = 0
        self.underinvested_days: int = 0  # Track days with < half max_positions and > 50% cash
        self.is_seed_day: bool = False  # G: Track seed day to skip conviction sizing
        self.last_correction_end_date: Optional[date] = None  # For post-correction accelerated deployment

        # Recovery seed tracking for fast scale-up
        self.recovery_seed_date: Optional[date] = None  # When recovery seeds were planted
        self.recovery_seed_tickers: List[str] = []  # Which tickers were recovery seeds

        # Nibble mode flag (set during CORRECTION when nibble config is enabled)
        self.nibble_mode_active: bool = False

        # Correction zone flag: SPY below 50MA but above 200MA (profile-configurable)
        self.correction_zone_active: bool = False

        # Experimental position isolation: track realized losses from nibble/V-bottom
        # positions separately so they don't inflate circuit breaker drawdown
        self.experimental_realized_losses: float = 0.0  # Cumulative $ lost on experimental positions
        self.experimental_capital_deployed: float = 0.0  # Current $ in experimental positions

        # Buy throttle: track recent full-sell outcomes to slow buying during losing streaks
        # Stores realized_gain_pct for each completed (non-partial) sell
        self.recent_trade_outcomes: deque = deque(maxlen=10)

        # SPY cash sweep: park idle cash in SPY when SPY > 50MA
        self.spy_sweep_shares: float = 0.0

        # Score floor decay: count consecutive trading days under-invested in a bull market
        self.underinvested_days: int = 0

        # cs_bear / correction_zone overlay diagnostics — track whether the bear
        # overlay code paths actually fire during this backtest. Persisted to
        # BacktestRun.overlay_stats on completion. See canslim-rolling-matrix-may1.
        # Snapshot whether ML actually gated trades on this run, computed
        # against the resolved profile (YAML default + any profile_overrides).
        # ML training uses this to exclude self-influenced runs — relying only
        # on profile_overrides, like the original _is_ml_contaminated did,
        # silently mis-classifies every default-ML-on run as clean training data.
        ml_cfg = self.profile.get('ml_signal', {}) or {}
        ml_was_active = bool(ml_cfg.get('enabled', False)) and not bool(ml_cfg.get('log_only', True))

        # Optional per-backtest ML model override. Used by the model-graduation
        # gate (routes/ml.py: _run_evaluation_backtest) to score a candidate
        # model file without swapping the production active model. When set,
        # all ml predictions in this run use the override instead of the global
        # active model. None → use global ml_model_active.joblib as today.
        self._eval_model_payload: Optional[dict] = None
        eval_model_path = ml_cfg.get('model_path')
        if eval_model_path:
            try:
                from ml.trainer import load_model
                from pathlib import Path
                self._eval_model_payload = load_model(Path(eval_model_path))
                if self._eval_model_payload is None:
                    logger.warning(
                        f"Eval model override path {eval_model_path} did not load — "
                        f"falling back to global active model"
                    )
                else:
                    logger.info(
                        f"Backtest {backtest_id} using ML model override: {eval_model_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to load eval model override {eval_model_path}: {e}")
                self._eval_model_payload = None

        self.overlay_stats: dict = {
            'cz_active_days': 0,            # days correction_zone_active was True
            'cz_pre_filter_rejected': 0,    # candidates filtered by C+A+L / RS / base requirement
            'cz_cs_only_rejected': 0,       # candidates rejected by cs_only mode (non-CS)
            'cz_cs_confidence_rejected': 0, # candidates rejected by cs_only mode (CS confidence too low)
            'cz_pass': 0,                   # candidates that passed CZ filtering
            'cz_position_mult_applied': 0,  # times position size was reduced via _correction_zone_mult
            'bear_base_bonus_applied': 0,   # candidates that received bear base bonus
            'bear_base_bonus_total': 0,     # cumulative bonus points awarded
            'ml_was_active': ml_was_active, # True iff ML actively gated trades (read by ML training dedup)
        }

        # SPY tracking for benchmark
        self.spy_start_price: float = 0.0
        self.spy_shares: float = 0.0  # Hypothetical SPY buy-and-hold

        # Daily score cache: avoids recalculating scores for the same ticker+date
        # multiple times per day (circuit breaker, sell eval, buy eval all call _calculate_scores)
        self._score_cache: Dict[str, Dict[str, dict]] = {}  # {date_str: {ticker: score_dict}}
        self._score_cache_date: Optional[str] = None  # Current cache date

        # Persistent score cache (SQLite on disk, survives across backtest runs)
        self._persistent_cache_conn: Optional[sqlite3.Connection] = None
        self._persistent_cache_writes: int = 0  # Batch commit counter

    def _init_persistent_score_cache(self):
        """Initialize SQLite-based persistent score cache on disk."""
        try:
            os.makedirs(SCORE_CACHE_DIR, exist_ok=True)
            db_path = os.path.join(SCORE_CACHE_DIR, f"scores_v{SCORE_CACHE_VERSION}.db")
            self._persistent_cache_conn = sqlite3.connect(db_path, timeout=10)
            self._persistent_cache_conn.execute("PRAGMA journal_mode=WAL")
            self._persistent_cache_conn.execute("PRAGMA synchronous=NORMAL")
            self._persistent_cache_conn.execute("PRAGMA busy_timeout=5000")  # 5s retry on lock
            self._persistent_cache_conn.execute("""
                CREATE TABLE IF NOT EXISTS scores (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    score_json TEXT NOT NULL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            self._persistent_cache_conn.commit()
            count = self._persistent_cache_conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
            logger.info(f"Backtest {self.backtest.id}: Persistent score cache opened ({count:,} entries)")
        except Exception as e:
            logger.warning(f"Backtest {self.backtest.id}: Could not init persistent score cache: {e}")
            self._persistent_cache_conn = None

    def _get_persistent_scores(self, date_str: str, tickers: List[str]) -> Dict[str, dict]:
        """Batch-fetch cached scores from SQLite for a given date."""
        if not self._persistent_cache_conn or not tickers:
            return {}
        try:
            placeholders = ",".join("?" for _ in tickers)
            rows = self._persistent_cache_conn.execute(
                f"SELECT ticker, score_json FROM scores WHERE date = ? AND ticker IN ({placeholders})",
                [date_str] + tickers
            ).fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}
        except Exception:
            return {}

    def _save_persistent_scores(self, date_str: str, scores: Dict[str, dict]):
        """Batch-save computed scores to SQLite."""
        if not self._persistent_cache_conn or not scores:
            return
        try:
            rows = [(ticker, date_str, json.dumps(s)) for ticker, s in scores.items()]
            self._persistent_cache_conn.executemany(
                "INSERT OR IGNORE INTO scores (ticker, date, score_json) VALUES (?, ?, ?)",
                rows
            )
            self._persistent_cache_writes += len(rows)
            # Commit every 5000 writes to balance durability and performance
            if self._persistent_cache_writes >= 5000:
                self._persistent_cache_conn.commit()
                self._persistent_cache_writes = 0
        except Exception as e:
            logger.debug(f"Persistent cache write error: {e}")

    def _get_frozen_scores(self, date_str: str, tickers: List[str]) -> Dict[str, dict]:
        """Fetch frozen scanner snapshots from StockScore table for a given date.

        Returns only the CANSLIM fundamental scores (not price-derived fields).
        Uses the latest scan per stock per date (scanner may run multiple times).
        """
        if not tickers:
            return {}
        try:
            # Subquery: latest StockScore.id per stock_id for this date
            latest_ids = (
                self.db.query(func.max(StockScore.id).label('max_id'))
                .filter(StockScore.date == date_str)
                .group_by(StockScore.stock_id)
                .subquery()
            )
            rows = (
                self.db.query(Stock.ticker, StockScore)
                .join(Stock, StockScore.stock_id == Stock.id)
                .filter(StockScore.id.in_(
                    self.db.query(latest_ids.c.max_id)
                ))
                .filter(Stock.ticker.in_(tickers))
                .all()
            )
            result = {}
            for ticker, ss in rows:
                result[ticker] = {
                    "total_score": _nan_safe(ss.total_score, 0),
                    "c_score": _nan_safe(ss.c_score, 0),
                    "a_score": _nan_safe(ss.a_score, 0),
                    "n_score": _nan_safe(ss.n_score, 0),
                    "s_score": _nan_safe(ss.s_score, 0),
                    "l_score": _nan_safe(ss.l_score, 0),
                    "i_score": _nan_safe(ss.i_score, 0),
                    "m_score": _nan_safe(ss.m_score, 0),
                    "projected_growth": _nan_safe(ss.projected_growth, 0),
                    "_frozen_price": _nan_safe(ss.current_price, 0),
                    "_frozen_52w_high": _nan_safe(ss.week_52_high, 0),
                }
            return result
        except Exception as e:
            logger.debug(f"Frozen score query error: {e}")
            return {}

    def _build_score_from_frozen(self, ticker: str, current_date: date,
                                  frozen: dict) -> dict:
        """Build a complete score dict from frozen fundamentals + fresh price-derived fields."""
        price = self.data_provider.get_price_on_date(ticker, current_date)
        if not price or price <= 0 or price != price:
            return None

        high_52w, low_52w = self.data_provider.get_52_week_high_low(ticker, current_date)
        pct_from_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else 0

        rs_12m = self.data_provider.get_relative_strength(ticker, current_date, 12)
        rs_3m = self.data_provider.get_relative_strength(ticker, current_date, 3)

        is_breaking_out, volume_ratio_breakout, _ = self.data_provider.is_breaking_out(
            ticker, current_date
        )

        base_result = self.data_provider.detect_base_pattern(ticker, current_date)
        base_pattern = base_result if base_result else {"type": "none"}
        has_base = base_pattern.get("type", "none") not in ("none", None)
        weeks_in_base = base_pattern.get("weeks", 0) if has_base else 0
        pivot_price = base_pattern.get("pivot_price", high_52w)
        pct_from_pivot = ((price - pivot_price) / pivot_price * 100) if pivot_price > 0 else pct_from_high

        ad_data = self.data_provider.get_accumulation_distribution(ticker, current_date, 20)
        volume_dry_up = ad_data.get("volume_dry_up", False) if ad_data else False
        volume_dry_up_score = ad_data.get("volume_dry_up_score", 0) if ad_data else 0

        static_data = self.static_data.get(ticker, {})
        is_growth_stock = False  # Conservative default for frozen scores

        return {
            "total_score": frozen["total_score"],
            "c_score": frozen["c_score"],
            "a_score": frozen["a_score"],
            "n_score": frozen["n_score"],
            "s_score": frozen["s_score"],
            "l_score": frozen["l_score"],
            "i_score": frozen["i_score"],
            "m_score": frozen["m_score"],
            "rs_12m": rs_12m,
            "rs_3m": rs_3m,
            "projected_growth": frozen["projected_growth"],
            "pct_from_high": pct_from_high,
            "pct_from_pivot": pct_from_pivot,
            "pivot_price": pivot_price,
            "has_base_pattern": has_base,
            "base_pattern": base_pattern,
            "weeks_in_base": weeks_in_base,
            "is_growth_stock": is_growth_stock,
            "is_breaking_out": is_breaking_out,
            "volume_ratio": volume_ratio_breakout if volume_ratio_breakout else 1.0,
            "_volume_dry_up": volume_dry_up,
            "_volume_dry_up_score": volume_dry_up_score,
        }

    def _close_persistent_score_cache(self):
        """Flush and close the persistent score cache."""
        if self._persistent_cache_conn:
            try:
                self._persistent_cache_conn.commit()
                count = self._persistent_cache_conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
                logger.info(f"Backtest {self.backtest.id}: Persistent score cache closed ({count:,} entries)")
                self._persistent_cache_conn.close()
            except Exception as e:
                logger.debug(f"Backtest cache cleanup error: {e}")
            self._persistent_cache_conn = None

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        """Recursively merge override into base dict (in-place)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                BacktestEngine._deep_merge(base[key], value)
            else:
                base[key] = value

    def run(self) -> BacktestRun:
        """
        Execute the backtest day by day.
        """
        try:
            self.backtest.status = "running"
            self.db.commit()

            # Safety net: snapshot the live P1 cache if no snapshot was
            # written at creation time (e.g. internal callers, batch
            # endpoints, or backtests created before the snapshot
            # mechanism existed). create_backtest_static_snapshot is
            # idempotent — no-op when rows already exist.
            try:
                create_backtest_static_snapshot(self.db, self.backtest.id)
            except Exception as snap_err:
                logger.warning(
                    f"Engine-side snapshot failed for backtest {self.backtest.id}: {snap_err}"
                )

            # Get stock universe
            tickers = self._get_universe_tickers()
            logger.info(f"Backtest {self.backtest.id}: {len(tickers)} tickers, "
                        f"{self.backtest.start_date} to {self.backtest.end_date}")

            # Initialize data provider — capture today's date once so
            # _filter_available_earnings stays deterministic for this run
            self.data_provider = HistoricalDataProvider(tickers, data_reference_date=date.today())

            # Initialize persistent score cache (reused across backtest runs)
            self._init_persistent_score_cache()

            # Preload historical data
            def update_progress(pct):
                self.backtest.progress_pct = pct * 0.3  # Loading is 30% of total
                self.db.commit()
                # Check for cancellation during data loading
                self.db.refresh(self.backtest)
                if self.backtest.cancel_requested or self.backtest.status == "cancelled":
                    raise ValueError("Cancelled by user")

            success = self.data_provider.preload_data(
                self.backtest.start_date,
                self.backtest.end_date,
                progress_callback=update_progress
            )

            if not success:
                raise ValueError("Failed to load historical data")

            # Pre-compute market direction for all trading days (performance optimization)
            self.data_provider.precompute_market_direction()

            # Load static data (sector, earnings) from database
            self._load_static_data()

            # Fetch missing historical earnings for old backtest periods
            self._fetch_missing_earnings()

            # Filter for survivorship bias: only keep stocks with price data at backtest start
            # This prevents inflated returns from stocks that didn't exist or weren't tradeable
            available_tickers = self.data_provider.get_available_tickers()
            original_count = len(available_tickers)

            # Get the first trading day to check for data availability
            trading_days = self.data_provider.get_trading_days()
            if not trading_days:
                raise ValueError("No trading days in period")

            # Survivorship bias filter: exclude stocks that don't have price data at the START
            # of the backtest. This prevents us from trading stocks that IPO'd mid-backtest
            # or weren't actively traded at the beginning.
            first_day = trading_days[0]
            valid_tickers = []
            excluded_tickers = []

            for ticker in available_tickers:
                start_price = self.data_provider.get_price_on_date(ticker, first_day)
                if start_price and start_price > 0:
                    valid_tickers.append(ticker)
                else:
                    excluded_tickers.append(ticker)

            # Update the data provider to only use valid tickers
            self.data_provider.filter_tickers(valid_tickers)

            excluded_count = len(excluded_tickers)
            if excluded_count > 0:
                logger.info(f"Survivorship filter: excluded {excluded_count} stocks without "
                           f"price data at backtest start ({first_day})")
                if excluded_count <= 20:
                    logger.debug(f"Excluded tickers: {excluded_tickers}")

            logger.info(f"Universe after survivorship filter: {len(valid_tickers)} stocks "
                       f"(was {original_count})")

            # Initialize SPY benchmark
            self.spy_start_price = self.data_provider.get_spy_price_on_date(trading_days[0])
            if self.spy_start_price > 0:
                self.spy_shares = self.backtest.starting_cash / self.spy_start_price

            # Initialize market state machine from day-1 conditions
            if self.market_state_enabled:
                spy_day1 = self.data_provider.get_spy_daily_data(trading_days[0])
                self.market_state.initialize_state(
                    spy_close=spy_day1["close"],
                    spy_ma50=spy_day1["ma50"],
                    spy_ema21=spy_day1["ema21"],
                )

            # Seed initial positions on day 1 based on historical CANSLIM scores
            self.is_seed_day = True
            self._seed_initial_positions(trading_days[0])

            # Simulate each trading day
            total_days = len(trading_days)
            progress = 30.0  # Start at 30% (data loading complete)
            for i, current_date in enumerate(trading_days):
                # G: Clear seed day flag after day 1
                if i > 0:
                    self.is_seed_day = False
                # Check for cancellation request every 10 days
                if i % 10 == 0:
                    self.db.refresh(self.backtest)
                    if self.backtest.cancel_requested or self.backtest.status == "cancelled":
                        logger.info(f"Backtest {self.backtest.id} cancelled by user at {i}/{total_days} days")
                        self._close_persistent_score_cache()
                        self.backtest.status = "cancelled"
                        self.backtest.error_message = f"Cancelled by user at {progress:.0f}% progress"
                        self.db.commit()
                        return self.backtest

                self._simulate_day(current_date)

                # Update progress (30% loading + 70% simulation)
                progress = 30 + ((i + 1) / total_days * 70)
                self.backtest.progress_pct = progress
                if i % 10 == 0:  # Commit every 10 days
                    self.db.commit()

            # Calculate final metrics
            self._calculate_final_metrics()

            # Close persistent score cache (flush remaining writes)
            self._close_persistent_score_cache()

            self.backtest.status = "completed"
            self.backtest.completed_at = datetime.now(timezone.utc)
            self.backtest.progress_pct = 100
            overlay_payload = dict(self.overlay_stats)
            if hasattr(self, '_data_fingerprint'):
                overlay_payload['data_fingerprint'] = self._data_fingerprint
            self.backtest.overlay_stats = overlay_payload
            self.db.commit()

            logger.info(f"Backtest {self.backtest.id} completed: "
                        f"{self.backtest.total_return_pct:.1f}% return, "
                        f"{self.backtest.total_trades} trades")
            logger.info(f"Backtest {self.backtest.id} overlay_stats: {self.overlay_stats}")

            # Log market state summary
            if self.market_state_enabled and self.market_state.state_history:
                changes = self.market_state.state_history
                logger.info(f"Market state changes: {len(changes)} transitions")
                for c in changes:
                    logger.info(f"  {c['date']}: {c['from']} -> {c['to']} "
                               f"(SPY ${c['spy']:.2f}, dist={c['dist_count']})")

            return self.backtest

        except Exception as e:
            self._close_persistent_score_cache()
            if "Cancelled by user" in str(e):
                logger.info(f"Backtest {self.backtest.id} cancelled during data loading")
                self.backtest.status = "cancelled"
                self.backtest.error_message = "Cancelled by user"
                self.db.commit()
                return self.backtest
            logger.error(f"Backtest {self.backtest.id} failed: {e}")
            self.backtest.status = "failed"
            self.backtest.error_message = str(e)
            self.db.commit()
            raise

    def _get_universe_tickers(self) -> List[str]:
        """Get list of tickers to include in backtest"""
        if self.backtest.stock_universe == "custom" and self.backtest.custom_tickers:
            return self.backtest.custom_tickers

        # Try to get tickers from the ticker module (more reliable than database)
        try:
            from sp500_tickers import get_sp500_tickers, get_all_tickers

            if self.backtest.stock_universe == "sp500":
                tickers = get_sp500_tickers()
                logger.info(f"Loaded {len(tickers)} S&P 500 tickers from module")
            else:
                tickers = get_all_tickers()
                logger.info(f"Loaded {len(tickers)} tickers from module")

            if tickers:
                return tickers
        except Exception as e:
            logger.warning(f"Could not load tickers from module: {e}")

        # Fallback to database
        query = self.db.query(Stock.ticker)

        if self.backtest.stock_universe == "sp500":
            query = query.filter(Stock.market_cap >= 10_000_000_000)

        tickers = [row[0] for row in query.all()]
        return tickers

    def _load_static_data(self):
        """Load static stock data (sector, earnings, ROE) from database.

        P1-cache scalars (days_to_earnings, earnings_beat_streak,
        eps_estimate_revision_pct, industry_group_rank, weeks_in_base) are
        read from the per-backtest BacktestStaticSnapshot table when one
        exists for this backtest_id. That snapshot is populated at
        backtest creation, so re-running the same backtest later is
        deterministic regardless of how the live cache has drifted in
        between. Backtests created before the snapshot mechanism existed
        have no rows in that table and fall back to the live Stock
        columns — preserving legacy behavior.
        """
        from backend.database import StockDataCache, BacktestStaticSnapshot

        stocks = self.db.query(Stock).all()

        # Batch-load ROE and analyst data from StockDataCache
        cache_entries = self.db.query(StockDataCache).all()
        cache_by_ticker = {c.ticker: c for c in cache_entries}

        # Per-backtest P1 snapshot (May 2026 fix). When present, prefer
        # these over live Stock columns so the run reproduces.
        snapshots = self.db.query(BacktestStaticSnapshot).filter(
            BacktestStaticSnapshot.backtest_id == self.backtest.id
        ).all()
        snap_by_ticker = {s.ticker: s for s in snapshots}
        if snap_by_ticker:
            logger.info(
                f"Loaded {len(snap_by_ticker)} P1 snapshots for backtest "
                f"{self.backtest.id} (deterministic mode)"
            )

        def _decode_json(raw):
            """Snapshot serializes lists/dicts as JSON text; live Stock
            columns may already be parsed objects. Normalize."""
            if raw is None:
                return None
            if isinstance(raw, (list, dict)):
                return raw
            if isinstance(raw, (str, bytes)):
                try:
                    return json.loads(raw)
                except (ValueError, TypeError):
                    return None
            return None

        for stock in stocks:
            cache = cache_by_ticker.get(stock.ticker)
            snap = snap_by_ticker.get(stock.ticker)

            # Snap-preferred reader: returns the snapshot value if non-None,
            # otherwise the live value, otherwise the default.
            def _snap(field, live, default=None):
                if snap is not None:
                    snap_val = getattr(snap, field, None)
                    if snap_val is not None:
                        return snap_val
                return live if live is not None else default

            # ROE — snapshot preferred; fall back to live cache then
            # score_details JSON (legacy fallback path)
            roe = _snap('roe', cache.roe if (cache and cache.roe is not None) else None, None)
            if roe is None:
                if stock.score_details:
                    a_details = (stock.score_details or {}).get('a', {})
                    if isinstance(a_details, dict):
                        roe = a_details.get('roe', 0.0) or 0.0
                roe = roe or 0.0

            analyst_target = _snap(
                'analyst_target_price',
                getattr(cache, 'analyst_target_price', 0) if cache else 0,
                0,
            ) or 0
            # Latent-bug fix: the StockDataCache column is `analyst_count`,
            # not `num_analyst_opinions`. The original read used the wrong
            # attribute name and silently returned 0 for everything via the
            # getattr default — so growth_projector branches that test
            # >= 5/10/20 analysts have never fired in production. Now reads
            # the real column. Snapshots written before this commit have
            # num_analyst_opinions=None (was hard-coded to preserve the bug);
            # they fall through to the live cache.analyst_count value.
            num_analysts = _snap(
                'num_analyst_opinions',
                getattr(cache, 'analyst_count', 0) if cache else 0,
                0,
            ) or 0

            # Earnings/revenue arrays (snapshot stores as JSON text)
            quarterly_earnings = _decode_json(_snap('quarterly_earnings', None)) \
                if (snap and snap.quarterly_earnings is not None) \
                else (stock.quarterly_earnings or [])
            annual_earnings = _decode_json(_snap('annual_earnings', None)) \
                if (snap and snap.annual_earnings is not None) \
                else (stock.annual_earnings or [])
            quarterly_revenue = _decode_json(_snap('quarterly_revenue', None)) \
                if (snap and snap.quarterly_revenue is not None) \
                else (stock.quarterly_revenue or [])

            # score_details — snapshot prefer, used for institutional_pct only.
            # Variable named `score_details` to match the literal pattern asserted
            # in tests/test_bug_regressions.py::TestBacktesterInstitutionalPct.
            score_details = _decode_json(_snap('score_details', None)) \
                if (snap and snap.score_details is not None) \
                else (stock.score_details or {})
            institutional_pct = (score_details or {}).get('i', {}).get('institutional_pct', 0) or 0

            self.static_data[stock.ticker] = {
                "sector": _snap('sector', stock.sector, "Unknown") or "Unknown",
                "name": stock.name or stock.ticker,
                "institutional_holders_pct": institutional_pct,
                "roe": roe,
                "analyst_target_price": analyst_target,
                "num_analyst_opinions": num_analysts,
                "quarterly_earnings": quarterly_earnings or [],
                "annual_earnings": annual_earnings or [],
                "quarterly_revenue": quarterly_revenue or [],
                # Coiled Spring fields — P1 snapshot preferred when present
                "weeks_in_base": _snap('weeks_in_base', getattr(stock, 'weeks_in_base', 0), 0) or 0,
                "earnings_beat_streak": _snap('earnings_beat_streak', getattr(stock, 'earnings_beat_streak', 0), 0) or 0,
                "days_to_earnings": _snap('days_to_earnings', getattr(stock, 'days_to_earnings', None), None),
                "eps_estimate_revision_pct": _snap('eps_estimate_revision_pct', getattr(stock, 'eps_estimate_revision_pct', None), None),
                # C-score surprise bonus signal (May 6 — read by canslim_scorer
                # via the StockData adapter built in _calculate_scores).
                "earnings_surprise_pct": _snap('earnings_surprise_pct', getattr(stock, 'earnings_surprise_pct', None), None),
                # ML v11: sector rank
                "industry_group_rank": _snap('industry_group_rank', getattr(stock, 'industry_group_rank', 50), 50) or 50,
            }

    def _compute_fingerprint(self):
        """Compute MD5 fingerprint of earnings data for reproducibility tracking."""
        import hashlib
        fingerprint_data = ""
        for ticker in sorted(self.static_data.keys()):
            q = self.static_data[ticker].get("quarterly_earnings", [])
            a = self.static_data[ticker].get("annual_earnings", [])
            fingerprint_data += f"{ticker}:{q[:8]}:{a[:5]}|"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]

    def _get_earnings_cache_path(self):
        """Get path for the version-based earnings cache file (single file, not per-date)."""
        cache_dir = "/tmp/backtest_cache"
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "earnings_latest.json")

    def _load_earnings_cache(self):
        """Try to load cached FMP earnings data. Returns True if loaded.

        Uses a single persistent cache file (earnings_latest.json) instead of
        date-keyed files. Cache is reused indefinitely unless force_refresh is set.
        """
        import json

        # Check if force_refresh is requested
        if getattr(self.backtest, 'force_refresh', False):
            logger.info(f"Backtest {self.backtest.id}: force_refresh=True, skipping cache")
            return False

        cache_path = self._get_earnings_cache_path()
        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, 'r') as f:
                wrapper = json.load(f)

            # Support both old format (flat dict) and new format (with metadata)
            if "data" in wrapper and "fetched_at" in wrapper:
                cached = wrapper["data"]
                fetched_at = wrapper.get("fetched_at", "unknown")
                cached_fingerprint = wrapper.get("fingerprint", "unknown")
                ticker_count = wrapper.get("ticker_count", len(cached))
            else:
                # Old format: flat dict of ticker -> earnings data
                cached = wrapper
                fetched_at = "unknown (legacy)"
                cached_fingerprint = "unknown"
                ticker_count = len(cached)

            applied = 0
            for ticker, data in cached.items():
                if ticker in self.static_data:
                    if "quarterly_earnings" in data:
                        self.static_data[ticker]["quarterly_earnings"] = data["quarterly_earnings"]
                    if "quarterly_revenue" in data:
                        self.static_data[ticker]["quarterly_revenue"] = data["quarterly_revenue"]
                    if "annual_earnings" in data:
                        self.static_data[ticker]["annual_earnings"] = data["annual_earnings"]
                    applied += 1

            self._data_fingerprint = self._compute_fingerprint()
            logger.info(f"Backtest {self.backtest.id}: Loaded earnings cache for {applied} tickers "
                       f"(fetched: {fetched_at}, cached_fp: {cached_fingerprint}, "
                       f"current_fp: {self._data_fingerprint})")
            return True
        except Exception as e:
            logger.warning(f"Backtest {self.backtest.id}: Failed to load earnings cache: {e}")
            return False

    def _save_earnings_cache(self):
        """Save FMP earnings data to cache file for reuse by subsequent backtests.

        Saves as a single versioned file with metadata (fetched_at, fingerprint, ticker_count).
        Cleans up old date-based cache files from the previous scheme.
        """
        import json
        cache_path = self._get_earnings_cache_path()
        try:
            cached = {}
            for ticker, data in self.static_data.items():
                entry = {}
                if "quarterly_earnings" in data:
                    entry["quarterly_earnings"] = data["quarterly_earnings"]
                if "quarterly_revenue" in data:
                    entry["quarterly_revenue"] = data["quarterly_revenue"]
                if "annual_earnings" in data:
                    entry["annual_earnings"] = data["annual_earnings"]
                if entry:
                    cached[ticker] = entry

            wrapper = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": getattr(self, '_data_fingerprint', 'unknown'),
                "ticker_count": len(cached),
                "data": cached,
            }

            with open(cache_path, 'w') as f:
                json.dump(wrapper, f)

            # Clean up old date-based cache files (earnings_YYYY-MM-DD.json)
            cache_dir = os.path.dirname(cache_path)
            for fname in os.listdir(cache_dir):
                if fname.startswith("earnings_2") and fname.endswith(".json"):
                    os.remove(os.path.join(cache_dir, fname))

            logger.info(f"Backtest {self.backtest.id}: Saved earnings cache ({len(cached)} tickers, "
                        f"fingerprint: {getattr(self, '_data_fingerprint', 'unknown')})")
        except Exception as e:
            logger.warning(f"Backtest {self.backtest.id}: Failed to save earnings cache: {e}")

    def _wait_for_scan_to_finish(self):
        """Wait for any running scan to finish before starting FMP fetch.

        This prevents rate limit competition between the scanner and backtester,
        which was causing variable FMP failures (19 vs 88 tickers).
        """
        import time
        try:
            from backend.scheduler import _scan_config
        except ImportError:
            return

        max_wait = 1800  # 30 minutes
        waited = 0
        while _scan_config.get("is_scanning", False) and waited < max_wait:
            logger.info(f"Backtest {self.backtest.id}: Waiting for scan to finish before FMP fetch ({waited}s)...")
            time.sleep(10)
            waited += 10

        if waited > 0:
            logger.info(f"Backtest {self.backtest.id}: Scan finished after {waited}s wait, proceeding with FMP fetch")

    def _fetch_single_ticker_earnings(self, ticker, fmp_base, fmp_api_key, q_limit, a_limit):
        """Fetch earnings data for a single ticker from FMP. Returns True on success."""
        import requests

        try:
            q_url = f"{fmp_base}/income-statement?symbol={ticker}&period=quarter&limit={q_limit}&apikey={fmp_api_key}"
            q_resp = requests.get(q_url, timeout=10)

            a_url = f"{fmp_base}/income-statement?symbol={ticker}&limit={a_limit}&apikey={fmp_api_key}"
            a_resp = requests.get(a_url, timeout=10)

            updated = False

            if q_resp.status_code == 200:
                q_data = q_resp.json()
                if isinstance(q_data, list) and q_data:
                    q_eps = [q.get("eps", 0) or 0 for q in q_data]
                    q_rev = [q.get("revenue", 0) or 0 for q in q_data]
                    self.static_data[ticker]["quarterly_earnings"] = q_eps
                    self.static_data[ticker]["quarterly_revenue"] = q_rev
                    updated = True

            if a_resp.status_code == 200:
                a_data = a_resp.json()
                if isinstance(a_data, list) and a_data:
                    a_eps = [a.get("eps", 0) or 0 for a in a_data]
                    self.static_data[ticker]["annual_earnings"] = a_eps
                    updated = True

            return updated
        except Exception:
            return False

    def _fetch_missing_earnings(self):
        """
        Fetch earnings from FMP for ALL tickers to ensure deterministic backtests.

        Uses a persistent version-based cache (earnings_latest.json) that is reused
        across days unless force_refresh is set. This eliminates the main source of
        non-determinism (FMP data changing between fetches).

        Key improvements over the old date-based cache:
        - A) Waits for any running scan to finish (prevents rate limit competition)
        - B) Retries failed tickers up to 2 times with increasing delays
        - C) Cache persists indefinitely (not date-keyed), use force_refresh to update

        Only updates self.static_data (NOT the DB) to avoid interfering with scans.
        """
        import requests
        import time

        # Try loading from cache first (skipped if force_refresh=True)
        if self._load_earnings_cache():
            return

        fmp_api_key = os.environ.get('FMP_API_KEY', '')
        if not fmp_api_key:
            logger.warning("Backtest: No FMP_API_KEY, cannot fetch earnings")
            return

        # A) Wait for scan to finish before starting FMP fetch
        self._wait_for_scan_to_finish()

        # Calculate how many quarters we need
        today = date.today()
        days_diff = (today - self.backtest.start_date).days

        quarters_to_skip = max(0, (days_diff + 45) // 91) if days_diff > 0 else 0
        quarters_needed = quarters_to_skip + 8
        annuals_to_skip = max(0, (days_diff + 45) // 365) if days_diff > 0 else 0
        annuals_needed = annuals_to_skip + 5

        all_tickers = list(self.static_data.keys())

        logger.info(f"Backtest {self.backtest.id}: Fetching FMP earnings for "
                     f"{len(all_tickers)} tickers (need {quarters_needed}q/{annuals_needed}a, "
                     f"backtest starts {self.backtest.start_date})")

        fmp_base = "https://financialmodelingprep.com/stable"
        q_limit = min(quarters_needed + 4, 40)
        a_limit = min(annuals_needed + 2, 15)
        fetched_count = 0
        failed_tickers = []
        batch_size = 50

        # Main fetch loop
        for i, ticker in enumerate(all_tickers):
            success = self._fetch_single_ticker_earnings(ticker, fmp_base, fmp_api_key, q_limit, a_limit)
            if success:
                fetched_count += 1
            else:
                failed_tickers.append(ticker)

            # Rate limiting: pause every batch_size requests
            if (i + 1) % batch_size == 0:
                time.sleep(1.0)
                if (i + 1) % 200 == 0:
                    logger.info(f"Backtest {self.backtest.id}: FMP earnings fetch progress: "
                               f"{i + 1}/{len(all_tickers)} tickers "
                               f"({fetched_count} ok, {len(failed_tickers)} failed)")

        # B) Retry failed tickers (2 rounds with increasing delay)
        for retry_round in range(2):
            if not failed_tickers:
                break
            logger.info(f"Backtest {self.backtest.id}: Retry round {retry_round + 1}: "
                       f"{len(failed_tickers)} tickers")
            time.sleep(5 * (retry_round + 1))  # 5s, then 10s

            still_failed = []
            for i, ticker in enumerate(failed_tickers):
                success = self._fetch_single_ticker_earnings(ticker, fmp_base, fmp_api_key, q_limit, a_limit)
                if success:
                    fetched_count += 1
                else:
                    still_failed.append(ticker)

                if (i + 1) % batch_size == 0:
                    time.sleep(1.0)

            logger.info(f"Backtest {self.backtest.id}: Retry round {retry_round + 1} recovered "
                       f"{len(failed_tickers) - len(still_failed)} tickers")
            failed_tickers = still_failed

        self._data_fingerprint = self._compute_fingerprint()

        logger.info(f"Backtest {self.backtest.id}: FMP earnings fetched for {fetched_count} tickers "
                     f"({len(failed_tickers)} failed after retries), "
                     f"data fingerprint: {self._data_fingerprint}")

        # Save to cache for subsequent backtests
        self._save_earnings_cache()

    def _calculate_coiled_spring_for_backtest(self, ticker: str, score_data: dict, static_data: dict, cs_config: dict) -> dict:
        """
        Calculate Coiled Spring score for backtesting using score_data and static_data dicts.

        This is a simplified version that uses the pre-calculated score data from
        _calculate_scores() rather than live Stock objects.

        Returns dict with:
        - is_coiled_spring: bool
        - cs_score: float (bonus points)
        - cs_details: str (explanation)
        - allow_pre_earnings_buy: bool
        """
        # Create mock objects for calculate_coiled_spring_score
        class MockData:
            pass

        class MockScore:
            pass

        data = MockData()
        data.weeks_in_base = static_data.get('weeks_in_base', 0)
        data.earnings_beat_streak = static_data.get('earnings_beat_streak', 0)
        data.days_to_earnings = static_data.get('days_to_earnings')
        data.institutional_holders_pct = static_data.get('institutional_holders_pct', 0)

        score = MockScore()
        score.c_score = score_data.get('c_score', 0)
        score.l_score = score_data.get('l_score', 0)
        score.total_score = score_data.get('total_score', 0)

        return calculate_coiled_spring_score(data, score, cs_config)

    def _calculate_portfolio_heat(self, current_date: date) -> float:
        """
        Calculate total portfolio heat: sum of (position_size_pct × distance_to_stop_pct).
        Higher heat = more total risk exposure.
        """
        portfolio_value = self._get_portfolio_value(current_date)
        if portfolio_value <= 0:
            return 0.0

        stop_loss_config = config.get('ai_trader.stops', {})
        base_stop = stop_loss_config.get('normal_stop_loss_pct', 8.0)

        total_heat = 0.0
        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue
            position_value = position.shares * price
            position_pct = (position_value / portfolio_value) * 100
            # Distance to stop = current gain% + stop loss%
            # e.g., stock up 5% with 8% stop = 13% distance
            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100 if position.cost_basis > 0 else 0
            distance_to_stop = base_stop + gain_pct if gain_pct > 0 else base_stop + gain_pct
            total_heat += position_pct * (distance_to_stop / 100)

        return total_heat

    def _get_buy_throttle_limit(self) -> int:
        """
        Check recent trade outcomes and return max buys allowed today.

        Buy throttle reduces buying during losing streaks to prevent churn.
        Counts consecutive losses from the most recent trade backward.
        Returns 999 (unlimited) when throttle is off or not triggered.
        """
        throttle_config = self.profile.get('buy_throttle', {})
        if not throttle_config.get('enabled', False):
            return 999

        # Need minimum trades before throttle can activate
        min_trades = throttle_config.get('min_trades', 3)
        if len(self.recent_trade_outcomes) < min_trades:
            return 999

        # Count consecutive losses from most recent trade backward
        consecutive_losses = 0
        for outcome in reversed(self.recent_trade_outcomes):
            if outcome < 0:
                consecutive_losses += 1
            else:
                break  # A win breaks the streak

        pause_after = throttle_config.get('pause_after', 5)
        reduce_after = throttle_config.get('reduce_after', 3)

        if consecutive_losses >= pause_after:
            logger.info(f"BUY THROTTLE: PAUSED ({consecutive_losses} consecutive losses)")
            return 0
        elif consecutive_losses >= reduce_after:
            logger.debug(f"BUY THROTTLE: REDUCED to 1 buy ({consecutive_losses} consecutive losses)")
            return 1

        return 999  # No throttle active

    def _liquidate_spy_sweep(self, current_date: date, reason: str = ""):
        """Sell all SPY sweep shares and return cash to portfolio."""
        if self.spy_sweep_shares <= 0:
            return
        spy_price = self.data_provider.get_spy_price_on_date(current_date)
        if not spy_price:
            return
        sweep_value = self.spy_sweep_shares * spy_price
        self.cash += sweep_value
        logger.debug(f"SPY SWEEP SOLD: {self.spy_sweep_shares:.1f} shares @ ${spy_price:.2f} "
                     f"= ${sweep_value:,.0f} ({reason})")
        self.spy_sweep_shares = 0.0

    def _handle_spy_sweep(self, current_date: date):
        """
        Park idle cash in SPY when SPY is above its 50MA.
        Sell sweep when SPY drops below 50MA (bear protection).
        """
        sweep_config = self.profile.get('spy_sweep', {})
        if not sweep_config.get('enabled', False):
            return

        spy_price = self.data_provider.get_spy_price_on_date(current_date)
        if not spy_price or spy_price <= 0:
            return

        spy_data = self.data_provider.get_spy_daily_data(current_date)
        spy_ma50 = spy_data.get('ma50', 0)
        spy_above_50ma = spy_price > spy_ma50 if spy_ma50 > 0 else False

        # Sell sweep if SPY below 50MA (bear protection)
        if self.spy_sweep_shares > 0 and not spy_above_50ma:
            self._liquidate_spy_sweep(current_date, "SPY < 50MA")
            return

        # Buy sweep with idle cash if SPY above 50MA
        if spy_above_50ma and self.cash > 0:
            portfolio_value = self._get_portfolio_value(current_date)
            min_idle_pct = sweep_config.get('min_idle_pct', 20) / 100
            cash_reserve = portfolio_value * 0.05  # Keep 5% cash reserve
            idle_cash = self.cash - cash_reserve

            if idle_cash > portfolio_value * min_idle_pct:
                sweep_shares = idle_cash / spy_price
                self.cash -= idle_cash
                self.spy_sweep_shares += sweep_shares
                logger.debug(f"SPY SWEEP BUY: {sweep_shares:.1f} shares @ ${spy_price:.2f} "
                             f"= ${idle_cash:,.0f}")

    def _simulate_day(self, current_date: date):
        """Simulate one trading day with two-tier scoring for performance."""
        # Update position prices and peak tracking
        self._update_positions(current_date)

        # ===== MARKET STATE MACHINE =====
        # Replaces binary regime gate + advisory FTD with graduated exposure system.
        # In bull markets (SPY > 50MA), state stays TRENDING at 100% — identical to before.
        if self.market_state_enabled:
            spy_daily = self.data_provider.get_spy_daily_data(current_date)
            market_state_result = self.market_state.update(
                current_date=current_date,
                spy_close=spy_daily["close"],
                spy_prev_close=spy_daily["prev_close"],
                spy_volume=spy_daily["volume"],
                spy_prev_volume=spy_daily["prev_volume"],
                spy_ma50=spy_daily["ma50"],
                spy_ema21=spy_daily["ema21"],
                spy_ma200=spy_daily["ma200"],
            )

        # ===== DRAWDOWN CIRCUIT BREAKER =====
        portfolio_value = self._get_portfolio_value(current_date)
        current_drawdown = 0.0
        if self.peak_portfolio_value > 0:
            current_drawdown = ((self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value) * 100

        # Adjusted drawdown: exclude experimental position losses (nibble/V-bottom)
        # so they don't cascade into killing legitimate recovery positions.
        # The raw drawdown is still tracked for reporting (max_drawdown_pct).
        adjusted_drawdown = current_drawdown
        if self.experimental_realized_losses > 0 and self.peak_portfolio_value > 0:
            experimental_drawdown_pct = (self.experimental_realized_losses / self.peak_portfolio_value) * 100
            adjusted_drawdown = max(0.0, current_drawdown - experimental_drawdown_pct)
            if adjusted_drawdown < current_drawdown - 0.1:
                logger.debug(f"CB isolation: raw DD {current_drawdown:.1f}% -> adjusted {adjusted_drawdown:.1f}% "
                           f"(excluded ${self.experimental_realized_losses:.0f} experimental losses)")

        drawdown_config = config.get('ai_trader.drawdown_protection', {})
        halt_threshold = drawdown_config.get('halt_new_buys_pct', 15.0)
        liquidate_threshold = drawdown_config.get('liquidate_all_pct', 25.0)
        recovery_threshold = drawdown_config.get('recovery_pct', 10.0)

        if adjusted_drawdown >= liquidate_threshold and self.positions:
            # Emergency: liquidate all positions
            exp_note = f" (raw {current_drawdown:.1f}%, excl ${self.experimental_realized_losses:.0f} experimental)" if self.experimental_realized_losses > 0 else ""
            logger.warning(f"CIRCUIT BREAKER: {adjusted_drawdown:.1f}% adjusted drawdown >= {liquidate_threshold}% threshold - LIQUIDATING ALL{exp_note}")
            self.drawdown_halt = True
            held_tickers = list(self.positions.keys())
            held_scores = self._calculate_scores(current_date, tickers=held_tickers) if held_tickers else {}
            for ticker in list(self.positions.keys()):
                position = self.positions[ticker]
                price = self.data_provider.get_price_on_date(ticker, current_date)
                if price and price > 0:
                    score = held_scores.get(ticker, {}).get("total_score", 0)
                    cb_gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100 if position.cost_basis > 0 else 0
                    trade = SimulatedTrade(
                        ticker=ticker, action="SELL", shares=position.shares,
                        price=price, reason=f"CIRCUIT BREAKER: Portfolio drawdown {current_drawdown:.1f}%",
                        score=score, priority=0
                    )
                    trade._signal_factors = {"sell_reason": "CIRCUIT BREAKER", "gain_pct": round(cb_gain_pct, 1), "drawdown_pct": round(current_drawdown, 1)}
                    self._execute_sell(current_date, trade)
            self._take_snapshot(current_date)
            return

        if adjusted_drawdown >= halt_threshold:
            if not self.drawdown_halt:
                logger.warning(f"CIRCUIT BREAKER: {adjusted_drawdown:.1f}% drawdown - halting new buys")
            self.drawdown_halt = True
        elif self.drawdown_halt and adjusted_drawdown < recovery_threshold:
            logger.info(f"CIRCUIT BREAKER LIFTED: Drawdown recovered to {adjusted_drawdown:.1f}%")
            self.drawdown_halt = False

        # Tier 1: Score HELD positions every day (needed for sell/pyramid triggers)
        held_tickers = list(self.positions.keys())
        held_scores = self._calculate_scores(current_date, tickers=held_tickers) if held_tickers else {}

        # Evaluate and execute sells first
        sells = self._evaluate_sells(current_date, held_scores)
        for trade in sells:
            self._execute_sell(current_date, trade)

        # Tier 2: Check if we have cash to buy new positions
        portfolio_value = self._get_portfolio_value(current_date)

        # Dynamic cash reserve based on market regime
        market_for_cash = self.data_provider.get_market_direction(current_date)
        weighted_signal_cash = market_for_cash.get("weighted_signal", 0) if market_for_cash else 0
        alloc_config = config.get('ai_trader.allocation', {})
        if weighted_signal_cash >= 1.5:
            min_cash_pct = alloc_config.get('cash_reserve_strong_bull', 0.05)
        elif weighted_signal_cash >= 1.0:
            min_cash_pct = alloc_config.get('cash_reserve_bull', 0.10)
        elif weighted_signal_cash >= 0:
            min_cash_pct = alloc_config.get('cash_reserve_neutral', 0.20)
        elif weighted_signal_cash >= -0.5:
            min_cash_pct = alloc_config.get('cash_reserve_bear', 0.40)
        else:
            min_cash_pct = alloc_config.get('cash_reserve_strong_bear', 0.60)

        # Portfolio heat check — advisory penalty, not hard block
        heat_config = config.get('portfolio_heat', {})
        self.heat_penalty_active = False
        if heat_config.get('enabled', True):
            portfolio_heat = self._calculate_portfolio_heat(current_date)
            max_heat = heat_config.get('max_heat_pct', 15.0)
            if portfolio_heat > max_heat:
                self.heat_penalty_active = True
                logger.debug(f"Portfolio heat {portfolio_heat:.1f}% > {max_heat}% - applying score penalty + half-size buys")

        # Follow-Through Day check — advisory penalty, matches ai_trader fallback
        # Only applies when market state is disabled (nostate strategies)
        self.ftd_penalty_active = False
        if not self.market_state_enabled:
            market_timing_config = config.get('market_timing', {})
            ftd_config = market_timing_config.get('follow_through_day', {})
            if ftd_config.get('enabled', True) and self.data_provider:
                try:
                    ftd_status = self.data_provider.get_follow_through_day_status(current_date)
                    if not ftd_status.get("can_buy", True):
                        self.ftd_penalty_active = True
                        logger.debug(f"FTD penalty: {ftd_status['state']} on {current_date}")
                except Exception:
                    pass  # FTD is advisory — don't block on failure

        profile_max_positions = self.profile.get('max_positions', self.backtest.max_positions)
        can_buy = (not self.drawdown_halt and
                   self.cash / portfolio_value >= min_cash_pct and
                   len(self.positions) < profile_max_positions and
                   self.cash >= 100)

        # Track idle days (no positions, all cash) for re-seeding
        if not self.positions:
            self.idle_days += 1
        else:
            self.idle_days = 0

        # Track under-invested days (few positions, lots of cash) for supplemental seeding
        # Threshold is market-state-aware:
        #   - Volatile states (RECOVERY, early CONFIRMED): strict <= 2 prevents stop-out cascade
        #   - Established trends (TRENDING 10+ days, CONFIRMED 10+ days): relaxed < half_max
        #     catches the "zombie" state where 3 positions + 50% cash sits idle for months
        #   - Post-correction window (90 days): immediately relaxed, no 30-day wait
        in_post_correction_window = (
            self.last_correction_end_date is not None
            and (current_date - self.last_correction_end_date).days <= 90
        )
        underinvested_threshold = 2  # default: strict
        if (self.market_state_enabled and self.market_state.state in (MarketState.TRENDING, MarketState.CONFIRMED)
                and (self.market_state.state_days_count >= 30 or in_post_correction_window)):
            half_max = profile_max_positions // 2
            underinvested_threshold = max(half_max - 1, 2)  # e.g. 8//2 - 1 = 3

        if 0 < len(self.positions) <= underinvested_threshold:
            pv = self._get_portfolio_value(current_date)
            cash_pct = self.cash / pv if pv > 0 else 0
            if cash_pct > 0.50:
                self.underinvested_days += 1
            else:
                self.underinvested_days = 0
        else:
            self.underinvested_days = 0

        # ===== MARKET STATE-AWARE RE-SEEDING =====
        # When market transitions to RECOVERY after a correction and we have no positions,
        # seed with higher-quality stocks instead of sitting in cash forever.
        if self.market_state_enabled:
            ms = self.market_state

            # Reset peak after recovery transitions to prevent circuit breaker doom loop.
            # After a crash, old peak is irrelevant — seeds would get immediately liquidated
            # because portfolio value (e.g. $18K) is still far below pre-crash peak ($25K).
            if (market_state_result.get("changed") and
                    ms.state in (MarketState.RECOVERY, MarketState.CONFIRMED, MarketState.TRENDING)):
                old_peak = self.peak_portfolio_value
                new_peak = self._get_portfolio_value(current_date)
                if new_peak < old_peak * 0.90:  # Only reset if meaningful gap (>10%)
                    self.peak_portfolio_value = new_peak
                    self.drawdown_halt = False
                    # Reset experimental loss tracker — old losses are baked into the new peak
                    if self.experimental_realized_losses > 0:
                        logger.info(f"Backtest {self.backtest.id}: Resetting experimental loss tracker "
                                   f"(${self.experimental_realized_losses:.0f}) on peak reset")
                        self.experimental_realized_losses = 0.0
                    logger.info(f"Backtest {self.backtest.id}: PEAK RESET on {current_date} "
                               f"({ms.state.value}): ${old_peak:.0f} → ${new_peak:.0f}")

            # Track correction end dates for post-correction accelerated deployment.
            # When we exit CORRECTION, record the date so we can relax under-invested
            # thresholds during the 90-day recovery window.
            if (market_state_result.get("changed") and ms.state_history
                    and ms.state_history[-1]["from"] == "correction"):
                self.last_correction_end_date = current_date
                logger.info(f"Backtest {self.backtest.id}: CORRECTION ENDED on {current_date}, "
                           f"accelerated deployment window active for 90 days")

            # === V-BOTTOM DETECTOR ===
            # When SPY crashes 15%+ and rallies 8%+ from the low within 10 days,
            # this is a V-bottom pattern where FTD methodology is too slow.
            # Deploy a small number of positions before waiting for the formal FTD.
            v_bottom_config = self.profile.get('v_bottom', {})
            if (v_bottom_config.get('enabled', False)
                    and market_state_result.get("v_bottom", False)
                    and ms.state == MarketState.CORRECTION
                    and len(self.positions) <= 1):
                v_seed_count = v_bottom_config.get('seed_count', 2)
                v_seed_pct = v_bottom_config.get('seed_pct', 10)
                logger.info(f"Backtest {self.backtest.id}: V-BOTTOM SEED on {current_date} "
                           f"({len(self.positions)} positions, deploying {v_seed_count} positions)")
                self.is_seed_day = True
                self.recovery_seed_date = current_date
                pre_seed_tickers = set(self.positions.keys())
                self._seed_initial_positions(current_date, recovery_mode=True)
                self.recovery_seed_tickers = list(self.positions.keys())
                # Tag V-bottom seeds as experimental (isolated from circuit breaker)
                for ticker in self.positions:
                    if ticker not in pre_seed_tickers:
                        self.positions[ticker].is_experimental = True
                self._take_snapshot(current_date)
                return

            # Recovery seed: FTD just fired, we have 0-1 positions → seed
            # Only when truly depleted — 3 positions is normal portfolio churn, not depletion
            if (market_state_result.get("changed") and ms.state == MarketState.RECOVERY
                    and len(self.positions) <= 1):
                logger.info(f"Backtest {self.backtest.id}: RECOVERY SEED triggered on {current_date} "
                           f"(FTD detected, {len(self.positions)} positions)")
                self.is_seed_day = True
                self.recovery_seed_date = current_date
                self._seed_initial_positions(current_date, recovery_mode=True)
                self.recovery_seed_tickers = list(self.positions.keys())
                self._take_snapshot(current_date)
                return

            # Fast-track seed: CORRECTION → TRENDING via fast-track with depleted portfolio.
            # When fast-track skips RECOVERY entirely, recovery seed never fires.
            # This catches that case and seeds with normal parameters.
            if (market_state_result.get("changed") and ms.state == MarketState.TRENDING
                    and ms.last_transition_was_fast_track
                    and len(self.positions) <= 1):
                logger.info(f"Backtest {self.backtest.id}: FAST-TRACK SEED on {current_date} "
                           f"({len(self.positions)} positions, seeding to recover)")
                self.is_seed_day = True
                self._seed_initial_positions(current_date, recovery_mode=False)
                self._take_snapshot(current_date)
                return

            # Confirmation scale-up seed: when market advances from RECOVERY to
            # CONFIRMED or TRENDING and we still have few positions (< half max),
            # add more positions to scale up from the initial recovery seed.
            # In bull markets this never fires (no correction → no depletion).
            # Guard: only from recovery/confirmed transitions. NOT correction fast-tracks,
            # which fire repeatedly during jitter and cause deploy→stop-out doom loops.
            half_max_scaleup = profile_max_positions // 2
            if (market_state_result.get("changed")
                    and ms.state in (MarketState.CONFIRMED, MarketState.TRENDING)
                    and 0 < len(self.positions) < half_max_scaleup
                    and ms.state_history
                    and ms.state_history[-1]["from"] in ("recovery", "confirmed")):
                logger.info(f"Backtest {self.backtest.id}: SCALE-UP SEED on {current_date} "
                           f"({len(self.positions)}/{half_max_scaleup} positions, "
                           f"{ms.state_history[-1]['from']} → {ms.state.value})")
                self.is_seed_day = True
                self._seed_initial_positions(current_date, recovery_mode=False)
                self._take_snapshot(current_date)
                return

            # === FAST SCALE-UP: if recovery seeds are working, add more positions ===
            # Don't wait 15 days for under-invested timer — if seeds are up 3%+ after
            # 3 days, the recovery is confirmed. Deploy more capital immediately.
            fast_scaleup_config = self.profile.get('fast_scaleup', {})
            if (fast_scaleup_config.get('enabled', False)
                    and self.recovery_seed_date is not None
                    and ms.state in (MarketState.RECOVERY, MarketState.CONFIRMED, MarketState.TRENDING)
                    and ms.can_buy):
                days_since_seed = (current_date - self.recovery_seed_date).days
                check_after = fast_scaleup_config.get('check_after_days', 3)
                min_gain = fast_scaleup_config.get('min_seed_gain_pct', 3.0)
                max_additional = fast_scaleup_config.get('additional_positions', 2)

                if (days_since_seed >= check_after
                        and self.recovery_seed_tickers
                        and len(self.positions) < profile_max_positions
                        and len(self.positions) <= len(self.recovery_seed_tickers) + 1):
                    # Check if recovery seeds are profitable
                    seed_gains = []
                    for ticker in self.recovery_seed_tickers:
                        if ticker in self.positions:
                            pos = self.positions[ticker]
                            price = self.data_provider.get_price_on_date(ticker, current_date)
                            if price and pos.cost_basis > 0:
                                gain = ((price - pos.cost_basis) / pos.cost_basis) * 100
                                seed_gains.append(gain)

                    if seed_gains:
                        avg_gain = sum(seed_gains) / len(seed_gains)
                        if avg_gain >= min_gain:
                            logger.info(f"Backtest {self.backtest.id}: FAST SCALE-UP on {current_date} "
                                       f"(recovery seeds avg +{avg_gain:.1f}% after {days_since_seed} days, "
                                       f"adding up to {max_additional} positions)")
                            self.is_seed_day = True
                            self._seed_initial_positions(current_date, recovery_mode=False)
                            self.recovery_seed_date = None  # Don't re-trigger
                            self.recovery_seed_tickers = []
                            self._take_snapshot(current_date)
                            return

            # Fallback idle re-seed: respects market state (must be RECOVERY+ to seed)
            if (can_buy and not self.positions and self.idle_days >= 10
                    and ms.can_seed):
                logger.info(f"Backtest {self.backtest.id}: Portfolio idle {self.idle_days} days, "
                           f"re-seeding in {ms.state.value} state on {current_date}")
                self.is_seed_day = True
                recovery_mode = ms.state in (MarketState.RECOVERY, MarketState.CONFIRMED)
                self._seed_initial_positions(current_date, recovery_mode=recovery_mode)
                self._take_snapshot(current_date)
                return

            # Under-invested re-seed: in TRENDING/CONFIRMED with few positions
            # and > 50% cash for N+ days → the "zombie" state. Seed back to normal.
            # Patience adapts to market stability:
            #   - Post-correction window: 7 days (capital needs deploying, correction already proved downturn)
            #   - Established TRENDING (30+ days): 10 days (market proven, deploy capital)
            #   - Default: 15 days (still validating, be patient)
            # Guard: skip if drawdown is high — circuit breaker would just kill the positions.
            if in_post_correction_window:
                underinvested_patience = 7  # Accelerated: correction already validated the downturn
            elif ms.state == MarketState.TRENDING and ms.state_days_count >= 30:
                underinvested_patience = 10
            else:
                underinvested_patience = 15
            current_dd = 0.0
            if self.peak_portfolio_value > 0:
                current_dd = ((self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value) * 100
            if (self.underinvested_days >= underinvested_patience
                    and ms.state in (MarketState.TRENDING, MarketState.CONFIRMED)
                    and ms.can_buy
                    and current_dd < halt_threshold):
                logger.info(f"Backtest {self.backtest.id}: UNDER-INVESTED SEED on {current_date} "
                           f"({len(self.positions)} positions, {self.underinvested_days} days under-invested, "
                           f"state={ms.state.value} for {ms.state_days_count} days, dd={current_dd:.1f}%)")
                self.is_seed_day = True
                self._seed_initial_positions(current_date, recovery_mode=False)
                self.underinvested_days = 0
                self._take_snapshot(current_date)
                return

            # Market state replaces binary regime gate
            # BUT: nibble mode allows small buys during CORRECTION
            nibble_config = self.profile.get('correction_nibble', {})
            if not ms.can_buy:
                if (nibble_config.get('enabled', False)
                        and ms.state == MarketState.CORRECTION
                        and len(self.positions) < nibble_config.get('max_positions', 2)):
                    # Nibble mode: allow buying but with tight exposure cap
                    # The exposure cap enforcement in the buy loop (line ~1050) will
                    # limit actual deployment to nibble_config.max_exposure_pct
                    can_buy = True  # Override the gate — exposure cap handles the limit
                    self.nibble_mode_active = True
                    logger.debug(f"NIBBLE MODE active: allowing up to "
                               f"{nibble_config.get('max_positions', 2)} positions "
                               f"at {nibble_config.get('max_exposure_pct', 15)}% max exposure")
                else:
                    can_buy = False
                    self.nibble_mode_active = False
            else:
                self.nibble_mode_active = False
        else:
            # Legacy fallback: binary regime gate (for backward compat if market_state disabled)
            regime_gate_config = config.get('ai_trader.market_regime_gate', {})
            # Profile-level override: allow disabling the regime gate entirely
            profile_gate = self.profile.get('regime_gate', {})
            gate_enabled = profile_gate.get('enabled', regime_gate_config.get('enabled', True))
            self.correction_zone_active = False  # Reset each day
            if gate_enabled and can_buy:
                spy_data = market_for_cash.get('spy', {}) if market_for_cash else {}
                spy_price = spy_data.get('price', 0)
                spy_ma50 = spy_data.get('ma_50', 0)
                spy_ma200 = spy_data.get('ma_200', 0)
                if not spy_price or not spy_ma50:
                    # Missing SPY data — assume bearish (conservative)
                    logger.debug(f"REGIME GATE: SPY data missing (price={spy_price}, ma50={spy_ma50}), skipping buys")
                    can_buy = False
                elif spy_price < spy_ma50:
                    # SPY below 50MA — check for correction zone override
                    cz_config = self.profile.get('correction_zone', {})
                    if (cz_config.get('enabled', False)
                            and spy_ma200 > 0
                            and spy_price >= spy_ma200):
                        # CORRECTION ZONE: SPY between 200MA and 50MA
                        # Allow buying with tighter filters + reduced position size
                        self.correction_zone_active = True
                        self.overlay_stats['cz_active_days'] += 1
                        can_buy = True
                        logger.debug(
                            f"CORRECTION ZONE: SPY ${spy_price:.2f} between "
                            f"200MA ${spy_ma200:.2f} and 50MA ${spy_ma50:.2f}, "
                            f"allowing filtered buys")
                    else:
                        # Full bear: SPY below both 50MA and 200MA, or correction zone disabled
                        logger.debug(f"REGIME GATE: SPY ${spy_price:.2f} below 50MA ${spy_ma50:.2f}, skipping buys")
                        can_buy = False

            # Score floor decay: track under-invested days in bull market
            decay_config = self.profile.get('score_floor_decay', {})
            if decay_config.get('enabled', False):
                max_positions_for_decay = decay_config.get('max_positions', 2)
                if can_buy and len(self.positions) <= max_positions_for_decay:
                    self.underinvested_days += 1
                elif len(self.positions) > max_positions_for_decay:
                    self.underinvested_days = 0
                # Note: when SPY < 50MA (can_buy=False), counter freezes — we WANT to hold cash in bears

            # Legacy idle re-seed
            if can_buy and not self.positions and self.idle_days >= 10:
                logger.info(f"Backtest {self.backtest.id}: Portfolio idle {self.idle_days} days, re-seeding on {current_date}")
                self._liquidate_spy_sweep(current_date, "idle re-seed")
                self.is_seed_day = True
                self._seed_initial_positions(current_date)
                self._handle_spy_sweep(current_date)
                self._take_snapshot(current_date)
                return

        # Liquidate SPY sweep before evaluating buys (active picks get first dibs on cash)
        if can_buy and self.spy_sweep_shares > 0:
            self._liquidate_spy_sweep(current_date, "freeing cash for active buys")

        if can_buy:
            # Score full universe when we can buy
            all_scores = self._calculate_scores(current_date)
            all_scores.update(held_scores)  # Ensure held positions have scores

            # Evaluate pyramids with full scores
            pyramids = self._evaluate_pyramids(current_date, all_scores)
            for trade in pyramids[:3]:
                self._execute_pyramid(current_date, trade)

            # Evaluate and execute buys (with market state exposure cap + buy throttle)
            buys = self._evaluate_buys(current_date, all_scores)
            buy_throttle_limit = self._get_buy_throttle_limit()
            buys_executed_today = 0
            for trade in buys:
                if self.cash < 100:
                    break
                if buys_executed_today >= buy_throttle_limit:
                    break

                # In nibble mode, cap positions at nibble limit (not profile max)
                nibble_active = getattr(self, 'nibble_mode_active', False)
                nibble_cfg = self.profile.get('correction_nibble', {})
                if nibble_active:
                    nibble_max_pos = nibble_cfg.get('max_positions', 2)
                    if len(self.positions) >= nibble_max_pos:
                        break
                elif len(self.positions) >= profile_max_positions:
                    break

                # Enforce market state exposure cap (nibble mode uses its own cap)
                if self.market_state_enabled:
                    if nibble_active:
                        max_exposure = nibble_cfg.get('max_exposure_pct', 15) / 100
                    else:
                        max_exposure = self.market_state.max_exposure_pct
                    current_invested = sum(
                        pos.shares * (self.data_provider.get_price_on_date(t, current_date) or pos.cost_basis)
                        for t, pos in self.positions.items()
                    )
                    pv = self._get_portfolio_value(current_date)
                    available_invest = (max_exposure * pv) - current_invested
                    if available_invest < 100:
                        logger.debug(f"Exposure cap: {max_exposure:.0%} reached, skipping remaining buys")
                        break
                    # Scale down trade if it would exceed exposure cap
                    trade_value = trade.shares * trade.price
                    if trade_value > available_invest:
                        trade.shares = available_invest / trade.price
                    # Skip if adjusted position is too small to be meaningful
                    if trade.shares * trade.price < 100:
                        continue
                self._execute_buy(current_date, trade)
                buys_executed_today += 1
        else:
            # Only evaluate pyramids for held positions
            pyramids = self._evaluate_pyramids(current_date, held_scores)
            for trade in pyramids[:3]:
                self._execute_pyramid(current_date, trade)

        # Re-sweep idle cash into SPY after buys are done
        self._handle_spy_sweep(current_date)

        # Take daily snapshot
        self._take_snapshot(current_date)

    def _seed_initial_positions(self, first_day: date, recovery_mode: bool = False):
        """
        Seed the portfolio with top CANSLIM stocks.

        In normal mode (day 1): Uses low score floor (35) since C/A scores are
        typically 0 on backtest day 1 due to earnings data filtering.

        In recovery mode (post-FTD): Uses higher score floor (55) and fewer
        positions (2 instead of 3) for cautious re-entry after a correction.
        This prevents the old problem of seeding garbage-quality stocks (score 43-50)
        into a falling market.
        """
        scores = self._calculate_scores(first_day)
        if not scores:
            return

        # Recovery mode uses higher quality floor to prevent crash-period garbage seeding
        recovery_config = config.get('market_state.recovery_seed', {})
        if recovery_mode:
            seed_min_score = recovery_config.get('min_score', 55)
        else:
            # Low floor since C and A scores are typically 0 on backtest day 1
            seed_min_score = 35

        # Skip C/A quality filters — earnings data is unavailable for day 1 scoring.
        # Instead rely on L score (relative strength) as the primary quality signal.
        candidates = []
        for ticker, data in scores.items():
            total_score = data.get("total_score", 0)
            if total_score < seed_min_score:
                continue

            # Require decent relative strength (the one reliable signal on day 1)
            l_score = data.get('l_score', 0)
            min_l = recovery_config.get('min_l_score', 6) if recovery_mode else 8
            if l_score < min_l:
                continue

            price = self.data_provider.get_price_on_date(ticker, first_day)
            if not price or price <= 0:
                continue

            candidates.append((ticker, data, price, total_score))

        if not candidates:
            mode_str = "recovery" if recovery_mode else "initial"
            logger.info(f"Backtest {self.backtest.id}: No stocks qualify for {mode_str} seeding on {first_day}")
            return

        # Sort by total score descending, take top N
        candidates.sort(key=lambda x: x[3], reverse=True)
        if recovery_mode:
            max_seed_positions = recovery_config.get('max_positions', 2)
            seed_pct_per_position = recovery_config.get('seed_pct', 10.0)
        else:
            max_seed_positions = self.profile.get('seed_count', 5)
            seed_pct_per_position = self.profile.get('seed_pct', 10.0)
        seeds = candidates[:max_seed_positions]

        mode_str = "RECOVERY" if recovery_mode else "INITIAL"
        logger.info(f"Backtest {self.backtest.id}: {mode_str} seeding {len(seeds)} positions on {first_day}: "
                    f"{[s[0] for s in seeds]}")

        # E: Cap total seed investment at max_seed_investment_pct of portfolio
        # Recovery mode uses its own cap (typically 30%) to avoid over-committing
        if recovery_mode:
            max_seed_invest_pct = recovery_config.get('max_exposure_pct', 30)
        else:
            max_seed_invest_pct = self.profile.get('max_seed_investment_pct', 60)
        max_seed_cash = self.cash * (max_seed_invest_pct / 100)
        seed_cash_spent = 0

        for ticker, data, price, total_score in seeds:
            portfolio_value = self._get_portfolio_value(first_day)
            position_value = portfolio_value * (seed_pct_per_position / 100)
            # E: Respect seed investment cap
            remaining_seed_budget = max_seed_cash - seed_cash_spent
            position_value = min(position_value, remaining_seed_budget, self.cash * 0.70)

            if position_value < 100 or self.cash < 100:
                break

            shares = position_value / price
            has_base = data.get("has_base_pattern", False)
            base_type = data.get("base_pattern", {}).get("type", "none") if has_base else "none"
            pct_from_high = data.get("pct_from_high", 0)

            seed_label = "RECOVERY SEED" if recovery_mode else "INITIAL SEED"
            reason = f"🏁 {seed_label} | Score {total_score:.0f} | {pct_from_high:.1f}% from high"
            if has_base:
                reason += f" | Base: {base_type}"

            trade = SimulatedTrade(
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                reason=reason,
                score=total_score,
                priority=0
            )
            self._execute_buy(first_day, trade)
            seed_cash_spent += position_value

        self._take_snapshot(first_day)

    def _update_positions(self, current_date: date):
        """Update position prices and track peak for trailing stops"""
        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if price and price > 0:
                # Track peak price for trailing stop
                if price > position.peak_price:
                    position.peak_price = price
                    position.peak_date = current_date

    def _calculate_scores(self, current_date: date, tickers: List[str] = None) -> Dict[str, dict]:
        """
        Calculate full CANSLIM scores for stocks, with daily caching.

        Scores are cached per (ticker, date) so that multiple calls on the same day
        (circuit breaker, sell eval, buy eval) don't recalculate from scratch.
        Cache is invalidated when the date changes.

        ── 4-LAYER SCORE CACHE HIERARCHY (read order) ──────────────────────
        Layer 1 — in-memory  : self._score_cache, per-day, cleared on date change.
        Layer 2 — StockScore : frozen scanner snapshots from the live DB.
                              Gated by `backtester.use_frozen_scores` (default
                              true — KEEP IT TRUE; see "WHY LAYER 4 IS WRONG").
                              When a scanner snapshot exists for (stock, date),
                              the backtester serves it INSTEAD of computing.
        Layer 3 — SQLite     : /tmp/backtest_cache/scores_v{N}.db, persists
                              ACROSS container restarts (volume mount). Once a
                              score for (ticker, date) is written, never updates
                              (INSERT OR IGNORE).
        Layer 4 — fresh      : compute via self._scorer (canslim_scorer) +
                              inline A/N/S/L/I/M blocks. ONLY runs when Layers
                              1-3 all miss. This path is semantically broken
                              for historical replay — see below.

        ── WHY LAYER 4 IS SEMANTICALLY WRONG FOR HISTORICAL REPLAY ─────────
        BacktestStaticSnapshot freezes the live DB's POINT-IN-TIME SCALARS
        (earnings_surprise_pct, eps_estimate_revision_pct, eps_beat_streak,
        roe, institutional_holders_pct) at backtest-creation time and applies
        them to every historical replay date. The earnings/revenue ARRAYS get
        filtered to as_of_date by HistoricalDataProvider._filter_available_earnings,
        so those are time-correct. The scalars are NOT filtered — Layer 4
        scoring a 2022 trading day will read today's "latest surprise %",
        today's "current beat streak", today's ROE.

        The frozen StockScore rows (Layer 2) are correct *because* the live
        scanner wrote them at each scan date with then-current scalars. Layer
        4 has no analog of that — without a per-date scalar history, fresh
        compute is a category error for historical replay.

        We learned this May 7 2026 testing Approach 2: flipping
        use_frozen_scores=false dropped a known-good 4yr backtest from
        +136.84% to -14.99%. The candidate (with cap) drifted to +111.95%
        and looked superficially "better" only because both fresh-compute
        runs were broken in different ways. Reverted to true in commit 6cf001e.

        ── HOW TO EVALUATE A canslim_scorer CHANGE ─────────────────────────
        Backtest replay can NOT validate a new scoring rule. The historical
        StockScore rows are frozen output of the OLD scorer. To evaluate:
          1. Deploy the new code.
          2. Let the live scanner write fresh StockScore rows under the new
             logic for going-forward dates.
          3. Run a forward (out-of-sample) period to compare.
        This is more expensive than backtesting, but it's the only honest
        path. See canslim_scorer-refactor-may6.md memory note.

        ── INVALIDATION (Layer 3 cache) ────────────────────────────────────
        When canslim_scorer logic changes, Layer 3 caches stale fresh-compute
        outputs. Bump SCORE_CACHE_VERSION to fork to a new SQLite file. Old
        cache is orphaned (not deleted; reclaim disk manually if you care).
        ────────────────────────────────────────────────────────────────────

        Args:
            current_date: Date to calculate scores for
            tickers: Optional list of tickers to score. If None, scores all available.
        """
        date_str = str(current_date)

        # Invalidate cache on new day
        if self._score_cache_date != date_str:
            self._score_cache = {}
            self._score_cache_date = date_str

        scores = {}

        # Get market direction once (same for all stocks)
        market = self.data_provider.get_market_direction(current_date)

        ticker_list = tickers if tickers is not None else self.data_provider.get_available_tickers()

        # Layer 1: Check in-memory cache (same-day reuse)
        uncached_tickers = []
        for ticker in ticker_list:
            if ticker in self._score_cache:
                scores[ticker] = self._score_cache[ticker]
            else:
                uncached_tickers.append(ticker)

        # Layer 2: Frozen StockScore snapshots (scanner-generated, deterministic)
        use_frozen = config.get('backtester.use_frozen_scores', True)
        if use_frozen and uncached_tickers:
            frozen_fundamentals = self._get_frozen_scores(date_str, uncached_tickers)
            if frozen_fundamentals:
                frozen_built = {}
                for ticker, frozen in frozen_fundamentals.items():
                    score_dict = self._build_score_from_frozen(ticker, current_date, frozen)
                    if score_dict:
                        frozen_built[ticker] = score_dict
                        scores[ticker] = score_dict
                        self._score_cache[ticker] = score_dict
                if frozen_built:
                    self._save_persistent_scores(date_str, frozen_built)
                    uncached_tickers = [t for t in uncached_tickers if t not in frozen_built]

        # Layer 3: Check persistent disk cache (cross-run reuse)
        if uncached_tickers:
            persistent_hits = self._get_persistent_scores(date_str, uncached_tickers)
            if persistent_hits:
                for ticker, cached_score in persistent_hits.items():
                    scores[ticker] = cached_score
                    self._score_cache[ticker] = cached_score  # Promote to in-memory
                uncached_tickers = [t for t in uncached_tickers if t not in persistent_hits]

        # Layer 3: Compute scores for tickers not in any cache
        newly_computed = {}
        for ticker in uncached_tickers:
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0:
                continue

            # Get historical stock data with point-in-time earnings
            static_data = self.static_data.get(ticker, {})
            stock_data = self.data_provider.get_stock_data_on_date(ticker, current_date, static_data)

            high_52w, low_52w = self.data_provider.get_52_week_high_low(ticker, current_date)
            rs_12m = self.data_provider.get_relative_strength(ticker, current_date, 12)
            rs_3m = self.data_provider.get_relative_strength(ticker, current_date, 3)

            # ===== C SCORE (15 pts): Current Quarterly Earnings =====
            # Routed through canslim_scorer._score_current_earnings (May 6 —
            # replaces inline duplicate that drifted from the live scorer.
            # Critically, the inline path was missing the surprise / beat-streak
            # / estimate-revision bonus block (`_apply_earnings_bonuses`), so
            # backtests silently lost up to ~10pt of C-score signal that live
            # trading uses. Approach 2's excellence-tier cap was a no-op partly
            # for this reason — signal #4 (surprise≥10%) was unreachable.
            quarterly_earnings = [e for e in (stock_data.quarterly_earnings or [])
                                  if e is not None and not (isinstance(e, float) and math.isnan(e))]

            # Sector thresholds — kept here because the A-score block below
            # also references c_excellent / c_good (the A score reuses the
            # same sector-adjusted thresholds).
            sector = static_data.get("sector", "default")
            sector_thresholds = {
                'Technology': {'excellent': 30, 'good': 20},
                'Communication Services': {'excellent': 25, 'good': 18},
                'Consumer Discretionary': {'excellent': 25, 'good': 18},
                'Healthcare': {'excellent': 25, 'good': 15},
                'Financials': {'excellent': 20, 'good': 12},
                'Industrials': {'excellent': 20, 'good': 12},
                'Consumer Staples': {'excellent': 15, 'good': 10},
                'Materials': {'excellent': 18, 'good': 12},
                'Energy': {'excellent': 18, 'good': 10},
                'Real Estate': {'excellent': 15, 'good': 10},
                'Utilities': {'excellent': 12, 'good': 8},
            }.get(sector, {'excellent': 25, 'good': 15})
            c_excellent = sector_thresholds['excellent']
            c_good = sector_thresholds['good']

            c_score, _c_detail = self._scorer._score_current_earnings(stock_data)

            # eps_growth — derived independently for projected_growth (a
            # downstream trading-decision signal that ai_trader gates on).
            # Mirrors the prior inline derivation so projected_growth stays
            # numerically stable across this refactor.
            eps_growth = 0.0
            if len(quarterly_earnings) >= 5:
                if len(quarterly_earnings) >= 8:
                    current_ttm = sum(quarterly_earnings[0:4])
                    prior_ttm = sum(quarterly_earnings[4:8])
                else:
                    current_ttm = quarterly_earnings[0]
                    prior_ttm = quarterly_earnings[4]

                if current_ttm > 0 and prior_ttm < 0:
                    eps_growth = ((current_ttm - prior_ttm) / abs(prior_ttm)) * 100
                elif current_ttm < 0:
                    eps_growth = 0
                elif abs(prior_ttm) < 0.01:
                    eps_growth = 100 if current_ttm > 0 else 0
                else:
                    eps_growth = ((current_ttm - prior_ttm) / abs(prior_ttm)) * 100
            elif len(quarterly_earnings) >= 2:
                current_eps = quarterly_earnings[0]
                prior_eps = quarterly_earnings[1]
                if prior_eps > 0 and current_eps > 0:
                    eps_growth = ((current_eps - prior_eps) / abs(prior_eps)) * 100
                elif current_eps > 0 and prior_eps <= 0:
                    eps_growth = 50

            # ===== A SCORE (15 pts): Annual Earnings Growth =====
            # Synced with canslim_scorer.py: sector-adjusted CAGR + ROE + turnaround
            a_score = 0
            annual_cagr = 0.0  # Track for projected_growth calculation
            annual_earnings = [e for e in (stock_data.annual_earnings or [])
                               if e is not None and not (isinstance(e, float) and math.isnan(e))]
            roe = static_data.get("roe", 0)

            if len(annual_earnings) >= 3:
                # Calculate 3-year CAGR
                current_annual = annual_earnings[0]
                three_years_ago = annual_earnings[2]

                if three_years_ago > 0 and current_annual > 0:
                    cagr = ((current_annual / three_years_ago) ** (1/3) - 1) * 100
                    annual_cagr = cagr  # Save for projected_growth
                    # Sector-adjusted thresholds (reuse from C score)
                    if cagr >= c_excellent:
                        a_score = 12
                    elif cagr >= c_good:
                        range_pct = (cagr - c_good) / (c_excellent - c_good)
                        a_score = round(12 * (0.6 + 0.4 * range_pct), 1)
                    elif cagr >= 0:
                        a_score = round((cagr / c_good) * 12 * 0.6, 1)
                    else:
                        a_score = 0
                elif current_annual > 0 and three_years_ago <= 0:
                    # Turnaround: negative->positive annual earnings
                    a_score = 10  # 70% of max, synced with canslim_scorer
                    annual_cagr = 50  # Proxy for projected_growth

                # ROE bonus (17%+ is quality threshold)
                roe_pct = roe * 100 if abs(roe) < 5 else roe  # Handle decimal vs pct format
                if roe_pct >= 17:
                    a_score = min(15, a_score + 3)
                elif roe_pct >= 12:
                    a_score = min(15, a_score + 1)

            # ===== N SCORE (15 pts): New Highs / Pivot Proximity =====
            base_pattern = self.data_provider.detect_base_pattern(ticker, current_date)
            has_base = base_pattern["type"] != "none"
            pivot_price = base_pattern.get("pivot_price", 0) if has_base else high_52w
            weeks_in_base = base_pattern.get("weeks", 0)

            n_score = 0
            pct_from_pivot = 0
            pct_from_high = ((high_52w - price) / high_52w) * 100 if high_52w > 0 else 100

            if pivot_price > 0:
                pct_from_pivot = ((pivot_price - price) / pivot_price) * 100

                if has_base:
                    # Proper base pattern scoring
                    if 0 < pct_from_pivot <= 5:
                        n_score = 15  # Optimal pre-breakout
                    elif -3 <= pct_from_pivot <= 0:
                        n_score = 14  # At breakout point
                    elif 5 < pct_from_pivot <= 10:
                        n_score = 12
                    elif -8 <= pct_from_pivot < -3:
                        n_score = 10
                    elif pct_from_pivot < -8:
                        n_score = 5  # Too extended
                    elif pct_from_pivot > 10:
                        n_score = 6
                    else:
                        n_score = 4
                else:
                    # No base pattern: reduced scores
                    if pct_from_high <= 5:
                        n_score = 10
                    elif pct_from_high <= 10:
                        n_score = 9
                    elif pct_from_high <= 15:
                        n_score = 7
                    elif pct_from_high <= 25:
                        n_score = 4
                    else:
                        n_score = 0

            # ===== S SCORE (15 pts): Supply/Demand (Volume + Price Trend + A/D) =====
            # Synced with canslim_scorer.py: vol (6) + price trend (5) + accum/distrib (4)
            s_score = 0
            avg_volume = self.data_provider.get_50_day_avg_volume(ticker, current_date)
            current_volume = self.data_provider.get_volume_on_date(ticker, current_date) or 0

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume if current_volume else 1.0

                # Volume component (up to 6 points)
                if volume_ratio >= 2.0:
                    s_score = 6
                elif volume_ratio >= 1.5:
                    s_score = 5
                elif volume_ratio >= 1.2:
                    s_score = 3

                # Price trend component (up to 5 points)
                try:
                    price_history = self.data_provider.get_price_history(ticker, current_date, 20)
                    if price_history is not None and isinstance(price_history, list) and len(price_history) >= 2:
                        first_price = price_history[0] if price_history[0] > 0 else 1
                        price_change_20d = (price_history[-1] - first_price) / first_price
                        if price_change_20d > 0.05 and volume_ratio > 1.2:
                            s_score += 5
                        elif price_change_20d > 0.03:
                            s_score += 3
                        elif price_change_20d > 0:
                            s_score += 2
                except (TypeError, AttributeError):
                    pass  # Data provider method may not exist or return unexpected type

                # Accumulation/Distribution component (up to 4 points)
                try:
                    ad_data = self.data_provider.get_accumulation_distribution(ticker, current_date, 20)
                    if ad_data is not None and isinstance(ad_data, dict):
                        ad_ratio = ad_data.get("ad_ratio", 1.0)
                        if isinstance(ad_ratio, (int, float)) and ad_ratio == ad_ratio:  # NaN guard
                            if ad_ratio >= 1.8:
                                s_score += 4
                            elif ad_ratio >= 1.3:
                                s_score += 3
                            elif ad_ratio >= 1.0:
                                s_score += 1
                            elif ad_ratio < 0.7:
                                s_score = max(0, s_score - 1)
                except (TypeError, AttributeError):
                    pass  # Data provider method may not exist or return unexpected type

                s_score = min(s_score, 15)

            # ===== L SCORE (15 pts): Leader/Laggard (Relative Strength) =====
            # Synced with canslim_scorer.py: 3-timeframe RS + improving bonus
            try:
                rs_1m = self.data_provider.get_relative_strength(ticker, current_date, 1)
                if not isinstance(rs_1m, (int, float)) or rs_1m != rs_1m:
                    rs_1m = 1.0
            except (TypeError, AttributeError):
                rs_1m = 1.0
            # Cap RS values at 3.0 to prevent crash-period inflation
            rs_12m_capped = min(rs_12m, 3.0) if isinstance(rs_12m, (int, float)) else 1.0
            rs_3m_capped = min(rs_3m, 3.0) if isinstance(rs_3m, (int, float)) else 1.0
            rs_1m_capped = min(rs_1m, 3.0)
            # Weighted RS: 40% 12-month + 35% 3-month + 25% 1-month
            combined_rs = rs_12m_capped * 0.40 + rs_3m_capped * 0.35 + rs_1m_capped * 0.25

            # Smoother threshold transitions (linear interpolation)
            if combined_rs >= 1.5:
                l_score = 15
            elif combined_rs >= 1.3:
                l_score = 15 * (0.85 + 0.15 * (combined_rs - 1.3) / 0.2)
            elif combined_rs >= 1.15:
                l_score = 15 * (0.70 + 0.15 * (combined_rs - 1.15) / 0.15)
            elif combined_rs >= 1.0:
                l_score = 15 * (0.50 + 0.20 * (combined_rs - 1.0) / 0.15)
            elif combined_rs >= 0.85:
                l_score = 15 * (0.30 + 0.20 * (combined_rs - 0.85) / 0.15)
            elif combined_rs >= 0.7:
                l_score = 15 * (0.10 + 0.20 * (combined_rs - 0.7) / 0.15)
            else:
                l_score = 0

            # RS improving bonus
            if rs_3m_capped > rs_12m_capped and l_score > 0:
                l_score = min(l_score + 1, 15)
            l_score = round(l_score, 1)

            # ===== I SCORE (10 pts): Institutional Ownership =====
            i_score = 0
            inst_pct = static_data.get("institutional_holders_pct", 0)
            # inst_pct is stored as percentage (e.g., 65 = 65%), matching live scorer
            if 25 <= inst_pct <= 75:
                i_score = 10
            elif 15 <= inst_pct < 25 or 75 < inst_pct <= 85:
                i_score = 7
            elif 10 <= inst_pct < 15 or 85 < inst_pct <= 90:
                i_score = 4
            elif inst_pct > 0:
                i_score = 2
            # If no data, give benefit of doubt with middle score
            else:
                i_score = 5

            # ===== M SCORE (15 pts): Market Direction =====
            # Granular scoring from continuous per-index analysis
            # Mirrors data_fetcher.calculate_index_m_score() used in live trading
            composite_m = market.get("composite_m", 0.5) if market else 0.5
            m_score = round(composite_m * 15.0, 1)

            # ===== PROJECTED GROWTH (independent of CANSLIM score) =====
            # Combine EPS growth rate, annual CAGR, and price momentum
            # Weights tuned to reduce FMP-dependent signal (eps+cagr = 55%, was 70%)
            momentum_pct = (combined_rs - 1.0) * 100  # RS ratio to %
            projected_growth = (
                eps_growth * 0.30 +      # 30% quarterly EPS growth rate (was 40%)
                annual_cagr * 0.25 +     # 25% annual earnings CAGR (was 30%)
                momentum_pct * 0.45      # 45% price momentum vs market (was 30%)
            )

            # ===== TOTAL SCORE (100 pts) =====
            total_score = c_score + a_score + n_score + s_score + l_score + i_score + m_score

            # Get breakout status for buy decisions
            is_breaking_out, volume_ratio_breakout, _ = self.data_provider.is_breaking_out(
                ticker, current_date
            )

            # Determine if this is a growth stock (negative/no earnings but high revenue growth)
            is_growth_stock = False
            if len(quarterly_earnings) >= 2 and quarterly_earnings[0] <= 0:
                quarterly_revenue = stock_data.quarterly_revenue or []
                if len(quarterly_revenue) >= 5:
                    current_rev = quarterly_revenue[0]
                    year_ago_rev = quarterly_revenue[4]
                    if year_ago_rev > 0 and current_rev > 0:
                        rev_growth = ((current_rev - year_ago_rev) / year_ago_rev) * 100
                        if rev_growth >= 30:
                            is_growth_stock = True

            # Volume dry-up detection (uses price history)
            ad_data = self.data_provider.get_accumulation_distribution(ticker, current_date, 20)
            _volume_dry_up = ad_data.get("volume_dry_up", False) if ad_data else False
            _volume_dry_up_score = ad_data.get("volume_dry_up_score", 0) if ad_data else 0

            score_result = {
                "total_score": total_score,
                "c_score": c_score,
                "a_score": a_score,
                "n_score": n_score,
                "s_score": s_score,
                "l_score": l_score,
                "i_score": i_score,
                "m_score": m_score,
                "rs_12m": rs_12m,
                "rs_3m": rs_3m,
                "projected_growth": projected_growth,
                "pct_from_high": pct_from_high,
                "pct_from_pivot": pct_from_pivot,
                "pivot_price": pivot_price,
                "has_base_pattern": has_base,
                "base_pattern": base_pattern,
                "weeks_in_base": weeks_in_base,
                "is_growth_stock": is_growth_stock,
                "is_breaking_out": is_breaking_out,
                "volume_ratio": volume_ratio_breakout if volume_ratio_breakout else (current_volume / avg_volume if avg_volume > 0 else 1.0),
                "_volume_dry_up": _volume_dry_up,
                "_volume_dry_up_score": _volume_dry_up_score,
            }
            scores[ticker] = score_result
            self._score_cache[ticker] = score_result  # Cache for reuse within same day
            newly_computed[ticker] = score_result

        # Persist newly computed scores to disk for future backtest runs
        if newly_computed:
            self._save_persistent_scores(date_str, newly_computed)

        # Post-process: compute industry group rankings and adjust L scores
        self._apply_industry_group_bonus(scores)

        return scores

    def _compute_sector_rs_rank(self, ticker: str, scores: dict) -> float:
        """Compute the percentile rank of this stock's sector based on median RS."""
        sector = self.static_data.get(ticker, {}).get("sector", "Unknown")
        # Group all scored stocks by sector, collect rs_12m
        sector_rs = {}
        for t, data in scores.items():
            s = self.static_data.get(t, {}).get("sector", "Unknown")
            rs = data.get("rs_12m")
            if rs is not None and rs == rs:  # NaN guard
                sector_rs.setdefault(s, []).append(rs)
        if not sector_rs or sector not in sector_rs:
            return 50.0
        # Compute median per sector
        sector_medians = {}
        for s, vals in sector_rs.items():
            sorted_vals = sorted(vals)
            sector_medians[s] = sorted_vals[len(sorted_vals) // 2]
        # Rank this sector among all sectors (percentile)
        sorted_medians = sorted(sector_medians.values())
        my_median = sector_medians[sector]
        rank = sorted_medians.index(my_median)
        n = max(1, len(sorted_medians) - 1)
        return round((rank / n) * 100, 1)

    def _update_score_history(self, ticker: str, score: float):
        """Track score history for stability checks"""
        if ticker not in self.score_history:
            self.score_history[ticker] = []
        self.score_history[ticker].append(score)
        # Keep last 5 scores (about 5 trading days worth)
        if len(self.score_history[ticker]) > 5:
            self.score_history[ticker] = self.score_history[ticker][-5:]

    def _check_score_stability(self, ticker: str, current_score: float, threshold: float = 50) -> dict:
        """
        Check if a low score is consistent across recent scans (not a one-time blip).
        Delegates to check_score_stability_from_history() in trading_engine.py.
        """
        history = self.score_history.get(ticker, [])
        return check_score_stability_from_history(history, current_score, threshold)

    def _evaluate_sells(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """Evaluate positions for sells using ai_trader logic"""
        sells = []

        # Update score history for all positions
        for ticker in self.positions:
            score_data = scores.get(ticker, {})
            current_score = score_data.get("total_score", 0)
            self._update_score_history(ticker, current_score)

        # Get market condition for market-aware stop losses
        market = self.data_provider.get_market_direction(current_date)
        spy_data = market.get('spy', {})
        is_bearish_market = spy_data.get('price', 0) < spy_data.get('ma_50', 0)

        # Get stop loss config — strategy profile overrides YAML defaults
        stop_loss_config = config.get('ai_trader.stops', {})
        use_atr_stops = stop_loss_config.get('use_atr_stops', True)
        atr_multiplier = stop_loss_config.get('atr_multiplier', 2.5)
        atr_period = stop_loss_config.get('atr_period', 14)
        max_stop_pct = stop_loss_config.get('max_stop_pct', 20.0)

        # Partial profit config is now handled by get_partial_profit_action() in trading_engine

        # Partial trailing stop config
        partial_trailing_config = config.get('ai_trader.trailing_stops', {})
        partial_on_trailing = partial_trailing_config.get('partial_on_trailing', True)
        partial_min_pyramid = partial_trailing_config.get('partial_min_pyramid_count', 2)
        partial_min_score = partial_trailing_config.get('partial_min_score', 65)
        partial_sell_pct = partial_trailing_config.get('partial_sell_pct', 50)

        # VIX-regime stop adjustment (proxy fetched here; math via shared helper)
        vix_config = config.get('vix_stops', {})
        vix_proxy = (
            self.data_provider.get_vix_proxy(current_date)
            if vix_config.get('enabled', True) else None
        )

        base_stop_loss_pct = select_effective_stop_loss_pct(
            profile=self.profile,
            stop_loss_config=stop_loss_config,
            default_normal_pct=self.backtest.stop_loss_pct,
            is_bearish_market=is_bearish_market,
            vix_proxy=vix_proxy,
            vix_config=vix_config,
        )

        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0 or price != price:  # Guard NaN (IEEE 754 self-inequality)
                continue

            # Guard against division by zero on cost_basis
            if position.cost_basis <= 0:
                continue

            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100
            score_data = scores.get(ticker, {})
            current_score = _nan_safe(score_data.get("total_score", 0))

            # ATR-based stop loss: volatile stocks get wider stops
            effective_stop_loss_pct = base_stop_loss_pct
            if use_atr_stops:
                atr_pct = self.data_provider.get_atr(ticker, current_date, period=atr_period)
                if atr_pct > 0:
                    atr_stop_pct = atr_multiplier * atr_pct
                    effective_stop_loss_pct = max(base_stop_loss_pct, atr_stop_pct)
                    effective_stop_loss_pct = min(effective_stop_loss_pct, max_stop_pct)  # Cap

            # F: NEW POSITION GUARD — tighter stop for new positions in first N days
            guard_config = config.get('ai_trader.new_position_guard', {})
            if guard_config.get('enabled', False):
                guard_days = guard_config.get('guard_days', 21)
                guard_stop_pct = guard_config.get('guard_stop_pct', 8.0)
                skip_if_pyramided = guard_config.get('skip_if_pyramided', True)
                holding_days = (current_date - position.purchase_date).days
                if holding_days <= guard_days:
                    if not (skip_if_pyramided and position.pyramid_count > 0):
                        # Use the tighter guard stop instead of the wider ATR-adjusted stop
                        effective_stop_loss_pct = min(effective_stop_loss_pct, guard_stop_pct)

            # Market-aware stop loss check
            if gain_pct <= -effective_stop_loss_pct:
                market_note = " (bearish market)" if is_bearish_market else ""
                atr_note = f" (ATR-adj {effective_stop_loss_pct:.1f}%)" if use_atr_stops and effective_stop_loss_pct != base_stop_loss_pct else ""
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"STOP LOSS: Down {abs(gain_pct):.1f}%{market_note}{atr_note}",
                    score=current_score,
                    priority=1
                )
                trade._signal_factors = {"sell_reason": "STOP LOSS", "gain_pct": round(gain_pct, 1), "stop_pct": round(effective_stop_loss_pct, 1)}
                sells.append(trade)
                continue

            # Trailing stop check
            if position.peak_price > 0:
                drop_from_peak = ((position.peak_price - price) / position.peak_price) * 100
                peak_gain_pct = ((position.peak_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0

                # Dynamic trailing stop thresholds — strategy profile overrides
                trailing_stop_pct = get_trailing_stop_pct(peak_gain_pct, self.profile)

                if trailing_stop_pct:
                    # Widen trailing stop for pyramided positions (high conviction)
                    effective_trailing_stop = apply_pyramid_widening(trailing_stop_pct, position.pyramid_count)
                    pyramid_widening = effective_trailing_stop - trailing_stop_pct

                    if drop_from_peak >= effective_trailing_stop:
                        # High conviction: partial sell + widen stop on remainder
                        # (shared helper — mirrored in ai_trader.evaluate_sells)
                        if should_take_partial_on_trailing_stop(
                            pyramid_count=position.pyramid_count,
                            score=current_score,
                            partial_on_trailing=partial_on_trailing,
                            partial_min_pyramid_count=partial_min_pyramid,
                            partial_min_score=partial_min_score,
                        ):
                            shares_to_sell = position.shares * (partial_sell_pct / 100)
                            trade = SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=shares_to_sell,
                                price=price,
                                reason=f"TRAILING STOP (PARTIAL {partial_sell_pct}%): Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%)",
                                score=current_score,
                                priority=2,
                                is_partial=True,
                                sell_pct=partial_sell_pct
                            )
                            trade._signal_factors = {"sell_reason": "PARTIAL TRAILING", "gain_pct": round(gain_pct, 1), "drop_from_peak": round(drop_from_peak, 1), "sell_pct": partial_sell_pct}
                            sells.append(trade)
                            # Reset peak to current price so remaining shares get a fresh wider stop
                            position.peak_price = price
                            position.peak_date = current_date
                        else:
                            # Standard: full sell
                            pyramid_note = f" (pyramid +{pyramid_widening:.0f}%)" if pyramid_widening > 0 else ""
                            trade = SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=position.shares,
                                price=price,
                                reason=f"TRAILING STOP: Peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%){pyramid_note}",
                                score=current_score,
                                priority=2
                            )
                            trade._signal_factors = {"sell_reason": "TRAILING STOP", "gain_pct": round(gain_pct, 1), "drop_from_peak": round(drop_from_peak, 1)}
                            sells.append(trade)
                        continue

            # PRE-EARNINGS TRAILING STOP TIGHTENING
            # Non-CS positions approaching earnings get tighter trailing stops to protect gains
            # CS (Coiled Spring) positions are earnings plays — they hold through intentionally
            earnings_tighten_config = config.get('ai_trader.earnings_tighten', {})
            if earnings_tighten_config.get('enabled', True):
                earnings_tighten_days = earnings_tighten_config.get('days_before', 5)
                earnings_tighten_factor = earnings_tighten_config.get('stop_tighten_factor', 0.50)
                earnings_min_gain_for_partial = earnings_tighten_config.get('min_gain_for_partial', 10)
                days_to_earn = self.static_data.get(ticker, {}).get('days_to_earnings')
                is_cs = getattr(position, 'is_coiled_spring', False)

                if (days_to_earn is not None and isinstance(days_to_earn, (int, float))
                        and 0 < days_to_earn <= earnings_tighten_days and not is_cs):
                    # Tighten trailing stop for non-CS positions near earnings
                    if position.peak_price > 0 and gain_pct > 0:
                        drop_from_peak = ((position.peak_price - price) / position.peak_price) * 100
                        peak_gain_pct = ((position.peak_price / position.cost_basis) - 1) * 100 if position.cost_basis > 0 else 0

                        # Use tighter trailing stop (50% of normal)
                        tightened_trailing = get_tightened_trailing_stop(peak_gain_pct, self.profile, tighten_factor=earnings_tighten_factor)

                        if tightened_trailing and drop_from_peak >= tightened_trailing:
                            trade = SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=position.shares,
                                price=price,
                                reason=f"PRE-EARNINGS STOP: {int(days_to_earn)}d to earnings, peak ${position.peak_price:.2f} -> ${price:.2f} (-{drop_from_peak:.1f}%, tightened to {tightened_trailing:.0f}%)",
                                score=current_score,
                                priority=2
                            )
                            trade._signal_factors = {"sell_reason": "PRE-EARNINGS STOP", "gain_pct": round(gain_pct, 1), "days_to_earnings": int(days_to_earn)}
                            sells.append(trade)
                            continue

                    # Partial profit protection for profitable positions near earnings
                    if gain_pct >= earnings_min_gain_for_partial and position.partial_profit_taken < 25:
                        take_pct = 25 - position.partial_profit_taken
                        if take_pct > 0:
                            shares_to_sell = position.shares * (take_pct / 100)
                            trade = SimulatedTrade(
                                ticker=ticker,
                                action="SELL",
                                shares=shares_to_sell,
                                price=price,
                                reason=f"PRE-EARNINGS PARTIAL: {int(days_to_earn)}d to earnings, up {gain_pct:.1f}% - protecting gains",
                                score=current_score,
                                priority=3,
                                is_partial=True,
                                sell_pct=take_pct
                            )
                            trade._signal_factors = {"sell_reason": "PRE-EARNINGS PARTIAL", "gain_pct": round(gain_pct, 1), "days_to_earnings": int(days_to_earn)}
                            sells.append(trade)
                            continue

            # Skip all score-dependent sells if score data is missing (matches ai_trader)
            score_available = current_score > 0
            if not score_available:
                logger.debug(f"{ticker}: Score=0 (data missing), skipping score-dependent sells")
                continue

            # Score crash check with stability verification and profitability exception
            # Check score stability first
            score_crash_config = config.get('ai_trader.score_crash', {})
            score_threshold = score_crash_config.get('threshold', 50)
            consecutive_required = score_crash_config.get('consecutive_required', 3)
            stability = self._check_score_stability(ticker, current_score, threshold=score_threshold)

            # Evaluate score crash using engine function
            crash_result = evaluate_score_crash(
                current_score=current_score,
                purchase_score=position.purchase_score,
                gain_pct=gain_pct,
                stability=stability,
                profile=self.profile,
                yaml_config=config,
            )

            if crash_result and crash_result.get("should_sell"):
                score_drop = position.purchase_score - current_score
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=crash_result["reason"],
                    score=current_score,
                    priority=3
                )
                trade._signal_factors = {"sell_reason": "SCORE CRASH", "gain_pct": round(gain_pct, 1), "score_drop": round(score_drop, 1)}
                sells.append(trade)
                continue

            # ── EXPERIMENT (gated, default OFF): score-floor exit ──────────────
            # Hypothesis: cut laggards whose CANSLIM score sits below an absolute
            # floor for N consecutive scans — even if modestly green — to free
            # capital for fresh >=min_score candidates. Distinct from SCORE CRASH
            # (which needs a big DROP from purchase AND ignores profitable names),
            # so it catches names that simply decay to mediocrity without a crash.
            # Winners up >= min_gain_to_exempt are exempt: trailing-stop / take-
            # profit manage them (avoids cutting future winners on a score dip —
            # the concentration mistake that sank the rotation experiment).
            # Enabled only via profile_overrides; absent => champion unchanged.
            # NOT mirrored to ai_trader.py: offline backtest research only until a
            # post-June-18 ship decision (keeps the live eval surface untouched).
            floor_cfg = self.profile.get('score_floor_exit', {})
            if (floor_cfg.get('enabled')
                    and gain_pct < floor_cfg.get('min_gain_to_exempt', 20)
                    and current_score < floor_cfg.get('floor', 60)):
                floor = floor_cfg.get('floor', 60)
                need = floor_cfg.get('consecutive', 3)
                stab = self._check_score_stability(ticker, current_score, threshold=floor)
                if stab.get('consecutive_low', 0) >= need:
                    trade = SimulatedTrade(
                        ticker=ticker,
                        action="SELL",
                        shares=position.shares,
                        price=price,
                        reason=f"SCORE FLOOR: {current_score:.0f} < {floor:.0f} for {stab.get('consecutive_low')} scans, gain {gain_pct:+.1f}%",
                        score=current_score,
                        priority=4,
                    )
                    trade._signal_factors = {"sell_reason": "SCORE FLOOR", "gain_pct": round(gain_pct, 1), "consecutive_low": stab.get('consecutive_low')}
                    sells.append(trade)
                    continue

            # PARTIAL PROFIT TAKING - let winners run while locking in gains
            partial_taken = position.partial_profit_taken
            partial_action = get_partial_profit_action(gain_pct, current_score, partial_taken)
            if partial_action:
                take_pct = partial_action["take_pct"]
                shares_to_sell = position.shares * (take_pct / 100)
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=shares_to_sell,
                    price=price,
                    reason=partial_action["reason"],
                    score=current_score,
                    priority=partial_action["priority"],
                    is_partial=True,
                    sell_pct=take_pct
                )
                trade._signal_factors = {"sell_reason": "PARTIAL PROFIT", "gain_pct": round(gain_pct, 1), "sell_pct": take_pct}
                sells.append(trade)
                continue

            # Winners (gain >= 20%): protect gains or take profit (matches ai_trader structure)
            profile_take_profit = self.profile.get('take_profit_pct', 75.0)
            if gain_pct >= 20:
                if current_score < self.backtest.sell_score_threshold:
                    # PROTECT GAINS: winner with weak score — sell to lock in gains
                    trade = SimulatedTrade(
                        ticker=ticker,
                        action="SELL",
                        shares=position.shares,
                        price=price,
                        reason=f"PROTECT GAINS: Up {gain_pct:.1f}% but score weak ({current_score:.0f})",
                        score=current_score,
                        priority=6
                    )
                    trade._signal_factors = {"sell_reason": "PROTECT GAINS", "gain_pct": round(gain_pct, 1)}
                    sells.append(trade)
                elif gain_pct >= profile_take_profit and current_score < position.purchase_score - 15:
                    # TAKE PROFIT: big winner with significantly declining score
                    trade = SimulatedTrade(
                        ticker=ticker,
                        action="SELL",
                        shares=position.shares,
                        price=price,
                        reason=f"TAKE PROFIT: Up {gain_pct:.1f}%, score declining significantly ({position.purchase_score:.0f} -> {current_score:.0f})",
                        score=current_score,
                        priority=7
                    )
                    trade._signal_factors = {"sell_reason": "TAKE PROFIT", "gain_pct": round(gain_pct, 1)}
                    sells.append(trade)

            # Weak flat/losing positions — cut and redeploy capital
            elif gain_pct < 10 and current_score < self.backtest.sell_score_threshold:
                trade = SimulatedTrade(
                    ticker=ticker,
                    action="SELL",
                    shares=position.shares,
                    price=price,
                    reason=f"WEAK POSITION: {gain_pct:+.1f}%, score {current_score:.0f}",
                    score=current_score,
                    priority=6
                )
                trade._signal_factors = {"sell_reason": "WEAK POSITION", "gain_pct": round(gain_pct, 1)}
                sells.append(trade)

        # Sort by priority
        sells.sort(key=lambda x: x.priority)
        return sells

    def _evaluate_buys(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """
        Evaluate stocks for buys with enhanced breakout and base pattern detection.

        Key improvements:
        - Pre-breakout bonus: +20 for stocks 5-15% below pivot with valid base
        - Breakout bonus: +25 for confirmed breakouts, +35 for strong volume breakouts
        - Extended penalty: -10 to -20 for stocks >5% above pivot (chasing)
        - Chase penalty: -15 for buying at highs without base pattern
        - Volume confirmation: Larger positions for breakouts with strong volume
        - Base pattern bonus: Extra points for stocks with proper consolidation
        """
        # Reset ML prediction cache for this trading day
        clear_prediction_cache()
        buys = []
        current_tickers = set(self.positions.keys())

        # Apply market regime score adjustment (match ai_trader logic)
        market = self.data_provider.get_market_direction(current_date)
        weighted_signal = market.get("weighted_signal", 0)

        regime_config = config.get('ai_trader.market_regime', {})
        bullish_threshold = regime_config.get('bullish_threshold', 1.5)
        bearish_threshold = regime_config.get('bearish_threshold', -0.5)
        bearish_adj = regime_config.get('bearish_min_score_adj', 10)
        bear_exception_min_cal = regime_config.get('bear_exception_min_cal', 35)
        bear_exception_position_mult = regime_config.get('bear_exception_position_mult', 0.50)

        # Regime-based position sizing (matches live trader get_market_regime())
        # Profile-level overrides take precedence over global config
        profile_regime = self.profile.get('regime_position_caps', {})
        bullish_max_pct = profile_regime.get('bullish', regime_config.get('bullish_max_position_pct', 15.0))
        bearish_max_pct = profile_regime.get('bearish', regime_config.get('bearish_max_position_pct', 8.0))
        neutral_max_pct = profile_regime.get('neutral', regime_config.get('neutral_max_position_pct', 12.0))

        # Use strategy profile min_score (falls back to backtest record value)
        profile_min_score = self.profile.get('min_score', self.backtest.min_score_to_buy)

        # Score floor decay: lower min_score when under-invested in a bull market
        decay_config = self.profile.get('score_floor_decay', {})
        if decay_config.get('enabled', False) and self.underinvested_days > 0:
            steps = decay_config.get('steps', [
                {'after_days': 15, 'min_score': 68},
                {'after_days': 30, 'min_score': 65},
                {'after_days': 45, 'min_score': 62},
            ])
            # Apply the most aggressive step that has been reached
            for step in sorted(steps, key=lambda s: s['after_days'], reverse=True):
                if self.underinvested_days >= step['after_days']:
                    old_score = profile_min_score
                    profile_min_score = step['min_score']
                    logger.info(f"SCORE DECAY: {self.underinvested_days} under-invested days, "
                                f"min_score {old_score} -> {profile_min_score}")
                    break

        if weighted_signal >= bullish_threshold:
            effective_min_score = profile_min_score - 5  # Easier in bull
            regime_max_pct = bullish_max_pct
        elif weighted_signal <= bearish_threshold:
            effective_min_score = profile_min_score + bearish_adj  # Harder in bear
            regime_max_pct = bearish_max_pct
        else:
            effective_min_score = profile_min_score
            regime_max_pct = neutral_max_pct

        # Detect if earnings data is limited (historical backtests lose C/A scores)
        # When _filter_available_earnings strips too many quarters, C=0 and A=0 for
        # all stocks. Without these 30 points, the normal threshold (72) is unreachable.
        #
        # NOTE: We require BOTH C AND A to be limited before bypassing quality filters.
        # When only C is limited (mid-2021 in backtester), A data is still available,
        # and buying stocks without verified quarterly earnings is harmful — it was
        # tested and produced -7.8% vs +63.9% baseline. The idle cash during C-limited
        # periods is a backtester data artifact (live trading always has FMP/Yahoo C data).
        max_c_in_universe = max((data.get('c_score', 0) for _, data in scores.items()), default=0)
        max_a_in_universe = max((data.get('a_score', 0) for _, data in scores.items()), default=0)
        earnings_data_limited = (max_c_in_universe < 5 and max_a_in_universe < 5)

        if earnings_data_limited:
            # C+A worth up to 30 points — reduce threshold to match available range
            # This mirrors seeding logic which uses threshold 35 instead of 72
            effective_min_score = max(effective_min_score - 30, 35)
            logger.debug(f"Earnings data limited (max C={max_c_in_universe:.0f}, A={max_a_in_universe:.0f}), "
                         f"adjusted min_score to {effective_min_score:.0f}")

        # PERCENTILE-BASED THRESHOLD: Adapt to score distribution
        # Use the lower of (regime-adjusted threshold, top 5% percentile)
        # This ensures we always have candidates even when scores are compressed
        percentile_pct = config.get('ai_trader.allocation.percentile_threshold_pct', 5)
        score_floor = config.get('ai_trader.allocation.percentile_score_floor', 45)

        all_scores_list = sorted([data["total_score"] for _, data in scores.items()
                                  if data["total_score"] > 0], reverse=True)
        if all_scores_list:
            top_idx = max(1, len(all_scores_list) * percentile_pct // 100)
            percentile_threshold = all_scores_list[min(top_idx - 1, len(all_scores_list) - 1)]
            # Use the lower of regime-adjusted and percentile, floored at score_floor
            effective_min_score = max(score_floor, min(effective_min_score, percentile_threshold))

        # SOFT THRESHOLD: Allow stocks slightly below min_score with reduced position sizes
        soft_config = config.get('ai_trader.allocation.soft_threshold', {})
        # Profile-level soft zone overrides
        profile_soft = self.profile.get('soft_threshold', {})
        soft_enabled = profile_soft.get('enabled', soft_config.get('enabled', False))
        soft_zone_width = profile_soft.get('zone_width', soft_config.get('zone_width', 4))
        soft_mult_edge = profile_soft.get('multiplier_at_edge', soft_config.get('multiplier_at_edge', 0.25))
        soft_mult_top = profile_soft.get('multiplier_at_top', soft_config.get('multiplier_at_top', 0.75))

        # SCORE STABILITY: Override for strong deterministic scores
        stability_config = config.get('ai_trader.score_stability', {})
        stability_enabled = stability_config.get('enabled', False)
        stable_min_for_override = stability_config.get('stable_min_for_override', 55)

        # H: Wider soft zone with deterministic gate
        require_strong_det = soft_config.get('require_strong_deterministic', False)
        det_min_for_soft = soft_config.get('deterministic_min', 50)

        # Diagnostic: log score distribution when decay is active (helps debug empty candidate pools)
        if self.underinvested_days >= 15:
            above_threshold = sum(1 for _, d in scores.items() if d["total_score"] >= effective_min_score and _ not in current_tickers)
            top5 = sorted([(t, d["total_score"], d.get("c_score", 0), d.get("l_score", 0))
                           for t, d in scores.items() if t not in current_tickers],
                          key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"DECAY DIAG: {self.underinvested_days}d idle, eff_min={effective_min_score:.0f}, "
                        f"{above_threshold} candidates above threshold, "
                        f"top5: {[(t, f's={s:.0f} c={c:.0f} l={l:.0f}') for t, s, c, l in top5]}")

        # Get candidates: full-position (above threshold) + soft-zone (graduated)
        soft_zone_floor = effective_min_score - soft_zone_width if soft_enabled else effective_min_score
        candidates = []
        for ticker, data in scores.items():
            if ticker in current_tickers:
                continue
            total_score = data["total_score"]
            if total_score >= effective_min_score:
                # Full position candidate
                data["_soft_zone_multiplier"] = 1.0
                candidates.append((ticker, data))
            elif soft_enabled and total_score >= soft_zone_floor:
                # H: If require_strong_deterministic, only allow soft zone for stocks
                # with strong N+S+L+I+M scores (these are Yahoo-based, deterministic)
                det_score = (_nan_safe(data.get("n_score", 0)) + _nan_safe(data.get("s_score", 0)) +
                             _nan_safe(data.get("l_score", 0)) + _nan_safe(data.get("i_score", 0)) +
                             _nan_safe(data.get("m_score", 0)))
                if require_strong_det and det_score < det_min_for_soft:
                    continue  # Skip weak deterministic stocks in soft zone

                # Soft zone candidate — graduated position size
                # Linear interpolation: edge of zone → multiplier_at_edge, top of zone → multiplier_at_top
                zone_position = (total_score - soft_zone_floor) / max(1, soft_zone_width)
                soft_mult = soft_mult_edge + zone_position * (soft_mult_top - soft_mult_edge)

                # STABLE OVERRIDE: If deterministic scores are very strong, upgrade to top multiplier
                if stability_enabled:
                    if det_score >= stable_min_for_override:
                        soft_mult = soft_mult_top
                        data["_stable_override"] = True
                        data["_stable_score"] = det_score

                data["_soft_zone_multiplier"] = round(soft_mult, 2)
                data["_in_soft_zone"] = True
                candidates.append((ticker, data))

        # BEAR MARKET EXCEPTION: Allow very strong stocks at reduced position size
        # If C+A+L >= 35 (out of 45 max), the stock has excellent fundamentals
        # regardless of market conditions
        is_bearish = weighted_signal <= bearish_threshold
        if not candidates and is_bearish:
            bear_candidates = []
            for ticker, data in scores.items():
                if ticker in current_tickers:
                    continue
                if data["total_score"] < self.backtest.min_score_to_buy:
                    continue
                cal_score = (data.get("c_score", 0) + data.get("a_score", 0) + data.get("l_score", 0))
                if cal_score >= bear_exception_min_cal:
                    data["_bear_market_entry"] = True
                    bear_candidates.append((ticker, data))
            candidates = bear_candidates

        # CORRECTION ZONE FILTER: When SPY between 200MA and 50MA,
        # apply tighter quality filters to candidates
        cz_config = self.profile.get('correction_zone', {})
        if self.correction_zone_active and cz_config.get('enabled', False):
            cz_min_cal = cz_config.get('min_cal_score', 38)
            cz_require_rs = cz_config.get('require_relative_strength', False)
            cz_require_base = cz_config.get('require_base', False)
            cz_min_base_weeks = cz_config.get('min_base_weeks', 0)
            cz_cs_only = cz_config.get('cs_only', False)
            cz_min_cs_confidence = cz_config.get('min_cs_confidence', 60)
            cz_position_mult = cz_config.get('position_mult', 0.50)

            filtered = []
            for ticker, data in candidates:
                # CS-only mode: defer filtering to per-candidate loop where CS is computed.
                # Mark all candidates as needing CS validation later.
                if cz_cs_only:
                    data["_correction_zone_entry"] = True
                    data["_correction_zone_mult"] = cz_position_mult
                    data["_cz_cs_only_required"] = True
                    data["_cz_min_cs_confidence"] = cz_min_cs_confidence
                    filtered.append((ticker, data))
                    continue

                # C+A+L quality filter
                cal = (_nan_safe(data.get("c_score", 0)) +
                       _nan_safe(data.get("a_score", 0)) +
                       _nan_safe(data.get("l_score", 0)))
                if cal < cz_min_cal:
                    self.overlay_stats['cz_pre_filter_rejected'] += 1
                    continue

                # Base requirement
                if cz_require_base:
                    base = data.get("base_type", "none")
                    weeks = data.get("weeks_in_base", 0)
                    if base in ("none", None) or weeks < cz_min_base_weeks:
                        self.overlay_stats['cz_pre_filter_rejected'] += 1
                        continue

                # Relative strength: stock price above its own 21MA
                if cz_require_rs:
                    stock_ma21 = self.data_provider.get_moving_average(ticker, current_date, 21)
                    stock_price = self.data_provider.get_price_on_date(ticker, current_date)
                    if stock_ma21 > 0 and stock_price and stock_price < stock_ma21:
                        self.overlay_stats['cz_pre_filter_rejected'] += 1
                        continue

                data["_correction_zone_entry"] = True
                data["_correction_zone_mult"] = cz_position_mult
                filtered.append((ticker, data))
                self.overlay_stats['cz_pass'] += 1
                logger.debug(f"CZ pass: {ticker} score={data['total_score']:.0f} C+A+L={cal:.0f}")

            candidates = filtered

        portfolio_value = self._get_portfolio_value(current_date)

        # Funnel diagnostic counters (only active during decay idle periods)
        _funnel_diag = self.underinvested_days >= 15
        _funnel = {"candidates": len(candidates), "no_price": 0, "cooldown": 0, "sector": 0,
                   "c_filter": 0, "l_filter": 0, "volume": 0, "earnings_prox": 0, "passed": 0}

        for ticker, score_data in candidates:
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0 or price != price:  # Guard NaN
                _funnel["no_price"] += 1
                continue

            score = score_data["total_score"]
            pct_from_high = score_data.get("pct_from_high", 100)
            pct_from_pivot = score_data.get("pct_from_pivot", pct_from_high)
            has_base = score_data.get("has_base_pattern", False)
            pivot_price = score_data.get("pivot_price", 0)
            weeks_in_base = score_data.get("weeks_in_base", 0)

            # DEAD STOCK GUARD: Skip stocks with no relative strength data
            # Catches acquisition targets (pinned price), halted stocks, tickers with stale data
            if score_data.get("rs_12m") is None and score_data.get("rs_3m") is None:
                _funnel["dead_stock"] = _funnel.get("dead_stock", 0) + 1
                continue

            # Re-entry cooldown: don't re-buy recently stopped-out stocks
            cooldown_config = config.get('ai_trader.re_entry_cooldown', {})
            # Profile-level cooldown overrides
            profile_cooldown = self.profile.get('re_entry_cooldown', {})
            stop_loss_cooldown = profile_cooldown.get('stop_loss_days', cooldown_config.get('stop_loss_days', 5))
            trailing_stop_cooldown = profile_cooldown.get('trailing_stop_days', cooldown_config.get('trailing_stop_days', 3))

            if ticker in self.recently_sold:
                sold_date, sold_reason = self.recently_sold[ticker]
                days_since = (current_date - sold_date).days
                if "STOP LOSS" in sold_reason:
                    cooldown = stop_loss_cooldown
                elif "TRAILING STOP" in sold_reason and "PARTIAL" not in sold_reason:
                    cooldown = trailing_stop_cooldown
                else:
                    cooldown = 0  # No cooldown for score crash, protect gains, etc.
                if cooldown > 0 and days_since < cooldown:
                    _funnel["cooldown"] += 1
                    continue

            # Check sector limits
            sector = self.static_data.get(ticker, {}).get("sector", "Unknown")
            if not self._check_sector_limit(sector):
                _funnel["sector"] += 1
                continue

            # QUALITY FILTERS: Only buy stocks with strong fundamentals
            # Strategy profile overrides YAML defaults
            quality_config = config.get('ai_trader.quality_filters', {})
            profile_quality = self.profile.get('quality_filters', {})
            min_c_score = profile_quality.get('min_c_score', quality_config.get('min_c_score', 10))
            min_l_score = profile_quality.get('min_l_score', quality_config.get('min_l_score', 8))
            min_volume_ratio = quality_config.get('min_volume_ratio', 1.2)
            skip_growth = quality_config.get('skip_in_growth_mode', True)

            # Get individual scores from score_data
            c_score = _nan_safe(score_data.get('c', 0) or score_data.get('c_score', 0))
            l_score = _nan_safe(score_data.get('l', 0) or score_data.get('l_score', 0))
            volume_ratio = _nan_safe(score_data.get('volume_ratio', 1.0) or 1.0, default=1.0)
            is_growth_stock = score_data.get('is_growth_stock', False)

            # Skip if not meeting quality thresholds (unless growth stock or limited earnings)
            if not (is_growth_stock and skip_growth) and not earnings_data_limited:
                if c_score < min_c_score:
                    logger.debug(f"Skipping {ticker}: C score {c_score} < {min_c_score}")
                    _funnel["c_filter"] += 1
                    continue
                if l_score < min_l_score:
                    logger.debug(f"Skipping {ticker}: L score {l_score} < {min_l_score}")
                    _funnel["l_filter"] += 1
                    continue
            elif is_growth_stock and skip_growth:
                # Growth stocks bypass normal thresholds but must have SOME fundamental data
                # C=0 AND A=0 means no earnings data at all (e.g., acquired/dead stock)
                a_score = _nan_safe(score_data.get('a', 0) or score_data.get('a_score', 0))
                if c_score == 0 and a_score == 0:
                    logger.debug(f"Skipping growth {ticker}: C=0 and A=0 (no earnings data)")
                    _funnel["c_filter"] += 1
                    continue
            elif earnings_data_limited:
                if l_score < min_l_score:
                    _funnel["l_filter"] += 1
                    continue

            # VOLUME GATE: Only buy when volume confirms interest
            vol_gate_config = config.get('volume_gate', {})
            if vol_gate_config.get('enabled', True):
                if score_data.get('is_breaking_out', False):
                    vol_threshold = vol_gate_config.get('breakout_min_volume_ratio', 1.5)
                elif has_base and pct_from_pivot is not None and 0 <= pct_from_pivot <= 15:
                    vol_threshold = vol_gate_config.get('pre_breakout_min_volume_ratio', 0.8)
                else:
                    vol_threshold = vol_gate_config.get('min_volume_ratio', 1.0)
                if volume_ratio < vol_threshold:
                    logger.debug(f"Skipping {ticker}: Volume ratio {volume_ratio:.2f} < {vol_threshold} (volume gate)")
                    _funnel["volume"] += 1
                    continue
            elif volume_ratio < min_volume_ratio and not score_data.get('is_breaking_out', False):
                logger.debug(f"Skipping {ticker}: Volume ratio {volume_ratio:.2f} < {min_volume_ratio}")
                _funnel["volume"] += 1
                continue

            # Earnings proximity check with Coiled Spring exception
            # Synced with ai_trader.py evaluate_buys() earnings logic
            static_data = self.static_data.get(ticker, {})
            days_to_earnings = static_data.get('days_to_earnings')
            cs_config = config.get('coiled_spring', {})
            earnings_config = config.get('ai_trader.earnings', {})
            allow_buy_days = cs_config.get('earnings_window', {}).get('allow_buy_days', 7)
            block_days = cs_config.get('earnings_window', {}).get('block_days', 1)
            # Profile-level override for non-CS earnings avoidance window. Lets us
            # A/B test alternate gate widths (e.g. nostate_optimized_avoid3) without
            # touching the live YAML default that the AI trader also reads.
            avoidance_days = self.profile.get('earnings_avoidance_days',
                                              earnings_config.get('avoidance_days', 5))
            cs_override_enabled = earnings_config.get('cs_override', True)

            # Initialize CS result
            cs_result = None
            coiled_spring_bonus = 0

            if days_to_earnings is not None and 0 < days_to_earnings <= max(allow_buy_days, avoidance_days):
                # Check for Coiled Spring qualification using static_data
                cs_result = self._calculate_coiled_spring_for_backtest(ticker, score_data, static_data, cs_config)

                if cs_result["is_coiled_spring"] and days_to_earnings > block_days and cs_override_enabled:
                    # Check confidence gating — skip LOW confidence CS setups
                    cs_confidence = cs_result.get('confidence', 50)
                    min_cs_confidence = cs_config.get('min_confidence', 30)
                    if cs_confidence < min_cs_confidence:
                        logger.debug(
                            f"Backtest CS SKIPPED (low confidence): {ticker} "
                            f"confidence={cs_confidence} < {min_cs_confidence}"
                        )
                        # Treat as non-CS — apply normal avoidance window
                        if days_to_earnings <= avoidance_days:
                            _funnel["earnings_prox"] += 1
                            continue
                    else:
                        # ALLOW - high conviction earnings catalyst
                        coiled_spring_bonus = cs_result.get('cs_score', 0)
                        logger.debug(f"Backtest CS: {ticker} confidence={cs_confidence} ({cs_result['cs_details']})")
                else:
                    # NON-CS: Block stocks within avoidance window only
                    if days_to_earnings <= avoidance_days:
                        _funnel["earnings_prox"] += 1
                        continue

            # Correction zone CS-only gate: reject non-CS candidates
            if score_data.get("_cz_cs_only_required"):
                if coiled_spring_bonus <= 0:
                    self.overlay_stats['cz_cs_only_rejected'] += 1
                    continue  # Not a CS candidate — skip in cs_only mode
                if cs_result and cs_result.get("confidence", 0) < score_data.get("_cz_min_cs_confidence", 60):
                    self.overlay_stats['cz_cs_confidence_rejected'] += 1
                    continue  # CS confidence too low
                # Mirrors ai_trader.evaluate_buys: count the cs_only successes too.
                self.overlay_stats['cz_pass'] += 1

            # Get breakout status and volume ratio from cached scores
            is_breaking_out = score_data.get("is_breaking_out", False)
            volume_ratio = score_data.get("volume_ratio", 1.0)
            base_pattern = score_data.get("base_pattern", {"type": "none"})

            # Base pattern quality bonus (up to 15 points)
            base_type = base_pattern.get("type", "none")
            base_quality_bonus = calculate_base_quality_bonus(base_type, weeks_in_base)

            # Reconstruct week_52_high from pct_from_high for engine function
            # pct_from_high = ((high_52w - price) / high_52w) * 100
            # So: high_52w = price / (1 - pct_from_high/100)
            if pct_from_high >= 100:
                week_52_high = price  # Edge case: pct_from_high >= 100 means price ~ 0
            else:
                week_52_high = price / (1 - pct_from_high / 100) if pct_from_high < 100 else price

            # Calculate entry signals (momentum, breakout, pre-breakout, extended penalties)
            entry_signals = calculate_entry_signals(
                current_price=price,
                week_52_high=week_52_high,
                pivot_price=pivot_price,
                base_type=base_type,
                weeks_in_base=weeks_in_base,
                is_breaking_out=is_breaking_out,
                breakout_volume_ratio=volume_ratio,  # backtester uses volume_ratio for both
                volume_ratio=volume_ratio,
                effective_score=score,  # total_score serves as effective_score
            )
            momentum_score = entry_signals["momentum_score"]
            breakout_bonus = entry_signals["breakout_bonus"]
            pre_breakout_bonus = entry_signals["pre_breakout_bonus"]
            extended_penalty = entry_signals["extended_penalty"]

            # RS LINE NEW HIGH: Leading indicator when RS makes new high before price
            rs_line_bonus = 0
            rs_line_config = config.get('rs_line', {})
            if rs_line_config.get('enabled', True):
                rs_line_data = self.data_provider.get_rs_line_new_high(ticker, current_date)
                if rs_line_data.get("rs_leading"):
                    rs_line_bonus = rs_line_config.get('bonus_points', 8)
                elif rs_line_data.get("rs_lagging"):
                    rs_line_bonus = rs_line_config.get('penalty_divergence', -5)

            # EARNINGS SURPRISE DRIFT: Post-earnings momentum for big beats
            earnings_drift_bonus = 0
            drift_config = config.get('earnings_drift', {})
            if drift_config.get('enabled', True):
                static_data = self.static_data.get(ticker, {})
                days_to_earnings = static_data.get('days_to_earnings')
                beat_streak = static_data.get('earnings_beat_streak', 0)
                # If stock has recent strong earnings beat (proxy: high beat streak + near earnings)
                if beat_streak >= 3 and days_to_earnings is not None:
                    # Post-earnings drift window: within 60 days after earnings
                    # Proxy: if days_to_earnings < 0 (past), we're in drift window
                    # Since we don't have negative days_to_earnings, use beat_streak as signal
                    if beat_streak >= 4:
                        earnings_drift_bonus = drift_config.get('bonus_points', 5)
                    elif beat_streak >= 3:
                        earnings_drift_bonus = 3

            # ANALYST REVISION BONUS: Reward stocks where analysts are raising estimates
            estimate_revision_bonus = 0
            revision_pct = self.static_data.get(ticker, {}).get('eps_estimate_revision_pct')
            revision_config = config.get('ai_trader.analyst_revisions', {})
            if revision_pct is not None and revision_pct == revision_pct:  # Guard NaN
                strong_up_threshold = revision_config.get('strong_up_threshold', 10)
                strong_up_bonus = revision_config.get('strong_up_bonus', 5)
                mod_up_bonus = revision_config.get('mod_up_bonus', 3)
                strong_down_penalty = revision_config.get('strong_down_penalty', -5)
                mod_down_penalty = revision_config.get('mod_down_penalty', -2)

                if revision_pct >= strong_up_threshold:
                    estimate_revision_bonus = strong_up_bonus
                elif revision_pct >= 5:
                    estimate_revision_bonus = mod_up_bonus
                elif revision_pct <= -10:
                    estimate_revision_bonus = strong_down_penalty
                elif revision_pct <= -5:
                    estimate_revision_bonus = mod_down_penalty

            # MOMENTUM CONFIRMATION: Penalize stocks where recent momentum is fading
            # If 3-month RS is significantly weaker than 12-month RS, momentum is weakening
            rs_12m = score_data.get("rs_12m", 1.0)
            rs_3m = score_data.get("rs_3m", 1.0)
            momentum_penalty = 0
            if rs_12m > 0 and rs_3m < rs_12m * 0.95:
                # Recent momentum fading - apply 15% penalty to composite
                momentum_penalty = -0.15

            # Growth mode: adjust CANSLIM score for growth strategy
            # Add extra weight to C/L/N scores within the score component
            effective_score = score
            if self.strategy == "growth":
                c_weight = self.profile.get('c_score_weight', 1.0)
                l_weight = self.profile.get('l_score_weight', 1.0)
                n_weight = self.profile.get('n_score_weight', 1.0)
                c_sc = _nan_safe(score_data.get('c_score', 0))
                l_sc = _nan_safe(score_data.get('l_score', 0))
                n_sc = _nan_safe(score_data.get('n_score', 0))
                # Add the weighted bonus on top of total score
                effective_score += c_sc * (c_weight - 1.0) + l_sc * (l_weight - 1.0) + n_sc * (n_weight - 1.0)

            # Guard against NaN: float('nan') passes `or` check and poisons composite_score
            _pg = score_data.get("projected_growth", effective_score * 0.3)
            if _pg is None or (_pg != _pg):  # NaN != NaN is True
                _pg = effective_score * 0.3
            growth_projection = min(_pg, 50)

            # Composite score using trading engine function
            composite_score = calculate_composite_score(
                growth_projection=growth_projection,
                effective_score=effective_score,
                momentum_score=momentum_score,
                breakout_bonus=breakout_bonus,
                pre_breakout_bonus=pre_breakout_bonus,
                base_quality_bonus=base_quality_bonus,
                extended_penalty=extended_penalty,
                coiled_spring_bonus=coiled_spring_bonus,
                rs_line_bonus=rs_line_bonus,
                earnings_drift_bonus=earnings_drift_bonus,
                estimate_revision_bonus=estimate_revision_bonus,
                profile=self.profile,
            )

            # Apply momentum penalty after base composite calculation
            if momentum_penalty < 0:
                composite_score *= (1 + momentum_penalty)  # Reduce by 15%

            # Market state penalty: non-TRENDING states get a mild score reduction
            # This makes the system prefer buying in confirmed uptrends
            if self.market_state_enabled and self.market_state.state in (
                    MarketState.RECOVERY, MarketState.CONFIRMED):
                composite_score -= 5  # Mild penalty, not blocking
            elif self.market_state_enabled and self.market_state.state == MarketState.PRESSURE:
                composite_score -= 8  # Moderate penalty
            elif self.ftd_penalty_active:
                ftd_penalty_val = config.get('market_timing.follow_through_day.penalty', 8)
                composite_score -= ftd_penalty_val

            # Heat penalty: too much risk exposure → mild score reduction
            if self.heat_penalty_active:
                composite_score -= 5

            # DETERMINISTIC BOOST: Bonus for stocks with strong non-FMP scores (N+S+L+I+M)
            # This raises their ranking so they get bought first and with larger positions
            det_config = config.get('ai_trader.deterministic_boost', {})
            deterministic_boost_val = 0
            if det_config.get('enabled', False):
                stable_score = (_nan_safe(score_data.get("n_score", 0)) + _nan_safe(score_data.get("s_score", 0)) +
                                _nan_safe(score_data.get("l_score", 0)) + _nan_safe(score_data.get("i_score", 0)) +
                                _nan_safe(score_data.get("m_score", 0)))
                strong_thresh = det_config.get('strong_threshold', 60)
                stable_thresh = det_config.get('stable_bonus_threshold', 55)
                if stable_score >= strong_thresh:
                    deterministic_boost_val = det_config.get('strong_bonus', 8)
                elif stable_score >= stable_thresh:
                    deterministic_boost_val = det_config.get('stable_bonus', 5)
                composite_score += deterministic_boost_val

            # Volume dry-up bonus: volume contracting in a base = stored energy before breakout
            volume_dry_up_bonus = 0
            volume_dry_up = score_data.get("_volume_dry_up", False)
            if volume_dry_up and has_base and not is_breaking_out:
                volume_dry_up_bonus = 5
                composite_score += volume_dry_up_bonus

            # Bear-base readiness bonus (mirrors ai_trader.evaluate_buys, commit b972da3).
            # Boosts buy ranking for stocks that built quality bases during a bear so they
            # rank higher when the SPY gate flips bullish.
            bear_base_bonus, bb_readiness, bb_days, bear_base_reason = (
                self._bear_base_bonus_for_candidate(
                    ticker, current_date, score_data, static_data, price
                )
            )
            if bear_base_bonus > 0:
                composite_score += bear_base_bonus
                self.overlay_stats['bear_base_bonus_applied'] += 1
                self.overlay_stats['bear_base_bonus_total'] += bear_base_bonus

            # Composite score used for ranking/priority only — CANSLIM score is the quality gate

            # Position sizing: conviction-based (higher scores get larger positions)
            max_positions = self.profile.get('max_positions', self.backtest.max_positions)
            conv_config = config.get('ai_trader.conviction_sizing', {})
            # G: Skip conviction sizing on seed days — equal weight for initial entries
            skip_conv_on_seeds = conv_config.get('skip_initial_seeds', True)
            use_conviction = conv_config.get('enabled', False) and not (self.is_seed_day and skip_conv_on_seeds)

            # Calculate base position size using engine function
            # Note: Seed day skip for conviction sizing is handled by compute a modified profile
            if not use_conviction and self.is_seed_day and skip_conv_on_seeds:
                # For seed days with skip enabled, force fallback positioning by using modified profile
                sizing_profile = dict(self.profile)
                sizing_profile['conviction_sizing'] = {'enabled': False}
            else:
                sizing_profile = self.profile

            position_pct = calculate_position_size_pct(
                composite_score=composite_score,
                max_positions=max_positions,
                profile=sizing_profile,
                regime_max_pct=regime_max_pct,
                yaml_config=config,
            )

            # POSITION SIZE MULTIPLIERS
            # Note: Backtester has bear market exception positioning and correlation-aware sizing
            # that are not in the trading_engine.apply_position_size_multipliers() function.
            # Therefore, these multipliers are applied inline rather than using the engine function.
            # This keeps backtester logic intact while ai_trader.py uses the engine function.

            # Half-size positions when portfolio heat is elevated
            if self.heat_penalty_active:
                position_pct *= 0.50

            # Market state position sizing: smaller positions during recovery/pressure
            if self.market_state_enabled:
                position_pct *= self.market_state.position_size_multiplier

            # PREDICTIVE POSITION SIZING: Pre-breakout stocks get largest positions
            # These are the ideal entries - before the crowd notices
            pre_breakout_mult = self.profile.get('pre_breakout_multiplier', 1.40)
            if pre_breakout_bonus >= 35 and has_base:
                position_pct *= pre_breakout_mult  # Larger for best pre-breakout entries
            elif pre_breakout_bonus >= 25 and has_base:
                position_pct *= (pre_breakout_mult * 0.93)  # Slightly less for good pre-breakout
            elif is_breaking_out and volume_ratio >= 1.5:
                position_pct *= 1.0   # No boost - already extended, entry is late

            # Coiled Spring position boost
            if coiled_spring_bonus > 0:
                cs_multiplier = cs_config.get('position_multiplier', 1.25)
                position_pct *= cs_multiplier

            # Reduce position size for bear market exception entries
            if score_data.get("_bear_market_entry"):
                position_pct *= bear_exception_position_mult

            # Reduce position size for correction zone entries
            if score_data.get("_correction_zone_entry"):
                position_pct *= score_data.get("_correction_zone_mult", 0.50)
                self.overlay_stats['cz_position_mult_applied'] += 1

            # CORRELATION-AWARE SIZING: Reduce position if highly correlated with existing holdings
            corr_config = config.get('correlation_sizing', {})
            if corr_config.get('enabled', False) and self.positions:
                corr_threshold = corr_config.get('high_correlation_threshold', 0.70)
                corr_multiplier = corr_config.get('high_correlation_multiplier', 0.75)
                max_correlated = corr_config.get('max_correlated_positions', 3)
                lookback = corr_config.get('lookback_days', 30)

                high_corr_count = 0
                for held_ticker in list(self.positions.keys())[:5]:  # Check top 5 for performance
                    corr = self.data_provider.get_stock_correlation(
                        ticker, held_ticker, current_date, lookback
                    )
                    if corr > corr_threshold:
                        high_corr_count += 1

                if high_corr_count >= max_correlated:
                    logger.debug(f"{ticker}: {high_corr_count} correlated positions, skipping")
                    continue
                elif high_corr_count > 0:
                    position_pct *= corr_multiplier

            # SOFT ZONE: Apply graduated position multiplier for stocks below hard threshold
            soft_zone_mult = score_data.get("_soft_zone_multiplier", 1.0)
            if soft_zone_mult < 1.0:
                position_pct *= soft_zone_mult

            # Cap at profile max or market regime max AFTER all multipliers (matches live trader)
            profile_max_pct = self.profile.get('max_single_position_pct', 25)
            position_pct = min(position_pct, regime_max_pct, profile_max_pct)

            position_value = portfolio_value * (position_pct / 100)

            # Budget cash evenly across remaining position slots (matches live trader)
            remaining_slots = max(1, max_positions - len(self.positions))
            available_cash = self.cash * 0.90  # Keep 10% liquid buffer
            per_slot_budget = available_cash / remaining_slots
            # Allow high-conviction entries up to 1.3x the per-slot budget
            if pre_breakout_bonus >= 35 or is_breaking_out:
                cash_limit = per_slot_budget * 1.3
            else:
                cash_limit = per_slot_budget
            cash_limit = max(cash_limit, 500)  # Floor
            position_value = min(position_value, cash_limit)

            if position_value < 100:
                continue

            # Check sector allocation % limit (shared helper — mirrored in ai_trader.check_sector_limit)
            if portfolio_value > 0:
                sector = self.static_data.get(ticker, {}).get("sector", "Unknown")
                sector_value = sum(
                    p.shares * (self.data_provider.get_price_on_date(p.ticker, current_date) or p.cost_basis)
                    for p in self.positions.values() if p.sector == sector
                )
                adjusted_value = apply_sector_allocation_cap(
                    requested_position_value=position_value,
                    current_sector_value=sector_value,
                    portfolio_value=portfolio_value,
                    max_sector_allocation=MAX_SECTOR_ALLOCATION,
                )
                if adjusted_value == 0.0:
                    _funnel["sector"] += 1
                    continue
                position_value = adjusted_value

            # CORRELATION GUARD: Check price correlation with held positions
            corr_config = config.get('ai_trader.correlation_guard', {})
            if corr_config.get('enabled', True) and self.positions:
                corr_threshold = corr_config.get('threshold', 0.70)
                corr_reduction = corr_config.get('reduction', 0.50)
                corr_lookback = corr_config.get('lookback_days', 30)
                max_corr, most_corr_ticker = self._check_correlation(
                    ticker, current_date, corr_lookback
                )
                if max_corr >= corr_threshold:
                    excess = (max_corr - corr_threshold) / (1.0 - corr_threshold)
                    multiplier = max(0.25, corr_reduction * (1.0 - excess * 0.5))
                    old_val = position_value
                    position_value *= multiplier
                    if position_value < 100:
                        continue
                    logger.debug(f"CORR GUARD: {ticker} corr={max_corr:.2f} with "
                                 f"{most_corr_ticker}, ${old_val:.0f} -> ${position_value:.0f}")

            shares = position_value / price

            # Build reason string
            reason_parts = []
            # Coiled Spring indicator (highest priority)
            if coiled_spring_bonus > 0:
                days_to_earn = static_data.get('days_to_earnings', 0) or 0
                reason_parts.append(f"🌀 COILED SPRING ({days_to_earn}d to earnings)")
            if is_breaking_out:
                base_type = base_pattern.get("type", "none")
                reason_parts.append(f"🚀 BREAKOUT ({base_type}) {volume_ratio:.1f}x vol")
            elif pre_breakout_bonus >= 15:
                base_type = base_pattern.get("type", "none")
                reason_parts.append(f"📈 PRE-BREAKOUT ({base_type}) {pct_from_pivot:.0f}% below pivot")
            elif extended_penalty < 0:
                reason_parts.append(f"⚠️ Extended {abs(pct_from_pivot):.0f}% above pivot")
            # Soft zone annotation
            if score_data.get("_in_soft_zone"):
                sz_pct = int(soft_zone_mult * 100)
                if score_data.get("_stable_override"):
                    reason_parts.append(f"Soft zone {sz_pct}% pos (stable override {score_data.get('_stable_score', 0):.0f}/70)")
                else:
                    reason_parts.append(f"Soft zone {sz_pct}% pos")
            reason_parts.append(f"Score {score:.0f}")
            if has_base and not is_breaking_out and pre_breakout_bonus < 15:
                reason_parts.append(f"Base: {base_pattern.get('type', 'none')} {weeks_in_base}w")
            if not is_breaking_out and pre_breakout_bonus < 15:
                reason_parts.append(f"{pct_from_high:.1f}% from high")
            if volume_ratio >= 1.5 and not is_breaking_out:
                reason_parts.append(f"Vol {volume_ratio:.1f}x")
            if estimate_revision_bonus >= 5:
                reason_parts.append(f"📊 Est↑ {revision_pct:+.0f}%")
            elif estimate_revision_bonus <= -5:
                reason_parts.append(f"📉 Est↓ {revision_pct:+.0f}%")
            if deterministic_boost_val > 0:
                reason_parts.append(f"Det+{deterministic_boost_val}")
            if bear_base_reason:
                reason_parts.append(bear_base_reason)

            # Compute price action features for ML signal factors
            try:
                _ma21 = _nan_safe(self.data_provider.get_moving_average(ticker, current_date, 21))
                _ma50 = _nan_safe(self.data_provider.get_moving_average(ticker, current_date, 50))
                _pct_from_21ma = round(((price / _ma21) - 1) * 100, 2) if isinstance(_ma21, (int, float)) and _ma21 > 0 else 0.0
                _pct_from_50ma = round(((price / _ma50) - 1) * 100, 2) if isinstance(_ma50, (int, float)) and _ma50 > 0 else 0.0
                _atr_pct = _nan_safe(self.data_provider.get_atr(ticker, current_date, 14))
                _days_since_pullback = self.data_provider.get_days_since_spy_pullback(current_date)
                _sector_rs = self._compute_sector_rs_rank(ticker, scores)
            except Exception:
                _pct_from_21ma = 0.0
                _pct_from_50ma = 0.0
                _atr_pct = 0
                _days_since_pullback = 30
                _sector_rs = 50.0

            # Compute SPY % above 50MA for ML (continuous market gradient)
            try:
                _spy_daily = self.data_provider.get_spy_daily_data(current_date)
                _spy_close = _spy_daily.get("close", 0) if _spy_daily else 0
                _spy_ma50 = _spy_daily.get("ma50", 0) if _spy_daily else 0
                _spy_pct_above_50ma = round(((_spy_close / _spy_ma50) - 1) * 100, 2) if _spy_ma50 and _spy_ma50 > 0 and _spy_close else 0.0
            except Exception:
                _spy_pct_above_50ma = 0.0

            # Build trade journal signal factors
            signal_factors = {
                "entry_type": "breakout" if is_breaking_out else ("pre-breakout" if pre_breakout_bonus >= 15 else "standard"),
                "market_regime": "bullish" if weighted_signal >= bullish_threshold else ("bearish" if weighted_signal <= bearish_threshold else "neutral"),
                "market_timing_state": self.market_state.state.value if self.market_state_enabled else "trending",
                "rs_line_bonus": rs_line_bonus,
                "earnings_drift_bonus": earnings_drift_bonus,
                "estimate_revision_bonus": estimate_revision_bonus,
                "composite_score": round(composite_score, 1),
                # Price action features
                "relative_volume": round(volume_ratio, 2),
                "pct_from_21ma": _pct_from_21ma,
                "pct_from_50ma": _pct_from_50ma,
                "atr_pct": round(_atr_pct, 2),
                "sector_rs_rank": _sector_rs,
                "days_since_spy_pullback": _days_since_pullback,
                # v11 ML features: individual CANSLIM components
                "c_score": _nan_safe(score_data.get('c', 0) or score_data.get('c_score', 0)),
                "a_score": _nan_safe(score_data.get('a', 0) or score_data.get('a_score', 0)),
                "n_score": _nan_safe(score_data.get('n', 0) or score_data.get('n_score', 0)),
                "s_score": _nan_safe(score_data.get('s', 0) or score_data.get('s_score', 0)),
                "l_score": _nan_safe(score_data.get('l', 0) or score_data.get('l_score', 0)),
                "i_score": _nan_safe(score_data.get('i', 0) or score_data.get('i_score', 0)),
                # v11 ML features: market + sector context
                "spy_pct_above_50ma": _spy_pct_above_50ma,
                "industry_group_rank": self.static_data.get(ticker, {}).get('industry_group_rank', 50),
                # Continuous dry-up: captured ALWAYS so ML sees full distribution, not just bonus-firing cases
                "volume_dry_up_score": _nan_safe(score_data.get("_volume_dry_up_score", 0)),
            }
            if volume_dry_up_bonus > 0:
                signal_factors["volume_dry_up"] = True
            if coiled_spring_bonus > 0:
                signal_factors["coiled_spring"] = True
                # Enrich with CS detail for ML training
                if cs_result and cs_result.get("factors"):
                    cs_factors = cs_result["factors"]
                    signal_factors["cs_weeks_in_base"] = cs_factors.get("weeks_in_base", 0)
                    signal_factors["cs_beat_streak"] = cs_factors.get("earnings_beat_streak", 0)
                    signal_factors["cs_days_to_earnings"] = cs_factors.get("days_to_earnings", 0)
                    signal_factors["cs_bonus"] = cs_result.get("cs_score", 0)
                    signal_factors["cs_c_score"] = cs_factors.get("c_score", 0)
                    signal_factors["cs_institutional_pct"] = cs_factors.get("institutional_pct", 0)
                    signal_factors["cs_quality_rank"] = cs_result.get("quality_rank", 0)
                    signal_factors["cs_confidence"] = cs_result.get("confidence", 0)
            if score_data.get("_bear_market_entry"):
                signal_factors["bear_exception"] = True
            if score_data.get("_in_soft_zone"):
                signal_factors["soft_zone"] = True
                signal_factors["soft_zone_multiplier"] = soft_zone_mult
            if deterministic_boost_val > 0:
                signal_factors["deterministic_boost"] = deterministic_boost_val

            # ML Signal Layer: advisory ranking modifier (identical to ai_trader.py)
            ml_bonus = 0
            ml_confidence = None
            ml_config = self.profile.get('ml_signal', {})
            if ml_config.get('enabled', False):
                try:
                    # Pass all signal_factors through — model extracts FEATURE_COLUMNS
                    _ml_features = dict(signal_factors)
                    _ml_features["entry_type"] = ENTRY_TYPE_MAP_ML.get(signal_factors.get("entry_type", "standard"), 2)
                    _ml_features["market_regime"] = REGIME_MAP_ML.get(signal_factors.get("market_regime", "neutral"), 1)
                    _ml_features["total_score"] = effective_score
                    _ml_features["coiled_spring"] = 1 if coiled_spring_bonus > 0 else 0
                    # Pruned 2026-05-14: soft_zone + volume_dry_up no longer in
                    # FEATURE_COLUMNS, so the predictor ignores these kwargs.
                    if self._eval_model_payload is not None:
                        from ml.model import get_ml_prediction_with_model
                        ml_confidence = get_ml_prediction_with_model(
                            self._eval_model_payload, **_ml_features
                        )
                    else:
                        ml_confidence = get_ml_prediction(**_ml_features)
                    if ml_confidence is not None and ml_confidence == ml_confidence and not ml_config.get('log_only', True):
                        ml_weight = ml_config.get('weight', 20)
                        ml_bonus = (ml_confidence - 0.5) * ml_weight
                        composite_score += ml_bonus
                except Exception as e:
                    logger.debug(f"ML prediction skipped for {ticker}: {e}")
            if ml_confidence is not None and ml_confidence == ml_confidence:
                signal_factors["ml_confidence"] = round(ml_confidence, 3)
                signal_factors["ml_bonus"] = round(ml_bonus, 1)
                # NOTE: MLPrediction audit rows are written in ai_trader.py only.
                # Mirroring here would dump hundreds of thousands of simulated
                # predictions into the production audit table per backtest run.
                # The gating logic below IS mirrored — only the persistence diverges.
            # ML confidence gating: veto or reduce low-confidence candidates
            ml_min_confidence = ml_config.get('min_confidence', 0.0)
            if ml_confidence is not None and ml_confidence == ml_confidence and ml_min_confidence > 0:
                if ml_confidence < ml_min_confidence:
                    veto_action = ml_config.get('veto_action', 'skip')
                    if veto_action == 'skip':
                        signal_factors["ml_vetoed"] = True
                        logger.debug(f"ML VETO: {ticker} confidence {ml_confidence:.3f} < {ml_min_confidence}")
                        continue
                    elif veto_action == 'reduce':
                        position_value = shares * price * 0.5
                        shares = position_value / price
                        signal_factors["ml_reduced"] = True

            _funnel["passed"] += 1
            buys.append(SimulatedTrade(
                ticker=ticker,
                action="BUY",
                shares=shares,
                price=price,
                reason=" | ".join(reason_parts),
                score=score,
                priority=-composite_score  # Higher score = lower priority number
            ))
            # Stash signal_factors on trade for recording (sanitize NaN/Inf)
            buys[-1]._signal_factors = sanitize_signal_factors(signal_factors)

        # Log funnel diagnostics when decay is active
        if _funnel_diag and _funnel["candidates"] > 0:
            logger.info(f"DECAY FUNNEL: {_funnel['candidates']} candidates -> "
                        f"no_price={_funnel['no_price']}, cooldown={_funnel['cooldown']}, "
                        f"sector={_funnel['sector']}, c_filter={_funnel['c_filter']}, "
                        f"l_filter={_funnel['l_filter']}, volume={_funnel['volume']}, "
                        f"earnings_prox={_funnel['earnings_prox']} -> "
                        f"{_funnel['passed']} passed -> {len(buys)} buys")

        buys.sort(key=lambda x: x.priority)

        # === NIBBLE MODE FILTER ===
        # During correction with nibble mode active, only allow high-quality
        # defensive stocks that are holding up despite the market falling.
        nibble_active = getattr(self, 'nibble_mode_active', False)
        if nibble_active:
            nibble_cfg = self.profile.get('correction_nibble', {})
            nibble_min_score = nibble_cfg.get('min_score', 55)
            nibble_min_l = nibble_cfg.get('min_l_score', 10)
            nibble_position_pct = nibble_cfg.get('position_pct', 7)
            defensive_sectors = set(nibble_cfg.get('defensive_sectors', []))

            filtered_buys = []
            for buy in buys:
                # Apply stricter score filter
                if buy.score < nibble_min_score:
                    continue

                # Check L score (relative strength) — must be a leader
                buy_data = scores.get(buy.ticker, {})
                l_score = buy_data.get('l_score', 0)
                if l_score < nibble_min_l:
                    continue

                # Prefer defensive sectors if configured
                if defensive_sectors:
                    sector = self.static_data.get(buy.ticker, {}).get('sector', '')
                    if sector and sector not in defensive_sectors:
                        continue

                # Cap position size at nibble level
                pv = self._get_portfolio_value(current_date)
                max_nibble_value = pv * (nibble_position_pct / 100)
                trade_value = buy.shares * buy.price
                if trade_value > max_nibble_value:
                    buy.shares = max_nibble_value / buy.price

                buy.reason = f"🔍 NIBBLE: {buy.reason}"
                filtered_buys.append(buy)

            buys = filtered_buys
            if buys:
                logger.info(f"NIBBLE MODE: {len(buys)} candidates passed filter on {current_date}")

        # Log breakout candidates for debugging
        breakout_buys = [b for b in buys[:10] if "BREAKOUT" in b.reason]
        if breakout_buys:
            logger.info(f"Breakout candidates on {current_date}: "
                       f"{[b.ticker for b in breakout_buys]}")

        return buys

    def _evaluate_pyramids(self, current_date: date, scores: Dict[str, dict]) -> List[SimulatedTrade]:
        """Evaluate positions for pyramid opportunities, prioritizing breakouts"""
        pyramids = []
        portfolio_value = self._get_portfolio_value(current_date)

        for ticker, position in list(self.positions.items()):
            price = self.data_provider.get_price_on_date(ticker, current_date)
            if not price or price <= 0 or price != price:  # Guard NaN
                continue

            score_data = scores.get(ticker, {})
            current_score = _nan_safe(score_data.get("total_score", 0))

            if position.cost_basis <= 0:
                continue
            gain_pct = ((price - position.cost_basis) / position.cost_basis) * 100
            current_value = position.shares * price
            current_allocation = current_value / portfolio_value if portfolio_value > 0 else 0

            # SCALE-IN ON PULLBACKS: Add to winners pulling back on low volume
            scale_in_config = config.get('scale_in_pullbacks', {})
            if scale_in_config.get('enabled', True) and position.pyramid_count >= 2:
                min_gain = scale_in_config.get('min_gain_pct', 10.0)
                pullback_min = scale_in_config.get('pullback_pct', 3.0)
                pullback_max = scale_in_config.get('max_pullback_pct', 5.0)
                low_vol_ratio = scale_in_config.get('low_volume_ratio', 0.8)
                min_score_pullback = scale_in_config.get('min_score', 70)

                if gain_pct >= min_gain and current_score >= min_score_pullback:
                    # Check if pulling back from peak on low volume
                    if position.peak_price > 0:
                        drop_from_peak = ((position.peak_price - price) / position.peak_price) * 100
                        vol_ratio = self.data_provider.get_volume_ratio(ticker, current_date)
                        if pullback_min <= drop_from_peak <= pullback_max and vol_ratio < low_vol_ratio:
                            add_pct = scale_in_config.get('add_pct', 30.0) / 100
                            original_cost = position.shares * position.cost_basis
                            scale_amount = min(original_cost * add_pct, self.cash * 0.3)
                            # Cap at remaining room under max position allocation (sync with ai_trader)
                            max_pos_pct = self.profile.get('max_single_position_pct', 25) / 100
                            remaining_room = (max_pos_pct - current_allocation) * portfolio_value
                            scale_amount = min(scale_amount, remaining_room)
                            if scale_amount >= 100:
                                shares = scale_amount / price
                                pyramids.append(SimulatedTrade(
                                    ticker=ticker, action="PYRAMID", shares=shares,
                                    price=price,
                                    reason=f"SCALE-IN PULLBACK: +{gain_pct:.0f}% winner, -{drop_from_peak:.1f}% pullback, {vol_ratio:.1f}x vol",
                                    score=current_score, priority=-25
                                ))
                                continue  # Don't evaluate standard pyramid

            # O'Neil: add after 2-3% confirmation (lowered from 5%)
            if gain_pct < 2.5 or current_score < 70:
                continue

            # Max 2 pyramids per O'Neil 50/30/20 method
            if position.pyramid_count >= 2:
                continue

            # 1-day cooldown between pyramids (and from initial buy)
            last_pyr = self.last_pyramid_date.get(ticker)
            if last_pyr and last_pyr >= current_date:
                continue
            if position.pyramid_count == 0 and position.purchase_date >= current_date:
                continue

            # Skip if at max allocation
            if current_allocation >= MAX_POSITION_ALLOCATION:
                continue

            # Skip if not enough cash
            if self.cash < 200:
                continue

            # Check if breaking out (prefer pyramiding into breakouts)
            is_breaking_out = score_data.get("is_breaking_out", False)
            volume_ratio = score_data.get("volume_ratio", 1.0)

            # O'Neil 50/30/20: second add = 60% of original, third add = 40%
            original_cost = position.shares * position.cost_basis
            if position.pyramid_count == 0:
                pyramid_pct = 0.60  # ~30% of full position
            else:
                pyramid_pct = 0.40  # ~20% of full position
            pyramid_amount = original_cost * pyramid_pct

            # Cap by remaining room
            remaining_room = (MAX_POSITION_ALLOCATION - current_allocation) * portfolio_value
            pyramid_amount = min(pyramid_amount, remaining_room, self.cash * 0.5)

            if pyramid_amount < 100:
                continue

            shares = pyramid_amount / price

            # Build reason and set priority
            reason_parts = [f"Winner +{gain_pct:.0f}%", f"Score {current_score:.0f}"]
            priority = 0

            if is_breaking_out:
                priority = -20  # Higher priority for breakouts
                reason_parts.append("Breakout!")
            if volume_ratio >= 1.5:
                priority -= 5
                reason_parts.append(f"Vol {volume_ratio:.1f}x")

            pyramids.append(SimulatedTrade(
                ticker=ticker,
                action="PYRAMID",
                shares=shares,
                price=price,
                reason=" | ".join(reason_parts),
                score=current_score,
                priority=priority
            ))

        # Sort by priority (breakouts first)
        pyramids.sort(key=lambda x: x.priority)
        return pyramids

    def _execute_buy(self, current_date: date, trade: SimulatedTrade):
        """Execute a buy trade"""
        total_cost = trade.shares * trade.price

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        sector = self.static_data.get(trade.ticker, {}).get("sector", "Unknown")

        # Tag nibble buys as experimental (isolated from circuit breaker)
        is_experimental = getattr(self, 'nibble_mode_active', False) and "NIBBLE" in trade.reason

        self.positions[trade.ticker] = SimulatedPosition(
            ticker=trade.ticker,
            shares=trade.shares,
            cost_basis=trade.price,
            purchase_date=current_date,
            purchase_score=trade.score,
            peak_price=trade.price,
            peak_date=current_date,
            is_growth_stock=trade.is_growth_stock,
            sector=sector,
            signal_factors=sanitize_signal_factors(getattr(trade, '_signal_factors', {})),
            is_experimental=is_experimental,
        )

        self._record_trade(current_date, trade, cost_basis=trade.price)
        self.trades_executed += 1

    def _execute_sell(self, current_date: date, trade: SimulatedTrade):
        """Execute a sell trade (full or partial)"""
        if trade.ticker not in self.positions:
            return

        position = self.positions[trade.ticker]
        cost_basis = position.cost_basis
        total_value = trade.shares * trade.price
        realized_gain = total_value - (trade.shares * cost_basis)

        # Track experimental position losses separately for circuit breaker isolation
        if position.is_experimental and realized_gain < 0:
            self.experimental_realized_losses += abs(realized_gain)

        self.cash += total_value

        purchase_date = position.purchase_date

        if trade.is_partial:
            # PARTIAL SELL - reduce position but don't delete
            # Guard: never sell more shares than position holds
            if trade.shares > position.shares:
                logger.error(f"SELL GUARD: {trade.ticker} trying to sell {trade.shares:.2f} "
                             f"of {position.shares:.2f} shares, capping")
                trade.shares = position.shares
            position.shares -= trade.shares
            position.partial_profit_taken += trade.sell_pct

            logger.debug(f"PARTIAL SELL {trade.ticker}: {trade.sell_pct}% "
                        f"({trade.shares:.2f} shares), remaining: {position.shares:.2f}")
        else:
            # FULL SELL - remove position entirely and record for cooldown
            del self.positions[trade.ticker]
            self.recently_sold[trade.ticker] = (current_date, trade.reason)

            # Track outcome for buy throttle (full sells only, not partials)
            gain_pct = ((trade.price - cost_basis) / cost_basis) * 100 if cost_basis > 0 else 0
            self.recent_trade_outcomes.append(gain_pct)

        self._record_trade(
            current_date, trade,
            cost_basis=cost_basis,
            realized_gain=realized_gain,
            purchase_date=purchase_date
        )

        self.trades_executed += 1
        self.sells_executed += 1  # Track sells for win rate calculation
        if realized_gain > 0:
            self.profitable_trades += 1

    def _execute_pyramid(self, current_date: date, trade: SimulatedTrade):
        """Execute a pyramid (add to position)"""
        if trade.ticker not in self.positions:
            return

        position = self.positions[trade.ticker]
        total_cost = trade.shares * trade.price

        if total_cost > self.cash:
            return

        self.cash -= total_cost

        # Update cost basis (weighted average)
        old_value = position.shares * position.cost_basis
        new_value = trade.shares * trade.price
        position.shares += trade.shares
        position.cost_basis = (old_value + new_value) / position.shares
        position.pyramid_count += 1

        # Track pyramid date for 1-day cooldown
        self.last_pyramid_date[trade.ticker] = current_date

        self._record_trade(current_date, trade)
        self.trades_executed += 1

    def _record_trade(self, current_date: date, trade: SimulatedTrade,
                      cost_basis: float = None, realized_gain: float = None,
                      purchase_date: date = None):
        """Record trade in database"""
        position = self.positions.get(trade.ticker)
        holding_days = None
        realized_gain_pct = None

        if trade.action == "SELL" and cost_basis:
            sell_purchase_date = purchase_date or (position.purchase_date if position else None)
            holding_days = (current_date - sell_purchase_date).days if sell_purchase_date else 0
            realized_gain_pct = (realized_gain / (trade.shares * cost_basis)) * 100 if cost_basis else 0

        # Get signal_factors from trade (stashed during evaluation)
        signal_factors = getattr(trade, '_signal_factors', None)
        # For sells, get from position if available
        if trade.action == "SELL" and not signal_factors and position:
            signal_factors = position.signal_factors
        # Sanitize NaN/Inf before JSON serialization
        if signal_factors:
            signal_factors = sanitize_signal_factors(signal_factors)

        db_trade = BacktestTrade(
            backtest_id=self.backtest.id,
            date=current_date,
            ticker=trade.ticker,
            action=trade.action,
            shares=trade.shares,
            price=trade.price,
            total_value=trade.shares * trade.price,
            reason=trade.reason,
            canslim_score=trade.score,
            is_growth_stock=trade.is_growth_stock,
            cost_basis=cost_basis,
            realized_gain=realized_gain,
            realized_gain_pct=realized_gain_pct,
            holding_days=holding_days,
            signal_factors=signal_factors
        )
        self.db.add(db_trade)

    def _take_snapshot(self, current_date: date):
        """Take daily portfolio snapshot"""
        positions_value = sum(
            pos.shares * (self.data_provider.get_price_on_date(pos.ticker, current_date) or 0)
            for pos in self.positions.values()
        )
        # Include SPY sweep value
        sweep_value = 0.0
        if self.spy_sweep_shares > 0:
            spy_price = self.data_provider.get_spy_price_on_date(current_date)
            sweep_value = self.spy_sweep_shares * (spy_price or 0)
        total_value = self.cash + positions_value + sweep_value

        # Track peak and drawdown
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        drawdown = ((self.peak_portfolio_value - total_value) / self.peak_portfolio_value) * 100
        if drawdown > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown

        # Calculate returns
        cumulative_return_pct = ((total_value / self.backtest.starting_cash) - 1) * 100

        if self.daily_returns:
            prev_value = self.backtest.starting_cash * (1 + self.daily_returns[-1] / 100)
            daily_return_pct = ((total_value / prev_value) - 1) * 100
        else:
            daily_return_pct = cumulative_return_pct

        self.daily_returns.append(cumulative_return_pct)

        # SPY benchmark
        spy_price = self.data_provider.get_spy_price_on_date(current_date)
        spy_value = self.spy_shares * spy_price if spy_price else 0
        spy_return_pct = ((spy_price / self.spy_start_price) - 1) * 100 if self.spy_start_price else 0

        snapshot = BacktestSnapshot(
            backtest_id=self.backtest.id,
            date=current_date,
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            positions_count=len(self.positions),
            daily_return_pct=daily_return_pct,
            cumulative_return_pct=cumulative_return_pct,
            spy_price=spy_price,
            spy_value=spy_value,
            spy_return_pct=spy_return_pct
        )
        self.db.add(snapshot)

    def _get_portfolio_value(self, current_date: date = None) -> float:
        """Get current portfolio value using actual prices, not peak prices"""
        if current_date and self.data_provider:
            # Use actual current prices for accurate valuation
            positions_value = sum(
                pos.shares * (self.data_provider.get_price_on_date(pos.ticker, current_date) or pos.cost_basis)
                for pos in self.positions.values()
            )
            # Include SPY cash sweep value
            sweep_value = 0.0
            if self.spy_sweep_shares > 0:
                spy_price = self.data_provider.get_spy_price_on_date(current_date)
                sweep_value = self.spy_sweep_shares * (spy_price or 0)
        else:
            # Fallback to cost basis if no date provided
            positions_value = sum(
                pos.shares * pos.cost_basis
                for pos in self.positions.values()
            )
            sweep_value = 0.0
        return self.cash + positions_value + sweep_value

    def _apply_industry_group_bonus(self, scores: dict):
        """
        Post-process scores: compute industry group rankings from RS data,
        then adjust L scores with group bonus/penalty (matching live scorer).
        """
        from backend.industry_group import get_industry_group_bonus

        # Group tickers by industry, compute average RS
        industry_rs = {}  # industry -> [rs values]
        ticker_industry = {}  # ticker -> industry

        for ticker, data in scores.items():
            industry = self.static_data.get(ticker, {}).get("industry", "")
            if not industry:
                continue
            ticker_industry[ticker] = industry
            rs_12m = data.get("rs_12m")
            rs_3m = data.get("rs_3m")
            if rs_12m is not None and rs_3m is not None and rs_12m == rs_12m and rs_3m == rs_3m:
                industry_rs.setdefault(industry, []).append((rs_12m, rs_3m))

        if not industry_rs:
            return

        # Compute composite RS per group (same weights as live: 40% 12m + 60% 3m)
        group_composites = {}
        for industry, rs_pairs in industry_rs.items():
            if len(rs_pairs) < 2:
                continue  # Need 2+ stocks for meaningful group RS
            avg_12m = sum(r[0] for r in rs_pairs) / len(rs_pairs)
            avg_3m = sum(r[1] for r in rs_pairs) / len(rs_pairs)
            group_composites[industry] = avg_12m * 0.40 + avg_3m * 0.60

        if not group_composites:
            return

        # Rank groups by composite RS
        sorted_groups = sorted(group_composites.items(), key=lambda x: x[1])
        total = len(sorted_groups)
        group_ranks = {}
        for i, (industry, _) in enumerate(sorted_groups):
            group_ranks[industry] = max(1, min(100, round(((i + 1) / total) * 100)))

        # Apply bonus/penalty to L scores
        for ticker, data in scores.items():
            industry = ticker_industry.get(ticker)
            if not industry or industry not in group_ranks:
                continue
            rank = group_ranks[industry]
            bonus = get_industry_group_bonus(rank)
            if bonus != 0:
                old_l = data["l_score"]
                new_l = max(0, min(15, old_l + bonus))
                data["l_score"] = round(new_l, 1)
                # Recompute total
                data["total_score"] = (
                    data["c_score"] + data["a_score"] + data["n_score"] +
                    data["s_score"] + data["l_score"] + data["i_score"] + data["m_score"]
                )

    def _check_correlation(self, candidate_ticker: str, current_date, lookback_days: int = 30):
        """Check price correlation between candidate and held positions using cached data."""
        import numpy as np
        max_corr = 0.0
        most_correlated = None

        # Get candidate's recent returns
        candidate_prices = self.data_provider.get_price_series(
            candidate_ticker, current_date, lookback_days
        )
        if candidate_prices is None or len(candidate_prices) < 15:
            return 0.0, None

        candidate_returns = np.diff(candidate_prices) / candidate_prices[:-1]

        for held_ticker in self.positions:
            held_prices = self.data_provider.get_price_series(
                held_ticker, current_date, lookback_days
            )
            if held_prices is None or len(held_prices) < 15:
                continue

            held_returns = np.diff(held_prices) / held_prices[:-1]

            # Align lengths
            min_len = min(len(candidate_returns), len(held_returns))
            if min_len < 10:
                continue

            cr = candidate_returns[-min_len:]
            hr = held_returns[-min_len:]

            # Compute Pearson correlation
            if np.std(cr) == 0 or np.std(hr) == 0:
                continue
            corr = np.corrcoef(cr, hr)[0, 1]
            if corr != corr:  # NaN
                continue

            if abs(corr) > max_corr:
                max_corr = abs(corr)
                most_correlated = held_ticker

        return round(max_corr, 3), most_correlated

    def _check_sector_limit(self, sector: str) -> bool:
        """Check if we can add another position in this sector"""
        sector_count = sum(1 for p in self.positions.values() if p.sector == sector)
        return sector_count < MAX_STOCKS_PER_SECTOR

    def _bear_base_bonus_for_candidate(
        self, ticker, current_date, score_data, static_data, price
    ):
        """Mirror ai_trader.evaluate_buys bear-base ranking added in commit b972da3.

        Live source-of-truth is BearBaseCandidate (populated by
        bear_base.update_bear_base_candidates when SPY < 50MA). Backtester
        recomputes readiness on the fly from score_data and tracks a
        per-engine watchlist so the bonus persists for 30 days after a
        stock's last qualifying scan — matching the live behavior where
        BearBaseCandidate rows survive into the post-bear period (which
        is exactly when this bonus matters most).

        Returns: (bonus_pts, readiness, days_on_list, reason_or_None)

        Backtest-only deviations vs live:
          - institutional_accumulation, insider_sentiment not tracked → up
            to 8 of 15 accumulation pts unavailable. Affects all candidates
            uniformly; relative ranking preserved.
          - Watchlist eligibility derives from already-evaluated buy
            candidates instead of the full universe. Live's
            BearBaseCandidate is broader, but the candidates we'd boost
            are those passing quality filters anyway — same effective set.
        """
        from types import SimpleNamespace

        base_pattern = score_data.get("base_pattern") or {}
        base_type = base_pattern.get("type", "none")
        weeks = score_data.get("weeks_in_base", 0) or 0
        canslim = score_data.get("total_score") or 0
        rs_3m = score_data.get("rs_3m")

        has_valid_base = (
            base_type and base_type not in ('none', '')
            and weeks >= 3 and rs_3m is not None
        )
        has_dryup = (
            score_data.get("_volume_dry_up") and canslim >= 50
            and rs_3m is not None
        )

        readiness = 0
        if has_valid_base or has_dryup:
            try:
                atr_raw = self.data_provider.get_atr(ticker, current_date, 14)
                atr_pct = (
                    _nan_safe(atr_raw)
                    if isinstance(atr_raw, (int, float)) and not isinstance(atr_raw, bool)
                    else None
                )
            except Exception:
                atr_pct = None
            proxy = SimpleNamespace(
                c_score=score_data.get("c_score") or 0,
                a_score=score_data.get("a_score") or 0,
                l_score=score_data.get("l_score") or 0,
                canslim_score=canslim,
                rs_3m=rs_3m,
                rs_12m=score_data.get("rs_12m"),
                base_type=base_type,
                weeks_in_base=weeks,
                atr_pct=atr_pct,
                volume_dry_up=score_data.get("_volume_dry_up", False),
                institutional_accumulation=False,
                insider_sentiment="",
                industry_group_rank=static_data.get("industry_group_rank", 50) or 50,
                pivot_price=score_data.get("pivot_price", 0),
                current_price=price,
            )
            readiness, _ = compute_readiness_score(proxy)

        # Update watchlist on every scan where stock qualifies (>=20).
        if readiness >= 20:
            existing = self._bear_base_watchlist.get(ticker)
            if existing:
                existing['readiness'] = readiness
                existing['last_seen'] = current_date
            else:
                self._bear_base_watchlist[ticker] = {
                    'readiness': readiness,
                    'first_seen': current_date,
                    'last_seen': current_date,
                }

        bb = self._bear_base_watchlist.get(ticker)
        if not bb or (current_date - bb['last_seen']).days > 30:
            return 0, readiness, 0, None

        days_on_list = (current_date - bb['first_seen']).days
        if bb['readiness'] >= 70:
            bonus = 15
        elif bb['readiness'] >= 50:
            bonus = 10
        elif bb['readiness'] >= 30:
            bonus = 5
        else:
            bonus = 0
        if days_on_list >= 14:
            bonus += 3
        elif days_on_list >= 7:
            bonus += 1

        reason = f"BearBase+{bonus} (ready={bb['readiness']:.0f})" if bonus > 0 else None
        return bonus, bb['readiness'], days_on_list, reason

    def _calculate_final_metrics(self):
        """Calculate final backtest metrics"""
        # Get final snapshot using end date for accurate valuation
        final_value = self._get_portfolio_value(self.backtest.end_date)
        self.backtest.final_value = final_value
        self.backtest.total_return_pct = ((final_value / self.backtest.starting_cash) - 1) * 100
        self.backtest.max_drawdown_pct = self.max_drawdown_pct
        self.backtest.total_trades = self.trades_executed

        # Win rate - calculated from sells only (not all trades including buys)
        if self.sells_executed > 0:
            self.backtest.win_rate = (self.profitable_trades / self.sells_executed) * 100
        else:
            self.backtest.win_rate = 0

        # Sharpe ratio (simplified - using daily returns)
        if len(self.daily_returns) > 1:
            import numpy as np
            daily_returns_array = np.diff(self.daily_returns)  # Convert cumulative to daily
            if len(daily_returns_array) > 0 and np.std(daily_returns_array) > 0:
                avg_return = np.mean(daily_returns_array)
                std_return = np.std(daily_returns_array)
                # Annualized Sharpe (assuming 252 trading days)
                self.backtest.sharpe_ratio = (avg_return / std_return) * math.sqrt(252)
            else:
                self.backtest.sharpe_ratio = 0
        else:
            self.backtest.sharpe_ratio = 0

        # SPY benchmark
        trading_days = self.data_provider.get_trading_days()
        if trading_days:
            spy_end_price = self.data_provider.get_spy_price_on_date(trading_days[-1])
            if spy_end_price and self.spy_start_price:
                self.backtest.spy_final_value = self.spy_shares * spy_end_price
                self.backtest.spy_return_pct = ((spy_end_price / self.spy_start_price) - 1) * 100


def run_backtest(db: Session, backtest_id: int, profile_overrides: dict = None) -> BacktestRun:
    """Run a backtest by ID, with optional profile overrides for A/B testing."""
    engine = BacktestEngine(db, backtest_id, profile_overrides=profile_overrides)
    return engine.run()
