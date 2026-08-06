"""
Database models for CANSLIM Analyzer Web App
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Date, Text, ForeignKey, Index, JSON
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime, date, timezone
from pathlib import Path

# Database setup
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Support PostgreSQL via DATABASE_URL env var, fallback to SQLite
DATABASE_URL = os.environ.get('DATABASE_URL', f"sqlite:///{DATA_DIR}/canslim.db")

# SQLite needs check_same_thread=False; PostgreSQL does not
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# Increase pool for concurrent backtests + scanning + API requests
pool_kwargs = {}
if not DATABASE_URL.startswith("sqlite"):
    pool_kwargs = {"pool_size": 20, "max_overflow": 30, "pool_timeout": 60}

engine = create_engine(DATABASE_URL, connect_args=connect_args, **pool_kwargs)
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
    """Initialize database tables and run migrations.

    Iterates each table in Base.metadata explicitly with checkfirst=True
    instead of relying on Base.metadata.create_all. The bulk call has
    bitten us once already (commit befca59 — backtest_static_snapshot
    silently failed to materialize on first deploy even though create_all
    raised no error). The explicit loop logs SUCCESS / SKIP / FAIL per
    table, so a missing table after deploy is immediately visible rather
    than masquerading as 'migrations complete'.
    """
    import logging
    from sqlalchemy import inspect

    log = logging.getLogger(__name__)

    inspector = inspect(engine)
    existing_before = set(inspector.get_table_names())

    created, skipped, failed = [], [], []
    for table in Base.metadata.sorted_tables:
        if table.name in existing_before:
            skipped.append(table.name)
            continue
        try:
            table.create(bind=engine, checkfirst=True)
            created.append(table.name)
        except Exception as te:
            failed.append((table.name, str(te)))
            log.warning(f"init_db: failed to create {table.name}: {te}")

    log.info(f"init_db: {len(created)} created, {len(skipped)} already present, {len(failed)} failed")
    if created:
        log.info(f"init_db: new tables this run: {created}")
    if failed:
        log.error(f"init_db: {len(failed)} tables failed to create: {failed}")

    # Cross-check: any table still missing after the loop is a real bug.
    inspector = inspect(engine)
    existing_after = set(inspector.get_table_names())
    expected = set(Base.metadata.tables.keys())
    still_missing = expected - existing_after
    if still_missing:
        log.error(f"init_db: tables in metadata but missing from DB after init: {still_missing}")

    run_migrations()


def coerce_json_list(value):
    """Return a JSON-column value as a list, decoding legacy TEXT-column
    strings. Columns ALTER-added as TEXT on Postgres read back as raw JSON
    strings (the driver only auto-decodes real json/jsonb columns) — the
    run_migrations jsonb repair fixes the schema, this guards any straggler
    row or pre-migration read so a string can never 500 a response model."""
    if isinstance(value, str):
        try:
            import json
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except Exception:
            return []
    return value or []


def run_migrations():
    """Add any missing columns to existing tables and fix constraints.
    Uses SQLAlchemy inspect() for database-agnostic migrations (SQLite + PostgreSQL).
    """
    from sqlalchemy import inspect, text
    import logging

    logger = logging.getLogger(__name__)

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    if not existing_tables:
        logger.info("No tables yet, skipping migrations")
        return

    # Define column migrations: (table, column, type)
    migrations = [
        ("stocks", "previous_score", "FLOAT"),
        ("stocks", "score_change", "FLOAT"),
        ("ai_portfolio_snapshots", "timestamp", "TIMESTAMP"),
        ("ai_portfolio_snapshots", "prev_value", "FLOAT"),
        ("ai_portfolio_snapshots", "value_change", "FLOAT"),
        ("ai_portfolio_snapshots", "value_change_pct", "FLOAT"),
        ("stock_scores", "timestamp", "TIMESTAMP"),
        ("stock_scores", "week_52_high", "FLOAT"),
        ("stocks", "growth_mode_score", "FLOAT"),
        ("stocks", "growth_mode_details", "TEXT"),
        ("stocks", "is_growth_stock", "BOOLEAN"),
        ("stocks", "eps_acceleration", "BOOLEAN"),
        ("stocks", "earnings_surprise_pct", "FLOAT"),
        ("stocks", "revenue_growth_pct", "FLOAT"),
        ("stocks", "volume_ratio", "FLOAT"),
        ("stocks", "weeks_in_base", "INTEGER"),
        ("stocks", "base_type", "TEXT"),
        ("stocks", "pivot_price", "FLOAT"),
        ("stocks", "is_breaking_out", "BOOLEAN"),
        ("stocks", "breakout_volume_ratio", "FLOAT"),
        ("ai_portfolio_positions", "is_growth_stock", "BOOLEAN DEFAULT FALSE"),
        ("ai_portfolio_positions", "purchase_growth_score", "FLOAT"),
        ("ai_portfolio_positions", "current_growth_score", "FLOAT"),
        ("ai_portfolio_trades", "growth_mode_score", "FLOAT"),
        ("ai_portfolio_trades", "is_growth_stock", "BOOLEAN DEFAULT FALSE"),
        ("market_snapshots", "timestamp", "TIMESTAMP"),
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
        ("ai_portfolio_positions", "peak_price", "FLOAT"),
        ("ai_portfolio_positions", "peak_date", "TIMESTAMP"),
        ("stocks", "insider_buy_count", "INTEGER"),
        ("stocks", "insider_sell_count", "INTEGER"),
        ("stocks", "insider_net_shares", "FLOAT"),
        ("stocks", "insider_sentiment", "TEXT"),
        ("stocks", "insider_updated_at", "TIMESTAMP"),
        ("stocks", "short_interest_pct", "FLOAT"),
        ("stocks", "short_ratio", "FLOAT"),
        ("stocks", "short_updated_at", "TIMESTAMP"),
        ("stocks", "score_details", "TEXT"),
        ("stocks", "quarterly_earnings", "TEXT"),
        ("stocks", "annual_earnings", "TEXT"),
        ("stocks", "quarterly_revenue", "TEXT"),
        ("backtest_runs", "cancel_requested", "BOOLEAN DEFAULT FALSE"),
        ("stocks", "rs_12m", "FLOAT"),
        ("stocks", "rs_3m", "FLOAT"),
        ("ai_portfolio_positions", "partial_profit_taken", "FLOAT DEFAULT 0"),
        ("stocks", "next_earnings_date", "DATE"),
        ("stocks", "days_to_earnings", "INTEGER"),
        ("stocks", "earnings_beat_streak", "INTEGER"),
        ("stocks", "earnings_calendar_updated_at", "TIMESTAMP"),
        ("stocks", "eps_estimate_current", "FLOAT"),
        ("stocks", "eps_estimate_prior", "FLOAT"),
        ("stocks", "eps_estimate_revision_pct", "FLOAT"),
        ("stocks", "estimate_revision_trend", "TEXT"),
        ("stocks", "analyst_estimates_updated_at", "TIMESTAMP"),
        ("stocks", "insider_buy_value", "FLOAT"),
        ("stocks", "insider_sell_value", "FLOAT"),
        ("stocks", "insider_net_value", "FLOAT"),
        ("stocks", "insider_largest_buy", "FLOAT"),
        ("stocks", "insider_largest_buyer_title", "TEXT"),
        ("stock_data_cache", "next_earnings_date", "DATE"),
        ("stock_data_cache", "days_to_earnings", "INTEGER"),
        ("stock_data_cache", "earnings_beat_streak", "INTEGER"),
        ("stock_data_cache", "latest_surprise_pct", "FLOAT"),
        ("stock_data_cache", "earnings_calendar_updated_at", "TIMESTAMP"),
        ("stock_data_cache", "eps_estimate_current", "FLOAT"),
        ("stock_data_cache", "eps_estimate_prior", "FLOAT"),
        ("stock_data_cache", "eps_estimate_revision_pct", "FLOAT"),
        ("stock_data_cache", "analyst_estimates_updated_at", "TIMESTAMP"),
        ("stock_data_cache", "short_interest_pct", "FLOAT"),
        ("stock_data_cache", "short_ratio", "FLOAT"),
        ("stock_data_cache", "short_updated_at", "TIMESTAMP"),
        ("stocks", "volume_dry_up", "BOOLEAN DEFAULT FALSE"),
        ("stocks", "institutional_accumulation", "BOOLEAN DEFAULT FALSE"),
        ("ai_portfolio_positions", "pyramid_count", "INTEGER DEFAULT 0"),
        ("ai_portfolio_config", "peak_portfolio_value", "FLOAT DEFAULT 0"),
        ("backtest_trades", "signal_factors", "TEXT"),
        ("ai_portfolio_trades", "signal_factors", "TEXT"),
        # Paper Trading Mode (Feb 2026)
        ("ai_portfolio_config", "paper_mode", "BOOLEAN DEFAULT FALSE"),
        ("ai_portfolio_trades", "is_paper", "BOOLEAN DEFAULT FALSE"),
        # Backtest force refresh (Feb 2026)
        ("backtest_runs", "force_refresh", "BOOLEAN DEFAULT FALSE"),
        # Multi-user support (Mar 2026) — user_id FK on all user-scoped tables
        ("ai_portfolio_config", "user_id", "INTEGER"),
        ("ai_portfolio_positions", "user_id", "INTEGER"),
        ("ai_portfolio_trades", "user_id", "INTEGER"),
        ("ai_portfolio_snapshots", "user_id", "INTEGER"),
        ("portfolio_positions", "user_id", "INTEGER"),
        ("watchlist", "user_id", "INTEGER"),
        ("backtest_runs", "user_id", "INTEGER"),
        ("fidelity_snapshots", "user_id", "INTEGER"),
        ("fidelity_trades", "user_id", "INTEGER"),
        # ML v2 regression support (Mar 2026)
        ("ml_models", "model_type", "TEXT DEFAULT 'classifier'"),
        ("ml_models", "spearman", "FLOAT"),
        ("ml_models", "r2_score", "FLOAT"),
        ("ml_models", "mae", "FLOAT"),
        ("ml_models", "direction_accuracy", "FLOAT"),
        # ML A/B comparison support (Mar 2026)
        ("backtest_runs", "profile_overrides", "TEXT"),
        # CS confidence scoring (Mar 2026)
        ("coiled_spring_alerts", "confidence", "INTEGER"),
        # Price action features for ML (Mar 2026)
        ("stocks", "ma_21", "FLOAT"),
        ("stocks", "ma_50", "FLOAT"),
        ("stocks", "atr_pct", "FLOAT"),
        # Industry group strength rankings (Mar 2026)
        ("stocks", "industry_group_rank", "INTEGER"),
        # Continuous volume dry-up score 0-100 (Apr 2026, ML feature)
        ("stocks", "volume_dry_up_score", "INTEGER DEFAULT 0"),
        # Per-user notification webhook (Apr 2026)
        ("users", "webhook_url", "VARCHAR"),
        # Per-user notification preferences (May 2026): per-kind mute + quiet hours
        ("users", "mute_kinds", "TEXT"),
        ("users", "quiet_hours_start", "INTEGER"),
        ("users", "quiet_hours_end", "INTEGER"),
        # Per-user min-score gate for score-bearing alerts (May 2026)
        ("users", "score_alert_threshold", "INTEGER"),
        # cs_bear/correction_zone overlay firing counters (May 2026)
        ("backtest_runs", "overlay_stats", "TEXT"),
        # Per-backtest static snapshot extended fields (May 2026 second pass —
        # add the rest of the static_data fields beyond the original P1 set)
        ("backtest_static_snapshot", "sector", "VARCHAR"),
        ("backtest_static_snapshot", "roe", "FLOAT"),
        ("backtest_static_snapshot", "analyst_target_price", "FLOAT"),
        ("backtest_static_snapshot", "num_analyst_opinions", "INTEGER"),
        ("backtest_static_snapshot", "quarterly_earnings", "TEXT"),
        ("backtest_static_snapshot", "annual_earnings", "TEXT"),
        ("backtest_static_snapshot", "quarterly_revenue", "TEXT"),
        ("backtest_static_snapshot", "score_details", "TEXT"),
        # C-score surprise signal (May 6 2026 — added when backtester routed
        # through canslim_scorer; surprise/beat-streak bonus needs this field).
        ("backtest_static_snapshot", "earnings_surprise_pct", "FLOAT"),
        # Model graduation gate metrics (May 2026 — see _run_evaluation_backtest):
        # the eval backtest's portfolio metrics are stored on the MLModel row so
        # future candidates can be compared apples-to-apples against the
        # incumbent without re-running the incumbent's backtest each time.
        ("ml_models", "eval_backtest_id", "INTEGER"),
        ("ml_models", "eval_return_pct", "FLOAT"),
        ("ml_models", "eval_sharpe", "FLOAT"),
        ("ml_models", "eval_max_drawdown_pct", "FLOAT"),
        # Top-decile WR pre-gate metric (May 2026 — see _decile_wr_gate_decision):
        # cheap pre-flight before the slow eval-backtest gate. Cached on the
        # incumbent so candidates can be compared without re-scoring v12 each run.
        ("ml_models", "eval_decile_wr", "FLOAT"),
        # Pre-excellence-cap C value (Shadow Step 7, May 2026 — Approach 2 /
        # ec73f83 verdict requires a parallel-regime control arm). Persisted
        # forward-only; existing 1.6M rows stay NULL and are ignored by the
        # shadow override path until they age out of the eval window.
        ("stock_scores", "c_score_uncapped", "FLOAT"),
        # Analyst price-target range (May 2026 — surfaced on StockDetail's
        # Analyst Consensus card). consensus was already cached; high/low are
        # the new spread bounds for the range bar.
        ("stock_data_cache", "analyst_target_high", "FLOAT"),
        ("stock_data_cache", "analyst_target_low", "FLOAT"),
        # Strategy active at execution time (Jul 2026 — A/B attribution was
        # by CURRENT config membership, so strategy switches retroactively
        # reshuffled trade history; see AIPortfolioTrade.strategy comment).
        ("ai_portfolio_trades", "strategy", "VARCHAR"),
    ]

    # Build a cache of existing columns per table
    columns_cache = {}
    for table_name in set(t for t, _, _ in migrations):
        if table_name in existing_tables:
            columns_cache[table_name] = {c['name'] for c in inspector.get_columns(table_name)}
        else:
            columns_cache[table_name] = set()

    with engine.begin() as conn:
        for table, column, col_type in migrations:
            if table not in existing_tables:
                continue
            if column in columns_cache.get(table, set()):
                continue
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                logger.info(f"Migration: Added {table}.{column}")
            except Exception as e:
                # The columns_cache pre-check already skips existing columns,
                # so reaching here means a REAL failure (lock, disk, syntax) —
                # a missing column surfaces later as confusing runtime errors.
                logger.warning(f"Migration: failed to add {table}.{column}: {e}")

    # Fix (Jul 2026): JSON-model columns that were ALTER-added as TEXT.
    # On Postgres, SQLAlchemy's JSON type serializes on write but relies on
    # psycopg2 to deserialize on read — which only happens for real
    # json/jsonb columns. A TEXT column therefore reads back as a raw
    # string, and UserResponse(mute_kinds=<str>) 500s every /api/auth/me
    # after the user's FIRST prefs save (live incident 2026-07-22).
    # SQLite never showed it: its dialect json.loads on read.
    if not DATABASE_URL.startswith("sqlite"):
        jsonb_repairs = [("users", "mute_kinds"),
                         ("backtest_runs", "overlay_stats"),
                         # profile_overrides is physically TEXT under a Column(JSON)
                         # model (GRID/PYRGATE sweep-arm overrides); without this a
                         # new reader gets a str and silently drops the overrides.
                         ("backtest_runs", "profile_overrides")]
        with engine.begin() as conn:
            for table, column in jsonb_repairs:
                if table not in existing_tables:
                    continue
                try:
                    cur_type = conn.execute(text(
                        "SELECT data_type FROM information_schema.columns "
                        "WHERE table_name = :t AND column_name = :c"
                    ), {"t": table, "c": column}).scalar()
                    if cur_type in ("text", "character varying"):
                        conn.execute(text(
                            f"ALTER TABLE {table} ALTER COLUMN {column} "
                            f"TYPE JSONB USING NULLIF({column}, '')::jsonb"
                        ))
                        logger.info(f"Migration: {table}.{column} TEXT -> JSONB")
                except Exception as e:
                    logger.warning(f"Migration: jsonb repair {table}.{column} failed: {e}")

    # Fix: Remove unique constraint on ai_portfolio_snapshots.date (SQLite only)
    is_sqlite = DATABASE_URL.startswith("sqlite")
    if is_sqlite and "ai_portfolio_snapshots" in existing_tables:
        import sqlite3
        db_path = DATA_DIR / "canslim.db"
        if db_path.exists():
            sqlite_conn = sqlite3.connect(str(db_path))
            cursor = sqlite_conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='ix_ai_portfolio_snapshots_date' AND sql LIKE '%UNIQUE%'")
            has_unique = cursor.fetchone()

            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='ai_portfolio_snapshots'")
            table_sql = cursor.fetchone()
            needs_rebuild = has_unique or (table_sql and 'UNIQUE' in table_sql[0] and 'date' in table_sql[0].lower())

            if needs_rebuild:
                logger.info("Rebuilding ai_portfolio_snapshots table to remove unique constraint on date")
                try:
                    # NOTE: schema must include EVERY column later migrations
                    # add (user_id!) — this rebuild predated multi-user and
                    # silently dropped user_id values on legacy dev DBs.
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
                            date DATE,
                            user_id INTEGER
                        )
                    ''')

                    cursor.execute('PRAGMA table_info(ai_portfolio_snapshots)')
                    old_cols = [row[1] for row in cursor.fetchall()]

                    new_cols = ['id', 'timestamp', 'total_value', 'cash', 'positions_value', 'positions_count',
                                'total_return', 'total_return_pct', 'prev_value', 'value_change', 'value_change_pct',
                                'date', 'user_id']
                    common_cols = [c for c in new_cols if c in old_cols]
                    cols_str = ', '.join(common_cols)

                    cursor.execute(f'''
                        INSERT INTO ai_portfolio_snapshots_new ({cols_str})
                        SELECT {cols_str} FROM ai_portfolio_snapshots
                    ''')

                    cursor.execute('DROP TABLE ai_portfolio_snapshots')
                    cursor.execute('ALTER TABLE ai_portfolio_snapshots_new RENAME TO ai_portfolio_snapshots')

                    cursor.execute('CREATE INDEX IF NOT EXISTS ix_ai_portfolio_snapshots_timestamp ON ai_portfolio_snapshots(timestamp)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS ix_ai_portfolio_snapshots_date ON ai_portfolio_snapshots(date)')

                    logger.info("Successfully rebuilt ai_portfolio_snapshots table")
                except Exception as e:
                    logger.error(f"Failed to rebuild ai_portfolio_snapshots: {e}")

            sqlite_conn.commit()
            sqlite_conn.close()

    # Backfill user_id=1 for all existing data (multi-user migration)
    user_id_tables = [
        "ai_portfolio_config", "ai_portfolio_positions", "ai_portfolio_trades",
        "ai_portfolio_snapshots", "portfolio_positions", "watchlist",
        "backtest_runs", "fidelity_snapshots", "fidelity_trades",
    ]
    with engine.begin() as conn:
        for table in user_id_tables:
            if table not in existing_tables:
                continue
            if "user_id" not in columns_cache.get(table, set()):
                continue  # Column wasn't added yet (first run will add it above)
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE user_id IS NULL"))
                null_count = result.scalar()
                if null_count and null_count > 0:
                    conn.execute(text(f"UPDATE {table} SET user_id = 1 WHERE user_id IS NULL"))
                    logger.info(f"Migration: Backfilled {null_count} rows in {table} with user_id=1")
            except Exception as e:
                logger.warning(f"Failed to backfill user_id in {table}: {e}")

    # Make hashed_password nullable (Google Sign-In migration — no passwords needed)
    if "users" in existing_tables and not is_sqlite:
        try:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE users ALTER COLUMN hashed_password DROP NOT NULL"))
        except Exception:
            pass  # Already nullable or doesn't exist

    # One-time backfill: copy CANSLIM_WEBHOOK_URL into user 1 so the admin keeps
    # receiving notifications without manually re-entering the URL. Other users
    # remain null (silent) until they configure their own.
    # Try the UPDATE unconditionally — if the column doesn't exist yet (first
    # deploy edge case where ALTER TABLE hasn't happened), the UPDATE fails
    # harmlessly and the next startup picks it up.
    if "users" in existing_tables:
        env_url = os.environ.get("CANSLIM_WEBHOOK_URL", "").strip()
        if env_url:
            try:
                with engine.begin() as conn:
                    result = conn.execute(text(
                        "UPDATE users SET webhook_url = :url "
                        "WHERE id = 1 AND (webhook_url IS NULL OR webhook_url = '')"
                    ), {"url": env_url})
                    if result.rowcount:
                        logger.info(f"Migration: Backfilled webhook_url for admin user (id=1)")
            except Exception as e:
                logger.debug(f"webhook_url backfill skipped: {e}")

    # Create indexes (database-agnostic)
    index_migrations = [
        ('ix_stocks_sector', 'stocks', 'sector'),
        ('ix_stocks_canslim', 'stocks', 'canslim_score'),
        ('ix_stocks_price', 'stocks', 'current_price'),
        ('ix_stocks_score_price', 'stocks', 'canslim_score, current_price'),
        ('ix_stock_scores_stock_date', 'stock_scores', 'stock_id, date'),
        ('ix_stock_scores_stock_timestamp', 'stock_scores', 'stock_id, timestamp'),
        ('ix_stocks_growth_mode', 'stocks', 'growth_mode_score'),
        ('ix_stocks_breaking_out_idx', 'stocks', 'is_breaking_out'),
        ('ix_backtest_runs_status', 'backtest_runs', 'status'),
        ('ix_backtest_snapshots_backtest_date', 'backtest_snapshots', 'backtest_id, date'),
        ('ix_backtest_trades_backtest_date', 'backtest_trades', 'backtest_id, date'),
        ('ix_backtest_positions_backtest', 'backtest_positions', 'backtest_id'),
        ('ix_coiled_spring_alerts_ticker_date', 'coiled_spring_alerts', 'ticker, alert_date'),
        ('ix_coiled_spring_alerts_date', 'coiled_spring_alerts', 'alert_date'),
        ('ix_stocks_cs_candidates', 'stocks',
         'days_to_earnings, weeks_in_base, earnings_beat_streak, canslim_score'),
        ('ix_stocks_earnings', 'stocks', 'days_to_earnings, canslim_score'),
        ('ix_earnings_audits_ticker', 'earnings_audits', 'ticker'),
        ('ix_earnings_audits_ticker_date', 'earnings_audits', 'ticker, audited_at'),
        # Multi-user indexes
        ('ix_ai_portfolio_config_user', 'ai_portfolio_config', 'user_id'),
        ('ix_ai_portfolio_positions_user', 'ai_portfolio_positions', 'user_id'),
        ('ix_ai_portfolio_trades_user', 'ai_portfolio_trades', 'user_id'),
        ('ix_ai_portfolio_snapshots_user', 'ai_portfolio_snapshots', 'user_id'),
        ('ix_portfolio_positions_user', 'portfolio_positions', 'user_id'),
        ('ix_watchlist_user', 'watchlist', 'user_id'),
        ('ix_backtest_runs_user', 'backtest_runs', 'user_id'),
        ('ix_fidelity_snapshots_user', 'fidelity_snapshots', 'user_id'),
        ('ix_fidelity_trades_user', 'fidelity_trades', 'user_id'),
        # Composite performance indexes — optimize hot query paths
        ('ix_ai_portfolio_trades_user_action_date', 'ai_portfolio_trades', 'user_id, action, executed_at'),
        ('ix_ai_portfolio_trades_user_ticker_date', 'ai_portfolio_trades', 'user_id, ticker, executed_at'),
        ('ix_ai_portfolio_snapshots_user_timestamp', 'ai_portfolio_snapshots', 'user_id, timestamp'),
        ('ix_fidelity_trades_user_date_symbol_action', 'fidelity_trades', 'user_id, run_date, symbol, action'),
        ('ix_fidelity_trades_user_rundate', 'fidelity_trades', 'user_id, run_date'),
        ('ix_backtest_runs_user_created', 'backtest_runs', 'user_id, created_at'),
        ('ix_coiled_spring_alerts_outcome_date', 'coiled_spring_alerts', 'outcome, alert_date'),
        ('ix_earnings_audits_date_confidence', 'earnings_audits', 'audited_at, fundamental_confidence'),
        # In-app notifications (Apr 2026)
        ('ix_notifications_user_read_created', 'notifications', 'user_id, read_at, created_at'),
        ('ix_notifications_user_created', 'notifications', 'user_id, created_at'),
        # Web Push subscriptions (Apr 2026)
        ('ix_push_subscriptions_user', 'push_subscriptions', 'user_id'),
    ]

    with engine.begin() as conn:
        for idx_name, table, columns in index_migrations:
            if table not in existing_tables:
                continue
            try:
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({columns})'))
            except Exception as e:
                pass  # Expected: index already exists or table issue

    # Seed default admin user if users table is empty
    if "users" in existing_tables:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM users"))
                count = result.scalar()
                if count == 0:
                    admin_email = os.environ.get("DEFAULT_ADMIN_EMAIL", "admin@canslim.local")
                    conn.execute(text(
                        "INSERT INTO users (email, hashed_password, display_name, is_active, is_admin, created_at, updated_at) "
                        "VALUES (:email, '', :name, true, true, NOW(), NOW())"
                    ), {"email": admin_email, "name": "Owner"})
                    conn.commit()
                    logger.info(f"Seeded default admin user: {admin_email} (id=1, Google Sign-In)")
        except Exception as e:
            logger.warning(f"Failed to seed admin user: {e}")

    logger.info("Database migrations complete")


# ============== Models ==============

class User(Base):
    """Application user for multi-user support."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True, default="")
    display_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    # Per-user notification webhook (e.g. ntfy topic URL). Null = silent.
    webhook_url = Column(String, nullable=True)
    # Per-kind notification mute list. JSON array of `kind` strings the user
    # never wants delivered (push + ntfy). In-app DB row is always written;
    # only OUTBOUND delivery is suppressed so the bell still shows context.
    mute_kinds = Column(JSON, nullable=True)
    # Quiet hours window in America/Chicago local hours, [start, end). If
    # start == end the window is empty (effectively disabled). Crossing
    # midnight is allowed (e.g. start=22, end=7).
    quiet_hours_start = Column(Integer, nullable=True)
    quiet_hours_end = Column(Integer, nullable=True)
    # Minimum CANSLIM score required for score-bearing notifications
    # (breakout, coiled_spring, etc) to be delivered. Null = no threshold,
    # all alerts pass. Urgent priority alerts (stop_loss, circuit breakers)
    # always bypass — same gating model as mute_kinds and quiet_hours.
    score_alert_threshold = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class SystemSetting(Base):
    """Generic key/JSON-value store for global settings that need to survive
    container restarts. Use sparingly — per-user prefs belong on User, and
    per-strategy config belongs on its own model. This table is for *global*
    knobs the admin can change at runtime (scanner cadence, etc).
    """
    __tablename__ = "system_settings"

    key = Column(String, primary_key=True)
    # JSON-encoded value. Decoded by get_system_setting / encoded by
    # set_system_setting — callers see native Python types.
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


def get_system_setting(key: str, default=None):
    """Read a system setting by key. Returns `default` if absent or unparseable.

    Fail-soft on DB errors — a degraded DB shouldn't break boot. Callers
    using this in boot paths should always pass a sensible default.
    """
    import json
    try:
        db = SessionLocal()
        try:
            row = db.query(SystemSetting).filter_by(key=key).first()
            if row is None:
                return default
            return json.loads(row.value)
        finally:
            db.close()
    except Exception:
        return default


def set_system_setting(key: str, value) -> bool:
    """Upsert a system setting. Returns True on success, False on failure.

    Fail-soft on DB errors — persistence failure should never block the
    runtime operation that triggered the save.
    """
    import json
    try:
        db = SessionLocal()
        try:
            encoded = json.dumps(value)
            row = db.query(SystemSetting).filter_by(key=key).first()
            if row is None:
                db.add(SystemSetting(key=key, value=encoded))
            else:
                row.value = encoded
            db.commit()
            return True
        finally:
            db.close()
    except Exception:
        return False


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
    pivot_price = Column(Float)  # Breakout pivot point from base pattern
    is_breaking_out = Column(Boolean, default=False)  # Price breaking out with volume
    breakout_volume_ratio = Column(Float)  # Volume surge on breakout day

    # Relative Strength (for momentum confirmation)
    rs_12m = Column(Float)  # 12-month relative strength vs S&P 500
    rs_3m = Column(Float)  # 3-month relative strength vs S&P 500

    # Volume Profile Analysis (Feb 2026)
    volume_dry_up = Column(Boolean, default=False)  # Recent volume < 70% of baseline (bullish in base)
    volume_dry_up_score = Column(Integer, default=0)  # Continuous 0-100 (Apr 2026, ML feature)
    institutional_accumulation = Column(Boolean, default=False)  # High up/down ratio with above-avg volume
    ma_21 = Column(Float)   # 21-day simple moving average
    ma_50 = Column(Float)   # 50-day simple moving average
    atr_pct = Column(Float) # 14-day ATR as % of price

    # Industry Group Strength (Mar 2026)
    industry_group_rank = Column(Integer)  # Percentile rank 1-100 (100 = strongest group)

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

    # Earnings Calendar (Feb 2026)
    next_earnings_date = Column(Date)  # Next expected earnings date
    days_to_earnings = Column(Integer)  # Days until next earnings
    earnings_beat_streak = Column(Integer)  # Consecutive quarters beating estimates
    earnings_calendar_updated_at = Column(DateTime)

    # Analyst Estimate Revisions (Feb 2026)
    eps_estimate_current = Column(Float)  # Current consensus EPS estimate
    eps_estimate_prior = Column(Float)  # Prior period EPS estimate
    eps_estimate_revision_pct = Column(Float)  # % change in estimates
    estimate_revision_trend = Column(String)  # 'up', 'down', 'stable'
    analyst_estimates_updated_at = Column(DateTime)

    # Insider Value Tracking (Feb 2026)
    insider_buy_value = Column(Float)  # Total $ value of insider buys (3 months)
    insider_sell_value = Column(Float)  # Total $ value of insider sells (3 months)
    insider_net_value = Column(Float)  # Net $ value (buys - sells)
    insider_largest_buy = Column(Float)  # Largest single insider purchase $
    insider_largest_buyer_title = Column(String)  # Title of largest buyer (CEO, CFO, etc.)

    # Metadata
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    scores = relationship("StockScore", back_populates="stock", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_stocks_sector', 'sector'),
        Index('ix_stocks_canslim', 'canslim_score'),
        Index('ix_stocks_price', 'current_price'),
        Index('ix_stocks_score_price', 'canslim_score', 'current_price'),  # Composite for filtered queries
        Index('ix_stocks_breaking_out', 'is_breaking_out', 'canslim_score'),  # For breakout queries
        Index('ix_stocks_growth', 'is_growth_stock', 'growth_mode_score'),  # For growth stock queries
        Index('ix_stocks_sector_score', 'sector', 'canslim_score'),  # For sector-based filtering
        Index('ix_stocks_price_growth', 'current_price', 'is_growth_stock'),  # For price + growth filtering
    )


class StockScore(Base):
    """Historical CANSLIM scores for tracking changes - one record per scan"""
    __tablename__ = "stock_scores"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    date = Column(Date, nullable=False)  # Kept for easy daily grouping (indexed via __table_args__)

    # CANSLIM breakdown
    total_score = Column(Float)
    c_score = Column(Float)
    # Pre-excellence-cap C value (Approach 2 / ec73f83). Equals c_score when
    # no cap was applied. Persisted forward-only so shadow strategies
    # (Step 7) can reconstruct an un-capped canslim_score for parallel-regime
    # A/B against the capped baseline. Nullable for backfill safety on the
    # ~1.6M historical rows written before this column existed.
    c_score_uncapped = Column(Float, nullable=True)
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
        Index('ix_stock_scores_date', 'date'),  # For date-range queries across all stocks
    )


class PortfolioPosition(Base):
    """User's portfolio positions"""
    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
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

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Watchlist(Base):
    """Stocks user is watching"""
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    notes = Column(Text)
    target_price = Column(Float)  # Alert when reaches this price
    alert_score = Column(Float)  # Alert when CANSLIM score reaches this

    # Alert tracking fields
    alert_triggered_at = Column(DateTime)  # When alert was triggered
    alert_sent = Column(Boolean, default=False)  # Has email been sent
    last_check_price = Column(Float)  # Price at last check (for comparison)


class CoiledSpringAlert(Base):
    """
    Track Coiled Spring earnings catalyst alerts for analysis and limiting.

    A Coiled Spring setup identifies stocks with explosive earnings potential:
    - Long consolidation (stored energy)
    - Consistent earnings beats
    - Low institutional ownership (room to buy)
    - Rising relative strength
    - Approaching earnings
    """
    __tablename__ = "coiled_spring_alerts"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    alert_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Snapshot at alert time
    days_to_earnings = Column(Integer)
    weeks_in_base = Column(Integer)
    beat_streak = Column(Integer)
    c_score = Column(Float)
    total_score = Column(Float)
    cs_bonus = Column(Float)
    price_at_alert = Column(Float)

    # Additional context
    base_type = Column(String)  # flat, cup, cup_with_handle, etc.
    institutional_pct = Column(Float)
    l_score = Column(Float)

    # Outcome tracking (filled after earnings)
    price_after_earnings = Column(Float)
    price_change_pct = Column(Float)
    outcome = Column(String)  # 'big_win', 'win', 'flat', 'loss'
    outcome_updated_at = Column(DateTime)

    # Confidence scoring (0-100, rule-based from historical patterns)
    confidence = Column(Integer)

    # Alert status
    email_sent = Column(Boolean, default=False)

    __table_args__ = (
        Index('ix_coiled_spring_alerts_ticker_date', 'ticker', 'alert_date'),
        Index('ix_coiled_spring_alerts_date', 'alert_date'),
    )


class BreakoutAlert(Base):
    """
    Track intraday breakout alerts with DB-backed dedup + daily-cap enforcement.

    Mirrors the CoiledSpringAlert pattern (DB-backed, survives restart) so the
    breakout monitor's per-ticker cooldown and daily cap are not wiped every
    container redeploy. Pre-fix, the dedup lived in a Python module-level dict
    that reset on every restart, producing repeat alerts for the same ticker
    multiple times per day across redeploys.

    Cooldown semantics:
      - One alert per ticker per 24h (queried by created_at).
      - Hard daily cap (queried by alert_date), default 10.
    """
    __tablename__ = "breakout_alerts"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    alert_date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Snapshot at alert time
    pivot_price = Column(Float)
    current_price = Column(Float)
    vol_ratio = Column(Float)
    label = Column(String)  # "BREAKOUT" or "BREAKOUT + VOLUME"

    __table_args__ = (
        Index('ix_breakout_alerts_ticker_created', 'ticker', 'created_at'),
        Index('ix_breakout_alerts_date', 'alert_date'),
    )


class SystemState(Base):
    """
    Tiny key/value store for cross-restart system state that doesn't justify
    its own table.

    Use cases (May 2026):
      - last_spy_gate_state: BULLISH/BEARISH; survives restart so the
        SPY-flip notification only fires on a real state change.

    Keep this lean. If a use case needs more than (key, value, updated_at),
    give it its own table.
    """
    __tablename__ = "system_state"

    key = Column(String, primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class EarningsAudit(Base):
    """
    Deep fundamental audit of buy candidates using FMP data.

    Runs between scan and AI trading phases. Enriches top candidates
    with analyst targets, earnings quality, financial health, insider
    conviction, and estimate revision data to compute a fundamental_confidence
    score (0-100) that modifies the composite buy score.
    """
    __tablename__ = "earnings_audits"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    audited_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Analyst target consensus
    analyst_avg_target = Column(Float)       # Average price target
    analyst_high_target = Column(Float)      # Highest target
    analyst_low_target = Column(Float)       # Lowest target
    analyst_num = Column(Integer)            # Number of analysts
    analyst_upside_pct = Column(Float)       # % upside to avg target from current price

    # Earnings beat quality
    beat_streak = Column(Integer)            # Consecutive quarterly beats
    avg_beat_magnitude = Column(Float)       # Average EPS surprise %
    last_beat_pct = Column(Float)            # Most recent quarter surprise %

    # Financial health
    roe = Column(Float)                      # Return on equity
    debt_to_equity = Column(Float)           # Debt/equity ratio
    free_cash_flow_per_share = Column(Float) # FCF per share
    current_ratio = Column(Float)            # Current assets / current liabilities

    # Insider conviction
    insider_net_value = Column(Float)        # Net insider buy value (90d)
    insider_cluster_buys = Column(Integer)   # Number of distinct insider buyers (90d)

    # Estimate revisions
    eps_revision_pct = Column(Float)         # EPS estimate revision %
    revenue_revision_pct = Column(Float)     # Revenue estimate revision %

    # Composite confidence score
    fundamental_confidence = Column(Float)   # 0-100 composite score
    confidence_breakdown = Column(JSON)      # Per-component scores for transparency

    # Current price at audit time (for upside calc)
    price_at_audit = Column(Float)

    __table_args__ = (
        Index('ix_earnings_audits_ticker_date', 'ticker', 'audited_at'),
    )


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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MarketSnapshot(Base):
    """Daily market direction snapshot with multi-index support"""
    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

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

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SpyIntradayPrice(Base):
    """Append-only intraday SPY price log, written (throttled) on each
    market-direction refresh. MarketSnapshot above is one row per DAY
    updated in place, which destroys intraday history as it's written —
    this table preserves it so the AI Portfolio performance chart can show
    SPY moving in parallel with the per-scan portfolio snapshots.
    """
    __tablename__ = "spy_intraday_prices"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, nullable=False,
                       default=lambda: datetime.now(timezone.utc))
    price = Column(Float, nullable=False)


# ============== AI Portfolio Models ==============

class AIPortfolioConfig(Base):
    """AI Portfolio configuration"""
    __tablename__ = "ai_portfolio_config"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    starting_cash = Column(Float, default=25000.0)
    current_cash = Column(Float, default=25000.0)
    max_positions = Column(Integer, default=15)
    max_position_pct = Column(Float, default=10.0)  # Max % of portfolio per position
    min_score_to_buy = Column(Integer, default=75)
    sell_score_threshold = Column(Integer, default=50)  # Sell if score drops below
    take_profit_pct = Column(Float, default=25.0)  # Take profits at this gain %
    stop_loss_pct = Column(Float, default=15.0)  # Stop loss at this loss %
    is_active = Column(Boolean, default=True)
    strategy = Column(String, default="balanced")  # balanced, growth
    peak_portfolio_value = Column(Float, default=0.0)  # Track peak for drawdown circuit breaker
    spy_sweep_shares = Column(Float, default=0.0)  # SPY cash sweep: shares parked in SPY
    paper_mode = Column(Boolean, default=False)  # Simulate trades without mutating real cash/positions
    # NOTE: a raw migration (see _MIGRATIONS in this file) also adds this column
    # to existing prod tables; mapping it here is what makes the ORM actually
    # load/persist it (otherwise getattr(config,'paper_mode') was always False).
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AIPortfolioPosition(Base):
    """AI Portfolio current positions"""
    __tablename__ = "ai_portfolio_positions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    shares = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False)  # Price per share when bought
    purchase_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
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

    # Pyramiding tracking (O'Neil 60/40 sizing)
    pyramid_count = Column(Integer, default=0)  # Number of times position has been pyramided (max 2)

    # Partial profit taking tracking
    partial_profit_taken = Column(Float, default=0)  # Cumulative % of position sold as partial profits

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_positions_user_ticker', 'user_id', 'ticker'),
    )


class AIPortfolioTrade(Base):
    """AI Portfolio trade history"""
    __tablename__ = "ai_portfolio_trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
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
    holding_days = Column(Integer)  # Days held (sell trades only)

    # Trade Journal / Performance Attribution
    signal_factors = Column(JSON)  # {"entry_type": "pre-breakout", "market_regime": "bullish", ...}

    # Strategy active at execution time (Jul 2026). Before this column, A/B
    # attribution resolved users from their CURRENT config — so switching a
    # user's strategy retroactively reshuffled historical trade attribution
    # (live incident: nostate_optimized's June baseline vanished after users
    # 1+2 moved to nostate_cs_bear). NULL on legacy rows.
    strategy = Column(String, nullable=True)

    executed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index('ix_trades_user_executed', 'user_id', 'executed_at'),
    )


class AIPortfolioSnapshot(Base):
    """AI Portfolio snapshots for performance chart - taken after each scan"""
    __tablename__ = "ai_portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    timestamp = Column(DateTime, index=True, nullable=False, default=lambda: datetime.now(timezone.utc))

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
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String)  # User-friendly name
    status = Column(String, default="pending")  # pending, running, completed, failed

    # Configuration
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    starting_cash = Column(Float, default=25000.0)
    stock_universe = Column(String, default="all")  # sp500, all, custom
    strategy = Column(String, default="balanced")  # balanced, growth
    custom_tickers = Column(JSON)  # If universe is custom

    # AI Config snapshot (frozen at backtest start)
    max_positions = Column(Integer, default=8)
    max_position_pct = Column(Float, default=12.0)
    min_score_to_buy = Column(Integer, default=72)
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)
    error_message = Column(Text)
    progress_pct = Column(Float, default=0.0)  # 0-100 progress during run
    cancel_requested = Column(Boolean, default=False)  # Flag to request cancellation
    force_refresh = Column(Boolean, default=False)  # Force fresh FMP earnings fetch (ignore cache)
    profile_overrides = Column(JSON, nullable=True)  # Optional strategy profile overrides for A/B testing
    overlay_stats = Column(JSON, nullable=True)  # cs_bear/correction_zone overlay firing counters for diagnosis

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

    # Trade Journal / Performance Attribution
    signal_factors = Column(JSON)  # {"entry_type": "pre-breakout", "market_regime": "bullish", ...}

    backtest_run = relationship("BacktestRun", back_populates="trades")

    __table_args__ = (
        Index('ix_backtest_trades_backtest_date', 'backtest_id', 'date'),
    )


class BacktestHoldSnapshot(Base):
    """Per-day snapshot of a HELD position during a backtest — training data for
    the exit/hold ML model. Captured only when the ``hold_snapshot_capture``
    backtester lever is enabled (default OFF), so normal runs are unaffected.

    ``features`` is the position-state feature vector at the hold decision point
    (gain, drop_from_peak, days_held, score trajectory, ATR, regime, …).
    ``fwd_return_pct`` is the label: the position's return over ``horizon_days``
    forward trading days. It is NULL when the forward window runs past the sim's
    end (those rows are dropped at training time).
    """
    __tablename__ = "backtest_hold_snapshots"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String, nullable=False)

    features = Column(JSON)             # position-state feature vector
    fwd_return_pct = Column(Float)      # forward-horizon return label (nullable)
    horizon_days = Column(Integer)      # forward horizon actually used (trading days)

    __table_args__ = (
        Index('ix_backtest_hold_snap_bt_date', 'backtest_id', 'date'),
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


class BacktestStaticSnapshot(Base):
    """Per-ticker snapshot of mutable P1-cache scalars taken at backtest
    creation. Solves the cross-day reproducibility problem from
    canslim-livescan-churn-investigation.md: stock_data_cache is a
    single-row-per-ticker mutable cache, so two backtests created days
    apart against the same window read different P1 values and produce
    different trades. Reading from a snapshot keyed by backtest_id makes
    re-runs identical and lets historical comparisons stay meaningful.

    Backwards compat: existing backtests with no snapshot rows fall back
    to the live cache in BacktestEngine._load_static_data, preserving
    legacy behavior.
    """
    __tablename__ = "backtest_static_snapshot"

    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    ticker = Column(String, nullable=False, index=True)

    # P1 fields (May 4 first-pass — caught the 50pp leak)
    days_to_earnings = Column(Integer)
    earnings_beat_streak = Column(Integer)
    eps_estimate_revision_pct = Column(Float)
    industry_group_rank = Column(Integer)
    weeks_in_base = Column(Integer)

    # Extended fields (May 4 second-pass — close the residual 0.5pp drift).
    # Every other static_data input read in BacktestEngine._load_static_data
    # so a re-run with this snapshot reproduces identically.
    sector = Column(String)
    roe = Column(Float)
    analyst_target_price = Column(Float)
    num_analyst_opinions = Column(Integer)
    quarterly_earnings = Column(Text)  # JSON array
    annual_earnings = Column(Text)     # JSON array
    quarterly_revenue = Column(Text)   # JSON array
    score_details = Column(Text)       # JSON object — used for institutional_holders_pct

    # C-score signal (May 6 — added when backtester routed through canslim_scorer
    # which reads `earnings_surprise_pct` for its surprise/beat-streak bonus block.
    # Without this column the snapshot couldn't reproduce historical C scores
    # because the bonus was unreachable in replay.
    earnings_surprise_pct = Column(Float)

    snapshot_taken_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_backtest_static_snap_btid_ticker', 'backtest_id', 'ticker', unique=True),
    )


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
    analyst_target_high = Column(Float)
    analyst_target_low = Column(Float)
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

    # P1 Cache Fields (Feb 2026)
    # Earnings Calendar
    next_earnings_date = Column(Date)
    days_to_earnings = Column(Integer)
    earnings_beat_streak = Column(Integer)
    latest_surprise_pct = Column(Float)  # Most recent quarter's EPS surprise % (added May 2026)
    earnings_calendar_updated_at = Column(DateTime)

    # Analyst Estimates
    eps_estimate_current = Column(Float)
    eps_estimate_prior = Column(Float)
    eps_estimate_revision_pct = Column(Float)
    analyst_estimates_updated_at = Column(DateTime)

    # Short Interest
    short_interest_pct = Column(Float)
    short_ratio = Column(Float)
    short_updated_at = Column(DateTime)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


# ============== Fidelity Sync Models ==============

class FidelitySnapshot(Base):
    """A point-in-time snapshot from a Fidelity positions CSV upload."""
    __tablename__ = "fidelity_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Account summary
    account_number = Column(String, default="")
    cash_balance = Column(Float, default=0)
    total_value = Column(Float, default=0)
    positions_count = Column(Integer, default=0)

    # Relationships
    positions = relationship("FidelityPosition", back_populates="snapshot", cascade="all, delete-orphan")


class FidelityPosition(Base):
    """Individual position within a Fidelity snapshot."""
    __tablename__ = "fidelity_positions"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(Integer, ForeignKey("fidelity_snapshots.id"), nullable=False, index=True)

    symbol = Column(String, nullable=False, index=True)
    description = Column(String)
    quantity = Column(Float, nullable=False)
    last_price = Column(Float)
    current_value = Column(Float)
    total_gain_loss = Column(Float)
    total_gain_loss_pct = Column(Float)
    cost_basis_total = Column(Float)
    average_cost_basis = Column(Float)
    percent_of_account = Column(Float)
    position_type = Column(String)  # Margin, Cash

    snapshot = relationship("FidelitySnapshot", back_populates="positions")


class FidelityTrade(Base):
    """Parsed trade from a Fidelity activity CSV upload."""
    __tablename__ = "fidelity_trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    run_date = Column(Date, nullable=False, index=True)
    action = Column(String, nullable=False)  # BUY, SELL
    symbol = Column(String, nullable=False, index=True)
    description = Column(String)
    price = Column(Float)
    quantity = Column(Float)
    amount = Column(Float)
    commission = Column(Float, default=0)
    fees = Column(Float, default=0)
    settlement_date = Column(Date)
    raw_action = Column(String)  # Original Fidelity action text

    __table_args__ = (
        Index('ix_fidelity_trades_symbol_date', 'symbol', 'run_date'),
    )


class MLModel(Base):
    """ML model training run metadata and metrics."""
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(Integer, nullable=False)
    strategy = Column(String, nullable=False, default="nostate_optimized")
    status = Column(String, default="training")  # training, completed, failed, active

    training_samples = Column(Integer)
    feature_count = Column(Integer)
    backtest_ids = Column(JSON)  # List of backtest run IDs used
    hyperparameters = Column(JSON)

    model_type = Column(String, default="classifier")  # classifier or regression

    # Classifier metrics
    roc_auc = Column(Float)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1 = Column(Float)
    brier_score = Column(Float)

    # Regression metrics
    spearman = Column(Float)
    r2_score = Column(Float)
    mae = Column(Float)
    direction_accuracy = Column(Float)

    cv_results = Column(JSON)  # Per-fold details
    feature_importance = Column(JSON)

    model_path = Column(String)
    error_message = Column(String)

    # Model graduation gate metrics: portfolio outcome of running this model
    # in a standardized eval backtest. Used by _run_training to compare a
    # new candidate against the incumbent's stored eval_return_pct + eval_sharpe.
    # Per-trade WR + AUC on the OOS holdout were both shown to disagree with
    # actual portfolio return (May 5 diagnostic), so we gate on the thing we
    # actually care about — return + Sharpe — not a proxy.
    eval_backtest_id = Column(Integer)
    eval_return_pct = Column(Float)
    eval_sharpe = Column(Float)
    eval_max_drawdown_pct = Column(Float)

    # Top-decile WR on the OOS holdout. The May 5 v12-vs-v17 diagnostic showed
    # the strategy only ever trades from the top decile (after score>=72 +
    # max_positions=8 + sector caps), so backtest-relevant model quality lives
    # there — not in full-distribution AUC. Used by the decile-WR pre-gate to
    # cheaply reject candidates whose top-decile WR is materially below the
    # incumbent's, before paying for the ~10 min eval backtest.
    eval_decile_wr = Column(Float)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    activated_at = Column(DateTime)

    predictions = relationship("MLPrediction", back_populates="model")


class MLPrediction(Base):
    """Audit log of ML predictions for tracking accuracy."""
    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id"), nullable=False, index=True)

    ticker = Column(String, nullable=False, index=True)
    prediction_date = Column(Date, nullable=False, index=True)
    ml_confidence = Column(Float, nullable=False)
    features = Column(JSON)

    actual_outcome = Column(Integer)  # 1=win, 0=loss, NULL=pending
    actual_gain_pct = Column(Float)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    model = relationship("MLModel", back_populates="predictions")

    __table_args__ = (
        Index('ix_ml_predictions_model_date', 'model_id', 'prediction_date'),
    )


class DelistedTicker(Base):
    """
    Tracks tickers that are delisted, invalid, or consistently fail to fetch.
    These are excluded from future scans to avoid wasting API calls.
    """
    __tablename__ = "delisted_tickers"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    reason = Column(String)  # "404_not_found", "no_price_data", "delisted", etc.
    source = Column(String)  # Which index/source it came from
    failure_count = Column(Integer, default=1)  # Number of consecutive failures
    first_failed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_failed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Allow re-checking after some time (ticker might be re-listed or data fixed)
    recheck_after = Column(DateTime)  # If set, can be rechecked after this date


class BearBaseCandidate(Base):
    """
    Stocks building quality bases during bear markets.
    Updated after each scan when SPY is below 50MA.
    Ranked by readiness score — ready-to-buy list when market turns.
    """
    __tablename__ = "bear_base_candidates"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)

    # Quality metrics (snapshot at scan time)
    canslim_score = Column(Float)
    c_score = Column(Float)
    a_score = Column(Float)
    l_score = Column(Float)
    rs_12m = Column(Float)
    rs_3m = Column(Float)
    industry_group_rank = Column(Integer)

    # Base characteristics
    base_type = Column(String)  # flat_base, cup, cup_with_handle, double_bottom
    weeks_in_base = Column(Integer)
    atr_pct = Column(Float)  # Tightness of the base
    pivot_price = Column(Float)
    current_price = Column(Float)
    pct_from_pivot = Column(Float)  # How close to breakout

    # Accumulation signals
    volume_dry_up = Column(Boolean, default=False)
    institutional_accumulation = Column(Boolean, default=False)
    insider_sentiment = Column(String)

    # Composite readiness score (0-100, higher = more ready)
    readiness_score = Column(Float)
    readiness_factors = Column(JSON)  # Breakdown of score components

    # Tracking
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    days_on_list = Column(Integer, default=0)

    __table_args__ = (
        Index('ix_bear_base_readiness', 'readiness_score'),
        Index('ix_bear_base_ticker_updated', 'ticker', 'last_updated'),
    )


class PushSubscription(Base):
    """Web Push subscription for a user's device.

    One row per (user, device). Created when the user enables push from the
    Settings page; the browser hands us a unique endpoint URL that we POST
    to when we want to deliver a push. Endpoint URLs are unguessable random
    strings — the UNIQUE constraint prevents duplicate rows when a phone
    re-subscribes (e.g. after clearing site data).
    """
    __tablename__ = "push_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    endpoint = Column(Text, nullable=False, unique=True)
    # ECDH public key (Base64URL) the browser uses to encrypt push payloads.
    p256dh_key = Column(String, nullable=False)
    # 16-byte random shared-secret seed (Base64URL) for the same encryption.
    auth_key = Column(String, nullable=False)
    user_agent = Column(String, nullable=True)  # for the device list UI
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('ix_push_subscriptions_user', 'user_id'),
    )


class Notification(Base):
    """Per-user in-app notification.

    Dual-write target: the existing send_*_webhook helpers continue to fire ntfy
    (when a user has webhook_url set); they also create a row here so the user
    can read recent activity inside the app even if their phone push failed or
    their URL is blank. Scoped strictly by user_id — never list across users.
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # 'trade', 'stop_loss', 'score_crash', etc. Free-form so future event types
    # can be added without a migration.
    kind = Column(String, nullable=False)
    title = Column(String, nullable=False)
    body = Column(Text, nullable=False, default="")

    # Mirrors ntfy priority: 'low', 'default', 'high', 'urgent'
    priority = Column(String, default="default")
    tags = Column(JSON, nullable=True)   # list[str], optional ntfy emoji tags
    data = Column(JSON, nullable=True)   # arbitrary structured payload (ticker, price, etc.)

    read_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        # Hot path: list a user's notifications newest-first, unread on top.
        Index('ix_notifications_user_read_created', 'user_id', 'read_at', 'created_at'),
        Index('ix_notifications_user_created', 'user_id', 'created_at'),
    )


# ── Shadow paper-trading models ───────────────────────────────────────────────
# Forward-only scoring evaluation: alternative scoring stacks run alongside
# the live scanner and emit virtual BUY/SELL decisions. The /admin/strategy-
# ab-eval framework reads from these tables when invoked with source=shadow,
# letting us evaluate scoring changes WITHOUT committing capital. See
# docs/shadow-paper-trading-design.md for the full design.

class ShadowStrategy(Base):
    """A candidate scoring stack that runs in parallel with the live scanner.

    config_snapshot freezes the YAML profile at registration time so a
    long-running shadow comparison cannot drift if the YAML is later edited
    (same defensive pattern as BacktestRun.profile_overrides).
    """
    __tablename__ = "shadow_strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    parent_strategy = Column(String, nullable=False)
    config_snapshot = Column(JSON, nullable=False)
    scorer_overrides = Column(JSON, nullable=False, default=dict)
    description = Column(String)
    starting_value = Column(Float, nullable=False, default=25000.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    activated_at = Column(DateTime)
    archived_at = Column(DateTime, index=True)


class ShadowTrade(Base):
    """A virtual BUY or SELL emitted by a shadow strategy.

    Columns mirror AIPortfolioTrade so the existing _summarize_window /
    _decide helpers in routes/admin.py work without per-source code paths.
    No user_id column — shadow trades belong to a strategy, not a user.
    """
    __tablename__ = "shadow_trades"

    id = Column(Integer, primary_key=True, index=True)
    shadow_strategy_id = Column(Integer, ForeignKey("shadow_strategies.id"), nullable=False, index=True)
    ticker = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    reason = Column(String)
    canslim_score = Column(Float)
    cost_basis = Column(Float)
    realized_gain = Column(Float)
    holding_days = Column(Integer)
    signal_factors = Column(JSON)
    executed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    __table_args__ = (
        Index('ix_shadow_trades_strategy_executed', 'shadow_strategy_id', 'executed_at'),
    )


class ShadowPositionPeak(Base):
    """Persisted peak-price state for open shadow positions.

    The FIFO rebuild in shadow_trader derives positions from the trade log
    alone, so peaks collapsed to max(cost_basis, current_price) every tick —
    above water drop-from-peak was ~0, underwater peak_gain was 0 (below the
    lowest trailing tier), so shadow trailing stops could NEVER fire
    (verified live 2026-07-25: zero trailing exits in 2.5 months of shadow
    history vs ~20 on the lead book). This table ratchets the peak forward
    across ticks, keyed by position generation (strategy, ticker, opened_at).
    Rows are deleted when the position fully closes.
    """
    __tablename__ = "shadow_position_peaks"

    id = Column(Integer, primary_key=True, index=True)
    shadow_strategy_id = Column(Integer, ForeignKey("shadow_strategies.id"), nullable=False, index=True)
    ticker = Column(String, nullable=False)
    opened_at = Column(DateTime, nullable=False)
    peak_price = Column(Float, nullable=False)
    peak_date = Column(DateTime)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_shadow_peaks_strategy_ticker', 'shadow_strategy_id', 'ticker'),
    )


# ── SystemState helpers ───────────────────────────────────────────────────────
# Tiny accessors that hide the ORM boilerplate at every call site. The
# SPY-flip detector (ai_trader) uses the two-call pattern (read previous,
# write current after acting).

def get_system_state(db, key: str, default=None) -> str:
    """Read a SystemState value by key. Returns ``default`` if absent."""
    row = db.query(SystemState).filter(SystemState.key == key).first()
    if row is None:
        return default
    return row.value


def set_system_state(db, key: str, value) -> None:
    """Upsert a SystemState (key, value). Caller commits.

    ``value`` is coerced to str (the column is Text). Pass dates as ISO strings.
    """
    row = db.query(SystemState).filter(SystemState.key == key).first()
    str_value = None if value is None else str(value)
    if row is None:
        db.add(SystemState(key=key, value=str_value,
                           updated_at=datetime.now(timezone.utc)))
    else:
        row.value = str_value
        row.updated_at = datetime.now(timezone.utc)
