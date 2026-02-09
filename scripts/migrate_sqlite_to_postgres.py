#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL.

Usage:
    python scripts/migrate_sqlite_to_postgres.py

Requires DATABASE_URL env var pointing to PostgreSQL.
SQLite database is read from data/canslim.db.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker

# Source: SQLite
DATA_DIR = Path(__file__).parent.parent / "data"
SQLITE_URL = f"sqlite:///{DATA_DIR}/canslim.db"

# Target: PostgreSQL from env
PG_URL = os.environ.get('DATABASE_URL')
if not PG_URL:
    print("ERROR: Set DATABASE_URL environment variable to your PostgreSQL connection string")
    print("Example: DATABASE_URL=postgresql://canslim:password@localhost:5432/canslim")
    sys.exit(1)

print(f"Source: {SQLITE_URL}")
print(f"Target: {PG_URL.split('@')[0]}@****")

# Create engines
sqlite_engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
pg_engine = create_engine(PG_URL)

SqliteSession = sessionmaker(bind=sqlite_engine)
PgSession = sessionmaker(bind=pg_engine)

# Import all models to create tables
from backend.database import Base, init_db

# Create all tables in PostgreSQL
print("\nCreating tables in PostgreSQL...")
os.environ['DATABASE_URL'] = PG_URL
Base.metadata.create_all(bind=pg_engine)

# Get table names in dependency order (parents before children)
TABLE_ORDER = [
    "stocks",
    "stock_scores",
    "stock_data_cache",
    "delisted_tickers",
    "portfolio_positions",
    "watchlist",
    "analysis_jobs",
    "market_snapshots",
    "ai_portfolio_config",
    "ai_portfolio_positions",
    "ai_portfolio_trades",
    "ai_portfolio_snapshots",
    "backtest_runs",
    "backtest_snapshots",
    "backtest_trades",
    "backtest_positions",
    "coiled_spring_alerts",
]

BATCH_SIZE = 1000

sqlite_inspector = inspect(sqlite_engine)
sqlite_tables = sqlite_inspector.get_table_names()

pg_inspector = inspect(pg_engine)
pg_tables = pg_inspector.get_table_names()

total_migrated = 0

for table_name in TABLE_ORDER:
    if table_name not in sqlite_tables:
        print(f"  SKIP {table_name} (not in SQLite)")
        continue
    if table_name not in pg_tables:
        print(f"  SKIP {table_name} (not created in PostgreSQL)")
        continue

    # Get columns that exist in both
    sqlite_cols = {c['name'] for c in sqlite_inspector.get_columns(table_name)}
    pg_cols = {c['name'] for c in pg_inspector.get_columns(table_name)}
    common_cols = sorted(sqlite_cols & pg_cols)

    if not common_cols:
        print(f"  SKIP {table_name} (no common columns)")
        continue

    # Count rows in SQLite
    with sqlite_engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

    if count == 0:
        print(f"  SKIP {table_name} (empty)")
        continue

    print(f"  Migrating {table_name}: {count} rows...", end="", flush=True)

    cols_str = ", ".join(common_cols)
    placeholders = ", ".join(f":{c}" for c in common_cols)

    # Read from SQLite and write to PostgreSQL in batches
    migrated = 0
    with sqlite_engine.connect() as src_conn:
        result = src_conn.execute(text(f"SELECT {cols_str} FROM {table_name}"))

        batch = []
        for row in result:
            row_dict = dict(zip(common_cols, row))
            batch.append(row_dict)

            if len(batch) >= BATCH_SIZE:
                with pg_engine.begin() as dst_conn:
                    dst_conn.execute(
                        text(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"),
                        batch
                    )
                migrated += len(batch)
                batch = []

        # Final batch
        if batch:
            with pg_engine.begin() as dst_conn:
                dst_conn.execute(
                    text(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"),
                    batch
                )
            migrated += len(batch)

    # Reset sequences for tables with serial IDs (PostgreSQL)
    if 'id' in common_cols:
        try:
            with pg_engine.begin() as conn:
                conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table_name}), 1))"
                ))
        except Exception:
            pass

    total_migrated += migrated
    print(f" {migrated} rows migrated")

print(f"\nMigration complete! {total_migrated} total rows migrated.")
print("\nNext steps:")
print("1. Verify data: psql -U canslim -d canslim -c 'SELECT COUNT(*) FROM stocks;'")
print("2. Set DATABASE_URL in docker-compose.yml (already done)")
print("3. Redeploy: docker-compose down && docker-compose up -d --build")
print("4. Keep SQLite volume for 1 week as backup")
