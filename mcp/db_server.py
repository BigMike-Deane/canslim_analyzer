#!/usr/bin/env python3
"""PostgreSQL MCP Server for CANSLIM Analyzer.

Provides Claude Code with direct read-only access to the CANSLIM database.
"""

import os
import json

import psycopg2
import psycopg2.extras
from mcp.server.fastmcp import FastMCP

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://canslim:canslim_secure_pw@100.104.189.36:5432/canslim",
)

mcp = FastMCP("canslim-database")


def _get_conn():
    return psycopg2.connect(DB_DSN)


def _query(sql: str, params: tuple = (), limit: int = 100) -> list[dict]:
    """Execute a read-only query and return results as list of dicts."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchmany(limit)
            return [dict(r) for r in rows]
    finally:
        conn.close()


def _serialize(obj):
    """JSON-serialize with datetime/Decimal handling."""
    import datetime
    from decimal import Decimal

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


@mcp.tool()
def execute_sql(query: str, limit: int = 50) -> str:
    """Execute a read-only SQL query against the CANSLIM database.

    Use SELECT queries only. The database contains stock data, portfolio positions,
    backtest results, trade history, and CANSLIM scores.

    Args:
        query: SQL SELECT query to execute
        limit: Max rows to return (default 50, max 500)
    """
    q = query.strip().rstrip(";")
    if not q.upper().startswith("SELECT") and not q.upper().startswith("WITH"):
        return "Error: Only SELECT queries are allowed for safety."
    limit = min(limit, 500)
    try:
        rows = _query(q, limit=limit)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Query error: {e}"


@mcp.tool()
def list_tables() -> str:
    """List all tables in the CANSLIM database with row counts."""
    try:
        rows = _query("""
            SELECT schemaname, tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
        """, limit=100)
        result = []
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                for r in rows:
                    table = r["tablename"]
                    cur.execute(f'SELECT count(*) FROM "{table}"')
                    count = cur.fetchone()[0]
                    result.append({"table": table, "rows": count})
        finally:
            conn.close()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get column names, types, and constraints for a table.

    Args:
        table_name: Name of the table to describe
    """
    try:
        rows = _query("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,), limit=200)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_stock(ticker: str) -> str:
    """Get full stock data for a ticker including CANSLIM scores and technical data.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
    """
    try:
        rows = _query(
            "SELECT * FROM stocks WHERE ticker = %s",
            (ticker.upper(),),
            limit=1,
        )
        if not rows:
            return f"No stock found for {ticker}"
        return json.dumps(rows[0], indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_portfolio_positions() -> str:
    """Get all current AI portfolio positions with P&L."""
    try:
        rows = _query("""
            SELECT p.*, s.canslim_score, s.current_price, s.sector
            FROM ai_portfolio_positions p
            LEFT JOIN stocks s ON p.ticker = s.ticker
            ORDER BY p.purchase_date DESC
        """, limit=50)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_recent_trades(limit: int = 20) -> str:
    """Get recent AI portfolio trades.

    Args:
        limit: Number of trades to return (default 20)
    """
    limit = min(limit, 100)
    try:
        rows = _query(
            "SELECT * FROM ai_portfolio_trades ORDER BY trade_date DESC",
            limit=limit,
        )
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_backtest_results(limit: int = 10) -> str:
    """Get recent backtest run summaries.

    Args:
        limit: Number of backtests to return (default 10)
    """
    limit = min(limit, 50)
    try:
        rows = _query(
            "SELECT * FROM backtest_runs ORDER BY id DESC",
            limit=limit,
        )
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_top_stocks(min_score: int = 70, limit: int = 20) -> str:
    """Get top-scoring stocks by CANSLIM score.

    Args:
        min_score: Minimum CANSLIM score (default 70)
        limit: Max results (default 20)
    """
    limit = min(limit, 100)
    try:
        rows = _query(
            """SELECT ticker, canslim_score, current_price, sector, market_cap,
                      c_score, a_score, n_score, s_score, l_score, i_score, m_score,
                      is_breaking_out, base_type, volume_ratio
               FROM stocks
               WHERE canslim_score >= %s
               ORDER BY canslim_score DESC""",
            (min_score,),
            limit=limit,
        )
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
