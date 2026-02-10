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
            "SELECT * FROM ai_portfolio_trades ORDER BY executed_at DESC",
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


@mcp.tool()
def get_score_history(ticker: str, days: int = 30) -> str:
    """Get CANSLIM score history for a ticker over time.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL')
        days: Number of days of history (default 30)
    """
    days = min(days, 365)
    try:
        rows = _query("""
            SELECT ss.date, ss.timestamp, ss.total_score,
                   ss.c_score, ss.a_score, ss.n_score, ss.s_score,
                   ss.l_score, ss.i_score, ss.m_score,
                   ss.current_price, ss.projected_growth
            FROM stock_scores ss
            JOIN stocks s ON ss.stock_id = s.id
            WHERE s.ticker = %s
              AND ss.date >= CURRENT_DATE - make_interval(days => %s)
            ORDER BY ss.date DESC, ss.timestamp DESC
        """, (ticker.upper(), days), limit=200)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_sector_breakdown() -> str:
    """Get sector breakdown for portfolio positions and stock universe."""
    try:
        portfolio = _query("""
            SELECT s.sector, COUNT(*) as positions,
                   SUM(p.current_value) as total_value
            FROM ai_portfolio_positions p
            LEFT JOIN stocks s ON p.ticker = s.ticker
            GROUP BY s.sector
            ORDER BY total_value DESC
        """, limit=50)

        universe = _query("""
            SELECT sector, COUNT(*) as stock_count,
                   ROUND(AVG(canslim_score)::numeric, 1) as avg_score,
                   SUM(CASE WHEN is_breaking_out = true THEN 1 ELSE 0 END) as breakouts
            FROM stocks
            WHERE sector IS NOT NULL
            GROUP BY sector
            ORDER BY avg_score DESC
        """, limit=50)

        return json.dumps({
            "portfolio_sectors": portfolio,
            "universe_sectors": universe,
        }, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_backtest_trades(backtest_id: int) -> str:
    """Get all trades from a specific backtest run.

    Args:
        backtest_id: ID of the backtest run
    """
    try:
        rows = _query("""
            SELECT date, ticker, action, shares, price, total_value,
                   reason, canslim_score, cost_basis, realized_gain,
                   realized_gain_pct, holding_days, signal_factors
            FROM backtest_trades
            WHERE backtest_id = %s
            ORDER BY date, id
        """, (backtest_id,), limit=500)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_portfolio_snapshots(days: int = 30) -> str:
    """Get AI portfolio value snapshots over time.

    Args:
        days: Number of days of history (default 30)
    """
    days = min(days, 365)
    try:
        rows = _query("""
            SELECT timestamp, total_value, cash, positions_value,
                   positions_count, total_return_pct, value_change_pct
            FROM ai_portfolio_snapshots
            WHERE timestamp >= NOW() - make_interval(days => %s)
            ORDER BY timestamp DESC
        """, (days,), limit=500)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_scan_freshness() -> str:
    """Get data freshness info: last scan times, stock counts, and update coverage."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            result = {}

            cur.execute("SELECT MAX(last_updated) as last_stock_scan FROM stocks")
            result["last_stock_scan"] = cur.fetchone()["last_stock_scan"]

            cur.execute("SELECT MAX(timestamp) as last_score FROM stock_scores")
            result["last_score_recorded"] = cur.fetchone()["last_score"]

            cur.execute("SELECT COUNT(*) as total FROM stocks")
            result["total_stocks"] = cur.fetchone()["total"]

            cur.execute("""
                SELECT COUNT(*) as recent
                FROM stocks
                WHERE last_updated >= NOW() - INTERVAL '24 hours'
            """)
            result["stocks_updated_24h"] = cur.fetchone()["recent"]

            cur.execute("SELECT COUNT(*) as cnt FROM ai_portfolio_positions")
            result["portfolio_positions"] = cur.fetchone()["cnt"]

            cur.execute("SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT 1")
            row = cur.fetchone()
            result["latest_market_snapshot"] = dict(row) if row else None

        import datetime
        from decimal import Decimal

        def serialize(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            return str(obj)

        return json.dumps(result, indent=2, default=serialize)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


@mcp.tool()
def get_coiled_spring_alerts(limit: int = 20) -> str:
    """Get recent coiled spring (earnings catalyst) alerts.

    Args:
        limit: Number of alerts to return (default 20)
    """
    limit = min(limit, 100)
    try:
        rows = _query("""
            SELECT ticker, alert_date, days_to_earnings, weeks_in_base,
                   beat_streak, total_score, cs_bonus, price_at_alert,
                   base_type, outcome, price_change_pct
            FROM coiled_spring_alerts
            ORDER BY alert_date DESC
        """, limit=limit)
        return json.dumps(rows, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_stock_comparison(ticker1: str, ticker2: str) -> str:
    """Compare two stocks side by side on key metrics.

    Args:
        ticker1: First stock ticker
        ticker2: Second stock ticker
    """
    try:
        rows = _query("""
            SELECT ticker, canslim_score, current_price, market_cap, sector,
                   c_score, a_score, n_score, s_score, l_score, i_score, m_score,
                   projected_growth, growth_confidence, volume_ratio,
                   is_breaking_out, base_type, weeks_in_base,
                   rs_12m, rs_3m, short_interest_pct, insider_sentiment,
                   earnings_beat_streak, days_to_earnings,
                   is_growth_stock, growth_mode_score
            FROM stocks
            WHERE ticker IN (%s, %s)
        """, (ticker1.upper(), ticker2.upper()), limit=2)

        result = {}
        for r in rows:
            key = "stock_1" if r["ticker"] == ticker1.upper() else "stock_2"
            result[key] = r

        if len(result) < 2:
            missing = [t for t in [ticker1, ticker2]
                       if t.upper() not in [r.get("ticker") for r in rows]]
            return f"Not found: {', '.join(missing)}"

        return json.dumps(result, indent=2, default=_serialize)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_system_health() -> str:
    """Get comprehensive system health: scan freshness, portfolio summary, recent activity."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            result = {}

            # Scan freshness
            cur.execute("SELECT MAX(last_updated) as last_scan FROM stocks")
            result["last_scan"] = cur.fetchone()["last_scan"]

            cur.execute("""
                SELECT COUNT(*) as stale_1h FROM stocks
                WHERE last_updated < NOW() - INTERVAL '1 hour'
            """)
            result["stocks_stale_1h"] = cur.fetchone()["stale_1h"]

            cur.execute("""
                SELECT COUNT(*) as stale_24h FROM stocks
                WHERE last_updated < NOW() - INTERVAL '24 hours'
            """)
            result["stocks_stale_24h"] = cur.fetchone()["stale_24h"]

            cur.execute("SELECT COUNT(*) as total FROM stocks")
            result["total_stocks"] = cur.fetchone()["total"]

            # Portfolio summary
            cur.execute("""
                SELECT COUNT(*) as positions,
                       COALESCE(SUM(current_value), 0) as total_value,
                       COALESCE(SUM(unrealized_gain), 0) as total_pnl
                FROM ai_portfolio_positions
            """)
            result["portfolio"] = dict(cur.fetchone())

            # Recent trades (7 days)
            cur.execute("""
                SELECT
                    SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buys,
                    SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sells
                FROM ai_portfolio_trades
                WHERE executed_at >= NOW() - INTERVAL '7 days'
            """)
            result["trades_7d"] = dict(cur.fetchone())

            # Backtest status
            cur.execute("""
                SELECT status, COUNT(*) as cnt
                FROM backtest_runs
                GROUP BY status
            """)
            result["backtests"] = {r["status"]: r["cnt"] for r in cur.fetchall()}

            # Score records
            cur.execute("SELECT COUNT(*) as cnt FROM stock_scores")
            result["score_records"] = cur.fetchone()["cnt"]

            # Market signal
            cur.execute("""
                SELECT weighted_signal, timestamp
                FROM market_snapshots
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cur.fetchone()
            result["market_signal"] = dict(row) if row else None

            # Delisted tickers
            cur.execute("SELECT COUNT(*) as cnt FROM delisted_tickers")
            result["delisted_tickers"] = cur.fetchone()["cnt"]

            cur.execute("SELECT NOW() as server_time")
            result["server_time"] = cur.fetchone()["server_time"]

        import datetime
        from decimal import Decimal

        def serialize(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return float(obj)
            return str(obj)

        return json.dumps(result, indent=2, default=serialize)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
