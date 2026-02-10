#!/usr/bin/env python3
"""FMP (Financial Modeling Prep) MCP Server for CANSLIM Analyzer.

Provides Claude Code with direct access to FMP financial data APIs.
"""

import os
import json
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"

mcp = FastMCP("fmp-financial-data")


async def _fmp_get(path: str, params: dict[str, Any] | None = None) -> dict | list | None:
    """Make a GET request to FMP API."""
    params = params or {}
    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE}/{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def get_quote(symbol: str) -> str:
    """Get real-time stock quote including price, volume, market cap, day change, and 52-week range."""
    data = await _fmp_get("quote", {"symbol": symbol})
    if not data:
        return f"No quote data found for {symbol}"
    return json.dumps(data[0] if isinstance(data, list) else data, indent=2)


@mcp.tool()
async def get_company_profile(symbol: str) -> str:
    """Get company profile: sector, industry, description, market cap, beta, CEO, employees, website."""
    data = await _fmp_get("profile", {"symbol": symbol})
    if not data:
        return f"No profile found for {symbol}"
    item = data[0] if isinstance(data, list) else data
    keys = ["companyName", "sector", "industry", "mktCap", "beta", "ceo",
            "fullTimeEmployees", "website", "description", "country", "exchange"]
    return json.dumps({k: item.get(k) for k in keys if k in item}, indent=2)


@mcp.tool()
async def get_earnings(symbol: str, period: str = "quarterly", limit: int = 8) -> str:
    """Get earnings history (EPS actual vs estimate, revenue, surprise %).

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 8)
    """
    data = await _fmp_get("earnings", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No earnings data for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_income_statement(symbol: str, period: str = "quarterly", limit: int = 8) -> str:
    """Get income statement: revenue, net income, EPS, operating income, gross profit margin.

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 8)
    """
    data = await _fmp_get("income-statement", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No income statement data for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_key_metrics(symbol: str, period: str = "quarterly", limit: int = 4) -> str:
    """Get key financial metrics: ROE, P/E, PEG, debt/equity, free cash flow per share, book value.

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 4)
    """
    data = await _fmp_get("key-metrics", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No key metrics for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_analyst_estimates(symbol: str, period: str = "quarterly", limit: int = 4) -> str:
    """Get analyst consensus estimates: EPS and revenue estimates for upcoming quarters.

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 4)
    """
    data = await _fmp_get("analyst-estimates", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No analyst estimates for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_price_target(symbol: str) -> str:
    """Get analyst price target consensus: average, high, low targets and number of analysts."""
    data = await _fmp_get("price-target-consensus", {"symbol": symbol})
    if not data:
        return f"No price target data for {symbol}"
    return json.dumps(data[0] if isinstance(data, list) else data, indent=2)


@mcp.tool()
async def get_insider_trading(symbol: str, limit: int = 20) -> str:
    """Get recent insider trading activity: buys, sells, transaction amounts.

    Args:
        symbol: Stock ticker
        limit: Number of transactions (default 20)
    """
    data = await _fmp_get("insider-trading", {"symbol": symbol, "limit": limit})
    if not data:
        return f"No insider trading data for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_earnings_calendar(from_date: str = "", to_date: str = "") -> str:
    """Get upcoming earnings dates for all companies.

    Args:
        from_date: Start date (YYYY-MM-DD), defaults to today
        to_date: End date (YYYY-MM-DD), defaults to 2 weeks out
    """
    params = {}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    data = await _fmp_get("earnings-calendar", params)
    if not data:
        return "No upcoming earnings found"
    return json.dumps(data[:50] if isinstance(data, list) else data, indent=2)


@mcp.tool()
async def search_ticker(query: str, limit: int = 10) -> str:
    """Search for stock tickers by company name or symbol.

    Args:
        query: Search term (company name or partial ticker)
        limit: Max results (default 10)
    """
    data = await _fmp_get("search", {"query": query, "limit": limit})
    if not data:
        return f"No results for '{query}'"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_stock_screener(
    market_cap_min: int = 0,
    market_cap_max: int = 0,
    sector: str = "",
    country: str = "US",
    limit: int = 20,
) -> str:
    """Screen stocks by market cap, sector, and country.

    Args:
        market_cap_min: Minimum market cap (e.g. 1000000000 for $1B)
        market_cap_max: Maximum market cap (0 = no limit)
        sector: Filter by sector (e.g. 'Technology', 'Healthcare')
        country: Country code (default 'US')
        limit: Max results (default 20)
    """
    params = {"country": country, "limit": limit}
    if market_cap_min:
        params["marketCapMoreThan"] = market_cap_min
    if market_cap_max:
        params["marketCapLowerThan"] = market_cap_max
    if sector:
        params["sector"] = sector
    data = await _fmp_get("stock-screener", params)
    if not data:
        return "No stocks match the criteria"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_balance_sheet(symbol: str, period: str = "quarterly", limit: int = 4) -> str:
    """Get balance sheet: total assets, liabilities, equity, cash, debt.

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 4)
    """
    data = await _fmp_get("balance-sheet-statement", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No balance sheet data for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_cash_flow(symbol: str, period: str = "quarterly", limit: int = 4) -> str:
    """Get cash flow statement: operating, investing, financing cash flows, free cash flow.

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 4)
    """
    data = await _fmp_get("cash-flow-statement", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No cash flow data for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_financial_ratios(symbol: str, period: str = "quarterly", limit: int = 4) -> str:
    """Get financial ratios: profitability, liquidity, leverage, efficiency ratios.

    Args:
        symbol: Stock ticker
        period: 'quarterly' or 'annual'
        limit: Number of periods (default 4)
    """
    data = await _fmp_get("ratios", {"symbol": symbol, "period": period, "limit": limit})
    if not data:
        return f"No financial ratios for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_market_movers() -> str:
    """Get today's biggest gainers, losers, and most active stocks."""
    gainers = await _fmp_get("biggest-gainers") or []
    losers = await _fmp_get("biggest-losers") or []
    most_active = await _fmp_get("most-active") or []
    return json.dumps({
        "gainers": (gainers[:10] if isinstance(gainers, list) else []),
        "losers": (losers[:10] if isinstance(losers, list) else []),
        "most_active": (most_active[:10] if isinstance(most_active, list) else []),
    }, indent=2)


@mcp.tool()
async def get_sector_performance() -> str:
    """Get sector performance snapshot: daily change % for each sector."""
    data = await _fmp_get("sector-performance-snapshot")
    if not data:
        return "No sector performance data"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


@mcp.tool()
async def get_economic_calendar(from_date: str = "", to_date: str = "") -> str:
    """Get upcoming economic events: GDP, CPI, jobs, Fed decisions.

    Args:
        from_date: Start date (YYYY-MM-DD), defaults to today
        to_date: End date (YYYY-MM-DD), defaults to 2 weeks out
    """
    params = {}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    data = await _fmp_get("economic-calendar", params)
    if not data:
        return "No upcoming economic events"
    return json.dumps(data[:50] if isinstance(data, list) else data, indent=2)


@mcp.tool()
async def get_historical_price(symbol: str, from_date: str = "", to_date: str = "") -> str:
    """Get historical daily price data (OHLCV) for a stock.

    Args:
        symbol: Stock ticker
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
    """
    params = {"symbol": symbol}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    data = await _fmp_get("historical-price-eod/light", params)
    if not data:
        return f"No historical price data for {symbol}"
    return json.dumps(data if isinstance(data, list) else [data], indent=2)


if __name__ == "__main__":
    mcp.run()
