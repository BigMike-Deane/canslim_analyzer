#!/usr/bin/env python3
"""
CANSLIM Stock Analyzer
Analyzes S&P 500 stocks using William O'Neil's CANSLIM method
and identifies top stocks with highest projected 6-month growth.
"""

import argparse
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm

from sp500_tickers import get_sp500_tickers, get_russell2000_tickers, get_all_tickers
from data_fetcher import DataFetcher, StockData
from canslim_scorer import CANSLIMScorer, CANSLIMScore
from growth_projector import GrowthProjector, GrowthProjection


@dataclass
class AnalysisResult:
    """Complete analysis result for a stock"""
    ticker: str
    name: str
    sector: str
    current_price: float
    canslim_score: CANSLIMScore
    growth_projection: GrowthProjection
    stock_data: StockData


def print_header(is_bullish: bool, pct_above_200: float, stocks_analyzed: int):
    """Print the report header"""
    print()
    print("=" * 70)
    print("           CANSLIM STOCK ANALYZER - TOP 5 PICKS")
    print("=" * 70)

    market_status = "BULLISH" if is_bullish else "BEARISH"
    market_color = "" if is_bullish else ""
    print(f"  Market Direction: {market_status} (S&P 500 {pct_above_200:+.1f}% vs 200-day MA)")
    print(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Stocks Analyzed: {stocks_analyzed}")
    print("=" * 70)


def print_stock_result(rank: int, result: AnalysisResult):
    """Print detailed result for a single stock"""
    score = result.canslim_score
    proj = result.growth_projection
    data = result.stock_data

    print()
    print(f"#{rank} {result.ticker} - {result.name}")
    print(f"   Sector: {result.sector}")
    print(f"   Current Price: ${result.current_price:.2f}")

    # Show analyst target if available
    if proj.analyst_target > 0:
        print(f"   Analyst Target: ${proj.analyst_target:.2f} ({proj.analyst_upside:+.1f}% upside) [{proj.num_analysts} analysts]")
        if data.analyst_recommendation:
            print(f"   Consensus: {data.analyst_recommendation.upper()}")

    print(f"   CANSLIM Score: {score.total_score:.0f}/100")
    print(f"   Projected 6-Month Growth: {proj.projected_growth_pct:+.1f}% ({proj.confidence} confidence)")
    print()
    print("   CANSLIM Score Breakdown:")
    print(f"   +-- C (Current Earnings):    {score.c_score:>5.1f}/15  ({score.c_detail})")
    print(f"   +-- A (Annual Earnings):     {score.a_score:>5.1f}/15  ({score.a_detail})")
    print(f"   +-- N (New Highs):           {score.n_score:>5.1f}/15  ({score.n_detail})")
    print(f"   +-- S (Supply/Demand):       {score.s_score:>5.1f}/15  ({score.s_detail})")
    print(f"   +-- L (Leader):              {score.l_score:>5.1f}/15  ({score.l_detail})")
    print(f"   +-- I (Institutional):       {score.i_score:>5.1f}/10  ({score.i_detail})")
    print(f"   +-- M (Market):              {score.m_score:>5.1f}/15  ({score.m_detail})")
    print()
    print("   Growth Projection Components:")
    print(f"   +-- Analyst Target (30%):    {proj.analyst_projection:+.1f}%")
    print(f"   +-- Momentum (25%):          {proj.momentum_projection:+.1f}%")
    print(f"   +-- Earnings (20%):          {proj.earnings_projection:+.1f}%")
    print(f"   +-- CANSLIM Factor (15%):    {proj.canslim_factor:+.1f}%")
    print(f"   +-- Sector Bonus (10%):      {proj.sector_bonus:+.1f}%")
    print("-" * 70)


def print_footer():
    """Print the report footer with disclaimer"""
    print()
    print("=" * 70)
    print("                           DISCLAIMER")
    print("  This analysis is for educational and informational purposes only.")
    print("  It is NOT financial advice. Past performance does not guarantee")
    print("  future results. Always do your own research before investing.")
    print("=" * 70)
    print()


def analyze_stocks(tickers: list[str], top_n: int = 5, min_price: float = 0, max_price: float = float('inf'), sector_filter: str = None) -> list[AnalysisResult]:
    """
    Analyze all tickers and return top N by projected growth.
    Supports price and sector filtering.
    """
    fetcher = DataFetcher()
    scorer = CANSLIMScorer(fetcher)
    projector = GrowthProjector(fetcher)

    results: list[AnalysisResult] = []
    skipped = 0
    price_filtered = 0
    sector_filtered = 0

    print("\nFetching stock data and calculating scores...")
    if min_price > 0 or max_price < float('inf'):
        print(f"Price filter: ${min_price:.2f} - ${max_price:.2f}")
    if sector_filter:
        print(f"Sector filter: {sector_filter}")
    print("(This may take several minutes for the full S&P 500)\n")

    for ticker in tqdm(tickers, desc="Analyzing stocks", unit="stock"):
        try:
            # Fetch data
            stock_data = fetcher.get_stock_data(ticker)

            if not stock_data.is_valid:
                skipped += 1
                continue

            # Apply price filter
            if stock_data.current_price < min_price or stock_data.current_price > max_price:
                price_filtered += 1
                continue

            # Apply sector filter
            if sector_filter and sector_filter.lower() not in stock_data.sector.lower():
                sector_filtered += 1
                continue

            # Calculate CANSLIM score
            canslim_score = scorer.score_stock(stock_data)

            # Skip stocks with very low CANSLIM scores
            if canslim_score.total_score < 30:
                skipped += 1
                continue

            # Project growth
            growth_projection = projector.project_growth(stock_data, canslim_score)

            # Create result
            result = AnalysisResult(
                ticker=ticker,
                name=stock_data.name,
                sector=stock_data.sector,
                current_price=stock_data.current_price,
                canslim_score=canslim_score,
                growth_projection=growth_projection,
                stock_data=stock_data,
            )
            results.append(result)

        except Exception as e:
            skipped += 1
            continue

    print(f"\nAnalyzed {len(results)} stocks successfully, skipped {skipped}, price filtered {price_filtered}, sector filtered {sector_filtered}")

    # Sort by projected growth (descending)
    results.sort(key=lambda r: r.growth_projection.projected_growth_pct, reverse=True)

    return results[:top_n]


def main():
    parser = argparse.ArgumentParser(
        description="CANSLIM Stock Analyzer - Find top growth stocks using O'Neil's method"
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=5,
        help="Number of top stocks to display (default: 5)"
    )
    parser.add_argument(
        "-t", "--tickers",
        type=str,
        nargs="+",
        help="Specific tickers to analyze (overrides S&P 500 list)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: analyze only top 50 S&P 500 stocks by market cap"
    )
    parser.add_argument(
        "--russell",
        action="store_true",
        help="Include Russell 2000 small-cap stocks"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="include_all",
        help="Include both S&P 500 and Russell 2000 stocks"
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0,
        help="Minimum stock price filter (default: 0)"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=float('inf'),
        help="Maximum stock price filter (default: no limit)"
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Filter by sector (e.g., 'Technology', 'Healthcare', 'Financials')"
    )

    args = parser.parse_args()

    # Get tickers to analyze
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"Analyzing {len(tickers)} specified tickers...")
    elif args.quick:
        # Quick mode: just analyze major stocks
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "PFE", "TMO", "MCD",
            "CSCO", "WMT", "ACN", "ABT", "DHR", "CRM", "VZ", "ADBE", "CMCSA",
            "NKE", "NFLX", "INTC", "TXN", "AMD", "QCOM", "HON", "UNP", "PM",
            "LOW", "NEE", "UPS", "RTX", "BA"
        ]
        print(f"Quick mode: analyzing top 50 stocks...")
    elif args.include_all:
        tickers = get_all_tickers()
        print(f"Fetched {len(tickers)} tickers (S&P 500 + Russell 2000)")
    elif args.russell:
        tickers = get_russell2000_tickers()
        print(f"Fetched {len(tickers)} Russell 2000 small-cap tickers")
    else:
        tickers = get_sp500_tickers()
        print(f"Fetched {len(tickers)} S&P 500 tickers")

    # Initialize fetcher to check market direction
    fetcher = DataFetcher()
    is_bullish, pct_above_200, pct_above_50 = fetcher.get_market_direction()

    if not is_bullish:
        print("\n*** WARNING: Market appears bearish (below 200-day MA) ***")
        print("*** CANSLIM strategy works best in bullish markets ***\n")

    # Analyze stocks
    top_results = analyze_stocks(tickers, top_n=args.top, min_price=args.min_price, max_price=args.max_price, sector_filter=args.sector)

    if not top_results:
        print("\nNo stocks passed the CANSLIM criteria. Try again later or check data sources.")
        return

    # Print results
    print_header(is_bullish, pct_above_200, len(tickers))

    for rank, result in enumerate(top_results, 1):
        print_stock_result(rank, result)

    print_footer()


if __name__ == "__main__":
    main()
