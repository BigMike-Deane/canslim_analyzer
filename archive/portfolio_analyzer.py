#!/usr/bin/env python3
"""
Portfolio Analyzer
Analyzes your current stock positions and provides buy/hold/sell recommendations
based on CANSLIM scores and growth projections.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from data_fetcher import DataFetcher, StockData
from canslim_scorer import CANSLIMScorer, CANSLIMScore
from growth_projector import GrowthProjector, GrowthProjection

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.csv"


@dataclass
class Position:
    """A single stock position"""
    ticker: str
    shares: float
    cost_basis: float  # Average cost per share

    @property
    def total_cost(self) -> float:
        return self.shares * self.cost_basis


@dataclass
class PositionAnalysis:
    """Complete analysis of a position"""
    position: Position
    stock_data: StockData
    canslim_score: CANSLIMScore
    growth_projection: GrowthProjection

    # Calculated fields
    current_value: float = 0.0
    gain_loss: float = 0.0
    gain_loss_pct: float = 0.0
    recommendation: str = "HOLD"
    recommendation_reason: str = ""
    confidence: str = "low"


class PortfolioAnalyzer:
    """Analyzes a portfolio of positions"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.scorer = CANSLIMScorer(self.fetcher)
        self.projector = GrowthProjector(self.fetcher)

    def analyze_position(self, position: Position) -> Optional[PositionAnalysis]:
        """Analyze a single position and generate recommendation"""

        # Fetch stock data
        stock_data = self.fetcher.get_stock_data(position.ticker)
        if not stock_data.is_valid:
            print(f"  Warning: Could not fetch data for {position.ticker}")
            return None

        # Calculate CANSLIM score
        canslim_score = self.scorer.score_stock(stock_data)

        # Project growth
        growth_projection = self.projector.project_growth(stock_data, canslim_score)

        # Create analysis
        analysis = PositionAnalysis(
            position=position,
            stock_data=stock_data,
            canslim_score=canslim_score,
            growth_projection=growth_projection,
        )

        # Calculate gain/loss
        analysis.current_value = position.shares * stock_data.current_price
        analysis.gain_loss = analysis.current_value - position.total_cost
        if position.total_cost > 0:
            analysis.gain_loss_pct = (analysis.gain_loss / position.total_cost) * 100

        # Generate recommendation
        analysis.recommendation, analysis.recommendation_reason, analysis.confidence = \
            self._generate_recommendation(analysis)

        return analysis

    def _generate_recommendation(self, analysis: PositionAnalysis) -> tuple[str, str, str]:
        """
        Generate buy/hold/sell recommendation based on multiple factors.
        Returns: (recommendation, reason, confidence)
        """
        score = analysis.canslim_score.total_score
        projected_growth = analysis.growth_projection.projected_growth_pct
        gain_loss_pct = analysis.gain_loss_pct
        proj_confidence = analysis.growth_projection.confidence

        reasons = []
        buy_signals = 0
        sell_signals = 0
        hold_signals = 0

        # Factor 1: CANSLIM Score
        if score >= 70:
            buy_signals += 2
            reasons.append(f"Strong CANSLIM score ({score:.0f}/100)")
        elif score >= 50:
            hold_signals += 1
            reasons.append(f"Moderate CANSLIM score ({score:.0f}/100)")
        elif score >= 35:
            hold_signals += 1
            reasons.append(f"Below-average CANSLIM score ({score:.0f}/100)")
        else:
            sell_signals += 2
            reasons.append(f"Weak CANSLIM score ({score:.0f}/100)")

        # Factor 2: Projected Growth
        if projected_growth >= 20:
            buy_signals += 2
            reasons.append(f"Strong growth projection ({projected_growth:+.1f}%)")
        elif projected_growth >= 10:
            buy_signals += 1
            reasons.append(f"Positive growth projection ({projected_growth:+.1f}%)")
        elif projected_growth >= 0:
            hold_signals += 1
            reasons.append(f"Flat growth projection ({projected_growth:+.1f}%)")
        elif projected_growth >= -10:
            sell_signals += 1
            reasons.append(f"Negative growth projection ({projected_growth:+.1f}%)")
        else:
            sell_signals += 2
            reasons.append(f"Poor growth projection ({projected_growth:+.1f}%)")

        # Factor 3: Current Gain/Loss (tax and momentum considerations)
        if gain_loss_pct >= 50:
            # Big winner - consider taking some profits
            hold_signals += 1
            reasons.append(f"Large gain ({gain_loss_pct:+.1f}%) - consider partial profit-taking")
        elif gain_loss_pct >= 20:
            hold_signals += 1
            reasons.append(f"Solid gain ({gain_loss_pct:+.1f}%)")
        elif gain_loss_pct >= 0:
            hold_signals += 1
            reasons.append(f"Small gain ({gain_loss_pct:+.1f}%)")
        elif gain_loss_pct >= -10:
            hold_signals += 1
            reasons.append(f"Minor loss ({gain_loss_pct:+.1f}%)")
        elif gain_loss_pct >= -20:
            # Moderate loss - evaluate carefully
            if score < 50 and projected_growth < 5:
                sell_signals += 1
                reasons.append(f"Moderate loss ({gain_loss_pct:+.1f}%) with weak outlook")
            else:
                hold_signals += 1
                reasons.append(f"Moderate loss ({gain_loss_pct:+.1f}%) but outlook okay")
        else:
            # Large loss - cut losses if fundamentals weak
            if score < 40:
                sell_signals += 2
                reasons.append(f"Large loss ({gain_loss_pct:+.1f}%) with weak fundamentals")
            else:
                hold_signals += 1
                reasons.append(f"Large loss ({gain_loss_pct:+.1f}%) but fundamentals intact")

        # Factor 4: Analyst sentiment
        if analysis.growth_projection.analyst_upside >= 30:
            buy_signals += 1
            reasons.append(f"Analysts see {analysis.growth_projection.analyst_upside:.0f}% upside")
        elif analysis.growth_projection.analyst_upside <= -10:
            sell_signals += 1
            reasons.append(f"Analysts see {analysis.growth_projection.analyst_upside:.0f}% downside")

        # Factor 5: Near 52-week high/low
        if analysis.stock_data.current_price > 0 and analysis.stock_data.high_52w > 0:
            pct_from_high = ((analysis.stock_data.high_52w - analysis.stock_data.current_price)
                           / analysis.stock_data.high_52w) * 100
            if pct_from_high <= 5:
                buy_signals += 1
                reasons.append("Near 52-week high (momentum)")
            elif pct_from_high >= 40:
                if score >= 50:
                    buy_signals += 1
                    reasons.append("Well below 52-week high (potential value)")
                else:
                    sell_signals += 1
                    reasons.append("Far from 52-week high with weak score")

        # Make final recommendation
        total_signals = buy_signals + sell_signals + hold_signals

        if sell_signals >= 4 or (sell_signals >= 3 and buy_signals == 0):
            recommendation = "SELL"
        elif buy_signals >= 4 or (buy_signals >= 3 and sell_signals == 0):
            recommendation = "BUY MORE"
        elif sell_signals > buy_signals + 1:
            recommendation = "SELL"
        elif buy_signals > sell_signals + 1:
            recommendation = "BUY MORE"
        else:
            recommendation = "HOLD"

        # Confidence based on signal agreement and data quality
        signal_agreement = max(buy_signals, sell_signals, hold_signals) / max(total_signals, 1)
        if signal_agreement >= 0.6 and proj_confidence == "high":
            confidence = "high"
        elif signal_agreement >= 0.4 or proj_confidence == "medium":
            confidence = "medium"
        else:
            confidence = "low"

        # Build reason string
        top_reasons = reasons[:3]  # Top 3 reasons
        reason_str = "; ".join(top_reasons)

        return recommendation, reason_str, confidence


def load_positions_from_csv(filepath: Path = PORTFOLIO_FILE) -> list[Position]:
    """Load positions from CSV file"""
    if not filepath.exists():
        return []

    positions = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append(Position(
                ticker=row['ticker'].upper(),
                shares=float(row['shares']),
                cost_basis=float(row['cost_basis'])
            ))
    return positions


def parse_positions(position_str: str) -> list[Position]:
    """
    Parse position string into Position objects.
    Format: TICKER:SHARES:COST_BASIS or TICKER:SHARES (cost basis defaults to 0)
    Example: "AAPL:100:150.50,NVDA:50:450,MSFT:25"
    """
    positions = []

    for item in position_str.split(","):
        item = item.strip().upper()
        if not item:
            continue

        parts = item.split(":")
        ticker = parts[0].strip()

        if len(parts) >= 2:
            try:
                shares = float(parts[1].strip())
            except ValueError:
                shares = 0
        else:
            shares = 0

        if len(parts) >= 3:
            try:
                cost_basis = float(parts[2].strip())
            except ValueError:
                cost_basis = 0
        else:
            cost_basis = 0

        if ticker:
            positions.append(Position(ticker=ticker, shares=shares, cost_basis=cost_basis))

    return positions


def print_analysis(analysis: PositionAnalysis):
    """Print detailed analysis for a position"""
    pos = analysis.position
    data = analysis.stock_data
    score = analysis.canslim_score
    proj = analysis.growth_projection

    # Recommendation colors (using text markers)
    rec_marker = {
        "BUY MORE": "[+++]",
        "HOLD": "[===]",
        "SELL": "[---]",
    }.get(analysis.recommendation, "[???]")

    print()
    print(f"{rec_marker} {pos.ticker} - {data.name}")
    print(f"    Recommendation: {analysis.recommendation} ({analysis.confidence} confidence)")
    print(f"    Reason: {analysis.recommendation_reason}")
    print()
    print(f"    Position: {pos.shares:.2f} shares @ ${pos.cost_basis:.2f} avg cost")
    print(f"    Current Price: ${data.current_price:.2f}")
    print(f"    Current Value: ${analysis.current_value:,.2f}")
    print(f"    Gain/Loss: ${analysis.gain_loss:+,.2f} ({analysis.gain_loss_pct:+.1f}%)")
    print()
    print(f"    CANSLIM Score: {score.total_score:.0f}/100")
    print(f"    6-Month Projection: {proj.projected_growth_pct:+.1f}%")

    if proj.analyst_target > 0:
        print(f"    Analyst Target: ${proj.analyst_target:.2f} ({proj.analyst_upside:+.1f}% upside)")

    print()
    print(f"    Score Breakdown: C:{score.c_score:.0f} A:{score.a_score:.0f} N:{score.n_score:.0f} "
          f"S:{score.s_score:.0f} L:{score.l_score:.0f} I:{score.i_score:.0f} M:{score.m_score:.0f}")
    print("-" * 70)


def print_portfolio_summary(analyses: list[PositionAnalysis]):
    """Print portfolio summary"""
    total_cost = sum(a.position.total_cost for a in analyses)
    total_value = sum(a.current_value for a in analyses)
    total_gain = total_value - total_cost
    total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0

    buy_count = sum(1 for a in analyses if a.recommendation == "BUY MORE")
    hold_count = sum(1 for a in analyses if a.recommendation == "HOLD")
    sell_count = sum(1 for a in analyses if a.recommendation == "SELL")

    print()
    print("=" * 70)
    print("                    PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"  Total Positions: {len(analyses)}")
    print(f"  Total Cost Basis: ${total_cost:,.2f}")
    print(f"  Current Value: ${total_value:,.2f}")
    print(f"  Total Gain/Loss: ${total_gain:+,.2f} ({total_gain_pct:+.1f}%)")
    print()
    print(f"  Recommendations: {buy_count} BUY MORE | {hold_count} HOLD | {sell_count} SELL")
    print("=" * 70)

    # Show action items
    if sell_count > 0:
        print()
        print("  ACTION ITEMS - Consider Selling:")
        for a in analyses:
            if a.recommendation == "SELL":
                print(f"    - {a.position.ticker}: {a.recommendation_reason[:50]}...")

    if buy_count > 0:
        print()
        print("  ACTION ITEMS - Consider Buying More:")
        for a in analyses:
            if a.recommendation == "BUY MORE":
                print(f"    + {a.position.ticker}: {a.recommendation_reason[:50]}...")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze your stock portfolio with CANSLIM method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from portfolio.csv (default)
  python portfolio_analyzer.py

  # Specify positions directly
  python portfolio_analyzer.py -p "AAPL:100:150.50,NVDA:50:450.00"

  # Load from a different CSV file
  python portfolio_analyzer.py --file my_portfolio.csv

  # Interactive mode
  python portfolio_analyzer.py --interactive
        """
    )

    parser.add_argument(
        "-p", "--positions",
        type=str,
        help="Positions as TICKER:SHARES:COST_BASIS (comma-separated)"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Load positions from CSV file (default: portfolio.csv)"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode - enter positions one by one"
    )

    args = parser.parse_args()

    positions = []

    if args.interactive:
        print("\nEnter your positions (type 'done' when finished)")
        print("Format: TICKER SHARES COST_BASIS (e.g., AAPL 100 150.50)")
        print()

        while True:
            try:
                line = input("Position> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if line.lower() == 'done' or not line:
                if positions:
                    break
                print("Please enter at least one position.")
                continue

            parts = line.split()
            ticker = parts[0].upper()
            shares = float(parts[1]) if len(parts) > 1 else 0
            cost_basis = float(parts[2]) if len(parts) > 2 else 0

            positions.append(Position(ticker=ticker, shares=shares, cost_basis=cost_basis))
            print(f"  Added: {ticker} - {shares} shares @ ${cost_basis:.2f}")

    elif args.positions:
        positions = parse_positions(args.positions)

    elif args.file:
        filepath = Path(args.file)
        positions = load_positions_from_csv(filepath)
        if not positions:
            print(f"No positions found in {filepath}")
            return

    else:
        # Default: load from portfolio.csv
        positions = load_positions_from_csv()
        if not positions:
            print(f"No positions found in {PORTFOLIO_FILE}")
            print("Use -p to specify positions or create portfolio.csv")
            return

    if not positions:
        print("No valid positions to analyze.")
        return

    print()
    print("=" * 70)
    print("           CANSLIM PORTFOLIO ANALYZER")
    print("=" * 70)
    print(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Positions to Analyze: {len(positions)}")
    print("=" * 70)

    # Get market direction
    analyzer = PortfolioAnalyzer()
    is_bullish, pct_200, pct_50 = analyzer.fetcher.get_market_direction()
    market_status = "BULLISH" if is_bullish else "BEARISH"
    print(f"  Market Direction: {market_status} (S&P 500 {pct_200:+.1f}% vs 200-day MA)")

    if not is_bullish:
        print("  *** WARNING: Bearish market - be cautious with new buys ***")

    print("=" * 70)

    # Analyze each position
    analyses = []
    for pos in positions:
        print(f"\nAnalyzing {pos.ticker}...", end="", flush=True)
        analysis = analyzer.analyze_position(pos)
        if analysis:
            analyses.append(analysis)
            print(" done")
        else:
            print(" failed")

    if not analyses:
        print("\nNo positions could be analyzed.")
        return

    # Sort by recommendation priority (SELL first, then HOLD, then BUY)
    rec_order = {"SELL": 0, "HOLD": 1, "BUY MORE": 2}
    analyses.sort(key=lambda a: (rec_order.get(a.recommendation, 1), -a.canslim_score.total_score))

    # Print individual analyses
    print("\n" + "=" * 70)
    print("                 POSITION ANALYSIS")
    print("=" * 70)

    for analysis in analyses:
        print_analysis(analysis)

    # Print summary
    print_portfolio_summary(analyses)

    print("\nDISCLAIMER: This is for educational purposes only. Not financial advice.")
    print("Always do your own research before making investment decisions.\n")


if __name__ == "__main__":
    main()
