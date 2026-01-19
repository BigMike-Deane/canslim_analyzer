#!/usr/bin/env python3
"""
CANSLIM Daily Email Report
Runs two analyses and sends combined results via email:
1. Top 10 stocks under $20 (S&P 500 + Russell 2000)
2. Top 5 stocks from S&P 500 (no price filter)
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from io import StringIO
import sys
from pathlib import Path

# Load .env file if it exists
def load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Import our modules
from sp500_tickers import get_sp500_tickers, get_russell2000_tickers, get_all_tickers
from data_fetcher import DataFetcher, StockData
from canslim_scorer import CANSLIMScorer, CANSLIMScore
from growth_projector import GrowthProjector, GrowthProjection
from main import analyze_stocks, AnalysisResult
from portfolio_analyzer import (
    PortfolioAnalyzer, Position, PositionAnalysis, load_positions_from_csv
)
from score_history import record_score, get_score_delta, get_biggest_movers


# ============ CONFIGURATION ============
GMAIL_ADDRESS = os.environ.get('CANSLIM_EMAIL', 'your-email@gmail.com')
GMAIL_APP_PASSWORD = os.environ.get('CANSLIM_APP_PASSWORD', 'your-app-password')
RECIPIENT_EMAIL = os.environ.get('CANSLIM_RECIPIENT', GMAIL_ADDRESS)
# =======================================


def generate_stock_card_html(rank: int, result: AnalysisResult) -> str:
    """Generate HTML for a single stock card"""
    score = result.canslim_score
    proj = result.growth_projection
    growth_class = "" if proj.projected_growth_pct >= 0 else "negative"

    html = f"""
    <div class="stock-card">
        <div class="stock-header">
            <div>
                <span class="ticker">#{rank} {result.ticker}</span>
                <span style="color: #666;"> - {result.name}</span>
            </div>
            <span class="score">{score.total_score:.0f}/100</span>
        </div>

        <div class="details">
            <p><strong>Sector:</strong> {result.sector} | <strong>Price:</strong> ${result.current_price:.2f}</p>
    """

    if proj.analyst_target > 0:
        html += f"""
            <p><strong>Analyst Target:</strong> ${proj.analyst_target:.2f} ({proj.analyst_upside:+.1f}% upside) [{proj.num_analysts} analysts]</p>
        """

    html += f"""
            <p class="growth {growth_class}">Projected 6-Month Growth: {proj.projected_growth_pct:+.1f}% ({proj.confidence} confidence)</p>
        </div>

        <div class="breakdown">
            C: {score.c_score:.1f}/15 | A: {score.a_score:.1f}/15 | N: {score.n_score:.1f}/15 | S: {score.s_score:.1f}/15 | L: {score.l_score:.1f}/15 | I: {score.i_score:.1f}/10 | M: {score.m_score:.1f}/15
        </div>
    </div>
    """
    return html


def generate_portfolio_card_html(analysis: PositionAnalysis, score_delta: dict = None) -> str:
    """Generate HTML for a single portfolio position with score delta"""
    pos = analysis.position
    score = analysis.canslim_score
    proj = analysis.growth_projection

    # Color coding for recommendation
    rec_colors = {
        "BUY MORE": ("#28a745", "#d4edda"),  # green
        "HOLD": ("#6c757d", "#e9ecef"),       # gray
        "SELL": ("#dc3545", "#f8d7da"),       # red
    }
    text_color, bg_color = rec_colors.get(analysis.recommendation, ("#6c757d", "#e9ecef"))

    gain_color = "#28a745" if analysis.gain_loss_pct >= 0 else "#dc3545"

    # Score delta display
    delta_html = ""
    if score_delta and score_delta.get('has_history'):
        delta = score_delta['total_delta']
        trend = score_delta['trend']

        if trend == 'improving':
            delta_html = f'<span style="color: #28a745; font-weight: bold;"> ▲ +{delta:.0f}</span>'
        elif trend == 'degrading':
            delta_html = f'<span style="color: #dc3545; font-weight: bold;"> ▼ {delta:.0f}</span>'
        else:
            delta_html = f'<span style="color: #6c757d;"> ● {delta:+.0f}</span>'
    else:
        delta_html = '<span style="color: #999; font-size: 0.8em;"> (new)</span>'

    # Highlight degrading stocks with a warning background
    card_style = f"border-left: 4px solid {text_color};"
    if score_delta and score_delta.get('trend') == 'degrading':
        card_style += " background: #fff5f5;"

    html = f"""
    <div class="stock-card" style="{card_style}">
        <div class="stock-header">
            <div>
                <span class="ticker">{pos.ticker}</span>
                <span style="color: #666;"> - {analysis.stock_data.name}</span>
            </div>
            <span style="background: {bg_color}; color: {text_color}; padding: 5px 10px; border-radius: 5px; font-weight: bold;">
                {analysis.recommendation}
            </span>
        </div>

        <div class="details">
            <p><strong>Position:</strong> {pos.shares:.2f} shares @ ${pos.cost_basis:.2f} | <strong>Current:</strong> ${analysis.stock_data.current_price:.2f}</p>
            <p><strong>Value:</strong> ${analysis.current_value:,.2f} |
               <strong style="color: {gain_color};">Gain/Loss: ${analysis.gain_loss:+,.2f} ({analysis.gain_loss_pct:+.1f}%)</strong></p>
            <p><strong>CANSLIM:</strong> {score.total_score:.0f}/100{delta_html} | <strong>6-Mo Projection:</strong> {proj.projected_growth_pct:+.1f}%</p>
            <p style="font-size: 0.85em; color: #666;"><em>{analysis.recommendation_reason}</em></p>
        </div>
    </div>
    """
    return html


def generate_html_report(under20_results: list[AnalysisResult],
                         sp500_results: list[AnalysisResult],
                         portfolio_analyses: list[PositionAnalysis],
                         portfolio_deltas: dict,
                         market_info: tuple) -> str:
    """Generate an HTML email report with three sections"""
    is_bullish, pct_above_200, pct_above_50 = market_info

    market_status = "BULLISH" if is_bullish else "BEARISH"
    market_color = "#28a745" if is_bullish else "#dc3545"

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            h2 {{ color: #007bff; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
            .header-info {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .market-status {{ color: {market_color}; font-weight: bold; }}
            .section-desc {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
            .stock-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; }}
            .stock-header {{ display: flex; justify-content: space-between; align-items: center; }}
            .ticker {{ font-size: 1.4em; font-weight: bold; color: #007bff; }}
            .score {{ background: #007bff; color: white; padding: 5px 10px; border-radius: 5px; }}
            .growth {{ font-size: 1.2em; color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; }}
            .details {{ margin-top: 10px; font-size: 0.9em; color: #666; }}
            .breakdown {{ background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace; font-size: 0.85em; }}
            .disclaimer {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 20px; font-size: 0.85em; }}
        </style>
    </head>
    <body>
        <h1>CANSLIM Stock Analyzer - Daily Report</h1>

        <div class="header-info">
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')} CST</p>
            <p><strong>Market Direction:</strong> <span class="market-status">{market_status}</span> (S&P 500 {pct_above_200:+.1f}% vs 200-day MA)</p>
        </div>

        <h2>Top 10 Value Picks (Under $20)</h2>
        <p class="section-desc">Small and mid-cap opportunities from S&P 500 + Russell 2000</p>
    """

    if under20_results:
        for rank, result in enumerate(under20_results, 1):
            html += generate_stock_card_html(rank, result)
    else:
        html += "<p><em>No stocks under $20 passed the CANSLIM criteria today.</em></p>"

    html += """
        <h2>Top 5 S&P 500 Picks (All Prices)</h2>
        <p class="section-desc">Large-cap leaders from the S&P 500</p>
    """

    if sp500_results:
        for rank, result in enumerate(sp500_results, 1):
            html += generate_stock_card_html(rank, result)
    else:
        html += "<p><em>No S&P 500 stocks passed the CANSLIM criteria today.</em></p>"

    # Portfolio section
    if portfolio_analyses:
        total_cost = sum(a.position.total_cost for a in portfolio_analyses)
        total_value = sum(a.current_value for a in portfolio_analyses)
        total_gain = total_value - total_cost
        total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
        gain_color = "#28a745" if total_gain >= 0 else "#dc3545"

        buy_count = sum(1 for a in portfolio_analyses if a.recommendation == "BUY MORE")
        hold_count = sum(1 for a in portfolio_analyses if a.recommendation == "HOLD")
        sell_count = sum(1 for a in portfolio_analyses if a.recommendation == "SELL")

        # Count score changes
        improving_count = sum(1 for d in portfolio_deltas.values() if d.get('trend') == 'improving')
        degrading_count = sum(1 for d in portfolio_deltas.values() if d.get('trend') == 'degrading')

        html += f"""
        <h2>Your Portfolio Analysis</h2>
        <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <p><strong>Total Value:</strong> ${total_value:,.2f} |
               <strong style="color: {gain_color};">Total Gain/Loss: ${total_gain:+,.2f} ({total_gain_pct:+.1f}%)</strong></p>
            <p><strong>Recommendations:</strong>
               <span style="color: #28a745;">{buy_count} BUY MORE</span> |
               <span style="color: #6c757d;">{hold_count} HOLD</span> |
               <span style="color: #dc3545;">{sell_count} SELL</span></p>
            <p><strong>Score Changes:</strong>
               <span style="color: #28a745;">▲ {improving_count} improving</span> |
               <span style="color: #dc3545;">▼ {degrading_count} degrading</span></p>
        </div>
        """

        # Alert for degrading scores
        degrading_tickers = [a.position.ticker for a in portfolio_analyses
                           if portfolio_deltas.get(a.position.ticker, {}).get('trend') == 'degrading']
        if degrading_tickers:
            html += f"""
        <div style="background: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #dc3545;">
            <strong style="color: #dc3545;">⚠ Score Alert:</strong> {', '.join(degrading_tickers)} showing declining CANSLIM scores
        </div>
            """

        # Sort by recommendation (SELL first, then HOLD, then BUY)
        rec_order = {"SELL": 0, "HOLD": 1, "BUY MORE": 2}
        sorted_portfolio = sorted(portfolio_analyses,
                                  key=lambda a: (rec_order.get(a.recommendation, 1), -a.canslim_score.total_score))

        for analysis in sorted_portfolio:
            delta = portfolio_deltas.get(analysis.position.ticker, {})
            html += generate_portfolio_card_html(analysis, delta)

    html += """
        <div class="disclaimer">
            <strong>DISCLAIMER:</strong> This analysis is for educational and informational purposes only.
            It is NOT financial advice. Past performance does not guarantee future results.
            Always do your own research before investing.
        </div>
    </body>
    </html>
    """

    return html


def generate_text_report(under20_results: list[AnalysisResult],
                         sp500_results: list[AnalysisResult],
                         portfolio_analyses: list[PositionAnalysis],
                         portfolio_deltas: dict,
                         market_info: tuple) -> str:
    """Generate a plain text email report with three sections"""
    is_bullish, pct_above_200, pct_above_50 = market_info
    market_status = "BULLISH" if is_bullish else "BEARISH"

    text = f"""
CANSLIM STOCK ANALYZER - DAILY REPORT
{'=' * 60}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} CST
Market Direction: {market_status} (S&P 500 {pct_above_200:+.1f}% vs 200-day MA)

{'=' * 60}
TOP 10 VALUE PICKS (UNDER $20)
Small and mid-cap opportunities from S&P 500 + Russell 2000
{'=' * 60}
"""

    if under20_results:
        for rank, result in enumerate(under20_results, 1):
            score = result.canslim_score
            proj = result.growth_projection
            text += f"""
#{rank} {result.ticker} - {result.name}
    Sector: {result.sector}
    Price: ${result.current_price:.2f}
    CANSLIM Score: {score.total_score:.0f}/100
    Projected 6-Month Growth: {proj.projected_growth_pct:+.1f}% ({proj.confidence} confidence)
"""
            if proj.analyst_target > 0:
                text += f"    Analyst Target: ${proj.analyst_target:.2f} ({proj.analyst_upside:+.1f}% upside)\n"
            text += f"    Score: C:{score.c_score:.0f} A:{score.a_score:.0f} N:{score.n_score:.0f} S:{score.s_score:.0f} L:{score.l_score:.0f} I:{score.i_score:.0f} M:{score.m_score:.0f}\n"
            text += "-" * 60 + "\n"
    else:
        text += "\nNo stocks under $20 passed the CANSLIM criteria today.\n"

    text += f"""
{'=' * 60}
TOP 5 S&P 500 PICKS (ALL PRICES)
Large-cap leaders from the S&P 500
{'=' * 60}
"""

    if sp500_results:
        for rank, result in enumerate(sp500_results, 1):
            score = result.canslim_score
            proj = result.growth_projection
            text += f"""
#{rank} {result.ticker} - {result.name}
    Sector: {result.sector}
    Price: ${result.current_price:.2f}
    CANSLIM Score: {score.total_score:.0f}/100
    Projected 6-Month Growth: {proj.projected_growth_pct:+.1f}% ({proj.confidence} confidence)
"""
            if proj.analyst_target > 0:
                text += f"    Analyst Target: ${proj.analyst_target:.2f} ({proj.analyst_upside:+.1f}% upside)\n"
            text += f"    Score: C:{score.c_score:.0f} A:{score.a_score:.0f} N:{score.n_score:.0f} S:{score.s_score:.0f} L:{score.l_score:.0f} I:{score.i_score:.0f} M:{score.m_score:.0f}\n"
            text += "-" * 60 + "\n"
    else:
        text += "\nNo S&P 500 stocks passed the CANSLIM criteria today.\n"

    # Portfolio section
    if portfolio_analyses:
        total_cost = sum(a.position.total_cost for a in portfolio_analyses)
        total_value = sum(a.current_value for a in portfolio_analyses)
        total_gain = total_value - total_cost
        total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0

        buy_count = sum(1 for a in portfolio_analyses if a.recommendation == "BUY MORE")
        hold_count = sum(1 for a in portfolio_analyses if a.recommendation == "HOLD")
        sell_count = sum(1 for a in portfolio_analyses if a.recommendation == "SELL")

        improving_count = sum(1 for d in portfolio_deltas.values() if d.get('trend') == 'improving')
        degrading_count = sum(1 for d in portfolio_deltas.values() if d.get('trend') == 'degrading')

        text += f"""
{'=' * 60}
YOUR PORTFOLIO ANALYSIS
{'=' * 60}
Total Value: ${total_value:,.2f}
Total Gain/Loss: ${total_gain:+,.2f} ({total_gain_pct:+.1f}%)
Recommendations: {buy_count} BUY MORE | {hold_count} HOLD | {sell_count} SELL
Score Changes: {improving_count} improving | {degrading_count} degrading
{'-' * 60}
"""
        # Alert for degrading scores
        degrading_tickers = [a.position.ticker for a in portfolio_analyses
                           if portfolio_deltas.get(a.position.ticker, {}).get('trend') == 'degrading']
        if degrading_tickers:
            text += f"*** ALERT: {', '.join(degrading_tickers)} showing declining scores ***\n"
            text += "-" * 60 + "\n"

        rec_order = {"SELL": 0, "HOLD": 1, "BUY MORE": 2}
        sorted_portfolio = sorted(portfolio_analyses,
                                  key=lambda a: (rec_order.get(a.recommendation, 1), -a.canslim_score.total_score))

        for analysis in sorted_portfolio:
            pos = analysis.position
            delta = portfolio_deltas.get(pos.ticker, {})

            # Format delta string
            if delta.get('has_history'):
                d = delta['total_delta']
                if delta['trend'] == 'improving':
                    delta_str = f" [+{d:.0f} improving]"
                elif delta['trend'] == 'degrading':
                    delta_str = f" [{d:.0f} DEGRADING]"
                else:
                    delta_str = f" [{d:+.0f}]"
            else:
                delta_str = " [new]"

            text += f"""
[{analysis.recommendation}] {pos.ticker} - {analysis.stock_data.name}
    Position: {pos.shares:.2f} shares @ ${pos.cost_basis:.2f}
    Current: ${analysis.stock_data.current_price:.2f} | Value: ${analysis.current_value:,.2f}
    Gain/Loss: ${analysis.gain_loss:+,.2f} ({analysis.gain_loss_pct:+.1f}%)
    CANSLIM: {analysis.canslim_score.total_score:.0f}/100{delta_str} | Projection: {analysis.growth_projection.projected_growth_pct:+.1f}%
    {analysis.recommendation_reason}
{'-' * 60}
"""

    text += """
DISCLAIMER: This analysis is for educational purposes only.
Not financial advice. Always do your own research.
"""

    return text


def send_email(subject: str, html_content: str, text_content: str):
    """Send email via Gmail SMTP"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL

    part1 = MIMEText(text_content, 'plain')
    part2 = MIMEText(html_content, 'html')
    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        print(f"Email sent successfully to {RECIPIENT_EMAIL}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def main():
    print(f"Starting CANSLIM analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get market direction first
    fetcher = DataFetcher()
    market_info = fetcher.get_market_direction()
    is_bullish, pct_above_200, pct_above_50 = market_info

    if not is_bullish:
        print("WARNING: Market appears bearish")

    # ============ ANALYSIS 1: Under $20 stocks ============
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Top 10 stocks under $20 (S&P 500 + Russell 2000)")
    print("=" * 60)

    all_tickers = get_all_tickers()
    print(f"Scanning {len(all_tickers)} tickers...")

    under20_results = analyze_stocks(
        all_tickers,
        top_n=10,
        min_price=1,
        max_price=20,
        sector_filter=None
    )
    print(f"Found {len(under20_results)} stocks under $20")

    # ============ ANALYSIS 2: S&P 500 all prices ============
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Top 5 S&P 500 stocks (all prices)")
    print("=" * 60)

    sp500_tickers = get_sp500_tickers()
    print(f"Scanning {len(sp500_tickers)} S&P 500 tickers...")

    sp500_results = analyze_stocks(
        sp500_tickers,
        top_n=5,
        min_price=0,
        max_price=float('inf'),
        sector_filter=None
    )
    print(f"Found {len(sp500_results)} S&P 500 stocks")

    # ============ ANALYSIS 3: Your Portfolio ============
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Your Portfolio")
    print("=" * 60)

    portfolio_analyses = []
    portfolio_deltas = {}
    positions = load_positions_from_csv()

    if positions:
        print(f"Analyzing {len(positions)} positions from portfolio.csv...")
        portfolio_analyzer = PortfolioAnalyzer()

        for pos in positions:
            print(f"  Analyzing {pos.ticker}...", end="", flush=True)
            analysis = portfolio_analyzer.analyze_position(pos)
            if analysis:
                portfolio_analyses.append(analysis)

                # Get score components for history tracking
                score = analysis.canslim_score
                components = {
                    'c': score.c_score,
                    'a': score.a_score,
                    'n': score.n_score,
                    's': score.s_score,
                    'l': score.l_score,
                    'i': score.i_score,
                    'm': score.m_score
                }

                # Calculate delta from previous day
                delta = get_score_delta(pos.ticker, score.total_score, components)
                portfolio_deltas[pos.ticker] = delta

                # Record today's score
                record_score(pos.ticker, score.total_score, components)

                # Print with delta info
                if delta.get('has_history'):
                    trend_symbol = "▲" if delta['trend'] == 'improving' else ("▼" if delta['trend'] == 'degrading' else "●")
                    print(f" {analysis.recommendation} ({score.total_score:.0f} {trend_symbol}{delta['total_delta']:+.0f})")
                else:
                    print(f" {analysis.recommendation} ({score.total_score:.0f} new)")
            else:
                print(" failed")

        print(f"Analyzed {len(portfolio_analyses)} positions")

        # Show biggest movers
        movers = get_biggest_movers(threshold=3.0)
        if movers['degraders']:
            print(f"  ⚠ Degrading: {', '.join([t for t, d, s in movers['degraders']])}")
        if movers['improvers']:
            print(f"  ✓ Improving: {', '.join([t for t, d, s in movers['improvers']])}")
    else:
        print("No portfolio.csv found - skipping portfolio analysis")

    # ============ Generate and send report ============
    if not under20_results and not sp500_results and not portfolio_analyses:
        print("No stocks passed the CANSLIM criteria in any analysis")
        subject = f"CANSLIM Report {datetime.now().strftime('%Y-%m-%d')} - No Results"
        html = "<html><body><p>No stocks passed the CANSLIM criteria today.</p></body></html>"
        text = "No stocks passed the CANSLIM criteria today."
        send_email(subject, html, text)
        return

    # Generate reports
    html_report = generate_html_report(under20_results, sp500_results, portfolio_analyses, portfolio_deltas, market_info)
    text_report = generate_text_report(under20_results, sp500_results, portfolio_analyses, portfolio_deltas, market_info)

    # Send email
    total_picks = len(under20_results) + len(sp500_results)
    subject = f"CANSLIM Report {datetime.now().strftime('%Y-%m-%d')} - {total_picks} Picks + Portfolio"
    send_email(subject, html_report, text_report)

    print(f"\nAnalysis complete. Market picks: {total_picks}, Portfolio positions: {len(portfolio_analyses)}")


if __name__ == "__main__":
    main()
