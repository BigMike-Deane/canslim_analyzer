#!/usr/bin/env python3
"""
Send a demo email showing the full report format with sample data
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

# Load .env file
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

GMAIL_ADDRESS = os.environ.get('CANSLIM_EMAIL')
GMAIL_APP_PASSWORD = os.environ.get('CANSLIM_APP_PASSWORD')
RECIPIENT_EMAIL = os.environ.get('CANSLIM_RECIPIENT', GMAIL_ADDRESS)

# Sample data for demo
UNDER_20_PICKS = [
    {"rank": 1, "ticker": "SOFI", "name": "SoFi Technologies Inc", "sector": "Financial Services",
     "price": 12.45, "score": 72, "growth": 34.2, "target": 18.50, "upside": 48.6, "analysts": 18,
     "c": 12, "a": 11, "n": 13, "s": 10, "l": 12, "i": 7, "m": 7},
    {"rank": 2, "ticker": "PLTR", "name": "Palantir Technologies", "sector": "Technology",
     "price": 18.92, "score": 68, "growth": 28.5, "target": 25.00, "upside": 32.1, "analysts": 14,
     "c": 10, "a": 12, "n": 11, "s": 12, "l": 10, "i": 6, "m": 7},
    {"rank": 3, "ticker": "LCID", "name": "Lucid Group Inc", "sector": "Consumer Cyclical",
     "price": 4.82, "score": 61, "growth": 42.1, "target": 7.50, "upside": 55.6, "analysts": 12,
     "c": 8, "a": 7, "n": 14, "s": 11, "l": 9, "i": 5, "m": 7},
    {"rank": 4, "ticker": "RIVN", "name": "Rivian Automotive", "sector": "Consumer Cyclical",
     "price": 14.33, "score": 58, "growth": 31.8, "target": 20.00, "upside": 39.6, "analysts": 22,
     "c": 6, "a": 8, "n": 12, "s": 10, "l": 11, "i": 4, "m": 7},
    {"rank": 5, "ticker": "UPST", "name": "Upstart Holdings", "sector": "Financial Services",
     "price": 15.67, "score": 55, "growth": 25.4, "target": 22.00, "upside": 40.4, "analysts": 11,
     "c": 9, "a": 6, "n": 10, "s": 9, "l": 10, "i": 4, "m": 7},
    {"rank": 6, "ticker": "IONQ", "name": "IonQ Inc", "sector": "Technology",
     "price": 11.25, "score": 54, "growth": 38.7, "target": 16.00, "upside": 42.2, "analysts": 8,
     "c": 7, "a": 5, "n": 13, "s": 8, "l": 12, "i": 3, "m": 6},
    {"rank": 7, "ticker": "OPEN", "name": "Opendoor Technologies", "sector": "Real Estate",
     "price": 3.45, "score": 52, "growth": 29.3, "target": 5.00, "upside": 44.9, "analysts": 9,
     "c": 8, "a": 4, "n": 11, "s": 9, "l": 8, "i": 5, "m": 7},
    {"rank": 8, "ticker": "DNA", "name": "Ginkgo Bioworks", "sector": "Healthcare",
     "price": 1.89, "score": 49, "growth": 45.2, "target": 3.00, "upside": 58.7, "analysts": 7,
     "c": 5, "a": 6, "n": 12, "s": 7, "l": 9, "i": 4, "m": 6},
    {"rank": 9, "ticker": "SKLZ", "name": "Skillz Inc", "sector": "Technology",
     "price": 7.82, "score": 47, "growth": 22.1, "target": 10.00, "upside": 27.9, "analysts": 6,
     "c": 6, "a": 5, "n": 9, "s": 8, "l": 8, "i": 5, "m": 6},
    {"rank": 10, "ticker": "WISH", "name": "ContextLogic Inc", "sector": "Consumer Cyclical",
     "price": 5.12, "score": 44, "growth": 35.6, "target": 7.50, "upside": 46.5, "analysts": 5,
     "c": 4, "a": 3, "n": 11, "s": 7, "l": 10, "i": 3, "m": 6},
]

SP500_PICKS = [
    {"rank": 1, "ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology",
     "price": 485.22, "score": 89, "growth": 42.3, "target": 650.00, "upside": 34.0, "analysts": 45,
     "c": 14, "a": 15, "n": 14, "s": 13, "l": 15, "i": 8, "m": 10},
    {"rank": 2, "ticker": "META", "name": "Meta Platforms Inc", "sector": "Technology",
     "price": 352.18, "score": 82, "growth": 28.7, "target": 425.00, "upside": 20.7, "analysts": 52,
     "c": 13, "a": 14, "n": 12, "s": 11, "l": 14, "i": 8, "m": 10},
    {"rank": 3, "ticker": "AMZN", "name": "Amazon.com Inc", "sector": "Consumer Cyclical",
     "price": 178.45, "score": 78, "growth": 24.5, "target": 220.00, "upside": 23.3, "analysts": 58,
     "c": 12, "a": 13, "n": 11, "s": 12, "l": 12, "i": 8, "m": 10},
    {"rank": 4, "ticker": "GOOGL", "name": "Alphabet Inc", "sector": "Technology",
     "price": 142.89, "score": 75, "growth": 21.2, "target": 175.00, "upside": 22.5, "analysts": 48,
     "c": 11, "a": 12, "n": 10, "s": 11, "l": 13, "i": 8, "m": 10},
    {"rank": 5, "ticker": "LLY", "name": "Eli Lilly & Co", "sector": "Healthcare",
     "price": 612.33, "score": 73, "growth": 32.1, "target": 800.00, "upside": 30.6, "analysts": 28,
     "c": 12, "a": 11, "n": 13, "s": 10, "l": 11, "i": 7, "m": 9},
]

# Your portfolio with sample CANSLIM data and deltas
PORTFOLIO = [
    {"ticker": "CMPS", "name": "Compass Pathways", "shares": 100, "cost": 7.46, "current": 8.12,
     "score": 42, "growth": 28.5, "rec": "HOLD", "reason": "Speculative biotech with pipeline potential",
     "delta": -3, "trend": "degrading", "c": 3, "a": 4, "n": 10, "s": 6, "l": 8, "i": 4, "m": 7},
    {"ticker": "ZETA", "name": "Zeta Global Holdings", "shares": 100, "cost": 15.11, "current": 18.45,
     "score": 58, "growth": 22.1, "rec": "HOLD", "reason": "Growing marketing platform with solid fundamentals",
     "delta": 4, "trend": "improving", "c": 9, "a": 8, "n": 11, "s": 8, "l": 10, "i": 5, "m": 7},
    {"ticker": "HUMA", "name": "Humacyte Inc", "shares": 430, "cost": 5.37, "current": 4.82,
     "score": 35, "growth": 18.3, "rec": "HOLD", "reason": "Early-stage biotech, high risk/reward",
     "delta": 0, "trend": "stable", "c": 2, "a": 3, "n": 9, "s": 5, "l": 7, "i": 3, "m": 6},
    {"ticker": "LCTX", "name": "Lineage Cell Therapeutics", "shares": 1000, "cost": 1.69, "current": 1.45,
     "score": 28, "growth": 15.2, "rec": "SELL", "reason": "Weak fundamentals, below cost basis with limited catalyst",
     "delta": -5, "trend": "degrading", "c": 1, "a": 2, "n": 7, "s": 4, "l": 5, "i": 3, "m": 6},
    {"ticker": "XYZ", "name": "Block Inc", "shares": 9.621, "cost": 77.95, "current": 72.33,
     "score": 52, "growth": 16.8, "rec": "HOLD", "reason": "Payments leader facing margin pressure",
     "delta": -2, "trend": "stable", "c": 8, "a": 7, "n": 8, "s": 7, "l": 9, "i": 6, "m": 7},
    {"ticker": "IRWD", "name": "Ironwood Pharmaceuticals", "shares": 200, "cost": 4.42, "current": 5.18,
     "score": 48, "growth": 12.4, "rec": "HOLD", "reason": "Stable GI franchise, limited growth catalysts",
     "delta": 2, "trend": "stable", "c": 7, "a": 6, "n": 8, "s": 6, "l": 9, "i": 5, "m": 7},
    {"ticker": "GOSS", "name": "Gossamer Bio Inc", "shares": 250, "cost": 2.50, "current": 1.85,
     "score": 22, "growth": 25.6, "rec": "SELL", "reason": "High cash burn, weak clinical pipeline progress",
     "delta": -8, "trend": "degrading", "c": 1, "a": 1, "n": 6, "s": 3, "l": 4, "i": 2, "m": 5},
    {"ticker": "TMQ", "name": "Trilogy Metals Inc", "shares": 100, "cost": 5.63, "current": 4.92,
     "score": 38, "growth": 21.3, "rec": "HOLD", "reason": "Copper exposure, dependent on commodity prices",
     "delta": 1, "trend": "stable", "c": 4, "a": 5, "n": 9, "s": 5, "l": 6, "i": 3, "m": 6},
    {"ticker": "ARM", "name": "Arm Holdings", "shares": 32.082, "cost": 111.10, "current": 148.75,
     "score": 74, "growth": 28.9, "rec": "BUY MORE", "reason": "AI chip leader with strong licensing growth",
     "delta": 6, "trend": "improving", "c": 12, "a": 11, "n": 13, "s": 10, "l": 12, "i": 7, "m": 9},
    {"ticker": "NANC", "name": "Unusual Whales ETF", "shares": 70, "cost": 38.15, "current": 42.50,
     "score": 55, "growth": 14.2, "rec": "HOLD", "reason": "Congressional trading ETF, moderate momentum",
     "delta": 0, "trend": "stable", "c": 8, "a": 7, "n": 10, "s": 7, "l": 9, "i": 6, "m": 8},
    {"ticker": "VERI", "name": "Veritone Inc", "shares": 100, "cost": 4.44, "current": 3.28,
     "score": 31, "growth": 32.1, "rec": "SELL", "reason": "AI company struggling with profitability path",
     "delta": -6, "trend": "degrading", "c": 3, "a": 2, "n": 8, "s": 4, "l": 6, "i": 3, "m": 5},
    {"ticker": "STE", "name": "STERIS plc", "shares": 2.071, "cost": 241.40, "current": 225.80,
     "score": 62, "growth": 11.5, "rec": "HOLD", "reason": "Stable healthcare equipment, defensive play",
     "delta": 1, "trend": "stable", "c": 9, "a": 10, "n": 8, "s": 8, "l": 11, "i": 7, "m": 9},
    {"ticker": "ONDS", "name": "Ondas Holdings", "shares": 50, "cost": 13.03, "current": 2.45,
     "score": 18, "growth": 45.2, "rec": "SELL", "reason": "Severe losses, questionable path to profitability",
     "delta": -4, "trend": "degrading", "c": 1, "a": 0, "n": 5, "s": 2, "l": 3, "i": 2, "m": 5},
    {"ticker": "CAVA", "name": "CAVA Group Inc", "shares": 15.767, "cost": 73.40, "current": 89.22,
     "score": 68, "growth": 24.8, "rec": "BUY MORE", "reason": "Fast-casual leader with strong unit economics",
     "delta": 5, "trend": "improving", "c": 11, "a": 10, "n": 12, "s": 9, "l": 11, "i": 6, "m": 9},
]


def generate_html_report():
    """Generate the full HTML email report"""

    # Calculate portfolio totals
    total_cost = sum(p['shares'] * p['cost'] for p in PORTFOLIO)
    total_value = sum(p['shares'] * p['current'] for p in PORTFOLIO)
    total_gain = total_value - total_cost
    total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
    gain_color = "#28a745" if total_gain >= 0 else "#dc3545"

    buy_count = sum(1 for p in PORTFOLIO if p['rec'] == "BUY MORE")
    hold_count = sum(1 for p in PORTFOLIO if p['rec'] == "HOLD")
    sell_count = sum(1 for p in PORTFOLIO if p['rec'] == "SELL")

    improving_count = sum(1 for p in PORTFOLIO if p['trend'] == 'improving')
    degrading_count = sum(1 for p in PORTFOLIO if p['trend'] == 'degrading')

    degrading_tickers = [p['ticker'] for p in PORTFOLIO if p['trend'] == 'degrading']

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
            .container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a1a2e; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
            h2 {{ color: #007bff; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
            .header-info {{ background: #e8f4fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #007bff; }}
            .market-status {{ color: #28a745; font-weight: bold; font-size: 1.1em; }}
            .section-desc {{ color: #666; font-size: 0.9em; margin-bottom: 15px; font-style: italic; }}
            .stock-card {{ border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; margin-bottom: 15px; background: #fafafa; }}
            .stock-card:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .stock-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
            .ticker {{ font-size: 1.3em; font-weight: bold; color: #1a1a2e; }}
            .score {{ background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; }}
            .growth {{ font-size: 1.1em; color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; }}
            .details {{ font-size: 0.9em; color: #555; line-height: 1.6; }}
            .breakdown {{ background: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: 'Courier New', monospace; font-size: 0.85em; color: #444; }}
            .portfolio-summary {{ background: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .alert-box {{ background: #fff5f5; padding: 12px 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #dc3545; }}
            .rec-buy {{ background: #d4edda; color: #155724; padding: 5px 12px; border-radius: 15px; font-weight: bold; }}
            .rec-hold {{ background: #e9ecef; color: #495057; padding: 5px 12px; border-radius: 15px; font-weight: bold; }}
            .rec-sell {{ background: #f8d7da; color: #721c24; padding: 5px 12px; border-radius: 15px; font-weight: bold; }}
            .delta-up {{ color: #28a745; font-weight: bold; }}
            .delta-down {{ color: #dc3545; font-weight: bold; }}
            .delta-stable {{ color: #6c757d; }}
            .disclaimer {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin-top: 30px; font-size: 0.85em; border-left: 4px solid #ffc107; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CANSLIM Stock Analyzer - Daily Report</h1>

            <div class="header-info">
                <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')} 9:00 AM CST</p>
                <p><strong>Market Direction:</strong> <span class="market-status">BULLISH</span> (S&P 500 +3.2% vs 200-day MA)</p>
                <p><strong>Stocks Analyzed:</strong> 1,860 (S&P 500 + Russell 2000)</p>
            </div>

            <h2>Top 10 Value Picks (Under $20)</h2>
            <p class="section-desc">Small and mid-cap opportunities from S&P 500 + Russell 2000 with strong CANSLIM scores</p>
    """

    # Under $20 picks
    for s in UNDER_20_PICKS:
        html += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <div>
                        <span class="ticker">#{s['rank']} {s['ticker']}</span>
                        <span style="color: #666;"> - {s['name']}</span>
                    </div>
                    <span class="score">{s['score']}/100</span>
                </div>
                <div class="details">
                    <p><strong>Sector:</strong> {s['sector']} | <strong>Price:</strong> ${s['price']:.2f}</p>
                    <p><strong>Analyst Target:</strong> ${s['target']:.2f} ({s['upside']:+.1f}% upside) [{s['analysts']} analysts]</p>
                    <p class="growth">Projected 6-Month Growth: {s['growth']:+.1f}% (medium confidence)</p>
                </div>
                <div class="breakdown">
                    C: {s['c']}/15 | A: {s['a']}/15 | N: {s['n']}/15 | S: {s['s']}/15 | L: {s['l']}/15 | I: {s['i']}/10 | M: {s['m']}/15
                </div>
            </div>
        """

    html += """
            <h2>Top 5 S&P 500 Picks (All Prices)</h2>
            <p class="section-desc">Large-cap market leaders with the strongest CANSLIM profiles</p>
    """

    # S&P 500 picks
    for s in SP500_PICKS:
        html += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <div>
                        <span class="ticker">#{s['rank']} {s['ticker']}</span>
                        <span style="color: #666;"> - {s['name']}</span>
                    </div>
                    <span class="score">{s['score']}/100</span>
                </div>
                <div class="details">
                    <p><strong>Sector:</strong> {s['sector']} | <strong>Price:</strong> ${s['price']:.2f}</p>
                    <p><strong>Analyst Target:</strong> ${s['target']:.2f} ({s['upside']:+.1f}% upside) [{s['analysts']} analysts]</p>
                    <p class="growth">Projected 6-Month Growth: {s['growth']:+.1f}% (high confidence)</p>
                </div>
                <div class="breakdown">
                    C: {s['c']}/15 | A: {s['a']}/15 | N: {s['n']}/15 | S: {s['s']}/15 | L: {s['l']}/15 | I: {s['i']}/10 | M: {s['m']}/15
                </div>
            </div>
        """

    # Portfolio section
    html += f"""
            <h2>Your Portfolio Analysis</h2>
            <div class="portfolio-summary">
                <p><strong>Total Value:</strong> ${total_value:,.2f} |
                   <strong style="color: {gain_color};">Total Gain/Loss: ${total_gain:+,.2f} ({total_gain_pct:+.1f}%)</strong></p>
                <p><strong>Recommendations:</strong>
                   <span style="color: #28a745;">{buy_count} BUY MORE</span> |
                   <span style="color: #6c757d;">{hold_count} HOLD</span> |
                   <span style="color: #dc3545;">{sell_count} SELL</span></p>
                <p><strong>Score Changes (vs yesterday):</strong>
                   <span class="delta-up">&#9650; {improving_count} improving</span> |
                   <span class="delta-down">&#9660; {degrading_count} degrading</span></p>
            </div>
    """

    if degrading_tickers:
        html += f"""
            <div class="alert-box">
                <strong style="color: #dc3545;">&#9888; Score Alert:</strong> {', '.join(degrading_tickers)} showing declining CANSLIM scores - consider reviewing these positions
            </div>
        """

    # Sort portfolio: SELL first, then HOLD, then BUY
    rec_order = {"SELL": 0, "HOLD": 1, "BUY MORE": 2}
    sorted_portfolio = sorted(PORTFOLIO, key=lambda p: (rec_order.get(p['rec'], 1), -p['score']))

    for p in sorted_portfolio:
        value = p['shares'] * p['current']
        cost = p['shares'] * p['cost']
        gain = value - cost
        gain_pct = (gain / cost * 100) if cost > 0 else 0
        pos_gain_color = "#28a745" if gain >= 0 else "#dc3545"

        # Recommendation styling
        if p['rec'] == "BUY MORE":
            rec_class = "rec-buy"
        elif p['rec'] == "SELL":
            rec_class = "rec-sell"
        else:
            rec_class = "rec-hold"

        # Delta styling
        if p['trend'] == 'improving':
            delta_html = f'<span class="delta-up">&#9650; +{p["delta"]}</span>'
        elif p['trend'] == 'degrading':
            delta_html = f'<span class="delta-down">&#9660; {p["delta"]}</span>'
        else:
            delta_html = f'<span class="delta-stable">&#9679; {p["delta"]:+d}</span>'

        # Card background for degrading
        card_style = "border-left: 4px solid "
        if p['rec'] == "BUY MORE":
            card_style += "#28a745;"
        elif p['rec'] == "SELL":
            card_style += "#dc3545;"
        else:
            card_style += "#6c757d;"

        if p['trend'] == 'degrading':
            card_style += " background: #fff8f8;"

        html += f"""
            <div class="stock-card" style="{card_style}">
                <div class="stock-header">
                    <div>
                        <span class="ticker">{p['ticker']}</span>
                        <span style="color: #666;"> - {p['name']}</span>
                    </div>
                    <span class="{rec_class}">{p['rec']}</span>
                </div>
                <div class="details">
                    <p><strong>Position:</strong> {p['shares']:.2f} shares @ ${p['cost']:.2f} | <strong>Current:</strong> ${p['current']:.2f}</p>
                    <p><strong>Value:</strong> ${value:,.2f} |
                       <strong style="color: {pos_gain_color};">Gain/Loss: ${gain:+,.2f} ({gain_pct:+.1f}%)</strong></p>
                    <p><strong>CANSLIM:</strong> {p['score']}/100 {delta_html} | <strong>6-Mo Projection:</strong> {p['growth']:+.1f}%</p>
                    <p style="font-size: 0.85em; color: #666; font-style: italic;">{p['reason']}</p>
                </div>
                <div class="breakdown">
                    C: {p['c']}/15 | A: {p['a']}/15 | N: {p['n']}/15 | S: {p['s']}/15 | L: {p['l']}/15 | I: {p['i']}/10 | M: {p['m']}/15
                </div>
            </div>
        """

    html += """
            <div class="disclaimer">
                <strong>DISCLAIMER:</strong> This analysis is for educational and informational purposes only.
                It is NOT financial advice. Past performance does not guarantee future results.
                Always do your own research and consult a financial advisor before making investment decisions.
            </div>
        </div>
    </body>
    </html>
    """

    return html


def generate_text_report():
    """Generate plain text version"""

    total_cost = sum(p['shares'] * p['cost'] for p in PORTFOLIO)
    total_value = sum(p['shares'] * p['current'] for p in PORTFOLIO)
    total_gain = total_value - total_cost
    total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0

    text = f"""
CANSLIM STOCK ANALYZER - DAILY REPORT
{'=' * 60}
Date: {datetime.now().strftime('%B %d, %Y')} 9:00 AM CST
Market Direction: BULLISH (S&P 500 +3.2% vs 200-day MA)
{'=' * 60}

TOP 10 VALUE PICKS (UNDER $20)
{'-' * 60}
"""

    for s in UNDER_20_PICKS:
        text += f"""
#{s['rank']} {s['ticker']} - {s['name']}
    Price: ${s['price']:.2f} | Score: {s['score']}/100
    Projected Growth: {s['growth']:+.1f}% | Target: ${s['target']:.2f}
"""

    text += f"""
{'=' * 60}
TOP 5 S&P 500 PICKS
{'-' * 60}
"""

    for s in SP500_PICKS:
        text += f"""
#{s['rank']} {s['ticker']} - {s['name']}
    Price: ${s['price']:.2f} | Score: {s['score']}/100
    Projected Growth: {s['growth']:+.1f}% | Target: ${s['target']:.2f}
"""

    text += f"""
{'=' * 60}
YOUR PORTFOLIO
Total Value: ${total_value:,.2f} | Gain/Loss: ${total_gain:+,.2f} ({total_gain_pct:+.1f}%)
{'-' * 60}
"""

    for p in PORTFOLIO:
        value = p['shares'] * p['current']
        gain = value - (p['shares'] * p['cost'])
        delta_str = f"+{p['delta']}" if p['delta'] > 0 else str(p['delta'])
        text += f"""
[{p['rec']}] {p['ticker']} - Score: {p['score']}/100 ({delta_str})
    Value: ${value:,.2f} | Gain: ${gain:+,.2f}
"""

    text += """
DISCLAIMER: For educational purposes only. Not financial advice.
"""

    return text


def send_email(subject, html_content, text_content):
    """Send email via Gmail"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL

    msg.attach(MIMEText(text_content, 'plain'))
    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        print(f"Email sent successfully to {RECIPIENT_EMAIL}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


if __name__ == "__main__":
    print("Generating demo email report...")

    html = generate_html_report()
    text = generate_text_report()

    subject = f"CANSLIM Daily Report - {datetime.now().strftime('%Y-%m-%d')} - 15 Picks + Portfolio"

    send_email(subject, html, text)
    print("Done!")
