#!/usr/bin/env python3
"""
Portfolio Manager
Simple CLI tool to manage your stock positions in portfolio.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.csv"


def load_portfolio() -> list[dict]:
    """Load portfolio from CSV file"""
    if not PORTFOLIO_FILE.exists():
        return []

    with open(PORTFOLIO_FILE, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_portfolio(positions: list[dict]):
    """Save portfolio to CSV file"""
    with open(PORTFOLIO_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ticker', 'shares', 'cost_basis'])
        writer.writeheader()
        writer.writerows(positions)


def add_position(ticker: str, shares: float, cost_basis: float):
    """Add a new position or update existing one"""
    positions = load_portfolio()
    ticker = ticker.upper()

    # Check if position already exists
    for pos in positions:
        if pos['ticker'].upper() == ticker:
            print(f"Position {ticker} already exists with {pos['shares']} shares @ ${pos['cost_basis']}")
            print(f"Use 'update' command to modify it.")
            return False

    positions.append({
        'ticker': ticker,
        'shares': str(shares),
        'cost_basis': str(cost_basis)
    })

    save_portfolio(positions)
    print(f"Added: {ticker} - {shares} shares @ ${cost_basis:.2f}")
    return True


def update_position(ticker: str, shares: float, cost_basis: float):
    """Update an existing position"""
    positions = load_portfolio()
    ticker = ticker.upper()

    for pos in positions:
        if pos['ticker'].upper() == ticker:
            old_shares = pos['shares']
            old_cost = pos['cost_basis']
            pos['shares'] = str(shares)
            pos['cost_basis'] = str(cost_basis)
            save_portfolio(positions)
            print(f"Updated: {ticker}")
            print(f"  Old: {old_shares} shares @ ${old_cost}")
            print(f"  New: {shares} shares @ ${cost_basis:.2f}")
            return True

    print(f"Position {ticker} not found. Use 'add' to create it.")
    return False


def remove_position(ticker: str):
    """Remove a position"""
    positions = load_portfolio()
    ticker = ticker.upper()

    new_positions = [p for p in positions if p['ticker'].upper() != ticker]

    if len(new_positions) == len(positions):
        print(f"Position {ticker} not found.")
        return False

    save_portfolio(new_positions)
    print(f"Removed: {ticker}")
    return True


def list_positions():
    """List all positions"""
    positions = load_portfolio()

    if not positions:
        print("No positions in portfolio.")
        return

    print()
    print("=" * 50)
    print("           YOUR PORTFOLIO")
    print("=" * 50)
    print(f"{'Ticker':<8} {'Shares':>12} {'Cost Basis':>12}")
    print("-" * 50)

    total_cost = 0
    for pos in positions:
        ticker = pos['ticker']
        shares = float(pos['shares'])
        cost = float(pos['cost_basis'])
        position_cost = shares * cost
        total_cost += position_cost
        print(f"{ticker:<8} {shares:>12.3f} ${cost:>10.2f}")

    print("-" * 50)
    print(f"Total Positions: {len(positions)}")
    print(f"Total Cost Basis: ${total_cost:,.2f}")
    print("=" * 50)
    print()


def parse_position_string(pos_str: str) -> tuple[str, float, float]:
    """Parse TICKER:SHARES:COST_BASIS format"""
    parts = pos_str.split(":")
    ticker = parts[0].strip().upper()
    shares = float(parts[1]) if len(parts) > 1 else 0
    cost_basis = float(parts[2]) if len(parts) > 2 else 0
    return ticker, shares, cost_basis


def main():
    parser = argparse.ArgumentParser(
        description="Manage your stock portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  add TICKER:SHARES:COST      Add a new position
  update TICKER:SHARES:COST   Update existing position
  remove TICKER               Remove a position (sold all)
  list                        Show all positions

Examples:
  python portfolio_manager.py add AAPL:100:175.50
  python portfolio_manager.py update AAPL:150:180.00
  python portfolio_manager.py remove AAPL
  python portfolio_manager.py list
        """
    )

    parser.add_argument('command', choices=['add', 'update', 'remove', 'list'],
                        help='Command to execute')
    parser.add_argument('position', nargs='?', default=None,
                        help='Position in TICKER:SHARES:COST_BASIS format')

    args = parser.parse_args()

    if args.command == 'list':
        list_positions()

    elif args.command == 'add':
        if not args.position:
            print("Error: Position required. Format: TICKER:SHARES:COST_BASIS")
            sys.exit(1)
        ticker, shares, cost = parse_position_string(args.position)
        add_position(ticker, shares, cost)

    elif args.command == 'update':
        if not args.position:
            print("Error: Position required. Format: TICKER:SHARES:COST_BASIS")
            sys.exit(1)
        ticker, shares, cost = parse_position_string(args.position)
        update_position(ticker, shares, cost)

    elif args.command == 'remove':
        if not args.position:
            print("Error: Ticker required.")
            sys.exit(1)
        ticker = args.position.split(":")[0].upper()
        remove_position(ticker)


if __name__ == "__main__":
    main()
