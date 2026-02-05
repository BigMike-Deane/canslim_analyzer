#!/usr/bin/env python3
"""
Score History Tracker
Stores and retrieves historical CANSLIM scores for portfolio positions
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

HISTORY_FILE = Path(__file__).parent / "portfolio_history.json"


def load_history() -> dict:
    """Load score history from JSON file"""
    if not HISTORY_FILE.exists():
        return {}

    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_history(history: dict):
    """Save score history to JSON file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def record_score(ticker: str, score: float, components: dict):
    """
    Record today's CANSLIM score for a ticker.

    Args:
        ticker: Stock ticker symbol
        score: Total CANSLIM score (0-100)
        components: Dict with individual scores {c, a, n, s, l, i, m}
    """
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')

    if ticker not in history:
        history[ticker] = {}

    history[ticker][today] = {
        'total': score,
        'components': components,
        'timestamp': datetime.now().isoformat()
    }

    # Keep only last 90 days of history per ticker
    if len(history[ticker]) > 90:
        dates = sorted(history[ticker].keys())
        for old_date in dates[:-90]:
            del history[ticker][old_date]

    save_history(history)


def get_previous_score(ticker: str, days_back: int = 1) -> Optional[dict]:
    """
    Get the score from a previous day.

    Args:
        ticker: Stock ticker symbol
        days_back: How many days to look back (default: 1 for yesterday)

    Returns:
        Dict with 'total' and 'components', or None if not found
    """
    history = load_history()

    if ticker not in history:
        return None

    # Get all dates for this ticker, sorted
    dates = sorted(history[ticker].keys(), reverse=True)

    if len(dates) < 2:
        return None

    # Skip today if it exists, get the previous entry
    today = datetime.now().strftime('%Y-%m-%d')
    previous_dates = [d for d in dates if d != today]

    if not previous_dates:
        return None

    # Return the most recent previous date
    prev_date = previous_dates[0]
    return history[ticker][prev_date]


def get_score_delta(ticker: str, current_score: float, current_components: dict) -> dict:
    """
    Calculate the change in score from the previous day.

    Returns:
        Dict with:
        - 'total_delta': Change in total score
        - 'component_deltas': Dict of changes per component
        - 'trend': 'improving', 'stable', or 'degrading'
        - 'has_history': Whether we have previous data
    """
    result = {
        'total_delta': 0,
        'component_deltas': {},
        'trend': 'stable',
        'has_history': False,
        'prev_date': None
    }

    prev = get_previous_score(ticker)

    if prev is None:
        return result

    result['has_history'] = True
    result['total_delta'] = current_score - prev['total']

    # Calculate component deltas
    prev_components = prev.get('components', {})
    for key in ['c', 'a', 'n', 's', 'l', 'i', 'm']:
        current_val = current_components.get(key, 0)
        prev_val = prev_components.get(key, 0)
        result['component_deltas'][key] = current_val - prev_val

    # Determine trend
    if result['total_delta'] >= 3:
        result['trend'] = 'improving'
    elif result['total_delta'] <= -3:
        result['trend'] = 'degrading'
    else:
        result['trend'] = 'stable'

    return result


def get_trend_history(ticker: str, days: int = 7) -> list:
    """
    Get the score trend over the last N days.

    Returns:
        List of (date, total_score) tuples, most recent first
    """
    history = load_history()

    if ticker not in history:
        return []

    dates = sorted(history[ticker].keys(), reverse=True)[:days]

    return [(d, history[ticker][d]['total']) for d in dates]


def get_biggest_movers(threshold: float = 5.0) -> dict:
    """
    Find stocks with the biggest score changes from yesterday.

    Returns:
        Dict with 'improvers' and 'degraders' lists
    """
    history = load_history()
    today = datetime.now().strftime('%Y-%m-%d')

    improvers = []
    degraders = []

    for ticker, dates in history.items():
        if today not in dates:
            continue

        # Find previous day
        sorted_dates = sorted(dates.keys(), reverse=True)
        prev_dates = [d for d in sorted_dates if d != today]

        if not prev_dates:
            continue

        prev_date = prev_dates[0]
        current_score = dates[today]['total']
        prev_score = dates[prev_date]['total']
        delta = current_score - prev_score

        if delta >= threshold:
            improvers.append((ticker, delta, current_score))
        elif delta <= -threshold:
            degraders.append((ticker, delta, current_score))

    # Sort by magnitude of change
    improvers.sort(key=lambda x: x[1], reverse=True)
    degraders.sort(key=lambda x: x[1])

    return {
        'improvers': improvers,
        'degraders': degraders
    }


if __name__ == "__main__":
    # Test the module
    print("Score History Tracker")
    print("=" * 40)

    history = load_history()
    print(f"Tracking {len(history)} tickers")

    for ticker, dates in list(history.items())[:5]:
        print(f"\n{ticker}: {len(dates)} days of history")
        trend = get_trend_history(ticker, 5)
        for date, score in trend:
            print(f"  {date}: {score:.1f}")
