"""
Shared ticker-identity helpers.

Duplicate ticker groups (same company, different share classes) and the
helpers that deduplicate/expand across them. Lives outside backend.main so
route modules (backend/routes/fidelity.py) can use these without importing
main — a module-level `from backend.main import ...` in a router creates a
circular import that only survives if backend.main happens to be fully
loaded first (it broke whenever main.py was imported cold under the
top-level name `main`, e.g. tests/test_bug_regressions.py run standalone).
"""

DUPLICATE_TICKERS = [
    {'GOOGL', 'GOOG'},  # Alphabet Class A vs Class C
    # Add more pairs here if needed (e.g., BRK.A/BRK.B)
]


def expand_tickers_with_duplicates(tickers: set) -> set:
    """Expand a set of tickers to include all related duplicates"""
    expanded = set(tickers)
    for ticker in list(expanded):
        for group in DUPLICATE_TICKERS:
            if ticker in group:
                expanded.update(group)
    return expanded


def filter_duplicate_stocks(stocks, limit: int):
    """Filter out duplicate tickers, keeping highest scorer from each group"""
    seen_groups = set()
    filtered = []
    for stock in stocks:
        # Check if this ticker belongs to a duplicate group
        ticker_group = None
        for group in DUPLICATE_TICKERS:
            if stock.ticker in group:
                ticker_group = frozenset(group)
                break

        # Skip if we've already seen this group
        if ticker_group and ticker_group in seen_groups:
            continue

        if ticker_group:
            seen_groups.add(ticker_group)

        filtered.append(stock)
        if len(filtered) >= limit:
            break
    return filtered
