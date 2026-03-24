"""
Shared Trading Utilities

Common functions and constants used by both ai_trader.py and backtester.py.
Extracted to eliminate duplication and ensure these stay in sync.
"""

import sys
import os

# Add parent directory to path for config_loader import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import config


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _nan_safe(val, default=0):
    """Convert None/NaN to a safe default. float('nan') is truthy and passes `or 0`."""
    if val is None:
        return default
    try:
        if val != val:  # NaN != NaN per IEEE 754
            return default
    except (TypeError, ValueError):
        pass
    return val


def get_strategy_profile(strategy_name: str = "balanced") -> dict:
    """Load strategy profile from YAML config, falling back to balanced defaults."""
    profiles = config.get('strategy_profiles', {})
    profile = profiles.get(strategy_name, profiles.get('balanced', {}))
    return profile


# ---------------------------------------------------------------------------
# ML Signal Layer ordinal maps (must match ml/feature_extractor.py)
# ---------------------------------------------------------------------------

ENTRY_TYPE_MAP_ML = {"breakout": 0, "pre-breakout": 1, "standard": 2}
REGIME_MAP_ML = {"bearish": 0, "neutral": 1, "bullish": 2}


# ---------------------------------------------------------------------------
# Trading allocation limits - loaded from config with fallbacks
# ---------------------------------------------------------------------------

MIN_CASH_RESERVE_PCT = config.get('ai_trader.allocation.min_cash_reserve_pct', default=0.10)
MAX_SECTOR_ALLOCATION = config.get('ai_trader.allocation.max_sector_allocation', default=0.30)
MAX_STOCKS_PER_SECTOR = config.get('ai_trader.allocation.max_stocks_per_sector', default=4)
MAX_POSITION_ALLOCATION = config.get('ai_trader.allocation.max_single_position', default=0.15)
