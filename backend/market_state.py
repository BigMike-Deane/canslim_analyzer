"""
Market State Machine for CANSLIM Trading

Implements O'Neil's Follow-Through Day (FTD) methodology with IBD Market School
graduated exposure system. Replaces the binary regime gate (SPY < 50MA = block all)
with a 5-state system that provides structured re-entry after corrections.

CRITICAL: In a bull market (SPY > 50MA), the state stays TRENDING at 100% exposure,
making behavior IDENTICAL to the previous system. The graduated states only activate
during/after corrections.

States:
- TRENDING:    100% max exposure — full buying + pyramids (default state)
- PRESSURE:     60% max exposure — distribution accumulating, reduce new buys
- CORRECTION:    0% max exposure — no new buys, protect capital
- RECOVERY:     30% max exposure — FTD confirmed, cautious re-entry (2-3 positions)
- CONFIRMED:    55% max exposure — recovery validated, normal buying resumes

References:
- O'Neil, "How to Make Money in Stocks" (Follow-Through Day)
- IBD Market School (graduated exposure count system)
- Faber, "A Quantitative Approach to Tactical Asset Allocation" (200-day MA)
"""

from enum import Enum
from datetime import date
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MarketState(Enum):
    TRENDING = "trending"
    PRESSURE = "pressure"
    CORRECTION = "correction"
    RECOVERY = "recovery"
    CONFIRMED = "confirmed"


# Default configuration
DEFAULT_CONFIG = {
    "enabled": True,

    # Exposure limits per state (fraction of portfolio)
    "exposure_limits": {
        "trending": 1.00,
        "pressure": 0.60,
        "correction": 0.00,
        "recovery": 0.30,
        "confirmed": 0.55,
    },

    # FTD detection thresholds
    "ftd": {
        "min_gain_pct": 1.7,          # Stricter than O'Neil's 1.25%
        "earliest_day": 4,             # Day 4 of rally attempt
        "latest_day": 25,              # Max days to wait for FTD
    },

    # Distribution day tracking
    "distribution": {
        "min_decline_pct": -0.2,       # Minimum decline to count
        "expiry_calendar_days": 35,    # ~25 trading days
        "rally_off_pct": 6.0,          # % rally removes distribution day
        "pressure_threshold": 5,       # Count that triggers PRESSURE
    },

    # State transition thresholds
    "transitions": {
        "correction_below_ma50_pct": 0.0,   # SPY must be below 50MA to enter CORRECTION from PRESSURE
        "trending_deep_drop_pct": 3.0,      # % below 50MA for direct TRENDING → CORRECTION
        "pressure_below_21ema_days": 3,     # Days below 21EMA + below 50MA = PRESSURE
        "recovery_confirm_days": 3,         # Days above 21EMA to advance RECOVERY → CONFIRMED
        "trending_confirm_days": 5,         # Days above 21EMA to advance CONFIRMED → TRENDING
        "trending_max_dist_days": 3,        # Max distribution days to re-enter TRENDING
        "min_correction_days": 3,           # Minimum days in CORRECTION before allowing fast-track exit
        "min_confirmed_days": 3,            # Minimum days in CONFIRMED before allowing back to CORRECTION
        # "post_correction_damping_days": 10, # DISABLED: caused -4.9pp bull regression by capping pyramids
    },

    # Recovery seeding (when portfolio is depleted after a correction)
    "recovery_seed": {
        "enabled": True,
        "min_score": 45,               # Lower than normal (72) since crash depresses all scores
        "max_positions": 3,            # Match normal seed count for adequate diversification
        "max_exposure_pct": 50,        # Match normal max_seed_investment_pct
        "seed_pct": 12,                # Slightly larger per-position (vs normal 10%)
        "min_l_score": 6,              # Lower than normal (8) since RS is meaningless after crash
    },
}


class MarketStateManager:
    """
    Graduated market exposure system inspired by IBD Market School.

    In a bull market (SPY consistently above 50MA), the state stays TRENDING
    with 100% exposure — making behavior identical to the previous binary
    regime gate system.

    The graduated states only activate during/after corrections, providing
    a structured re-entry path instead of the previous binary on/off gate.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.state = MarketState.TRENDING  # Start optimistic

        # FTD / rally tracking
        self.rally_attempt_day: int = 0
        self.rally_attempt_low: Optional[float] = None
        self.rally_attempt_start_date: Optional[date] = None
        self.last_ftd_date: Optional[date] = None

        # Distribution day tracking
        self.distribution_days: List[dict] = []

        # Moving average relationship tracking
        self.consecutive_above_21ema: int = 0
        self.consecutive_below_21ema: int = 0

        # Recovery confirmation counter
        self.recovery_confirmations: int = 0

        # State entry date tracking (for minimum stay enforcement)
        self.state_entry_date: Optional[date] = None
        self.state_days_count: int = 0

        # State change history for debugging/analysis
        self.state_history: List[dict] = []

        # Post-correction damping: DISABLED — caused -4.9pp bull regression
        # by keeping system in PRESSURE (60% exposure) and reducing pyramids (24→19).
        # The jitter prevention is handled by min_correction_days and min_confirmed_days instead.
        self.last_correction_exit_date: Optional[date] = None

    @property
    def max_exposure_pct(self) -> float:
        """Maximum portfolio exposure (0.0 to 1.0) allowed in current state."""
        limits = self.config.get("exposure_limits", DEFAULT_CONFIG["exposure_limits"])
        return limits.get(self.state.value, 1.0)

    @property
    def can_buy(self) -> bool:
        """Whether new buys are allowed (exposure > 0)."""
        return self.state != MarketState.CORRECTION

    @property
    def can_seed(self) -> bool:
        """Whether recovery seeding is allowed."""
        return self.state in (
            MarketState.RECOVERY,
            MarketState.CONFIRMED,
            MarketState.TRENDING,
        )

    @property
    def position_size_multiplier(self) -> float:
        """Scale factor for position sizes based on market state."""
        if self.state == MarketState.RECOVERY:
            return 0.60  # Smaller positions during early recovery
        elif self.state == MarketState.PRESSURE:
            return 0.75  # Slightly smaller during pressure
        elif self.state == MarketState.CONFIRMED:
            return 0.85  # Moderate sizing during confirmation
        return 1.0  # Full size in TRENDING

    def initialize_state(self, spy_close: float, spy_ma50: float,
                         spy_ema21: float = 0) -> None:
        """
        Set initial state based on market conditions at backtest start.
        Ensures day-1 behavior matches the current system.
        """
        deep_drop_pct = self.config.get("transitions", {}).get("trending_deep_drop_pct", 3.0)

        if spy_close > spy_ma50:
            self.state = MarketState.TRENDING
        elif spy_close > spy_ma50 * (1 - deep_drop_pct / 100):
            self.state = MarketState.PRESSURE
        else:
            self.state = MarketState.CORRECTION
            self.rally_attempt_low = spy_close

        logger.info(f"Market state initialized: {self.state.value} "
                    f"(SPY ${spy_close:.2f}, 50MA ${spy_ma50:.2f})")

    def update(self, current_date: date, spy_close: float, spy_prev_close: float,
               spy_volume: float, spy_prev_volume: float,
               spy_ma50: float, spy_ema21: float,
               spy_ma200: float = 0) -> dict:
        """
        Update market state based on today's SPY data. Called once per trading day.

        Returns:
            dict with state info: state, max_exposure, can_buy, can_seed, changed
        """
        old_state = self.state

        # --- Track days in current state ---
        self.state_days_count += 1

        # --- Update distribution days ---
        self._update_distribution_days(current_date, spy_close, spy_prev_close,
                                       spy_volume, spy_prev_volume)

        # --- Update 21 EMA tracking ---
        if spy_ema21 > 0:
            if spy_close > spy_ema21:
                self.consecutive_above_21ema += 1
                self.consecutive_below_21ema = 0
            else:
                self.consecutive_below_21ema += 1
                self.consecutive_above_21ema = 0

        # --- State transitions ---
        dist_count = len(self.distribution_days)

        if self.state == MarketState.TRENDING:
            self._check_trending_exit(current_date, spy_close, spy_ma50,
                                      spy_ema21, dist_count)

        elif self.state == MarketState.PRESSURE:
            self._check_pressure_transitions(current_date, spy_close, spy_ma50,
                                             spy_ema21, dist_count)

        elif self.state == MarketState.CORRECTION:
            self._check_correction_exit(current_date, spy_close, spy_prev_close,
                                        spy_volume, spy_prev_volume,
                                        spy_ma50, spy_ema21)

        elif self.state == MarketState.RECOVERY:
            self._check_recovery_transitions(current_date, spy_close, spy_ma50,
                                             spy_ema21, dist_count)

        elif self.state == MarketState.CONFIRMED:
            self._check_confirmed_transitions(current_date, spy_close, spy_ma50,
                                              spy_ema21, dist_count)

        # Log state changes
        changed = self.state != old_state
        if changed:
            # Track correction exits for post-correction damping
            if old_state == MarketState.CORRECTION and self.state != MarketState.CORRECTION:
                self.last_correction_exit_date = current_date

            self.state_entry_date = current_date
            self.state_days_count = 0
            change_record = {
                "date": current_date,
                "from": old_state.value,
                "to": self.state.value,
                "spy": spy_close,
                "ma50": spy_ma50,
                "ema21": spy_ema21,
                "dist_count": dist_count,
            }
            self.state_history.append(change_record)
            logger.info(
                f"MARKET STATE: {old_state.value} -> {self.state.value} on {current_date} "
                f"(SPY ${spy_close:.2f}, 50MA ${spy_ma50:.2f}, 21EMA ${spy_ema21:.2f}, "
                f"dist_days={dist_count})"
            )

        return {
            "state": self.state.value,
            "max_exposure": self.max_exposure_pct,
            "can_buy": self.can_buy,
            "can_seed": self.can_seed,
            "dist_count": dist_count,
            "changed": changed,
        }

    # ---- Distribution day tracking ----

    def _update_distribution_days(self, current_date: date, spy_close: float,
                                   spy_prev_close: float, spy_volume: float,
                                   spy_prev_volume: float) -> None:
        """Track and expire distribution days (institutional selling)."""
        if spy_prev_close <= 0:
            return

        min_decline = self.config.get("distribution", {}).get("min_decline_pct", -0.2)
        expiry_days = self.config.get("distribution", {}).get("expiry_calendar_days", 35)
        rally_off = self.config.get("distribution", {}).get("rally_off_pct", 6.0)

        pct_change = ((spy_close - spy_prev_close) / spy_prev_close) * 100
        volume_up = spy_volume > spy_prev_volume

        # New distribution day: decline on higher volume
        if pct_change <= min_decline and volume_up:
            self.distribution_days.append({
                "date": current_date,
                "close": spy_close,
                "decline_pct": pct_change,
            })

        # Expire old and rallied-off distribution days
        self.distribution_days = [
            d for d in self.distribution_days
            if (current_date - d["date"]).days <= expiry_days
            and ((spy_close - d["close"]) / d["close"] * 100) < rally_off
        ]

    # ---- State transition methods ----

    def _check_trending_exit(self, current_date: date, spy_close: float,
                              spy_ma50: float, spy_ema21: float,
                              dist_count: int) -> None:
        """TRENDING -> PRESSURE or CORRECTION."""
        transitions = self.config.get("transitions", {})
        deep_drop_pct = transitions.get("trending_deep_drop_pct", 3.0)
        pressure_threshold = self.config.get("distribution", {}).get("pressure_threshold", 5)
        below_21ema_days = transitions.get("pressure_below_21ema_days", 3)

        # Direct to CORRECTION if SPY drops well below 50MA
        if spy_ma50 > 0 and spy_close < spy_ma50 * (1 - deep_drop_pct / 100):
            self.state = MarketState.CORRECTION
            self._start_rally_tracking(spy_close, current_date)
            return

        # To PRESSURE if distribution accumulates
        if dist_count >= pressure_threshold:
            self.state = MarketState.PRESSURE
            return

        # To PRESSURE if below 21EMA for multiple days AND below 50MA
        if (self.consecutive_below_21ema >= below_21ema_days
                and spy_ma50 > 0 and spy_close < spy_ma50):
            self.state = MarketState.PRESSURE

    def _check_pressure_transitions(self, current_date: date, spy_close: float,
                                     spy_ma50: float, spy_ema21: float,
                                     dist_count: int) -> None:
        """PRESSURE -> CORRECTION or back to TRENDING."""
        transitions = self.config.get("transitions", {})
        trending_max_dist = transitions.get("trending_max_dist_days", 3)

        # To CORRECTION if SPY breaks below 50MA
        if spy_ma50 > 0 and spy_close < spy_ma50:
            self.state = MarketState.CORRECTION
            self._start_rally_tracking(spy_close, current_date)
            return

        # Back to TRENDING if pressure relieved:
        # Distribution days drop AND above 21 EMA for several days
        if (dist_count < trending_max_dist
                and spy_ema21 > 0 and spy_close > spy_ema21
                and self.consecutive_above_21ema >= 3):
            self.state = MarketState.TRENDING

    def _check_correction_exit(self, current_date: date, spy_close: float,
                                spy_prev_close: float, spy_volume: float,
                                spy_prev_volume: float, spy_ma50: float,
                                spy_ema21: float) -> None:
        """CORRECTION -> RECOVERY (via FTD) or fast-track to TRENDING."""
        ftd_config = self.config.get("ftd", {})
        transitions = self.config.get("transitions", {})
        min_gain = ftd_config.get("min_gain_pct", 1.7)
        earliest_day = ftd_config.get("earliest_day", 4)
        latest_day = ftd_config.get("latest_day", 25)
        min_correction_days = transitions.get("min_correction_days", 3)

        # FAST-TRACK: If SPY recovers above 50MA AND 21EMA, skip directly to TRENDING.
        # But only after min_correction_days to prevent jittery bouncing on mild pullbacks.
        if (spy_ma50 > 0 and spy_close > spy_ma50
                and spy_ema21 > 0 and spy_close > spy_ema21
                and self.state_days_count >= min_correction_days):
            self.state = MarketState.TRENDING
            self.rally_attempt_day = 0
            self.recovery_confirmations = 0
            logger.info(f"FAST-TRACK: SPY back above 50MA+21EMA on {current_date} "
                       f"(after {self.state_days_count} days in correction), skipping to TRENDING")
            return

        # Track rally attempt
        if spy_prev_close > 0 and spy_close > spy_prev_close:
            # Up day
            if self.rally_attempt_day == 0:
                # Start new rally attempt
                self._start_rally_tracking(spy_close, current_date)
            self.rally_attempt_day += 1
        elif self.rally_attempt_low is not None and spy_close < self.rally_attempt_low:
            # Rally attempt failed — undercut the low
            self._start_rally_tracking(spy_close, current_date)

        # Check for FTD: Day 4+, SPY up >= 1.7% on higher volume
        if (earliest_day <= self.rally_attempt_day <= latest_day
                and spy_prev_close > 0):
            gain_pct = ((spy_close - spy_prev_close) / spy_prev_close) * 100
            volume_up = spy_volume > spy_prev_volume

            if gain_pct >= min_gain and volume_up:
                self.state = MarketState.RECOVERY
                self.last_ftd_date = current_date
                self.recovery_confirmations = 1
                self.distribution_days = []  # Reset on new uptrend
                logger.info(
                    f"FTD DETECTED on {current_date}: +{gain_pct:.1f}% on higher volume, "
                    f"day {self.rally_attempt_day} of rally"
                )

    def _check_recovery_transitions(self, current_date: date, spy_close: float,
                                     spy_ma50: float, spy_ema21: float,
                                     dist_count: int) -> None:
        """RECOVERY -> CONFIRMED, TRENDING, or back to CORRECTION."""
        transitions = self.config.get("transitions", {})
        confirm_days = transitions.get("recovery_confirm_days", 3)

        # Back to CORRECTION if rally fails (undercuts rally low)
        if self.rally_attempt_low is not None and spy_close < self.rally_attempt_low:
            self.state = MarketState.CORRECTION
            self.recovery_confirmations = 0
            self._start_rally_tracking(spy_close, current_date)
            logger.info(f"RECOVERY FAILED on {current_date}: undercut rally low "
                       f"${self.rally_attempt_low:.2f}")
            return

        # Fast-track to TRENDING if above 50MA with strong confirmation
        if (spy_ma50 > 0 and spy_close > spy_ma50
                and spy_ema21 > 0 and spy_close > spy_ema21
                and self.consecutive_above_21ema >= 3):
            self.state = MarketState.TRENDING
            return

        # Advance to CONFIRMED with additional confirmations (days above 21 EMA)
        if spy_ema21 > 0 and spy_close > spy_ema21:
            self.recovery_confirmations += 1

        if self.recovery_confirmations >= confirm_days:
            self.state = MarketState.CONFIRMED

    def _check_confirmed_transitions(self, current_date: date, spy_close: float,
                                      spy_ma50: float, spy_ema21: float,
                                      dist_count: int) -> None:
        """CONFIRMED -> TRENDING, PRESSURE, or CORRECTION."""
        transitions = self.config.get("transitions", {})
        trending_days = transitions.get("trending_confirm_days", 5)
        trending_max_dist = transitions.get("trending_max_dist_days", 3)
        deep_drop_pct = transitions.get("trending_deep_drop_pct", 3.0)
        pressure_threshold = self.config.get("distribution", {}).get("pressure_threshold", 5)
        min_confirmed_days = transitions.get("min_confirmed_days", 3)

        # To TRENDING when fully recovered
        if (spy_ma50 > 0 and spy_close > spy_ma50
                and self.consecutive_above_21ema >= trending_days
                and dist_count < trending_max_dist):
            self.state = MarketState.TRENDING
            return

        # Back to CORRECTION if market breaks down again — but only after min_confirmed_days
        # to prevent the immediate CONFIRMED→CORRECTION reversal seen in COVID (Apr 8→Apr 9).
        if (spy_ma50 > 0 and spy_close < spy_ma50 * (1 - deep_drop_pct / 100)
                and self.state_days_count >= min_confirmed_days):
            self.state = MarketState.CORRECTION
            self._start_rally_tracking(spy_close, current_date)
            return

        # To PRESSURE if distribution builds
        if dist_count >= pressure_threshold:
            self.state = MarketState.PRESSURE

    # ---- Helper methods ----

    def _start_rally_tracking(self, spy_close: float, current_date: date) -> None:
        """Initialize or reset rally attempt tracking."""
        self.rally_attempt_day = 0
        self.rally_attempt_low = spy_close
        self.rally_attempt_start_date = current_date

    @property
    def last_transition_was_fast_track(self) -> bool:
        """Whether the most recent state change was a fast-track from CORRECTION."""
        if not self.state_history:
            return False
        last = self.state_history[-1]
        return last["from"] == "correction" and last["to"] == "trending"

    def get_state_summary(self) -> dict:
        """Get current state summary for logging/API."""
        return {
            "state": self.state.value,
            "max_exposure_pct": self.max_exposure_pct,
            "can_buy": self.can_buy,
            "can_seed": self.can_seed,
            "position_size_multiplier": self.position_size_multiplier,
            "distribution_days": len(self.distribution_days),
            "rally_attempt_day": self.rally_attempt_day,
            "last_ftd_date": str(self.last_ftd_date) if self.last_ftd_date else None,
            "state_changes": len(self.state_history),
        }
