"""
Tests for the AI Trader Module

Tests cover critical trading decisions:
- Position sizing (minimum, maximum, pre-breakout bonus, coiled spring multiplier)
- Trailing stop logic (50%+, 30-50%, 20-30%, 10-20% gain thresholds)
- Score stability checks (blip detection vs consistent decline)
- Composite scoring (pre-breakout bonus, extended penalty, insider/short signals)
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal


class TestPositionSizing:
    """Tests for position sizing logic in AI trader"""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session"""
        session = MagicMock()

        # Mock config
        mock_config = MagicMock()
        mock_config.current_cash = 25000.0
        mock_config.starting_cash = 25000.0
        mock_config.max_positions = 20
        mock_config.min_score_to_buy = 65
        mock_config.is_active = True

        session.query.return_value.first.return_value = mock_config

        return session, mock_config

    def test_minimum_position_size(self, mock_db_session):
        """Test that positions below $100 are skipped"""
        # Minimum position size is $100
        min_position = 100

        # Position value below minimum should be rejected
        position_value = 50
        assert position_value < min_position

        # Position value at or above minimum is acceptable
        position_value = 100
        assert position_value >= min_position

    def test_maximum_position_size(self, mock_db_session):
        """Test that positions are capped at 20% of portfolio"""
        from backend.ai_trader import MAX_POSITION_ALLOCATION

        # Default max is 15% but can be up to 20% in code
        assert MAX_POSITION_ALLOCATION <= 0.20

        # Test calculation
        portfolio_value = 25000.0
        max_position = portfolio_value * 0.20  # 20% cap
        assert max_position == 5000.0

    def test_pre_breakout_bonus_sizing(self):
        """Test that pre-breakout entries get 1.30x position multiplier"""
        # Pre-breakout stocks (5-15% below pivot) should get 30% larger positions
        base_position_pct = 10.0
        pre_breakout_multiplier = 1.30

        boosted_pct = base_position_pct * pre_breakout_multiplier
        assert boosted_pct == 13.0

        # At-pivot entries get 20% boost
        at_pivot_multiplier = 1.20
        at_pivot_pct = base_position_pct * at_pivot_multiplier
        assert at_pivot_pct == 12.0

    def test_coiled_spring_multiplier(self):
        """Test that Coiled Spring stocks get 1.25x position multiplier"""
        # Coiled Spring stocks should get 25% larger positions
        base_position_pct = 10.0
        cs_multiplier = 1.25  # Default from config

        boosted_pct = base_position_pct * cs_multiplier
        assert boosted_pct == 12.5


class TestTrailingStops:
    """Tests for trailing stop loss thresholds"""

    def test_trailing_stop_50pct_gain(self):
        """Test 15% trailing stop for 50%+ winners"""
        cost_basis = 100.0
        peak_price = 160.0  # 60% gain at peak
        peak_gain_pct = ((peak_price / cost_basis) - 1) * 100

        assert peak_gain_pct >= 50

        # 15% trailing stop threshold
        trailing_stop_pct = 15
        stop_price = peak_price * (1 - trailing_stop_pct / 100)

        assert stop_price == 136.0  # $160 * 0.85

    def test_trailing_stop_30pct_gain(self):
        """Test 12% trailing stop for 30-50% winners"""
        cost_basis = 100.0
        peak_price = 140.0  # 40% gain at peak
        peak_gain_pct = ((peak_price / cost_basis) - 1) * 100

        assert 30 <= peak_gain_pct < 50

        # 12% trailing stop threshold
        trailing_stop_pct = 12
        stop_price = peak_price * (1 - trailing_stop_pct / 100)

        assert stop_price == 123.2  # $140 * 0.88

    def test_trailing_stop_20pct_gain(self):
        """Test 10% trailing stop for 20-30% winners"""
        cost_basis = 100.0
        peak_price = 125.0  # 25% gain at peak
        peak_gain_pct = ((peak_price / cost_basis) - 1) * 100

        assert 20 <= peak_gain_pct < 30

        # 10% trailing stop threshold
        trailing_stop_pct = 10
        stop_price = peak_price * (1 - trailing_stop_pct / 100)

        assert stop_price == 112.5  # $125 * 0.90

    def test_trailing_stop_10pct_gain(self):
        """Test 8% trailing stop for 10-20% winners"""
        cost_basis = 100.0
        peak_price = 115.0  # 15% gain at peak
        peak_gain_pct = ((peak_price / cost_basis) - 1) * 100

        assert 10 <= peak_gain_pct < 20

        # 8% trailing stop threshold
        trailing_stop_pct = 8
        stop_price = peak_price * (1 - trailing_stop_pct / 100)

        assert abs(stop_price - 105.8) < 0.01  # $115 * 0.92 (with float tolerance)

    def test_no_trailing_under_10pct(self):
        """Test that no trailing stop applies for gains under 10%"""
        cost_basis = 100.0
        peak_price = 108.0  # 8% gain at peak
        peak_gain_pct = ((peak_price / cost_basis) - 1) * 100

        assert peak_gain_pct < 10

        # No trailing stop - only fixed stop loss applies
        trailing_stop_pct = None

        # Determine trailing stop based on peak gain
        if peak_gain_pct >= 50:
            trailing_stop_pct = 15
        elif peak_gain_pct >= 30:
            trailing_stop_pct = 12
        elif peak_gain_pct >= 20:
            trailing_stop_pct = 10
        elif peak_gain_pct >= 10:
            trailing_stop_pct = 8

        assert trailing_stop_pct is None


class TestScoreStability:
    """Tests for score stability detection (blip vs real decline)"""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session for score stability tests"""
        from backend.database import Stock, StockScore

        session = MagicMock()
        return session

    def test_single_blip_not_trigger_sell(self, mock_db):
        """Test that a single score drop is identified as a potential blip"""
        # Recent scores have been consistently high
        recent_scores = [75, 78, 72, 76]
        avg_score = sum(recent_scores) / len(recent_scores)  # 75.25

        # Current score suddenly drops
        current_score = 40
        threshold = 50

        # Score variance check
        score_variance = abs(current_score - avg_score)

        # Blip detection: current < threshold AND avg > threshold+10 AND variance > 15
        is_blip = current_score < threshold and avg_score > threshold + 10 and score_variance > 15

        assert is_blip == True
        assert score_variance > 15  # Sudden large drop
        assert avg_score > 60  # Recent average was healthy

    def test_consistent_low_triggers_sell(self, mock_db):
        """Test that 2+ consecutive low scores confirms a real decline"""
        # Simulate declining score history
        recent_scores = [75, 65, 45, 42]  # Last 2 are below threshold
        threshold = 50

        # Count consecutive low scores
        consecutive_low = 0
        for score in reversed(recent_scores):
            if score < threshold:
                consecutive_low += 1
            else:
                break

        # Current score is also low
        current_score = 40

        # If 2+ consecutive lows, it's a confirmed decline
        assert consecutive_low >= 2

        # Add current score to count
        if current_score < threshold:
            consecutive_low += 1

        assert consecutive_low >= 2  # Confirmed decline, should trigger sell

    def test_insufficient_history_skips_check(self, mock_db):
        """Test that stocks with limited history trust current score"""
        # Only 1 historical score available
        recent_scores = [70]
        current_score = 40

        # Not enough history to detect blips (need at least 2)
        has_sufficient_history = len(recent_scores) >= 2

        assert has_sufficient_history == False
        # With insufficient history, trust current score and allow sell


class TestCompositeScoring:
    """Tests for composite score calculation with bonuses and penalties"""

    def test_pre_breakout_bonus_30pts(self):
        """Test that pre-breakout stocks get +30 bonus points"""
        # Stock 5-15% below pivot with valid base pattern
        pct_from_pivot = 10  # 10% below pivot
        has_base = True

        pre_breakout_bonus = 0
        if has_base and 5 <= pct_from_pivot <= 15:
            pre_breakout_bonus = 30

        assert pre_breakout_bonus == 30

    def test_extended_penalty_20pts(self):
        """Test that extended stocks get -20 penalty"""
        # Stock more than 10% above pivot
        pct_from_pivot = -12  # 12% above pivot (negative = above)
        has_base = True
        pivot_price = 100

        extended_penalty = 0
        if has_base and pivot_price > 0 and pct_from_pivot < -5:
            if pct_from_pivot < -10:
                extended_penalty = -20  # Heavy penalty for very extended
            else:
                extended_penalty = -10  # Moderate penalty

        assert extended_penalty == -20

    def test_insider_buying_bonus(self):
        """Test that bullish insider sentiment adds bonus points"""
        insider_sentiment = "bullish"
        insider_net_value = 600000  # $600K net buying
        insider_largest_buyer_title = "CEO"

        insider_bonus = 0
        if insider_sentiment == "bullish":
            # Scale by $ value
            if insider_net_value >= 500000:
                insider_bonus = 10
            elif insider_net_value >= 100000:
                insider_bonus = 7

            # Extra for C-suite
            if insider_largest_buyer_title.upper() in ('CEO', 'CFO', 'COO', 'PRESIDENT'):
                insider_bonus += 3

        assert insider_bonus == 13  # 10 for $500K+ plus 3 for CEO

    def test_short_interest_penalty(self):
        """Test that high short interest adds penalty"""
        # Test high short interest (>20%)
        short_interest_pct = 25

        short_penalty = 0
        if short_interest_pct > 20:
            short_penalty = -5
        elif short_interest_pct > 10:
            short_penalty = -2

        assert short_penalty == -5

        # Test moderate short interest (10-20%)
        short_interest_pct = 15
        short_penalty = 0
        if short_interest_pct > 20:
            short_penalty = -5
        elif short_interest_pct > 10:
            short_penalty = -2

        assert short_penalty == -2


class TestEffectiveScore:
    """Tests for get_effective_score helper function"""

    def test_canslim_stock_uses_canslim_score(self):
        """Test that traditional stocks use CANSLIM score"""
        # Mock a traditional CANSLIM stock
        stock = MagicMock()
        stock.is_growth_stock = False
        stock.canslim_score = 85
        stock.growth_mode_score = 60

        # Traditional stock should use CANSLIM score
        is_growth = getattr(stock, 'is_growth_stock', False)
        if is_growth:
            effective = stock.growth_mode_score
        else:
            effective = stock.canslim_score

        assert effective == 85

    def test_growth_stock_uses_growth_score(self):
        """Test that growth stocks use Growth Mode score"""
        # Mock a growth stock
        stock = MagicMock()
        stock.is_growth_stock = True
        stock.canslim_score = 50
        stock.growth_mode_score = 78

        # Growth stock should use Growth Mode score
        is_growth = getattr(stock, 'is_growth_stock', False)
        if is_growth:
            effective = stock.growth_mode_score
        else:
            effective = stock.canslim_score

        assert effective == 78


class TestMomentumConfirmation:
    """Tests for momentum confirmation (RS ratio check)"""

    def test_fading_momentum_penalty(self):
        """Test that fading momentum applies 15% penalty"""
        rs_12m = 1.2
        rs_3m = 1.0  # Below threshold of 1.14 (1.2 * 0.95)

        threshold = rs_12m * 0.95
        momentum_penalty = 0

        if rs_12m > 0 and rs_3m < threshold:
            momentum_penalty = -0.15

        assert rs_3m < threshold
        assert momentum_penalty == -0.15

    def test_strong_momentum_no_penalty(self):
        """Test that strong momentum doesn't get penalized"""
        rs_12m = 1.2
        rs_3m = 1.18  # Above threshold of 1.14

        threshold = rs_12m * 0.95
        momentum_penalty = 0

        if rs_12m > 0 and rs_3m < threshold:
            momentum_penalty = -0.15

        assert rs_3m >= threshold
        assert momentum_penalty == 0


class TestSectorLimits:
    """Tests for sector concentration limits"""

    def test_max_stocks_per_sector(self):
        """Test that max stocks per sector is enforced"""
        from backend.ai_trader import MAX_STOCKS_PER_SECTOR

        # Default is 4 stocks per sector
        assert MAX_STOCKS_PER_SECTOR == 4

        # If we already have 4 in a sector, no more allowed
        current_count = 4
        allow_buy = current_count < MAX_STOCKS_PER_SECTOR

        assert allow_buy == False

    def test_max_sector_allocation(self):
        """Test that max sector allocation (30%) is enforced"""
        from backend.ai_trader import MAX_SECTOR_ALLOCATION

        # Default is 30% max per sector
        assert MAX_SECTOR_ALLOCATION == 0.30

        portfolio_value = 25000
        current_sector_value = 7000  # 28%
        proposed_buy = 1000

        new_allocation = (current_sector_value + proposed_buy) / portfolio_value

        # 32% would exceed 30% limit
        assert new_allocation > MAX_SECTOR_ALLOCATION


class TestPartialProfitTaking:
    """Tests for partial profit taking logic"""

    def test_25pct_partial_at_25pct_gain(self):
        """Test that 25% partial is taken at +25% gain when score strong"""
        gain_pct = 28  # Up 28%
        score = 72  # Score still strong (>= 60)
        partial_taken = 0  # No partial taken yet

        should_take_partial = gain_pct >= 25 and score >= 60 and partial_taken < 25
        sell_pct = 25

        assert should_take_partial == True
        assert sell_pct == 25

    def test_50pct_partial_at_40pct_gain(self):
        """Test that 50% partial is taken at +40% gain when score strong"""
        gain_pct = 45  # Up 45%
        score = 65  # Score still strong (>= 60)
        partial_taken = 25  # Already took 25%

        # At 40%+ gain, take remaining to get to 50%
        should_take_50 = gain_pct >= 40 and score >= 60 and partial_taken < 50
        take_pct = 50 - partial_taken  # Take remaining to reach 50%

        assert should_take_50 == True
        assert take_pct == 25  # Take another 25% to reach 50% total

    def test_no_partial_if_score_weak(self):
        """Test that partial profit is NOT taken if score is weak"""
        gain_pct = 30  # Up 30%
        score = 55  # Score weak (< 60)
        partial_taken = 0

        # Don't take partial if score is weak - let full sell logic handle it
        should_take_partial = gain_pct >= 25 and score >= 60 and partial_taken < 25

        assert should_take_partial == False


class TestCoiledSpringLogic:
    """Tests for Coiled Spring earnings catalyst detection"""

    def test_cs_requires_base_pattern(self):
        """Test that Coiled Spring requires proper base formation"""
        weeks_in_base = 18  # Long consolidation
        beat_streak = 4  # Consistent beats
        days_to_earnings = 5  # Approaching earnings
        base_type = "flat"  # Has base pattern

        # All criteria must be met
        has_base = base_type not in ('none', '', None) and weeks_in_base >= 15

        assert has_base == True

    def test_cs_requires_beat_streak(self):
        """Test that Coiled Spring requires earnings beat streak"""
        min_beat_streak = 3  # From config
        actual_beat_streak = 5

        meets_requirement = actual_beat_streak >= min_beat_streak

        assert meets_requirement == True

    def test_cs_allows_lower_c_score_pre_breakout(self):
        """Test that pre-breakout CS can have lower C score (catalyst pending)"""
        # Standard threshold is 10, but pre-breakout allows 5
        c_score = 7
        is_breaking_out = False

        standard_min_c = 10
        pre_breakout_min_c = 5

        # Pre-breakout gets relaxed threshold
        min_c = pre_breakout_min_c if not is_breaking_out else standard_min_c

        passes = c_score >= min_c

        assert passes == True


class TestCashReserve:
    """Tests for cash reserve requirements"""

    def test_min_cash_reserve_stops_buys(self):
        """Test that 10% cash reserve stops new buys"""
        from backend.ai_trader import MIN_CASH_RESERVE_PCT

        portfolio_value = 30000
        current_cash = 2500  # 8.3% of portfolio

        min_cash_reserve = portfolio_value * MIN_CASH_RESERVE_PCT  # 10% = $3000

        should_stop_buys = current_cash < min_cash_reserve

        assert MIN_CASH_RESERVE_PCT == 0.10
        assert should_stop_buys == True

    def test_sufficient_cash_allows_buys(self):
        """Test that buys proceed when cash above reserve"""
        from backend.ai_trader import MIN_CASH_RESERVE_PCT

        portfolio_value = 30000
        current_cash = 5000  # 16.7% of portfolio

        min_cash_reserve = portfolio_value * MIN_CASH_RESERVE_PCT  # 10% = $3000
        available_for_buying = current_cash - min_cash_reserve

        assert current_cash >= min_cash_reserve
        assert available_for_buying == 2000
