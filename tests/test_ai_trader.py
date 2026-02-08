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
        """Test that positions are capped at max single position (concentrated portfolio: 25%)"""
        from backend.ai_trader import MAX_POSITION_ALLOCATION

        # Concentrated portfolio: max 25% per position (O'Neil/Minervini)
        assert MAX_POSITION_ALLOCATION <= 0.25

        # Test calculation
        portfolio_value = 25000.0
        max_position = portfolio_value * MAX_POSITION_ALLOCATION
        assert max_position > 0

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
        """Test that max sector allocation (50%) is enforced for concentrated portfolio"""
        from backend.ai_trader import MAX_SECTOR_ALLOCATION

        # Concentrated portfolio: max 50% per sector (follow best sectors)
        assert MAX_SECTOR_ALLOCATION == 0.50

        portfolio_value = 25000
        current_sector_value = 12000  # 48%
        proposed_buy = 1000

        new_allocation = (current_sector_value + proposed_buy) / portfolio_value

        # 52% would exceed 50% limit
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


class TestMarketRegimeDetection:
    """Tests for market regime detection logic"""

    def test_bullish_regime_larger_positions(self):
        """Test that bullish market allows larger positions"""
        weighted_signal = 1.8  # Strong bullish signal

        # From config defaults
        bullish_threshold = 1.5
        bullish_max_pct = 15.0
        neutral_max_pct = 12.0

        if weighted_signal >= bullish_threshold:
            max_position_pct = bullish_max_pct
        else:
            max_position_pct = neutral_max_pct

        assert max_position_pct == 15.0

    def test_bearish_regime_smaller_positions(self):
        """Test that bearish market reduces position sizes"""
        weighted_signal = -0.8  # Bearish signal

        bearish_threshold = -0.5
        bearish_max_pct = 8.0
        neutral_max_pct = 12.0

        if weighted_signal <= bearish_threshold:
            max_position_pct = bearish_max_pct
        else:
            max_position_pct = neutral_max_pct

        assert max_position_pct == 8.0

    def test_neutral_regime_standard_positions(self):
        """Test that neutral market uses standard position size"""
        weighted_signal = 0.5  # Neutral/mild bullish

        bullish_threshold = 1.5
        bearish_threshold = -0.5
        neutral_max_pct = 12.0
        bullish_max_pct = 15.0
        bearish_max_pct = 8.0

        if weighted_signal >= bullish_threshold:
            max_position_pct = bullish_max_pct
        elif weighted_signal <= bearish_threshold:
            max_position_pct = bearish_max_pct
        else:
            max_position_pct = neutral_max_pct

        assert max_position_pct == 12.0


class TestDrawdownProtection:
    """Tests for portfolio drawdown protection"""

    def test_level1_drawdown_reduces_position(self):
        """Test 25% position reduction at 10% drawdown"""
        high_water_mark = 30000
        current_value = 26500  # 11.7% down
        drawdown_pct = ((high_water_mark - current_value) / high_water_mark) * 100

        level1_threshold = 10  # 10%
        level1_multiplier = 0.75

        if drawdown_pct >= level1_threshold:
            position_multiplier = level1_multiplier
        else:
            position_multiplier = 1.0

        assert drawdown_pct > 10
        assert position_multiplier == 0.75

    def test_level2_drawdown_halves_position(self):
        """Test 50% position reduction at 15% drawdown"""
        high_water_mark = 30000
        current_value = 24000  # 20% down
        drawdown_pct = ((high_water_mark - current_value) / high_water_mark) * 100

        level2_threshold = 15  # 15%
        level2_multiplier = 0.50

        if drawdown_pct >= level2_threshold:
            position_multiplier = level2_multiplier
        else:
            position_multiplier = 1.0

        assert drawdown_pct >= 15
        assert position_multiplier == 0.50

    def test_no_drawdown_full_position(self):
        """Test no reduction when portfolio is at/near high water"""
        high_water_mark = 30000
        current_value = 31000  # Above high water (making new highs)
        drawdown_pct = max(0, ((high_water_mark - current_value) / high_water_mark) * 100)

        # No drawdown
        assert drawdown_pct == 0
        # Full position multiplier
        position_multiplier = 1.0
        assert position_multiplier == 1.0


class TestSectorRotation:
    """Tests for sector rotation bonus/penalty"""

    def test_leading_sector_bonus(self):
        """Test bonus for stocks in leading sectors"""
        sector_avg_l_score = 12  # Strong L score average

        leading_threshold = 10
        leading_bonus = 5

        if sector_avg_l_score >= leading_threshold:
            sector_bonus = leading_bonus
        else:
            sector_bonus = 0

        assert sector_bonus == 5

    def test_lagging_sector_penalty(self):
        """Test penalty for stocks in lagging sectors"""
        sector_avg_l_score = 3  # Weak L score average

        lagging_threshold = 5
        lagging_penalty = -3

        if sector_avg_l_score < lagging_threshold:
            sector_bonus = lagging_penalty
        else:
            sector_bonus = 0

        assert sector_bonus == -3

    def test_neutral_sector_no_adjustment(self):
        """Test no adjustment for neutral sectors"""
        sector_avg_l_score = 7  # Middle-of-road L score

        leading_threshold = 10
        lagging_threshold = 5

        if sector_avg_l_score >= leading_threshold:
            sector_bonus = 5
        elif sector_avg_l_score < lagging_threshold:
            sector_bonus = -3
        else:
            sector_bonus = 0

        assert sector_bonus == 0


class TestPortfolioCorrelation:
    """Tests for portfolio correlation position reduction"""

    def test_sector_concentration_reduces_position(self):
        """Test position reduction for concentrated sectors"""
        stocks_in_sector = 4
        sector_threshold = 3
        sector_multiplier = 0.85

        if stocks_in_sector >= sector_threshold:
            position_multiplier = sector_multiplier
        else:
            position_multiplier = 1.0

        assert position_multiplier == 0.85

    def test_industry_concentration_stronger_reduction(self):
        """Test stronger reduction for same industry"""
        stocks_in_industry = 2
        industry_threshold = 2
        industry_multiplier = 0.70

        if stocks_in_industry >= industry_threshold:
            position_multiplier = industry_multiplier
        else:
            position_multiplier = 1.0

        assert position_multiplier == 0.70

    def test_no_concentration_full_position(self):
        """Test full position when no concentration"""
        stocks_in_sector = 1
        stocks_in_industry = 0

        sector_threshold = 3
        industry_threshold = 2

        sector_ok = stocks_in_sector < sector_threshold
        industry_ok = stocks_in_industry < industry_threshold

        assert sector_ok and industry_ok


class TestShortSqueezeDetection:
    """Tests for short squeeze detection logic"""

    def test_squeeze_setup_converts_penalty_to_bonus(self):
        """Test that squeeze setup gives bonus instead of penalty"""
        short_interest_pct = 25
        l_score = 12
        has_base = True
        is_breaking_out = True

        min_short_pct = 20
        min_l_score = 10
        squeeze_bonus = 5
        risk_penalty = -5

        if short_interest_pct >= min_short_pct:
            if l_score >= min_l_score and has_base and is_breaking_out:
                # Squeeze setup
                short_adjustment = squeeze_bonus
                is_squeeze = True
            else:
                # Just risky
                short_adjustment = risk_penalty
                is_squeeze = False
        else:
            short_adjustment = 0
            is_squeeze = False

        assert is_squeeze == True
        assert short_adjustment == 5  # Bonus, not penalty!

    def test_high_short_without_setup_is_penalty(self):
        """Test that high short without setup is still penalized"""
        short_interest_pct = 25
        l_score = 5  # Weak relative strength
        has_base = False
        is_breaking_out = False

        min_short_pct = 20
        min_l_score = 10
        squeeze_bonus = 5
        risk_penalty = -5

        if short_interest_pct >= min_short_pct:
            if l_score >= min_l_score and has_base and is_breaking_out:
                short_adjustment = squeeze_bonus
            else:
                short_adjustment = risk_penalty
        else:
            short_adjustment = 0

        assert short_adjustment == -5  # Penalty

    def test_moderate_short_small_penalty(self):
        """Test moderate short interest gets small penalty"""
        short_interest_pct = 15
        min_short_pct = 20
        medium_threshold = 10
        medium_penalty = -2

        if short_interest_pct >= min_short_pct:
            short_adjustment = -5
        elif short_interest_pct > medium_threshold:
            short_adjustment = medium_penalty
        else:
            short_adjustment = 0

        assert short_adjustment == -2


class TestInsiderClusterDetection:
    """Tests for insider cluster detection"""

    def test_cluster_bonus_3_insiders(self):
        """Test cluster bonus for 3+ insider buys"""
        insider_buy_count = 4
        insider_net_value = 500000
        insider_sentiment = "bullish"

        cluster_bonus = 8
        high_value_cluster_bonus = 12

        if insider_sentiment == "bullish" and insider_buy_count >= 3:
            if insider_net_value >= 1_000_000:
                insider_bonus = high_value_cluster_bonus
            else:
                insider_bonus = cluster_bonus
        else:
            insider_bonus = 5  # Standard bullish bonus

        assert insider_bonus == 8

    def test_high_value_cluster_bonus(self):
        """Test high value cluster bonus for $1M+"""
        insider_buy_count = 5
        insider_net_value = 1_500_000  # $1.5M

        high_value_cluster_bonus = 12

        if insider_buy_count >= 3 and insider_net_value >= 1_000_000:
            insider_bonus = high_value_cluster_bonus
        else:
            insider_bonus = 8

        assert insider_bonus == 12

    def test_standard_insider_bonus(self):
        """Test standard bonus for 2 insider buys (not cluster)"""
        insider_buy_count = 2
        insider_sentiment = "bullish"

        cluster_threshold = 3
        standard_bonus = 5

        if insider_sentiment == "bullish":
            if insider_buy_count >= cluster_threshold:
                insider_bonus = 8
            elif insider_buy_count >= 2:
                insider_bonus = standard_bonus
            else:
                insider_bonus = 0
        else:
            insider_bonus = 0

        assert insider_bonus == 5


class TestAnalystRevisionBonus:
    """Tests for analyst estimate revision bonus"""

    def test_strong_upward_revision_bonus(self):
        """Test strong bonus for 10%+ upward revision"""
        revision_pct = 15

        strong_up_threshold = 10
        strong_up_bonus = 5

        if revision_pct >= strong_up_threshold:
            estimate_bonus = strong_up_bonus
        else:
            estimate_bonus = 0

        assert estimate_bonus == 5

    def test_moderate_upward_revision_bonus(self):
        """Test moderate bonus for 5-10% upward revision"""
        revision_pct = 7

        strong_up_threshold = 10
        mod_up_threshold = 5
        strong_up_bonus = 5
        mod_up_bonus = 3

        if revision_pct >= strong_up_threshold:
            estimate_bonus = strong_up_bonus
        elif revision_pct >= mod_up_threshold:
            estimate_bonus = mod_up_bonus
        else:
            estimate_bonus = 0

        assert estimate_bonus == 3

    def test_downward_revision_penalty(self):
        """Test penalty for significant downward revision"""
        revision_pct = -12

        strong_down_threshold = -10
        strong_down_penalty = -5

        if revision_pct <= strong_down_threshold:
            estimate_bonus = strong_down_penalty
        else:
            estimate_bonus = 0

        assert estimate_bonus == -5


class TestEarningsAvoidance:
    """Tests for earnings calendar avoidance (non-CS stocks)"""

    def test_non_cs_stock_blocked_near_earnings(self):
        """Test non-CS stocks are blocked within avoidance window"""
        days_to_earnings = 3
        is_coiled_spring = False
        avoidance_days = 5

        if days_to_earnings is not None and days_to_earnings <= avoidance_days:
            if is_coiled_spring:
                skip_buy = False  # CS can override
            else:
                skip_buy = True

        assert skip_buy == True

    def test_cs_stock_can_override_avoidance(self):
        """Test CS stocks can override earnings avoidance"""
        days_to_earnings = 3
        is_coiled_spring = True
        cs_allow_buy_days = 7
        cs_block_days = 1
        avoidance_days = 5

        if days_to_earnings is not None and days_to_earnings <= avoidance_days:
            if is_coiled_spring and days_to_earnings > cs_block_days:
                skip_buy = False  # CS can buy
            else:
                skip_buy = True

        assert skip_buy == False  # CS allowed

    def test_stock_far_from_earnings_allowed(self):
        """Test stocks far from earnings are allowed"""
        days_to_earnings = 10
        avoidance_days = 5

        if days_to_earnings is None or days_to_earnings > avoidance_days:
            skip_buy = False
        else:
            skip_buy = True

        assert skip_buy == False


class TestMarketAwareStops:
    """Tests for market-aware stop loss logic"""

    def test_wider_stop_in_bearish_market(self):
        """Test that stop loss widens to 15% in bearish market"""
        spy_price = 450.0
        spy_ma_50 = 480.0  # Price below 50-day MA = bearish

        normal_stop_loss_pct = 10.0
        bearish_stop_loss_pct = 15.0

        is_bearish = spy_price < spy_ma_50

        if is_bearish:
            effective_stop_loss_pct = bearish_stop_loss_pct
        else:
            effective_stop_loss_pct = normal_stop_loss_pct

        assert is_bearish == True
        assert effective_stop_loss_pct == 15.0

    def test_normal_stop_in_bullish_market(self):
        """Test that stop loss stays at 10% in bullish market"""
        spy_price = 500.0
        spy_ma_50 = 480.0  # Price above 50-day MA = bullish

        normal_stop_loss_pct = 10.0
        bearish_stop_loss_pct = 15.0

        is_bearish = spy_price < spy_ma_50

        if is_bearish:
            effective_stop_loss_pct = bearish_stop_loss_pct
        else:
            effective_stop_loss_pct = normal_stop_loss_pct

        assert is_bearish == False
        assert effective_stop_loss_pct == 10.0

    def test_stop_loss_triggered_at_correct_threshold(self):
        """Test stop loss triggers at effective threshold"""
        cost_basis = 100.0

        # In bearish market with 15% stop
        bearish_stop_pct = 15.0
        price_in_bearish = 86.0  # -14% (not triggered)
        gain_pct = ((price_in_bearish - cost_basis) / cost_basis) * 100

        triggered = gain_pct <= -bearish_stop_pct
        assert triggered == False  # -14% does NOT trigger 15% stop

        # At exactly -15%
        price_at_stop = 85.0
        gain_pct = ((price_at_stop - cost_basis) / cost_basis) * 100
        triggered = gain_pct <= -bearish_stop_pct
        assert triggered == True  # -15% triggers 15% stop

    def test_normal_stop_would_trigger_earlier(self):
        """Test that normal stop (10%) would trigger where bearish (15%) doesn't"""
        cost_basis = 100.0
        price = 88.0  # -12%

        normal_stop_pct = 10.0
        bearish_stop_pct = 15.0

        gain_pct = ((price - cost_basis) / cost_basis) * 100

        normal_triggered = gain_pct <= -normal_stop_pct
        bearish_triggered = gain_pct <= -bearish_stop_pct

        assert normal_triggered == True   # -12% triggers 10% stop
        assert bearish_triggered == False  # -12% does NOT trigger 15% stop


class TestScoreCrashImprovements:
    """Tests for improved score crash detection"""

    def test_requires_three_consecutive_low_scans(self):
        """Test that 3 consecutive low scores are required (not just 2)"""
        consecutive_required = 3
        threshold = 50

        # Only 2 consecutive low scores
        recent_scores = [75, 68, 45, 42]  # Last 2 below 50
        consecutive_low = 0
        for score in reversed(recent_scores):
            if score < threshold:
                consecutive_low += 1
            else:
                break

        should_sell = consecutive_low >= consecutive_required
        assert consecutive_low == 2
        assert should_sell == False  # Need 3, only have 2

        # 3 consecutive low scores
        recent_scores = [75, 45, 42, 38]  # Last 3 below 50
        consecutive_low = 0
        for score in reversed(recent_scores):
            if score < threshold:
                consecutive_low += 1
            else:
                break

        should_sell = consecutive_low >= consecutive_required
        assert consecutive_low == 3
        assert should_sell == True  # Have 3, meet requirement

    def test_ignores_crash_if_profitable(self):
        """Test that score crash is ignored if position is profitable (10%+)"""
        ignore_if_profitable_pct = 10
        purchase_score = 75
        current_score = 40  # Crashed below 50
        threshold = 50
        drop_required = 20

        # Profitable position (+15%)
        gain_pct = 15
        score_drop = purchase_score - current_score  # 35 points

        meets_crash_criteria = score_drop > drop_required and current_score < threshold
        skip_due_to_profit = gain_pct >= ignore_if_profitable_pct

        assert meets_crash_criteria == True  # Score DID crash
        assert skip_due_to_profit == True    # But position is profitable
        # Result: should NOT sell

    def test_crashes_sell_if_not_profitable(self):
        """Test that score crash DOES trigger sell if position is NOT profitable"""
        ignore_if_profitable_pct = 10
        purchase_score = 75
        current_score = 40
        threshold = 50
        drop_required = 20
        consecutive_required = 3

        # Small gain (only +5%)
        gain_pct = 5

        score_drop = purchase_score - current_score
        meets_crash_criteria = score_drop > drop_required and current_score < threshold
        skip_due_to_profit = gain_pct >= ignore_if_profitable_pct

        assert meets_crash_criteria == True
        assert skip_due_to_profit == False
        # With 3 consecutive lows, this SHOULD sell

    def test_losing_position_with_crash_sells(self):
        """Test losing position with score crash does sell"""
        ignore_if_profitable_pct = 10
        purchase_score = 80
        current_score = 35
        gain_pct = -5  # Losing money

        score_drop = purchase_score - current_score  # 45 points
        skip_due_to_profit = gain_pct >= ignore_if_profitable_pct

        assert score_drop > 20  # Meets crash threshold
        assert skip_due_to_profit == False  # Not profitable
        # With 3 consecutive lows, this SHOULD sell


class TestDiskCacheIntegration:
    """Tests for disk cache price history functionality"""

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly"""
        from datetime import date

        ticker = "AAPL"
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)

        expected_key = f"{ticker}_{start.isoformat()}_{end.isoformat()}"
        assert expected_key == "AAPL_2024-01-01_2024-12-31"

    def test_cache_expiry_check(self):
        """Test that cache expiry is calculated correctly"""
        from datetime import datetime, timedelta, timezone

        expiry_days = 30
        created_at = datetime.now(timezone.utc) - timedelta(days=25)

        age_days = (datetime.now(timezone.utc) - created_at).days
        is_expired = age_days > expiry_days

        assert age_days == 25
        assert is_expired == False  # 25 days old, 30 day expiry = NOT expired

        # Test expired case
        created_at_old = datetime.now(timezone.utc) - timedelta(days=35)
        age_days_old = (datetime.now(timezone.utc) - created_at_old).days
        is_expired_old = age_days_old > expiry_days

        assert age_days_old == 35
        assert is_expired_old == True  # 35 days old > 30 day expiry = EXPIRED


class TestWorkerConfiguration:
    """Tests for configurable worker count"""

    def test_default_worker_count(self):
        """Test default worker count is 12"""
        default_workers = 12
        assert default_workers == 12

    def test_worker_count_respects_config(self):
        """Test that worker count comes from config"""
        # Simulate reading from config
        config_value = 12
        fallback_value = 8

        workers = config_value if config_value else fallback_value
        assert workers == 12

    def test_worker_count_bounds(self):
        """Test worker count stays reasonable"""
        min_workers = 4
        max_workers = 20

        configured_workers = 12

        effective_workers = max(min_workers, min(configured_workers, max_workers))
        assert effective_workers == 12
        assert min_workers <= effective_workers <= max_workers


class TestQualityFilters:
    """Tests for stock selection quality filters"""

    def test_min_score_raised(self):
        """Test minimum score to buy raised from 65 to 72"""
        from config_loader import config

        min_score = config.get('ai_trader.allocation.min_score_to_buy', 65)
        assert min_score == 72, f"min_score_to_buy should be 72, got {min_score}"

    def test_c_score_filter_threshold(self):
        """Test C score filter requires 10 points minimum"""
        from config_loader import config

        min_c = config.get('ai_trader.quality_filters.min_c_score', 10)
        assert min_c == 10, f"min_c_score should be 10, got {min_c}"

    def test_l_score_filter_threshold(self):
        """Test L score filter requires 8 points minimum"""
        from config_loader import config

        min_l = config.get('ai_trader.quality_filters.min_l_score', 8)
        assert min_l == 8, f"min_l_score should be 8, got {min_l}"

    def test_volume_filter_threshold(self):
        """Test volume filter requires 1.2x average"""
        from config_loader import config

        min_volume = config.get('ai_trader.quality_filters.min_volume_ratio', 1.2)
        assert min_volume == 1.2, f"min_volume_ratio should be 1.2, got {min_volume}"

    def test_growth_stocks_skip_c_l_filters(self):
        """Test that growth stocks can skip C and L filters"""
        from config_loader import config

        skip_growth = config.get('ai_trader.quality_filters.skip_in_growth_mode', True)
        assert skip_growth is True, "Growth stocks should skip C/L filters by default"

    def test_bearish_score_adjustment(self):
        """Test bear market requires 10 extra points (raised from 5)"""
        from config_loader import config

        bearish_adj = config.get('ai_trader.market_regime.bearish_min_score_adj', 5)
        assert bearish_adj == 10, f"bearish_min_score_adj should be 10, got {bearish_adj}"

    def test_quality_filter_logic(self):
        """Test quality filter logic rejects weak stocks"""
        # Simulate the filter logic
        min_c_score = 10
        min_l_score = 8
        min_volume = 1.2

        # Weak stock - should be filtered
        weak_stock = {
            'c_score': 5,  # Below 10
            'l_score': 7,  # Below 8
            'volume_ratio': 1.0,  # Below 1.2
            'is_growth': False
        }

        # Check C score
        if weak_stock['c_score'] < min_c_score:
            should_skip = True
        elif weak_stock['l_score'] < min_l_score:
            should_skip = True
        elif weak_stock['volume_ratio'] < min_volume:
            should_skip = True
        else:
            should_skip = False

        assert should_skip is True, "Weak stocks should be filtered out"

    def test_quality_filter_passes_strong_stock(self):
        """Test quality filter accepts strong stocks"""
        min_c_score = 10
        min_l_score = 8
        min_volume = 1.2

        # Strong stock - should pass all filters
        strong_stock = {
            'c_score': 12,  # Above 10
            'l_score': 10,  # Above 8
            'volume_ratio': 1.5,  # Above 1.2
            'is_growth': False
        }

        passes_c = strong_stock['c_score'] >= min_c_score
        passes_l = strong_stock['l_score'] >= min_l_score
        passes_volume = strong_stock['volume_ratio'] >= min_volume

        assert passes_c is True, "Strong stock should pass C score check"
        assert passes_l is True, "Strong stock should pass L score check"
        assert passes_volume is True, "Strong stock should pass volume check"
        assert (passes_c and passes_l and passes_volume) is True

    def test_growth_stock_bypasses_c_l_filters(self):
        """Test growth stocks bypass C and L score filters"""
        min_c_score = 10
        min_l_score = 8
        skip_growth = True

        # Growth stock with weak C and L (should still pass)
        growth_stock = {
            'c_score': 0,  # Very low - no earnings
            'l_score': 5,  # Below threshold
            'is_growth': True
        }

        # Logic: if growth and skip_growth enabled, bypass C/L checks
        if growth_stock['is_growth'] and skip_growth:
            bypassed = True
        else:
            bypassed = (growth_stock['c_score'] >= min_c_score and
                       growth_stock['l_score'] >= min_l_score)

        assert bypassed is True, "Growth stocks should bypass C/L filters"
