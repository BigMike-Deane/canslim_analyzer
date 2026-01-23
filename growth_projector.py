"""
Growth Projector Module
Projects 6-month growth potential based on multiple factors
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from data_fetcher import StockData, DataFetcher
from canslim_scorer import CANSLIMScore


@dataclass
class GrowthProjection:
    """Container for growth projection data"""
    ticker: str
    projected_growth_pct: float = 0.0
    confidence: str = "low"  # low, medium, high

    # Component projections
    momentum_projection: float = 0.0
    earnings_projection: float = 0.0
    analyst_projection: float = 0.0  # analyst price targets
    valuation_factor: float = 0.0    # NEW: P/E relative to growth
    canslim_factor: float = 0.0
    sector_bonus: float = 0.0

    # Additional info
    analyst_target: float = 0.0
    analyst_upside: float = 0.0
    num_analysts: int = 0


class GrowthProjector:
    """Projects 6-month stock growth based on multiple factors"""

    # REFINED weights - now includes valuation factor
    MOMENTUM_WEIGHT = 0.20      # Price momentum
    EARNINGS_WEIGHT = 0.15      # Earnings trajectory
    ANALYST_WEIGHT = 0.25       # Analyst price targets
    VALUATION_WEIGHT = 0.15     # NEW: P/E relative to growth (PEG-style)
    CANSLIM_WEIGHT = 0.15       # CANSLIM score factor
    SECTOR_WEIGHT = 0.10        # Sector momentum

    # Sector momentum data (would be fetched in production)
    SECTOR_PERFORMANCE: dict[str, float] = {}

    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self._calculate_sector_performance()

    def _calculate_sector_performance(self):
        """Calculate recent performance by sector using ETF proxies"""
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC',
        }

        for sector, etf in sector_etfs.items():
            try:
                # Use price-only fetch for ETFs (avoids yfinance fundamental errors)
                data = self.fetcher.get_price_data_only(etf)
                if data.is_valid and not data.price_history.empty:
                    prices = data.price_history['Close']
                    if len(prices) >= 126:  # ~6 months of trading days
                        perf = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100
                        self.SECTOR_PERFORMANCE[sector] = perf
            except Exception:
                pass

    def project_growth(self, stock_data: StockData, canslim_score: CANSLIMScore) -> GrowthProjection:
        """
        Project 6-month growth for a stock.
        REFINED: Includes analyst targets, valuation, and dynamic weighting.
        """
        projection = GrowthProjection(ticker=stock_data.ticker)

        if not stock_data.is_valid or stock_data.price_history.empty:
            return projection

        # 1. Momentum Projection (20% weight)
        projection.momentum_projection = self._calculate_momentum_projection(stock_data)

        # 2. Earnings Trajectory Projection (15% weight)
        projection.earnings_projection = self._calculate_earnings_projection(stock_data)

        # 3. Analyst Price Target Projection (25% weight)
        analyst_proj, target, upside, num = self._calculate_analyst_projection(stock_data)
        projection.analyst_projection = analyst_proj
        projection.analyst_target = target
        projection.analyst_upside = upside
        projection.num_analysts = num

        # 4. NEW: Valuation Factor (15% weight) - P/E relative to growth
        projection.valuation_factor = self._calculate_valuation_factor(stock_data)

        # 5. CANSLIM Score Factor (15% weight)
        projection.canslim_factor = self._calculate_canslim_factor(canslim_score)

        # 6. Sector Momentum Bonus (10% weight)
        projection.sector_bonus = self._calculate_sector_bonus(stock_data.sector)

        # Combined projection with dynamic weighting
        # If analyst data is weak, redistribute weight to other factors
        if projection.num_analysts >= 5:
            # Good analyst coverage - use standard weights
            projection.projected_growth_pct = (
                projection.momentum_projection * self.MOMENTUM_WEIGHT +
                projection.earnings_projection * self.EARNINGS_WEIGHT +
                projection.analyst_projection * self.ANALYST_WEIGHT +
                projection.valuation_factor * self.VALUATION_WEIGHT +
                projection.canslim_factor * self.CANSLIM_WEIGHT +
                projection.sector_bonus * self.SECTOR_WEIGHT
            )
        else:
            # Low analyst coverage - redistribute analyst weight
            adj_momentum = self.MOMENTUM_WEIGHT + (self.ANALYST_WEIGHT * 0.3)
            adj_earnings = self.EARNINGS_WEIGHT + (self.ANALYST_WEIGHT * 0.2)
            adj_valuation = self.VALUATION_WEIGHT + (self.ANALYST_WEIGHT * 0.25)
            adj_canslim = self.CANSLIM_WEIGHT + (self.ANALYST_WEIGHT * 0.25)

            projection.projected_growth_pct = (
                projection.momentum_projection * adj_momentum +
                projection.earnings_projection * adj_earnings +
                projection.analyst_projection * 0 +  # Ignore weak analyst data
                projection.valuation_factor * adj_valuation +
                projection.canslim_factor * adj_canslim +
                projection.sector_bonus * self.SECTOR_WEIGHT
            )

        # Determine confidence level
        projection.confidence = self._assess_confidence(stock_data, canslim_score)

        return projection

    def _calculate_momentum_projection(self, data: StockData) -> float:
        """
        Project growth based on price momentum using linear regression.
        Extrapolates 6-month trend forward.
        """
        prices = data.price_history['Close']

        if len(prices) < 126:  # Need ~6 months of data
            return 0.0

        # Use last 6 months for trend
        recent_prices = prices.iloc[-126:]

        # Linear regression
        x = np.arange(len(recent_prices))
        y = recent_prices.values

        # Calculate slope using least squares
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)

        # Project forward 126 trading days (~6 months)
        current_price = prices.iloc[-1]
        projected_price = current_price + (slope * 126)

        # Calculate projected growth percentage
        if current_price > 0:
            momentum_growth = ((projected_price - current_price) / current_price) * 100
        else:
            momentum_growth = 0

        # Cap extreme projections
        momentum_growth = max(-50, min(100, momentum_growth))

        return momentum_growth

    def _calculate_analyst_projection(self, data: StockData) -> tuple[float, float, float, int]:
        """
        NEW: Project growth based on analyst consensus price targets.
        Returns: (projection_pct, target_price, upside_pct, num_analysts)
        """
        target = data.analyst_target_price
        current = data.current_price
        num_analysts = data.num_analyst_opinions

        if target <= 0 or current <= 0:
            return 0.0, 0.0, 0.0, 0

        # Calculate upside to target
        upside_pct = ((target - current) / current) * 100

        # Analyst targets are typically 12-month targets
        # For 6-month projection, we take ~60% of the upside
        # (accounting for mean reversion and timing uncertainty)
        six_month_factor = 0.6

        # Apply confidence adjustments based on analyst coverage
        if num_analysts >= 20:
            confidence_mult = 1.0  # High confidence in consensus
        elif num_analysts >= 10:
            confidence_mult = 0.9
        elif num_analysts >= 5:
            confidence_mult = 0.8
        else:
            confidence_mult = 0.5  # Low confidence

        # Also adjust based on recommendation
        rec = data.analyst_recommendation.lower()
        if 'strong buy' in rec:
            rec_mult = 1.1
        elif 'buy' in rec:
            rec_mult = 1.0
        elif 'hold' in rec:
            rec_mult = 0.7  # Discount hold ratings
        elif 'sell' in rec:
            rec_mult = 0.3  # Heavy discount for sell ratings
        else:
            rec_mult = 0.8

        # Calculate projection
        analyst_projection = upside_pct * six_month_factor * confidence_mult * rec_mult

        # Cap extreme projections
        analyst_projection = max(-30, min(60, analyst_projection))

        return analyst_projection, target, upside_pct, num_analysts

    def _calculate_valuation_factor(self, data: StockData) -> float:
        """
        NEW: Calculate valuation-based growth factor.
        Uses P/E relative to earnings growth (PEG-style analysis).
        - Low P/E + high growth = undervalued = positive factor
        - High P/E + low growth = overvalued = negative factor
        Also considers earnings yield and FCF yield for quality.
        """
        pe = data.trailing_pe
        earnings_yield = getattr(data, 'earnings_yield', 0) or 0
        fcf_yield = getattr(data, 'fcf_yield', 0) or 0

        # Estimate growth rate from quarterly earnings if available
        growth_rate = 0
        if len(data.quarterly_earnings) >= 5:
            current_q = data.quarterly_earnings[0]
            year_ago_q = data.quarterly_earnings[4] if len(data.quarterly_earnings) > 4 else 0
            if year_ago_q > 0 and current_q > 0:
                growth_rate = ((current_q / year_ago_q) - 1) * 100

        # Use forward estimate if available and more reliable
        if data.earnings_growth_estimate > 0:
            growth_rate = max(growth_rate, data.earnings_growth_estimate * 100)

        valuation_score = 0

        # PEG-style analysis
        if pe > 0 and growth_rate > 0:
            peg = pe / growth_rate

            if peg < 0.5:
                # Deeply undervalued relative to growth
                valuation_score = 30
            elif peg < 1.0:
                # Undervalued (PEG < 1 is classic "buy" signal)
                valuation_score = 20
            elif peg < 1.5:
                # Fairly valued
                valuation_score = 5
            elif peg < 2.0:
                # Slightly overvalued
                valuation_score = -5
            elif peg < 3.0:
                # Overvalued
                valuation_score = -15
            else:
                # Extremely overvalued
                valuation_score = -25
        elif pe > 0 and pe < 15 and growth_rate <= 0:
            # Low P/E even with no growth - could be value play
            valuation_score = 5
        elif pe > 50 and growth_rate <= 20:
            # High P/E without strong growth justification
            valuation_score = -20

        # Earnings yield bonus (inverse of P/E, higher is better)
        # Earnings yield > 5% is generally attractive
        if earnings_yield > 0.08:  # > 8%
            valuation_score += 10
        elif earnings_yield > 0.05:  # > 5%
            valuation_score += 5
        elif earnings_yield < 0.02 and earnings_yield > 0:  # < 2%
            valuation_score -= 5

        # FCF yield bonus (free cash flow relative to price)
        # FCF yield > 5% indicates strong cash generation
        if fcf_yield > 0.08:
            valuation_score += 10
        elif fcf_yield > 0.05:
            valuation_score += 5
        elif fcf_yield < 0 and fcf_yield != 0:
            # Negative FCF is a red flag
            valuation_score -= 10

        # Cap the valuation factor
        return max(-30, min(40, valuation_score))

    def _calculate_earnings_projection(self, data: StockData) -> float:
        """
        REFINED: Project growth based on earnings trajectory with anomaly filtering.
        Uses forward estimates when available, falls back to historical with filtering.
        """
        # First, try to use forward earnings growth estimate if available
        if data.earnings_growth_estimate > 0:
            # Forward estimate is usually annual, scale for 6 months
            forward_projection = data.earnings_growth_estimate * 100 * 0.5

            # Apply P/E consideration via PEG ratio
            if data.peg_ratio > 0:
                if data.peg_ratio < 1:
                    forward_projection *= 1.2  # Undervalued growth
                elif data.peg_ratio > 2:
                    forward_projection *= 0.7  # Expensive growth

            return max(-30, min(60, forward_projection))

        # Fallback: calculate from historical earnings with anomaly filtering
        if len(data.quarterly_earnings) < 4:
            return 0.0

        earnings = data.quarterly_earnings[:4]

        # Filter out zeros and negatives for growth calculation
        valid_earnings = [e for e in earnings if e and e > 0]

        if len(valid_earnings) < 2:
            return 0.0

        # Calculate growth rates with anomaly filtering
        growths = []
        for i in range(len(valid_earnings) - 1):
            if valid_earnings[i + 1] > 0:
                growth = (valid_earnings[i] - valid_earnings[i + 1]) / valid_earnings[i + 1]
                # ANOMALY FILTER: ignore extreme swings (>100% change)
                if abs(growth) <= 1.0:
                    growths.append(growth)

        if not growths:
            return 0.0

        # Use median instead of mean to be robust to outliers
        avg_quarterly_growth = np.median(growths)

        # Project 2 quarters forward (6 months)
        earnings_projection = avg_quarterly_growth * 2 * 100

        # More conservative P/E expansion factor
        if earnings_projection > 15:
            earnings_projection *= 1.2  # Reduced from 1.5

        # Cap extreme projections
        earnings_projection = max(-30, min(50, earnings_projection))

        return earnings_projection

    def _calculate_canslim_factor(self, score: CANSLIMScore) -> float:
        """
        Convert CANSLIM score to a growth factor.
        Higher scores correlate with better performance.
        """
        # Score is out of 100
        # Top scores (80+) get significant boost
        # Average scores (50-80) get moderate boost
        # Low scores (<50) get penalty

        if score.total_score >= 80:
            return 30 + (score.total_score - 80) * 1.5  # 30-60%
        elif score.total_score >= 60:
            return 10 + (score.total_score - 60) * 1.0  # 10-30%
        elif score.total_score >= 40:
            return (score.total_score - 40) * 0.5  # 0-10%
        else:
            return -10  # Penalty for low scores

    def _calculate_sector_bonus(self, sector: str) -> float:
        """
        Apply sector momentum bonus.
        Stocks in leading sectors get a boost.
        """
        sector_perf = self.SECTOR_PERFORMANCE.get(sector, 0)

        # Scale sector performance as a bonus factor
        # Strong sector (>10% 6mo return) -> up to 20% bonus
        # Weak sector (<-10%) -> penalty

        if sector_perf > 15:
            return 20
        elif sector_perf > 10:
            return 15
        elif sector_perf > 5:
            return 10
        elif sector_perf > 0:
            return 5
        elif sector_perf > -5:
            return 0
        elif sector_perf > -10:
            return -5
        else:
            return -10

    def _assess_confidence(self, data: StockData, score: CANSLIMScore) -> str:
        """
        REFINED: Assess confidence level of the projection.
        Now includes analyst coverage as a major confidence factor.
        """
        confidence_score = 0

        # Data quality checks
        if len(data.price_history) >= 250:
            confidence_score += 2
        elif len(data.price_history) >= 126:
            confidence_score += 1

        if len(data.quarterly_earnings) >= 4:
            confidence_score += 1
        if len(data.quarterly_earnings) >= 8:
            confidence_score += 1  # Bonus for TTM data

        if data.institutional_holders_pct > 0:
            confidence_score += 1

        # NEW: Analyst coverage is a major confidence factor
        if data.num_analyst_opinions >= 20:
            confidence_score += 3
        elif data.num_analyst_opinions >= 10:
            confidence_score += 2
        elif data.num_analyst_opinions >= 5:
            confidence_score += 1

        # NEW: Forward estimates available
        if data.earnings_growth_estimate > 0:
            confidence_score += 1

        # Signal consistency
        if score.total_score >= 70:
            confidence_score += 2
        elif score.total_score >= 50:
            confidence_score += 1

        # Check if multiple CANSLIM factors are strong
        strong_factors = sum([
            score.c_score >= 10,
            score.a_score >= 10,
            score.l_score >= 10,
        ])
        confidence_score += strong_factors

        if confidence_score >= 10:
            return "high"
        elif confidence_score >= 6:
            return "medium"
        else:
            return "low"


if __name__ == "__main__":
    from canslim_scorer import CANSLIMScorer

    fetcher = DataFetcher()
    scorer = CANSLIMScorer(fetcher)
    projector = GrowthProjector(fetcher)

    test_tickers = ["NVDA", "AAPL", "MSFT"]

    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Growth Projection for {ticker}")
        print('='*50)

        data = fetcher.get_stock_data(ticker)
        if not data.is_valid:
            print(f"Could not fetch data")
            continue

        canslim = scorer.score_stock(data)
        projection = projector.project_growth(data, canslim)

        print(f"Projected 6-Month Growth: {projection.projected_growth_pct:+.1f}%")
        print(f"Confidence: {projection.confidence}")
        print(f"\nComponents:")
        print(f"  Momentum:  {projection.momentum_projection:+.1f}% (weight: 40%)")
        print(f"  Earnings:  {projection.earnings_projection:+.1f}% (weight: 30%)")
        print(f"  CANSLIM:   {projection.canslim_factor:+.1f}% (weight: 20%)")
        print(f"  Sector:    {projection.sector_bonus:+.1f}% (weight: 10%)")
