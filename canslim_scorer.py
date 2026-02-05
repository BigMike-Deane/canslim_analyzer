"""
CANSLIM Scorer Module
Implements scoring logic for all 7 CANSLIM criteria
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from data_fetcher import StockData, DataFetcher, get_cached_market_direction


@dataclass
class CANSLIMScore:
    """Container for CANSLIM scores and details"""
    ticker: str
    total_score: float = 0.0

    # Individual scores (max points in parentheses)
    c_score: float = 0.0  # Current Quarterly Earnings (15)
    a_score: float = 0.0  # Annual Earnings Growth (15)
    n_score: float = 0.0  # New Highs (15)
    s_score: float = 0.0  # Supply & Demand (15)
    l_score: float = 0.0  # Leader vs Laggard (15)
    i_score: float = 0.0  # Institutional Ownership (10)
    m_score: float = 0.0  # Market Direction (15)

    # Details for display
    c_detail: str = ""
    a_detail: str = ""
    n_detail: str = ""
    s_detail: str = ""
    l_detail: str = ""
    i_detail: str = ""
    m_detail: str = ""


class CANSLIMScorer:
    """Calculates CANSLIM scores for stocks"""

    MAX_SCORES = {
        'C': 15,
        'A': 15,
        'N': 15,
        'S': 15,
        'L': 15,
        'I': 10,
        'M': 15,
    }

    # Sector-adjusted growth thresholds
    # A 20% growth for Industrial is excellent; for Tech it's mediocre
    SECTOR_GROWTH_THRESHOLDS = {
        'Technology': {'excellent': 30, 'good': 20},
        'Communication Services': {'excellent': 25, 'good': 18},
        'Consumer Discretionary': {'excellent': 25, 'good': 18},
        'Healthcare': {'excellent': 25, 'good': 15},
        'Financials': {'excellent': 20, 'good': 12},
        'Industrials': {'excellent': 20, 'good': 12},
        'Consumer Staples': {'excellent': 15, 'good': 10},
        'Materials': {'excellent': 18, 'good': 12},
        'Energy': {'excellent': 18, 'good': 10},
        'Real Estate': {'excellent': 15, 'good': 10},
        'Utilities': {'excellent': 12, 'good': 8},
        # Default for unknown sectors
        'default': {'excellent': 25, 'good': 15},
    }

    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self._market_score: float | None = None
        self._market_detail: str = ""

    def _get_sector_thresholds(self, sector: str) -> dict:
        """Get growth thresholds for a given sector"""
        return self.SECTOR_GROWTH_THRESHOLDS.get(
            sector,
            self.SECTOR_GROWTH_THRESHOLDS['default']
        )

    def score_stock(self, stock_data: StockData) -> CANSLIMScore:
        """Calculate complete CANSLIM score for a stock"""
        score = CANSLIMScore(ticker=stock_data.ticker)

        if not stock_data.is_valid:
            return score

        # Calculate each criterion
        score.c_score, score.c_detail = self._score_current_earnings(stock_data)
        score.a_score, score.a_detail = self._score_annual_earnings(stock_data)
        score.n_score, score.n_detail = self._score_new_highs(stock_data)
        score.s_score, score.s_detail = self._score_supply_demand(stock_data)
        score.l_score, score.l_detail = self._score_leader(stock_data)
        score.i_score, score.i_detail = self._score_institutional(stock_data)
        score.m_score, score.m_detail = self._score_market()

        score.total_score = (
            score.c_score + score.a_score + score.n_score +
            score.s_score + score.l_score + score.i_score + score.m_score
        )

        return score

    def _score_current_earnings(self, data: StockData) -> tuple[float, str]:
        """
        C - Current Quarterly Earnings (15 pts max)
        REFINED: Uses TTM comparison + bonuses for acceleration and earnings surprise.
        - Base: TTM EPS growth vs prior year (up to 10 pts)
        - Bonus: EPS acceleration (current Q growth > prior Q growth) (up to 3 pts)
        - Bonus: Earnings surprise (beat estimates by 5%+) (up to 2 pts)
        """
        max_score = self.MAX_SCORES['C']
        base_max = 10  # Base score for TTM growth
        accel_max = 5  # Bonus for acceleration (increased from 3 - strong predictor)
        surprise_max = 2  # Bonus for earnings surprise

        # Filter out None values from earnings
        earnings = [e for e in data.quarterly_earnings if e is not None]

        # Need at least 8 quarters for TTM vs prior TTM comparison
        if len(earnings) < 8:
            # Fallback to simpler comparison with anomaly filtering
            if len(earnings) >= 4:
                return self._score_earnings_with_anomaly_filter(data, max_score)
            return 0, "Insufficient data"

        # Calculate TTM (sum of last 4 quarters)
        current_ttm = sum(earnings[0:4])
        prior_ttm = sum(earnings[4:8])

        # Companies with negative TTM earnings get reduced scores
        # Give partial credit for improving losses (turnaround potential)
        if current_ttm < 0:
            if prior_ttm < 0 and current_ttm > prior_ttm:
                # Losses shrinking - give partial credit (up to 40% of max)
                # Guard against division by near-zero
                if abs(prior_ttm) < 0.01:
                    return round(max_score * 0.3, 1), "Losses improving"
                # Formula: how much did losses shrink as % of prior loss
                # e.g., prior=-10, current=-5: improvement = 50% (losses cut in half)
                improvement_pct = ((current_ttm - prior_ttm) / abs(prior_ttm)) * 100
                partial_score = max(0, min(max_score * 0.4, (improvement_pct / 50) * max_score * 0.4))
                return round(partial_score, 1), f"Losses improving ({improvement_pct:+.0f}%)"
            else:
                # Losses worsening or stable negative
                return 0, f"TTM loss: ${current_ttm:.2f}"

        # Handle zero or near-zero prior TTM (avoid division issues)
        if abs(prior_ttm) < 0.01:
            if current_ttm > 0:
                return max_score * 0.8, "Turnaround (TTM)"
            return 0, "No prior TTM earnings"

        ttm_growth = ((current_ttm - prior_ttm) / abs(prior_ttm)) * 100

        # Anomaly filter: cap extreme values that are likely data errors
        if abs(ttm_growth) > 500:
            if data.earnings_growth_estimate > 0:
                ttm_growth = min(ttm_growth, data.earnings_growth_estimate * 100 * 2)
            else:
                ttm_growth = max(-100, min(200, ttm_growth))

        # Get sector-adjusted thresholds
        sector = getattr(data, 'sector', None) or 'default'
        thresholds = self._get_sector_thresholds(sector)
        excellent_threshold = thresholds['excellent']
        good_threshold = thresholds['good']

        # Base score from TTM growth (up to 10 pts) - sector adjusted
        if ttm_growth >= excellent_threshold:
            base_score = base_max
        elif ttm_growth >= good_threshold:
            # Scale between good and excellent
            range_pct = (ttm_growth - good_threshold) / (excellent_threshold - good_threshold)
            base_score = base_max * (0.6 + 0.4 * range_pct)
        elif ttm_growth >= 0:
            base_score = (ttm_growth / good_threshold) * base_max * 0.6
        else:
            base_score = max(0, (1 + ttm_growth / 50) * base_max * 0.3)

        # EPS Acceleration bonus (up to 3 pts)
        # Check if most recent quarter's YoY growth > prior quarter's YoY growth
        accel_score = 0
        accel_detail = ""
        if len(data.quarterly_earnings) >= 5:
            # Current quarter vs same quarter last year
            current_q = data.quarterly_earnings[0]
            prior_year_q = data.quarterly_earnings[4] if len(data.quarterly_earnings) > 4 else 0

            # Previous quarter vs same quarter last year
            prev_q = data.quarterly_earnings[1]
            prev_prior_year_q = data.quarterly_earnings[5] if len(data.quarterly_earnings) > 5 else 0

            if prior_year_q > 0 and prev_prior_year_q > 0:
                current_q_growth = ((current_q - prior_year_q) / abs(prior_year_q)) * 100
                prev_q_growth = ((prev_q - prev_prior_year_q) / abs(prev_prior_year_q)) * 100

                if current_q_growth > prev_q_growth and current_q_growth > 0:
                    # Accelerating earnings growth
                    accel_score = accel_max
                    accel_detail = " +accel"
                elif current_q_growth > 0 and current_q_growth >= prev_q_growth * 0.9:
                    # Maintaining strong growth
                    accel_score = accel_max * 0.5
                    accel_detail = " steady"

        # Earnings surprise bonus (up to 2 pts)
        # Reward companies that beat analyst estimates
        surprise_score = 0
        surprise_detail = ""
        earnings_surprise = getattr(data, 'earnings_surprise_pct', 0) or 0
        eps_beat_streak = getattr(data, 'eps_beat_streak', 0) or 0

        if earnings_surprise >= 10:
            surprise_score = surprise_max
            surprise_detail = f" +beat {earnings_surprise:.0f}%"
        elif earnings_surprise >= 5:
            surprise_score = surprise_max * 0.75
            surprise_detail = f" +beat"
        elif earnings_surprise > 0:
            surprise_score = surprise_max * 0.5

        # Extra bonus for consistent beats (4+ quarters)
        beat_streak_bonus = 0
        beat_streak_detail = ""
        if eps_beat_streak >= 4:
            # +1 for 4 beats, +2 for 5+ beats (capped at 2)
            beat_streak_bonus = min(2, eps_beat_streak - 3)
            beat_streak_detail = f" +{eps_beat_streak}beats"
            surprise_score = min(surprise_score + beat_streak_bonus, surprise_max)

        # Analyst estimate revision bonus/penalty (P1 feature Feb 2026)
        # Reward stocks where analysts are raising estimates - scaled by magnitude
        revision_bonus = 0
        revision_detail = ""
        estimate_revision_pct = getattr(data, 'eps_estimate_revision_pct', None)
        if estimate_revision_pct is not None:
            if estimate_revision_pct >= 20:
                revision_bonus = 8  # Strong upward revision
                revision_detail = f" +est↑↑{estimate_revision_pct:.0f}%"
            elif estimate_revision_pct >= 15:
                revision_bonus = 6
                revision_detail = f" +est↑{estimate_revision_pct:.0f}%"
            elif estimate_revision_pct >= 10:
                revision_bonus = 4
                revision_detail = f" +est↑{estimate_revision_pct:.0f}%"
            elif estimate_revision_pct >= 5:
                revision_bonus = 2
                revision_detail = f" +est↑"
            elif estimate_revision_pct <= -10:
                revision_bonus = -4  # Strong downward revision
                revision_detail = f" est↓↓"
            elif estimate_revision_pct <= -5:
                revision_bonus = -2
                revision_detail = f" est↓"

        total_score = min(base_score + accel_score + surprise_score + revision_bonus, max_score)
        total_score = max(total_score, 0)  # Don't go below 0
        return round(total_score, 1), f"TTM: {ttm_growth:+.0f}%{accel_detail}{surprise_detail}{beat_streak_detail}{revision_detail}"

    def _score_earnings_with_anomaly_filter(self, data: StockData, max_score: float) -> tuple[float, str]:
        """
        Fallback scoring with anomaly filtering for stocks with limited data.
        Filters out extreme QoQ swings (>50%) that are likely one-time items.
        IMPORTANT: Companies with negative earnings get penalized regardless of "growth" trend.
        """
        # Filter out None values first
        earnings = [e for e in data.quarterly_earnings[:4] if e is not None]

        if len(earnings) < 2:
            return 0, "Insufficient data"

        # Check if earnings are negative (company is losing money)
        # Give partial credit for improving losses, but cap below profitable companies
        recent_earnings = earnings[:2]
        if all(e < 0 for e in recent_earnings):
            # Company is losing money in recent quarters
            if earnings[0] < earnings[1]:
                # Losses are getting WORSE (more negative)
                return 0, "Losses worsening"
            else:
                # Losses are shrinking - give partial credit (up to 35% of max)
                # Q0 > Q1 (less negative), so improvement = (Q0 - Q1) / |Q1|
                # Guard against near-zero division
                if abs(earnings[1]) < 0.001:
                    return round(max_score * 0.2, 1), "Losses near zero"
                improvement = ((earnings[0] - earnings[1]) / abs(earnings[1])) * 100
                partial_score = max(0, min(max_score * 0.35, (improvement / 50) * max_score * 0.35))
                return round(max(partial_score, max_score * 0.1), 1), f"Losses shrinking ({improvement:+.0f}%)"

        # Calculate growth rates between consecutive quarters
        growth_rates = []
        for i in range(len(earnings) - 1):
            if earnings[i + 1] != 0 and abs(earnings[i + 1]) >= 0.001:
                rate = ((earnings[i] - earnings[i + 1]) / abs(earnings[i + 1])) * 100
                # Filter anomalies: ignore swings > 100%
                if abs(rate) <= 100:
                    growth_rates.append(rate)

        if not growth_rates:
            # All data was anomalous, use forward estimate if available
            if data.earnings_growth_estimate > 0:
                est_growth = data.earnings_growth_estimate * 100
                score = min(max_score, (est_growth / 25) * max_score)
                return round(score, 1), f"Est: {est_growth:+.0f}%"
            return 0, "Data anomaly"

        # Use median to be robust to outliers
        avg_growth = np.median(growth_rates)

        if avg_growth >= 25:
            score = max_score
        elif avg_growth >= 0:
            score = (avg_growth / 25) * max_score
        else:
            score = max(0, (1 + avg_growth / 50) * max_score * 0.3)

        return round(score, 1), f"Avg QoQ: {avg_growth:+.0f}%"

    def _score_annual_earnings(self, data: StockData) -> tuple[float, str]:
        """
        A - Annual Earnings Growth (15 pts max)
        REFINED: 3-year CAGR (up to 12 pts) + ROE quality check (up to 3 pts)
        - O'Neil recommends 25%+ CAGR and 17%+ ROE
        """
        max_score = self.MAX_SCORES['A']
        cagr_max = 12  # Base score for CAGR
        roe_max = 3    # Bonus for strong ROE

        if len(data.annual_earnings) < 3:
            return 0, "Insufficient data"

        # Calculate 3-year CAGR
        recent = data.annual_earnings[0]
        older = data.annual_earnings[2]  # 3 years ago

        if older <= 0 or recent <= 0:
            if recent > 0 and older <= 0:
                return max_score * 0.7, "Turnaround"
            return 0, "Negative earnings"

        cagr = ((recent / older) ** (1 / 3) - 1) * 100

        # Get sector-adjusted thresholds
        sector = getattr(data, 'sector', None) or 'default'
        thresholds = self._get_sector_thresholds(sector)
        excellent_threshold = thresholds['excellent']
        good_threshold = thresholds['good']

        # Base score from CAGR (up to 12 pts) - sector adjusted
        if cagr >= excellent_threshold:
            cagr_score = cagr_max
        elif cagr >= good_threshold:
            # Scale between good and excellent
            range_pct = (cagr - good_threshold) / (excellent_threshold - good_threshold)
            cagr_score = cagr_max * (0.6 + 0.4 * range_pct)
        elif cagr >= 0:
            cagr_score = (cagr / good_threshold) * cagr_max * 0.6
        else:
            cagr_score = 0

        # ROE quality bonus (up to 3 pts)
        # O'Neil recommends ROE of 17% or higher
        roe_score = 0
        roe_detail = ""
        roe = getattr(data, 'roe', 0) or 0

        # ROE from FMP is already a decimal (e.g., 0.25 = 25%)
        roe_pct = roe * 100 if roe < 1 else roe  # Handle both formats

        if roe_pct >= 25:
            roe_score = roe_max
            roe_detail = f", ROE {roe_pct:.0f}%"
        elif roe_pct >= 17:
            roe_score = roe_max * 0.7
            roe_detail = f", ROE {roe_pct:.0f}%"
        elif roe_pct >= 10:
            roe_score = roe_max * 0.3
            roe_detail = f", ROE {roe_pct:.0f}%"
        elif roe_pct > 0:
            roe_detail = f", low ROE"

        total_score = min(cagr_score + roe_score, max_score)
        return round(total_score, 1), f"3yr CAGR: {cagr:+.0f}%{roe_detail}"

    def _score_new_highs(self, data: StockData) -> tuple[float, str]:
        """
        N - New Highs / Near 52-week High (15 pts max)
        REFINED: Price proximity to high (up to 12 pts) + Volume confirmation (up to 3 pts)
        - Breakouts on high volume are more significant
        """
        max_score = self.MAX_SCORES['N']
        price_max = 12  # Base score for price proximity
        vol_max = 3     # Bonus for volume confirmation

        if data.high_52w <= 0 or data.current_price <= 0:
            return 0, "No price data"

        pct_from_high = ((data.high_52w - data.current_price) / data.high_52w) * 100

        # Base score from price proximity to 52-week high (up to 12 pts)
        if pct_from_high <= 5:
            price_score = price_max
            price_detail = f"{pct_from_high:.0f}% from high"
        elif pct_from_high <= 10:
            price_score = price_max * 0.75
            price_detail = f"{pct_from_high:.0f}% from high"
        elif pct_from_high <= 15:
            price_score = price_max * 0.5
            price_detail = f"{pct_from_high:.0f}% from high"
        elif pct_from_high <= 25:
            price_score = price_max * 0.25
            price_detail = f"{pct_from_high:.0f}% below high"
        else:
            price_score = 0
            price_detail = f"{pct_from_high:.0f}% below high"

        # Volume confirmation bonus (up to 3 pts)
        # If near highs AND volume is elevated, it's a stronger signal
        vol_score = 0
        vol_detail = ""

        if data.avg_volume_50d > 0 and data.current_volume > 0:
            vol_ratio = data.current_volume / data.avg_volume_50d

            # Only give volume bonus if price is within 15% of high
            if pct_from_high <= 15:
                if vol_ratio >= 1.5:
                    vol_score = vol_max
                    vol_detail = f", vol {vol_ratio:.1f}x"
                elif vol_ratio >= 1.2:
                    vol_score = vol_max * 0.6
                    vol_detail = f", vol {vol_ratio:.1f}x"
                elif vol_ratio >= 1.0:
                    vol_score = vol_max * 0.3

        total_score = min(price_score + vol_score, max_score)
        return round(total_score, 1), f"{price_detail}{vol_detail}"

    def _score_supply_demand(self, data: StockData) -> tuple[float, str]:
        """
        S - Supply & Demand (15 pts max)
        Based on volume trends and price action
        """
        max_score = self.MAX_SCORES['S']

        if data.price_history.empty or data.avg_volume_50d <= 0:
            return 0, "No volume data"

        # Calculate volume ratio with NaN handling
        try:
            if data.current_volume > 0:
                recent_vol = data.current_volume
            else:
                vol_series = data.price_history['Volume'].dropna().iloc[-5:]
                recent_vol = float(vol_series.mean()) if len(vol_series) > 0 else 0

            if recent_vol <= 0 or data.avg_volume_50d <= 0:
                return 0, "No volume data"

            vol_ratio = recent_vol / data.avg_volume_50d
        except Exception:
            return 0, "Volume calc error"

        # Check price trend (last 20 days) with NaN handling
        price_change = 0
        try:
            if len(data.price_history) >= 20:
                recent_prices = data.price_history['Close'].dropna().iloc[-20:]
                if len(recent_prices) >= 2 and recent_prices.iloc[0] > 0:
                    price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        except (KeyError, IndexError, TypeError, ValueError):
            pass  # Expected when price_history is incomplete or malformed

        # Score based on volume surge with positive price action
        score = 0
        details = []

        # Volume component (up to 8 points)
        if vol_ratio >= 2.0:
            score += 8
            details.append(f"vol {vol_ratio:.1f}x avg")
        elif vol_ratio >= 1.5:
            score += 6
            details.append(f"vol {vol_ratio:.1f}x avg")
        elif vol_ratio >= 1.2:
            score += 4
            details.append(f"vol {vol_ratio:.1f}x avg")
        else:
            details.append(f"vol {vol_ratio:.1f}x avg")

        # Price trend component (up to 7 points)
        if price_change > 0.05 and vol_ratio > 1.2:  # Rising price with high volume
            score += 7
            details.append("rising")
        elif price_change > 0:
            score += 4
            details.append("rising")
        elif price_change < -0.05:
            details.append("falling")

        return min(round(score, 1), max_score), ", ".join(details)

    def _score_leader(self, data: StockData) -> tuple[float, str]:
        """
        L - Leader vs Laggard (15 pts max)
        REFINED: Multi-timeframe relative strength with momentum weighting
        - 12-month RS (60% weight) + 3-month RS (40% weight)
        - Approximates IBD RS Rating by rewarding consistent outperformers
        """
        max_score = self.MAX_SCORES['L']

        if data.price_history.empty or len(data.price_history) < 63:
            return 0, "Insufficient price data"

        sp500_history = self.fetcher.get_sp500_history()
        if sp500_history.empty:
            return max_score * 0.5, "No benchmark data"

        # Drop NaN values from price history to avoid calculation errors
        prices = data.price_history['Close'].dropna()
        sp500_prices = sp500_history['Close'].dropna()

        if len(prices) < 63 or len(sp500_prices) < 63:
            return 0, "Insufficient valid price data"

        # Calculate 12-month RS (if enough data)
        if len(prices) >= 252 and len(sp500_prices) >= 252:
            stock_return_12m = (prices.iloc[-1] / prices.iloc[-252]) - 1
            sp500_return_12m = (sp500_prices.iloc[-1] / sp500_prices.iloc[-252]) - 1
        else:
            # Use available data
            stock_return_12m = (prices.iloc[-1] / prices.iloc[0]) - 1
            sp500_return_12m = (sp500_prices.iloc[-1] / sp500_prices.iloc[0]) - 1

        # Calculate 3-month RS (more recent momentum)
        lookback_3m = min(63, len(prices) - 1, len(sp500_prices) - 1)
        stock_return_3m = (prices.iloc[-1] / prices.iloc[-lookback_3m]) - 1
        sp500_return_3m = (sp500_prices.iloc[-1] / sp500_prices.iloc[-lookback_3m]) - 1

        # Calculate RS ratios with protection against extreme market crashes
        # Use minimum denominator of 0.1 (90% market crash) to avoid inflated RS values
        sp500_denom_12m = max(1 + sp500_return_12m, 0.1)
        sp500_denom_3m = max(1 + sp500_return_3m, 0.1)
        rs_12m = (1 + stock_return_12m) / sp500_denom_12m
        rs_3m = (1 + stock_return_3m) / sp500_denom_3m

        # Weighted RS: 60% 12-month + 40% 3-month (emphasizes recent momentum)
        weighted_rs = rs_12m * 0.6 + rs_3m * 0.4

        # RS trend bonus: is 3-month RS stronger than 12-month? (improving)
        rs_improving = rs_3m > rs_12m

        # Score based on weighted RS (approximates percentile ranking)
        # RS of 1.5+ typically represents top 10-20% of stocks
        if weighted_rs >= 1.5:
            score = max_score
        elif weighted_rs >= 1.3:
            score = max_score * 0.9
        elif weighted_rs >= 1.15:
            score = max_score * 0.75
        elif weighted_rs >= 1.0:
            score = max_score * 0.55
        elif weighted_rs >= 0.85:
            score = max_score * 0.35
        elif weighted_rs >= 0.7:
            score = max_score * 0.15
        else:
            score = 0

        # Small bonus for improving RS trend
        if rs_improving and score > 0:
            score = min(score + 1, max_score)

        trend_indicator = "↑" if rs_improving else "↓" if rs_3m < rs_12m * 0.95 else "→"
        return round(score, 1), f"RS: {weighted_rs:.2f} {trend_indicator}"

    def extract_rs_values(self, data: StockData) -> dict:
        """
        Extract RS (relative strength) values for persistence.
        Returns dict with rs_12m, rs_3m for saving to database.
        """
        if data.price_history.empty or len(data.price_history) < 63:
            return {"rs_12m": None, "rs_3m": None}

        sp500_history = self.fetcher.get_sp500_history()
        if sp500_history.empty:
            return {"rs_12m": None, "rs_3m": None}

        prices = data.price_history['Close'].dropna()
        sp500_prices = sp500_history['Close'].dropna()

        if len(prices) < 63 or len(sp500_prices) < 63:
            return {"rs_12m": None, "rs_3m": None}

        # Calculate 12-month RS
        if len(prices) >= 252 and len(sp500_prices) >= 252:
            stock_return_12m = (prices.iloc[-1] / prices.iloc[-252]) - 1
            sp500_return_12m = (sp500_prices.iloc[-1] / sp500_prices.iloc[-252]) - 1
        else:
            stock_return_12m = (prices.iloc[-1] / prices.iloc[0]) - 1
            sp500_return_12m = (sp500_prices.iloc[-1] / sp500_prices.iloc[0]) - 1

        # Calculate 3-month RS
        lookback_3m = min(63, len(prices) - 1, len(sp500_prices) - 1)
        stock_return_3m = (prices.iloc[-1] / prices.iloc[-lookback_3m]) - 1
        sp500_return_3m = (sp500_prices.iloc[-1] / sp500_prices.iloc[-lookback_3m]) - 1

        sp500_denom_12m = max(1 + sp500_return_12m, 0.1)
        sp500_denom_3m = max(1 + sp500_return_3m, 0.1)
        rs_12m = (1 + stock_return_12m) / sp500_denom_12m
        rs_3m = (1 + stock_return_3m) / sp500_denom_3m

        return {
            "rs_12m": round(rs_12m, 4),
            "rs_3m": round(rs_3m, 4)
        }

    def _score_institutional(self, data: StockData) -> tuple[float, str]:
        """
        I - Institutional Ownership (10 pts max)
        Sweet spot: 25-75% institutional ownership (expanded from 20-60%)
        Many quality growth stocks have 60-75% institutional ownership.
        """
        max_score = self.MAX_SCORES['I']
        inst_pct = data.institutional_holders_pct

        if inst_pct <= 0:
            # Default to neutral score when data unavailable (e.g., rate limited)
            # Most large-cap stocks have 40-80% institutional ownership
            return max_score * 0.5, "No data (neutral)"

        # Ideal range is 25-75% (expanded to include more quality growth stocks)
        if 25 <= inst_pct <= 75:
            score = max_score
        elif 15 <= inst_pct < 25:
            score = max_score * 0.7
        elif 75 < inst_pct <= 85:
            score = max_score * 0.7
        elif inst_pct < 15:
            score = max_score * 0.3  # Too little institutional interest
        else:
            score = max_score * 0.4  # Too crowded (>85%)

        return round(score, 1), f"{inst_pct:.0f}% inst."

    def _score_market(self) -> tuple[float, str]:
        """
        M - Market Direction (15 pts max)
        Based on weighted multi-index analysis: SPY (50%), QQQ (30%), DIA (20%)
        Uses cached market direction to avoid rate limiting during scans.
        """
        if self._market_score is not None:
            return self._market_score, self._market_detail

        max_score = self.MAX_SCORES['M']

        # Get cached multi-index market direction
        market_data = get_cached_market_direction()

        if market_data.get("success"):
            # Use pre-calculated score from multi-index analysis
            score = market_data.get("market_score", max_score * 0.5)
            trend = market_data.get("market_trend", "neutral")
            weighted_signal = market_data.get("weighted_signal", 0)

            # Build detail string showing index breakdown
            indexes = market_data.get("indexes", {})
            index_details = []
            for ticker in ["SPY", "QQQ", "DIA"]:
                idx = indexes.get(ticker, {})
                if idx.get("status") == "ok":
                    signal = idx.get("signal", 0)
                    signal_str = {2: "++", 1: "+", 0: "~", -1: "-"}.get(signal, "?")
                    index_details.append(f"{ticker}{signal_str}")

            if index_details:
                detail = f"{trend} ({', '.join(index_details)})"
            else:
                detail = trend
        else:
            # Fallback to old single-index method if multi-index fails
            is_bullish, pct_200, pct_50 = self.fetcher.get_market_direction()

            if is_bullish:
                if pct_50 > 0:
                    score = max_score
                    detail = "bullish (SPY above MAs)"
                else:
                    score = max_score * 0.7
                    detail = "neutral-bullish"
            else:
                if pct_50 > 0:
                    score = max_score * 0.5
                    detail = "recovery mode"
                else:
                    score = max_score * 0.2
                    detail = "bearish (SPY below MAs)"

        self._market_score = round(score, 1)
        self._market_detail = detail

        return self._market_score, self._market_detail


def calculate_coiled_spring_score(data, score, config: dict = None) -> dict:
    """
    Detect "Coiled Spring" earnings catalyst setup.

    A Coiled Spring setup identifies stocks with explosive earnings potential:
    - Long consolidation (15+ weeks in base = stored energy)
    - Consistent earnings beats (3+ consecutive)
    - Strong current earnings (C score >= 12)
    - Quality stock overall (total score >= 65)
    - Low institutional ownership (room for institutions to buy)
    - Rising relative strength (L score >= 8)
    - Approaching earnings (1-14 days out)

    Args:
        data: StockData object with stock fundamentals
        score: CANSLIMScore object with scoring breakdown
        config: Optional config dict (defaults loaded from config_loader)

    Returns dict with:
    - is_coiled_spring: bool
    - cs_score: float (bonus points to add to composite)
    - cs_details: str (explanation for logs/alerts)
    - allow_pre_earnings_buy: bool (override earnings block)
    - factors: dict (detailed breakdown for debugging)
    """
    # Load config if not provided
    if config is None:
        try:
            from config_loader import config as app_config
            config = app_config.get('coiled_spring', {})
        except ImportError:
            config = {}

    # Get thresholds with defaults
    thresholds = config.get('thresholds', {})
    min_weeks_in_base = thresholds.get('min_weeks_in_base', 15)
    min_beat_streak = thresholds.get('min_beat_streak', 3)
    min_c_score = thresholds.get('min_c_score', 12)
    min_total_score = thresholds.get('min_total_score', 65)
    max_institutional_pct = thresholds.get('max_institutional_pct', 40)
    min_l_score = thresholds.get('min_l_score', 8)

    # Get earnings window settings
    earnings_window = config.get('earnings_window', {})
    alert_days = earnings_window.get('alert_days', 14)
    allow_buy_days = earnings_window.get('allow_buy_days', 7)
    block_days = earnings_window.get('block_days', 1)

    # Get scoring settings
    scoring = config.get('scoring', {})
    base_bonus = scoring.get('base_bonus', 20)
    long_base_bonus = scoring.get('long_base_bonus', 10)
    strong_beat_bonus = scoring.get('strong_beat_bonus', 5)
    max_bonus = scoring.get('max_bonus', 35)

    # Initialize result
    result = {
        "is_coiled_spring": False,
        "cs_score": 0,
        "cs_details": "",
        "allow_pre_earnings_buy": False,
        "factors": {}
    }

    # Extract data fields
    weeks_in_base = getattr(data, 'weeks_in_base', 0) or 0
    earnings_beat_streak = getattr(data, 'earnings_beat_streak', 0) or 0
    days_to_earnings = getattr(data, 'days_to_earnings', None)
    institutional_pct = getattr(data, 'institutional_holders_pct', 0) or 0

    # Get scores
    c_score = score.c_score if hasattr(score, 'c_score') else 0
    l_score = score.l_score if hasattr(score, 'l_score') else 0
    total_score = score.total_score if hasattr(score, 'total_score') else 0

    # Store factors for debugging
    factors = {
        "weeks_in_base": weeks_in_base,
        "earnings_beat_streak": earnings_beat_streak,
        "days_to_earnings": days_to_earnings,
        "institutional_pct": institutional_pct,
        "c_score": c_score,
        "l_score": l_score,
        "total_score": total_score,
        "thresholds": {
            "min_weeks_in_base": min_weeks_in_base,
            "min_beat_streak": min_beat_streak,
            "min_c_score": min_c_score,
            "min_total_score": min_total_score,
            "max_institutional_pct": max_institutional_pct,
            "min_l_score": min_l_score,
        }
    }
    result["factors"] = factors

    # Check ALL criteria (must all pass)
    criteria_met = []
    criteria_failed = []

    # 1. Long consolidation
    if weeks_in_base >= min_weeks_in_base:
        criteria_met.append(f"{weeks_in_base}w base")
    else:
        criteria_failed.append(f"weeks_in_base ({weeks_in_base} < {min_weeks_in_base})")

    # 2. Consistent earnings beats
    if earnings_beat_streak >= min_beat_streak:
        criteria_met.append(f"{earnings_beat_streak} beats")
    else:
        criteria_failed.append(f"beat_streak ({earnings_beat_streak} < {min_beat_streak})")

    # 3. Strong current earnings
    if c_score >= min_c_score:
        criteria_met.append(f"C:{c_score:.0f}")
    else:
        criteria_failed.append(f"c_score ({c_score:.1f} < {min_c_score})")

    # 4. Quality stock overall
    if total_score >= min_total_score:
        criteria_met.append(f"score:{total_score:.0f}")
    else:
        criteria_failed.append(f"total_score ({total_score:.1f} < {min_total_score})")

    # 5. Low institutional ownership (room to buy)
    if institutional_pct <= max_institutional_pct:
        criteria_met.append(f"{institutional_pct:.0f}% inst")
    else:
        criteria_failed.append(f"institutional ({institutional_pct:.0f}% > {max_institutional_pct}%)")

    # 6. Rising relative strength
    if l_score >= min_l_score:
        criteria_met.append(f"L:{l_score:.0f}")
    else:
        criteria_failed.append(f"l_score ({l_score:.1f} < {min_l_score})")

    # 7. Approaching earnings (within alert window)
    if days_to_earnings is not None and block_days < days_to_earnings <= alert_days:
        criteria_met.append(f"earnings {days_to_earnings}d")
    else:
        if days_to_earnings is None:
            criteria_failed.append("no earnings date")
        elif days_to_earnings <= block_days:
            criteria_failed.append(f"too close to earnings ({days_to_earnings}d <= {block_days}d)")
        else:
            criteria_failed.append(f"earnings too far ({days_to_earnings}d > {alert_days}d)")

    # All criteria must pass
    if len(criteria_failed) == 0:
        result["is_coiled_spring"] = True

        # Calculate bonus score
        cs_score = base_bonus

        # Extra bonus for very long consolidation (20+ weeks)
        if weeks_in_base >= 20:
            cs_score += long_base_bonus

        # Extra bonus for strong beat streak (5+)
        if earnings_beat_streak >= 5:
            cs_score += strong_beat_bonus

        # Cap at max bonus
        cs_score = min(cs_score, max_bonus)
        result["cs_score"] = cs_score

        # Build details string
        result["cs_details"] = f"CS: {', '.join(criteria_met)}"

        # Allow pre-earnings buy if beyond the hard block
        if days_to_earnings is not None and days_to_earnings > block_days:
            result["allow_pre_earnings_buy"] = True

        # Calculate quality ranking score (higher = better candidate)
        # Used to rank and prioritize CS candidates
        ranking_weights = config.get('ranking_weights', {})
        w_base = ranking_weights.get('weeks_in_base', 1.5)
        w_beats = ranking_weights.get('beat_streak', 3.0)
        w_l = ranking_weights.get('l_score', 2.0)
        w_total = ranking_weights.get('total_score', 0.5)
        low_inst_bonus = ranking_weights.get('low_inst_bonus', 10)

        quality_rank = (
            weeks_in_base * w_base +           # Longer base = more stored energy
            earnings_beat_streak * w_beats +   # More consistency = better
            l_score * w_l +                    # Strong RS = momentum
            total_score * w_total              # Overall quality
        )
        # Bonus for truly low institutional (< 30%)
        if institutional_pct < 30:
            quality_rank += low_inst_bonus

        result["quality_rank"] = round(quality_rank, 1)

    else:
        # Not a coiled spring - store what failed for debugging
        factors["criteria_failed"] = criteria_failed
        result["cs_details"] = f"Not CS: {criteria_failed[0]}"
        result["quality_rank"] = 0

    return result


@dataclass
class GrowthModeScore:
    """Container for Growth Mode scores (alternative for pre-revenue companies)"""
    ticker: str
    total_score: float = 0.0
    is_growth_stock: bool = True  # Flag to indicate this uses growth mode

    # Individual scores (max points in parentheses)
    r_score: float = 0.0  # Revenue Growth (20)
    f_score: float = 0.0  # Funding/Financial Health (15)
    n_score: float = 0.0  # New Highs (15)
    s_score: float = 0.0  # Supply & Demand (15)
    l_score: float = 0.0  # Leader vs Laggard (15)
    i_score: float = 0.0  # Institutional Ownership (10)
    m_score: float = 0.0  # Market Direction (10)

    # Details for display
    r_detail: str = ""
    f_detail: str = ""
    n_detail: str = ""
    s_detail: str = ""
    l_detail: str = ""
    i_detail: str = ""
    m_detail: str = ""


class GrowthModeScorer:
    """
    Alternative scoring for pre-revenue/high-growth companies.

    Unlike traditional CANSLIM which penalizes companies without earnings,
    Growth Mode evaluates companies based on:
    - R: Revenue growth (replaces C+A earnings metrics)
    - F: Funding health (cash runway, debt levels)
    - N: New highs (same as CANSLIM)
    - S: Supply & demand (same as CANSLIM)
    - L: Leader (same as CANSLIM)
    - I: Institutional (same as CANSLIM)
    - M: Market (same as CANSLIM)
    """

    MAX_SCORES = {
        'R': 20,  # Revenue replaces C+A
        'F': 15,  # Funding health
        'N': 15,
        'S': 15,
        'L': 15,
        'I': 10,
        'M': 10,  # Slightly reduced to balance R
    }

    def __init__(self, data_fetcher: DataFetcher, canslim_scorer: CANSLIMScorer = None):
        self.fetcher = data_fetcher
        # Reuse CANSLIM scorer for common methods
        self.canslim_scorer = canslim_scorer or CANSLIMScorer(data_fetcher)

    def should_use_growth_mode(self, stock_data: StockData) -> bool:
        """
        Determine if a stock should be scored using Growth Mode.

        Growth Mode is for companies where CANSLIM scoring doesn't work well:
        - Pre-revenue or consistently negative earnings (C score can't be calculated)
        - Minimal profit margin < 1% (early-stage companies reinvesting everything)

        Profitable companies with positive earnings should use CANSLIM, even if
        they have high revenue growth. CANSLIM's C and A scores already reward
        earnings growth.
        """
        if not stock_data.quarterly_earnings:
            return True  # No earnings data - use growth mode

        recent_earnings = stock_data.quarterly_earnings[:4]
        if not recent_earnings:
            return True  # No recent earnings data

        # Filter out None values
        valid_earnings = [e for e in recent_earnings if e is not None]
        if not valid_earnings:
            return True  # No valid earnings

        # Pre-revenue or loss-making: ALL recent quarters are negative or zero
        if all(e <= 0 for e in valid_earnings):
            return True

        # Early-stage with minimal profit margin (< 1% of revenue)
        # These companies are reinvesting everything and C/A scores don't reflect true potential
        if stock_data.quarterly_revenue and len(stock_data.quarterly_revenue) >= 4:
            revenue_sum = sum(stock_data.quarterly_revenue[:4])
            earnings_sum = sum(valid_earnings)
            if revenue_sum > 0 and earnings_sum / revenue_sum < 0.01:
                return True

        # Profitable companies with positive earnings should use CANSLIM
        return False

    def score_stock(self, stock_data: StockData) -> GrowthModeScore:
        """Calculate Growth Mode score for a stock"""
        score = GrowthModeScore(ticker=stock_data.ticker)

        if not stock_data.is_valid:
            return score

        # Calculate each criterion
        score.r_score, score.r_detail = self._score_revenue_growth(stock_data)
        score.f_score, score.f_detail = self._score_funding_health(stock_data)

        # Reuse CANSLIM scoring for these components
        score.n_score, score.n_detail = self.canslim_scorer._score_new_highs(stock_data)
        score.s_score, score.s_detail = self.canslim_scorer._score_supply_demand(stock_data)
        score.l_score, score.l_detail = self.canslim_scorer._score_leader(stock_data)
        score.i_score, score.i_detail = self.canslim_scorer._score_institutional(stock_data)

        # Market score (slightly reduced max for Growth Mode)
        m_score, m_detail = self.canslim_scorer._score_market()
        # Scale from 15 to 10 max
        score.m_score = round(m_score * (10 / 15), 1)
        score.m_detail = m_detail

        score.total_score = (
            score.r_score + score.f_score + score.n_score +
            score.s_score + score.l_score + score.i_score + score.m_score
        )

        return score

    def _score_revenue_growth(self, data: StockData) -> tuple[float, str]:
        """
        R - Revenue Growth (20 pts max)
        Replaces C+A for growth stocks. Focuses on revenue momentum.
        - YoY quarterly revenue growth (up to 15 pts)
        - Revenue acceleration bonus (up to 5 pts)
        """
        max_score = self.MAX_SCORES['R']
        growth_max = 15
        accel_max = 5

        if not data.quarterly_revenue or len(data.quarterly_revenue) < 5:
            # Try to use annual revenue
            if data.annual_revenue and len(data.annual_revenue) >= 2:
                current = data.annual_revenue[0]
                prior = data.annual_revenue[1]
                if prior > 0:
                    growth = ((current - prior) / prior) * 100
                    score = self._revenue_growth_to_score(growth, growth_max)
                    return round(score, 1), f"Annual rev: {growth:+.0f}%"
            return 0, "No revenue data"

        # Calculate YoY quarterly revenue growth
        current_q = data.quarterly_revenue[0]
        prior_year_q = data.quarterly_revenue[4] if len(data.quarterly_revenue) > 4 else 0

        if prior_year_q <= 0:
            if current_q > 0:
                return max_score * 0.6, "New revenue"
            return 0, "No revenue"

        yoy_growth = ((current_q - prior_year_q) / prior_year_q) * 100

        # Base score from YoY growth
        growth_score = self._revenue_growth_to_score(yoy_growth, growth_max)

        # Revenue acceleration bonus
        accel_score = 0
        accel_detail = ""
        if len(data.quarterly_revenue) >= 6:
            # Compare current quarter's YoY growth to previous quarter's YoY growth
            prev_q = data.quarterly_revenue[1]
            prev_prior_year_q = data.quarterly_revenue[5] if len(data.quarterly_revenue) > 5 else 0

            if prev_prior_year_q > 0:
                prev_yoy_growth = ((prev_q - prev_prior_year_q) / prev_prior_year_q) * 100

                if yoy_growth > prev_yoy_growth and yoy_growth > 0:
                    accel_score = accel_max
                    accel_detail = " +accel"
                elif yoy_growth >= prev_yoy_growth * 0.9 and yoy_growth > 20:
                    accel_score = accel_max * 0.5
                    accel_detail = " steady"

        total_score = min(growth_score + accel_score, max_score)
        return round(total_score, 1), f"Rev YoY: {yoy_growth:+.0f}%{accel_detail}"

    def _revenue_growth_to_score(self, growth_pct: float, max_score: float) -> float:
        """Convert revenue growth percentage to a score"""
        if growth_pct >= 50:
            return max_score
        elif growth_pct >= 30:
            return max_score * 0.85
        elif growth_pct >= 20:
            return max_score * 0.7
        elif growth_pct >= 10:
            return max_score * 0.5
        elif growth_pct >= 0:
            return max_score * 0.3
        else:
            # Declining revenue
            return max(0, max_score * 0.1 * (1 + growth_pct / 50))

    def _score_funding_health(self, data: StockData) -> tuple[float, str]:
        """
        F - Funding/Financial Health (15 pts max)
        For pre-revenue companies, cash runway and debt levels matter.
        - Cash runway estimation (up to 10 pts)
        - Debt-to-cash ratio (up to 5 pts)
        """
        max_score = self.MAX_SCORES['F']
        runway_max = 10
        debt_max = 5

        cash = data.cash_and_equivalents
        debt = data.total_debt

        # If no balance sheet data, give partial neutral score
        if cash <= 0 and debt <= 0:
            # Check for institutional backing as proxy
            if data.institutional_holders_pct >= 30:
                return max_score * 0.5, "No data (inst. backed)"
            return max_score * 0.3, "No balance sheet data"

        # Estimate quarterly cash burn from operating losses
        burn_rate = 0
        if data.quarterly_earnings and len(data.quarterly_earnings) >= 2:
            recent_earnings = data.quarterly_earnings[:2]
            avg_loss = sum(e for e in recent_earnings if e < 0) / max(1, len([e for e in recent_earnings if e < 0]))
            if avg_loss < 0:
                # Rough estimate: loss * shares outstanding = quarterly burn
                shares = data.shares_outstanding if data.shares_outstanding > 0 else 1
                burn_rate = abs(avg_loss) * shares

        # Cash runway scoring
        runway_score = 0
        runway_detail = ""

        if burn_rate > 0 and cash > 0:
            quarters_runway = cash / burn_rate
            if quarters_runway >= 12:  # 3+ years
                runway_score = runway_max
                runway_detail = f"{quarters_runway:.0f}Q runway"
            elif quarters_runway >= 8:  # 2+ years
                runway_score = runway_max * 0.8
                runway_detail = f"{quarters_runway:.0f}Q runway"
            elif quarters_runway >= 4:  # 1+ years
                runway_score = runway_max * 0.5
                runway_detail = f"{quarters_runway:.0f}Q runway"
            else:
                runway_score = runway_max * 0.2
                runway_detail = f"Low runway ({quarters_runway:.0f}Q)"
        elif cash > 0:
            # Profitable or no burn rate data - just check if they have cash
            if cash >= 100_000_000:  # $100M+
                runway_score = runway_max * 0.9
                runway_detail = f"${cash/1e9:.1f}B cash"
            elif cash >= 50_000_000:  # $50M+
                runway_score = runway_max * 0.7
                runway_detail = f"${cash/1e6:.0f}M cash"
            else:
                runway_score = runway_max * 0.4
                runway_detail = f"${cash/1e6:.0f}M cash"

        # Debt-to-cash ratio scoring
        debt_score = 0
        debt_detail = ""

        if cash > 0:
            debt_ratio = debt / cash if debt > 0 else 0
            if debt_ratio <= 0.3:  # Very low debt
                debt_score = debt_max
                debt_detail = ", low debt"
            elif debt_ratio <= 0.7:  # Manageable debt
                debt_score = debt_max * 0.7
                debt_detail = ", mod debt"
            elif debt_ratio <= 1.5:  # High but covered
                debt_score = debt_max * 0.4
                debt_detail = ", high debt"
            else:
                debt_detail = ", excess debt"
        else:
            # No cash data, just check debt level
            if debt <= 0:
                debt_score = debt_max * 0.8
                debt_detail = ", no debt"
            elif debt < 50_000_000:
                debt_score = debt_max * 0.5
                debt_detail = f", ${debt/1e6:.0f}M debt"

        total_score = min(runway_score + debt_score, max_score)
        return round(total_score, 1), f"{runway_detail}{debt_detail}"


class TechnicalAnalyzer:
    """Technical analysis helpers for base detection and breakout alerts"""

    @staticmethod
    def detect_base_pattern(weekly_data: list) -> dict:
        """
        Detect consolidation base patterns from weekly price data.
        Returns: {type: 'flat'|'cup'|'cup_with_handle'|'double_bottom'|'none',
                  weeks: int, pivot_price: float, handle_low: float (if applicable)}

        Patterns detected:
        - Flat base: 5+ consecutive weeks with <15% weekly range
        - Cup with handle: U-shaped recovery with small pullback forming handle
        - Double bottom: Two distinct lows at similar levels with recovery between
        """
        if not weekly_data or len(weekly_data) < 5:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Filter out None values and get valid weeks
        valid_weeks = [w for w in weekly_data if w.get("high") and w.get("low") and w.get("close")]
        if len(valid_weeks) < 5:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Try each pattern detection in order of specificity

        # 1. Cup with handle (most specific)
        cup_handle = TechnicalAnalyzer._detect_cup_with_handle(valid_weeks)
        if cup_handle["type"] != "none":
            return cup_handle

        # 2. Double bottom
        double_bottom = TechnicalAnalyzer._detect_double_bottom(valid_weeks)
        if double_bottom["type"] != "none":
            return double_bottom

        # 3. Flat base (check for consecutive tight weeks)
        flat_base = TechnicalAnalyzer._detect_flat_base(valid_weeks)
        if flat_base["type"] != "none":
            return flat_base

        return {"type": "none", "weeks": 0, "pivot_price": 0}

    @staticmethod
    def _detect_flat_base(valid_weeks: list) -> dict:
        """
        Detect flat base pattern: 5+ weeks where the OVERALL price range is tight (<15%).

        Per O'Neil's CANSLIM methodology:
        - A flat base is a consolidation where the stock trades sideways
        - The TOTAL range (highest high to lowest low) should be < 15%
        - Duration: typically 5-15 weeks
        - Pivot point: the highest high within the base
        """
        recent_weeks = valid_weeks[-15:]  # Look at last 15 weeks for longer bases

        if len(recent_weeks) < 5:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        best_base = {"type": "none", "weeks": 0, "pivot_price": 0}

        # Try different window sizes from 5 to len(recent_weeks)
        # Find the longest valid flat base
        for window_size in range(5, len(recent_weeks) + 1):
            # Slide the window across recent weeks
            for start_idx in range(len(recent_weeks) - window_size + 1):
                window = recent_weeks[start_idx:start_idx + window_size]

                # Calculate the OVERALL range across all weeks in the window
                highest_high = max(w["high"] for w in window)
                lowest_low = min(w["low"] for w in window)

                if lowest_low <= 0:
                    continue

                # Total consolidation range as percentage
                total_range_pct = (highest_high - lowest_low) / lowest_low

                # A valid flat base has < 15% total range
                if total_range_pct < 0.15:
                    # Prefer longer bases (more significant)
                    if window_size > best_base["weeks"]:
                        best_base = {
                            "type": "flat",
                            "weeks": window_size,
                            "pivot_price": highest_high,
                            "base_low": lowest_low,
                            "base_depth": round(total_range_pct * 100, 1)
                        }

        return best_base

    @staticmethod
    def _detect_cup_with_handle(valid_weeks: list) -> dict:
        """
        Detect cup-with-handle pattern:
        - Cup: 15%+ decline from left high, then recovery to within 15% of that high
        - Handle: Small pullback (5-15%) after cup completion, lasting 1-4 weeks
        - Pivot: Top of the handle (not the cup's highest point)
        """
        if len(valid_weeks) < 10:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Use more data for longer patterns (up to 26 weeks / 6 months)
        analysis_weeks = valid_weeks[-26:] if len(valid_weeks) >= 26 else valid_weeks[-10:]

        closes = [w["close"] for w in analysis_weeks]
        highs = [w["high"] for w in analysis_weeks]
        lows = [w["low"] for w in analysis_weeks]

        # Find the cup's left high (peak before the decline)
        # Look for highest point in first half of the data
        first_half = len(closes) // 2
        left_high_idx = closes[:first_half + 1].index(max(closes[:first_half + 1]))
        left_high = closes[left_high_idx]

        # Find the cup's bottom (lowest point after left high)
        bottom_idx = left_high_idx + closes[left_high_idx:].index(min(closes[left_high_idx:]))
        bottom = closes[bottom_idx]

        # Calculate decline
        decline = (left_high - bottom) / left_high if left_high > 0 else 0

        # Need at least 15% decline to form a valid cup
        if decline < 0.15:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Check for recovery after bottom (right side of cup)
        if bottom_idx >= len(closes) - 2:  # Need at least 2 weeks after bottom
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        post_bottom = closes[bottom_idx:]
        right_high = max(post_bottom)
        right_high_idx = bottom_idx + post_bottom.index(right_high)

        # Recovery should bring price back to within 15% of left high
        recovery_level = right_high / left_high if left_high > 0 else 0
        if recovery_level < 0.85:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Now look for handle formation (small pullback after right high)
        if right_high_idx >= len(closes) - 1:
            # No room for handle, but we have a valid cup - return cup without handle
            return {
                "type": "cup",
                "weeks": len(analysis_weeks),
                "pivot_price": max(highs[right_high_idx:right_high_idx + 1])
            }

        # Handle detection: look for pullback of 5-15% after right high
        handle_weeks = closes[right_high_idx:]
        if len(handle_weeks) >= 2:
            handle_low = min(handle_weeks[1:])  # Exclude the high itself
            handle_low_idx = right_high_idx + 1 + handle_weeks[1:].index(handle_low)
            handle_decline = (right_high - handle_low) / right_high if right_high > 0 else 0

            # Valid handle: 5-15% pullback, lasting 1-4 weeks
            handle_length = handle_low_idx - right_high_idx
            if 0.05 <= handle_decline <= 0.15 and 1 <= handle_length <= 4:
                # Pivot is the high of the handle (top of consolidation after pullback)
                handle_highs = highs[right_high_idx:handle_low_idx + 2] if handle_low_idx + 2 <= len(highs) else highs[right_high_idx:]
                pivot_price = max(handle_highs) if handle_highs else right_high

                return {
                    "type": "cup_with_handle",
                    "weeks": len(analysis_weeks),
                    "pivot_price": pivot_price,
                    "handle_low": handle_low,
                    "cup_depth": round(decline * 100, 1),
                    "handle_depth": round(handle_decline * 100, 1)
                }

        # Valid cup but no handle yet
        return {
            "type": "cup",
            "weeks": len(analysis_weeks),
            "pivot_price": highs[right_high_idx]
        }

    @staticmethod
    def _detect_double_bottom(valid_weeks: list) -> dict:
        """
        Detect double-bottom pattern (W pattern):
        - Two distinct lows within 3% of each other
        - Recovery of at least 10% between the two lows
        - Second low should not break below first low significantly
        - Pivot: The high point between the two lows
        """
        if len(valid_weeks) < 8:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        analysis_weeks = valid_weeks[-20:] if len(valid_weeks) >= 20 else valid_weeks

        closes = [w["close"] for w in analysis_weeks]
        highs = [w["high"] for w in analysis_weeks]
        lows = [w["low"] for w in analysis_weeks]

        # Find the two lowest points
        # First, find the absolute lowest
        first_low_idx = lows.index(min(lows))
        first_low = lows[first_low_idx]

        # Find second low: must be at least 3 weeks apart from first
        second_low = float('inf')
        second_low_idx = -1

        for i, low in enumerate(lows):
            if abs(i - first_low_idx) >= 3:  # At least 3 weeks separation
                if low < second_low:
                    second_low = low
                    second_low_idx = i

        if second_low_idx == -1:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Ensure first_low_idx < second_low_idx for chronological order
        if first_low_idx > second_low_idx:
            first_low_idx, second_low_idx = second_low_idx, first_low_idx
            first_low, second_low = second_low, first_low

        # Check if lows are within 3% of each other
        low_diff_pct = abs(first_low - second_low) / first_low if first_low > 0 else 1
        if low_diff_pct > 0.03:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Find the middle peak (between the two lows)
        middle_section = closes[first_low_idx:second_low_idx + 1]
        if len(middle_section) < 3:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        middle_peak = max(middle_section)
        middle_peak_idx = first_low_idx + middle_section.index(middle_peak)

        # Check for adequate recovery between lows (at least 10%)
        recovery_pct = (middle_peak - first_low) / first_low if first_low > 0 else 0
        if recovery_pct < 0.10:
            return {"type": "none", "weeks": 0, "pivot_price": 0}

        # Pivot is the high point between the two lows
        pivot_price = max(highs[first_low_idx:second_low_idx + 1])

        return {
            "type": "double_bottom",
            "weeks": second_low_idx - first_low_idx + 1,
            "pivot_price": pivot_price,
            "first_low": first_low,
            "second_low": second_low,
            "middle_peak": middle_peak
        }

    @staticmethod
    def calculate_volume_ratio(stock_data: StockData) -> float:
        """
        Calculate volume ratio vs 50-day average.
        Uses max(today, yesterday) to avoid misleading partial-day readings.
        """
        if stock_data.avg_volume_50d <= 0:
            return 1.0

        # Get today's volume
        today_volume = stock_data.current_volume or 0

        # Try to get yesterday's volume from price_history
        yesterday_volume = 0
        if hasattr(stock_data, 'price_history') and not stock_data.price_history.empty:
            try:
                volumes = stock_data.price_history['Volume'].tolist()
                if len(volumes) >= 2:
                    # volumes[-1] is today (possibly partial), volumes[-2] is yesterday (complete)
                    yesterday_volume = volumes[-2] if volumes[-2] and volumes[-2] > 0 else 0
            except (KeyError, IndexError, TypeError, AttributeError):
                pass  # Expected when price_history lacks Volume column or has insufficient data

        # Use max of today and yesterday to avoid partial-day issues
        # If today's volume already exceeds yesterday, it's likely a significant move
        effective_volume = max(today_volume, yesterday_volume)

        if effective_volume > 0:
            return effective_volume / stock_data.avg_volume_50d

        return 1.0

    @staticmethod
    def calculate_multiday_volume_confirmation(stock_data: StockData, days: int = 5) -> dict:
        """
        Check volume trend over multiple days for stronger breakout confirmation.
        Returns: {avg_ratio: float, days_above_avg: int, volume_trend: 'increasing'|'stable'|'decreasing'}

        Legitimate breakouts typically show:
        - Multiple days with above-average volume
        - Increasing volume trend during the breakout
        """
        result = {
            "avg_ratio": 1.0,
            "days_above_avg": 0,
            "volume_trend": "stable",
            "confirmation_score": 0  # 0-100 score for volume confirmation strength
        }

        # Try to get daily volume data from price_history DataFrame
        if hasattr(stock_data, 'price_history') and not stock_data.price_history.empty:
            try:
                volumes = stock_data.price_history['Volume'].tail(days).tolist()
                avg_volume = stock_data.avg_volume_50d

                if avg_volume > 0 and len(volumes) >= 3:
                    # Calculate ratios for each day
                    ratios = [v / avg_volume for v in volumes if v and v > 0]

                    if ratios:
                        result["avg_ratio"] = sum(ratios) / len(ratios)
                        result["days_above_avg"] = sum(1 for r in ratios if r > 1.0)

                        # Determine volume trend (compare first half to second half)
                        mid = len(ratios) // 2
                        if mid > 0:
                            first_half_avg = sum(ratios[:mid]) / mid
                            second_half_avg = sum(ratios[mid:]) / len(ratios[mid:])

                            if second_half_avg > first_half_avg * 1.1:
                                result["volume_trend"] = "increasing"
                            elif second_half_avg < first_half_avg * 0.9:
                                result["volume_trend"] = "decreasing"

                        # Calculate confirmation score (0-100)
                        # Factors: avg ratio, days above average, trend
                        score = 0

                        # Avg ratio contribution (up to 40 points)
                        if result["avg_ratio"] >= 2.0:
                            score += 40
                        elif result["avg_ratio"] >= 1.5:
                            score += 30
                        elif result["avg_ratio"] >= 1.2:
                            score += 20
                        elif result["avg_ratio"] >= 1.0:
                            score += 10

                        # Days above average contribution (up to 30 points)
                        score += min(result["days_above_avg"] * 6, 30)

                        # Trend contribution (up to 30 points)
                        if result["volume_trend"] == "increasing":
                            score += 30
                        elif result["volume_trend"] == "stable":
                            score += 15
                        # Decreasing trend gets 0 points

                        result["confirmation_score"] = min(score, 100)

            except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                pass  # Expected when price_history lacks Volume column or has insufficient data

        # Fallback to weekly data if daily not available
        elif stock_data.weekly_price_history:
            recent_weeks = stock_data.weekly_price_history[-4:]  # Last 4 weeks
            avg_volume = stock_data.avg_volume_50d

            if avg_volume > 0 and recent_weeks:
                volumes = [w.get("volume", 0) for w in recent_weeks if w.get("volume")]
                if volumes:
                    # Weekly volumes are cumulative, so compare to weekly average
                    weekly_avg = avg_volume * 5  # Approximate weekly volume
                    ratios = [v / weekly_avg for v in volumes if v > 0]

                    if ratios:
                        result["avg_ratio"] = sum(ratios) / len(ratios)
                        result["days_above_avg"] = sum(1 for r in ratios if r > 1.0)

                        # Simple score for weekly data
                        result["confirmation_score"] = min(int(result["avg_ratio"] * 30) + result["days_above_avg"] * 10, 100)

        return result

    @staticmethod
    def is_breaking_out(stock_data: StockData, base_pattern: dict) -> tuple[bool, float]:
        """
        Check if stock is actively breaking out of its base (actionable buy point).
        Returns: (is_breaking_out, volume_ratio)

        STRICT breakout criteria - only marks TRUE breakouts, not extended stocks:
        - Pre-breakout: Price within 3% BELOW pivot (building for breakout)
        - Active breakout: Price 0-5% ABOVE pivot with volume confirmation
        - Extended stocks (>5% above pivot) are NOT marked as breaking out
          because the optimal entry point has passed

        This ensures the breakout list shows ACTIONABLE opportunities, not
        stocks that already broke out days/weeks ago.
        """
        vol_ratio = TechnicalAnalyzer.calculate_volume_ratio(stock_data)
        vol_confirmation = TechnicalAnalyzer.calculate_multiday_volume_confirmation(stock_data)

        # Effective volume score combines single-day and multi-day analysis
        effective_vol_score = max(
            vol_ratio * 50,  # Single day contribution (1.5x = 75 points)
            vol_confirmation["confirmation_score"]  # Multi-day contribution (0-100)
        )

        if base_pattern["type"] == "none" or base_pattern["pivot_price"] <= 0:
            # No base pattern detected - be VERY strict
            # Only mark as breakout with exceptional volume (2.5x+) AND at new 52-week high
            if stock_data.high_52w and stock_data.high_52w > 0 and stock_data.current_price and stock_data.current_price > 0:
                pct_from_high = (stock_data.high_52w - stock_data.current_price) / stock_data.high_52w
                # Must be within 2% of 52-week high AND have exceptional volume
                if pct_from_high <= 0.02 and vol_ratio >= 2.5:
                    return True, vol_ratio
            return False, vol_ratio

        pivot = base_pattern["pivot_price"]
        current = stock_data.current_price

        if current > 0 and pivot > 0:
            pct_from_pivot = (current - pivot) / pivot

            # EXTENDED CHECK: If stock is >5% above pivot, it's NO LONGER a breakout
            # The buy point has passed - don't chase extended stocks
            if pct_from_pivot > 0.05:
                return False, vol_ratio

            # Active breakout: price 0-5% above pivot with volume confirmation
            # This is the optimal buy zone per CANSLIM methodology
            if 0 <= pct_from_pivot <= 0.05:
                # Require decent volume (1.2x+ single day OR 50+ multi-day score)
                if effective_vol_score >= 50:
                    return True, vol_ratio

            # Pre-breakout: price within 3% below pivot, building for breakout
            # Slightly more lenient on volume - looking for accumulation
            if -0.03 <= pct_from_pivot < 0:
                if effective_vol_score >= 40:
                    return True, vol_ratio

            # Special case for cup_with_handle: can be slightly below pivot
            if base_pattern.get("type") == "cup_with_handle":
                handle_low = base_pattern.get("handle_low", 0)
                if handle_low > 0 and current > handle_low:
                    # Price above handle low and within 5% below pivot
                    if -0.05 <= pct_from_pivot < 0 and effective_vol_score >= 40:
                        return True, vol_ratio

        return False, vol_ratio

    @staticmethod
    def calculate_accumulation_distribution(stock_data, days: int = 20) -> dict:
        """
        Calculate up-day vs down-day volume ratio to detect institutional accumulation.
        Returns dict with:
        - rating: "A" (strong accum) to "E" (strong distrib)
        - score_bonus: -5 to +10 points for composite scoring
        - up_down_ratio: ratio of up-day volume to down-day volume
        - detail: explanation string
        """
        if stock_data.price_history.empty or len(stock_data.price_history) < days:
            return {
                "rating": "C",
                "score_bonus": 0,
                "up_down_ratio": 1.0,
                "detail": "Insufficient data"
            }

        try:
            # Get recent price/volume data
            recent = stock_data.price_history.tail(days)
            closes = recent['Close'].values
            volumes = recent['Volume'].values

            # Calculate daily price changes
            up_volume = 0
            down_volume = 0

            for i in range(1, len(closes)):
                if closes[i] is None or closes[i-1] is None or volumes[i] is None:
                    continue

                price_change = closes[i] - closes[i-1]
                volume = volumes[i]

                if price_change > 0:
                    up_volume += volume
                elif price_change < 0:
                    down_volume += volume

            if down_volume == 0:
                up_down_ratio = 2.0 if up_volume > 0 else 1.0
            else:
                up_down_ratio = up_volume / down_volume

            # Determine rating and score bonus
            if up_down_ratio >= 1.8:
                rating = "A"
                score_bonus = 10
                detail = f"Strong accumulation ({up_down_ratio:.2f}x)"
            elif up_down_ratio >= 1.4:
                rating = "B"
                score_bonus = 5
                detail = f"Accumulation ({up_down_ratio:.2f}x)"
            elif up_down_ratio >= 0.9:
                rating = "C"
                score_bonus = 0
                detail = f"Neutral ({up_down_ratio:.2f}x)"
            elif up_down_ratio >= 0.7:
                rating = "D"
                score_bonus = -3
                detail = f"Distribution ({up_down_ratio:.2f}x)"
            else:
                rating = "E"
                score_bonus = -5
                detail = f"Strong distribution ({up_down_ratio:.2f}x)"

            return {
                "rating": rating,
                "score_bonus": score_bonus,
                "up_down_ratio": round(up_down_ratio, 2),
                "detail": detail
            }

        except Exception as e:
            return {
                "rating": "C",
                "score_bonus": 0,
                "up_down_ratio": 1.0,
                "detail": f"Error: {str(e)}"
            }


if __name__ == "__main__":
    # Test the scorer
    fetcher = DataFetcher()
    scorer = CANSLIMScorer(fetcher)

    test_tickers = ["AAPL", "NVDA", "MSFT"]

    for ticker in test_tickers:
        print(f"\n{'='*50}")
        print(f"Scoring {ticker}")
        print('='*50)

        data = fetcher.get_stock_data(ticker)
        if not data.is_valid:
            print(f"Could not fetch data: {data.error_message}")
            continue

        score = scorer.score_stock(data)

        print(f"Total Score: {score.total_score}/100")
        print(f"C (Current Earnings):  {score.c_score}/15 - {score.c_detail}")
        print(f"A (Annual Earnings):   {score.a_score}/15 - {score.a_detail}")
        print(f"N (New Highs):         {score.n_score}/15 - {score.n_detail}")
        print(f"S (Supply/Demand):     {score.s_score}/15 - {score.s_detail}")
        print(f"L (Leader):            {score.l_score}/15 - {score.l_detail}")
        print(f"I (Institutional):     {score.i_score}/10 - {score.i_detail}")
        print(f"M (Market):            {score.m_score}/15 - {score.m_detail}")
