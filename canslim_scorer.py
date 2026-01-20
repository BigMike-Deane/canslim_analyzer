"""
CANSLIM Scorer Module
Implements scoring logic for all 7 CANSLIM criteria
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from data_fetcher import StockData, DataFetcher


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

    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self._market_score: float | None = None
        self._market_detail: str = ""

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
        REFINED: Uses TTM comparison + EPS acceleration bonus.
        - Base: TTM EPS growth vs prior year (up to 12 pts)
        - Bonus: EPS acceleration (current Q growth > prior Q growth) (up to 3 pts)
        """
        max_score = self.MAX_SCORES['C']
        base_max = 12  # Base score for TTM growth
        accel_max = 3  # Bonus for acceleration

        # Need at least 8 quarters for TTM vs prior TTM comparison
        if len(data.quarterly_earnings) < 8:
            # Fallback to simpler comparison with anomaly filtering
            if len(data.quarterly_earnings) >= 4:
                return self._score_earnings_with_anomaly_filter(data, max_score)
            return 0, "Insufficient data"

        # Calculate TTM (sum of last 4 quarters)
        current_ttm = sum(data.quarterly_earnings[0:4])
        prior_ttm = sum(data.quarterly_earnings[4:8])

        if prior_ttm == 0:
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

        # Base score from TTM growth (up to 12 pts)
        if ttm_growth >= 25:
            base_score = base_max
        elif ttm_growth >= 0:
            base_score = (ttm_growth / 25) * base_max
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

        total_score = min(base_score + accel_score, max_score)
        return round(total_score, 1), f"TTM: {ttm_growth:+.0f}%{accel_detail}"

    def _score_earnings_with_anomaly_filter(self, data: StockData, max_score: float) -> tuple[float, str]:
        """
        Fallback scoring with anomaly filtering for stocks with limited data.
        Filters out extreme QoQ swings (>50%) that are likely one-time items.
        """
        earnings = data.quarterly_earnings[:4]

        # Calculate growth rates between consecutive quarters
        growth_rates = []
        for i in range(len(earnings) - 1):
            if earnings[i + 1] != 0 and earnings[i + 1] is not None:
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

        # Base score from CAGR (up to 12 pts)
        if cagr >= 25:
            cagr_score = cagr_max
        elif cagr >= 0:
            cagr_score = (cagr / 25) * cagr_max
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

        # Calculate volume ratio
        recent_vol = data.current_volume if data.current_volume > 0 else data.price_history['Volume'].iloc[-5:].mean()
        vol_ratio = recent_vol / data.avg_volume_50d

        # Check price trend (last 20 days)
        if len(data.price_history) >= 20:
            recent_prices = data.price_history['Close'].iloc[-20:]
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        else:
            price_change = 0

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

        prices = data.price_history['Close']
        sp500_prices = sp500_history['Close']

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

        # Calculate RS ratios
        rs_12m = (1 + stock_return_12m) / (1 + sp500_return_12m) if sp500_return_12m != -1 else 1.0
        rs_3m = (1 + stock_return_3m) / (1 + sp500_return_3m) if sp500_return_3m != -1 else 1.0

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

    def _score_institutional(self, data: StockData) -> tuple[float, str]:
        """
        I - Institutional Ownership (10 pts max)
        Sweet spot: 20-60% institutional ownership
        """
        max_score = self.MAX_SCORES['I']
        inst_pct = data.institutional_holders_pct

        if inst_pct <= 0:
            # Default to neutral score when data unavailable (e.g., rate limited)
            # Most large-cap stocks have 40-80% institutional ownership
            return max_score * 0.5, "No data (neutral)"

        # Ideal range is 20-60%
        if 20 <= inst_pct <= 60:
            score = max_score
        elif 10 <= inst_pct < 20:
            score = max_score * 0.7
        elif 60 < inst_pct <= 80:
            score = max_score * 0.7
        elif inst_pct < 10:
            score = max_score * 0.3  # Too little institutional interest
        else:
            score = max_score * 0.4  # Too crowded

        return round(score, 1), f"{inst_pct:.0f}% inst."

    def _score_market(self) -> tuple[float, str]:
        """
        M - Market Direction (15 pts max)
        Based on S&P 500 position relative to moving averages
        """
        if self._market_score is not None:
            return self._market_score, self._market_detail

        max_score = self.MAX_SCORES['M']

        is_bullish, pct_200, pct_50 = self.fetcher.get_market_direction()

        if is_bullish:
            if pct_50 > 0:  # Above both MAs
                score = max_score
                detail = "bullish (above 50/200 MA)"
            else:  # Above 200 but below 50
                score = max_score * 0.7
                detail = "neutral-bullish"
        else:
            if pct_50 > 0:  # Below 200 but above 50 (recovery)
                score = max_score * 0.5
                detail = "recovery mode"
            else:  # Below both
                score = max_score * 0.2
                detail = "bearish (below MAs)"

        self._market_score = round(score, 1)
        self._market_detail = detail

        return self._market_score, self._market_detail


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
