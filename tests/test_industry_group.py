"""
Tests for industry_group.py — Industry Group Strength Rankings

Tests the group-level RS computation, bonus/penalty calculation,
and stock ranking updates.
"""
import pytest
from unittest.mock import MagicMock, patch


# ── get_industry_group_bonus ──────────────────────────────────────────────────

class TestGetIndustryGroupBonus:
    """Test the scoring bonus/penalty calculation from group rank."""

    def test_top_group_max_bonus(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(100)
        assert bonus == pytest.approx(3.0, abs=0.1)

    def test_top_group_rank_80_gives_bonus_2(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(80)
        assert bonus == pytest.approx(2.0, abs=0.1)

    def test_rank_90_gives_bonus_between_2_and_3(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(90)
        assert 2.0 < bonus < 3.0

    def test_rank_70_gives_moderate_bonus(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(70)
        assert 0.5 < bonus < 2.0

    def test_neutral_zone_rank_50(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(50)
        assert bonus == 0

    def test_neutral_zone_rank_40(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(40)
        assert bonus == 0

    def test_weak_group_rank_30_gives_penalty(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(30)
        assert bonus < 0

    def test_bottom_group_rank_1_gives_max_penalty(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(1)
        assert bonus <= -2.5

    def test_rank_20_gives_penalty(self):
        from backend.industry_group import get_industry_group_bonus
        bonus = get_industry_group_bonus(20)
        assert bonus < 0
        assert bonus >= -2.0

    def test_bonus_is_graduated_not_binary(self):
        """Ensure smooth graduation across ranks."""
        from backend.industry_group import get_industry_group_bonus
        bonuses = [get_industry_group_bonus(r) for r in range(1, 101)]
        # Should be monotonically non-decreasing
        for i in range(1, len(bonuses)):
            assert bonuses[i] >= bonuses[i-1], \
                f"Rank {i+1} bonus {bonuses[i]} < rank {i} bonus {bonuses[i-1]}"


# ── compute_industry_group_rankings ───────────────────────────────────────────

class TestComputeIndustryGroupRankings:
    """Test the SQL-based group ranking computation."""

    def test_returns_empty_dict_when_no_data(self):
        from backend.industry_group import compute_industry_group_rankings
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.group_by.return_value.having.return_value.all.return_value = []
        result = compute_industry_group_rankings(mock_db)
        assert result == {}

    def test_ranking_assigns_percentiles(self):
        """Verify that groups are ranked 1-100 by composite RS."""
        from backend.industry_group import compute_industry_group_rankings

        # Create mock rows for 5 industry groups
        mock_rows = []
        for i, (industry, rs_12m, rs_3m) in enumerate([
            ("Software", 1.5, 1.8),
            ("Banks", 0.8, 0.7),
            ("Semiconductors", 1.3, 1.5),
            ("Retail", 1.0, 1.0),
            ("Oil & Gas", 0.6, 0.5),
        ]):
            row = MagicMock()
            row.industry = industry
            row.avg_rs_12m = rs_12m
            row.avg_rs_3m = rs_3m
            row.stock_count = 10
            mock_rows.append(row)

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.group_by.return_value.having.return_value.all.return_value = mock_rows

        result = compute_industry_group_rankings(mock_db)

        assert len(result) == 5
        # Software should be ranked highest (best RS)
        assert result["Software"]["rank"] > result["Banks"]["rank"]
        assert result["Software"]["rank"] > result["Oil & Gas"]["rank"]
        # Oil & Gas should be ranked lowest
        assert result["Oil & Gas"]["rank"] < result["Retail"]["rank"]

    def test_composite_rs_weights_3m_more(self):
        """Composite RS should weight 3m at 60% (recent momentum matters more)."""
        from backend.industry_group import compute_industry_group_rankings

        row = MagicMock()
        row.industry = "Test"
        row.avg_rs_12m = 1.0
        row.avg_rs_3m = 2.0
        row.stock_count = 5

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.group_by.return_value.having.return_value.all.return_value = [row]

        result = compute_industry_group_rankings(mock_db)
        # Expected: 1.0 * 0.40 + 2.0 * 0.60 = 1.60
        assert result["Test"]["composite_rs"] == pytest.approx(1.60, abs=0.01)


# ── Integration with scorer ───────────────────────────────────────────────────

class TestScorerIntegration:
    """Test that industry_group_rank is properly passed through scoring."""

    def test_score_stock_with_group_rank_applies_bonus(self, mock_stock_data):
        """Verify that passing industry_group_rank modifies the L score."""
        from canslim_scorer import CANSLIMScorer
        from unittest.mock import MagicMock

        mock_fetcher = MagicMock()
        mock_fetcher.get_sp500_history.return_value = mock_stock_data.price_history
        scorer = CANSLIMScorer(mock_fetcher)

        # Score without group rank
        score_no_rank = scorer.score_stock(mock_stock_data)
        # Score with top group rank
        score_top_rank = scorer.score_stock(mock_stock_data, industry_group_rank=95)
        # Score with bottom group rank
        score_bottom_rank = scorer.score_stock(mock_stock_data, industry_group_rank=5)

        # Top rank should boost L score
        assert score_top_rank.l_score >= score_no_rank.l_score
        # Bottom rank should reduce L score
        assert score_bottom_rank.l_score <= score_no_rank.l_score

    def test_score_stock_without_group_rank_unchanged(self, mock_stock_data):
        """Verify that None group rank doesn't change behavior."""
        from canslim_scorer import CANSLIMScorer

        mock_fetcher = MagicMock()
        mock_fetcher.get_sp500_history.return_value = mock_stock_data.price_history
        scorer = CANSLIMScorer(mock_fetcher)

        score_none = scorer.score_stock(mock_stock_data, industry_group_rank=None)
        score_default = scorer.score_stock(mock_stock_data)

        assert score_none.total_score == score_default.total_score
