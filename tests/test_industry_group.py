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


# ── update_stock_group_ranks ──────────────────────────────────────────────────


def _make_stock(industry, current_rank):
    """Mock Stock row with industry + industry_group_rank attrs."""
    s = MagicMock()
    s.industry = industry
    s.industry_group_rank = current_rank
    return s


def _db_returning_stocks(stocks):
    """Build a mock Session whose .query(Stock).filter().all() returns stocks."""
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = stocks
    return db


class TestUpdateStockGroupRanks:
    """Covers backend/industry_group.py:130-170 — the batch-update path that
    pulls new ranks from `rankings` into each Stock's industry_group_rank.

    Four conditional branches per stock:
      1. industry in rankings, rank differs    -> assign + count
      2. industry in rankings, rank matches    -> skip
      3. industry NOT in rankings, rank != None -> null out + count
      4. industry NOT in rankings, rank == None -> skip

    Plus the commit gate: db.commit() runs only when `updated > 0`.
    """

    def test_empty_rankings_returns_zero_and_skips_commit(self):
        """Short-circuit on empty rankings — no stock query, no commit."""
        from backend.industry_group import update_stock_group_ranks
        db = MagicMock()
        result = update_stock_group_ranks(db, {})
        assert result == 0
        db.commit.assert_not_called()
        # Stock query short-circuits before .query() is hit
        db.query.assert_not_called()

    def test_stock_with_changed_rank_is_updated(self):
        """Branch 1: industry in rankings AND rank differs → assign new rank."""
        from backend.industry_group import update_stock_group_ranks
        stock = _make_stock("Software", current_rank=50)
        db = _db_returning_stocks([stock])
        rankings = {"Software": {"rank": 85, "composite_rs": 1.5,
                                  "stock_count": 10, "avg_rs_12m": 1.4,
                                  "avg_rs_3m": 1.6}}
        result = update_stock_group_ranks(db, rankings)
        assert result == 1
        assert stock.industry_group_rank == 85
        db.commit.assert_called_once()

    def test_stock_with_matching_rank_is_not_updated(self):
        """Branch 2: already-correct rank → no write, no count, no commit."""
        from backend.industry_group import update_stock_group_ranks
        stock = _make_stock("Software", current_rank=85)
        db = _db_returning_stocks([stock])
        rankings = {"Software": {"rank": 85, "composite_rs": 1.5,
                                  "stock_count": 10, "avg_rs_12m": 1.4,
                                  "avg_rs_3m": 1.6}}
        result = update_stock_group_ranks(db, rankings)
        assert result == 0
        assert stock.industry_group_rank == 85
        db.commit.assert_not_called()

    def test_stock_missing_from_rankings_with_existing_rank_is_nulled(self):
        """Branch 3: industry has too few members to rank → set rank to None.

        Guards against stale ranks lingering when a stock's industry drops
        out of the universe (e.g., delisted siblings).
        """
        from backend.industry_group import update_stock_group_ranks
        stock = _make_stock("Tiny Niche", current_rank=42)
        db = _db_returning_stocks([stock])
        rankings = {"Software": {"rank": 85, "composite_rs": 1.5,
                                  "stock_count": 10, "avg_rs_12m": 1.4,
                                  "avg_rs_3m": 1.6}}
        result = update_stock_group_ranks(db, rankings)
        assert result == 1
        assert stock.industry_group_rank is None
        db.commit.assert_called_once()

    def test_stock_missing_from_rankings_already_null_is_skipped(self):
        """Branch 4: industry missing AND rank already None → no churn."""
        from backend.industry_group import update_stock_group_ranks
        stock = _make_stock("Tiny Niche", current_rank=None)
        db = _db_returning_stocks([stock])
        rankings = {"Software": {"rank": 85, "composite_rs": 1.5,
                                  "stock_count": 10, "avg_rs_12m": 1.4,
                                  "avg_rs_3m": 1.6}}
        result = update_stock_group_ranks(db, rankings)
        assert result == 0
        assert stock.industry_group_rank is None
        db.commit.assert_not_called()

    def test_mixed_population_counts_only_changed_stocks(self):
        """All four branches in one batch — the counter should reflect only
        the stocks whose rank actually moved."""
        from backend.industry_group import update_stock_group_ranks
        stocks = [
            _make_stock("Software", current_rank=50),       # branch 1: update
            _make_stock("Software", current_rank=85),       # branch 2: skip
            _make_stock("Tiny Niche", current_rank=42),     # branch 3: null
            _make_stock("Tiny Niche", current_rank=None),   # branch 4: skip
        ]
        db = _db_returning_stocks(stocks)
        rankings = {"Software": {"rank": 85, "composite_rs": 1.5,
                                  "stock_count": 10, "avg_rs_12m": 1.4,
                                  "avg_rs_3m": 1.6}}
        result = update_stock_group_ranks(db, rankings)
        assert result == 2
        assert stocks[0].industry_group_rank == 85
        assert stocks[1].industry_group_rank == 85
        assert stocks[2].industry_group_rank is None
        assert stocks[3].industry_group_rank is None
        db.commit.assert_called_once()


# ── get_top_groups ────────────────────────────────────────────────────────────


class TestGetTopGroups:
    """Covers backend/industry_group.py:175-189 — top-N wrapper around
    compute_industry_group_rankings. Sort order is rank DESC (highest = strongest)."""

    def _rankings(self):
        # composite_rs is included to mirror the real shape, but the wrapper
        # only uses 'rank' for sorting.
        return {
            "Software": {"rank": 95, "composite_rs": 1.8, "stock_count": 12,
                         "avg_rs_12m": 1.5, "avg_rs_3m": 2.0},
            "Banks": {"rank": 30, "composite_rs": 0.7, "stock_count": 15,
                      "avg_rs_12m": 0.8, "avg_rs_3m": 0.6},
            "Semis": {"rank": 80, "composite_rs": 1.5, "stock_count": 8,
                      "avg_rs_12m": 1.3, "avg_rs_3m": 1.6},
            "Retail": {"rank": 50, "composite_rs": 1.0, "stock_count": 20,
                       "avg_rs_12m": 1.0, "avg_rs_3m": 1.0},
        }

    def test_returns_groups_sorted_by_rank_descending(self):
        from backend.industry_group import get_top_groups
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=self._rankings()):
            result = get_top_groups(MagicMock(), limit=10)
        names = [g["industry"] for g in result]
        assert names == ["Software", "Semis", "Retail", "Banks"]

    def test_respects_limit_param(self):
        """limit=2 should return only the top-2 groups."""
        from backend.industry_group import get_top_groups
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=self._rankings()):
            result = get_top_groups(MagicMock(), limit=2)
        assert len(result) == 2
        assert [g["industry"] for g in result] == ["Software", "Semis"]

    def test_flattens_industry_name_into_dict(self):
        """Each entry should carry 'industry' alongside the rank/RS fields."""
        from backend.industry_group import get_top_groups
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=self._rankings()):
            result = get_top_groups(MagicMock(), limit=1)
        top = result[0]
        assert top["industry"] == "Software"
        assert top["rank"] == 95
        assert top["composite_rs"] == 1.8
        assert top["stock_count"] == 12


# ── get_bottom_groups ─────────────────────────────────────────────────────────


class TestGetBottomGroups:
    """Covers backend/industry_group.py:183-190 — mirror of get_top_groups,
    sorted by rank ASC (weakest first)."""

    def _rankings(self):
        return {
            "Software": {"rank": 95, "composite_rs": 1.8, "stock_count": 12,
                         "avg_rs_12m": 1.5, "avg_rs_3m": 2.0},
            "Banks": {"rank": 30, "composite_rs": 0.7, "stock_count": 15,
                      "avg_rs_12m": 0.8, "avg_rs_3m": 0.6},
            "Oil": {"rank": 12, "composite_rs": 0.5, "stock_count": 18,
                    "avg_rs_12m": 0.6, "avg_rs_3m": 0.4},
            "Retail": {"rank": 50, "composite_rs": 1.0, "stock_count": 20,
                       "avg_rs_12m": 1.0, "avg_rs_3m": 1.0},
        }

    def test_returns_groups_sorted_by_rank_ascending(self):
        from backend.industry_group import get_bottom_groups
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=self._rankings()):
            result = get_bottom_groups(MagicMock(), limit=10)
        names = [g["industry"] for g in result]
        assert names == ["Oil", "Banks", "Retail", "Software"]

    def test_respects_limit_param(self):
        from backend.industry_group import get_bottom_groups
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=self._rankings()):
            result = get_bottom_groups(MagicMock(), limit=2)
        assert [g["industry"] for g in result] == ["Oil", "Banks"]


# ── get_group_rotation_summary ────────────────────────────────────────────────


class TestGetGroupRotationSummary:
    """Covers backend/industry_group.py:192-227 — bear-market-report rotation
    bucketing. Groups are split by `avg_rs_3m - avg_rs_12m`:
      diff >  0.05 → improving
      diff < -0.05 → deteriorating
      otherwise    → excluded from both buckets
    Each bucket is sorted by |diff| descending and capped at 15 entries.
    """

    @staticmethod
    def _rank_entry(industry, rank, rs_3m, rs_12m, stock_count=10):
        return industry, {
            "rank": rank,
            "composite_rs": rs_3m * 0.6 + rs_12m * 0.4,
            "stock_count": stock_count,
            "avg_rs_12m": rs_12m,
            "avg_rs_3m": rs_3m,
        }

    def test_empty_rankings_returns_empty_summary(self):
        from backend.industry_group import get_group_rotation_summary
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value={}):
            result = get_group_rotation_summary(MagicMock())
        assert result == {"improving": [], "deteriorating": [], "total_groups": 0}

    def test_strong_positive_diff_routes_to_improving(self):
        """rs_3m - rs_12m well above +0.05 → improving bucket."""
        from backend.industry_group import get_group_rotation_summary
        rankings = dict([
            self._rank_entry("HotMomentum", 90, rs_3m=2.0, rs_12m=1.0),  # diff +1.0
        ])
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=rankings):
            result = get_group_rotation_summary(MagicMock())
        assert len(result["improving"]) == 1
        assert result["improving"][0]["industry"] == "HotMomentum"
        assert result["improving"][0]["rs_diff"] == pytest.approx(1.0, abs=1e-6)
        assert result["deteriorating"] == []
        assert result["total_groups"] == 1

    def test_strong_negative_diff_routes_to_deteriorating(self):
        from backend.industry_group import get_group_rotation_summary
        rankings = dict([
            self._rank_entry("LosingSteam", 30, rs_3m=0.4, rs_12m=1.5),  # diff -1.1
        ])
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=rankings):
            result = get_group_rotation_summary(MagicMock())
        assert len(result["deteriorating"]) == 1
        assert result["deteriorating"][0]["industry"] == "LosingSteam"
        assert result["deteriorating"][0]["rs_diff"] == pytest.approx(-1.1, abs=1e-6)
        assert result["improving"] == []

    def test_neutral_diff_excluded_from_both_buckets(self):
        """A group whose 3m vs 12m RS differ by <= 0.05 in absolute value
        is considered flat — it should NOT appear in either bucket."""
        from backend.industry_group import get_group_rotation_summary
        rankings = dict([
            self._rank_entry("Flat", 50, rs_3m=1.02, rs_12m=1.00),  # diff +0.02
            self._rank_entry("BarelyDown", 49, rs_3m=0.98, rs_12m=1.00),  # diff -0.02
        ])
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=rankings):
            result = get_group_rotation_summary(MagicMock())
        assert result["improving"] == []
        assert result["deteriorating"] == []
        # total_groups still counts ALL groups, not just the bucketed ones
        assert result["total_groups"] == 2

    def test_caps_each_bucket_at_fifteen(self):
        """Many groups in the same direction should be truncated to 15 each,
        with the steepest movers retained (sort by rs_diff)."""
        from backend.industry_group import get_group_rotation_summary
        # 20 improving, 20 deteriorating — total 40 in rankings
        rankings = {}
        for i in range(20):
            # diff = +0.10 .. +2.0 (descending should keep the biggest diffs)
            name, payload = self._rank_entry(
                f"Imp_{i}", rank=80 + i, rs_3m=1.0 + (i + 1) * 0.1, rs_12m=1.0,
            )
            rankings[name] = payload
        for i in range(20):
            name, payload = self._rank_entry(
                f"Det_{i}", rank=20 - i, rs_3m=1.0 - (i + 1) * 0.1, rs_12m=1.0,
            )
            rankings[name] = payload
        with patch("backend.industry_group.compute_industry_group_rankings",
                   return_value=rankings):
            result = get_group_rotation_summary(MagicMock())
        assert len(result["improving"]) == 15
        assert len(result["deteriorating"]) == 15
        # Improving is sorted descending — first entry has the biggest diff
        diffs_imp = [g["rs_diff"] for g in result["improving"]]
        assert diffs_imp == sorted(diffs_imp, reverse=True)
        # Deteriorating is sorted ascending — most-negative first
        diffs_det = [g["rs_diff"] for g in result["deteriorating"]]
        assert diffs_det == sorted(diffs_det)
        assert result["total_groups"] == 40
