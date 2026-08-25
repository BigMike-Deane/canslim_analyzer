"""Tests for stock-split detection and position adjustment.

Regression coverage for the SFBS 2:1 split (effective 2026-08-21): raw
quotes halved while the position's cost basis stayed pre-split, which read
as a -50.2% collapse and fired a phantom STOP LOSS on user 3.
"""
import types
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

from backend import ai_trader
from backend.ai_trader import fetch_recent_split, maybe_apply_split_adjustment


@pytest.fixture(autouse=True)
def _clear_split_cache():
    """The per-day lookup cache is correct in production but leaks results
    across tests that mock different FMP responses for the same ticker."""
    ai_trader._split_lookup_cache.clear()
    yield
    ai_trader._split_lookup_cache.clear()


def make_position(**overrides):
    pos = types.SimpleNamespace(
        ticker="SFBS",
        shares=41.6571721544947,
        cost_basis=87.77,
        current_price=87.40,
        peak_price=91.00,
        user_id=3,
    )
    for key, value in overrides.items():
        setattr(pos, key, value)
    return pos


def fmp_response(rows, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = rows
    return resp


class TestFetchRecentSplit:
    @patch.dict("os.environ", {"FMP_API_KEY": "test-key"})
    @patch("requests.get")
    def test_recent_split_returns_factor(self, mock_get):
        recent = (date.today() - timedelta(days=2)).isoformat()
        mock_get.return_value = fmp_response([
            {"symbol": "SFBS", "date": recent, "numerator": 2, "denominator": 1},
            {"symbol": "SFBS", "date": "2016-12-21", "numerator": 2, "denominator": 1},
        ])
        assert fetch_recent_split("SFBS") == 2.0

    @patch.dict("os.environ", {"FMP_API_KEY": "test-key"})
    @patch("requests.get")
    def test_old_split_ignored(self, mock_get):
        mock_get.return_value = fmp_response([
            {"symbol": "SFBS", "date": "2016-12-21", "numerator": 2, "denominator": 1},
        ])
        assert fetch_recent_split("SFBS") is None

    @patch.dict("os.environ", {"FMP_API_KEY": "test-key"})
    @patch("requests.get")
    def test_reverse_split_factor(self, mock_get):
        recent = date.today().isoformat()
        mock_get.return_value = fmp_response([
            {"symbol": "XYZ", "date": recent, "numerator": 1, "denominator": 10},
        ])
        assert fetch_recent_split("XYZ") == 0.1

    @patch.dict("os.environ", {"FMP_API_KEY": ""})
    def test_no_api_key_returns_none(self):
        assert fetch_recent_split("SFBS") is None

    @patch.dict("os.environ", {"FMP_API_KEY": "test-key"})
    @patch("requests.get")
    def test_http_error_fails_soft(self, mock_get):
        mock_get.return_value = fmp_response([], status=429)
        assert fetch_recent_split("SFBS") is None

    @patch.dict("os.environ", {"FMP_API_KEY": "test-key"})
    @patch("requests.get")
    def test_malformed_rows_skipped(self, mock_get):
        mock_get.return_value = fmp_response([
            {"symbol": "SFBS", "date": None, "numerator": 2, "denominator": 1},
            {"symbol": "SFBS", "date": "not-a-date", "numerator": 2, "denominator": 1},
            {"symbol": "SFBS", "date": date.today().isoformat(), "numerator": 0, "denominator": 1},
        ])
        assert fetch_recent_split("SFBS") is None


class TestMaybeApplySplitAdjustment:
    @patch("backend.ai_trader.fetch_recent_split", return_value=2.0)
    def test_confirmed_split_rescales_position(self, mock_fetch):
        pos = make_position()
        assert maybe_apply_split_adjustment(pos, 43.71) is True
        assert abs(pos.shares - 41.6571721544947 * 2) < 1e-9
        assert abs(pos.cost_basis - 87.77 / 2) < 1e-9
        assert abs(pos.peak_price - 91.00 / 2) < 1e-9
        mock_fetch.assert_called_once_with("SFBS")

    @patch("backend.ai_trader.fetch_recent_split", return_value=2.0)
    def test_ordinary_move_never_queries_fmp(self, mock_fetch):
        pos = make_position()
        assert maybe_apply_split_adjustment(pos, 85.00) is False
        assert pos.cost_basis == 87.77
        mock_fetch.assert_not_called()

    @patch("backend.ai_trader.fetch_recent_split", return_value=None)
    def test_real_crash_without_split_untouched(self, mock_fetch):
        pos = make_position()
        assert maybe_apply_split_adjustment(pos, 43.71) is False
        assert pos.shares == 41.6571721544947
        assert pos.cost_basis == 87.77
        mock_fetch.assert_called_once()

    @patch("backend.ai_trader.fetch_recent_split", return_value=2.0)
    def test_idempotent_after_caller_updates_price(self, mock_fetch):
        pos = make_position()
        assert maybe_apply_split_adjustment(pos, 43.71) is True
        # update_position_prices stores the new price after adjusting;
        # the next cycle's ratio is ~1 so no second rescale happens.
        pos.current_price = 43.71
        assert maybe_apply_split_adjustment(pos, 43.90) is False
        assert abs(pos.cost_basis - 87.77 / 2) < 1e-9

    @patch("backend.ai_trader.fetch_recent_split", return_value=0.1)
    def test_reverse_split_rescales_up(self, mock_fetch):
        pos = make_position(ticker="XYZ", shares=1000.0, cost_basis=2.0,
                            current_price=1.9, peak_price=2.5)
        assert maybe_apply_split_adjustment(pos, 19.0) is True
        assert abs(pos.shares - 100.0) < 1e-9
        assert abs(pos.cost_basis - 20.0) < 1e-9
        assert abs(pos.peak_price - 25.0) < 1e-9

    def test_missing_prices_are_skipped(self):
        assert maybe_apply_split_adjustment(make_position(current_price=None), 43.71) is False
        assert maybe_apply_split_adjustment(make_position(), 0) is False
