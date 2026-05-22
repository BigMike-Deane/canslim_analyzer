"""
Coverage tests for backend/earnings_gapup.py.

Closes the 39.47% -> 100% gap (missing lines 67-105, 115-184, 194-202,
212, 223, 258, 261-263). Pure DB+compute logic; the Yahoo chart fetcher
inside _check_gap_up is patched at `requests.get`, and send_gapup_alert's
late imports are patched at the source module.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import earnings_gapup
from backend.earnings_gapup import (
    _check_gap_up,
    _is_actionable,
    find_earnings_gapups,
    send_gapup_alert,
)
from backend.database import Base, Stock


# ── Fixtures / helpers ────────────────────────────────────────────────────────


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _seed_stock(db, ticker="AAPL", **overrides):
    """Build a Stock row that passes find_earnings_gapups' base filter."""
    defaults = dict(
        ticker=ticker,
        name=f"{ticker} Inc.",
        sector="Technology",
        industry="Software",
        current_price=150.0,
        canslim_score=80.0,
        c_score=15.0,
        a_score=14.0,
        l_score=13.0,
        next_earnings_date=date.today() - timedelta(days=2),
        earnings_beat_streak=3,
        earnings_surprise_pct=10.0,
        base_type="flat",
        weeks_in_base=8,
        industry_group_rank=85,
        rs_3m=12.0,
    )
    defaults.update(overrides)
    stock = Stock(**defaults)
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


def _yahoo_response(timestamps, opens, closes, volumes, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {"open": opens, "close": closes, "volume": volumes}
                        ]
                    },
                }
            ]
        }
    }
    return resp


def _ts_for(d):
    return int(datetime(d.year, d.month, d.day).timestamp())


# ── find_earnings_gapups ──────────────────────────────────────────────────────


class TestFindEarningsGapups:
    """Covers lines 67-105: the main query function's loop, sort, log, return."""

    def test_returns_empty_when_no_stocks_match_query(self, db_session):
        assert find_earnings_gapups(db_session) == []

    def test_excludes_stock_with_null_canslim_score(self, db_session):
        # Query filter `canslim_score != None` removes this row before the loop
        _seed_stock(db_session, ticker="NONE", canslim_score=None)
        assert find_earnings_gapups(db_session) == []

    def test_excludes_stock_with_zero_beat_streak(self, db_session):
        _seed_stock(db_session, ticker="ZERO", earnings_beat_streak=0)
        assert find_earnings_gapups(db_session) == []

    def test_includes_stock_with_qualifying_gap(self, db_session):
        _seed_stock(db_session)
        with patch.object(
            earnings_gapup,
            "_check_gap_up",
            return_value={
                "gap_pct": 8.5,
                "volume_ratio": 2.0,
                "earnings_open": 162.5,
                "prior_close": 150.0,
            },
        ):
            result = find_earnings_gapups(db_session)
        assert len(result) == 1
        row = result[0]
        assert row["ticker"] == "AAPL"
        assert row["gap_pct"] == 8.5
        assert row["volume_ratio"] == 2.0
        assert row["sector"] == "Technology"
        assert row["beat_streak"] == 3
        assert row["base_type"] == "flat"
        assert row["earnings_date"] is not None
        assert isinstance(row["is_actionable"], bool)

    def test_excludes_gap_below_threshold(self, db_session):
        _seed_stock(db_session)
        with patch.object(
            earnings_gapup,
            "_check_gap_up",
            return_value={
                "gap_pct": 4.0,
                "volume_ratio": 1.5,
                "earnings_open": 156.0,
                "prior_close": 150.0,
            },
        ):
            assert find_earnings_gapups(db_session) == []

    def test_swallows_check_gap_up_exception(self, db_session):
        _seed_stock(db_session)
        with patch.object(
            earnings_gapup, "_check_gap_up", side_effect=RuntimeError("boom")
        ):
            assert find_earnings_gapups(db_session) == []

    def test_sorts_descending_by_gap_size_and_logs_count(self, db_session, caplog):
        _seed_stock(db_session, ticker="AAA")
        _seed_stock(db_session, ticker="BBB")
        _seed_stock(db_session, ticker="CCC")

        responses = {
            "AAA": {"gap_pct": 6.0, "volume_ratio": 1.0, "earnings_open": 0, "prior_close": 0},
            "BBB": {"gap_pct": 12.0, "volume_ratio": 2.0, "earnings_open": 0, "prior_close": 0},
            "CCC": {"gap_pct": 8.0, "volume_ratio": 1.5, "earnings_open": 0, "prior_close": 0},
        }
        with patch.object(
            earnings_gapup,
            "_check_gap_up",
            side_effect=lambda s, _m: responses[s.ticker],
        ):
            with caplog.at_level("INFO", logger="backend.earnings_gapup"):
                result = find_earnings_gapups(db_session)

        assert [r["ticker"] for r in result] == ["BBB", "CCC", "AAA"]
        assert any(
            "Found 3 post-earnings gap-ups" in rec.message for rec in caplog.records
        )


# ── _check_gap_up ─────────────────────────────────────────────────────────────


class TestCheckGapUp:
    """Covers lines 115-184: the Yahoo chart fetch + gap math."""

    def _stock(self, earnings_date=None):
        s = MagicMock()
        s.ticker = "AAPL"
        s.next_earnings_date = earnings_date or (date.today() - timedelta(days=2))
        return s

    def test_returns_none_on_non_200(self):
        with patch(
            "requests.get",
            return_value=_yahoo_response([], [], [], [], status=500),
        ):
            assert _check_gap_up(self._stock(), 5.0) is None

    def test_returns_none_on_empty_chart_result(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"chart": {"result": []}}
        with patch("requests.get", return_value=resp):
            assert _check_gap_up(self._stock(), 5.0) is None

    def test_returns_none_when_timestamps_missing(self):
        with patch(
            "requests.get",
            return_value=_yahoo_response([], [], [], []),
        ):
            assert _check_gap_up(self._stock(), 5.0) is None

    def test_returns_none_when_closes_empty(self):
        with patch(
            "requests.get",
            return_value=_yahoo_response(
                [_ts_for(date.today())], [100.0], [], []
            ),
        ):
            assert _check_gap_up(self._stock(), 5.0) is None

    def test_returns_none_when_all_dates_before_earnings(self):
        earnings_d = date.today()
        old_dates = [date.today() - timedelta(days=i) for i in range(10, 5, -1)]
        ts = [_ts_for(d) for d in old_dates]
        with patch(
            "requests.get",
            return_value=_yahoo_response(
                ts, [100.0] * 5, [100.0] * 5, [1000] * 5
            ),
        ):
            assert _check_gap_up(self._stock(earnings_date=earnings_d), 5.0) is None

    def test_returns_none_when_earnings_is_first_day_in_series(self):
        # earnings_idx = 0 -> no prior_close -> None
        earnings_d = date.today() - timedelta(days=5)
        dates = [earnings_d + timedelta(days=i) for i in range(5)]
        ts = [_ts_for(d) for d in dates]
        with patch(
            "requests.get",
            return_value=_yahoo_response(
                ts, [100.0] * 5, [100.0] * 5, [1000] * 5
            ),
        ):
            assert _check_gap_up(self._stock(earnings_date=earnings_d), 5.0) is None

    def test_returns_none_when_prior_close_is_zero(self):
        earnings_d = date.today() - timedelta(days=2)
        dates = [earnings_d - timedelta(days=2)] + [
            earnings_d + timedelta(days=i) for i in range(3)
        ]
        ts = [_ts_for(d) for d in dates]
        # prior_close = closes[0] = 0 -> falsy -> None
        with patch(
            "requests.get",
            return_value=_yahoo_response(
                ts,
                [100.0, 110.0, 100.0, 100.0],
                [0.0, 100.0, 100.0, 100.0],
                [1000] * 4,
            ),
        ):
            assert _check_gap_up(self._stock(earnings_date=earnings_d), 5.0) is None

    def test_returns_none_when_gap_below_minimum(self):
        earnings_d = date.today() - timedelta(days=2)
        dates = [
            earnings_d - timedelta(days=3),
            earnings_d - timedelta(days=2),
            earnings_d,
            earnings_d + timedelta(days=1),
        ]
        ts = [_ts_for(d) for d in dates]
        # close[1]=100, open[2]=102 -> 2% gap < 5% threshold -> None
        with patch(
            "requests.get",
            return_value=_yahoo_response(
                ts,
                [100.0, 100.0, 102.0, 102.0],
                [100.0, 100.0, 102.0, 102.0],
                [1000] * 4,
            ),
        ):
            assert _check_gap_up(self._stock(earnings_date=earnings_d), 5.0) is None

    def test_returns_gap_dict_when_gap_qualifies(self):
        earnings_d = date.today() - timedelta(days=2)
        dates = [
            earnings_d - timedelta(days=i) for i in range(5, 0, -1)
        ] + [earnings_d, earnings_d + timedelta(days=1)]
        ts = [_ts_for(d) for d in dates]
        opens = [100.0] * 5 + [110.0, 112.0]
        closes = [100.0] * 5 + [111.0, 112.0]
        volumes = [1000] * 5 + [5000, 4000]
        with patch(
            "requests.get",
            return_value=_yahoo_response(ts, opens, closes, volumes),
        ):
            result = _check_gap_up(self._stock(earnings_date=earnings_d), 5.0)
        assert result is not None
        assert result["gap_pct"] == 10.0
        assert result["volume_ratio"] == 5.0
        assert result["earnings_open"] == 110.0
        assert result["prior_close"] == 100.0

    def test_returns_none_on_requests_exception(self):
        with patch("requests.get", side_effect=ConnectionError("network down")):
            assert _check_gap_up(self._stock(), 5.0) is None

    def test_avg_volume_defaults_to_one_when_prior_volumes_empty(self):
        # All prior-day volumes are 0/None -> prior_volumes list comp is empty
        # -> avg_volume defaults to 1 via the `if prior_volumes else 1` branch
        earnings_d = date.today() - timedelta(days=2)
        dates = [earnings_d - timedelta(days=i) for i in range(3, 0, -1)] + [earnings_d]
        ts = [_ts_for(d) for d in dates]
        opens = [100.0] * 3 + [110.0]
        closes = [100.0] * 3 + [110.0]
        volumes = [0, None, 0, 5000]
        with patch(
            "requests.get",
            return_value=_yahoo_response(ts, opens, closes, volumes),
        ):
            result = _check_gap_up(self._stock(earnings_date=earnings_d), 5.0)
        assert result is not None
        assert result["gap_pct"] == 10.0


# ── _is_actionable ────────────────────────────────────────────────────────────


class TestIsActionable:
    """Covers lines 194-202: the four conditions controlling actionability."""

    def _stock(self, **overrides):
        s = MagicMock()
        s.canslim_score = overrides.get("canslim_score", 80.0)
        s.earnings_beat_streak = overrides.get("earnings_beat_streak", 3)
        return s

    def test_low_canslim_score_returns_false(self):
        assert _is_actionable(self._stock(canslim_score=50), {"volume_ratio": 2.0}) is False

    def test_none_canslim_score_treated_as_zero(self):
        assert _is_actionable(self._stock(canslim_score=None), {"volume_ratio": 2.0}) is False

    def test_low_volume_ratio_returns_false(self):
        assert _is_actionable(self._stock(), {"volume_ratio": 1.0}) is False

    def test_missing_volume_ratio_returns_false(self):
        assert _is_actionable(self._stock(), {}) is False

    def test_low_beat_streak_returns_false(self):
        assert _is_actionable(self._stock(earnings_beat_streak=1), {"volume_ratio": 2.0}) is False

    def test_none_beat_streak_treated_as_zero(self):
        assert _is_actionable(self._stock(earnings_beat_streak=None), {"volume_ratio": 2.0}) is False

    def test_all_conditions_met_returns_true(self):
        assert _is_actionable(
            self._stock(canslim_score=80, earnings_beat_streak=3),
            {"volume_ratio": 2.0},
        ) is True


# ── send_gapup_alert ──────────────────────────────────────────────────────────


def _gap(ticker="AAPL", actionable=True):
    return {
        "ticker": ticker,
        "is_actionable": actionable,
        "gap_pct": 8.0,
        "volume_ratio": 2.0,
        "canslim_score": 80,
    }


class TestSendGapupAlert:
    """Covers lines 212, 223, 258, 261-263 (+ the full happy/cooldown paths)."""

    def setup_method(self):
        earnings_gapup._recent_gapup_alerts.clear()

    def test_empty_list_returns_false(self):
        assert send_gapup_alert([]) is False

    def test_market_closed_returns_false(self):
        with patch("backend.ai_trader.is_market_open", return_value=False):
            assert send_gapup_alert([_gap()]) is False

    def test_no_actionable_entries_returns_false(self):
        with patch("backend.ai_trader.is_market_open", return_value=True):
            assert send_gapup_alert([_gap(actionable=False)]) is False

    def test_cooldown_blocks_recent_repeat(self):
        earnings_gapup._recent_gapup_alerts["AAPL"] = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        )
        with patch("backend.ai_trader.is_market_open", return_value=True), patch(
            "backend.email_utils.send_webhook_notification", return_value=True
        ) as send_mock:
            result = send_gapup_alert([_gap()])
        assert result is False
        send_mock.assert_not_called()

    def test_successful_send_marks_cooldown(self):
        with patch("backend.ai_trader.is_market_open", return_value=True), patch(
            "backend.email_utils.send_webhook_notification", return_value=True
        ) as send_mock:
            result = send_gapup_alert([_gap()])
        assert result is True
        send_mock.assert_called_once()
        assert "AAPL" in earnings_gapup._recent_gapup_alerts

    def test_failed_webhook_returns_false_and_skips_cooldown_update(self):
        with patch("backend.ai_trader.is_market_open", return_value=True), patch(
            "backend.email_utils.send_webhook_notification", return_value=False
        ):
            result = send_gapup_alert([_gap()])
        assert result is False
        assert "AAPL" not in earnings_gapup._recent_gapup_alerts

    def test_cleanup_evicts_stale_cooldown_entries(self):
        # > 2x cooldown window old -> should be evicted on next successful send
        earnings_gapup._recent_gapup_alerts["STALE"] = (
            datetime.now(timezone.utc) - timedelta(hours=72)
        )
        with patch("backend.ai_trader.is_market_open", return_value=True), patch(
            "backend.email_utils.send_webhook_notification", return_value=True
        ):
            send_gapup_alert([_gap()])
        assert "STALE" not in earnings_gapup._recent_gapup_alerts
        assert "AAPL" in earnings_gapup._recent_gapup_alerts

    def test_exception_in_alert_path_returns_false(self):
        with patch("backend.ai_trader.is_market_open", side_effect=RuntimeError("boom")):
            assert send_gapup_alert([_gap()]) is False
