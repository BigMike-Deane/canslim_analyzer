"""Tests for scripts/grow_training_pool.py.

The script is a thin CLI over POST /api/backtests with pre-flight filtering
against /api/admin/backtest-pairs. We verify:
  - the candidate grid generates well-formed (start < end, end <= today) entries
  - filter_fresh drops collisions with the existing pool
  - fetch_existing_pairs handles the API response shape
  - dry-run mode makes no POSTs, only the GET for existing pairs
  - --execute path POSTs the filtered sweep with the right payload
  - failure paths produce rc=1 and log helpfully
"""

import importlib.util
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "grow_training_pool.py"


def _load_script():
    """Import the script as a module without executing main() at import time."""
    spec = importlib.util.spec_from_file_location("grow_training_pool", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script():
    return _load_script()


@pytest.fixture
def mock_empty_pool(monkeypatch, script):
    """Mock fetch_existing_pairs to return no existing pairs.
    Lets the full candidate grid pass through filter_fresh."""
    get = MagicMock()
    get.return_value.json.return_value = {"pairs": [], "total": 0}
    get.return_value.raise_for_status.return_value = None
    monkeypatch.setattr(script.requests, "get", get)
    return get


class TestCandidateGrid:
    def test_grid_is_non_empty(self, script):
        grid = script.generate_candidate_grid()
        assert len(grid) > 0

    def test_grid_entries_well_formed(self, script):
        grid = script.generate_candidate_grid()
        for entry in grid:
            assert set(entry.keys()) == {"start_date", "end_date", "strategy"}
            assert entry["start_date"] < entry["end_date"]
            # No future windows.
            assert date.fromisoformat(entry["end_date"]) <= date.today()

    def test_grid_covers_both_strategies(self, script):
        grid = script.generate_candidate_grid()
        strategies = {e["strategy"] for e in grid}
        assert "nostate_optimized" in strategies
        assert "nostate_cs_bear" in strategies

    def test_grid_explicit_strategies_filter(self, script):
        grid = script.generate_candidate_grid(strategies=["nostate_optimized"])
        assert all(e["strategy"] == "nostate_optimized" for e in grid)

    def test_window_lengths_cover_3yr_and_4yr(self, script):
        grid = script.generate_candidate_grid()
        diffs = set()
        for e in grid:
            sd = date.fromisoformat(e["start_date"])
            ed = date.fromisoformat(e["end_date"])
            diffs.add(ed.year - sd.year)
        # Should include both 3 and 4 year windows.
        assert 3 in diffs
        assert 4 in diffs


class TestFilterFresh:
    def test_no_existing_returns_all(self, script):
        candidates = [
            {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "x"},
            {"start_date": "2021-01-01", "end_date": "2025-01-01", "strategy": "x"},
        ]
        assert script.filter_fresh(candidates, set()) == candidates

    def test_drops_exact_collision(self, script):
        candidates = [
            {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "x"},
            {"start_date": "2021-01-01", "end_date": "2025-01-01", "strategy": "x"},
        ]
        existing = {("2020-01-01", "2024-01-01", "x")}
        fresh = script.filter_fresh(candidates, existing)
        assert len(fresh) == 1
        assert fresh[0]["start_date"] == "2021-01-01"

    def test_strategy_distinguishes_pairs(self, script):
        """Same (start, end) but different strategy is NOT a collision."""
        candidates = [
            {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "a"},
            {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "b"},
        ]
        existing = {("2020-01-01", "2024-01-01", "a")}
        fresh = script.filter_fresh(candidates, existing)
        assert len(fresh) == 1
        assert fresh[0]["strategy"] == "b"


class TestFetchExistingPairs:
    def test_parses_response_shape(self, script, monkeypatch):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "pairs": [
                {"start_date": "2020-01-01", "end_date": "2024-01-01", "strategy": "nostate_optimized"},
                {"start_date": "2021-01-01", "end_date": "2025-01-01", "strategy": "nostate_cs_bear"},
            ],
            "total": 2,
        }
        monkeypatch.setattr(script.requests, "get", MagicMock(return_value=resp))
        pairs = script.fetch_existing_pairs("http://test.local", token="tok")
        assert pairs == {
            ("2020-01-01", "2024-01-01", "nostate_optimized"),
            ("2021-01-01", "2025-01-01", "nostate_cs_bear"),
        }

    def test_sends_auth_header(self, script, monkeypatch):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"pairs": [], "total": 0}
        get = MagicMock(return_value=resp)
        monkeypatch.setattr(script.requests, "get", get)
        script.fetch_existing_pairs("http://test.local", token="secret123")
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret123"

    def test_no_token_no_auth_header(self, script, monkeypatch):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"pairs": [], "total": 0}
        get = MagicMock(return_value=resp)
        monkeypatch.setattr(script.requests, "get", get)
        script.fetch_existing_pairs("http://test.local")
        assert "Authorization" not in get.call_args.kwargs.get("headers", {})


class TestBuildPayload:
    def test_payload_shape_matches_api(self, script):
        entry = {"start_date": "2022-01-01", "end_date": "2026-01-01", "strategy": "nostate_optimized"}
        payload = script.build_payload(entry, 25000.0, "all")
        assert payload == {
            "start_date": "2022-01-01",
            "end_date": "2026-01-01",
            "starting_cash": 25000.0,
            "stock_universe": "all",
            "strategy": "nostate_optimized",
            "profile_overrides": {"ml_signal": {"enabled": False}},
        }

    def test_payload_always_disables_ml_signal(self, script):
        """Non-negotiable: every payload built by this script must override
        ml_signal.enabled to False so the resulting backtest is eligible for
        the ML training pool. Pinning this contract pins the lesson from
        the 2026-05-13 sweep that yielded 0 new training samples."""
        entry = {"start_date": "2019-04-01", "end_date": "2022-04-01", "strategy": "nostate_cs_bear"}
        payload = script.build_payload(entry, 10000.0, "sp500")
        assert "profile_overrides" in payload
        assert payload["profile_overrides"] == {"ml_signal": {"enabled": False}}


class TestDryRun:
    def test_dry_run_makes_no_post_calls(self, script, capsys, monkeypatch, mock_empty_pool):
        def _explode(*a, **kw):
            raise AssertionError("dry-run must not call requests.post")
        monkeypatch.setattr(script.requests, "post", _explode)
        monkeypatch.setattr(sys, "argv", ["grow_training_pool.py"])
        rc = script.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Planned sweep" in out
        assert "--execute" in out

    def test_dry_run_empty_sweep_message(self, script, capsys, monkeypatch):
        """If every candidate is already in the pool, dry-run should say
        no fresh ranges left rather than printing an empty table."""
        # Mock the existing-pairs API to return EVERY candidate in the grid.
        candidates = script.generate_candidate_grid()
        all_existing = [
            {"start_date": c["start_date"], "end_date": c["end_date"], "strategy": c["strategy"]}
            for c in candidates
        ]
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"pairs": all_existing, "total": len(all_existing)}
        monkeypatch.setattr(script.requests, "get", MagicMock(return_value=resp))
        monkeypatch.setattr(sys, "argv", ["grow_training_pool.py"])
        rc = script.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "No fresh date ranges" in out


class TestExecute:
    def test_execute_posts_filtered_sweep(self, script, capsys, monkeypatch, mock_empty_pool):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 999}
        mock_resp.raise_for_status.return_value = None
        post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--api-base", "http://test.local",
             "--max-per-strategy", "2"],
        )
        rc = script.main()
        assert rc == 0
        # 2 per strategy * 2 strategies = 4 POSTs
        assert post.call_count == 4
        first_call = post.call_args_list[0]
        assert first_call.args[0] == "http://test.local/api/backtests"
        payload = first_call.kwargs["json"]
        assert "start_date" in payload and "strategy" in payload

    def test_execute_with_token_sets_auth_header(self, script, monkeypatch, mock_empty_pool):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 1}
        mock_resp.raise_for_status.return_value = None
        post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--token", "secret123",
             "--max-per-strategy", "1"],
        )
        script.main()
        headers = post.call_args_list[0].kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret123"

    def test_execute_empty_sweep_skips_posts(self, script, capsys, monkeypatch):
        """When the pool already covers every candidate, --execute should
        log the no-op and return 0 without issuing any POST."""
        candidates = script.generate_candidate_grid()
        all_existing = [
            {"start_date": c["start_date"], "end_date": c["end_date"], "strategy": c["strategy"]}
            for c in candidates
        ]
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"pairs": all_existing, "total": len(all_existing)}
        monkeypatch.setattr(script.requests, "get", MagicMock(return_value=resp))

        post = MagicMock()
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(sys, "argv", ["grow_training_pool.py", "--execute"])
        rc = script.main()
        assert rc == 0
        assert post.call_count == 0


class TestExecuteFailureHandling:
    """Cover failure branches: API-pairs fetch failure, per-backtest HTTPError,
    generic RequestException, and the post-loop failure report."""

    def test_pairs_fetch_failure_returns_rc_1(self, script, capsys, monkeypatch):
        import requests as _requests

        def _boom(*a, **kw):
            raise _requests.ConnectionError("pairs endpoint down")

        monkeypatch.setattr(script.requests, "get", _boom)
        monkeypatch.setattr(sys, "argv", ["grow_training_pool.py", "--execute"])
        rc = script.main()
        assert rc == 1
        err = capsys.readouterr().err
        assert "Failed to fetch existing pairs" in err

    def test_http_error_collects_and_returns_rc_1(self, script, capsys, monkeypatch, mock_empty_pool):
        import requests as _requests

        err_resp = MagicMock()
        err_resp.status_code = 503
        err_resp.text = "Service Unavailable" * 30
        http_err = _requests.HTTPError(response=err_resp)

        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = http_err

        post = MagicMock(return_value=bad_resp)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--max-per-strategy", "1"],
        )

        rc = script.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "FAILED" in out
        assert "Failures (re-run later):" in out
        assert "HTTP 503:" in out or "503" in out

    def test_request_exception_collects_and_returns_rc_1(self, script, capsys, monkeypatch, mock_empty_pool):
        import requests as _requests

        def _boom(*a, **kw):
            raise _requests.ConnectionError("connection refused")

        monkeypatch.setattr(script.requests, "post", _boom)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--max-per-strategy", "1"],
        )
        rc = script.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "FAILED" in out
        assert "connection refused" in out
        assert "Failures (re-run later):" in out

    def test_partial_failure_still_returns_rc_1(self, script, capsys, monkeypatch, mock_empty_pool):
        import requests as _requests

        ok_resp = MagicMock()
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {"id": 42}

        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.text = "internal"
        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = _requests.HTTPError(response=err_resp)

        # First call succeeds, the rest fail.
        responses = [ok_resp, bad_resp, bad_resp, bad_resp]
        post = MagicMock(side_effect=responses)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--max-per-strategy", "2"],
        )
        rc = script.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "queued bt=42" in out
        assert "FAILED" in out


class TestAddYears:
    """Date helper edge case: Feb 29 + N years where N moves to non-leap year."""

    def test_normal_add(self, script):
        assert script._add_years(date(2020, 1, 1), 4) == date(2024, 1, 1)

    def test_leap_to_non_leap_caps_at_28(self, script):
        assert script._add_years(date(2020, 2, 29), 1) == date(2021, 2, 28)
