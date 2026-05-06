"""Smoke tests for scripts/grow_training_pool.py.

The script is a thin CLI over POST /api/backtests. We verify:
  - dry-run mode prints the plan and makes NO network calls
  - --execute path posts to the right URL with the right payload shape
  - sweep is non-empty and well-formed (catches accidental list edits)
"""

import importlib.util
import sys
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


class TestSweep:
    def test_sweep_non_empty(self, script):
        assert len(script.SWEEP) >= 5, "sweep shouldn't be silently emptied"

    def test_sweep_entries_well_formed(self, script):
        for entry in script.SWEEP:
            assert set(entry.keys()) >= {"start_date", "end_date", "strategy"}
            assert entry["start_date"] < entry["end_date"]

    def test_sweep_covers_both_live_strategies(self, script):
        strategies = {e["strategy"] for e in script.SWEEP}
        assert "nostate_optimized" in strategies
        assert "nostate_cs_bear" in strategies


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
        }


class TestDryRun:
    def test_dry_run_makes_no_network_calls(self, script, capsys, monkeypatch):
        # If the dry-run path ever issues a real request, requests.post should
        # blow up loudly here rather than silently hit the production API.
        def _explode(*a, **kw):
            raise AssertionError("dry-run must not call requests.post")
        monkeypatch.setattr(script.requests, "post", _explode)
        monkeypatch.setattr(sys, "argv", ["grow_training_pool.py"])
        rc = script.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Planned sweep" in out
        assert "--execute" in out

    def test_execute_posts_to_api(self, script, capsys, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 999}
        mock_resp.raise_for_status.return_value = None
        post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--api-base", "http://test.local"],
        )
        rc = script.main()
        assert rc == 0
        assert post.call_count == len(script.SWEEP)
        # Verify the URL and payload shape on the first call
        first_call = post.call_args_list[0]
        assert first_call.args[0] == "http://test.local/api/backtests"
        payload = first_call.kwargs["json"]
        assert "start_date" in payload and "strategy" in payload

    def test_execute_with_token_sets_auth_header(self, script, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 1}
        mock_resp.raise_for_status.return_value = None
        post = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--token", "secret123"],
        )
        script.main()
        headers = post.call_args_list[0].kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret123"
