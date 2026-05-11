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


class TestExecuteFailureHandling:
    """Cover the failure branches at lines 169-185: HTTPError, generic
    RequestException, and the post-loop "Failures (re-run later):"
    report block (which only fires when failed[] is non-empty)."""

    def test_http_error_collects_and_returns_rc_1(
        self, script, capsys, monkeypatch
    ):
        """A 4xx/5xx response from /api/backtests should be caught,
        labeled with the HTTP status code + truncated body, and
        contribute to the rc=1 exit."""
        import requests as _requests

        err_resp = MagicMock()
        err_resp.status_code = 503
        err_resp.text = "Service Unavailable" * 30  # >200 chars on purpose
        http_err = _requests.HTTPError(response=err_resp)

        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = http_err

        post = MagicMock(return_value=bad_resp)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute", "--api-base", "http://test.local"],
        )

        rc = script.main()
        out = capsys.readouterr().out

        # Every entry in SWEEP raises -> every entry is a failure
        assert rc == 1
        assert post.call_count == len(script.SWEEP)
        assert "FAILED" in out
        # Failure-report block fired
        assert "Failures (re-run later):" in out
        # HTTP status code + truncated body landed in the failure line
        assert "HTTP 503:" in out or "503" in out

    def test_request_exception_collects_and_returns_rc_1(
        self, script, capsys, monkeypatch
    ):
        """Non-HTTP transport errors (ConnectionError, Timeout, etc.)
        hit the generic RequestException branch at line 172."""
        import requests as _requests

        def _boom(*a, **kw):
            raise _requests.ConnectionError("connection refused")

        monkeypatch.setattr(script.requests, "post", _boom)
        monkeypatch.setattr(
            sys, "argv",
            ["grow_training_pool.py", "--execute"],
        )

        rc = script.main()
        out = capsys.readouterr().out

        assert rc == 1
        assert "FAILED" in out
        assert "connection refused" in out
        assert "Failures (re-run later):" in out

    def test_partial_failure_still_returns_rc_1(
        self, script, capsys, monkeypatch
    ):
        """Mixed success+failure: even one failure returns rc=1 so the
        operator notices and can re-run the failing label."""
        import requests as _requests

        ok_resp = MagicMock()
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {"id": 42}

        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.text = "internal"
        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = _requests.HTTPError(response=err_resp)

        # First call succeeds, all subsequent fail
        responses = [ok_resp] + [bad_resp] * (len(script.SWEEP) - 1)
        post = MagicMock(side_effect=responses)
        monkeypatch.setattr(script.requests, "post", post)
        monkeypatch.setattr(
            sys, "argv", ["grow_training_pool.py", "--execute"],
        )

        rc = script.main()
        out = capsys.readouterr().out

        assert rc == 1
        # Should have logged at least one queued bt + at least one FAILED
        assert "queued bt=42" in out
        assert "FAILED" in out
