"""Tests for backend/backtest_queue.py — closes the 33% baseline (77/115 lines
uncovered as of 9a10e18).

The single-worker queue is load-bearing infra for the Approach 2 A/B eval
window (closes 2026-06-18): every backtest, including any catch-up runs to
calibrate the verdict, flows through `_worker_loop`. The `f17960c` defensive
JSON parse for `profile_overrides` (lines 135-141) — a production fix for a
TEXT-vs-JSON column mismatch on Postgres — currently has zero direct tests;
this file locks the dict / valid-JSON / malformed-JSON branches plus the
cancellation, exception-recovery, and stop paths.

Eval-safe: no scoring, no trading, no real DB writes, no real backtest run.
SessionLocal and run_backtest are patched at the SOURCE module
(backend.database / backend.backtester) because `_worker_loop` imports them
locally inside the function body — patching `backend.backtest_queue.X`
would not intercept the lookup.
"""

import os
import queue as queue_mod
from datetime import datetime
from unittest.mock import patch, MagicMock

os.environ.setdefault("REQUIRE_AUTH", "false")
os.environ.setdefault("CANSLIM_ENV", "development")
os.environ.setdefault("DISABLE_SCHEDULER", "true")

from backend.backtest_queue import BacktestQueueManager  # noqa: E402
from backend.backtest_queue import backtest_queue as _ORIGINAL_SINGLETON  # noqa: E402


class _SingletonResetMixin:
    """Reset the module singleton around each test, then restore the
    original instance on teardown so other test files (e.g. test_main_coverage
    which patches `backend.backtest_queue.backtest_queue.enqueue`) don't see
    drift between `BacktestQueueManager._instance` and the module-level
    `backtest_queue` reference."""

    def setup_method(self):
        BacktestQueueManager._instance = None

    def teardown_method(self):
        BacktestQueueManager._instance = _ORIGINAL_SINGLETON


def _bt(**fields):
    """Build a fake BacktestRun mock with sensible defaults plus overrides.

    Defaults match a freshly enqueued, not-yet-cancelled, not-yet-running
    row so the worker pre-check falls through to the run stage."""
    bt = MagicMock()
    bt.cancel_requested = False
    bt.status = "pending"
    bt.profile_overrides = None
    for k, v in fields.items():
        setattr(bt, k, v)
    return bt


def _one_iteration_then_stop(mgr):
    """Patch _stop_event so the worker loop runs exactly one iteration.

    is_set sequence: False (enter loop body once) → True (exit on next check)."""
    ev = MagicMock()
    ev.is_set.side_effect = [False, True]
    return patch.object(mgr, "_stop_event", ev)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
class TestSingleton(_SingletonResetMixin):
    def test_two_instantiations_return_same_object(self):
        a = BacktestQueueManager()
        b = BacktestQueueManager()
        assert a is b

    def test_init_short_circuits_on_second_call(self):
        """Once _initialized is True, __init__ must NOT replace _queue/_stop_event,
        otherwise queued items would be silently lost on every subsequent
        `BacktestQueueManager()` call. Exercises the line-31 guard."""
        a = BacktestQueueManager()
        a._queue.put(99)
        b = BacktestQueueManager()  # __init__ would reset _queue if not guarded
        assert b._queue.qsize() == 1
        assert list(b._queue.queue) == [99]


# ---------------------------------------------------------------------------
# enqueue + get_queue_position
# ---------------------------------------------------------------------------
class TestEnqueueAndPosition(_SingletonResetMixin):
    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    def test_enqueue_appends_and_returns_position_zero_for_first_item(self):
        pos = self.mgr.enqueue(42)
        assert pos == 0
        assert list(self.mgr._queue.queue) == [42]

    def test_second_enqueue_returns_position_one(self):
        self.mgr.enqueue(42)
        pos = self.mgr.enqueue(43)
        assert pos == 1

    def test_get_queue_position_returns_none_when_running(self):
        """Currently-running backtest is no longer in `_queue` — line 71-72
        explicitly returns None so the API doesn't surface position 0 for
        something already executing."""
        self.mgr.enqueue(42)
        self.mgr._current_backtest_id = 42
        assert self.mgr.get_queue_position(42) is None

    def test_get_queue_position_returns_none_when_absent(self):
        self.mgr.enqueue(42)
        assert self.mgr.get_queue_position(999) is None


# ---------------------------------------------------------------------------
# get_queue_snapshot
# ---------------------------------------------------------------------------
class TestQueueSnapshot(_SingletonResetMixin):
    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    def test_empty_queue(self):
        snap = self.mgr.get_queue_snapshot()
        assert snap == {"running": None, "queued": [], "queue_length": 0}

    def test_filters_out_none_poison_pills(self):
        """stop() puts None on the queue to unblock get(timeout=2). The snapshot
        must not surface that to /api/backtests/queue."""
        self.mgr._queue.put(10)
        self.mgr._queue.put(None)
        self.mgr._queue.put(20)
        snap = self.mgr.get_queue_snapshot()
        assert snap == {"running": None, "queued": [10, 20], "queue_length": 2}

    def test_running_id_surfaces(self):
        self.mgr._current_backtest_id = 7
        self.mgr._queue.put(8)
        snap = self.mgr.get_queue_snapshot()
        assert snap["running"] == 7
        assert snap["queued"] == [8]


# ---------------------------------------------------------------------------
# start / stop (no real threads)
# ---------------------------------------------------------------------------
class TestStartStop(_SingletonResetMixin):
    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    @patch("backend.backtest_queue.threading.Thread")
    def test_start_creates_and_starts_daemon_thread(self, mock_thread_cls):
        thread = MagicMock()
        mock_thread_cls.return_value = thread
        self.mgr.start()
        mock_thread_cls.assert_called_once()
        # Thread must be daemon=True so the worker dies with the process —
        # otherwise `docker stop` would hang on a stuck queue.get.
        kwargs = mock_thread_cls.call_args.kwargs
        assert kwargs.get("daemon") is True
        thread.start.assert_called_once()
        assert self.mgr._worker_thread is thread

    @patch("backend.backtest_queue.threading.Thread")
    def test_start_is_idempotent_when_worker_alive(self, mock_thread_cls):
        """A second start() while the worker is alive must NOT spawn a duplicate
        consumer (line 41-43). Two consumers on the same queue would race on
        BacktestRun status writes."""
        alive = MagicMock()
        alive.is_alive.return_value = True
        self.mgr._worker_thread = alive
        self.mgr.start()
        mock_thread_cls.assert_not_called()

    def test_stop_sets_event_and_puts_poison_pill_with_no_worker(self):
        # Worker thread was never started; stop must still flag the event +
        # push the None pill without trying to .join() something nonexistent.
        self.mgr.stop()
        assert self.mgr._stop_event.is_set() is True
        assert None in list(self.mgr._queue.queue)

    def test_stop_joins_alive_worker(self):
        alive = MagicMock()
        alive.is_alive.return_value = True
        self.mgr._worker_thread = alive
        self.mgr.stop()
        alive.join.assert_called_once_with(timeout=5)

    def test_stop_does_not_join_dead_worker(self):
        dead = MagicMock()
        dead.is_alive.return_value = False
        self.mgr._worker_thread = dead
        self.mgr.stop()
        dead.join.assert_not_called()


# ---------------------------------------------------------------------------
# Worker loop — happy path
# ---------------------------------------------------------------------------
class TestWorkerLoopHappyPath(_SingletonResetMixin):
    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    def test_runs_backtest_when_pending_with_no_overrides(self):
        bt_pre = _bt(status="pending", cancel_requested=False)
        bt_run = _bt(status="pending", profile_overrides=None)
        db_pre = MagicMock(); db_pre.get.return_value = bt_pre
        db_run = MagicMock(); db_run.get.return_value = bt_run
        self.mgr._queue.put(42)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal", side_effect=[db_pre, db_run]), \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        mock_run.assert_called_once_with(db_run, 42, profile_overrides=None)
        # _current_backtest_id must reset after each run (line 165) so the
        # next /api/backtests/queue poll doesn't lie about what's running.
        assert self.mgr._current_backtest_id is None
        # Both sessions closed (pre-check finally + outer finally).
        db_pre.close.assert_called_once()
        db_run.close.assert_called_once()

    def test_empty_queue_continues_without_error(self):
        """queue.get(timeout=2) raising Empty must NOT crash the worker — it's
        the heartbeat that lets _stop_event take effect on idle (line 99-101)."""
        ev = MagicMock()
        # Three loop checks: enter (Empty), enter (poison pill), exit.
        ev.is_set.side_effect = [False, False, True]
        get_mock = MagicMock(side_effect=[queue_mod.Empty, None])
        with patch.object(self.mgr, "_stop_event", ev), \
             patch.object(self.mgr._queue, "get", get_mock), \
             patch("backend.database.SessionLocal") as sl, \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        sl.assert_not_called()
        mock_run.assert_not_called()

    def test_poison_pill_skipped(self):
        """None pushed by stop() must hit the `continue` at line 105 and never
        try to look up BacktestRun(None)."""
        self.mgr._queue.put(None)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal") as sl, \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        sl.assert_not_called()
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Worker loop — cancellation paths
# ---------------------------------------------------------------------------
class TestWorkerLoopCancellation(_SingletonResetMixin):
    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    def test_cancel_pending_marks_cancelled_and_skips_run(self):
        bt = _bt(status="pending", cancel_requested=True)
        db = MagicMock()
        db.get.return_value = bt
        self.mgr._queue.put(42)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal", return_value=db), \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        assert bt.status == "cancelled"
        assert bt.error_message == "Cancelled before starting"
        assert isinstance(bt.completed_at, datetime)
        db.commit.assert_called_once()
        mock_run.assert_not_called()

    def test_cancel_already_running_does_not_overwrite_status(self):
        """If the cancel flag flipped AFTER the backtester moved bt.status to
        'running', we must not regress it back to 'cancelled' — the
        line-113 guard `if bt.status == "pending"` is what protects mid-flight
        progress writes."""
        bt = _bt(status="running", cancel_requested=True)
        db = MagicMock()
        db.get.return_value = bt
        self.mgr._queue.put(42)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal", return_value=db), \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        assert bt.status == "running"  # untouched
        db.commit.assert_not_called()
        mock_run.assert_not_called()

    def test_status_completed_skips_without_running(self):
        """Re-enqueued completed runs (e.g. retry-after-failure paths that
        landed mid-flight) must not be re-executed — the line-119 guard
        gates this."""
        bt = _bt(status="completed", cancel_requested=False)
        db = MagicMock()
        db.get.return_value = bt
        self.mgr._queue.put(42)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal", return_value=db), \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        mock_run.assert_not_called()
        db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Worker loop — profile_overrides parsing (the f17960c defensive fix)
# ---------------------------------------------------------------------------
class TestProfileOverridesParsing(_SingletonResetMixin):
    """The f17960c defensive parse exists because production Postgres has
    `profile_overrides` as TEXT (added before the ORM was migrated to JSON).
    Existing rows come back as raw strings; new rows arrive as dicts. All
    three branches must reach `run_backtest` with the right shape."""

    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    def _drive(self, overrides):
        bt_pre = _bt(status="pending")
        bt_run = _bt(status="pending", profile_overrides=overrides)
        db_pre = MagicMock(); db_pre.get.return_value = bt_pre
        db_run = MagicMock(); db_run.get.return_value = bt_run
        self.mgr._queue.put(42)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal", side_effect=[db_pre, db_run]), \
             patch("backend.backtester.run_backtest") as mock_run:
            self.mgr._worker_loop()
        return mock_run

    def test_dict_passes_through_unchanged(self):
        mock_run = self._drive({"min_score": 80})
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["profile_overrides"] == {"min_score": 80}

    def test_valid_json_string_is_parsed(self):
        """The exact f17960c fix: TEXT column returned a str, not a dict.
        Without `json.loads`, run_backtest would receive a string and crash
        at param-merge time."""
        mock_run = self._drive('{"min_score": 80, "max_positions": 6}')
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["profile_overrides"] == {
            "min_score": 80, "max_positions": 6
        }

    def test_malformed_json_string_falls_back_to_none(self):
        """Bad JSON must NOT propagate to run_backtest as a malformed string —
        we degrade to None and log a warning (line 138-141)."""
        mock_run = self._drive("{not json}")
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["profile_overrides"] is None

    def test_empty_string_short_circuits_to_none(self):
        """Line 131: `if bt_run and bt_run.profile_overrides else None`. Empty
        string is falsy → overrides=None without entering the parse branch.
        Defends against legacy rows that wrote '' instead of NULL."""
        mock_run = self._drive("")
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["profile_overrides"] is None


# ---------------------------------------------------------------------------
# Worker loop — exception path
# ---------------------------------------------------------------------------
class TestWorkerLoopExceptionPath(_SingletonResetMixin):
    def setup_method(self):
        super().setup_method()
        self.mgr = BacktestQueueManager()

    def _stage(self, run_exc, err_status="pending", err_commit_exc=None):
        """Common harness for exception-path tests. Returns (mock_run, bt_err,
        db_run, error_db) so each test can assert on whichever surface it
        cares about."""
        bt_pre = _bt(status="pending")
        bt_run = _bt(status="pending")
        bt_err = _bt(status=err_status)
        db_pre = MagicMock(); db_pre.get.return_value = bt_pre
        db_run = MagicMock(); db_run.get.return_value = bt_run
        error_db = MagicMock(); error_db.get.return_value = bt_err
        if err_commit_exc is not None:
            error_db.commit.side_effect = err_commit_exc
        self.mgr._queue.put(42)
        with _one_iteration_then_stop(self.mgr), \
             patch("backend.database.SessionLocal", side_effect=[db_pre, db_run, error_db]), \
             patch("backend.backtester.run_backtest", side_effect=run_exc) as mock_run:
            self.mgr._worker_loop()
        return mock_run, bt_err, db_run, error_db

    def test_run_backtest_raises_marks_failed_via_fresh_session(self):
        """The error-recovery path opens its OWN session (line 149) because the
        original may be in an aborted-transaction state. Asserts both that the
        original db.close() fired AND that the fresh error_db saw the commit."""
        mock_run, bt_err, db_run, error_db = self._stage(RuntimeError("boom"))
        assert bt_err.status == "failed"
        assert bt_err.error_message == "boom"
        error_db.commit.assert_called_once()
        # Original run-stage db must close (line 148) BEFORE we open error_db.
        db_run.close.assert_called_once()
        error_db.close.assert_called_once()
        # _current_backtest_id reset even on failure (line 165).
        assert self.mgr._current_backtest_id is None

    def test_long_error_message_truncated_to_500_chars(self):
        """Without the [:500] slice, a verbose traceback could blow past
        Postgres column limits and re-raise during the recovery write."""
        long_msg = "x" * 600
        _, bt_err, _, _ = self._stage(RuntimeError(long_msg))
        assert bt_err.error_message == "x" * 500
        assert len(bt_err.error_message) == 500

    def test_exception_does_not_overwrite_already_cancelled_status(self):
        """If a user cancelled mid-run, an exception during run_backtest must
        NOT regress the status to 'failed' (line 152 if-guard)."""
        _, bt_err, _, error_db = self._stage(RuntimeError("boom"), err_status="cancelled")
        assert bt_err.status == "cancelled"
        error_db.commit.assert_not_called()

    def test_exception_does_not_overwrite_already_completed_status(self):
        _, bt_err, _, error_db = self._stage(RuntimeError("boom"), err_status="completed")
        assert bt_err.status == "completed"
        error_db.commit.assert_not_called()

    def test_error_db_commit_failure_is_swallowed(self):
        """If even the FRESH error_db can't commit (e.g. DB fully down), the
        bare `except Exception` at line 157-158 must absorb it — the worker
        thread must not die. Also verifies error_db.close() still fires via
        the inner finally."""
        # If swallowing fails, _stage()'s context manager would re-raise.
        _, _, _, error_db = self._stage(
            RuntimeError("boom"),
            err_commit_exc=RuntimeError("db down"),
        )
        error_db.close.assert_called_once()
