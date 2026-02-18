"""
Backtest Queue Manager

Sequential backtest execution via a single-worker queue.
Ensures cache warming: backtest #1 populates the score cache,
then backtests #2-N run 15-37x faster on cached scores.
"""

import threading
import queue
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class BacktestQueueManager:
    """Singleton queue manager for sequential backtest execution."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._current_backtest_id = None
        self._initialized = True

    def start(self):
        """Start the queue worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Backtest queue worker already running")
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="backtest-queue-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Backtest queue worker started")

    def stop(self):
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        # Unblock the worker if it's waiting on queue.get()
        self._queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        logger.info("Backtest queue worker stopped")

    def enqueue(self, backtest_id: int):
        """Add a backtest to the queue. Returns queue position (0 = next up)."""
        self._queue.put(backtest_id)
        pos = self.get_queue_position(backtest_id)
        logger.info(f"Backtest {backtest_id} enqueued (position {pos})")
        return pos

    def get_queue_position(self, backtest_id: int) -> int | None:
        """Get queue position for a backtest. None = not queued (running or done)."""
        if self._current_backtest_id == backtest_id:
            return None  # Currently running
        # Snapshot the queue items (non-destructive)
        with self._queue.mutex:
            queued = list(self._queue.queue)
        try:
            return queued.index(backtest_id)
        except ValueError:
            return None  # Not in queue

    def get_queue_snapshot(self) -> dict:
        """Return current queue state for the /api/backtests/queue endpoint."""
        with self._queue.mutex:
            queued_ids = [x for x in self._queue.queue if x is not None]
        return {
            "running": self._current_backtest_id,
            "queued": queued_ids,
            "queue_length": len(queued_ids),
        }

    def _worker_loop(self):
        """Main worker loop: pull backtest IDs and run them sequentially."""
        from backend.database import SessionLocal, BacktestRun
        from backend.backtester import run_backtest

        logger.info("Backtest queue worker loop started")
        while not self._stop_event.is_set():
            try:
                backtest_id = self._queue.get(timeout=2)
            except queue.Empty:
                continue

            # None is the poison pill to unblock during shutdown
            if backtest_id is None:
                continue

            # Check if cancelled before starting
            db = SessionLocal()
            try:
                bt = db.query(BacktestRun).get(backtest_id)
                if bt and bt.cancel_requested:
                    logger.info(f"Backtest {backtest_id} was cancelled before starting, skipping")
                    if bt.status == "pending":
                        bt.status = "cancelled"
                        bt.completed_at = datetime.now(timezone.utc)
                        bt.error_message = "Cancelled before starting"
                        db.commit()
                    continue
                if bt and bt.status not in ("pending", "running"):
                    logger.info(f"Backtest {backtest_id} status is {bt.status}, skipping")
                    continue
            finally:
                db.close()

            # Run the backtest
            self._current_backtest_id = backtest_id
            logger.info(f"Backtest {backtest_id} starting (queue remaining: {self._queue.qsize()})")
            db = SessionLocal()
            try:
                run_backtest(db, backtest_id)
                logger.info(f"Backtest {backtest_id} completed successfully")
            except Exception as e:
                logger.error(f"Backtest {backtest_id} failed: {e}")
                try:
                    bt = db.query(BacktestRun).get(backtest_id)
                    if bt and bt.status not in ("cancelled", "completed"):
                        bt.status = "failed"
                        bt.error_message = str(e)[:500]
                        db.commit()
                except Exception:
                    logger.exception(f"Failed to update status for backtest {backtest_id}")
            finally:
                db.close()
                self._current_backtest_id = None

        logger.info("Backtest queue worker loop exited")


# Module-level singleton
backtest_queue = BacktestQueueManager()
