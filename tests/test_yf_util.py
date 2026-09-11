"""backend.yf_util: yf.download must not leave dead threads in multitasking's
registry (Sep-11 2026 LEAK_DIAG: ~18k retained Thread objects in 18h)."""

import threading

import multitasking
import pytest

from backend import yf_util


@pytest.fixture
def registry(monkeypatch):
    tasks = []
    monkeypatch.setitem(multitasking.config, "TASKS", tasks)
    return tasks


def _finished_thread():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def test_real_multitasking_tasks_are_pruned_once_finished(registry):
    # The same decorator yfinance's multi.py uses for each ticker.
    @multitasking.task
    def fetch():
        pass

    for _ in range(5):
        fetch()
    multitasking.wait_for_tasks()
    assert len(registry) == 5            # the leak: all five kept forever
    assert yf_util.prune_finished_tasks() == 5
    assert registry == []


def test_live_threads_stay(registry):
    release = threading.Event()
    live = threading.Thread(target=release.wait)
    live.start()
    try:
        registry.extend([_finished_thread(), live, None])
        assert yf_util.prune_finished_tasks() == 1
        assert registry == [live, None]
    finally:
        release.set()
        live.join()


def test_download_prunes_even_when_yfinance_raises(registry, monkeypatch):
    import yfinance

    def boom(*a, **k):
        registry.append(_finished_thread())
        raise RuntimeError("rate limited")
    monkeypatch.setattr(yfinance, "download", boom)
    with pytest.raises(RuntimeError):
        yf_util.download(["AAA"], period="5d")
    assert registry == []


def test_download_passes_through(registry, monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: ("df", a, k))
    assert yf_util.download(["AAA"], period="5d", threads=True) == \
        ("df", (["AAA"],), {"period": "5d", "threads": True})
