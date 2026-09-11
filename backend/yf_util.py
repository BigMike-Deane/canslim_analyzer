"""yfinance batch download without the thread leak.

yf.download(threads=True) -- its default -- runs each ticker on a thread from
the `multitasking` package, which appends every Thread it starts to a
module-level registry (multitasking.config["TASKS"]) and never removes it.
In a long-lived process that list only grows: LEAK_DIAG found ~18k dead
Thread objects (~2.2 KB each, with their Event/Condition/locks) after 18h of
scans, correlation checks and shadow arms (Sep-11 2026).

The registry is bookkeeping only (monitoring + wait_for_tasks at exit); the
thread-count limit lives in multitasking's pool semaphore, and yfinance never
reads TASKS. Dropping finished threads from it is safe.
"""


def download(*args, **kwargs):
    """yf.download, then forget the finished threads it left behind."""
    import yfinance as yf
    try:
        return yf.download(*args, **kwargs)
    finally:
        prune_finished_tasks()


def prune_finished_tasks() -> int:
    """Remove finished threads from multitasking's registry; returns how
    many. Removes in place, one at a time, so a thread another download
    appends meanwhile is never lost."""
    try:
        import multitasking
    except ImportError:
        return 0
    tasks = multitasking.config.get("TASKS")
    if not tasks:
        return 0
    done = [t for t in list(tasks) if t is not None and not t.is_alive()]
    for t in done:
        try:
            tasks.remove(t)
        except ValueError:
            pass    # another caller pruned it first
    return len(done)
