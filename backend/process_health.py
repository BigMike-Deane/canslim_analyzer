"""Process-level health: resident memory per scan, growth trend, and
unexpected-restart detection.

Why this exists (2026-09-08): the kernel OOM-killed uvicorn at 09:23 UTC
after 91 h alive. RSS had climbed from ~0.8 GB to 2.87 GB on a 3.8 GB VPS
with no swap and no container memory limit. Docker's ``restart:
unless-stopped`` hid the event completely: no alert, no log line, nothing
on the System Health page. Frequent deploys (never more than ~4 days
apart) had masked the growth all along.

Three jobs, all best-effort -- nothing here may raise into the scan path:

1. ``record_scan_sample`` -- after each scan: ``gc.collect()``, read
   VmRSS/VmHWM from /proc, keep a bounded history (persisted to Redis so
   the trend survives restarts), log one line, and fire a one-shot ops
   alert when RSS crosses ``RSS_ALERT_MB``.
2. ``check_restart`` -- at startup, classify this start as ``first``,
   ``deploy``, ``clean`` or ``unclean`` from two signals the previous
   process left behind: a clean-shutdown flag written in the lifespan
   teardown, and the Docker build stamp. ``unclean`` (no flag, same build)
   means crash or OOM kill -> ops alert with the last RSS seen.
3. ``get_process_health`` -- snapshot for ``/api/system-health``.

Leaf module: imports nothing from scheduler / ai_trader / main at import
time (email_utils and redis_cache are imported lazily inside functions).
"""

import gc
import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# One-shot alert threshold. 1800 MB is ~2/3 of the 2560 MB container limit
# (docker-compose mem_limit) -- at the ~23 MB/h slope seen before the OOM
# that leaves ~1.4 days of warning. Env override for experiments.
RSS_ALERT_MB = int(os.environ.get("RSS_ALERT_MB", "1800"))

# 16 scans/day -> ~25 days of history. Each sample is ~100 bytes.
MAX_SAMPLES = 400

REDIS_SAMPLES_KEY = "canslim:process_memory"
REDIS_START_KEY = "canslim:process_start"
REDIS_CLEAN_KEY = "canslim:process_clean_shutdown"

# ── Leak-hunt diagnostics (2026-09-10) ─────────────────────────────────
# MALLOC_ARENA_MAX=2 cut RSS growth from ~23.6 to ~13 MB/h -- so part of
# the growth was glibc fragmentation, and a residual remains. LEAK_DIAG=1
# adds three cheap probes to every post-scan sample, which together split
# the residual into its possible causes:
#
#   py_blocks      sys.getallocatedblocks() -- live Python allocations, O(1).
#                  Climbing with RSS => Python objects are accumulating.
#   trim_freed_mb  RSS released by glibc malloc_trim(0) -- free-but-held
#                  heap. Large, and post-trim RSS flat => fragmentation, and
#                  the trim itself is the fix.
#   census growth  live gc-tracked objects by type vs a post-warm-up
#                  baseline -- names WHAT is accumulating if py_blocks says
#                  something is.
#
# ⚑ tracemalloc was measured and REJECTED for this box: on 1M small objects
# it doubled traced memory (+280 MB of bookkeeping), slowed allocation ~8x,
# and one snapshot peaked ~950 MB above baseline -- enough to push a
# ~850 MB process through the 2560m container limit and OOM-kill the trader
# it was meant to protect. Note the census cannot see dicts holding only
# atomic values (CPython untracks them); py_blocks still counts those.
LEAK_DIAG = os.environ.get("LEAK_DIAG", "0").strip().lower() in ("1", "true", "yes")
LEAK_DIAG_WARMUP_H = float(os.environ.get("LEAK_DIAG_WARMUP_H", "3"))
CENSUS_TOP = 10

_lock = threading.Lock()
_samples: deque = deque(maxlen=MAX_SAMPLES)
_state = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "start_kind": None,      # first | deploy | clean | unclean
    "previous": None,        # previous process start record (dict) or None
    "alerted_mb": None,      # RSS at which the one-shot alert fired
    "limit_mb": None,        # cgroup memory limit, resolved lazily
    "census_baseline": None,     # {type: count} taken after warm-up
    "census_baseline_at": None,
    "census_growth": None,       # latest census_growth() result
}
_libc = {"handle": None, "tried": False}


# ---------------------------------------------------------------- readers

def read_memory() -> Optional[dict]:
    """Current RSS / high-water mark in MB from /proc/self/status. Returns
    None where /proc is unavailable (macOS dev boxes) -- callers skip."""
    try:
        rss = hwm = None
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) // 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) // 1024
        if rss is None:
            return None
        return {"rss_mb": rss, "hwm_mb": hwm if hwm is not None else rss}
    except OSError:
        return None


def container_limit_mb() -> Optional[int]:
    """cgroup memory limit in MB (v2 then v1 path); None when unlimited or
    unreadable. Cached after the first successful read."""
    if _state["limit_mb"] is not None:
        return _state["limit_mb"] or None
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw == "max":
                break
            val = int(raw)
            if val <= 0 or val > (1 << 50):      # v1 reports ~2^63 for "no limit"
                break
            _state["limit_mb"] = val // (1024 * 1024)
            return _state["limit_mb"]
        except (OSError, ValueError):
            continue
    _state["limit_mb"] = 0   # sentinel: checked, unlimited
    return None


def _redis():
    try:
        from redis_cache import get_redis_client
        return get_redis_client()
    except Exception:
        return None


def _parse_ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


# ---------------------------------------------------------------- pure logic

def growth_rate_mb_per_hour(samples, proc: str, window_hours: float = 24.0) -> Optional[float]:
    """Least-squares RSS slope (MB/h) over the newest ``window_hours`` of
    samples belonging to process ``proc``. None with fewer than 3 points or
    under 1 h of span -- a slope from two adjacent scans is noise."""
    pts = [s for s in samples if s.get("proc") == proc and s.get("rss_mb") is not None]
    if len(pts) < 3:
        return None
    newest = _parse_ts(pts[-1]["ts"])
    xs, ys = [], []
    for s in pts:
        t = _parse_ts(s["ts"])
        age_h = (newest - t).total_seconds() / 3600.0
        if age_h <= window_hours:
            xs.append(-age_h)
            ys.append(float(s["rss_mb"]))
    if len(xs) < 3 or (max(xs) - min(xs)) < 1.0:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return round(sxy / sxx, 1)


def classify_start(previous: Optional[dict], clean_flag: bool, build: str) -> str:
    """How did this process come to start?

    first   -- no record of a previous process (fresh install / Redis wiped)
    clean   -- previous process ran its lifespan teardown (docker stop,
               compose down, deploy) -- flag present regardless of build
    deploy  -- no flag, but the build stamp changed: the old container was
               replaced without a graceful stop (compose sometimes SIGKILLs
               after its 10 s grace when a scan thread is mid-fetch)
    unclean -- no flag and the SAME build came back: crash or OOM kill
    """
    if not previous:
        return "first"
    if clean_flag:
        return "clean"
    if previous.get("build") != build:
        return "deploy"
    return "unclean"


def census_growth(baseline: dict, current: dict, top: int = CENSUS_TOP) -> list:
    """Types whose live object count grew most since ``baseline``, largest
    first. Shrinking or unchanged types are omitted -- a leak only grows."""
    grown = []
    for name, count in current.items():
        delta = count - baseline.get(name, 0)
        if delta > 0:
            grown.append({"type": name, "delta": delta, "count": count})
    grown.sort(key=lambda g: -g["delta"])
    return grown[:top]


# ---------------------------------------------------------------- leak probes

def type_census() -> dict:
    """Live gc-tracked objects by fully-qualified type name. ~1 s on a few
    million objects; the transient list is pointers only (no per-object
    bookkeeping, unlike tracemalloc)."""
    counts: dict = {}
    for obj in gc.get_objects():
        t = type(obj)
        name = f"{t.__module__}.{t.__qualname__}"
        counts[name] = counts.get(name, 0) + 1
    return counts


def malloc_trim() -> Optional[dict]:
    """Ask glibc to return free heap pages to the OS; report RSS before and
    after. None off-glibc (musl, macOS) or where /proc is unavailable."""
    if not _libc["tried"]:
        _libc["tried"] = True
        try:
            import ctypes
            _libc["handle"] = ctypes.CDLL("libc.so.6")
            _libc["handle"].malloc_trim  # AttributeError on non-glibc libc
        except (OSError, AttributeError):
            _libc["handle"] = None
    if _libc["handle"] is None:
        return None
    before = read_memory()
    if before is None:
        return None
    _libc["handle"].malloc_trim(0)
    after = read_memory() or before
    return {"rss_pre_trim_mb": before["rss_mb"],
            "trim_freed_mb": max(before["rss_mb"] - after["rss_mb"], 0)}


def _leak_diag_step(uptime_h: float) -> dict:
    """Run the LEAK_DIAG probes. Best-effort: any failure yields a partial
    dict, never an exception into the scan path."""
    out: dict = {}
    try:
        import sys
        out["py_blocks"] = sys.getallocatedblocks()
    except Exception:
        pass
    try:
        census = type_census()
        if _state["census_baseline"] is None and uptime_h >= LEAK_DIAG_WARMUP_H:
            _state["census_baseline"] = census
            _state["census_baseline_at"] = datetime.now(timezone.utc).isoformat()
        elif _state["census_baseline"] is not None:
            _state["census_growth"] = census_growth(_state["census_baseline"], census)
        out["gc_objects"] = sum(census.values())
    except Exception as e:
        logger.debug(f"process_health: census failed: {e}")
    try:
        trim = malloc_trim()
        if trim:
            out.update(trim)
    except Exception as e:
        logger.debug(f"process_health: malloc_trim failed: {e}")
    return out


# ---------------------------------------------------------------- persistence

def _persist_samples():
    client = _redis()
    if not client:
        return
    try:
        with _lock:
            payload = list(_samples)
        client.set(REDIS_SAMPLES_KEY, json.dumps(payload))
    except Exception as e:
        logger.debug(f"process_health: could not persist samples: {e}")


def restore_history():
    """Load the sample history the previous process left in Redis."""
    client = _redis()
    if not client:
        return
    try:
        raw = client.get(REDIS_SAMPLES_KEY)
        if raw:
            data = json.loads(raw)
            with _lock:
                _samples.clear()
                _samples.extend(data[-MAX_SAMPLES:])
    except Exception as e:
        logger.debug(f"process_health: could not restore samples: {e}")


def mark_clean_shutdown():
    """Called first thing in the lifespan teardown. Absence of this flag at
    the next startup is the crash signal."""
    client = _redis()
    if not client:
        return
    try:
        client.set(REDIS_CLEAN_KEY, "1", ex=7 * 86400)
    except Exception as e:
        logger.debug(f"process_health: could not mark clean shutdown: {e}")


# ---------------------------------------------------------------- entry points

def _send_alert(title: str, message: str, data: dict = None):
    try:
        from backend.email_utils import send_ops_alert
        send_ops_alert(title, message, priority="urgent",
                       tags=["process_health"], data=data)
    except Exception as e:
        logger.warning(f"process_health: ops alert failed: {e}")


def check_restart(build: Optional[str] = None) -> str:
    """Startup hook: classify this start, alert on an unclean one, and leave
    this process's record for the next start to compare against."""
    now_iso = datetime.now(timezone.utc).isoformat()
    _state["started_at"] = now_iso
    if build is None:
        try:
            from backend.build_info import get_build_version
            build = get_build_version()
        except Exception:
            build = "unknown"

    previous, clean_flag = None, False
    client = _redis()
    if client:
        try:
            raw = client.get(REDIS_START_KEY)
            previous = json.loads(raw) if raw else None
            clean_flag = bool(client.get(REDIS_CLEAN_KEY))
            client.delete(REDIS_CLEAN_KEY)
        except Exception as e:
            logger.debug(f"process_health: could not read start record: {e}")

    kind = classify_start(previous, clean_flag, build)
    _state["start_kind"] = kind
    _state["previous"] = previous

    if previous:
        alive_h = None
        last_rss = None
        try:
            alive_h = round((_parse_ts(now_iso) - _parse_ts(previous["started_at"])).total_seconds() / 3600, 1)
        except Exception:
            pass
        with _lock:
            prev_samples = [s for s in _samples if s.get("proc") == previous.get("started_at")]
        if prev_samples:
            last_rss = prev_samples[-1].get("rss_mb")
        previous["alive_hours"] = alive_h
        previous["last_rss_mb"] = last_rss
        logger.info(f"process_health: start classified as '{kind}' "
                    f"(previous process alive {alive_h}h, last RSS {last_rss} MB, build {previous.get('build')} -> {build})")
        if kind == "unclean":
            _send_alert(
                "Backend restarted unexpectedly",
                f"The previous process (started {previous.get('started_at', '?')[:16]}Z, alive {alive_h}h) "
                f"exited without a clean shutdown and the same build came back -- "
                f"likely an OOM kill or crash. Last RSS sample before exit: "
                f"{last_rss if last_rss is not None else 'n/a'} MB. "
                f"Check `dmesg -T | grep -i 'out of memory'` on the VPS.",
                data={"previous_started_at": previous.get("started_at"),
                      "alive_hours": alive_h, "last_rss_mb": last_rss, "build": build},
            )
    else:
        logger.info(f"process_health: start classified as '{kind}' (build {build})")

    if client:
        try:
            client.set(REDIS_START_KEY, json.dumps({
                "started_at": now_iso, "build": build, "pid": os.getpid(),
            }))
        except Exception as e:
            logger.debug(f"process_health: could not write start record: {e}")
    return kind


def record_scan_sample(label: str = "scan") -> Optional[dict]:
    """Post-scan hook. gc.collect() first so RSS reflects what the process
    actually retains, not cyclic garbage awaiting the next automatic pass."""
    try:
        collected = gc.collect()
    except Exception:
        collected = -1
    diag = None
    if LEAK_DIAG and read_memory() is not None:
        # Before the RSS read, so rss_mb below is POST-trim (what the process
        # truly retains) and rss_pre_trim_mb keeps the untrimmed figure.
        diag = _leak_diag_step(
            (datetime.now(timezone.utc) - _parse_ts(_state["started_at"])).total_seconds() / 3600)
    mem = read_memory()
    if mem is None:
        return None
    sample = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "rss_mb": mem["rss_mb"],
        "hwm_mb": mem["hwm_mb"],
        "gc": collected,
        "proc": _state["started_at"],
        "label": label,
    }
    if diag:
        sample.update(diag)
    with _lock:
        _samples.append(sample)
        snapshot = list(_samples)
    rate = growth_rate_mb_per_hour(snapshot, _state["started_at"])
    uptime_h = (_parse_ts(sample["ts"]) - _parse_ts(_state["started_at"])).total_seconds() / 3600
    limit = container_limit_mb()
    logger.info(
        f"Process memory: RSS {sample['rss_mb']} MB (HWM {sample['hwm_mb']}"
        f"{f', limit {limit}' if limit else ''}) | gc freed {collected} | "
        f"trend {f'{rate:+.1f} MB/h' if rate is not None else 'n/a'} | uptime {uptime_h:.1f}h"
    )
    if diag:
        blocks = diag.get("py_blocks")
        logger.info(
            f"Leak diag: py blocks {f'{blocks / 1e6:.2f}M' if blocks is not None else 'n/a'} | "
            f"gc objects {diag.get('gc_objects', 'n/a')} | "
            f"trim freed {diag.get('trim_freed_mb', 'n/a')} MB "
            f"(pre-trim RSS {diag.get('rss_pre_trim_mb', 'n/a')} MB)"
        )
        growth = _state["census_growth"]
        if growth:
            logger.info("Leak census vs %s baseline: %s" % (
                (_state["census_baseline_at"] or "?")[:16],
                ", ".join(f"{g['type']} +{g['delta']}" for g in growth[:6])))
    _persist_samples()

    if sample["rss_mb"] >= RSS_ALERT_MB and _state["alerted_mb"] is None:
        _state["alerted_mb"] = sample["rss_mb"]
        eta = None
        if rate and rate > 0 and limit:
            eta = round((limit - sample["rss_mb"]) / rate, 1)
        _send_alert(
            f"Backend memory high: {sample['rss_mb']} MB",
            f"RSS crossed the {RSS_ALERT_MB} MB alert line after {uptime_h:.1f}h "
            f"(trend {f'{rate:+.1f} MB/h' if rate is not None else 'n/a'}"
            f"{f', ~{eta}h to the {limit} MB container limit' if eta is not None else ''}). "
            f"A restart clears it; the leak hunt is the fix. "
            f"`docker-compose restart canslim` on the VPS.",
            data={"rss_mb": sample["rss_mb"], "rate_mb_per_hour": rate,
                  "limit_mb": limit, "uptime_hours": round(uptime_h, 1)},
        )
    return sample


def get_process_health(recent: int = 64) -> dict:
    """Snapshot for /api/system-health."""
    mem = read_memory() or {"rss_mb": None, "hwm_mb": None}
    with _lock:
        snapshot = list(_samples)
    now = datetime.now(timezone.utc)
    try:
        uptime_h = round((now - _parse_ts(_state["started_at"])).total_seconds() / 3600, 2)
    except Exception:
        uptime_h = None
    return {
        "rss_mb": mem["rss_mb"],
        "hwm_mb": mem["hwm_mb"],
        "limit_mb": container_limit_mb(),
        "alert_mb": RSS_ALERT_MB,
        "alerted_mb": _state["alerted_mb"],
        "started_at": _state["started_at"],
        "uptime_hours": uptime_h,
        "start_kind": _state["start_kind"],
        "previous": _state["previous"],
        "growth_mb_per_hour": growth_rate_mb_per_hour(snapshot, _state["started_at"]),
        "sample_count": len(snapshot),
        "samples": [{"ts": s["ts"], "rss_mb": s["rss_mb"], "proc": s.get("proc"),
                     **{k: s[k] for k in ("py_blocks", "rss_pre_trim_mb", "trim_freed_mb", "gc_objects")
                        if k in s}}
                    for s in snapshot[-recent:]],
        "leak_diag": {
            "enabled": LEAK_DIAG,
            "census_baseline_at": _state["census_baseline_at"],
            "census_growth": _state["census_growth"],
        },
    }


def _reset_for_tests():
    with _lock:
        _samples.clear()
    _state.update({"start_kind": None, "previous": None, "alerted_mb": None, "limit_mb": None,
                   "started_at": datetime.now(timezone.utc).isoformat(),
                   "census_baseline": None, "census_baseline_at": None, "census_growth": None})
