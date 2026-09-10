"""process_health (2026-09-08 OOM follow-up): RSS sampling + one-shot alert,
restart classification, and the scheduler's non-trading-day scan throttle."""
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from backend import process_health as ph

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


class FakeRedis:
    def __init__(self):
        self.d = {}
    def get(self, k):
        return self.d.get(k)
    def set(self, k, v, ex=None):
        self.d[k] = v
    def delete(self, k):
        self.d.pop(k, None)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    ph._reset_for_tests()
    monkeypatch.setattr(ph, "_redis", lambda: None)
    alerts = []
    monkeypatch.setattr(ph, "_send_alert", lambda title, msg, data=None: alerts.append((title, msg, data)))
    yield alerts
    ph._reset_for_tests()


def _samples(proc, start, hours, base=800, slope=10, step_h=1.0):
    out = []
    t = start
    for i in range(int(hours / step_h) + 1):
        out.append({"ts": t.isoformat(), "rss_mb": base + slope * i * step_h, "proc": proc})
        t += timedelta(hours=step_h)
    return out


class TestClassifyStart:
    def test_first_when_no_previous(self):
        assert ph.classify_start(None, False, "b1") == "first"

    def test_clean_flag_wins_even_on_same_build(self):
        assert ph.classify_start({"build": "b1"}, True, "b1") == "clean"

    def test_build_change_without_flag_is_deploy(self):
        assert ph.classify_start({"build": "b0"}, False, "b1") == "deploy"

    def test_same_build_without_flag_is_unclean(self):
        assert ph.classify_start({"build": "b1"}, False, "b1") == "unclean"


class TestGrowthRate:
    def test_linear_slope_recovered(self):
        s = _samples("p", datetime(2026, 9, 8, tzinfo=UTC), hours=6, slope=10)
        assert ph.growth_rate_mb_per_hour(s, "p") == pytest.approx(10.0, abs=0.1)

    def test_needs_three_points(self):
        s = _samples("p", datetime(2026, 9, 8, tzinfo=UTC), hours=1, slope=10)
        assert len(s) == 2
        assert ph.growth_rate_mb_per_hour(s, "p") is None

    def test_ignores_other_process_samples(self):
        old = _samples("old", datetime(2026, 9, 7, tzinfo=UTC), hours=10, slope=50)
        new = _samples("p", datetime(2026, 9, 8, tzinfo=UTC), hours=1, slope=0)
        assert ph.growth_rate_mb_per_hour(old + new, "p") is None
        assert ph.growth_rate_mb_per_hour(old + new, "old") == pytest.approx(50.0, abs=0.1)

    def test_span_under_one_hour_is_noise(self):
        s = _samples("p", datetime(2026, 9, 8, tzinfo=UTC), hours=0.5, slope=10, step_h=0.25)
        assert ph.growth_rate_mb_per_hour(s, "p") is None

    def test_window_limits_fit_to_recent_samples(self):
        # 48h flat then 6h steep: 24h window sees mostly the steep part
        start = datetime(2026, 9, 6, tzinfo=UTC)
        flat = _samples("p", start, hours=48, slope=0)
        steep = _samples("p", start + timedelta(hours=49), hours=6, base=800, slope=20)
        narrow = ph.growth_rate_mb_per_hour(flat + steep, "p", window_hours=6)
        wide = ph.growth_rate_mb_per_hour(flat + steep, "p", window_hours=1000)
        assert narrow == pytest.approx(20.0, abs=0.1)   # window isolates the recent regime
        assert wide is not None and 0 < wide < narrow    # full history dilutes it


class TestRecordScanSample:
    def test_sample_shape_and_history(self, monkeypatch):
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": 900, "hwm_mb": 950})
        s = ph.record_scan_sample()
        assert s["rss_mb"] == 900 and s["hwm_mb"] == 950
        assert s["proc"] == ph._state["started_at"]
        assert isinstance(s["gc"], int)
        assert ph.get_process_health()["sample_count"] == 1

    def test_no_proc_means_no_sample(self, monkeypatch):
        monkeypatch.setattr(ph, "read_memory", lambda: None)
        assert ph.record_scan_sample() is None
        assert ph.get_process_health()["sample_count"] == 0

    def test_alert_fires_once_above_threshold(self, monkeypatch, _isolate):
        alerts = _isolate
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": ph.RSS_ALERT_MB + 100, "hwm_mb": 2000})
        ph.record_scan_sample()
        ph.record_scan_sample()
        assert len(alerts) == 1
        assert "memory high" in alerts[0][0].lower()
        assert ph.get_process_health()["alerted_mb"] == ph.RSS_ALERT_MB + 100

    def test_no_alert_below_threshold(self, monkeypatch, _isolate):
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": ph.RSS_ALERT_MB - 1, "hwm_mb": 1800})
        ph.record_scan_sample()
        assert _isolate == []


class TestCheckRestart:
    def test_first_start_without_redis(self, _isolate):
        assert ph.check_restart(build="b1") == "first"
        assert _isolate == []

    def test_unclean_restart_alerts_with_last_rss(self, monkeypatch, _isolate):
        r = FakeRedis()
        prev_start = "2026-09-04T14:17:40+00:00"
        r.set(ph.REDIS_START_KEY, json.dumps({"started_at": prev_start, "build": "b1"}))
        r.set(ph.REDIS_SAMPLES_KEY, json.dumps([
            {"ts": "2026-09-08T07:50:00+00:00", "rss_mb": 2790, "hwm_mb": 2790, "proc": prev_start},
        ]))
        monkeypatch.setattr(ph, "_redis", lambda: r)
        ph.restore_history()
        assert ph.check_restart(build="b1") == "unclean"
        assert len(_isolate) == 1
        title, msg, data = _isolate[0]
        assert "unexpectedly" in title
        assert data["last_rss_mb"] == 2790
        assert data["alive_hours"] and data["alive_hours"] > 90
        # new record written for the NEXT start to compare against
        new = json.loads(r.get(ph.REDIS_START_KEY))
        assert new["build"] == "b1" and new["started_at"] != prev_start
        health = ph.get_process_health()
        assert health["start_kind"] == "unclean"
        assert health["previous"]["last_rss_mb"] == 2790

    def test_clean_flag_consumed_and_no_alert(self, monkeypatch, _isolate):
        r = FakeRedis()
        r.set(ph.REDIS_START_KEY, json.dumps({"started_at": "2026-09-08T09:25:27+00:00", "build": "b1"}))
        r.set(ph.REDIS_CLEAN_KEY, "1")
        monkeypatch.setattr(ph, "_redis", lambda: r)
        assert ph.check_restart(build="b1") == "clean"
        assert _isolate == []
        assert r.get(ph.REDIS_CLEAN_KEY) is None

    def test_deploy_when_build_changes(self, monkeypatch, _isolate):
        r = FakeRedis()
        r.set(ph.REDIS_START_KEY, json.dumps({"started_at": "2026-09-08T09:25:27+00:00", "build": "b0"}))
        monkeypatch.setattr(ph, "_redis", lambda: r)
        assert ph.check_restart(build="b1") == "deploy"
        assert _isolate == []

    def test_mark_clean_shutdown_sets_flag(self, monkeypatch):
        r = FakeRedis()
        monkeypatch.setattr(ph, "_redis", lambda: r)
        ph.mark_clean_shutdown()
        assert r.get(ph.REDIS_CLEAN_KEY) == "1"


class TestGetProcessHealth:
    def test_shape(self):
        h = ph.get_process_health()
        for k in ("rss_mb", "hwm_mb", "limit_mb", "alert_mb", "started_at", "uptime_hours",
                  "start_kind", "previous", "growth_mb_per_hour", "sample_count", "samples"):
            assert k in h
        assert h["alert_mb"] == ph.RSS_ALERT_MB


# ---------------------------------------------------------------- leak diag

class TestLeakDiag:
    """LEAK_DIAG=1 (2026-09-10): split the residual ~13 MB/h RSS growth left
    after MALLOC_ARENA_MAX=2 into Python-object growth vs free-but-held heap,
    without tracemalloc (measured: too heavy for this box)."""

    def _hours_ago(self, h):
        ph._state["started_at"] = (datetime.now(UTC) - timedelta(hours=h)).isoformat()

    def test_off_by_default_runs_no_probes(self, monkeypatch):
        monkeypatch.setattr(ph, "LEAK_DIAG", False)
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": 900, "hwm_mb": 950})

        def boom(*a, **k):
            raise AssertionError("probe ran with LEAK_DIAG off")
        monkeypatch.setattr(ph, "malloc_trim", boom)
        monkeypatch.setattr(ph, "type_census", boom)
        s = ph.record_scan_sample()
        assert "py_blocks" not in s and "trim_freed_mb" not in s

    def test_on_sample_carries_probes_and_rss_is_post_trim(self, monkeypatch):
        monkeypatch.setattr(ph, "LEAK_DIAG", True)
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": 800, "hwm_mb": 950})
        monkeypatch.setattr(ph, "malloc_trim",
                            lambda: {"rss_pre_trim_mb": 900, "trim_freed_mb": 100})
        monkeypatch.setattr(ph, "type_census", lambda: {"builtins.dict": 10})
        s = ph.record_scan_sample()
        assert s["rss_mb"] == 800                  # post-trim: what is truly retained
        assert s["rss_pre_trim_mb"] == 900
        assert s["trim_freed_mb"] == 100
        assert isinstance(s["py_blocks"], int) and s["py_blocks"] > 0
        assert s["gc_objects"] == 10
        row = ph.get_process_health()["samples"][-1]
        assert row["trim_freed_mb"] == 100 and "py_blocks" in row

    def test_census_baseline_waits_for_warm_up(self, monkeypatch):
        # Cache warm-up is growth too; a baseline taken at boot would name the
        # caches as the leak.
        monkeypatch.setattr(ph, "LEAK_DIAG", True)
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": 800, "hwm_mb": 950})
        monkeypatch.setattr(ph, "malloc_trim", lambda: None)
        counts = {"v": {"a.X": 1}}
        monkeypatch.setattr(ph, "type_census", lambda: dict(counts["v"]))

        self._hours_ago(ph.LEAK_DIAG_WARMUP_H - 1)
        ph.record_scan_sample()
        assert ph._state["census_baseline"] is None

        self._hours_ago(ph.LEAK_DIAG_WARMUP_H + 1)
        ph.record_scan_sample()
        assert ph._state["census_baseline"] == {"a.X": 1}
        assert ph._state["census_growth"] is None

        counts["v"] = {"a.X": 5, "b.Y": 2}
        ph.record_scan_sample()
        growth = ph.get_process_health()["leak_diag"]["census_growth"]
        assert [(g["type"], g["delta"]) for g in growth] == [("a.X", 4), ("b.Y", 2)]

    def test_census_growth_ignores_shrinking_and_ranks(self):
        g = ph.census_growth({"a": 10, "b": 5, "c": 1}, {"a": 3, "b": 9, "c": 50, "d": 2}, top=2)
        assert [(x["type"], x["delta"]) for x in g] == [("c", 49), ("b", 4)]

    def test_a_failing_probe_never_breaks_the_sample(self, monkeypatch):
        monkeypatch.setattr(ph, "LEAK_DIAG", True)
        monkeypatch.setattr(ph, "read_memory", lambda: {"rss_mb": 800, "hwm_mb": 950})

        def boom():
            raise RuntimeError("probe exploded")
        monkeypatch.setattr(ph, "type_census", boom)
        monkeypatch.setattr(ph, "malloc_trim", boom)
        s = ph.record_scan_sample()
        assert s is not None and s["rss_mb"] == 800

    def test_real_probes_run_on_this_platform(self):
        # Unpatched: must return a well-formed result or None, never raise.
        census = ph.type_census()
        assert census and all(isinstance(v, int) for v in census.values())
        trim = ph.malloc_trim()
        assert trim is None or trim["trim_freed_mb"] >= 0


# ---------------------------------------------------------------- scan throttle

class TestIsTradingDay:
    def test_calendar(self):
        from backend.ai_trader import is_trading_day
        assert is_trading_day(datetime(2026, 9, 8, 12, tzinfo=ET)) is True     # Tue
        assert is_trading_day(datetime(2026, 9, 5, 12, tzinfo=ET)) is False    # Sat
        assert is_trading_day(datetime(2026, 9, 6, 12, tzinfo=ET)) is False    # Sun
        assert is_trading_day(datetime(2026, 9, 7, 12, tzinfo=ET)) is False    # Labor Day
        assert is_trading_day(datetime(2026, 9, 4, 23, tzinfo=ET)) is True     # Fri evening


class TestNonTradingDayThrottle:
    @pytest.fixture
    def skip(self):
        from backend.scheduler import _should_skip_non_trading_day_scan
        from backend.ai_trader import is_trading_day
        def _call(now_utc, last_iso):
            return _should_skip_non_trading_day_scan(now_utc, last_iso, is_trading_day)
        return _call

    def test_trading_day_never_skips(self, skip):
        assert skip(datetime(2026, 9, 8, 15, tzinfo=UTC), "2026-09-08T14:00:00+00:00") is None

    def test_first_weekend_scan_runs(self, skip):
        # Saturday 11:00 ET, last scan Friday night -> run (one refresh/day)
        assert skip(datetime(2026, 9, 5, 15, tzinfo=UTC), "2026-09-05T03:30:00+00:00") is None

    def test_second_weekend_scan_same_et_day_skips(self, skip):
        reason = skip(datetime(2026, 9, 5, 15, tzinfo=UTC), "2026-09-05T05:00:00+00:00")
        assert reason and "non-trading day" in reason and "Sat 2026-09-05" in reason

    def test_friday_evening_utc_saturday_is_still_trading_day_in_et(self, skip):
        # 01:00 UTC Sat = 21:00 ET Fri -> trading day, never throttled
        assert skip(datetime(2026, 9, 5, 1, tzinfo=UTC), "2026-09-05T00:00:00+00:00") is None

    def test_et_day_boundary_uses_eastern_not_utc(self, skip):
        # 03:00 UTC Sun = 23:00 ET Sat; last scan 15:00 UTC Sat = 11:00 ET Sat -> same ET day -> skip
        assert skip(datetime(2026, 9, 6, 3, tzinfo=UTC), "2026-09-05T15:00:00+00:00")
        # 05:00 UTC Sun = 01:00 ET Sun -> new ET day -> run
        assert skip(datetime(2026, 9, 6, 5, tzinfo=UTC), "2026-09-05T15:00:00+00:00") is None

    def test_holiday_throttled_like_weekend(self, skip):
        assert skip(datetime(2026, 9, 7, 18, tzinfo=UTC), "2026-09-07T06:00:00+00:00")

    def test_no_prior_scan_runs(self, skip):
        assert skip(datetime(2026, 9, 5, 15, tzinfo=UTC), None) is None

    def test_naive_iso_treated_as_utc(self, skip):
        assert skip(datetime(2026, 9, 5, 15, tzinfo=UTC), "2026-09-05T05:00:00")

    def test_last_skip_recorded_and_scan_declined(self, monkeypatch):
        from backend import scheduler as sch
        monkeypatch.setattr(sch, "_non_trading_day_skip_reason", lambda: "non-trading day (test)")
        called = []
        monkeypatch.setattr(sch, "_check_universe_shrink", lambda n: called.append(n))
        with sch._state_lock:
            sch._scan_config["last_skip"] = None
            sch._scan_config["is_scanning"] = False
        sch.run_continuous_scan()
        assert called == []
        assert sch._scan_config["last_skip"]["reason"] == "non-trading day (test)"
        assert sch._scan_config["is_scanning"] is False
