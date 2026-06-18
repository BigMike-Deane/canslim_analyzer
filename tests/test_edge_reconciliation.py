"""Tests for live-vs-backtest exit reconciliation (Edge Validation Phase 2)."""
from backend.edge_reconciliation import (
    summarize_exits, compare_exit_behavior,
    live_trade_to_record, backtest_trade_to_record,
)


def _t(reason, hold, ret):
    return {"reason": reason, "holding_days": hold, "realized_pct": ret}


class TestSummarizeExits:
    def test_groups_by_canonical_reason_with_fingerprints(self):
        trades = [
            _t("STOP LOSS: Down 7%", 5, -7.0),
            _t("STOP LOSS: Down 8%", 3, -8.0),
            _t("TRAILING STOP: Peak ...", 80, 12.0),
            _t("TRAILING STOP: Peak ...", 70, 4.0),
        ]
        s = summarize_exits(trades)
        assert s["STOP LOSS"]["count"] == 2
        assert s["STOP LOSS"]["win_rate"] == 0.0          # both losers
        assert s["STOP LOSS"]["avg_hold_days"] == 4.0
        assert s["TRAILING STOP"]["win_rate"] == 100.0    # both green
        assert s["TRAILING STOP"]["avg_hold_days"] == 75.0
        assert s["ALL"]["count"] == 4
        assert s["STOP LOSS"]["pct_of_sells"] == 50.0

    def test_handles_missing_fields(self):
        s = summarize_exits([_t("WEAK POSITION", None, None)])
        assert s["WEAK POSITION"]["count"] == 1
        assert s["WEAK POSITION"]["win_rate"] is None
        assert s["WEAK POSITION"]["avg_hold_days"] is None


class TestCompareExitBehavior:
    def test_aligned_when_fingerprints_match(self):
        live = summarize_exits([_t("TRAILING STOP: x", 78, 11.0)] * 5)
        bt = summarize_exits([_t("TRAILING STOP: x", 80, 12.0)] * 5)
        result = compare_exit_behavior(live, bt)
        assert result["verdict"] == "aligned"
        assert result["diverged_reasons"] == []

    def test_flags_hold_days_and_winrate_divergence(self):
        # The exact D1/cadence symptom: live trailing exits short & lossy.
        live = summarize_exits([_t("TRAILING STOP: x", 9, -0.1)] * 6)
        bt = summarize_exits([_t("TRAILING STOP: x", 81, 12.0)] * 6)
        result = compare_exit_behavior(live, bt)
        assert result["verdict"] == "diverged"
        flags = result["reasons"][0]["divergence_flags"]
        assert "hold_days" in flags and "realized_pct" in flags

    def test_presence_divergence_one_sided(self):
        # Live takes only STOP LOSS, backtest only TRAILING STOP → both flagged
        # as presence divergences (each engine takes an exit the other doesn't).
        live = summarize_exits([_t("STOP LOSS: x", 4, -7.0)] * 4)
        bt = summarize_exits([_t("TRAILING STOP: x", 80, 12.0)] * 4)
        result = compare_exit_behavior(live, bt)
        assert result["verdict"] == "diverged"
        assert "STOP LOSS" in result["diverged_reasons"]
        assert "TRAILING STOP" in result["diverged_reasons"]
        for row in result["reasons"]:
            if row["reason"] in ("STOP LOSS", "TRAILING STOP"):
                assert row["divergence_flags"] == ["presence"]

    def test_min_count_suppresses_thin_buckets(self):
        live = summarize_exits([_t("STOP LOSS: x", 2, -50.0)])   # n=1, noise
        bt = summarize_exits([_t("STOP LOSS: x", 30, -7.0)] * 10)
        result = compare_exit_behavior(live, bt, min_count=3)
        assert result["verdict"] == "aligned"  # thin live bucket not flagged


class TestNormalizers:
    def test_backtest_record(self):
        rec = backtest_trade_to_record({"reason": "STOP LOSS: x", "holding_days": 24, "gain_pct": -7.2})
        assert rec == {"reason": "STOP LOSS: x", "holding_days": 24, "realized_pct": -7.2}

    def test_live_record_computes_realized_pct(self):
        class FakeTrade:
            reason = "TRAILING STOP: x"; holding_days = 5
            cost_basis = 100.0; shares = 10.0; realized_gain = -240.0
        rec = live_trade_to_record(FakeTrade())
        assert rec["realized_pct"] == -24.0  # -240 / (100*10) * 100
        assert rec["holding_days"] == 5
