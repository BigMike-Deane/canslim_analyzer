"""Freeze-safe halves of June-19 minor items N1 (partial-stop notification) and
D2 (score-crash clock dedup). The ai_trader.py call sites that opt into these
are staged in docs/june19-minor-items-playbook.md; these pin the new behavior
so the June-19 hookup is a one-liner.
"""

from datetime import datetime

import backend.email_utils as eu
from backend.trading_engine import dedup_scores_to_daily


# ── N1: partial trailing stop must not look like a full liquidation ──────────

def _capture_webhook(monkeypatch):
    cap = {}
    monkeypatch.setattr(eu, "create_notification", lambda *a, **k: cap.update(k) or True)
    monkeypatch.setattr(eu, "send_webhook_notification",
                        lambda title, message, **k: cap.update({"wire_title": title, "wire_message": message}) or True)
    monkeypatch.setattr(eu, "get_user_webhook_url", lambda uid: None)
    return cap


def test_partial_stop_webhook_says_position_still_open(monkeypatch):
    cap = _capture_webhook(monkeypatch)
    eu.send_stop_loss_webhook("IESC", 4.71, 687.45, "TRAILING STOP", 2.0,
                              user_id=1, is_partial=True, shares_kept=4.71)
    assert "PARTIAL TRAILING STOP" in cap["title"]
    assert "STILL OPEN" in cap["body"]
    assert "4.71 shares kept" in cap["body"]
    assert cap["data"]["is_partial"] is True


def test_full_stop_webhook_unchanged_by_default(monkeypatch):
    cap = _capture_webhook(monkeypatch)
    eu.send_stop_loss_webhook("FSLR", 10.0, 100.0, "STOP LOSS", -8.0, user_id=1)
    assert cap["title"] == "STOP LOSS: FSLR"
    assert "Automatic stop triggered" in cap["body"]
    assert "PARTIAL" not in cap["title"]
    assert cap["data"]["is_partial"] is False


# ── D2: collapse scan-frequency scores to one per calendar day ───────────────

class _Row:
    def __init__(self, ts, score):
        self.timestamp = ts
        self.total_score = score


def test_dedup_keeps_latest_scan_per_day_capped():
    rows = [  # newest-first, as the live query returns
        _Row(datetime(2026, 6, 12, 15, 0), 40),  # day 3 — latest scan
        _Row(datetime(2026, 6, 12, 10, 0), 45),  # day 3 — earlier scan (dropped)
        _Row(datetime(2026, 6, 11, 15, 0), 50),  # day 2
        _Row(datetime(2026, 6, 10, 15, 0), 60),  # day 1
        _Row(datetime(2026, 6, 9, 15, 0), 70),   # day 0 — beyond lookback=3
    ]
    out = dedup_scores_to_daily(rows, lookback=3)
    assert [r.total_score for r in out] == [40, 50, 60]


def test_dedup_skips_missing_timestamp():
    rows = [_Row(None, 40), _Row(datetime(2026, 6, 12, 10, 0), 45)]
    assert [r.total_score for r in dedup_scores_to_daily(rows, lookback=3)] == [45]
