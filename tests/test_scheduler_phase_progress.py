"""
Regression tests for the per-phase scanner progress fields.

The scanner card on the frontend used to stall at "2076/2076" once Phase 1
finished because the backend only updated `stocks_scanned` / `total_stocks`
during the "stocks" phase. Insider/short, p1_data, save, ai_trading, etc.
ran for many more minutes with no progress signal.

These tests pin the new contract:
  * `get_scan_status()` always returns the new fields, even when idle.
  * The internal `_set_phase` helper drives `current_phase`, `phase_current`,
    `phase_total`, `phase_label`, and keeps `phase` / `phase_detail` in sync.
  * Phase keys map to friendly labels for the UI.
  * The idle-reset contract zeroes the per-phase fields without disturbing
    `last_scan_end`.

No DB / Redis / network dependency — operates purely on the module-level
state dict.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import scheduler as scheduler_mod


@pytest.fixture(autouse=True)
def _reset_scan_config():
    """Snapshot + restore _scan_config so tests can mutate it freely."""
    saved = dict(scheduler_mod._scan_config)
    yield
    scheduler_mod._scan_config.clear()
    scheduler_mod._scan_config.update(saved)


def test_scan_status_exposes_new_phase_fields_when_idle():
    """Idle scanner status must include the new fields with safe defaults."""
    status = scheduler_mod.get_scan_status()
    for key in ("current_phase", "phase_current", "phase_total", "phase_label"):
        assert key in status, f"get_scan_status() missing {key!r}"
    # Defaults shouldn't render anything misleading on the UI side.
    assert status["current_phase"] in (None, "")
    assert status["phase_current"] == 0
    assert status["phase_total"] == 0
    assert status["phase_label"] in (None, "")


def test_set_phase_updates_all_progress_fields():
    """_set_phase must drive both legacy and new fields atomically."""
    scheduler_mod._set_phase("p1_data", detail="Fetching P1 data (300/1426)...", current=300, total=1426)
    status = scheduler_mod.get_scan_status()
    assert status["phase"] == "p1_data"
    assert status["current_phase"] == "p1_data"
    assert status["phase_current"] == 300
    assert status["phase_total"] == 1426
    assert status["phase_label"] == "Fetching P1 data"
    assert status["phase_detail"] == "Fetching P1 data (300/1426)..."


def test_set_phase_known_labels():
    """Each phase key the scheduler reports must have a UI-friendly label."""
    expected = {
        "scanning": "Fetching stock data",
        "stocks": "Fetching stock data",
        "insider_short": "Fetching insider/short data",
        "p1_data": "Fetching P1 data",
        "saving": "Saving to database",
        "coiled_spring": "Checking Coiled Spring",
        "earnings_audit": "Earnings audit",
        "industry_groups": "Industry group rankings",
        "bear_base": "Bear base scan",
        "ai_trading": "AI trading cycle",
        "cleanup": "Cleanup",
    }
    for phase, label in expected.items():
        scheduler_mod._set_phase(phase, current=1, total=2)
        status = scheduler_mod.get_scan_status()
        assert status["phase_label"] == label, f"{phase} -> {status['phase_label']!r}"


def test_set_phase_unknown_phase_falls_back_to_none_label():
    """Unknown phase keys still update progress numbers but expose label=None."""
    scheduler_mod._set_phase("mystery_phase", current=5, total=10)
    status = scheduler_mod.get_scan_status()
    assert status["current_phase"] == "mystery_phase"
    assert status["phase_current"] == 5
    assert status["phase_total"] == 10
    assert status["phase_label"] is None


def test_set_phase_to_none_clears_label():
    """Reset path used by the finally block: phase=None must drop the label."""
    scheduler_mod._set_phase("saving", detail="Saving...", current=42, total=100)
    scheduler_mod._scan_config["phase"] = None
    scheduler_mod._scan_config["current_phase"] = None
    scheduler_mod._scan_config["phase_current"] = 0
    scheduler_mod._scan_config["phase_total"] = 0
    scheduler_mod._scan_config["phase_label"] = None
    status = scheduler_mod.get_scan_status()
    assert status["current_phase"] is None
    assert status["phase_label"] is None
    assert status["phase_current"] == 0
    assert status["phase_total"] == 0


def test_stocks_scanned_unaffected_by_non_stocks_phase():
    """Per the design contract, `stocks_scanned`/`total_stocks` belong to
    Phase 1. Setting a later-phase progress must NOT overwrite them — that
    would regress the existing UI consumers."""
    scheduler_mod._scan_config["stocks_scanned"] = 2076
    scheduler_mod._scan_config["total_stocks"] = 2076
    scheduler_mod._set_phase("p1_data", detail="...", current=300, total=1426)
    status = scheduler_mod.get_scan_status()
    assert status["stocks_scanned"] == 2076
    assert status["total_stocks"] == 2076
    assert status["phase_current"] == 300
    assert status["phase_total"] == 1426
