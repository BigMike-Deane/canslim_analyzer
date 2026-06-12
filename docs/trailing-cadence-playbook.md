# Trailing-Stop Cadence Redesign — June-19 Landing Playbook

**Status:** mechanism built + tested + shipped **default-off** 2026-06-12.
**The one frozen-file edit can't land until the freeze lifts (2026-06-18 17:00
UTC).** June-19 exit-parity item 2.

## The problem
Live `evaluate_sells()` runs every ~90-min scan, so **trailing stops fire
intraday** and shake winners out on ordinary daily ranges. The backtester
evaluates once per day at the close, so it never models this churn — a
trader↔backtester parity gap. Measured drag of intraday cadence: **−1..−17pp
per 2-yr window, 8/8 arms**; live-validated (live trailing exits held **9.2d /
50% WR / −0.1% avg** vs **81d / 76% / +3.6%** modeled). Hard stops (stop-loss,
score-crash) must stay intraday for crash protection — only **trailing** moves
to daily.

## Already shipped (2026-06-12, freeze-safe, default-OFF — no behavior change)
- `backend/trading_engine.trailing_stops_allowed_now(yaml_config, now_et)` —
  the cadence gate. Off → always True (current behavior). On → True only inside
  the close window.
- `config/default.yaml` → `ai_trader.trailing_cadence` block (`daily_only:
  false`, window 15:00–16:00 ET).
- `tests/test_trailing_cadence.py` — 7 tests pinning the window math.

So June-19 is just the gate + the flip below.

## The June-19 change — minimal, atomic

### 1. `backend/ai_trader.py` — gate the trailing-stop fire (ONE line + import)
Add `trailing_stops_allowed_now` to the existing `from backend.trading_engine
import (...)` block (~line 37).

Then at the trailing-stop fire (~**line 1651**), change:
```python
            if trailing_stop_pct and drop_from_peak >= trailing_stop_pct:
```
to:
```python
            if trailing_stop_pct and drop_from_peak >= trailing_stop_pct and trailing_stops_allowed_now(yaml_config):
```
`yaml_config` is already in scope (`evaluate_sells` line 1570). This gates BOTH
the full and the partial trailing stop (both live under this `if`). Out of the
window, trailing simply doesn't fire and the position falls through to the
other (hard) sell checks — which is correct.

### 2. Enable the lever
Flip `ai_trader.trailing_cadence.daily_only` → `true` (in `config/default.yaml`
or `config/production.yaml`).

### 3. `backend/backtester.py` — NO CHANGE
It already evaluates trailing once daily at the close (the honest control arm).
Enabling the live lever is what RESTORES parity; nothing to mirror.

## Design decisions (tune at landing)
- **Window = final hour before close (15:00–16:00 ET)**, config-driven. The
  ~15:30 ET scan lands inside it, so trailing evaluates once/day near the close.
  Tune via `window_start_hour_et`/`_minute_et` if you want it tighter.
- **Hard stops unaffected** — stop-loss and score-crash never call the gate.
- **Partial trailing stops are included** (same `if`), which is intended — a
  partial trailing exit is still trailing churn.
- **Open consideration:** the separate PRE-EARNINGS trailing-tighten block
  (`ai_trader.py` ~1689) is NOT gated by this. It's lower-churn and earnings-
  scoped; decide at landing whether to gate it too for full consistency
  (recommended: yes, wrap its fire in the same `trailing_stops_allowed_now`).

## Verify after landing
```bash
export CANSLIM_ENV=development
python3 -m pytest tests/test_trailing_cadence.py -q       # 7 passed
python3 -m pytest tests/test_ai_trader_coverage.py -q     # trailing-branch tests still green
# Manual: with daily_only=true, confirm a -drop position fires TRAILING STOP only
# when get_cst_now/ET is in 15:00-16:00, not on a 11:00 ET scan.
python3 -m pytest tests/ -q                               # full suite
```

## Parity nuance
Live evaluates trailing on the ~15:30 ET intraday price; the backtester uses the
16:00 close. That residual (~30 min) is far smaller than the current every-90-min
gap, and daily cadence is the audit's honest control. Acceptable; note it.

## Ship
Normal `main → deploy` post-June-18. Bump the `build_info.py` Central-time stamp.
