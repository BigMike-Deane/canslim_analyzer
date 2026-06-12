# D1 Stop-Loss Clamp (`new_position_guard`) — June-19 Landing Playbook

**Status:** staged + verified 2026-06-12. **Cannot land until the freeze lifts
(2026-06-18 17:00 UTC)** — it modifies `backend/ai_trader.py`, and the deploy
poller refuses any deploy touching `ai_trader.py`/`canslim_scorer.py` before
then. Land it as the first item of the June-19 exit-parity work.

## The bug (D1)
`evaluate_sells()` — the **only** sell path the scheduler runs in production —
goes straight from `calculate_atr_stop()` to the stop check with **no
new-position guard**. So a fast-falling new buy rides the ATR-widened stop (up
to the 20% cap) instead of being clamped to 8% in its first 21 days. The guard
**is** present in `_check_and_execute_stop_losses_impl` (the manual-only path)
and in the backtester — so live is backtest-optimistic and trader↔backtester
are out of sync. Confirmed live cost: FSLR −18%, AMD −13.7% (June-4 buys),
and a real −$967 FSLR stop-out on June 9.

## ⚠️ Do NOT use the existing `fix/stoploss-new-position-guard` branch
It is **stale** — based on an old `main`, so merging it would revert ~442 lines
of recent `backtester.py` work (exit-model capture, etc.) and delete files that
have since changed. Apply the change set below fresh onto current `main`.

## The change set — ONE atomic commit

### 1. `backend/ai_trader.py` — insert the guard in `evaluate_sells`
Right **after** `position_stop_pct = calculate_atr_stop(...)` (currently
~line 1623, just before the `if gain_pct <= -position_stop_pct:` check), insert:

```python
        # F: NEW POSITION GUARD — tighter stop for new positions in first N days.
        # MIRROR of _check_and_execute_stop_losses_impl (~1304-1314) and backtester
        # _evaluate_sells (~2646-2656). Without it, a fast-falling new buy's ATR stop
        # widens pro-cyclically toward the 20% cap and the intended 8% stop never fires
        # (jun-09 bug: live FSLR held to -18%, AMD -13.7%). Restores trader<->backtester sync.
        guard_config = yaml_config.get('ai_trader.new_position_guard', {})
        if guard_config.get('enabled', False) and position.purchase_date:
            guard_days = guard_config.get('guard_days', 21)
            guard_stop_pct = guard_config.get('guard_stop_pct', 8.0)
            skip_if_pyramided = guard_config.get('skip_if_pyramided', True)
            purchase_dt = position.purchase_date.date() if hasattr(position.purchase_date, 'date') else position.purchase_date
            holding_days = (date.today() - purchase_dt).days
            pyramid_count = getattr(position, 'pyramid_count', 0) or 0
            if holding_days <= guard_days:
                if not (skip_if_pyramided and pyramid_count > 0):
                    position_stop_pct = min(position_stop_pct, guard_stop_pct)
```

This is **byte-identical to the proven block** in `_check_and_execute_stop_losses_impl`
(`ai_trader.py:1304-1314`), which runs in production today. Verified in scope:
`yaml_config` is defined at `evaluate_sells` (`ai_trader.py:1570`), `date` is
imported (`ai_trader.py:8`).

### 2. Delete `backend/stop_guard_monitor.py`
The breach-alert sentinel exists only to cover this bug. Its own
`test_evaluate_sells_still_missing_guard` says delete it when the guard lands.

### 3. Delete `tests/test_stop_guard_monitor.py`
Same reason — it pins "evaluate_sells has NO guard," which is now false.

### 4. `backend/scheduler.py` — remove the monitor job
- Remove the `start_stop_guard_monitor_job()` call (~line 1881).
- Remove the `start_stop_guard_monitor_job` def (~2556) and `_run_stop_guard_monitor` (~2581).

### 5. `tests/test_evaluate_sells_d1_parity.py` — un-xfail the core test
Remove the `@pytest.mark.xfail(reason="D1 …", strict=True)` decorator on
`test_new_position_below_atr_stop_is_cut_by_guard`. With the guard present the
test passes; leaving the strict-xfail would turn the XPASS into a hard failure.
The two control tests are unaffected.

### 6. `backend/backtester.py` — NO CHANGE
The guard is already present (`backtester.py:2646-2656`,
`effective_stop_loss_pct = min(effective_stop_loss_pct, guard_stop_pct)`).
Parity is restored by step 1 alone; the trader↔backtester sync rule is satisfied.

## Verify after applying
```bash
export CANSLIM_ENV=development
python3 -m pytest tests/test_evaluate_sells_d1_parity.py -q   # expect 3 passed, 0 xfailed
python3 -m pytest tests/test_ai_trader_sync.py tests/test_backtester_trading_parity.py -q
python3 -m pytest tests/ -q                                   # full suite green
```
The D1 harness flipping from `1 xfailed` → `3 passed` (no strict-xfail failure)
is the proof the clamp fires.

## Ship
Normal `main → deploy` release. Post-June-18 the freeze guard permits the
`ai_trader.py` change. Bump the `build_info.py` Central-time stamp like any deploy.
