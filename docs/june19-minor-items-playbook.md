# June-19 Minor Items — N1 (partial-stop notification) + D2 (score-crash clock)

Both are one-liner hookups in the frozen `ai_trader.py`; the freeze-safe halves
(email template, dedup helper, tests) are **already shipped** 2026-06-12.
Land these with the other June-19 exit-parity work.

## N1 — partial trailing stop must not read as a full exit
**Bug (live 2026-06-10):** an IESC 50% partial trailing stop sent
`TRAILING STOP: IESC / 4.71 shares @ $687.45 (+2.0%)` — indistinguishable from a
full liquidation; owner reasonably thought the position closed. Cause:
`execute_trade` (`ai_trader.py:1508`) matches `"PARTIAL TRAILING STOP …"` via the
`"TRAILING STOP" in reason` substring and calls `send_stop_loss_webhook` with the
generic `stop_type` and no "still open" signal.

**Shipped (freeze-safe):** `email_utils.send_stop_loss_webhook` now takes
`is_partial=False` / `shares_kept=None` — when partial it titles `PARTIAL …` and
says `position is STILL OPEN`. Defaults preserve the old message exactly.

**June-19 change** — `ai_trader.py` ~1508, in `execute_trade`:
```python
            if "STOP LOSS" in reason or "TRAILING STOP" in reason:
                stop_type = "TRAILING STOP" if "TRAILING STOP" in reason else "STOP LOSS"
                send_stop_loss_webhook(ticker, shares, price, stop_type, gain_pct,
                                       user_id=user_id, is_partial=reason.startswith("PARTIAL"))
```
(`shares_kept` is optional — `execute_trade` doesn't have the remainder handy;
`is_partial` alone fixes the "looks like a full exit" confusion. Pass
`shares_kept` too if the remaining-share count is threaded through later.)

Verify: `tests/test_partial_stop_and_score_dedup.py` (already green) pins the
partial vs full message; after the hookup, trigger a partial trailing stop in a
test/sandbox and confirm the notification says STILL OPEN.

## D2 — score-crash clock: scans vs trading days (LOW priority)
**Note:** measured **low-impact and sign-mixed** (`score_crash_scan_clock` sweep,
runs 817-826: −3.7 / 0.0 / +0.5 / +10.4pp across 2-yr windows; fast clock *helped*
in the 2021-23 bear). It's parity hygiene, not a return lever — do it only if
touching `check_score_stability` anyway, and **not mid-bear** without re-checking.

`check_score_stability` (`ai_trader.py:509`) reads the last `lookback` *scans*
(every ~90 min), so "3 consecutive low" can confirm a crash in one bad afternoon;
the backtester appends once per *trading day*.

**Shipped (freeze-safe):** `trading_engine.dedup_scores_to_daily(rows, lookback)`
— one row per calendar day (latest scan), newest-first, capped.

**June-19 change** — `ai_trader.py` ~531, in `check_score_stability`:
```python
    from backend.trading_engine import dedup_scores_to_daily
    raw = db.query(StockScore).filter(
        StockScore.stock_id == stock.id
    ).order_by(StockScore.timestamp.desc()).limit(lookback * 8).all()
    recent_scores = dedup_scores_to_daily(raw, lookback)
```
(Fetch a wider window so `lookback` distinct days are available, then dedup. Rest
of the function is unchanged — it already consumes `recent_scores`.)

Verify: `tests/test_partial_stop_and_score_dedup.py` pins the dedup; after the
hookup, `tests/test_ai_trader_*` score-crash tests stay green.

## Ship
Bundle with the other June-19 `ai_trader.py` edits in one post-freeze
`main → deploy` release. Bump the `build_info.py` Central-time stamp.
