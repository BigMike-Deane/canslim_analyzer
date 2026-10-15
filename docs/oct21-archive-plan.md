# Oct-21 readout + shadow-arm archive plan (prep, 2026-10-15)

**Status: PREP — not a ruling.** The rulings are made from closes through
2026-10-21 under `docs/oct21-readout-rules.md`. This file sets out the
readout steps and the archive change in advance, so the owner's part on
Oct-21 is a single yes/no. The Oct-8 decision memo (`e3e269d`) says the
shadow strategies are archived after the binding Oct-21 readout, **with the
owner's approval**. That approval has not been given. Nothing in config
changes until it is.

## Preview (closing marks through 2026-10-14, σ = 4.4pp floor)

Δ = arm equity return − comparator equity return over the common window
(`shadow_equity_marks`). "Lever" = `lever_trades.lever_net_usd` from
`/api/admin/shadow-equity/{name}`.

| Arm | Comparator | Δ | Lever | What it shows | Rule row → likely outcome |
|---|---|---|---|---|---|
| small_cap_gate | vintage_sep23 | **−8.35pp (−1.9σ)** | −$1,896 (skipped 9 buys, net winners) | sat in cash all 16 sessions (IWM < 50MA) | clear loss → archive |
| wide_trail | baseline | +0.66pp (+0.15σ) | +$43 | exits 12/15, gate short | no effect → archive Oct-30 (read as-is) |
| cap50 | baseline | +0.01pp | $0 | 0 cap-tier fires | structurally dormant → archive |
| sector_relief | baseline | +0.08pp | $0 | 0 exempt pyramids | structurally dormant → archive |
| stop_band | vintage_sep16 | −0.04pp | $0 | 1/10 SPY closes in the band | no effect → its pre-reg says keep until the Nov/Dec fleet review |
| industry_cap | vintage_sep30 | +0.01pp | $0 | 0 names blocked | dormant so far |
| chop_damper / chop_spy / chop_entry_bar / chop_trim | — | read 2026-10-05 (D3, nothing wins) | — | — | archive Oct-30 (already decided) |
| ml_veto_off | — | read 2026-10-02 (keep the veto) | — | — | archive (already decided) |

Plain reading: no arm has changed the result by more than noise, except
small_cap_gate, which lost by sitting in cash. Five of the levers almost
never fired, so their books are copies of their controls.

## Proposed archive (needs owner approval on/after Oct-21)

**Retire 15 stacks:** chop_damper, chop_spy, chop_entry_bar, chop_trim,
ml_veto_off, wide_trail, cap50, sector_relief, stop_band, small_cap_gate,
industry_cap, vintage_sep02, vintage_sep16, vintage_sep23, vintage_sep30.

**Keep:**
- **shadow_model_v1**: the evidence-weighted model the owner tracks in the Lab.
- **shadow_baseline**: the weekly A/B email, `program_status` and
  `strategy-ab-eval` all name it (`ab_eval_email.SHADOW_BASELINE_NAME`,
  `scheduler._AB_EVAL_SHADOW_BASELINE`, `admin.py` arms list), and it is the
  champion's parity check against the live book.
- Live accounts (u1/u2/u4) and the Lab (A1, A5, model vs AI Portfolio mirrors).
- **Vintage path noise is not lost:** the `vintage_spread` cohort already
  reads the live u1/u2 same-day twins (191 days), not the synthetic copies.

**Two pre-registrations that Option A overrides (owner's call):**
stop_band's "keep until the Nov/Dec fleet review" and industry_cap's open
gate. Under Option A (stock-picking research stopped, Oct-8) neither read can
lead anywhere, so retire both.

**Why it helps beyond tidiness:** shadow stacks run one after another after
the live cycle, so shadow_model_v1 now trades about 8–10 minutes after the AI
Portfolio. With 2 stacks instead of 17 it trades just behind the live book,
closer to the owner's rule that test books must trade exactly as live would
(Sep-24). There are also fewer weekly-email sections and less load per cycle.

## Oct-21 steps (after the 3:00 PM CT close)

1. Re-run this preview from the Oct-21 closing marks. Record the rulings
   under a dated section in `oct21-readout-rules.md`, using the rule-file table B rows.
2. Owner yes/no on the archive list above.
3. On yes, remove the 15 entries from `config/default.yaml`
   `shadow_strategy_profiles` (CRLF file, binary-safe edit). The sync
   soft-archives each DB row (`archived_at`) and keeps all `shadow_trades` /
   `shadow_equity_marks` history. Then run the full test suite (tests that
   name a retired arm get updated), deploy after the close, and check that
   `program_status` lists only baseline + model_v1 and that the next
   cycle's model trade lands right after the live one.
4. Chop-family arms were due to archive Oct-30. Archiving them Oct-21 instead
   is fine: their read is final (Oct-5) and open lots do not reopen it.
