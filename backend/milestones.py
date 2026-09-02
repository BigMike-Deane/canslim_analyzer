"""Program milestone ledger — writers.

Two writers feed the program_milestones table:

1. `record_auto_milestones` — diffs the current /admin/experiment-gates
   payload against the ledger and inserts a row the first time a
   threshold is crossed (gate accrual met, stop-loss verdict fires,
   calendar clock comes due, first chop day for a chop arm). Idempotent
   via dedupe_key, so it is safe to run on every scheduler pass.
2. `seed_history` — one-time backfill of the program's history from the
   session record, with true occurred_at dates. Seeds also pre-claim
   dedupe keys in the auto namespace for events that were ALREADY true
   when the ledger went live (arm 10's day-one accrual, arms 3/6 first
   chop day), so the first auto pass cannot re-stamp them with deploy
   day's date. Idempotent by dedupe_key — runs harmlessly on every boot.

NB: gate-metric labels are part of the auto dedupe keys. Rewording a
label in ARM_GATES makes the event look new and it will re-fire once —
harmless, but keep labels stable when possible.
"""

import logging
from datetime import datetime, timezone

from backend.database import SessionLocal, ProgramMilestone

logger = logging.getLogger(__name__)

CATEGORIES = {"experiment", "verdict", "gate", "decision", "fix", "infra", "research"}

# Label of the generic >=5-closed-sells metric compute_experiment_gates appends
# to every arm. It gates whether the weekly A/B decision rule can run at all;
# it is not any arm's promotion gate. Kept as the dedupe-key label so rows
# already recorded under it (arms 8/9, 2026-09-01) never re-fire.
SUFFICIENCY_LABEL = "closed sells (weekly-email gate)"


def is_sufficiency_row(row) -> bool:
    """True for auto rows written for the generic sufficiency metric."""
    return f":{SUFFICIENCY_LABEL}:" in (getattr(row, "dedupe_key", None) or "")


def add_milestone(db, *, title, occurred_at=None, category="research",
                  detail=None, source="claude", dedupe_key=None):
    """Insert one milestone; returns the row, or None when the dedupe_key
    already exists (event previously recorded)."""
    if category not in CATEGORIES:
        category = "research"
    if dedupe_key:
        existing = db.query(ProgramMilestone).filter(
            ProgramMilestone.dedupe_key == dedupe_key).first()
        if existing:
            return None
    row = ProgramMilestone(
        occurred_at=occurred_at or datetime.now(timezone.utc),
        category=category,
        title=title,
        detail=detail,
        source=source,
        dedupe_key=dedupe_key,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def record_auto_milestones(db):
    """Diff current gate state into the ledger. Returns rows inserted."""
    # Lazy import: routes.admin imports this module for the manual-entry
    # endpoint; importing at module level would be circular.
    from backend.routes.admin import compute_experiment_gates

    gates = compute_experiment_gates(db)
    now = datetime.now(timezone.utc)
    inserted = 0

    clocks = gates.get("program_clocks", {})
    stop = clocks.get("stop_loss_recheck") or {}
    target = stop.get("target") or 5
    if (stop.get("n") or 0) >= target:
        if add_milestone(
            db, occurred_at=now, category="gate", source="auto",
            dedupe_key="clock:stop_loss_recheck:target",
            title=f"Stop-loss re-check cohort complete ({stop['n']}/{target} clean stops)",
            detail=(f"avg {stop.get('avg_loss_pct')}% vs {stop.get('bar_pct')}% bar — "
                    "mechanical verdict fires from the pre-registered rule."),
        ):
            inserted += 1
    if stop.get("verdict"):
        if add_milestone(
            db, occurred_at=now, category="verdict", source="auto",
            dedupe_key="clock:stop_loss_recheck:verdict",
            title=f"Stop-loss re-check verdict: {stop['verdict']}",
            detail=(f"avg {stop.get('avg_loss_pct')}% vs {stop.get('bar_pct')}% bar, "
                    f"n={stop.get('n')} (owner stops since Jun-24, split artifacts excluded)."),
        ):
            inserted += 1

    for c in clocks.get("calendar", []):
        if c.get("due"):
            if add_milestone(
                db, occurred_at=now, category="gate", source="auto",
                dedupe_key=f"calendar:{c['due_date']}",
                title=f"Calendar clock due: {c['label']}",
                detail=f"Pre-registered re-check date {c['due_date']} reached.",
            ):
                inserted += 1

    for arm in gates.get("arms", []):
        name = arm.get("name") or ""
        short = name.replace("shadow_", "")
        for m in arm.get("gate_metrics", []):
            label, n, tgt = m.get("label"), m.get("n") or 0, m.get("target")
            # Chop days are rare enough that the FIRST accrual is itself a
            # milestone (divergence between chop arms and baseline can only
            # begin on a chop day).
            if label and "chop days" in label and n >= 1:
                if add_milestone(
                    db, occurred_at=now, category="gate", source="auto",
                    dedupe_key=f"gate:{name}:{label}:first",
                    title=f"First chop day accrued for {short} ({n}/{tgt})",
                    detail="SPY closed 0-1.5% above its 50MA inside the arm's active window.",
                ):
                    inserted += 1
            if label and tgt and n >= tgt:
                if m.get("kind") == "sufficiency":
                    # Data-sufficiency threshold, not a gate crossing: the
                    # 2026-09-01 rows titled "'closed sells' accrual met" read
                    # as promotion events and triggered a false verdict read.
                    title = (f"{short}: weekly A/B data sufficiency reached "
                             f"({n}/{tgt}) — not a promotion gate")
                    detail = ("The shadow-vs-baseline decision rule now has enough "
                              "closed sells to evaluate this arm. The arm's own "
                              "pre-registered gate metrics are listed separately "
                              "and are what a verdict waits on.")
                else:
                    title = f"{short}: '{label}' accrual met ({n}/{tgt})"
                    detail = "Gate accrual only — the verdict comes from the weekly A/B decision rule."
                if add_milestone(
                    db, occurred_at=now, category="gate", source="auto",
                    dedupe_key=f"gate:{name}:{label}:target",
                    title=title, detail=detail,
                ):
                    inserted += 1

    return inserted


# One-time historical backfill. occurred_at dates and claims come from the
# session record (memory files + commit history); each entry's dedupe_key
# is stable so re-running the seed is a no-op. Entries in the auto
# namespace (gate:/clock:) deliberately pre-claim keys the auto writer
# would otherwise re-fire with a wrong date.
_SEEDS = [
    # (date, category, title, detail, dedupe_key-suffix or full auto key)
    ("2026-05-07", "experiment", "cs_bear scoring stack shipped to live trading",
     "First consumer of the live A/B framework (cutoff 2026-05-07). Founding rule: "
     "backtests cannot validate scorer changes — forward A/B only.", None),
    ("2026-06-18", "verdict", "Winner-protection study killed",
     "Give-back is the price of let-winners-run; don't tighten winner exits. "
     "Exit-parity queue landed the same day (6a93c7c).", None),
    ("2026-06-24", "experiment", "Stop-loss exit-fix cohort opened",
     "Owner stops from this date form the pre-registered re-check cohort; "
     "verdict fires mechanically at n>=5 vs the -10% bar.", None),
    ("2026-07-04", "verdict", "ML bonus removed; 0.30 veto stays pending kill-or-bless",
     "ML backtests untrustworthy (lookahead). The veto's fate rides on the "
     "shadow_ml_veto_off arm — do not flip it without that verdict.", None),
    ("2026-07-20", "verdict", "max_positions=8 confirmed — concentration IS the alpha",
     "Sweep verdict; pre-registered as do-not-re-sweep.", None),
    ("2026-07-22", "decision", "Beat-SPY program registered (owner gate)",
     "No real-money features until edge vs SPY is statistically proven. Signal "
     "audit the same day: ALL edge from trend days; chop bleeds.", None),
    ("2026-07-24", "experiment", "Exit Lab verdict: exits slightly too tight — wide_trail armed",
     "Stop slippage measured +3.4pp real; shadow_wide_trail (arm 4) activated.", None),
    ("2026-07-25", "fix", "Shadow FIFO rebuild was peak-blind (7b99e5f)",
     "Critical rebuild fix; stop-slippage investigation closed (residual = overnight gaps).", None),
    ("2026-07-29", "verdict", "Big-winner cap killed (5th time) — cap50 armed",
     "Caps below +40% destroy value; don't re-litigate. shadow_cap50 (arm 5) "
     "activated to settle it forward.", None),
    ("2026-07-30", "fix", "Shadow baseline rebased onto cs_bear (9a356fc)",
     "Baseline had drifted from the live stack; pre-Jul-30 baseline data non-comparable.", None),
    ("2026-08-03", "verdict", "Stop-loss fix: WORKED numerically, MARGINAL by pre-reg rule",
     "Re-check clock armed at n>=5. Chop arms damper/chop_spy gates pre-registered "
     "(beat baseline AND chop_damper, >=15 chop days).", None),
    ("2026-08-06", "research", "4-agent full audit: 25 findings, fully dispatched",
     "bb527e4, fd0c108, arm 9. Lesson: don't fan out parallel editing agents in one tree.", None),
    ("2026-08-17", "fix", "Window-dynamic P&L rebaseline (d9f39bd)",
     "All pre-Aug-17 backtest rows stale — same-day baselines only from here.", None),
    ("2026-08-18", "gate", "SECEX gate PASSED — sector_relief armed; ALL ARM CLOCKS RESET",
     "shadow_sector_relief (arm 7) + shadow pyramid mirroring (1d59606). Pre-Aug-18 "
     "shadow data not comparable (backup /root/shadow_stacks_backup_20260818.sql).", None),
    ("2026-08-19", "experiment", "CS confidence v2 recalibrated on 138 outcomes (9e39576)",
     "10-14d best bucket; tiers monotonic. Re-check ~Nov-2026 at ~200 post-Aug-19 outcomes.", None),
    ("2026-08-20", "experiment", "Arms 8/9/10 activated (cs_window14, cs_exempt, ml_veto_off)",
     "edcb7d1 + the ML-veto kill-or-bless arm. Gate Progress card shipped the same day (5c789cc).", None),
    ("2026-08-20", "infra", "CISO security sweep (e5def5e) + refresh-token rotation (fbfdbf4)",
     "Auth clean (0 IDOR), 8 fixes, dependency sweep; single-use jti with family-kill.", None),
    ("2026-08-20", "gate", "ml_veto_off: sub-0.30 buy accrual met on day one (8/5)",
     "All 8 activation-day buys carried ml_confidence < 0.30 — the arm's reason to exist "
     "accrued immediately.", "gate:shadow_ml_veto_off:sub-0.30-confidence buys taken:target"),
    ("2026-08-21", "fix", "SFBS 2:1 split phantom stop — detected and fixed same day (ea4d5d5)",
     "FMP stable/splits detection + rescale; shadow _reconcile_splits followed in 80f327d.", None),
    ("2026-08-24", "gate", "First chop day accrued for chop_damper (1/15)",
     "SPY closed +1.49% above its 50MA — first day inside the 0-1.5% chop band since "
     "the Aug-18 clock reset.", "gate:shadow_chop_damper:chop days:first"),
    ("2026-08-24", "gate", "First chop day accrued for chop_spy (1/15)",
     "Same tape as chop_damper — both Aug-19-reset chop arms started accruing together.",
     "gate:shadow_chop_spy:chop days:first"),
    ("2026-08-25", "research", "Chop-bleed mechanism identified: unrealized drift of HELD names",
     "Not realized whipsaw, not bad chop entries. Beta-hedge ruled out (bleed is residual); "
     "extension trap not supported in realized outcomes.", None),
    ("2026-08-25", "experiment", "Arms 11/12 activated (chop_entry_bar, chop_trim)",
     "Owner: \"ship both\". Divergence from baseline can only begin on a chop day; "
     "gates pre-registered in YAML and mirrored in the backtester.", None),
    ("2026-08-25", "infra", "PM program shipped (80f327d): bootstrap CIs, auto-verdict gates, clocks card",
     "P(edge>0) at ship time: 72.7% overall, 93.6% trend. Verdicts now fire mechanically.", None),
    ("2026-08-26", "infra", "Nightly offsite backup live (canslim-backups repo)",
     "Encrypted dump, round-trip restore verified, ntfy on failure — DR loop closed.", None),
    ("2026-08-26", "fix", "SPY-gate 4h-stale-cache bug fixed (aec4594)",
     "Market direction now force-refreshed per trade cycle; log-verified live Aug-27.", None),
    ("2026-08-27", "fix", "Gate counters made honest (67f96f3)",
     "Stop cohort was reading a u2 leak (true n=2, not 3); arm 11 suppression proxy no "
     "longer counts pre-activation baseline rows.", None),
    ("2026-08-27", "research", "Chop Lab + drift study: chop research CLOSED (185f0fb, edc7928)",
     "Bleed lives in 10-25%-extended held names + trend-day pyramid lots. All chop levers "
     "net-negative on trend-heavy windows — the arms are insurance, expected to trail in "
     "trend weeks. Nothing forecasts next-day drift (n=87). Only the arms' gates remain open.", None),
    ("2026-08-31", "infra", "First Monday email auto-check fired clean",
     "11 emails verified (10 shadow arms + cs_bear KEEP), clocks card present, no failure "
     "alert — silence-equals-pass confirmed manually on the first fire.", None),
    ("2026-08-31", "infra", "Program Ledger went live",
     "Event-sourced milestone history in the admin surface; auto rows fire on gate "
     "threshold crossings, verdicts, and calendar clocks.", None),
]


def seed_history(db):
    """Idempotent backfill of program history. Returns rows inserted."""
    inserted = 0
    for d, category, title, detail, auto_key in _SEEDS:
        # Keyed on date+title (not list index) so editing the seed list
        # never re-fires unrelated entries.
        key = auto_key or f"seed:{d}:{title[:48]}"
        # Noon UTC keeps seeded rows unambiguous across date-only renderers.
        occurred = datetime.fromisoformat(d).replace(hour=12, tzinfo=timezone.utc)
        if add_milestone(db, occurred_at=occurred, category=category, title=title,
                         detail=detail, source="claude", dedupe_key=key):
            inserted += 1
    return inserted


def _notify_new_milestones(rows):
    """Exception ping (2026-09-01 owner policy): a fresh auto milestone —
    gate crossed, verdict fired, calendar clock due — is exactly the event
    the retired weekly-email ritual existed to surface. Push it to the
    owner's devices instead. Fail-soft: a ping failure never fails the pass."""
    if not rows:
        return
    try:
        from backend.email_utils import create_notification
        titles = "\n".join(f"• {r.title}" for r in rows[:5])
        if len(rows) > 5:
            titles += f"\n(+{len(rows) - 5} more)"
        create_notification(
            1,  # owner — the ledger is an owner-only surface
            kind="program_milestone",
            title=(f"Program milestone: {rows[0].title}" if len(rows) == 1
                   else f"{len(rows)} new program milestones"),
            body=titles, priority="high",
            data={"url": "/admin/ab-eval"},
        )
    except Exception as e:
        logger.warning(f"Milestone exception ping failed: {e}")


def run_milestone_pass():
    """Scheduler entrypoint: seed (no-op after first run) then diff gates."""
    from sqlalchemy import func
    db = SessionLocal()
    try:
        # id watermark — anything auto-written above it this pass is new
        last_id = db.query(func.max(ProgramMilestone.id)).scalar() or 0
        seeded = seed_history(db)
        auto = record_auto_milestones(db)
        if seeded or auto:
            logger.info(f"milestone pass: {seeded} seeded, {auto} auto-recorded")
        if auto:
            new_rows = (db.query(ProgramMilestone)
                        .filter(ProgramMilestone.id > last_id,
                                ProgramMilestone.source == "auto")
                        .order_by(ProgramMilestone.id).all())
            # Sufficiency rows stay in the ledger but never ping — the
            # exception-ping channel is for gate crossings and verdicts.
            _notify_new_milestones([r for r in new_rows if not is_sufficiency_row(r)])
        return seeded + auto
    except Exception as e:
        logger.error(f"milestone pass failed: {e}")
        return 0
    finally:
        db.close()
