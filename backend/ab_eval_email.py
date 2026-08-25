"""Weekly A/B-eval snapshot email.

Renders a board-style HTML readout of /api/admin/strategy-ab-eval +
/strategy-ab-eval-trades for the active scoring experiment (Approach 2,
cutoff 2026-05-07). Built so the operator gets a verdict-and-deltas summary
on Monday mornings without logging into the admin page during the
2026-05-07 → 2026-06-18 evaluation window.

Calls the AB-eval helpers (_resolve_ab_window, _summarize_window,
_compute_delta, _decide) directly rather than the FastAPI endpoints —
no auth/HTTP overhead inside the scheduler process, and helpers raise
HTTPException(400/404) which the test endpoint propagates verbatim.
"""

import html
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from backend.email_utils import send_email
from backend.routes.admin import (
    _build_warnings,
    _compute_delta,
    _decide,
    _resolve_ab_window,
    _serialize_trade_row,
    _summarize_window,
    _trade_scope_id,
    compute_experiment_gates,
)

logger = logging.getLogger(__name__)


_DECISION_STYLE = {
    'keep':              {'bg': '#10b981', 'label': 'KEEP'},
    'revert':            {'bg': '#ef4444', 'label': 'REVERT'},
    'marginal':          {'bg': '#f59e0b', 'label': 'MARGINAL'},
    'insufficient_data': {'bg': '#6b7280', 'label': 'INSUFFICIENT DATA'},
}


def _fmt_pct(v, digits=2):
    if v is None:
        return '–'
    sign = '+' if v > 0 else ''
    return f'{sign}{v:.{digits}f}%'


def _fmt_num(v, digits=4):
    if v is None:
        return '–'
    return f'{v:.{digits}f}'


def _fmt_int(v):
    if v is None:
        return '–'
    return f'{int(v):,}'


def _delta_color(v, good_is_positive=True):
    if v is None or v == 0:
        return '#9ca3af'
    is_good = (v > 0) if good_is_positive else (v < 0)
    return '#10b981' if is_good else '#ef4444'


def _row(label, pre_val, post_val, delta_val, formatter=_fmt_pct, good_is_positive=True):
    delta_color = _delta_color(delta_val, good_is_positive)
    return f'''
        <tr style="border-bottom:1px solid #e5e7eb;">
          <td style="padding:8px 12px;color:#374151;font-size:13px;">{label}</td>
          <td style="padding:8px 12px;text-align:right;font-family:monospace;font-size:13px;color:#111827;">{formatter(pre_val)}</td>
          <td style="padding:8px 12px;text-align:right;font-family:monospace;font-size:13px;color:#111827;">{formatter(post_val)}</td>
          <td style="padding:8px 12px;text-align:right;font-family:monospace;font-size:13px;color:{delta_color};font-weight:bold;">{formatter(delta_val)}</td>
        </tr>'''


def _trade_rows(trades: list, label: str) -> str:
    """Render a top-N table of post-cutoff SELLs. Caller pre-sorts the list."""
    if not trades:
        return f'''
        <h3 style="margin:24px 0 8px 0;color:#111827;font-size:14px;">{label}</h3>
        <p style="color:#6b7280;font-size:13px;font-style:italic;margin:0;">
          No closed SELLs in the post-cutoff window yet — re-run when more trades have exited.
        </p>'''
    rows = ''
    for t in trades:
        realized_pct = t.get('realized_pct')
        color = _delta_color(realized_pct)
        ticker = html.escape(str(t.get('ticker') or '?'))
        executed = html.escape(str((t.get('executed_at') or '')[:10]))
        reason = html.escape(str(t.get('reason') or ''))[:60]
        hold_days = t.get('hold_days')
        rows += f'''
          <tr style="border-bottom:1px solid #e5e7eb;">
            <td style="padding:6px 10px;font-family:monospace;font-size:12px;color:#111827;">{ticker}</td>
            <td style="padding:6px 10px;font-family:monospace;font-size:12px;color:#374151;">{executed}</td>
            <td style="padding:6px 10px;text-align:right;font-family:monospace;font-size:12px;color:{color};font-weight:bold;">{_fmt_pct(realized_pct)}</td>
            <td style="padding:6px 10px;text-align:right;font-family:monospace;font-size:12px;color:#374151;">{_fmt_int(hold_days)}d</td>
            <td style="padding:6px 10px;font-size:12px;color:#6b7280;">{reason}</td>
          </tr>'''
    return f'''
        <h3 style="margin:24px 0 8px 0;color:#111827;font-size:14px;">{label}</h3>
        <table style="width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e7eb;border-radius:6px;overflow:hidden;">
          <thead>
            <tr style="background:#f9fafb;">
              <th style="padding:8px 10px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Ticker</th>
              <th style="padding:8px 10px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Date</th>
              <th style="padding:8px 10px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Realized %</th>
              <th style="padding:8px 10px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Hold</th>
              <th style="padding:8px 10px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Reason</th>
            </tr>
          </thead>
          <tbody>{rows}
          </tbody>
        </table>'''


def build_ab_eval_snapshot_html(
    strategy: str,
    cutoff_date: str,
    db: Session,
    pre_window_days: int = 30,
    post_window_days: Optional[int] = None,
    exclude_pyramids: bool = True,
    source: str = "live",
) -> dict:
    """Build the snapshot subject + HTML + plain-text bodies.

    Returns a dict with keys: subject, html, text, decision, post_sell_count,
    starting_value_reference. The decision/sell-count fields are surfaced so
    callers (the test endpoint, the scheduler, and tests) can assert on the
    state without re-parsing the HTML.

    `source` is passed through to `_resolve_ab_window`: 'live' (default) reads
    AIPortfolioTrade rows; 'shadow' reads ShadowTrade rows for the named
    ShadowStrategy. The subject and body get a `[Shadow]` prefix/badge when
    source != 'live' so the operator can tell at a glance which delivery was
    the live verdict and which were shadow stacks.
    """
    win = _resolve_ab_window(
        strategy, cutoff_date, pre_window_days, post_window_days,
        exclude_pyramids, db, source=source,
    )
    pre_trades = win['pre_trades']
    post_trades = win['post_trades']
    starting_value = win['starting_value']
    pre_window = win['pre_window']
    post_window = win['post_window']

    pre_summary = _summarize_window(pre_trades, win['pre_window_days'], starting_value)
    post_summary = _summarize_window(post_trades, win['post_window_days'], starting_value)
    delta = _compute_delta(pre_summary, post_summary)
    decision = _decide(pre_summary, post_summary, delta, {
        'min_return_delta_pp': -5.0,
        'min_sharpe_delta': 0.0,
        'min_post_sells': 5,
    })
    warnings = _build_warnings(pre_window, post_window, pre_summary, post_summary)

    # Per-trade rows for the best/worst tables — same path the dashboard uses,
    # via the shared serializer so realized_pct / hold_days math stays identical.
    # Key on _trade_scope_id, not t.user_id — ShadowTrade rows have no
    # user_id column and scope by shadow_strategy_id instead.
    prior_buy_map: dict = {}
    for t in pre_trades:
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, _trade_scope_id(t))] = t
    serialized_post = []
    for t in post_trades:
        serialized_post.append(_serialize_trade_row(t, prior_buy_map))
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, _trade_scope_id(t))] = t

    sells = [r for r in serialized_post if r['action'] == 'SELL' and r['realized_pct'] is not None]
    best_trades = sorted(sells, key=lambda r: r['realized_pct'], reverse=True)[:5]
    worst_trades = sorted(sells, key=lambda r: r['realized_pct'])[:5]

    style = _DECISION_STYLE.get(decision['decision'], _DECISION_STYLE['insufficient_data'])
    decision_label = style['label']
    decision_bg = style['bg']
    decision_reason = html.escape(decision['decision_reason'])

    # Window descriptors, used for the title strip and the side-by-side table.
    pre_range = f"{pre_window['start']} → {pre_window['end']} ({pre_window['days']}d)"
    post_range = f"{post_window['start']} → {post_window['end']} ({post_window['days']}d)"

    pre_sells = (pre_summary.get('realized_sell_pct') or {}).get('n', 0)
    post_sells = (post_summary.get('realized_sell_pct') or {}).get('n', 0)

    rows_html = ''.join([
        _row('Total return',     pre_summary.get('total_return_pct'),         post_summary.get('total_return_pct'),         delta.get('total_return_pct_delta')),
        _row('Capital eff.',     pre_summary.get('capital_efficiency_pct'),   post_summary.get('capital_efficiency_pct'),   delta.get('capital_efficiency_pct_delta')),
        _row('Sharpe / trade',   pre_summary.get('sharpe_per_trade'),         post_summary.get('sharpe_per_trade'),         delta.get('sharpe_per_trade_delta'),         formatter=_fmt_num),
        _row('Realized DD',      pre_summary.get('realized_max_drawdown_pct'),post_summary.get('realized_max_drawdown_pct'),delta.get('realized_max_drawdown_pct_delta'), good_is_positive=False),
        _row('Entries / day',    pre_summary.get('entry_rate_per_day'),       post_summary.get('entry_rate_per_day'),       delta.get('entry_rate_per_day_delta'),        formatter=_fmt_num),
        _row('Exits / day',      pre_summary.get('exit_rate_per_day'),        post_summary.get('exit_rate_per_day'),        delta.get('exit_rate_per_day_delta'),         formatter=_fmt_num),
    ])

    warnings_html = ''
    if warnings:
        items = ''.join(f'<li style="margin-bottom:4px;">{html.escape(w)}</li>' for w in warnings)
        warnings_html = f'''
        <div style="margin-top:16px;padding:12px;background:#fef3c7;border-left:3px solid #f59e0b;border-radius:4px;">
          <div style="font-weight:bold;color:#78350f;font-size:13px;margin-bottom:6px;">Caveats</div>
          <ul style="margin:0;padding-left:20px;color:#78350f;font-size:12px;">{items}</ul>
        </div>'''

    best_html = _trade_rows(best_trades, 'Top 5 post-cutoff SELLs (best)')
    worst_html = _trade_rows(worst_trades, 'Top 5 post-cutoff SELLs (worst)')

    strategy_safe = html.escape(strategy)
    cutoff_safe = html.escape(cutoff_date)
    is_shadow = source == "shadow"
    source_badge_html = ''
    if is_shadow:
        source_badge_html = (
            '<span style="display:inline-block;margin-left:8px;padding:2px 8px;'
            'background:#7c3aed;color:#fff;border-radius:4px;font-size:11px;'
            'font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">'
            'Shadow</span>'
        )

    html_body = f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:600px;margin:0 auto;padding:20px;">

    <div style="background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;border:1px solid #e5e7eb;">
      <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">A/B Evaluation Snapshot</div>
      <h1 style="margin:0;font-size:18px;color:#111827;">{strategy_safe} · cutoff {cutoff_safe}{source_badge_html}</h1>
      <p style="margin:4px 0 0 0;font-size:12px;color:#6b7280;">Pre {pre_range} · Post {post_range}</p>
    </div>

    <div style="background:{decision_bg};color:#fff;border-radius:8px;padding:18px;margin-bottom:16px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;opacity:0.9;margin-bottom:4px;">Decision</div>
      <div style="font-size:22px;font-weight:bold;margin-bottom:8px;">{decision_label}</div>
      <div style="font-size:13px;line-height:1.5;opacity:0.95;">{decision_reason}</div>
    </div>

    <div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:16px;border:1px solid #e5e7eb;">
      <h2 style="margin:0 0 12px 0;font-size:14px;color:#111827;">Side-by-side</h2>
      <div style="font-size:11px;color:#6b7280;margin-bottom:8px;">
        starting_value_reference: ${starting_value:,.2f} · pre SELLs={pre_sells} · post SELLs={post_sells}
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#f9fafb;">
            <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Metric</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Pre</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Post</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Δ</th>
          </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
      </table>
      {warnings_html}
    </div>

    <div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:16px;border:1px solid #e5e7eb;">
      {best_html}
      {worst_html}
    </div>

    <p style="font-size:11px;color:#9ca3af;text-align:center;margin:16px 0 0 0;">
      Generated by CANSLIM Analyzer · /api/admin/strategy-ab-eval
    </p>
  </div>
</body>
</html>'''

    return_delta = delta.get('total_return_pct_delta')
    sharpe_delta = delta.get('sharpe_per_trade_delta')
    source_text_label = "[Shadow] " if is_shadow else ""
    text_body = (
        f"{source_text_label}A/B Evaluation Snapshot — {strategy} · cutoff {cutoff_date}\n"
        f"Decision: {decision_label}\n"
        f"  {decision['decision_reason']}\n\n"
        f"Pre window:  {pre_range} ({pre_sells} SELLs)\n"
        f"Post window: {post_range} ({post_sells} SELLs)\n"
        f"Starting value reference: ${starting_value:,.2f}\n\n"
        f"Total return delta:  {_fmt_pct(return_delta)}\n"
        f"Sharpe per-trade delta: {_fmt_num(sharpe_delta)}\n"
    )

    subject_prefix = "[Shadow] " if is_shadow else ""
    subject = f"{subject_prefix}CANSLIM A/B [{decision_label}] {strategy} · cutoff {cutoff_date}"

    return {
        'subject': subject,
        'html': html_body,
        'text': text_body,
        'decision': decision['decision'],
        'post_sell_count': post_sells,
        'pre_sell_count': pre_sells,
        'starting_value_reference': round(starting_value, 2),
        'return_delta': return_delta,
        'sharpe_delta': sharpe_delta,
        'source': source,
    }


SHADOW_BASELINE_NAME = 'shadow_baseline'

# Shadow stacks are judged against the contemporaneous baseline stack, not
# against their own (empty) pre-history, so the decision labels read as
# pacing-vs-baseline rather than keep/revert-the-change.
_SHADOW_VS_BASELINE_STYLE = {
    'keep':              {'bg': '#10b981', 'label': 'ON PACE'},
    'revert':            {'bg': '#ef4444', 'label': 'LAGGING'},
    'marginal':          {'bg': '#f59e0b', 'label': 'MIXED'},
    'insufficient_data': {'bg': '#6b7280', 'label': 'INSUFFICIENT DATA'},
}


def _naive_utc(dt):
    """DB timestamps are timezone-naive UTC; strip tzinfo from anything
    aware so in-Python comparisons never mix naive and aware."""
    if dt is not None and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _gates_card_html(db: Session, shadow_name: str) -> str:
    """Pre-registered clocks card (PM program 2026-08-25): the same
    compute_experiment_gates payload the ABEval dashboard card renders,
    inlined into the weekly email so verdict-ready clocks announce
    themselves instead of waiting on a manual curl. Fail-soft — a
    monitoring email must always send, so any error renders nothing.
    """
    import html as _html
    try:
        gates = compute_experiment_gates(db)
    except Exception as exc:
        logger.warning(f"gates card skipped for {shadow_name}: {exc}")
        return ''

    def _pill(label, color):
        return (f'<span style="display:inline-block;padding:2px 8px;'
                f'background:{color};color:#fff;border-radius:4px;'
                f'font-size:11px;font-weight:600;">{label}</span>')

    def _tr(label, detail, pill):
        return (f'<tr>'
                f'<td style="padding:6px 12px;font-size:12px;color:#111827;">{_html.escape(label)}</td>'
                f'<td style="padding:6px 12px;font-size:12px;color:#6b7280;text-align:right;">{detail}</td>'
                f'<td style="padding:6px 12px;text-align:right;">{pill}</td>'
                f'</tr>')

    rows = []
    clock = (gates.get('program_clocks') or {}).get('stop_loss_recheck') or {}
    if clock:
        n = clock.get('n') or 0
        target = clock.get('target') or 5
        avg = clock.get('avg_loss_pct')
        verdict = clock.get('verdict')
        detail = (f"avg {avg:+.2f}% vs {clock.get('bar_pct')}% bar"
                  if avg is not None else 'no clean stops yet')
        if verdict == 'PASS':
            pill = _pill('PASS', '#059669')
        elif verdict == 'FAIL':
            pill = _pill('FAIL', '#dc2626')
        else:
            pill = _pill(f'{n}/{target}', '#6b7280')
        rows.append(_tr(clock.get('label') or 'Stop-loss re-check', detail, pill))

    # Date-based calendar clocks: pre-registered re-check dates self-report.
    for c in (gates.get('program_clocks') or {}).get('calendar') or []:
        if c.get('due'):
            pill = _pill('DUE', '#dc2626')
            detail = f"was due {c.get('due_date')}"
        else:
            pill = _pill(f"in {c.get('days_until')}d", '#6b7280')
            detail = f"due {c.get('due_date')}"
        rows.append(_tr(c.get('label') or '', detail, pill))

    arm = next((a for a in gates.get('arms') or []
                if a.get('name') == shadow_name), None)
    if arm:
        for gm in arm.get('gate_metrics') or []:
            n = gm.get('n') or 0
            target = gm.get('target')
            if target:
                pill = _pill(f'{n}/{target}', '#059669' if n >= target else '#6b7280')
                detail = 'accrued' if n >= target else 'accruing'
            else:
                pill = _pill(str(n), '#6b7280')
                detail = 'no target'
            rows.append(_tr(gm.get('label') or '', detail, pill))

    if not rows:
        return ''
    return f"""
    <div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:16px;border:1px solid #e5e7eb;">
      <h2 style="margin:0 0 12px 0;font-size:14px;color:#111827;">Pre-registered clocks</h2>
      <table style="width:100%;border-collapse:collapse;">
        <tbody>{''.join(rows)}
        </tbody>
      </table>
      <div style="font-size:11px;color:#9ca3af;margin-top:8px;">
        Same numbers as the ABEval Gate Progress card (one computation, two
        surfaces). Green = accrual target met; PASS/FAIL fires mechanically
        from the pre-registered rule.
      </div>
    </div>"""


def build_shadow_vs_baseline_snapshot_html(
    shadow_name: str,
    db: Session,
    baseline_name: str = SHADOW_BASELINE_NAME,
) -> dict:
    """Build a shadow-experiment snapshot: experiment vs baseline stack over
    the SAME calendar window.

    A pre/post-cutoff comparison is structurally wrong for shadow stacks —
    they have no trades before activation, so any 'pre' window is empty by
    construction and a delta against it is fabricated. The honest read is
    contemporaneous: both stacks' trades from the later of the two clock
    starts (first recorded trade — resets wipe trade history, so this is
    also the post-reset restart) through now. Promotion remains gated on
    each experiment's pre-registered criteria; this snapshot is monitoring.

    Raises ValueError when either stack is missing/archived, when asked to
    compare the baseline to itself, or when neither stack has traded yet
    (transient post-reset state — the caller's per-stack error isolation
    logs it without killing the fan-out).
    """
    from sqlalchemy import and_, or_
    from backend.database import ShadowStrategy, ShadowTrade

    def _resolve_stack(name, role):
        ss = db.query(ShadowStrategy).filter(ShadowStrategy.name == name).first()
        if ss is None:
            raise ValueError(f"{role} shadow strategy '{name}' not found")
        if ss.archived_at is not None:
            raise ValueError(f"{role} shadow strategy '{name}' is archived")
        return ss

    exp = _resolve_stack(shadow_name, "Experiment")
    base = _resolve_stack(baseline_name, "Baseline")
    if exp.id == base.id:
        raise ValueError("Refusing to compare the baseline stack against itself")

    def _stack_trades(ss):
        # Same pyramid exclusion as _resolve_ab_window: drop action='PYRAMID'
        # and BUY rows whose reason marks a pyramid add; keep NULL reasons.
        # 'SPY SWEEP' rows are cash-parking, not strategy trades — counting
        # them inflates sell counts toward the min_post_sells gate and
        # dilutes per-trade stats with ~0% round-trips. Their realized P&L
        # is surfaced separately as a caveat line.
        # 'SPLIT' rows are corporate-action bookkeeping (see shadow_trader
        # _SPLIT_REASON_PREFIX) — never strategy trades.
        return db.query(ShadowTrade).filter(
            ShadowTrade.shadow_strategy_id == ss.id,
            ~ShadowTrade.action.in_(['PYRAMID', 'SPLIT']),
            or_(
                ShadowTrade.reason.is_(None),
                and_(
                    ~ShadowTrade.reason.like('PYRAMID:%'),
                    ~ShadowTrade.reason.like('SPY SWEEP%'),
                ),
            ),
        ).order_by(ShadowTrade.executed_at).all()

    exp_trades_all = _stack_trades(exp)
    base_trades_all = _stack_trades(base)

    clock_starts = [
        _naive_utc(rows[0].executed_at)
        for rows in (exp_trades_all, base_trades_all) if rows
    ]
    if not clock_starts:
        raise ValueError(
            f"Neither '{shadow_name}' nor '{baseline_name}' has any trades "
            f"yet — no common window to compare."
        )
    window_start = max(clock_starts)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_days = max((now - window_start).days, 1)

    def _split(rows):
        prior = [t for t in rows if _naive_utc(t.executed_at) < window_start]
        window = [t for t in rows if _naive_utc(t.executed_at) >= window_start]
        return prior, window

    exp_prior, exp_window = _split(exp_trades_all)
    _, base_window = _split(base_trades_all)

    base_summary = _summarize_window(
        base_window, window_days, float(base.starting_value or 25000.0))
    exp_summary = _summarize_window(
        exp_window, window_days, float(exp.starting_value or 25000.0))
    # _compute_delta is post-minus-pre; feeding (baseline, experiment) makes
    # every delta experiment-minus-baseline, which is the sign we want.
    delta = _compute_delta(base_summary, exp_summary)
    decision = _decide(base_summary, exp_summary, delta, {
        'min_return_delta_pp': -5.0,
        'min_sharpe_delta': 0.0,
        'min_post_sells': 5,
    })

    base_sells = (base_summary.get('realized_sell_pct') or {}).get('n', 0)
    exp_sells = (exp_summary.get('realized_sell_pct') or {}).get('n', 0)

    warnings = []
    if window_days < 21:
        warnings.append(
            f"Common window is only {window_days} days — verdict clocks are "
            f"young; expect INSUFFICIENT DATA until more trades close."
        )
    if base_sells < 10:
        warnings.append(
            f"Baseline has only {base_sells} closed SELLs in the window — "
            f"baseline statistics unstable."
        )
    if exp_sells < 10:
        warnings.append(
            f"Experiment has only {exp_sells} closed SELLs in the window — "
            f"experiment statistics unstable."
        )

    # SPY sweep rows are excluded from the stats above; surface their
    # realized P&L so a sweeping stack's SPY drift stays visible. The
    # promotion gates read portfolio values, which include it either way.
    sweep_sells = db.query(ShadowTrade).filter(
        ShadowTrade.shadow_strategy_id == exp.id,
        ShadowTrade.action == 'SELL',
        ShadowTrade.reason.like('SPY SWEEP%'),
        ShadowTrade.executed_at >= window_start,
    ).all()
    if sweep_sells:
        sweep_pnl = sum(float(t.realized_gain or 0) for t in sweep_sells)
        warnings.append(
            f"SPY sweep excluded from trade stats: {len(sweep_sells)} sweep "
            f"SELL(s) realized ${sweep_pnl:+,.0f} in the window (cash-parking; "
            f"counted in the promotion gate's portfolio values, not here)."
        )

    # Best/worst tables for the EXPERIMENT stack — same serializer path as
    # the live snapshot; prior BUYs (pre-window history, possible when the
    # experiment's own clock predates the baseline's) seed the map first.
    prior_buy_map: dict = {}
    for t in exp_prior:
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, _trade_scope_id(t))] = t
    serialized_window = []
    for t in exp_window:
        serialized_window.append(_serialize_trade_row(t, prior_buy_map))
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, _trade_scope_id(t))] = t

    sells = [r for r in serialized_window
             if r['action'] == 'SELL' and r['realized_pct'] is not None]
    best_trades = sorted(sells, key=lambda r: r['realized_pct'], reverse=True)[:5]
    worst_trades = sorted(sells, key=lambda r: r['realized_pct'])[:5]

    style = _SHADOW_VS_BASELINE_STYLE.get(
        decision['decision'], _SHADOW_VS_BASELINE_STYLE['insufficient_data'])
    decision_label = style['label']
    decision_bg = style['bg']
    decision_reason = html.escape(decision['decision_reason'])

    window_range = (
        f"{window_start.date().isoformat()} → {now.date().isoformat()} "
        f"({window_days}d, both stacks)"
    )

    rows_html = ''.join([
        _row('Total return',   base_summary.get('total_return_pct'),          exp_summary.get('total_return_pct'),          delta.get('total_return_pct_delta')),
        _row('Capital eff.',   base_summary.get('capital_efficiency_pct'),    exp_summary.get('capital_efficiency_pct'),    delta.get('capital_efficiency_pct_delta')),
        _row('Sharpe / trade', base_summary.get('sharpe_per_trade'),          exp_summary.get('sharpe_per_trade'),          delta.get('sharpe_per_trade_delta'),          formatter=_fmt_num),
        _row('Realized DD',    base_summary.get('realized_max_drawdown_pct'), exp_summary.get('realized_max_drawdown_pct'), delta.get('realized_max_drawdown_pct_delta'), good_is_positive=False),
        _row('Entries / day',  base_summary.get('entry_rate_per_day'),        exp_summary.get('entry_rate_per_day'),        delta.get('entry_rate_per_day_delta'),        formatter=_fmt_num),
        _row('Exits / day',    base_summary.get('exit_rate_per_day'),         exp_summary.get('exit_rate_per_day'),         delta.get('exit_rate_per_day_delta'),         formatter=_fmt_num),
    ])

    warnings_html = ''
    if warnings:
        items = ''.join(f'<li style="margin-bottom:4px;">{html.escape(w)}</li>' for w in warnings)
        warnings_html = f'''
        <div style="margin-top:16px;padding:12px;background:#fef3c7;border-left:3px solid #f59e0b;border-radius:4px;">
          <div style="font-weight:bold;color:#78350f;font-size:13px;margin-bottom:6px;">Caveats</div>
          <ul style="margin:0;padding-left:20px;color:#78350f;font-size:12px;">{items}</ul>
        </div>'''

    best_html = _trade_rows(best_trades, 'Top 5 experiment SELLs (best)')
    worst_html = _trade_rows(worst_trades, 'Top 5 experiment SELLs (worst)')
    clocks_html = _gates_card_html(db, shadow_name)

    shadow_safe = html.escape(shadow_name)
    baseline_safe = html.escape(baseline_name)
    source_badge_html = (
        '<span style="display:inline-block;margin-left:8px;padding:2px 8px;'
        'background:#7c3aed;color:#fff;border-radius:4px;font-size:11px;'
        'font-weight:600;letter-spacing:0.05em;text-transform:uppercase;">'
        'Shadow</span>'
    )

    html_body = f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:600px;margin:0 auto;padding:20px;">

    <div style="background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;border:1px solid #e5e7eb;">
      <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Shadow vs Baseline Snapshot</div>
      <h1 style="margin:0;font-size:18px;color:#111827;">{shadow_safe} vs {baseline_safe}{source_badge_html}</h1>
      <p style="margin:4px 0 0 0;font-size:12px;color:#6b7280;">Window {window_range}</p>
    </div>

    <div style="background:{decision_bg};color:#fff;border-radius:8px;padding:18px;margin-bottom:16px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;opacity:0.9;margin-bottom:4px;">Pacing</div>
      <div style="font-size:22px;font-weight:bold;margin-bottom:8px;">{decision_label}</div>
      <div style="font-size:13px;line-height:1.5;opacity:0.95;">{decision_reason}</div>
    </div>

    <div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:16px;border:1px solid #e5e7eb;">
      <h2 style="margin:0 0 12px 0;font-size:14px;color:#111827;">Side-by-side</h2>
      <div style="font-size:11px;color:#6b7280;margin-bottom:8px;">
        baseline SELLs={base_sells} · experiment SELLs={exp_sells}
      </div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#f9fafb;">
            <th style="padding:8px 12px;text-align:left;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Metric</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Baseline</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Experiment</th>
            <th style="padding:8px 12px;text-align:right;font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Δ</th>
          </tr>
        </thead>
        <tbody>{rows_html}
        </tbody>
      </table>
      {warnings_html}
    </div>
{clocks_html}
    <div style="background:#fff;border-radius:8px;padding:16px;margin-bottom:16px;border:1px solid #e5e7eb;">
      {best_html}
      {worst_html}
    </div>

    <p style="font-size:11px;color:#9ca3af;text-align:center;margin:16px 0 0 0;">
      Contemporaneous comparison over the same window — promotion follows each
      experiment's pre-registered gate; this snapshot is monitoring only.<br>
      Generated by CANSLIM Analyzer · shadow_vs_baseline
    </p>
  </div>
</body>
</html>'''

    return_delta = delta.get('total_return_pct_delta')
    sharpe_delta = delta.get('sharpe_per_trade_delta')
    text_body = (
        f"[Shadow] {shadow_name} vs {baseline_name} — window {window_range}\n"
        f"Pacing: {decision_label}\n"
        f"  {decision['decision_reason']}\n\n"
        f"Baseline SELLs:   {base_sells}\n"
        f"Experiment SELLs: {exp_sells}\n\n"
        f"Total return delta (exp - base):  {_fmt_pct(return_delta)}\n"
        f"Sharpe per-trade delta:           {_fmt_num(sharpe_delta)}\n\n"
        f"Promotion follows the experiment's pre-registered gate; this "
        f"snapshot is monitoring only.\n"
    )

    subject = (
        f"[Shadow] CANSLIM A/B [{decision_label}] {shadow_name} vs "
        f"{baseline_name} · since {window_start.date().isoformat()}"
    )

    return {
        'subject': subject,
        'html': html_body,
        'text': text_body,
        'decision': decision['decision'],
        'window_start': window_start.date().isoformat(),
        'window_days': window_days,
        'window_sell_count': exp_sells,
        'baseline_sell_count': base_sells,
        'return_delta': return_delta,
        'sharpe_delta': sharpe_delta,
        'source': 'shadow_vs_baseline',
    }


def send_shadow_vs_baseline_snapshot(
    shadow_name: str,
    db: Session,
    recipient: Optional[str] = None,
    baseline_name: str = SHADOW_BASELINE_NAME,
) -> dict:
    """Build the shadow-vs-baseline snapshot and send via email_utils."""
    snapshot = build_shadow_vs_baseline_snapshot_html(
        shadow_name, db, baseline_name=baseline_name,
    )
    sent = send_email(
        snapshot['subject'],
        snapshot['html'],
        snapshot['text'],
        recipient=recipient,
    )
    from backend.email_utils import RECIPIENT_EMAIL
    resolved_recipient = recipient or RECIPIENT_EMAIL
    return {
        'sent': sent,
        'recipient': resolved_recipient,
        'subject': snapshot['subject'],
        'decision': snapshot['decision'],
        'window_sell_count': snapshot['window_sell_count'],
        'window_start': snapshot['window_start'],
        'source': snapshot['source'],
    }


def send_ab_eval_snapshot(
    strategy: str,
    cutoff_date: str,
    db: Session,
    recipient: Optional[str] = None,
    pre_window_days: int = 30,
    post_window_days: Optional[int] = None,
    exclude_pyramids: bool = True,
    source: str = "live",
) -> dict:
    """Build the snapshot and send via email_utils.send_email.

    Returns a dict with sent/recipient + the underlying snapshot fields, so
    the test endpoint can echo state to the operator without re-running the
    full eval.
    """
    snapshot = build_ab_eval_snapshot_html(
        strategy, cutoff_date, db,
        pre_window_days=pre_window_days,
        post_window_days=post_window_days,
        exclude_pyramids=exclude_pyramids,
        source=source,
    )
    sent = send_email(
        snapshot['subject'],
        snapshot['html'],
        snapshot['text'],
        recipient=recipient,
    )
    # send_email falls back to RECIPIENT_EMAIL when recipient is None — surface
    # the resolved address rather than echoing the optional override so the
    # caller always sees where the mail actually went.
    from backend.email_utils import RECIPIENT_EMAIL
    resolved_recipient = recipient or RECIPIENT_EMAIL
    return {
        'sent': sent,
        'recipient': resolved_recipient,
        'subject': snapshot['subject'],
        'decision': snapshot['decision'],
        'post_sell_count': snapshot['post_sell_count'],
        'source': source,
    }
