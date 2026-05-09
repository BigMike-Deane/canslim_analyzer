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
    prior_buy_map: dict = {}
    for t in pre_trades:
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, t.user_id)] = t
    serialized_post = []
    for t in post_trades:
        serialized_post.append(_serialize_trade_row(t, prior_buy_map))
        if t.action == 'BUY':
            prior_buy_map[(t.ticker, t.user_id)] = t

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
