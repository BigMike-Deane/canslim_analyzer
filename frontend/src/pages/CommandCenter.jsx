import { useState, useEffect, useCallback, useMemo, memo } from 'react'
import { Link } from 'react-router-dom'
import { api, formatCurrency, formatPercent, formatTime, formatRelativeTime, getScoreClass } from '../api'
import { computePositionSizing } from '../positionSizing'
import { useOwnerPrefs } from '../hooks/useOwnerPrefs'
import PositionHealthChip from '../components/PositionHealthChip'
import Card, { SectionLabel } from '../components/Card'
import { ScoreBadge, OutcomeBadge, ActionBadge, TagBadge, PnlText, CSConfidenceBadge } from '../components/Badge'
import StatGrid from '../components/StatGrid'
import Sparkline from '../components/Sparkline'
import CollapsibleSection from '../components/CollapsibleSection'
import Spinner from '../components/Spinner'
import EmptyState from '../components/EmptyState'
import { useToast } from '../components/Toast'

// Auto-refresh during market hours (M-F 8:30am-4pm CST).
// Gate is evaluated PER TICK, not at effect-mount, so a tab opened at 8:25am
// will start polling at the next tick after 8:30, and one opened at 3:55pm
// will stop firing past 4:00 without needing the user to reload.
function useMarketRefresh(callback, intervalMs = 60000) {
  useEffect(() => {
    const isMarketHours = () => {
      const now = new Date()
      const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Chicago', hour: 'numeric', minute: 'numeric', weekday: 'short', hour12: false }).formatToParts(now)
      const hour = parseInt(parts.find(p => p.type === 'hour')?.value || '0')
      const minute = parseInt(parts.find(p => p.type === 'minute')?.value || '0')
      const weekday = parts.find(p => p.type === 'weekday')?.value || 'Sun'
      const isWeekday = !['Sat', 'Sun'].includes(weekday)
      const totalMins = hour * 60 + minute
      return isWeekday && totalMins >= 510 && totalMins <= 960
    }

    const id = setInterval(() => {
      if (isMarketHours()) callback()
    }, intervalMs)
    return () => clearInterval(id)
  }, [callback, intervalMs])
}

// Binary SPY-vs-50MA gate. The winning strategy (`nostate_optimized` /
// live `nostate_cs_bear`) disables the 5-state machine and gates buys on
// this single signal, so it deserves the dominant header pill.
function BuyDayPill({ active }) {
  if (active === null || active === undefined) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border bg-dark-800 border-dark-700 text-dark-500">
        <span className="w-1.5 h-1.5 rounded-full bg-dark-500" />
        ---
      </span>
    )
  }
  return active ? (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border bg-emerald-500/15 border-emerald-500/40 text-emerald-300">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
      BUY DAY
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border bg-red-500/15 border-red-500/40 text-red-300">
      <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
      DEFENSE DAY
    </span>
  )
}

function IndexRow({ label, data }) {
  if (!data || !data.price) return null
  const aboveMa = data.ma50 ? data.price > data.ma50 : null
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-dark-400 text-xs font-medium w-9">{label}</span>
      <span className="font-data text-xs text-dark-100">{data.price?.toFixed(2)}</span>
      {data.ma50 && (
        <span className={`text-[10px] font-data ${aboveMa ? 'text-emerald-400' : 'text-red-400'}`}>
          {aboveMa ? '>' : '<'} 50MA
        </span>
      )}
    </div>
  )
}

function DefensePriorityCard({ positions }) {
  if (!positions || positions.length === 0) return null
  return (
    <Card as="section" aria-labelledby="cc-defense-heading" variant="accent" accent="red" className="bg-red-500/[0.04] mb-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span id="cc-defense-heading" className="text-sm font-semibold text-red-300">Defensive Priority</span>
          <TagBadge color="red">DEFENSE DAY</TagBadge>
        </div>
        <span className="text-[10px] text-dark-500">closest to stop</span>
      </div>
      <div>
        {positions.map((p, idx) => {
          const stopCls =
            p.stop_distance <= 2 ? 'text-red-400' :
            p.stop_distance <= 5 ? 'text-amber-400' :
            'text-dark-500'
          return (
            <Link
              key={p.ticker}
              to={`/stock/${p.ticker}`}
              className="flex items-center justify-between py-1.5 border-b border-dark-700/30 last:border-0 hover:bg-red-500/5 -mx-1 px-1 rounded transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-[10px] text-dark-500 font-data w-3">{idx + 1}</span>
                <span className="font-medium text-red-300 text-xs w-12">{p.ticker}</span>
                <span className={`text-[10px] font-data shrink-0 ${stopCls}`} title={`${p.stop_distance.toFixed(1)}% from stop`}>
                  stop {p.stop_distance.toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <span className={`text-[10px] font-data ${p.gain_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {p.gain_pct >= 0 ? '+' : ''}{p.gain_pct.toFixed(1)}%
                </span>
                {p.trail_from_peak != null && p.trail_from_peak > 5 && (
                  <span className="text-[10px] font-data text-amber-400" title="Down from peak">
                    ↓{p.trail_from_peak.toFixed(0)}%
                  </span>
                )}
              </div>
            </Link>
          )
        })}
      </div>
      <div className="mt-2 pt-2 border-t border-dark-700/30 text-[10px] text-dark-500">
        Tap a row to review. Defensive moves: tighten stop, trim, or exit.
      </div>
    </Card>
  )
}

function CoiledSpringSection({ cs }) {
  if (!cs) return null
  const { candidates, stats, recent_results } = cs
  const hasData = (candidates?.length > 0) || (stats?.total > 0)
  if (!hasData) return null

  // Collapsed by default (ui-revamp): a count badge carries the glanceable
  // signal; the full candidate list is one tap away. Duplicated surfaces
  // (AI Portfolio section, /coiled-spring page) carry the expanded view.
  return (
    <Card as="section" aria-labelledby="cc-coiled-spring-heading" variant="accent" accent="teal" className="bg-teal-500/[0.03]">
      <CollapsibleSection
        title="Coiled Spring"
        titleId="cc-coiled-spring-heading"
        defaultOpen={false}
        badge={
          <span className="flex items-center gap-1.5">
            <TagBadge color="teal">CATALYST</TagBadge>
            {candidates?.length > 0 && (
              <span className="text-[10px] font-data text-teal-300">{candidates.length} upcoming</span>
            )}
          </span>
        }
      >
      <div className="flex justify-end mb-2">
        <Link to="/coiled-spring/history" className="text-teal-400 text-[10px] hover:text-teal-300 transition-colors">
          History &rarr;
        </Link>
      </div>

      {stats?.total > 0 && (
        <StatGrid
          columns={4}
          className="mb-3"
          stats={[
            { label: 'Win Rate', value: formatPercent(stats.win_rate), color: 'text-teal-300' },
            { label: 'Wins', value: stats.wins, color: 'text-emerald-400' },
            { label: 'Losses', value: stats.losses, color: 'text-red-400' },
            { label: 'Flat', value: stats.flat, color: 'text-yellow-400' },
          ]}
        />
      )}

      {candidates?.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] text-dark-400 uppercase tracking-wider mb-1.5">Upcoming</div>
          {candidates.map(c => (
            <Link
              key={c.ticker}
              to={`/stock/${c.ticker}`}
              className="flex items-center justify-between py-1.5 border-b border-dark-700/30 last:border-0 hover:bg-teal-500/5 -mx-1 px-1 rounded transition-colors"
            >
              <div className="flex items-center gap-2">
                <span className="font-medium text-teal-300 text-xs w-10">{c.ticker}</span>
                {c.base_type && <TagBadge>{c.base_type}</TagBadge>}
              </div>
              <div className="flex items-center gap-2">
                <CSConfidenceBadge confidence={c.confidence} />
                <span className={`text-[10px] font-data ${c.days_to_earnings <= 7 ? 'text-red-400' : 'text-amber-400'}`}>
                  {c.days_to_earnings}d
                </span>
                <span className="text-[10px] text-dark-500 font-data">{c.beat_streak}x</span>
                <ScoreBadge score={c.score} ticker={c.ticker} size="xs" />
              </div>
            </Link>
          ))}
        </div>
      )}

      {recent_results?.length > 0 && (
        <div>
          <div className="text-[10px] text-dark-400 uppercase tracking-wider mb-1.5">Results</div>
          <div className="flex flex-wrap gap-x-3 gap-y-1.5">
            {recent_results.map((r, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-dark-200">{r.ticker}</span>
                <CSConfidenceBadge confidence={r.confidence} />
                <OutcomeBadge outcome={r.outcome} />
                <PnlText value={r.price_change_pct} className="text-[10px]" />
              </div>
            ))}
          </div>
        </div>
      )}
      </CollapsibleSection>
    </Card>
  )
}

// Improving Radar — score-velocity discovery. Backed by the 2026-07-21
// event study on point-in-time StockScore history: sub-65 stocks with a
// fast-rising score lead their static peers by ~1.1pp over the next 14d,
// while FAST risers already at 75+ went negative (extension, not
// emergence) — hence the separate chase-risk block.
function ImprovingRadarSection({ radar }) {
  if (!radar) return null
  const rising = radar.radar || []
  const cautions = radar.fast_risers || []
  if (rising.length === 0 && cautions.length === 0) return null

  const Row = ({ s, caution }) => (
    <Link
      to={`/stock/${s.ticker}`}
      className="flex items-center justify-between gap-2 py-1.5 px-2 -mx-2 rounded hover:bg-dark-750/50 transition-colors border-b border-dark-700/30 last:border-0"
    >
      <div className="flex items-center gap-2 min-w-0">
        <span className="font-medium text-dark-100 text-xs">{s.ticker}</span>
        <span className={`text-[10px] font-data ${caution ? 'text-amber-400' : 'text-emerald-400'}`}
              title={`Score ${s.prior_score} → ${s.score} over ${radar.lookback_days}d`}>
          {s.prior_score} → {s.score} (+{s.velocity})
        </span>
        {s.driver === 'fund_led' && (
          <span
            className="text-[9px] uppercase tracking-wide text-amber-300/90 bg-amber-500/10 border border-amber-500/20 rounded px-1 py-0.5 shrink-0"
            title="Rise driven by a fresh earnings report (C+A jump) — these historically stall for ~4 weeks (+1.7% vs +3.6% for momentum-led rises)"
          >
            EPS pop
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {s.spark?.length > 1 && (
          <Sparkline data={s.spark} width={56} height={18} color={caution ? '#f59e0b' : undefined} />
        )}
        <span className="text-[10px] font-data text-dark-400 w-14 text-right">
          {formatCurrency(s.current_price)}
        </span>
      </div>
    </Link>
  )

  // Collapsed by default (ui-revamp): discovery lens, never a candidate
  // pipeline — the count badge is the glance; expand to browse.
  return (
    <Card as="section" aria-labelledby="cc-radar-heading" variant="accent" accent="purple" className="bg-purple-500/[0.03]">
      <CollapsibleSection
        title="Improving Radar"
        titleId="cc-radar-heading"
        defaultOpen={false}
        badge={
          <span className="flex items-center gap-1.5">
            <TagBadge color="purple">SCORE Δ{radar.lookback_days}D</TagBadge>
            {rising.length > 0 && (
              <span className="text-[10px] font-data text-purple-300">{rising.length} rising</span>
            )}
          </span>
        }
      >
      <div className="text-[10px] text-dark-400 mb-2"
           title="Event-study backed: these beat static peers by ~1-2pp over the next 2-4 weeks. They are NOT a candidate pipeline — fast risers reach the 72+ buy bar LESS often than stable near-bar names (17% vs 29% within 30d); scores mean-revert even while prices run.">
        Score momentum igniting under the buy bar — historically outperforms for 2–4 weeks. Momentum lens, not a candidate pipeline.
      </div>
      {rising.length === 0 ? (
        <EmptyState bare compact message="No fast risers under the bar right now" />
      ) : (
        <div className="max-h-64 overflow-y-auto -mx-1 px-1">
          {rising.map(s => <Row key={s.ticker} s={s} />)}
        </div>
      )}
      {cautions.length > 0 && (
        <div className="mt-2 pt-2 border-t border-dark-700/40">
          <div className="text-[10px] text-amber-400/90 mb-1" title="High-score fast risers underperformed over the next 14d in the event study — a rapid score rise at the top usually reflects the price run-up itself (extension risk)">
            ⚠ Fast risers at 75+ — historically chase risk, not entries
          </div>
          {cautions.map(s => <Row key={s.ticker} s={s} caution />)}
        </div>
      )}
      </CollapsibleSection>
    </Card>
  )
}

// Industry Group Rotation — where money is rotating. Backed by
// /api/industry-groups: rotation.improving / rotation.deteriorating are
// groups whose 3-month RS meaningfully diverges from their 12-month RS
// (|rs_diff| > 0.05). RS values are ratios vs SPY (~1.0 = market-perform),
// so rs_diff is shown ×100 as "RS points" to keep one-decimal legibility.
// Hidden entirely when the one-shot fetch fails or returns nothing.
function GroupRotationSection({ data }) {
  const improving = data?.rotation?.improving || []
  const deteriorating = data?.rotation?.deteriorating || []
  if (improving.length === 0 && deteriorating.length === 0) return null

  const fmtDiff = (d) => {
    const v = (d ?? 0) * 100
    return `${v >= 0 ? '+' : ''}${v.toFixed(1)}`
  }

  const GroupRow = ({ g, tone }) => (
    <div className="flex items-center justify-between gap-2 py-1 border-b border-dark-700/20 last:border-0">
      <span
        className="text-[11px] text-dark-300 truncate min-w-0"
        title={`${g.industry} — group rank ${g.rank}/100 · 3m RS ${g.avg_rs_3m?.toFixed?.(2) ?? g.avg_rs_3m} vs 12m RS ${g.avg_rs_12m?.toFixed?.(2) ?? g.avg_rs_12m}`}
      >
        {g.industry}
      </span>
      <span className="flex items-center gap-1.5 shrink-0">
        <span className={`text-[10px] font-data font-medium ${tone}`}>{fmtDiff(g.rs_diff)}</span>
        <span className="text-[9px] font-data text-dark-500" title={`${g.stock_count} stocks in group`}>
          ({g.stock_count})
        </span>
      </span>
    </div>
  )

  return (
    <Card as="section" aria-labelledby="cc-group-rotation-heading" variant="glass">
      <CollapsibleSection
        title="Group Rotation"
        titleId="cc-group-rotation-heading"
        defaultOpen={false}
        badge={data?.as_of ? (
          <span className="text-[9px] font-data text-dark-500" title={data.as_of}>
            as of {formatRelativeTime(data.as_of)}
          </span>
        ) : undefined}
      >
        <div
          className="text-[10px] text-dark-400 mb-2"
          title="3-month group RS minus 12-month group RS (×100). Positive = the group is outperforming its own longer-term trend — money rotating in. IBD: ~37% of a stock's move comes from its industry group."
        >
          3m vs 12m group relative strength — where money is rotating.
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400 mb-1.5">
              Money Flowing In
            </div>
            {improving.length === 0 ? (
              <div className="text-[10px] text-dark-500 py-1">None</div>
            ) : (
              improving.slice(0, 8).map(g => (
                <GroupRow key={g.industry} g={g} tone="text-emerald-400" />
              ))
            )}
          </div>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-red-400 mb-1.5">
              Money Flowing Out
            </div>
            {deteriorating.length === 0 ? (
              <div className="text-[10px] text-dark-500 py-1">None</div>
            ) : (
              deteriorating.slice(0, 8).map(g => (
                <GroupRow key={g.industry} g={g} tone="text-red-400" />
              ))
            )}
          </div>
        </div>
      </CollapsibleSection>
    </Card>
  )
}

const PositionRow = memo(function PositionRow({ p, earningsDays }) {
  const stopDist = p.stop_distance
  const stopColor =
    stopDist == null ? 'text-dark-600' :
    stopDist <= 2 ? 'text-red-400' :
    stopDist <= 5 ? 'text-amber-400' :
    'text-dark-500'
  return (
    <Link
      to={`/stock/${p.ticker}`}
      className="flex items-center justify-between py-2 px-2 rounded-lg hover:bg-dark-750/50 transition-colors group"
    >
      <div className="flex items-center gap-2 min-w-0">
        <PositionHealthChip position={p} compact />
        <span className="text-xs font-semibold text-primary-400 w-11 shrink-0 group-hover:text-primary-300">{p.ticker}</span>
        <span className="text-[10px] font-data text-dark-500 w-7 shrink-0">{p.position_pct?.toFixed(0)}%</span>
        {stopDist != null && (
          <span
            className={`text-[10px] font-data ${stopColor} shrink-0`}
            title={`${stopDist.toFixed(1)}% from stop`}
          >
            stop {stopDist.toFixed(1)}%
          </span>
        )}
        {earningsDays != null && earningsDays <= 14 && (
          <span
            className={`text-[10px] font-data px-1.5 py-0.5 rounded shrink-0 ${
              earningsDays <= 7 ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'
            }`}
            title={`Earnings in ${earningsDays} day${earningsDays === 1 ? '' : 's'}`}
          >
            E:{earningsDays}d
          </span>
        )}
      </div>
      <div className="flex items-center gap-3 shrink-0">
        <ScoreBadge score={p.score} ticker={p.ticker} size="xs" />
        <span className={`font-data text-xs w-14 text-right ${p.gain_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
          {p.gain_pct >= 0 ? '+' : ''}{p.gain_pct?.toFixed(1)}%
        </span>
      </div>
    </Link>
  )
})

function CandidateRow({ c, portfolio }) {
  // When the portfolio summary is loaded we can render the actionable
  // "30sh @ $24.50" chip in place of the bare price. Sizing is a pure
  // derivation — no network, no extra render cost. Suppressed for
  // non-actionable kinds (extended / below_threshold / no_cash) so the
  // dashboard never invites a chase.
  const sizing = portfolio?.total_value && c.price && c.score
    ? computePositionSizing({
        ticker: c.ticker,
        currentPrice: c.price,
        pivotPrice: c.pivot_price,
        score: c.score,
        cash: portfolio.cash,
        totalValue: portfolio.total_value,
        positionsCount: portfolio.positions_count,
        maxPositions: portfolio.max_positions,
        stopLossPct: portfolio.stop_loss_pct,
        minScore: portfolio.min_score_to_buy,
      })
    : null
  const sizingActionable = sizing?.kind === 'actionable'
  // Owner's CLAUDE.md filter is "stocks under $25 that fit CANSLIM" — surface
  // that as a visual signal so the home screen does the filter pass without
  // requiring a tap into StockDetail. >=$25 stays visible but muted.
  const price = c.price
  const priceColor =
    price == null ? 'text-dark-600' :
    price < 25 ? 'text-emerald-400' :
    'text-dark-400'
  // EarningsAudit fundamental_confidence is 0-100; bucket into tiers that
  // mirror the score-band convention used elsewhere (>=70 strong, 40-69
  // mixed, <40 weak). Single-letter chip keeps the row tight at 390px.
  const conf = c.audit_confidence
  const confTier =
    conf == null ? null :
    conf >= 70 ? { label: 'H', cls: 'bg-emerald-500/10 text-emerald-400' } :
    conf >= 40 ? { label: 'M', cls: 'bg-amber-500/10 text-amber-400' } :
    { label: 'L', cls: 'bg-dark-700/40 text-dark-500' }
  // Distance from pivot using the natural CLAUDE.md convention:
  //   pct < 0      → price below pivot → pre-breakout zone (primary buy)
  //   0 ≤ pct ≤ 5  → active breakout (just broken out, still actionable)
  //   pct > 5      → extended (chasing, scoring-penalized)
  // Null when no base/pivot has been detected for this candidate yet.
  const pivot = c.pivot_price
  const pivotPct = (pivot != null && pivot > 0 && price != null)
    ? ((price - pivot) / pivot) * 100
    : null
  const pivotTier =
    pivotPct == null ? null :
    pivotPct >= -3 && pivotPct <= 0 ? { label: `${pivotPct.toFixed(1)}%`,  cls: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30' } :
    pivotPct > 0 && pivotPct <= 5   ? { label: `+${pivotPct.toFixed(1)}%`, cls: 'bg-amber-500/15 text-amber-300 border border-amber-500/30' } :
    pivotPct > 5                    ? { label: `+${pivotPct.toFixed(1)}%`, cls: 'bg-red-500/15 text-red-300 border border-red-500/30' } :
    null  // pivotPct < -3 → too far below pivot to matter today, suppress
  return (
    <Link
      to={`/stock/${c.ticker}`}
      className="flex items-center justify-between py-2 px-2 rounded-lg hover:bg-dark-750/50 transition-colors group"
    >
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-xs font-semibold text-primary-400 w-11 shrink-0 group-hover:text-primary-300">{c.ticker}</span>
        <span className="text-[10px] text-dark-500 truncate max-w-[60px] shrink-0">{c.sector?.split(' ')[0]}</span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <ScoreBadge score={c.score} ticker={c.ticker} size="xs" />
        {c.projected_growth > 0 && (
          <span className="text-emerald-400 text-[10px] font-data shrink-0">
            +{c.projected_growth?.toFixed(0)}%
          </span>
        )}
        {sizingActionable ? (
          <span
            className="text-[10px] font-data shrink-0 px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
            title={`Suggested: BUY ${sizing.shares} ${c.ticker} @ $${sizing.limitPrice.toFixed(2)} (stop $${sizing.stopPrice.toFixed(2)}, risk ${sizing.riskPct}% of portfolio)`}
          >
            {sizing.shares}sh@${sizing.limitPrice.toFixed(2)}
          </span>
        ) : price != null && (
          <span
            className={`text-[10px] font-data shrink-0 ${priceColor}`}
            title={price < 25 ? 'Under $25 — matches owner filter' : `$${price.toFixed(2)}`}
          >
            ${price.toFixed(2)}
          </span>
        )}
        {confTier && (
          <span
            className={`text-[9px] font-data px-1.5 py-0.5 rounded shrink-0 ${confTier.cls}`}
            title={`Audit confidence: ${conf.toFixed(0)}`}
          >
            {confTier.label}
          </span>
        )}
        {pivotTier && (
          <span
            className={`text-[9px] font-data px-1.5 py-0.5 rounded shrink-0 ${pivotTier.cls}`}
            title={
              pivotPct < 0  ? `Pre-breakout — ${Math.abs(pivotPct).toFixed(1)}% below pivot $${pivot.toFixed(2)}` :
              pivotPct <= 5 ? `Active breakout — ${pivotPct.toFixed(1)}% above pivot $${pivot.toFixed(2)}` :
                              `Extended — ${pivotPct.toFixed(1)}% above pivot $${pivot.toFixed(2)} (chasing)`
            }
          >
            {pivotTier.label}
          </span>
        )}
      </div>
    </Link>
  )
}

export default function CommandCenter() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [runningAction, setRunningAction] = useState(null)
  const toast = useToast()
  const { matchesPrefs } = useOwnerPrefs()

  const [radar, setRadar] = useState(null)
  // Breadth salvage (UI audit 2026-07-22): the Breadth PAGE is retired —
  // these two glances (A/D ratio, new highs/lows) move here. Fetched once
  // per visit; the endpoint aggregates all Stock rows so it stays off the
  // 60s market-hours poll.
  const [breadth, setBreadth] = useState(null)
  useEffect(() => {
    api.getMarketBreadth().then(setBreadth).catch(() => null)
  }, [])
  // Industry group rotation — same one-shot pattern as breadth: the endpoint
  // recomputes group rankings across all Stock rows, so it deliberately stays
  // off the 60s market-hours poll. Section hides itself when this stays null.
  const [industryGroups, setIndustryGroups] = useState(null)
  useEffect(() => {
    api.getIndustryGroups().then(setIndustryGroups).catch(() => null)
  }, [])

  const fetchData = useCallback(async () => {
    try {
      const [main, radarData] = await Promise.all([
        api.getCommandCenter(),
        api.getImprovingRadar().catch(() => null),
      ])
      setData(main)
      setRadar(radarData)
      setLastUpdate(new Date())
      setError(null)
    } catch (e) {
      setError(e?.message || 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchData() }, [fetchData])
  useMarketRefresh(fetchData)

  // Hooks must run before any early return. Reads data?.earnings so it
  // tolerates the loading state without changing hook count between renders.
  const earningsByTicker = useMemo(() => {
    const m = {}
    const earnings = data?.earnings
    if (Array.isArray(earnings)) {
      for (const e of earnings) {
        if (e?.ticker != null) m[e.ticker] = e.days
      }
    }
    return m
  }, [data?.earnings])

  const handleAction = async (action) => {
    setRunningAction(action)
    try {
      if (action === 'cycle') {
        await api.runAITradingCycle()
        toast.success('Trading cycle complete')
      }
      if (action === 'scan') {
        await api.startScanner('all', 35)
        toast.info('Scan started')
      }
      fetchData()
    } catch (e) {
      toast.error(e.message || `Failed to ${action}`)
    } finally {
      setRunningAction(null)
    }
  }

  if (loading && !data) {
    return (
      <div className="p-4 md:p-6">
        <div className="skeleton h-8 w-48 mb-5 rounded-lg" />
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 md:gap-4">
          <div className="md:col-span-3 space-y-3">
            <div className="skeleton h-40 rounded-xl" />
            <div className="skeleton h-28 rounded-xl" />
          </div>
          <div className="md:col-span-5 space-y-3">
            <div className="skeleton h-64 rounded-xl" />
            <div className="skeleton h-64 rounded-xl" />
          </div>
          <div className="md:col-span-4 space-y-3">
            <div className="skeleton h-48 rounded-xl" />
            <div className="skeleton h-36 rounded-xl" />
            <div className="skeleton h-36 rounded-xl" />
          </div>
        </div>
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="p-4 md:p-6">
        <Card className="text-center py-12">
          <div className="text-red-400 text-sm mb-2">Failed to load Command Center</div>
          <div className="text-dark-500 text-xs mb-4">{error}</div>
          <button onClick={fetchData} className="btn-primary text-xs">Retry</button>
        </Card>
      </div>
    )
  }

  const { market, portfolio, sparkline, positions, candidates, risk, earnings, trades, scanner, coiled_spring } = data || {}
  const spy = market?.spy
  const buyDayActive = (spy?.price != null && spy?.ma50 != null)
    ? spy.price > spy.ma50
    : null
  // Defense Day playbook — when the binary gate flips to defense, rank
  // positions by ascending stop_distance (closest-to-stop first), top 3.
  // Filter stop_distance <= 0 (already past stop, should be sold).
  // Strict `=== false` check keeps the panel suppressed on data-missing
  // renders (where buyDayActive is null).
  const defensivePositions = (buyDayActive === false && Array.isArray(positions))
    ? positions
        .filter(p => p.stop_distance != null && p.stop_distance > 0)
        .sort((a, b) => a.stop_distance - b.stop_distance)
        .slice(0, 3)
    : []
  const strategyName = portfolio?.strategy || 'balanced'
  const strategyLabel = strategyName.replace(/_/g, ' ')

  return (
    <div className="p-4 md:p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-4 md:mb-5">
        <div className="flex items-center gap-2 md:gap-3 min-w-0">
          <h1 className="text-base md:text-lg font-bold text-dark-50 shrink-0">Command Center</h1>
          <div className="flex items-center gap-2">
            {portfolio?.paper_mode && <TagBadge color="amber">PAPER</TagBadge>}
            <BuyDayPill active={buyDayActive} />
          </div>
        </div>
        <div className="flex items-center gap-2 md:gap-3 shrink-0">
          {/* Desktop Quick Actions */}
          <div className="hidden md:flex items-center gap-2">
            <button
              onClick={() => handleAction('cycle')}
              disabled={!!runningAction}
              className="text-[10px] font-medium px-3 py-1.5 rounded-lg bg-primary-600/15 text-primary-400 border border-primary-500/20 hover:bg-primary-600/25 transition-colors disabled:opacity-50"
            >
              {runningAction === 'cycle' ? <span className="inline-flex items-center gap-1.5"><Spinner size="xs" inline />Running…</span> : 'Run Cycle'}
            </button>
            <button
              onClick={() => handleAction('scan')}
              disabled={!!runningAction || scanner?.is_scanning}
              className="text-[10px] font-medium px-3 py-1.5 rounded-lg bg-dark-700 text-dark-300 border border-dark-600 hover:bg-dark-600 transition-colors disabled:opacity-50"
            >
              {runningAction === 'scan' ? <span className="inline-flex items-center gap-1.5"><Spinner size="xs" inline />Starting…</span> : 'Start Scan'}
            </button>
          </div>
          {/* Scanner status folded into the header (ui-revamp): a chip when
              idle; the progress card below only appears mid-scan. */}
          <span className="hidden sm:flex items-center gap-1.5 text-[10px] font-data">
            {scanner?.is_scanning ? (
              <span className="flex items-center gap-1.5 text-primary-400">
                <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-pulse-dot" />
                scan {scanner.stocks_scanned != null ? `${scanner.stocks_scanned}/${scanner.total_stocks}` : '…'}
              </span>
            ) : (
              <span className="text-dark-500" title={scanner?.last_scan_end ? `Last scan ${formatRelativeTime(scanner.last_scan_end)}` : ''}>
                scanner idle
              </span>
            )}
          </span>
          <span
            className="text-[10px] text-dark-500 font-data"
            title={lastUpdate ? formatTime(lastUpdate.toISOString()) : ''}
          >
            {lastUpdate ? `Updated ${formatRelativeTime(lastUpdate.toISOString())}` : ''}
          </span>
        </div>
      </header>

      {/* In-flight scan progress — full detail only while it matters */}
      {scanner?.is_scanning && (
        <Card variant="glass" padding="px-4 py-3" className="mb-3">
          <div className="flex items-center justify-between text-[10px] font-data text-dark-400">
            <span className="truncate">{scanner.phase_label || scanner.current_phase || 'scanning'}</span>
            <span className="ml-2 shrink-0">
              {scanner.phase_total > 0
                ? `${scanner.phase_current || 0}/${scanner.phase_total}`
                : `${scanner.stocks_scanned || 0}/${scanner.total_stocks || '…'}`}
            </span>
          </div>
          <div className="mt-2 h-1 bg-dark-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500/60 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, ((scanner.phase_total > 0 ? (scanner.phase_current || 0) / scanner.phase_total : (scanner.stocks_scanned || 0) / (scanner.total_stocks || 1))) * 100)}%` }}
            />
          </div>
        </Card>
      )}

      {/* ═══════════════════════════════════════════
          MOBILE LAYOUT: optimized card order
          Portfolio → Positions → Market → Candidates → Catalysts → rest
          ═══════════════════════════════════════════ */}

      {/* Mobile-only: Portfolio hero card */}
      <div className="md:hidden mb-3">
        <Card as="section" aria-labelledby="cc-mobile-portfolio-heading" variant="glass">
          <div className="flex items-center justify-between mb-2">
            <SectionLabel id="cc-mobile-portfolio-heading">Portfolio</SectionLabel>
            <span className="text-[10px] text-dark-500 capitalize font-medium">{strategyLabel}</span>
          </div>
          <div className="flex items-baseline gap-3 mb-2">
            <span className={`text-2xl font-bold font-data ${
              portfolio?.total_return >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {formatCurrency(portfolio?.total_value)}
            </span>
            <PnlText
              value={portfolio?.total_return_pct}
              className="text-sm"
              prefix={portfolio?.total_return_pct >= 0 ? '+' : ''}
              decimals={2}
            />
          </div>
          {sparkline && sparkline.length > 1 && (
            <div className="relative mb-2">
              <Sparkline data={sparkline} width={320} height={36} gradient className="w-full" />
              {/* Window label: the sparkline is the last 30 days, which can
                  run red while the all-time return above it is green —
                  labeling the window keeps that from reading as a bug. */}
              <span className="absolute top-0 right-0 text-[9px] font-data text-dark-500">30d</span>
            </div>
          )}
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-dark-700/30">
            <div>
              <div className="text-[10px] text-dark-500">Cash</div>
              <div className="text-xs font-data font-medium text-dark-200">{formatCurrency(portfolio?.cash)}</div>
            </div>
            <div>
              <div className="text-[10px] text-dark-500">Invested</div>
              <div className="text-xs font-data font-medium text-dark-200">{formatCurrency(portfolio?.invested)}</div>
            </div>
            <div>
              <div className="text-[10px] text-dark-500">Positions</div>
              <div className="text-xs font-data font-medium text-dark-200">{portfolio?.positions_count}/{portfolio?.max_positions}</div>
            </div>
          </div>
        </Card>
      </div>

      {/* Mobile-only: Quick Actions */}
      <div className="flex gap-2 md:hidden mb-3">
        <button
          onClick={() => handleAction('cycle')}
          disabled={!!runningAction}
          className="flex-1 text-xs font-medium py-2.5 rounded-lg bg-primary-600/15 text-primary-400 border border-primary-500/20 hover:bg-primary-600/25 transition-colors disabled:opacity-50"
        >
          {runningAction === 'cycle' ? 'Running...' : 'Run Cycle'}
        </button>
        <button
          onClick={() => handleAction('scan')}
          disabled={!!runningAction || scanner?.is_scanning}
          className="flex-1 text-xs font-medium py-2.5 rounded-lg bg-dark-700 text-dark-300 border border-dark-600 hover:bg-dark-600 transition-colors disabled:opacity-50"
        >
          {runningAction === 'scan' ? 'Starting...' : 'Start Scan'}
        </button>
      </div>

      <DefensePriorityCard positions={defensivePositions} />

      {/* ═══════════════════════════════════════════
          DESKTOP LAYOUT: 3-column grid
          ═══════════════════════════════════════════ */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 md:gap-4">

        {/* ═══ LEFT COLUMN (desktop only for market/portfolio/risk) ═══ */}
        <div className="md:col-span-3 space-y-3">
          {/* Market Regime */}
          <Card as="section" aria-labelledby="cc-market-heading" variant="glass" animate stagger={1}>
            <SectionLabel id="cc-market-heading">Market</SectionLabel>
            <div className="space-y-0.5">
              <IndexRow label="SPY" data={market?.spy} />
              <IndexRow label="QQQ" data={market?.qqq} />
              <IndexRow label="DIA" data={market?.dia} />
            </div>
            {breadth && (
              <div className="mt-2 pt-2 border-t border-dark-700/30 flex items-center justify-between text-[10px]"
                   title="Market breadth: advancers/decliners ratio and 52-week new highs vs new lows across the scanned universe (formerly the Breadth page)">
                <span className="text-dark-500">Breadth</span>
                <span className="font-data text-dark-300">
                  A/D {breadth.ad_ratio ?? '—'} · NH {breadth.new_highs ?? '—'} / NL {breadth.new_lows ?? '—'}
                </span>
              </div>
            )}
            {market?.weighted_signal != null && (
              <div className="mt-2 pt-2 border-t border-dark-700/30 flex items-center justify-between">
                <span className="text-[10px] text-dark-500">Signal</span>
                <span className={`text-xs font-data font-medium ${
                  market.weighted_signal > 0.5 ? 'text-emerald-400' :
                  market.weighted_signal < -0.5 ? 'text-red-400' : 'text-dark-300'
                }`}>
                  {market.weighted_signal?.toFixed(2)}
                </span>
              </div>
            )}
          </Card>

          {/* Portfolio Summary (desktop) */}
          <Card as="section" aria-labelledby="cc-portfolio-heading" variant="glass" animate stagger={2} className="hidden md:block">
            <div className="flex items-center justify-between mb-1">
              <SectionLabel id="cc-portfolio-heading">Portfolio</SectionLabel>
              <span className="text-[10px] text-dark-500 capitalize font-medium">{strategyLabel}</span>
            </div>
            <div className={`text-2xl font-bold font-data mb-1 ${
              portfolio?.total_return >= 0 ? 'text-emerald-400 glow-green' : 'text-red-400 glow-red'
            }`}>
              {formatCurrency(portfolio?.total_value)}
            </div>
            <div className="flex items-center gap-2 mb-3">
              <PnlText
                value={portfolio?.total_return_pct}
                className="text-xs"
                prefix={portfolio?.total_return_pct >= 0 ? '+' : ''}
                decimals={2}
              />
              <span className="text-dark-600">|</span>
              <span className="text-xs font-data text-dark-400">
                {formatCurrency(portfolio?.total_return)}
              </span>
            </div>

            {sparkline && sparkline.length > 1 && (
              <div className="mb-3 relative">
                <Sparkline data={sparkline} width={200} height={40} gradient className="w-full" />
                <span className="absolute top-0 right-0 text-[9px] font-data text-dark-500">30d</span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-dark-700/30">
              <div>
                <div className="text-[10px] text-dark-500">Cash</div>
                <div className="text-xs font-data font-medium text-dark-200">{formatCurrency(portfolio?.cash)}</div>
              </div>
              <div>
                <div className="text-[10px] text-dark-500">Positions</div>
                <div className="text-xs font-data font-medium text-dark-200">{portfolio?.positions_count}/{portfolio?.max_positions}</div>
              </div>
            </div>
          </Card>

          {/* Risk (desktop) */}
          <Card as="section" aria-labelledby="cc-risk-heading" variant="glass" animate stagger={3} className="hidden md:block">
            <SectionLabel id="cc-risk-heading">Risk</SectionLabel>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-dark-400">Heat</span>
              <span className={`text-sm font-data font-semibold ${
                risk?.heat_status === 'normal' ? 'text-emerald-400' :
                risk?.heat_status === 'warning' ? 'text-amber-400' : 'text-red-400'
              }`}>
                {risk?.portfolio_heat?.toFixed(1)}%
              </span>
            </div>
            {risk?.top_sectors?.slice(0, 3).map(s => (
              <div key={s.sector} className="flex justify-between text-[11px] py-1 border-b border-dark-700/20 last:border-0">
                <span className="text-dark-400 truncate max-w-[100px]">{s.sector}</span>
                <span className="font-data text-dark-300">{formatPercent(s.pct)}</span>
              </div>
            ))}
          </Card>
        </div>

        {/* ═══ CENTER COLUMN ═══ */}
        <div className="md:col-span-5 space-y-3">
          {/* Positions */}
          <Card as="section" aria-labelledby="cc-positions-heading" variant="glass" animate stagger={2}>
            <CollapsibleSection
              title="Positions"
              titleId="cc-positions-heading"
              badge={<span className="text-[10px] font-data text-dark-500">{positions?.length || 0}</span>}
            >
              <div className="max-h-80 overflow-y-auto -mx-1">
                {(!positions || positions.length === 0) && (
                  <EmptyState bare compact message="No active positions" hint="The AI portfolio holds no open positions right now." />
                )}
                {positions?.map(p => (
                  <PositionRow
                    key={p.ticker}
                    p={p}
                    earningsDays={earningsByTicker[p.ticker]}
                  />
                ))}
              </div>
              {positions?.length > 0 && (
                <Link to="/ai-portfolio" className="block text-center text-[10px] text-primary-400 hover:text-primary-300 mt-2 pt-2 border-t border-dark-700/30 transition-colors">
                  View All in AI Portfolio &rarr;
                </Link>
              )}
            </CollapsibleSection>
          </Card>

          {/* Top Candidates */}
          {(() => {
            const visibleCandidates = (candidates || []).filter(matchesPrefs)
            const hiddenByPrefs = (candidates?.length || 0) - visibleCandidates.length
            return (
          <Card as="section" aria-labelledby="cc-candidates-heading" variant="glass" animate stagger={3}>
            <CollapsibleSection
              title="Top Candidates"
              titleId="cc-candidates-heading"
              badge={
                <span className="text-[10px] font-data text-dark-500">
                  {visibleCandidates.length}
                  {hiddenByPrefs > 0 && (
                    <span
                      className="ml-1 text-amber-400/80"
                      title={`${hiddenByPrefs} hidden by Owner Preferences — edit in Settings`}
                    >
                      ({hiddenByPrefs} hidden)
                    </span>
                  )}
                </span>
              }
            >
              <div className="max-h-72 overflow-y-auto -mx-1">
                {visibleCandidates.length === 0 && (
                  <EmptyState
                    bare compact
                    message={hiddenByPrefs > 0 ? 'All candidates hidden by Owner Preferences' : 'No candidates above threshold'}
                    hint={hiddenByPrefs > 0 ? 'Adjust filters in Settings to widen the view.' : 'No stocks currently clear the buy score. Try a fresh scan.'}
                  />
                )}
                {visibleCandidates.map(c => <CandidateRow key={c.ticker} c={c} portfolio={portfolio} />)}
              </div>
              {candidates?.length > 0 && (
                <Link to="/screener" className="block text-center text-[10px] text-primary-400 hover:text-primary-300 mt-2 pt-2 border-t border-dark-700/30 transition-colors">
                  View All in Screener &rarr;
                </Link>
              )}
            </CollapsibleSection>
          </Card>
            )
          })()}
        </div>

        {/* ═══ RIGHT COLUMN ═══ */}
        <div className="md:col-span-4 space-y-3">
          {/* Coiled Spring */}
          <div className="animate-fade-in-up opacity-0 stagger-3">
            <CoiledSpringSection cs={coiled_spring} />
          </div>

          {/* Improving Radar — score-velocity discovery */}
          <div className="animate-fade-in-up opacity-0 stagger-4">
            <ImprovingRadarSection radar={radar} />
          </div>

          {/* Industry Group Rotation — one-shot fetch, hidden when unavailable */}
          <div className="animate-fade-in-up opacity-0 stagger-4">
            <GroupRotationSection data={industryGroups} />
          </div>

          {/* Earnings Countdown */}
          <Card as="section" aria-labelledby="cc-earnings-heading" variant="glass" animate stagger={4}>
            <CollapsibleSection
              title="Earnings"
              titleId="cc-earnings-heading"
              defaultOpen={false}
              badge={earnings?.length > 0 ? (
                <span className="text-[10px] font-data text-dark-500">
                  {earnings.length} upcoming{earnings.some(e => e.days <= 7) ? ' · one <7d' : ''}
                </span>
              ) : undefined}
            >
              {(!earnings || earnings.length === 0) ? (
                <EmptyState bare compact message="No upcoming earnings" />
              ) : (
                <div className="space-y-0.5">
                  {earnings.slice(0, 6).map(e => (
                    <div key={e.ticker} className="flex items-center justify-between py-1.5">
                      <Link to={`/stock/${e.ticker}`} className="text-xs font-medium text-primary-400 hover:text-primary-300 transition-colors">
                        {e.ticker}
                      </Link>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-data px-1.5 py-0.5 rounded ${
                          e.days <= 7 ? 'bg-red-500/10 text-red-400' :
                          e.days <= 14 ? 'bg-amber-500/10 text-amber-400' :
                          'bg-dark-700 text-dark-400'
                        }`}>
                          {e.days}d
                        </span>
                        {e.beat_streak > 0 && (
                          <span className="text-[10px] font-data text-dark-500">{e.beat_streak}x</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CollapsibleSection>
          </Card>

          {/* Recent Trades */}
          <Card as="section" aria-labelledby="cc-trades-heading" variant="glass" animate stagger={5}>
            <CollapsibleSection
              title="Trades"
              titleId="cc-trades-heading"
              defaultOpen={false}
              badge={trades?.length > 0 ? (
                <span className="text-[10px] font-data text-dark-500">
                  last {trades[0]?.executed_at ? formatRelativeTime(trades[0].executed_at) : '—'}
                </span>
              ) : undefined}
            >
              {(!trades || trades.length === 0) ? (
                <EmptyState bare compact message="No recent trades" />
              ) : (
                <>
                  <div className="space-y-0.5">
                    {trades.slice(0, 6).map((t, i) => (
                      <div key={i} className="flex items-center justify-between py-1.5">
                        <div className="flex items-center gap-2">
                          <ActionBadge action={t.action} />
                          <Link to={`/stock/${t.ticker}`} className="text-xs font-medium text-primary-400 hover:text-primary-300 transition-colors">
                            {t.ticker}
                          </Link>
                        </div>
                        <div className="flex items-center gap-2">
                          {t.realized_gain != null && (
                            <span className={`text-[10px] font-data font-medium ${t.realized_gain >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {t.realized_gain >= 0 ? '+' : ''}{formatCurrency(t.realized_gain)}
                            </span>
                          )}
                          <span className="text-[10px] font-data text-dark-500">
                            {t.executed_at ? formatRelativeTime(t.executed_at) : '-'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <Link to="/analytics" className="block text-center text-[10px] text-primary-400 hover:text-primary-300 mt-2 pt-2 border-t border-dark-700/30 transition-colors">
                    View All Trades &rarr;
                  </Link>
                </>
              )}
            </CollapsibleSection>
          </Card>

          {/* Mobile Risk (collapsed by default) */}
          <Card as="section" aria-labelledby="cc-mobile-risk-heading" variant="glass" className="md:hidden">
            <CollapsibleSection title="Risk" titleId="cc-mobile-risk-heading" defaultOpen={false}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-dark-400">Heat</span>
                <span className={`text-sm font-data font-semibold ${
                  risk?.heat_status === 'normal' ? 'text-emerald-400' :
                  risk?.heat_status === 'warning' ? 'text-amber-400' : 'text-red-400'
                }`}>
                  {risk?.portfolio_heat?.toFixed(1)}%
                </span>
              </div>
              {risk?.top_sectors?.slice(0, 3).map(s => (
                <div key={s.sector} className="flex justify-between text-[11px] py-1 border-b border-dark-700/20 last:border-0">
                  <span className="text-dark-400 truncate max-w-[100px]">{s.sector}</span>
                  <span className="font-data text-dark-300">{formatPercent(s.pct)}</span>
                </div>
              ))}
            </CollapsibleSection>
          </Card>

          {/* Scanner status lives in the page header now (ui-revamp); the
              in-flight progress card renders under the header during scans. */}

        </div>
      </div>
    </div>
  )
}
