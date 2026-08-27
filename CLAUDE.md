# CANSLIM Analyzer - Project Context

## Overview
A mobile-first web application for CANSLIM stock analysis with React frontend and FastAPI backend, deployed via Docker on a VPS.

**GitHub Repository**: [BigMike-Deane/canslim_analyzer](https://github.com/BigMike-Deane/canslim_analyzer)

## Architecture
- **Cache**: 3-tier (Memory → Redis → DB → API fetch)
- **Deployment**: Docker (3 containers + 3 volumes) on VPS at `/opt/canslim_analyzer`
- **Docker command**: Use `docker-compose` (with hyphen, old version on VPS)
- **Container name**: `canslim-analyzer`
- **Port**: 8001

## CRITICAL Rules
- **Multi-container VPS**: NEVER use `docker rm -f $(docker ps -aq)` — kills Finance Tracker too
- **AI Trader <> Backtester MUST stay in sync**: Every change to `ai_trader.py` trading logic MUST be mirrored in `backtester.py`
- **Asyncio primitives**: MUST create inside async context (not at module level)
- **Database sessions**: Always use try/finally
- **Score details**: Use lowercase keys `score_details["c"]`
- **Timezone**: Add `.replace(tzinfo=timezone.utc)` if naive
- **`scripts/` directory**: NOT copied into Docker image
- **FastAPI routes**: Literal routes BEFORE parameterized routes
- **Backtester universe**: MUST be "all" (winning stocks are mid/small-caps)
- **FMP API**: `/stable/` endpoints only (no batch); `/api/v3/` requires legacy subscription
- **Earnings avoidance**: `avoidance_days` != `allow_buy_days` — different purposes, NOT off-by-one
- **Force push**: User prefers normal push; avoid rewriting shared history

## Configuration System (YAML-based)
- `config/default.yaml` - Base configuration for all environments
- `config/development.yaml` - Dev overrides
- `config/production.yaml` - Production settings
- `CANSLIM_ENV=production` selects environment (default: development)

```python
from config_loader import config
workers = config.get('scanner.workers', default=4)
config.reload()  # Hot-reload
```

Key sections: `scanner`, `cache`, `scoring.canslim`, `scoring.growth_mode`, `market.indexes`, `ai_trader`, `technical`, `api`, `strategy_profiles`, `coiled_spring`

## Strategy System
7 profiles in YAML, configurable via API and frontend.
**Winner**: `nostate_optimized` — market state DISABLED, binary SPY gate only (SPY < 50MA = no buys).

Champion config: min_score=72, max_positions=8, stop_loss=7%, take_profit=75%, seed_count=4, trailing stops (50+: 25%, 30-50: 18%, 20-30: 12%, 10-20: 8%, 5-10: 4%)

Key finding: 5-state market state machine HURTS over full cycles. NoState's binary gate is crude but far more effective (8.6x better over 4yr).

## Admin Diagnostics
- `GET /api/admin/strategy-health` — pre/post-graduation health audit for a strategy.
- `GET /api/admin/strategy-ab-eval` — live A/B comparison framework: pre vs post-cutoff trade summary + decision (keep/revert/marginal/insufficient_data). Used to evaluate scoring-rule experiments shipped to live trading; backtest replay can't honestly evaluate scoring changes (snapshots freeze today's point-in-time scalars). First consumer: Approach 2 (commit `ec73f83`, deployed 2026-05-07).

## Where detailed conventions live
- Frontend patterns (`useApi` hook, UI grammar, setState rules): `frontend/CLAUDE.md`
- Test isolation / dependency-override conventions: `tests/CLAUDE.md`

## Deployment Commands
See CLAUDE.local.md for deployment commands with actual VPS addresses.
```bash
# Standard deploy pattern
ssh root@$VPS_IP 'cd /opt/canslim_analyzer && git pull && docker-compose down && docker-compose up -d --build'

# Start backtest (nostate_optimized)
curl -X POST https://canslim.duckdns.org/api/backtests -H "Content-Type: application/json" \
  -d '{"start_date": "2022-01-01", "end_date": "2026-02-19", "starting_cash": 25000, "stock_universe": "all", "strategy": "nostate_optimized"}'
```

## Testing
```bash
export CANSLIM_ENV=development
python3 -m pytest tests/ -v                           # All tests
```

## Owner's Trading Preferences
- Likes stocks under $25 that fit CANSLIM criteria
- Default scan: All Stocks at 90-minute intervals
- Prefers actionable pre-breakout entries over chasing extended stocks
