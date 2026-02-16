#!/usr/bin/env bash
# Run Lab Backtest: Deploy, start backtest, poll until complete, output results
#
# Usage: bash scripts/run_lab_backtest.sh [starting_cash] [start_date] [end_date]
#
# Defaults: $25000, 1 year ago → today
# Requires: ssh access to VPS, jq

set -euo pipefail

VPS="root@100.104.189.36"
VPS_DIR="/opt/canslim_analyzer"
API_BASE="http://100.104.189.36:8001"

STARTING_CASH="${1:-25000}"
END_DATE="${3:-$(date +%Y-%m-%d)}"
START_DATE="${2:-$(date -d '1 year ago' +%Y-%m-%d 2>/dev/null || date -v-1y +%Y-%m-%d)}"

STATE_FILE="/tmp/lab_results.json"

echo "=== LAB BACKTEST ==="
echo "Period: $START_DATE → $END_DATE"
echo "Cash: \$$STARTING_CASH"
echo "Strategy: lab"
echo ""

# Step 1: Commit config changes
echo "[1/5] Committing config changes..."
cd "$(dirname "$0")/.."
if git diff --name-only | grep -q "config/default.yaml"; then
    git add config/default.yaml
    git commit -m "lab: parameter adjustment for optimization

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    echo "  Committed config changes"
else
    echo "  No config changes to commit"
fi

# Step 2: Push and deploy
echo "[2/5] Pushing and deploying..."
git push
ssh "$VPS" "cd $VPS_DIR && git pull && docker-compose down && docker-compose up -d --build" 2>&1 | tail -5
echo "  Deployed"

# Step 3: Wait for container healthy
echo "[3/5] Waiting for container to be healthy..."
for i in $(seq 1 30); do
    if curl -sf "${API_BASE}/health" > /dev/null 2>&1; then
        echo "  Container healthy after ${i}0s"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Container did not become healthy after 5 minutes"
        exit 1
    fi
    sleep 10
done

# Step 4: Start backtest
echo "[4/5] Starting backtest..."
RESPONSE=$(curl -sf -X POST "${API_BASE}/api/backtests" \
    -H "Content-Type: application/json" \
    -d "{
        \"start_date\": \"$START_DATE\",
        \"end_date\": \"$END_DATE\",
        \"starting_cash\": $STARTING_CASH,
        \"stock_universe\": \"all\",
        \"strategy\": \"lab\"
    }")

BACKTEST_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")

if [ -z "$BACKTEST_ID" ]; then
    echo "  ERROR: Failed to start backtest"
    echo "  Response: $RESPONSE"
    exit 1
fi
echo "  Backtest #$BACKTEST_ID started"

# Step 5: Poll for completion
echo "[5/5] Polling for completion (this takes 10-20 minutes)..."
LAST_PROGRESS=""
while true; do
    STATUS=$(curl -sf "${API_BASE}/api/backtests/${BACKTEST_ID}/status" 2>/dev/null || echo '{}')
    BT_STATUS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
    PROGRESS=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('progress',0):.0f}%\")" 2>/dev/null || echo "?%")

    if [ "$PROGRESS" != "$LAST_PROGRESS" ]; then
        echo "  Progress: $PROGRESS ($BT_STATUS)"
        LAST_PROGRESS="$PROGRESS"
    fi

    if [ "$BT_STATUS" = "completed" ]; then
        break
    elif [ "$BT_STATUS" = "failed" ] || [ "$BT_STATUS" = "cancelled" ]; then
        echo "  ERROR: Backtest $BT_STATUS"
        exit 1
    fi
    sleep 30
done

# Get full results
echo ""
echo "=== RESULTS ==="
RESULTS_FILE="/tmp/lab_bt_results.json"
curl -sf "${API_BASE}/api/backtests/${BACKTEST_ID}" > "$RESULTS_FILE"

# Extract key metrics
python3 << PYEOF
import json, sys

with open("$RESULTS_FILE") as f:
    data = json.load(f)

# API returns {backtest: {...}, trades: [...], ...}
bt = data.get('backtest', data)
r = bt
bt_id = bt.get('id', '$BACKTEST_ID')
total_return = r.get('total_return_pct', 0)
spy_return = r.get('spy_return_pct', 0)
sharpe = r.get('sharpe_ratio', 0)
max_dd = r.get('max_drawdown_pct', 0)
win_rate = r.get('win_rate', 0)
total_trades = r.get('total_trades', 0)

vs_spy = total_return - spy_return

print(f"Backtest #{bt_id}")
print(f"  Return:   {total_return:+.1f}%")
print(f"  vs SPY:   {vs_spy:+.1f}%")
print(f"  Sharpe:   {sharpe:.2f}")
print(f"  Max DD:   {max_dd:.1f}%")
print(f"  Win Rate: {win_rate:.1f}%")
print(f"  Trades:   {total_trades}")
print()

# Update state file if exists
import os
state_file = "/tmp/lab_results.json"
if os.path.exists(state_file):
    with open(state_file) as f:
        state = json.load(f)

    iteration = {
        "backtest_id": bt_id,
        "return_pct": total_return,
        "vs_spy": vs_spy,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": total_trades,
    }
    state["iterations"].append(iteration)

    # Update best if improved
    if state["best"]["return_pct"] is None or total_return > state["best"]["return_pct"]:
        state["best"] = {
            "backtest_id": bt_id,
            "return_pct": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
        }
        print(f"  NEW BEST: +{total_return:.1f}%")
    else:
        print(f"  Best remains: +{state['best']['return_pct']:.1f}% (#{state['best']['backtest_id']})")

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  State saved to {state_file}")

PYEOF
