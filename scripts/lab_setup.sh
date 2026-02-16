#!/usr/bin/env bash
# Lab Setup: Initialize the optimization state tracker
# Run this ONCE before starting a Ralph Loop optimization session
#
# Usage: bash scripts/lab_setup.sh

set -euo pipefail

STATE_FILE="/tmp/lab_results.json"

cat > "$STATE_FILE" << 'EOF'
{
  "session_start": "$(date -Iseconds)",
  "baseline": null,
  "iterations": [],
  "best": {
    "backtest_id": null,
    "return_pct": null,
    "sharpe": null,
    "max_drawdown": null,
    "win_rate": null,
    "params_snapshot": null
  },
  "target_to_beat": 41.1,
  "max_iterations_without_improvement": 10
}
EOF

# Fix the date in the JSON
ACTUAL_DATE=$(date -Iseconds)
sed -i "s|\$(date -Iseconds)|$ACTUAL_DATE|g" "$STATE_FILE"

echo "Lab optimization state initialized at $STATE_FILE"
echo ""
echo "Next steps:"
echo "  1. Review config/default.yaml lab profile"
echo "  2. Run: bash scripts/run_lab_backtest.sh"
echo "  3. Read the prompt: scripts/optimize_prompt.md"
echo ""
echo "Target to beat: +41.1% (backtest #133)"
