#!/bin/bash
# CANSLIM Daily Report Runner
# Run this script at 9am CST daily

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Run the report
python3 email_report.py >> "$SCRIPT_DIR/logs/report_$(date +%Y%m%d).log" 2>&1

# Optional: Keep only last 30 days of logs
find "$SCRIPT_DIR/logs" -name "report_*.log" -mtime +30 -delete 2>/dev/null
