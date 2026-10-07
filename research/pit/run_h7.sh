#!/bin/bash
# H7 runner — safe to re-run after a restart; every step resumes or skips finished work.
#   cd /mnt/c/Users/bayer/canslim_analyzer && nohup setsid bash research/pit/run_h7.sh > ~/canslim_pit_data/meta/run_h7.out 2>&1 &
# Progress: ~/canslim_pit_data/meta/h7_chain.log
cd "$(dirname "$0")"
M=~/canslim_pit_data/meta
L=$M/h7_chain.log
log() { echo "$* $(date '+%m-%d %H:%M')" >> $L; }

# 1. daily PIT scores (219 chunks). Wait if a run is already going; else resume.
while pgrep -f "p7_daily_scores.py" > /dev/null; do sleep 60; done
rm -f $M/daily/*.tmp.gz
python3 - <<'PY'   # drop chunks a restart may have cut off mid-write
import glob, gzip, os
for f in glob.glob(os.path.expanduser("~/canslim_pit_data/meta/daily/*.csv.gz")):
    try:
        with gzip.open(f, "rt") as fh:
            for _ in fh: pass
    except Exception:
        print("removing truncated", f); os.remove(f)
PY
n=$(ls $M/daily/*.csv.gz 2>/dev/null | wc -l)
if [ "$n" -lt 219 ]; then
  log "daily resume ($n/219 chunks)"
  python3 p7_daily_scores.py --workers 8 >> $M/p7_daily.log 2>&1 || { log "FAIL daily"; exit 1; }
fi
log "daily complete ($(ls $M/daily/*.csv.gz | wc -l)/219)"

# 2. speed test (Q1 2016), once
if [ ! -f $M/h7/speedtest_q1_2016.json ]; then
  log "speedtest start"
  nice -n 5 python3 p7_backtest.py --offset 0 --end 2016-03-31 > $M/h7_speedtest.log 2>&1 || { log "FAIL speedtest"; exit 1; }
  mv $M/h7/vintage_0.json $M/h7/speedtest_q1_2016.json
  log "speedtest done: $(tail -1 $M/h7_speedtest.log)"
fi

# 3. five vintages, two at a time; finished vintages are skipped
for pair in "0 10" "20 30" "40"; do
  for o in $pair; do
    [ -f $M/h7/vintage_$o.json ] && { log "vintage $o already done"; continue; }
    log "vintage $o start"
    ( nice -n 5 python3 p7_backtest.py --offset $o > $M/h7_vintage_$o.log 2>&1 && log "vintage $o done" || log "FAIL vintage $o" ) &
  done
  wait
done
log "H7_EXIT"
