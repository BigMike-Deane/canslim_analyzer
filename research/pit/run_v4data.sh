#!/bin/bash
# "v4 data" rebuild (2026-10-07): market caps from FMP's daily series (m1_mcap.py) instead of
# SEC share counts (~6% of companies were off by > 2x; Alphabet absent 2016-mid-2024).
# Rules/models unchanged: every stage is regenerated, then v3/v4/v5 + diagnostics re-run.
# Resumable-ish: each stage is skipped if its done-marker exists in meta/v4data/.
set -u
M=~/canslim_pit_data/meta; D=$M/v4data; mkdir -p $D
cd "$(dirname "$0")"
log() { echo "$1 $(date '+%m-%d %H:%M')" >> $M/v4data_chain.log; }
stage() {  # stage <name> <command...>
  local n=$1; shift
  [ -f $D/$n.done ] && { log "$n already done"; return 0; }
  log "$n start"
  if nice -n 5 "$@" > $M/v4data_$n.log 2>&1; then touch $D/$n.done; log "$n done"
  else log "FAIL $n"; exit 1; fi
}

until grep -q "^done:" $M/m1_mcap.log; do
  pgrep -f "python3 m1_mcap.py" >/dev/null || { log "FAIL mcap fetch died"; exit 1; }
  sleep 60
done
log "mcap fetch complete: $(tail -1 $M/m1_mcap.log)"

if [ ! -f $D/backup.done ]; then
  for f in m3_panel p3_signals p4_signals p5_setups v3_price_features v3_dividends v3_table; do
    [ -f $M/$f.csv.gz ] && cp $M/$f.csv.gz $M/${f}_v3_oct07.csv.gz
  done
  [ -d $M/daily ] && [ ! -d $M/daily_v3_oct07 ] && mv $M/daily $M/daily_v3_oct07
  [ -d $M/v3 ] && [ ! -d $M/v3_results_v3data_oct07 ] && cp -r $M/v3 $M/v3_results_v3data_oct07
  touch $D/backup.done; log "backup done"
fi

stage panel     python3 m3_panel.py --workers 10
stage p3        python3 p3_signals.py --workers 10
stage p4        python3 p4_signals.py --workers 10
stage p5        python3 p5_setups.py --workers 10
stage daily     python3 p7_daily_scores.py --workers 10
stage pricefeat python3 v3_price_features.py --workers 10
stage divfetch  python3 v3_dividends.py fetch
stage divcomp   python3 v3_dividends.py compute
stage assemble  python3 v3_assemble.py
stage placebo3  python3 v3_model.py --placebo
stage v3        python3 v3_model.py
stage v4        python3 v4_model.py
stage v5        python3 v5_model.py
stage diag      python3 v5_diagnostics.py
log "V4DATA_EXIT"
