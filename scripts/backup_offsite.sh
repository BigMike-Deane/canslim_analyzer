#!/bin/bash
# Nightly encrypted off-box backup of the irreplaceable CANSLIM tables
# (PM program 2026-08-25). Installed by scripts/setup_offsite_backup.sh;
# runs from root's crontab on the VPS HOST (not the container — scripts/
# is excluded from the Docker image).
#
# What's protected: every table EXCEPT the rebuildable bulk (stock_scores,
# backtest_static_snapshot, backtest_snapshots, backtest_trades,
# backtest_hold_snapshots, stock_data_cache) — i.e. trades, shadow stacks,
# portfolios, users, alerts, market snapshots: the evidence the whole
# beat-SPY program rests on. The dump stays schema-complete (excluded
# tables keep their DDL) so pg_restore recreates a working database.
#
# Restore recipe (on any box with the key):
#   openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 \
#     -pass file:/root/.canslim_backup_key \
#     -in evidence_YYYYMMDD.dump.enc -out evidence.dump
#   pg_restore -U canslim -d canslim --clean --if-exists evidence.dump
# The key lives at /root/.canslim_backup_key on the VPS AND in the owner's
# CLAUDE.local.md — a backup nobody can decrypt protects nothing.
set -u
REPO_DIR=/root/canslim-backups
KEY_FILE=/root/.canslim_backup_key
ENV_FILE=/opt/canslim_analyzer/.env
STAMP=$(date -u +%Y%m%d)
TMP=$(mktemp /tmp/canslim_evidence.XXXXXX)

alert() {
  local url
  url=$(grep '^CANSLIM_WEBHOOK_URL=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '\r')
  [ -n "$url" ] && curl -s -H "Title: CANSLIM offsite backup FAILED" -H "Priority: high" \
    -d "offsite backup FAILED: $1 ($(date -u))" "$url" >/dev/null 2>&1
  echo "BACKUP FAILED: $1" >&2
}

fail() { alert "$1"; rm -f "$TMP"; exit 1; }

[ -f "$KEY_FILE" ] || fail "key file missing at $KEY_FILE"
[ -d "$REPO_DIR/.git" ] || fail "backup repo not initialized (run setup_offsite_backup.sh)"

docker exec canslim-postgres pg_dump -U canslim -d canslim -Fc \
  --exclude-table-data=stock_scores \
  --exclude-table-data=backtest_static_snapshot \
  --exclude-table-data=backtest_snapshots \
  --exclude-table-data=backtest_trades \
  --exclude-table-data=backtest_hold_snapshots \
  --exclude-table-data=stock_data_cache \
  > "$TMP" || fail "pg_dump failed"
[ -s "$TMP" ] || fail "empty dump"

openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt \
  -pass "file:$KEY_FILE" -in "$TMP" -out "$REPO_DIR/evidence_${STAMP}.dump.enc" \
  || fail "encryption failed"
rm -f "$TMP"

cd "$REPO_DIR" || fail "cd $REPO_DIR failed"
# Retention: newest 14 dumps in the working tree (git history keeps older
# blobs; revisit repo size roughly yearly).
ls -1t evidence_*.dump.enc 2>/dev/null | tail -n +15 | xargs -r git rm -q --ignore-unmatch --
git add -A
git -c user.name=canslim-backup -c user.email=backup@canslim.local \
  commit -q -m "evidence ${STAMP}" || true
git push -q origin HEAD || fail "git push failed"
date -u > /root/.canslim_backup_last_success
echo "backup OK: evidence_${STAMP}.dump.enc"
