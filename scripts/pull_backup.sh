#!/usr/bin/env bash
# Pull the newest CANSLIM DB backup off the VPS to this machine.
#
# Why: all automated backups live on the same VPS they protect (daily 2AM
# pg_dump + rotation in /opt/canslim_analyzer/data/backups). This script is
# the off-box leg. Run it from WSL whenever — weekly is plenty:
#   bash scripts/pull_backup.sh
#
# Keeps the newest LOCAL_KEEP dumps locally, prunes older ones.
# Restore procedure (into a scratch DB, never live — see backend/backup.py
# verify_latest_backup for the automated weekly equivalent on the VPS):
#   docker exec -i canslim-postgres pg_restore -U canslim -d <target_db> \
#     --no-owner < canslim_YYYYMMDD_HHMMSS.dump
set -euo pipefail

VPS="root@100.104.189.36"
REMOTE_DIR="/opt/canslim_analyzer/data/backups"
LOCAL_DIR="/mnt/c/Users/bayer/canslim_backups"
LOCAL_KEEP=4

mkdir -p "$LOCAL_DIR"

newest=$(ssh -o ConnectTimeout=15 "$VPS" "ls -t $REMOTE_DIR/canslim_*.dump | head -1")
[ -n "$newest" ] || { echo "No backups found on VPS" >&2; exit 1; }

base=$(basename "$newest")
if [ -f "$LOCAL_DIR/$base" ]; then
    echo "Already have newest: $base"
else
    echo "Pulling $base ..."
    scp -o ConnectTimeout=15 "$VPS:$newest" "$LOCAL_DIR/"
    echo "Saved $LOCAL_DIR/$base"
fi

# Prune: keep newest LOCAL_KEEP
ls -t "$LOCAL_DIR"/canslim_*.dump 2>/dev/null | tail -n +$((LOCAL_KEEP + 1)) | while read -r old; do
    echo "Pruning old local copy: $(basename "$old")"
    rm -f "$old"
done

echo "Local copies:"
ls -lh "$LOCAL_DIR"/canslim_*.dump
