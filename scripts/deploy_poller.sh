#!/usr/bin/env bash
# Auto-deploy poller — watches origin/deploy and rebuilds when it moves.
#
# Installed on the VPS as /opt/deploy-poller.sh (a copy OUTSIDE the repo
# tree so a deploy can't mutate the running script), cron'd every 10 min.
# This file is the reference copy. Set up 2026-06-11 for the owner's week
# on the road: cloud Claude Code sessions can't SSH to the VPS (Tailscale),
# so deploys go: PR -> merge to `deploy` branch -> this poller ships it.
# Re-armed 2026-08-07 (cron re-added, deploy branch reset to main).
#
# Safety rails:
#   - Lockfile: no concurrent runs.
#   - Kill switch: `touch /opt/canslim_analyzer/.deploy-poller-disabled`.
#   - Freeze guard: before 2026-06-19, refuses to deploy if ai_trader.py
#     or canslim_scorer.py would change (June-18 C-score-cap eval).
#   - Detached checkout of the exact SHA; last deployed SHA recorded in
#     /opt/.last-deployed-sha. To return to manual deploys: remove the
#     cron line and `git checkout main` in /opt/canslim_analyzer.

set -u
REPO=/opt/canslim_analyzer
LOCK=/tmp/canslim-deploy-poller.lock
DISABLE_FLAG=$REPO/.deploy-poller-disabled
SHA_FILE=/opt/.last-deployed-sha
FREEZE_LIFT=2026-06-19

log() { echo "[$(date -u +%FT%TZ)] $*"; }

[ -f "$DISABLE_FLAG" ] && { log "disabled via $DISABLE_FLAG, skipping"; exit 0; }

exec 9>"$LOCK"
if ! flock -n 9; then
    log "another run holds the lock, skipping"
    exit 0
fi

cd "$REPO" || { log "ERROR: cannot cd $REPO"; exit 1; }

git fetch origin deploy --quiet 2>&1 || { log "ERROR: git fetch failed"; exit 1; }
TARGET=$(git rev-parse origin/deploy 2>/dev/null) || { log "no deploy branch on origin, skipping"; exit 0; }
CURRENT=$(cat "$SHA_FILE" 2>/dev/null || git rev-parse HEAD)

if [ "$TARGET" = "$CURRENT" ]; then
    exit 0
fi

log "origin/deploy moved: $CURRENT -> $TARGET"

# Freeze guard: protect the June-18 eval surface until the freeze lifts.
if [ "$(date -u +%F)" \< "$FREEZE_LIFT" ]; then
    for f in backend/ai_trader.py canslim_scorer.py; do
        if ! git diff --quiet "$CURRENT" "$TARGET" -- "$f"; then
            log "FREEZE GUARD: $f changes between $CURRENT and $TARGET before $FREEZE_LIFT — NOT deploying"
            exit 0
        fi
    done
fi

log "deploying $TARGET"
git checkout --detach "$TARGET" --quiet || { log "ERROR: checkout failed"; exit 1; }
docker-compose down && docker-compose up -d --build
RC=$?
if [ $RC -eq 0 ]; then
    echo "$TARGET" > "$SHA_FILE"
    log "deploy OK at $TARGET"
else
    log "ERROR: docker-compose exited $RC — containers may be down, investigate"
fi
exit $RC
