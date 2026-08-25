#!/bin/bash
# One-time VPS setup for the nightly offsite backup (PM program 2026-08-25).
# Prerequisites:
#   1. Private GitHub repo BigMike-Deane/canslim-backups exists (the VPS's
#      account-level SSH key can then push to it — verified 2026-08-25).
#   2. /root/.canslim_backup_key exists (created 2026-08-25).
# Run ON THE VPS:  bash /opt/canslim_analyzer/scripts/setup_offsite_backup.sh
set -eu
[ -f /root/.canslim_backup_key ] || { echo "FATAL: /root/.canslim_backup_key missing"; exit 1; }

if [ ! -d /root/canslim-backups/.git ]; then
  git clone -q git@github.com:BigMike-Deane/canslim-backups.git /root/canslim-backups
  echo "cloned backup repo"
fi

chmod +x /opt/canslim_analyzer/scripts/backup_offsite.sh

# Nightly 06:10 UTC — after overnight jobs, well before the US open.
( crontab -l 2>/dev/null | grep -v backup_offsite.sh ; \
  echo "10 6 * * * /opt/canslim_analyzer/scripts/backup_offsite.sh >> /var/log/canslim_backup.log 2>&1" ) | crontab -
echo "cron installed:"
crontab -l | grep backup_offsite

echo "running first backup now..."
/opt/canslim_analyzer/scripts/backup_offsite.sh
