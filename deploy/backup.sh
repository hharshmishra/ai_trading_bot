#!/usr/bin/env bash
# WAL-safe online backup of the BitReinforceX SQLite DB (predictions, outcomes,
# rewards, sessions, news corpus). Uses sqlite3 .backup so it is consistent even
# while the runtime is writing. Schedule via cron, e.g. nightly:
#   0 3 * * *  /home/ubuntu/ai_trading_bot/deploy/backup.sh
# For continuous streaming backup instead, see deploy/litestream.yml.
set -euo pipefail

DB="${BITREINFORCEX_DB:-logs/bitreinforcex.db}"
DEST="${1:-backups}"
KEEP="${BACKUP_KEEP:-14}"

if [ ! -f "$DB" ]; then
  echo "no DB at $DB — nothing to back up" >&2
  exit 0
fi

mkdir -p "$DEST"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="$DEST/bitreinforcex-$TS.db"
sqlite3 "$DB" ".backup '$OUT'"

# retain the most recent $KEEP backups
ls -1t "$DEST"/bitreinforcex-*.db 2>/dev/null | tail -n +"$((KEEP + 1))" | xargs -r rm -f
echo "backup -> $OUT"
