# kb/scripts/backup.sh
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


BACKUP_DIR="$REPO_DIR/backups"
mkdir -p "$BACKUP_DIR"
TS="$(date +'%Y%m%d_%H%M%S')"
OUT="$BACKUP_DIR/kb_${TS}.sql"


echo "🧩 Dump alınıyor → $OUT"
$DC exec -T pg sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F p -c' > "$OUT"


echo "🧹 30 günden eski yedekler siliniyor..."
find "$BACKUP_DIR" -type f -name 'kb_*.sql' -mtime +30 -print -delete || true


echo "✅ Yedekleme tamam."