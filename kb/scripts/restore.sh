# kb/scripts/restore.sh
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


SQL="${1:-}"
if [[ -z "$SQL" ]]; then
echo "Kullanım: $0 /tam/yol/dosya.sql (veya kb/backups/altındaki dosya adı)" >&2
exit 1
fi


BACKUP_DIR="$REPO_DIR/backups"
if [[ ! -f "$SQL" && -f "$BACKUP_DIR/$SQL" ]]; then
SQL="$BACKUP_DIR/$SQL"
fi
if [[ ! -f "$SQL" ]]; then
echo "Dosya bulunamadı: $SQL" >&2
exit 1
fi


echo "⚠️ ${POSTGRES_DB:-kb} DROP/CREATE edilecek. Devam? [y/N]"
read -r ans
[[ "${ans,,}" == "y" ]] || { echo "İptal."; exit 0; }


echo "🗑️ Aktif bağlantılar sonlandırılıyor ve DB siliniyor..."
$DC exec -T pg psql -U "${POSTGRES_USER:-troy}" -d postgres -v ON_ERROR_STOP=1 -c \
"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${POSTGRES_DB:-kb}';
DROP DATABASE IF EXISTS \"${POSTGRES_DB:-kb}\";"


echo "📦 Yeni DB oluşturuluyor..."
$DC exec -T pg psql -U "${POSTGRES_USER:-troy}" -d postgres -v ON_ERROR_STOP=1 -c \
"CREATE DATABASE \"${POSTGRES_DB:-kb}\" ENCODING 'UTF8';"


echo "⬆️ İçeri aktarılıyor: $SQL"
cat "$SQL" | $DC exec -T pg psql -U "${POSTGRES_USER:-troy}" -d "${POSTGRES_DB:-kb}" -v ON_ERROR_STOP=1 -f -


echo "✅ Restore tamam."