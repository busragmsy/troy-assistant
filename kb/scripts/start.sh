#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


echo "🚀 Servisler başlatılıyor..."
$DC up -d pg adminer


echo "⏳ PostgreSQL hazır mı kontrol ediliyor..."
for i in {1..30}; do
if $DC exec -T pg pg_isready -h localhost >/dev/null 2>&1; then
echo "✅ PostgreSQL hazır."
break
fi
sleep 2
if [[ $i -eq 30 ]]; then
echo "⚠️ Zaman aşımı: pg hazır değil." >&2
exit 1
fi
done


echo "📊 Adminer (sunucuda): http://127.0.0.1:8080"
echo "🔗 PG bağlantısı: postgresql://${POSTGRES_USER:-troy}:***@<SUNUCU_IP>:5432/${POSTGRES_DB:-kb}"