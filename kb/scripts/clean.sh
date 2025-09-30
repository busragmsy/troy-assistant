# kb/scripts/clean.sh
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


echo "⚠️ Container'lar ve 'troy_pg_data' volume'ü silinecek."
read -p "Devam edilsin mi? [y/N] " -r ans
[[ "${ans,,}" == "y" ]] || { echo "İptal."; exit 0; }


$DC down -v || true
docker volume rm -f troy_pg_data || true


echo "✅ Temizlendi."