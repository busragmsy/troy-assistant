# kb/scripts/stop.sh
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


echo "🛑 Durduruluyor..."
$DC down
echo "✅ Tamamlandı"
