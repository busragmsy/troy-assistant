# kb/scripts/pdfs_ingest.sh
#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


DOC_DIR="${1:-kb/data/docs}"
DOC_TYPE="${2:-kullanici_kilavuzu}"
DEPT="${3:-GENEL}"


echo "📚 PDF'ler içeri alınıyor: $DOC_DIR (type=$DOC_TYPE dept=$DEPT)"
$DC --profile worker run --rm kb_worker bash -lc "python kb/ingest/process_pdfs.py --dir '$DOC_DIR' --document-type '$DOC_TYPE' --department '$DEPT'"


echo "✅ PDF ingest tamam."