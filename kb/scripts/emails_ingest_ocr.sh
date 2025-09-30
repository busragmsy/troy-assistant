#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_dc.sh"


EMAIL_DIR="${1:-kb/data/emails}"
THEME="${2:-Aylik Troy Egitimleri}"


echo "📨 E-posta OCR ingest: $EMAIL_DIR (tema: $THEME)"
$DC --profile worker run --rm kb_worker bash -lc "python kb/ingest/emails_ocr.py --dir '$EMAIL_DIR' --theme '$THEME'"


echo "✅ OCR ingest tamam."