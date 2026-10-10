#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
DATA="${DATA:-$PWD/reports/posttrain_data_v2}"
OUT="${OUT:-$PWD/reports/posttrain_k12_k30_v1}"
ADAPTER="${ADAPTER:-$PWD/reports/posttrain_full_20261008_192810/dpo_adapter}"
exec 9>"$PWD/rag_data/posttrain.lock"
flock -n 9 || { echo 'Another experiment is running'; exit 1; }
if [[ ! -f "$OUT/config.json" ]]; then
  "$PWD/.venv-gla/bin/python" -m region_rag.posttrain_k_compare recall \
    --data "$DATA" --index "$PWD/rag_data/index" --out "$OUT"
fi
.venv-posttrain/bin/python -m region_rag.posttrain_k_compare rerank \
  --data "$DATA" --adapter "$ADAPTER" --out "$OUT"
