#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ACTION="${1:-clean}"
export QUALITY_REVIEW="${QUALITY_REVIEW:-$PWD/region_rag/training_quality_v1.json}"
export DATA="${DATA:-$PWD/reports/posttrain_data_v2}"
export RUN="${RUN:-$PWD/reports/detail_ablation_clean_v1}"
if [[ "$ACTION" == clean ]]; then
  "${POSTTRAIN_PYTHON:-$PWD/.venv-posttrain/bin/python}" -m region_rag.training_quality \
    --data "$DATA" --review "$QUALITY_REVIEW" \
    --out "${CLEAN_OUT:-$PWD/reports/training_quality_v1}" \
    --experiment "${SOURCE_EXPERIMENT:-$PWD/reports/detail_ablation_v1}"
else
  bash tools/run_detail_ablation.sh "$ACTION"
fi
