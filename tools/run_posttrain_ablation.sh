#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PYTHON="${POSTTRAIN_PYTHON:-$PWD/.venv-posttrain/bin/python}"
GLA_PYTHON="${GLA_PYTHON:-$PWD/.venv-gla/bin/python}"
DATA="${DATA:-$PWD/reports/posttrain_data_v2}"
RUN="${RUN:-$PWD/reports/posttrain_$(date +%Y%m%d_%H%M%S)}"
ACTION="${1:-all}"
mkdir -p "$RUN"
exec 9>"$PWD/rag_data/posttrain.lock"
flock -n 9 || { echo 'Another post-training script is running'; exit 1; }
if [[ "$ACTION" == prepare ]]; then
  "$GLA_PYTHON" -m region_rag.posttrain_data --abo "${ABO_ROOT:-$HOME/TF_dataset/ABO}" \
    --index "$PWD/rag_data/index" --exclude-queries "$PWD/tools/posttrain_development_queries.json" \
    --out "$DATA" --k "${K:-12}" --train-queries "${TRAIN_QUERIES:-240}" \
    --val-queries "${VAL_QUERIES:-60}" --test-queries "${TEST_QUERIES:-120}" 2>&1 | tee "$RUN/prepare.log"
  exit 0
fi
[[ -f "$DATA/manifest.json" ]] || { echo "Prepare frozen data first: bash tools/run_posttrain_ablation.sh prepare"; exit 1; }
PIXELS="${PIXELS:-128}"
if [[ "$ACTION" == smoke ]]; then
  "$PYTHON" -m region_rag.posttrain_run train --mode sft --data "$DATA" --out "$RUN/sft_adapter" --steps 2 --accum 1 --limit-pairs 6 --pixels "$PIXELS" 2>&1 | tee "$RUN/sft.log"
  "$PYTHON" -m region_rag.posttrain_run train --mode dpo --data "$DATA" --out "$RUN/dpo_adapter" --sft "$RUN/sft_adapter" --steps 2 --accum 1 --limit-pairs 6 --pixels "$PIXELS" 2>&1 | tee "$RUN/dpo.log"
elif [[ "$ACTION" == all ]]; then
  "$PYTHON" -m region_rag.posttrain_run train --mode sft --data "$DATA" --out "$RUN/sft_adapter" --steps "${SFT_STEPS:-90}" --accum "${ACCUM:-8}" --pixels "$PIXELS" 2>&1 | tee "$RUN/sft.log"
  "$PYTHON" -m region_rag.posttrain_run train --mode dpo --data "$DATA" --out "$RUN/dpo_adapter" --sft "$RUN/sft_adapter" --steps "${DPO_STEPS:-90}" --accum "${ACCUM:-8}" --lr 5e-6 --pixels "$PIXELS" 2>&1 | tee "$RUN/dpo.log"
elif [[ "$ACTION" != evaluate ]]; then
  echo 'Usage: run_posttrain_ablation.sh prepare|smoke|all|evaluate'; exit 1
fi
for ARM in baseline sft dpo; do
  ADAPTER=(); LIMIT=()
  [[ "$ARM" == baseline ]] || ADAPTER=(--adapter "$RUN/${ARM}_adapter")
  [[ "$ACTION" != smoke ]] || LIMIT=(--limit 2)
  for SPLIT in val test; do
    "$PYTHON" -m region_rag.posttrain_run score --data "$DATA" --out "$RUN/$ARM/$SPLIT" --split "$SPLIT" \
      --pixels "$PIXELS" "${ADAPTER[@]}" "${LIMIT[@]}" 2>&1 | tee "$RUN/${ARM}_${SPLIT}.log"
  done
done
if [[ "$ACTION" != smoke ]]; then
  "$PYTHON" -m region_rag.posttrain_run report --data "$DATA" --root "$RUN"
fi
echo "Results: $RUN"
