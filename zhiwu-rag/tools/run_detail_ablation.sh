#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTION="${1:-all}"
PYTHON="${POSTTRAIN_PYTHON:-$PWD/.venv-posttrain/bin/python}"
GLA_PYTHON="${GLA_PYTHON:-$PWD/.venv-gla/bin/python}"
DATA="${DATA:-$PWD/reports/posttrain_data_v2}"
BASE="${BASE:-$PWD/reports/posttrain_full_20261008_192810}"
STEPS="${STEPS:-90}"; ACCUM="${ACCUM:-4}"; LIMIT=0
if [[ "$ACTION" == smoke ]]; then
  RUN="${RUN:-$PWD/reports/detail_ablation_smoke_v2}"; STEPS=2; ACCUM=1; LIMIT=2
else
  RUN="${RUN:-$PWD/reports/detail_ablation_v1}"
fi
mkdir -p "$RUN" "$PWD/rag_data"
exec 9>"$PWD/rag_data/posttrain.lock"
flock -n 9 || { echo 'Another experiment is running'; exit 1; }
COMMON=(--data "$DATA" --root "$BASE" --out "$RUN" --steps "$STEPS" --accum "$ACCUM"
        --lr "${LR:-5e-6}" --seed "${SEED:-20261009}" --margin "${MARGIN:-0.5}"
        --ce-weight "${CE_WEIGHT:-0.2}" --limit "$LIMIT")
if [[ -n "${QUALITY_REVIEW:-}" ]]; then
  COMMON+=(--quality-review "$QUALITY_REVIEW")
fi
prepare() {
  "$GLA_PYTHON" -m region_rag.detail_prepare --data "$DATA" --out "$RUN/prepare" --limit "$LIMIT" \
    --dino-hub "${DINO_HUB:-$HOME/.cache/torch/hub/facebookresearch_dino_main}" \
    --dino-weights "${DINO_WEIGHTS:-$HOME/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth}" \
    2>&1 | tee "$RUN/prepare.log"
}
run() { "$PYTHON" -m region_rag.detail_ablation "$@" "${COMMON[@]}"; }
arm() {
  local name="$1"
  if [[ "$name" == D || "$name" == E ]]; then
    run train --arm "$name" 2>&1 | tee "$RUN/${name}_train.log"
  fi
  for split in val test; do
    run score --arm "$name" --split "$split" 2>&1 | tee "$RUN/${name}_${split}.log"
  done
}
case "$ACTION" in
  prepare) prepare ;;
  mine) run mine 2>&1 | tee "$RUN/mine.log" ;;
  all|smoke)
    prepare
    run mine 2>&1 | tee "$RUN/mine.log"
    for name in A B C D E; do arm "$name"; done
    run report 2>&1 | tee "$RUN/report.log"
    ;;
  A|B|C|D|E) arm "$ACTION" ;;
  report) run report 2>&1 | tee "$RUN/report.log" ;;
  *) echo 'Usage: bash tools/run_detail_ablation.sh all|smoke|prepare|mine|A|B|C|D|E|report'; exit 2 ;;
esac
echo "Results: $RUN"
