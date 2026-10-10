#!/usr/bin/env bash
# Run the overlap-aware KV compression ablation on a validation set.
# Usage: bash tools/run_ablation.sh [DATASET] [NUM_IMAGES] [NUM_CLASSES]
set -euo pipefail
cd "$(dirname "$0")/.."

DS="${1:-voc21}"
N="${2:-1449}"
NC="${3:-21}"
CFG="./data/_gla_configs/cfg_${DS}.py"
mkdir -p reports

run () {
  local tag="$1"; shift
  echo "=================== ${tag} ==================="
  CUDA_VISIBLE_DEVICES="${GPU:-0}" python tools/subset_eval.py \
      --config "${CFG}" --num-images "${N}" --num-classes "${NC}" \
      --tag "${tag}" --out "reports/abl_${DS}_${tag}.json" "$@"
}

run baseline
run uniform80 --overlap_compression --compression_mode uniform --uniform_keep_ratio 0.8
run uniform50 --overlap_compression --compression_mode uniform --uniform_keep_ratio 0.5
run location  --overlap_compression --compression_mode location
run sim90     --overlap_compression --compression_mode sim --merge_tau 0.9
run sim95     --overlap_compression --compression_mode sim --merge_tau 0.95
run sim85     --overlap_compression --compression_mode sim --merge_tau 0.85
run sim90_nocount --overlap_compression --compression_mode sim --merge_tau 0.9 --no_count_adapt

echo "ALL DONE"
