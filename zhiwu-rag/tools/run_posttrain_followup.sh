#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
PYTHON="${POSTTRAIN_PYTHON:-$PWD/.venv-posttrain/bin/python}"
ROOT="${ROOT:-$PWD/reports/posttrain_full_20261008_192810}"
DATA="${DATA:-$PWD/reports/posttrain_data_v2}"
OUT="${OUT:-$ROOT/followup_budget_v1}"
mkdir -p "$OUT"
exec 9>"$PWD/rag_data/posttrain.lock"
flock -n 9 || { echo 'Another post-training script is running'; exit 1; }
"$PYTHON" -m region_rag.posttrain_followup --root "$ROOT" --data "$DATA" --out "$OUT/before_control" > "$OUT/before_control.log"
# Read original DPO settings rather than hardcoding a new training budget.
mapfile -t PARAMS < <("$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("\n".join(str(r[k]) for k in ("steps","accum","seed","lr","pixels","beta")))' "$ROOT/dpo_adapter/run.json")
[[ "${#PARAMS[@]}" == 6 ]] || { echo 'Invalid DPO training settings'; exit 1; }
if [[ ! -f "$OUT/control_adapter/run.json" ]]; then
  "$PYTHON" -m region_rag.posttrain_run train --mode continue_sft --data "$DATA" --out "$OUT/control_adapter" \
    --sft "$ROOT/sft_adapter" --steps "${PARAMS[0]}" --accum "${PARAMS[1]}" --seed "${PARAMS[2]}" \
    --lr "${PARAMS[3]}" --pixels "${PARAMS[4]}" --beta "${PARAMS[5]}" 2>&1 | tee "$OUT/control_train.log"
fi
for SPLIT in val test; do
  "$PYTHON" -m region_rag.posttrain_run score --data "$DATA" --out "$OUT/continue_sft/$SPLIT" \
    --adapter "$OUT/control_adapter" --split "$SPLIT" --pixels "${PARAMS[4]}" 2>&1 | tee "$OUT/control_${SPLIT}.log"
done
"$PYTHON" -m region_rag.posttrain_followup --root "$ROOT" --data "$DATA" --out "$OUT/analysis" --control "$OUT" 2>&1 | tee "$OUT/analysis.log"
echo "Results: $OUT/analysis/REPORT.md"
