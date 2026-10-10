#!/usr/bin/env bash
set -euo pipefail
REPO="${GLA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
PYTHON="${RAG_PYTHON:-$REPO/.venv-rag/bin/python}"
"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
from transformers import Sam2Model, AutoModelForZeroShotObjectDetection
for model in ('IDEA-Research/grounding-dino-tiny','facebook/sam2.1-hiera-tiny'):
    print('Preparing',model,flush=True)
    snapshot_download(model,allow_patterns=['*.json','*.txt','*.model','*.safetensors'],max_workers=3)
print('Instance repair checkpoints are ready for offline inference.')
PY
