#!/usr/bin/env bash
# Installs only the separate generation environment. Use --download for weights.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${GLA_PYTHON:-$PWD/.venv-gla/bin/python}"
if [[ ! -x .venv-rag/bin/python ]]; then
  "$BASE" -m venv --system-site-packages .venv-rag
fi
# A venv created from another venv does not inherit that venv's site-packages.
SITE="$(.venv-rag/bin/python -c 'import site; print(site.getsitepackages()[0])')"
"$BASE" -c 'import sys; print("\n".join(p for p in sys.path if p.endswith("site-packages")))' > "$SITE/gla_runtime.pth"
.venv-rag/bin/python -m pip install -r requirements/vlm.txt
.venv-rag/bin/python -c 'import torch; from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; print("CUDA:",torch.cuda.is_available(),"Torch:",torch.__version__)'
if [[ "${1:-}" == --download ]]; then
  .venv-rag/bin/python tools/download_models.py vlm
fi
echo 'Separate RAG environment is ready. Next: bash tools/download_models.sh'
