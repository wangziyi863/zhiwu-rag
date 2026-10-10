#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Separate environment, reuse installed GPU runtime without changing the serving venv.
BASE_PYTHON="${BASE_PYTHON:-$PWD/.venv-rag/bin/python}"
"$BASE_PYTHON" -m venv --system-site-packages .venv-posttrain
# Include the serving environment's Transformers dependencies as read-only imports.
SITE="$(.venv-posttrain/bin/python -c 'import site; print(site.getsitepackages()[0])')"
"$BASE_PYTHON" -c 'import sys; print("\n".join(p for p in sys.path if p.endswith("site-packages")))' > "$SITE/rag_runtime.pth"
.venv-posttrain/bin/python -m pip install 'peft==0.17.1'
.venv-posttrain/bin/python -c 'import torch,transformers,peft; print(torch.__version__,transformers.__version__,peft.__version__); assert torch.cuda.is_available()'
