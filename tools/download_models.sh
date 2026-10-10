#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ZHIWU_ROOT"
export HF_HUB_OFFLINE=0
"$GLA_PYTHON" tools/download_models.py vision
"$RAG_PYTHON" tools/download_models.py vlm
