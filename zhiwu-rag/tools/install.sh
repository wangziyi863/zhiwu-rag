#!/usr/bin/env bash
# Clean installation only. Existing users should configure .env, not reinstall working envs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ZHIWU_ROOT"
PYTHON="${PYTHON:-python3.10}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2]==(3,10), "Use Python 3.10"'
if [[ -e .venv-gla || -e .venv-rag || -e .venv-posttrain ]]; then
  echo 'An environment directory already exists. Configure .env to reuse it, or use a new checkout for a fresh install.' >&2
  exit 1
fi
"$PYTHON" -m venv .venv-gla
.venv-gla/bin/python -m pip install --upgrade pip
.venv-gla/bin/python -m pip install "torch==${TORCH_VERSION:-2.14.0}" "torchvision==${TORCHVISION_VERSION:-0.29.0}" --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
.venv-gla/bin/python -m pip install -r requirements/gla.txt
# GLA's inference path needs MMCV importable but does not call its CUDA custom ops.
# Build the CPU extension against the installed Torch, avoiding mismatched prebuilt wheels.
MMCV_WITH_OPS=1 FORCE_CUDA=0 CUDA_VISIBLE_DEVICES='' MAX_JOBS="${MAX_JOBS:-2}" \
 .venv-gla/bin/python -m pip install --no-build-isolation --no-binary=mmcv 'mmcv==2.1.0'
.venv-gla/bin/python -c 'from mmseg.models.segmentors import BaseSegmentor; import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
GLA_PYTHON="$ZHIWU_ROOT/.venv-gla/bin/python" bash tools/setup_region_rag.sh
BASE_PYTHON="$ZHIWU_ROOT/.venv-rag/bin/python" bash tools/setup_posttrain.sh
echo 'Environments installed. Next: bash tools/download_models.sh'
