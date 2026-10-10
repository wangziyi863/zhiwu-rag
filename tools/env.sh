#!/usr/bin/env bash
# Source this file; do not execute it to configure a parent shell.
ZHIWU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ZHIWU_ROOT/.env" ]]; then
  set -a
  source "$ZHIWU_ROOT/.env"
  set +a
fi
export GLA_PYTHON="${GLA_PYTHON:-$ZHIWU_ROOT/.venv-gla/bin/python}"
export RAG_PYTHON="${RAG_PYTHON:-$ZHIWU_ROOT/.venv-rag/bin/python}"
export DPO_PYTHON="${DPO_PYTHON:-$ZHIWU_ROOT/.venv-posttrain/bin/python}"
export POSTTRAIN_PYTHON="${POSTTRAIN_PYTHON:-$DPO_PYTHON}"
export GLA_RAG_HOME="${GLA_RAG_HOME:-$ZHIWU_ROOT/rag_data}"
export DINO_HUB="${DINO_HUB:-$ZHIWU_ROOT/vendor/dino}"
export DINO_WEIGHTS="${DINO_WEIGHTS:-$ZHIWU_ROOT/models/base/dino_vitbase8_pretrain.pth}"
export RAG_E_ROOT="${RAG_E_ROOT:-$ZHIWU_ROOT/models/clean_e}"
export RAG_DPO_ROOT="${RAG_DPO_ROOT:-$ZHIWU_ROOT/models/dpo}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export RAG_RERANK_MODEL="${RAG_RERANK_MODEL:-clean_e}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ZHIWU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
