#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
REPO="$ZHIWU_ROOT"
ABO_ROOT="${1:-${ABO_ROOT:-$REPO/data/ABO}}"
PYTHON="$GLA_PYTHON"
cd "$REPO"
RAG_HOME="${GLA_RAG_HOME:-$REPO/rag_data}"
mkdir -p "$RAG_HOME/abo" "$RAG_HOME/backups"
"$PYTHON" -m region_rag.abo --root "$ABO_ROOT" --output "$RAG_HOME/abo/items.jsonl"
if [[ -f "$RAG_HOME/index/index.npz" ]]; then
  cp -- "$RAG_HOME/index/index.npz" "$RAG_HOME/backups/index_before_abo_$(date +%Y%m%d_%H%M%S).npz"
fi
"$PYTHON" -m region_rag --home "$RAG_HOME" index --catalog "$RAG_HOME/abo/items.jsonl"
"$PYTHON" -m region_rag --home "$RAG_HOME" status
