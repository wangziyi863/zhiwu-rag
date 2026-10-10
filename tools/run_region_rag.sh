#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ZHIWU_ROOT"
exec "$GLA_PYTHON" -m region_rag serve --port "${PORT:-7860}"
