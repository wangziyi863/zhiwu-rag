#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ZHIWU_ROOT"
exec "$GLA_PYTHON" tools/check_install.py "$@"
