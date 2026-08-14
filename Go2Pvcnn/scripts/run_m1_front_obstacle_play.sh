#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M1_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$M1_ROOT"

exec "$SCRIPT_DIR/run_m1_contactfree_policy_play.sh" "$@"
