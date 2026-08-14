#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M1_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$M1_ROOT"

exec "${PYTHON:-python}" Go2Pvcnn/scripts/m1_play.py \
  --task Isaac-M1-Pvcnn-Crossing-60mm-Policy-Play-v0 \
  --checkpoint Go2Pvcnn/logs/m1_curriculum/stage2e_contactfree_policy_gate/m1_contactfree_policy_accepted.pt \
  --perception-checkpoint Go2Pvcnn/logs/m1_curriculum/stage2e_contactfree_policy_gate/m1_contactfree_perception_accepted.pt \
  --rolling-wheel-velocity 1.0 \
  --num_envs 1 \
  --steps 100000 \
  --clip-actions 1.0 \
  --disable-crossing-reset \
  "$@"
