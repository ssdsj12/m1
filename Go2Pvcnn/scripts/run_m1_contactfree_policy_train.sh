#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M1_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$M1_ROOT"

exec "${PYTHON:-python}" Go2Pvcnn/scripts/m1_wave_distill.py \
  --hierarchical-gate \
  --task Isaac-M1-Pvcnn-Crossing-60mm-ContactFree-Train-v0 \
  --policy-checkpoint Go2Pvcnn/logs/m1_curriculum/stage2e_contactfree_policy_gate/m1_contactfree_policy_accepted.pt \
  --perception-checkpoint Go2Pvcnn/logs/m1_curriculum/stage2e_contactfree_policy_gate/m1_contactfree_perception_accepted.pt \
  --num-envs 8 \
  --updates 4000 \
  --learning-rate 1e-4 \
  --checkpoint-interval 500 \
  --student-rollout-final-weight 1.0 \
  --teacher-forcing-fraction 0.20 \
  --gate-positive-weight 5.0 \
  --wheel-preservation-weight 0.20 \
  --run-name "m1_contactfree_gate_$(date +%Y%m%d_%H%M%S)" \
  --clip-actions 1.0 \
  --headless \
  "$@"
