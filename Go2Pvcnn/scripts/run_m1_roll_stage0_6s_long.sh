#!/usr/bin/env bash
set -euo pipefail

source /home/xk/miniconda3/etc/profile.d/conda.sh
conda activate go2pvcnn_ablation

export ISAACLAB_SITE=/home/xk/miniconda3/envs/go2pvcnn_ablation/lib/python3.11/site-packages/isaaclab/source
export PYTHONPATH="/home/xk/coding/M1/Go2Pvcnn:${ISAACLAB_SITE}/isaaclab:${ISAACLAB_SITE}/isaaclab_rl:${ISAACLAB_SITE}/isaaclab_tasks:${ISAACLAB_SITE}/isaaclab_mimic:${ISAACLAB_SITE}/isaaclab_assets:${ISAACLAB_SITE}/isaaclab_contrib:${PYTHONPATH:-}"

cd /home/xk/coding/M1
python Go2Pvcnn/scripts/m1_train.py \
  --headless \
  --task Isaac-M1-Roll-v0 \
  --num_envs 64 \
  --max_iterations 3000 \
  --run_name m1_roll_stage0_6s_long \
  --clip-actions 1.0
