# M1 + Panda Arm-MPC 8D Residual PPO Runbook

## Scope

This runbook covers the Phase 5 sensor-calibrated Arm-MPC gate and the Phase 6
fresh `103 -> 8` residual PPO path. The task is stationary M1 balance plus a
small six-axis Panda end-effector trajectory. It does not authorize rolling,
payloads, grasping, pushes, broad domain randomization, Student distillation or
real-hardware deployment.

## Environment

```bash
cd /home/xk/coding/M1
export PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/xk/miniconda3/envs/go2/bin/python
```

GPU jobs run directly with `$PY` and `--device cuda:0`; do not launch them via
`go2pvcnn.sh`.

## Phase 5 gate

Run each formal seed in an independent Isaac Sim process:

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 42 --headless
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 43 --headless
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 44 --headless
```

Do not enter Phase 6 unless all three summaries contain `accepted=true`.

## Phase 6 Stage 0

Stage 0 ignores the policy sample and sends an exact zero 8D residual into the
runtime. It validates the complete VecEnv, MPC/WBC, PPO update, diagnostics and
atomic checkpoint chain:

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage zero --device cuda:0 --num_envs 8 --max_iterations 10 --seed 42 \
  --headless --run_dir Go2Pvcnn/logs/m1_panda_arm_mpc_residual/zero_s42
```

The resulting `run_manifest.json` must report `status=safe_complete`, action
dimension `8`, observation dimension `103`, and `force_zero_residual=true`.
Stage 0 is a wiring check, not a policy promotion, so `accepted` remains false.

## Guarded short training

```bash
SHORT=Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_fixed_condition_v1
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage short --device cuda:0 --num_envs 8 --max_iterations 100 --seed 42 \
  --headless --run_dir "$SHORT"
```

The policy is initialized from scratch; no legacy 23D checkpoint is accepted.
The safety controller runs exactly 100 completed PPO updates and atomically
writes `candidate_u000/u025/u050/u075/u100.pt`. It may stop only for hard
failure, lost wheel contact, MPC/QP failure, non-finite optimization data, or
per-channel saturation at or above one percent. It does not rank candidates,
publish `model_best.pt`, or claim acceptance. A usable short manifest therefore
has `status=safe_complete`, `accepted=false`, and `promotion_required=true`.

## Fixed-condition promotion

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py \
  --short_manifest "$SHORT/run_manifest.json" --device cuda:0 --headless
```

The driver first executes three independent zero-vs-zero pairs for each seed
42/43/44 (nine fresh Isaac Sim processes) and calibrates metric tolerances from
measured PhysX noise. It then compares each of the five candidates against an
exact zero-residual baseline for all three seeds (15 additional fresh
processes), using 4000 steps per process. Selection is stability-first and
tolerance-aware; wrench/slip may not regress and every seed must pass hard
gates. Only the driver may atomically publish `model_best.pt` and an accepted
`promotion_manifest.json`. A rejected manifest must not start long training.

## Conditional long training

Long mode is refused unless the short manifest is accepted and its asset,
configuration and checkpoint hashes match the current files:

```bash
LONG=Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_fixed_condition_v1
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage long --device cuda:0 --num_envs 8 --max_iterations 3000 --seed 42 \
  --promotion_manifest "$SHORT/promotion_manifest.json" \
  --headless --run_dir "$LONG"
```

Long mode revalidates the short manifest, promotion manifest, selected
checkpoint, asset, configuration and reward SHA-256 chain before loading only
the promoted 8D checkpoint. Never delete these lineage files while long
training is running. A rejected or failed manifest is evidence, not a parent
checkpoint.

## GUI Play

```bash
$PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_play.py \
  --checkpoint "$SHORT/model_best.pt" \
  --device cuda:0 --num_envs 1 --seed 42
```

Play disables only the IsaacLab manager control window before `gym.make`; the
simulation viewport remains enabled. This avoids the delayed UI callback that
accessed a destroyed `viewport_camera_controller` during shutdown.
