# M1 smoke adaptation

This copy is a minimal M1 smoke-test adaptation of the Go2 PvCNN project. It registers a no-MPC IsaacLab environment for checking that the M1 USD, articulation names, action joints, contact bodies, and base termination body are wired correctly.

## Environment

Run from `/home/xk/coding/M1`:

```bash
source /home/xk/miniconda3/etc/profile.d/conda.sh
conda activate env1

export ISAACLAB_SITE=/home/xk/miniconda3/envs/env1/lib/python3.11/site-packages/isaaclab/source
export PYTHONPATH=/home/xk/coding/M1/Go2Pvcnn:$ISAACLAB_SITE/isaaclab:$ISAACLAB_SITE/isaaclab_rl:$ISAACLAB_SITE/isaaclab_tasks:$ISAACLAB_SITE/isaaclab_mimic:$ISAACLAB_SITE/isaaclab_assets:$ISAACLAB_SITE/isaaclab_contrib:$PYTHONPATH
```

## Registered task

```text
Isaac-M1-Smoke-v0
```

## Action contract

The M1 smoke environment uses a hybrid wheel-leg action split:

```text
12 leg joints: position control
4 wheel joints: velocity control
```

The leg joints are the `ABAD`, `HIP`, and `KNEE` joints on each leg. The wheel joints are:

```text
FAR_FOOT_JOINT
FBL_FOOT_JOINT
RAR_FOOT_JOINT
RBL_FOOT_JOINT
```

The environment exposes two mode names for higher-level controllers:

```text
rolling
wave
```

The task uses:

```text
/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd
```

## Smoke registration check

```bash
TERM=xterm PYTHONUNBUFFERED=1 python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
try:
    import gymnasium as gym
    import go2_pvcnn.tasks  # registers Isaac-M1-Smoke-v0
    spec = gym.spec("Isaac-M1-Smoke-v0")
    print(spec.id, spec.kwargs["env_cfg_entry_point"].__name__)
finally:
    app.close()
PY
```

Expected final line:

```text
Isaac-M1-Smoke-v0 M1SmokeEnvCfg
```

## M1 train

Basic M1 smoke training:

```bash
python Go2Pvcnn/scripts/m1_train.py --headless --num_envs 64 --max_iterations 1000
```

Quick startup check without training iterations:

```bash
python Go2Pvcnn/scripts/m1_train.py --headless --num_envs 1 --max_iterations 0 --run_name smoke_check
```

Training logs and checkpoints are written under:

```text
Go2Pvcnn/logs/m1_smoke/
```

## M1 play

Flat rolling smoke run:

```bash
python Go2Pvcnn/scripts/m1_play.py --headless --mode rolling --steps 1000 --num_envs 1
```

Wave smoke run:

```bash
python Go2Pvcnn/scripts/m1_play.py --headless --mode wave --steps 1000 --num_envs 1
```

Checkpoint policy play:

```bash
python Go2Pvcnn/scripts/m1_play.py --headless --checkpoint Go2Pvcnn/logs/m1_smoke/<run>/model_<iter>.pt --steps 1000 --num_envs 1
```

Useful tuning flags:

```text
--rolling-wheel-velocity 4.0
--wave-wheel-velocity 1.5
--wave-amplitude 0.08
--wave-frequency 1.0
```

## Current train/play status

The original `scripts/train.py` and `scripts/play.py` are still Go2 semantic-MPC scripts. They import the Go2 MPC config directly, so they are not M1 training/play commands yet.

For M1, the current usable commands are `scripts/m1_train.py` and `scripts/m1_play.py`. Training uses the simple M1 smoke PPO config and does not use the old Go2 MPC/PVCNN teacher path. Playback supports open-loop rolling/wave when no checkpoint is supplied, and trained policy playback when `--checkpoint` is supplied.
