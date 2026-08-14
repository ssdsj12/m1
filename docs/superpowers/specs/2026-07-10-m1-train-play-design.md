# M1 train/play design

## Goal

Add M1-specific train and play entrypoints for the adapted M1 project without changing the existing Go2 semantic-MPC scripts.

## Design

Training uses `Isaac-M1-Smoke-v0`, a lightweight RSL-RL wrapper that flattens the single `policy` observation group, and an MLP PPO config. This is a basic M1 smoke locomotion training path, not Go2 MPC/PVCNN teacher training.

Playback uses one M1-specific script. Without a checkpoint it runs the existing open-loop rolling/wave controller. With a checkpoint it builds the same M1 env and RSL-RL wrapper, loads `OnPolicyRunner`, gets the inference policy, and steps the environment with policy actions.

## Commands

```bash
python Go2Pvcnn/scripts/m1_train.py --headless --num_envs 64 --max_iterations 1000
python Go2Pvcnn/scripts/m1_play.py --headless --mode rolling --steps 1000
python Go2Pvcnn/scripts/m1_play.py --headless --checkpoint /path/to/model.pt --steps 1000
```

## Constraints

- Use conda env `go2pvcnn_ablation`.
- Keep original `scripts/train.py` and `scripts/play.py` behavior unchanged.
- Keep M1 task id as `Isaac-M1-Smoke-v0`.
