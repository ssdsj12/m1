# 2026-08-30 M1 + Panda Phase 6 PPO Scale/Normalization Execution

## Frozen implementation baseline

- Repository: `/home/xk/coding/M1`
- Branch: `main`
- Commit: `3a8dd51cf1b4d948c65a1c5191623440e93d6778`
- Device contract: `CUDA_VISIBLE_DEVICES=0`, Isaac/RSL device `cuda:0`
- Python: `/home/xk/miniconda3/envs/go2/bin/python`
- Focused CPU verification: `111 passed in 1.70s`
- Compile verification: `python -m compileall -q agent go2_pvcnn scripts rsl_rl/rsl_rl` exited `0`
- Whitespace verification: `git diff --check` emitted no output

## Phase 5 prerequisite

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 42 --headless \
  --summary-json Go2Pvcnn/logs/m1_panda_arm_mpc_probe/phase5_scale_norm_v3_s42.json
```

Authoritative result: `accepted=true`, steps `4000/4000`, MPC/QP rates
`1.0/1.0`, minimum wheel contacts `4`, base contacts `0`, joint-limit
violations `0`, resets `0`, maximum EE position error
`0.007537173326784423 m`, force/moment direction cosine
`0.9999999974317467 / 0.9999970430690538`.

## Fresh v3 artifact roots

- Pilot: `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_scale_norm_v3`
- Short: `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_scale_norm_v3`
- Long: `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_scale_norm_v3`

## Current status

Phase 5 prerequisite accepted. Exact 10-update pilot is the next authorized
operation. The 100-update short, 24 fixed-condition workers, and 3000-update
long have not started. Long remains fail-closed unless the fresh promotion
manifest contains `accepted=true`.
