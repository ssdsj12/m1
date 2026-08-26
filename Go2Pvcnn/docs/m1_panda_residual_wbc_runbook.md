# M1 + Panda 8D Residual WBC Runbook

This entrypoint validates the Phase 1–4 reference controller. It does not load or train a PPO policy.

## Zero-residual GPU0 smoke

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_residual_wbc_play.py \
  --task Isaac-M1-Panda-Residual-Wbc-v0 \
  --headless --device cuda:0 --warmup-steps 64 --steps 256 \
  --summary-json tests/artifacts/m1_panda_residual_wbc_zero.json
```

## One-axis manual probe

`--residual-axis` follows `[Fx,Fy,Fz,Mx,My,Mz,delta_height,delta_stance]`.

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_residual_wbc_play.py \
  --task Isaac-M1-Panda-Residual-Wbc-v0 \
  --headless --device cuda:0 --warmup-steps 64 --steps 128 \
  --residual-axis 0 --residual-value 0.1 \
  --summary-json tests/artifacts/m1_panda_residual_wbc_fx_positive.json
```

Run positive and negative `0.1` probes for axes `0..7`. Stop on any non-finite value, QP failure, lost wheel contact, base contact, bad orientation, joint-limit violation, unexpected reset, or safety termination.
