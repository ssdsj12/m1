# 2026-08-25 M1 + Panda folded-load training guard

## Purpose

Record Task 6 TDD evidence for always-on catastrophe handling, rolling eligibility, directional gates, and atomic three-seed acceptance.

## Stage

T400.10b / folded-load curriculum Task 6.

## Verification

```bash
cd Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_training_guard.py
```

Valid RED: guard module missing. GREEN: `8 passed in 0.77s`.

## Key Metrics

- Catastrophe works before any eligible checkpoint: hard failure `>0.50` for 2 updates and `>0.20` for 5 updates.
- Nonfinite state, any inactive-action leakage, and fold hard failure stop immediately.
- Command stages use 200 completed episodes; DR stages use 400.
- Shared timeout/contact/orientation/VX/yaw gates, four directional buckets with at least 25 episodes, per-bucket gates, and stationary speed gates pass boundary tests.
- Eligible-rank patience is 50 updates; stage cap is 600 updates.
- Seed 42/43/44 reports are atomic; only three passing reports copy `model_best.pt` byte-for-byte to `model_final.pt`.

## Boundary

Pure CPU guard only. Runner callback, checkpoint creation, manifest lifecycle, physical evaluation, and GPU training remain open.

## Git Refs

- Baseline Ref: `ef31b58`
- Candidate Ref: pending Task 6 commit
- Key Files:
  - `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py`
  - `Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py`
