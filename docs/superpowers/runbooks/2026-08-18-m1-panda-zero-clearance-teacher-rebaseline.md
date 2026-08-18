# M1 + Panda 零间隙 Teacher 重基线运行手册

## Accepted authority

- Combined asset: `assets/m1_panda/m1_panda.usd`
- Accepted SHA-256: `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`
- Panda SHA-256: `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
- Builder/runtime baseline commit: `eba7906`
- Isaac Sim `5.1`, GPU0 RTX 5070, driver `580.159.03`

Student S1 dataset and checkpoint manifests must carry the exact combined-asset SHA above. A mismatch fails before collection or loading.

## Asset gates

Run from `/home/xk/coding/M1/Go2Pvcnn`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_panda_wbc_env_static.py

OMNI_KIT_ACCEPT_EULA=Y /home/xk/miniconda3/envs/go2/bin/python \
  tests/run_m1_panda_asset_pxr_behavior.py

OMNI_KIT_ACCEPT_EULA=Y /home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --device cpu --headless
```

Required results: 25 DOF, one root `/M1Panda/BASE_LINK`, enabled fixed mount, `mount_plane_error_m <= 1e-6`, `abs(mount_surface_gap_m) <= 1e-6`, one-step relative delta `<1e-4`, and no validation errors. Copy the whole asset tree to a fresh `mktemp -d` directory and rerun the same verifier against the copy to prove relocation closure.

## Visual gate

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py --device cuda:0 --steps 0 --seed 42
```

Inspect the center mount from multiple views. Accept only visible attachment without air gap, mesh penetration or jitter. Close the GUI before headless acceptance runs.

## GPU0 Teacher gates

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py --headless --device cuda:0 \
  --steps 2000 --seed 42 --stats-interval 500 \
  --summary-json /tmp/m1_panda_zero_clearance_c0.json

CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py --headless --device cuda:0 \
  --steps 4000 --seed 42 --disable-target-motion --stats-interval 400 \
  --summary-json /tmp/m1_panda_zero_clearance_c1a_no_arm.json

CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py --headless --device cuda:0 \
  --steps 4000 --seed 42 --stats-interval 400 \
  --summary-json /tmp/m1_panda_zero_clearance_c1a_combined.json
```

C0 requires 2000 TRACK steps, QP rate `1.0`, and zero limit/base-contact/self-collision/reset/snap events. Both C1a summaries require `hard_gates_passed=true`, five phases of 800 steps, 4000 TRACK steps, QP rate `1.0`, four-wheel contact and all existing safety gates.

## Scope

This authority unlocks only the approved flat-ground DAgger Student S1 plan. It does not authorize turning, random external wrench, uneven terrain, PPO, grasping, payload testing or real hardware.
