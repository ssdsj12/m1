# M1 + Panda Prioritized WBC Teacher C0 Acceptance

## Result

C0 is complete on GPU0. The combined articulation maintained four-wheel balance while the Panda followed a seeded six-dimensional trajectory. This is deterministic Teacher play; no PPO runner, checkpoint, Student, external wrench, grasping task, or rolling command participated.

## Implementation outcome

- Corrected floating-base PhysX body-Jacobian indexing and wheel bottom-point contact Jacobians.
- Used relative hand orientation and a zero-velocity raised-cosine trajectory start.
- Added numerically scaled float64 KKT solves, feasible max-iteration handling, and strict residual reporting.
- Anchored legs and wheel velocity, used Panda `C+g` feed-forward, finite resolved-rate arm lookahead, and bounded wheel-speed integral braking.
- Separated 100 settling steps from scored mission steps and re-centered at the realized state.
- Paused/re-centered trajectory generation in safety HOLD/recovery and retained current arm bias compensation on a failed WBC cycle.
- Recorded safety reasons, motion failures, base activation, singularity crossing, and arm-target snap evidence.

## Verification

Pure/static command:

```bash
PYTHONPATH=/home/xk/coding/M1/Go2Pvcnn \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_wbc_contracts.py \
  tests/test_m1_panda_wbc_kinematics.py \
  tests/test_m1_panda_motion_distribution.py \
  tests/test_m1_panda_qp_backend.py \
  tests/test_m1_panda_standing_wbc.py \
  tests/test_m1_panda_wbc_safety.py \
  tests/test_m1_panda_wbc_teacher.py \
  tests/test_m1_panda_wbc_env_static.py \
  tests/test_m1_panda_wbc_play_static.py \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_teacher_env_cfg_static.py \
  tests/test_m1_panda_teacher_play_static.py
```

Result: `177 passed in 1.15s`, exit `0`. `py_compile` and `git diff --check` also exited `0`.

GPU: index `0`, NVIDIA GeForce RTX 5070, driver `580.159.03`.

Eight-step stationary smoke: exit `0`, `steps=8`, finite, QP rate `1.0`, maximum EE error `0.002141 m`, minimum singular value `0.185618`, maximum slip `0.001866 m/s`, roll/pitch `0.014726/0.002818 rad`, zero limit/base/self-collision/reset/snap counts.

Seed-42 moving acceptance: exit `0`, `steps=2000`, `exit_reason=steps_complete`, finite, QP `2000/2000`, maximum EE error `0.009566 m`, minimum singular value `0.141259`, maximum slip `0.007126 m/s`, roll/pitch `0.028045/0.008563 rad`, maximum arm-target step `0.005649 rad`, zero limit/base/self-collision/reset/snap counts, and all 2000 samples in `TRACK` with reason `safe`.

The code commit is `7a2548f`. Documentation commit is the commit containing this file.

## Boundary and next work

C0 proves stationary small-motion balance only. C1/C2 must separately design and validate rolling constraints and the physical mapping between planar base distribution and wheel motion before moving-base operation. C3 external-wrench randomization, Student training, grasping, and real hardware remain out of scope.
