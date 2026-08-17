# T400.8b Task 8 isolated WBC effort environment

## Purpose

Connect the deterministic C0 Teacher to an isolated Isaac Lab configuration without changing the legacy A0/A1 60-observation/16-action route.

## Stage

T400.8b / C0 deterministic Teacher / Task 8

## Procedure and evidence

1. RED: the new static contract produced `3 failed` because the WBC asset copy, environment file, and Gym ID did not exist.
2. Added `M1_PANDA_WBC_CFG` as an independent copy. Stiffness/damping are zero only for the ordered 23 WBC-controlled joints; both fingers retain the original position actuator and `0.04` open pose.
3. Added one preserved-order, unit-scale `JointEffortActionCfg`, 200 Hz physics/control timing, 50 Hz rendering, 20-second episodes, and inherited wheel/base contact sensing.
4. Registered `Isaac-M1-Panda-Wbc-Teacher-C0-v0` lazily with no RSL-RL runner.
5. Locked the legacy A0/A1 environment and play sources by SHA-256 and extended the existing complete registration-list test for the new ID.
6. Focused environment contract: `3 passed`.
7. Asset/A0/A1/smoke regression: `57 passed`.
8. Tasks 1–8 regression: `167 passed`; `py_compile` and `git diff --check` exit `0`.

## Result

- The new environment accepts exactly 23 efforts in canonical order.
- The original `M1_PANDA_CFG` and legacy Teacher source files remain unchanged.
- This is an environment boundary for deterministic WBC play, not a PPO/Student training configuration.

## Follow-up

Proceed to Task 9 PhysX tensor adapter and deterministic play entry point.

## Git refs

- Baseline Ref: `1b45bdf`
- Candidate Ref: `8d642e7`
- Key Files:
  - [WBC asset config](../../Go2Pvcnn/go2_pvcnn/assets/m1_panda.py)
  - [WBC environment config](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py)
  - [WBC environment tests](../../Go2Pvcnn/tests/test_m1_panda_wbc_env_static.py)
