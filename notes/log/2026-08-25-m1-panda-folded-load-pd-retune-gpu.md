# M1 + Panda Folded-Load `120/8` GPU0 Probe

## Purpose

Verify whether the user-approved folded-load-only Panda shoulder retune can hold the unchanged fold target over the full 256-step PPO horizon without weakening any safety gate.

## Stage And Todo

- Stage: T400.10b Task 9 physical qualification
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Design: [folded-load PD retune](../../docs/superpowers/specs/2026-08-25-m1-panda-folded-load-pd-retune-design.md)
- Plan: [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-pd-retune.md)

## Git Refs

- Baseline Ref: `8ff61a3`
- Controller Candidate: `7fca6b2`
- Probe/Runbook Candidate: `cf0289d`
- Current Work Ref: `codex/m1-panda-ppo-stability`

## Input Conditions

- GPU: GPU0, NVIDIA GeForce RTX 5070 12 GB
- Environment count: `8`
- Device: `cuda:0`
- Policy action: all-zero `[8,23]`; Panda coordinates `16:23` re-zeroed by wrapper
- Panda target: `(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)`
- Folded-load-only shoulder gains: `Kp=120`, `Kd=8`
- Forearm gains: unchanged `80/4`
- Effort limits and safety thresholds: unchanged
- External wrench: disabled

## Verification

Focused CPU/static tests:

```bash
cd Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_panda_folded_load_wrapper.py tests/test_m1_panda_folded_load_scripts.py tests/test_m1_panda_folded_load_probe_static.py -q
```

Result: `16 passed`.

The repository-wide ordinary-Python suite reached `276 passed` before its first unrelated baseline failure: `test_m1_wrapper_flattens_and_joins_policy_observation_groups` imports `isaaclab.utils` without launching the Isaac application. This was not patched as part of folded-load work.

GPU0 short probe:

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 16 --device cuda:0 --report logs/m1_panda_folded_load/probe-pd120-8x16.json --headless
```

Result: report `passed=true`.

- fold error max: `0.0659973621 rad`
- joint-limit proximity min: `0.0457024574 rad`
- effort utilization max: `1.0`
- inactive action max: `0.0`
- mount wrench norm max: `638.4578857`
- finite state/hard failure: `true` / `0.0`

GPU0 full-horizon probe:

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 256 --device cuda:0 --report logs/m1_panda_folded_load/probe-pd120-8x256.json --headless
```

Result: report `passed=false`; `joint_limit_margin=false`.

- fold error max: `0.2149407864 rad`
- joint-limit proximity min: `-0.1032409668 rad`
- effort utilization max: `1.0`
- inactive action max: `0.0`
- mount wrench norm max: `638.4578857`
- finite state/hard failure: `true` / `0.0`

The earlier `80/4` 256-step diagnostic was also `passed=false`: fold error `0.2620239258 rad`, joint margin `-0.1503241062 rad`, effort utilization `1.0`.

## Root Cause

Joint4's approved target has only about `0.1117 rad` clearance to IsaacLab's soft lower limit. The retuned full-horizon deflection is `0.21494 rad`, hence `0.1117 - 0.21494 = -0.10324 rad`, exactly matching the reported minimum margin. Both `80/4` and `120/8` reach effort utilization `1.0`; the unchanged `87 Nm` shoulder limit is the active authority boundary, so raising stiffness alone cannot remove the steady-state error.

## Result And Decision

Task-local controller isolation, explicit per-step fold targets, atomic probe report, and operational runbook are implemented. The approved `120/8` choice improves the 256-step fold error by about `0.0471 rad`, but does not pass the unchanged joint-margin gate.

Per the approved stop policy, the 8×1 smoke, 64×10 smoke, and long curriculum were not started. Failed reports remain diagnostic-only. The recurring PhysX warning for `/World/envs/env_0/Robot/Panda/root_joint` remains recorded and is not treated as a mechanical safety claim.

## Follow-Up

A new user-approved controller choice is required. At least one current constraint must change: move joint4's fold target farther from the soft lower limit, add model-based gravity compensation/feedforward torque, or revise actuator effort authority with separate physical justification. Increasing Kp beyond `120` while keeping the same saturated effort limit is not recommended.
