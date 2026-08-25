# 2026-08-25 M1 + Panda folded-load wrapper and DR

## Purpose

Record Task 5 evidence for the exact-zero arm boundary, episode command attribution, stage reset ranges, leg-only joint reset, and fold diagnostics.

## Stage

T400.10b / folded-load curriculum Task 5.

## Related Todo

- [T400 branch](../todo/T400-m1-panda-force-aware-teacher-student.md)
- [implementation plan](../../docs/superpowers/plans/2026-08-25-m1-panda-folded-load-locomotion-curriculum.md)

## Verification

```bash
cd Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_folded_load_wrapper.py \
  tests/test_m1_panda_folded_load_randomization.py \
  tests/test_m1_panda_folded_load_mdp.py \
  tests/test_m1_panda_folded_load_env_static.py \
  tests/test_m1_panda_coordinated_domain_randomization.py \
  tests/test_m1_panda_coordinated_wrapper_disturbance.py
```

The valid RED was a missing wrapper module. CPU import initially reached the full Isaac MDP package, so the test adopted the repository's existing minimal-MDP-stub pattern. A stale Task 4 assertion expecting old `arm_position_range=(0,0)` was replaced by the stricter leg-only reset contract.

## Key Metrics

- Final new plus legacy regression: `34 passed`.
- Arm action indices 16:23 are cloned and exact-zeroed before `env.step`; caller actions are not mutated.
- No external force/torque API occurs in the wrapper.
- Commands are seeded, episode-constant, and selectively replaced only for done environment IDs.
- Episode records preserve command, family, directional buckets, tracking sums, steps, timeout, base-contact, and orientation causes.
- L0/L1 deterministic and D1/D2/D3 exact root/leg/friction ranges pass.
- Dedicated reset writes selected environments only, randomizes only 12 leg positions, and preserves wheel/Panda positions and every default joint velocity.
- Fold error, effort utilization, joint-limit proximity, mount wrench, and inactive-action diagnostics are finite in the covered CPU path.

## Boundary

No Isaac application/GPU physics was run. Real material writes, mount-wrench values, PD effort response, contact attribution after Isaac auto-reset, and locomotion remain for the probe/smoke gates.

## Git Refs

- Baseline Ref: `d45a485`
- Candidate Ref: pending Task 5 commit
- Key Files:
  - `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py`
  - `Go2Pvcnn/go2_pvcnn/mdp/events.py`
