# 2026-09-05 M1 + Right Panda + Right O6 Asset Foundation

## Purpose

Complete T600.1: import the audited O6 source closure, build one M1 + right
Panda + right O6 articulation, freeze the 29-channel Isaac Lab contract, and
pass the 2000-step GPU0 physical gate.

## Scope

This acceptance covers only the asset and control interface foundation. It does
not implement or accept a Gym environment, free-space control, contact-aware
MPC, multimodal perception, Residual policy, or training entrypoint.

## Git Lineage

- Planning baseline: `c8bb5e8`
- O6 source closure: `26d6a2f53f72f2b3e4e8485640fedff60f876829`
- Portable source-test import: `c103de721e7273e9aac00aef8c0f5f08242a0dd8`
- Single-articulation builder and assets: `f6ae578dea94146938a0a75a151e22231d4da5a9`
- Isaac Lab 29-channel contract: `b673ad96f8b535b4f1c69cfada663b027f0a36a1`
- Physical verifier and accepted manifest: `8215b1b376286823725145df9a280a15a6bd9308`

## Build And Regression

The source closure contains 35 files, 12 meshes per hand, and all four USD
configuration layers per side. The builder consumes only the normalized right
O6 entry and produces `/M1SinglePandaO6/BASE_LINK` with two enabled assembly
joints.

The complete local regression command was:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_sources.py \
  tests/test_m1_single_panda_o6_asset_static.py \
  tests/test_m1_single_panda_o6_contracts.py \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_wbc_contracts.py
```

Result: `59 passed`. The normalizer, builder, verifier, and asset
contract modules also passed `py_compile`.

Repeated forced builds retained these formal output hashes:

- Combined asset SHA-256: `4c149d771ddf4caede277d94ae7a3349c43967c09ad7e813322fed8d37566705`
- Static build manifest SHA-256: `62f696f05d51a56d9960522372a8855ca36db482886e22ffc20628628ebdba6a`
- O6 source manifest SHA-256: `3ec630086d886739a2871ad6bad98dd009c9c31e8f27371682da741aa13c9227`

The Isaac importer and USD flattener changed only equivalent material child
ordering and `Flattened_Prototype_*` assignment in two generated intermediate
layers. Fixed `PYTHONHASHSEED=0` did not remove this C++ serialization behavior.
The accepted intermediate layers therefore remain the Git-LFS-pinned versions;
the final physical gate below was rerun against those pinned files.

## GPU0 Acceptance

Command:

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --steps 2000 \
  --device cuda:0 \
  --headless
```

Final evidence:

- `physics_steps=2000`
- `measured_physical_dof_count=34`
- `active_control_count=29`
- `four_wheel_contact_ratio=1.0`
- `max_mount_position_drift_m=2.384204549343849e-07`
- `max_mount_orientation_drift_rad=4.4802714521804273e-07`
- `o6_collision_count=12`
- `nonfinite_count=0`
- `hard_joint_limit_count=0`
- `unexpected_contact_count=0`
- `unexpected_reset_count=0`
- `base_instability_count=0`
- `offline_errors=[]`
- `hard_gates_passed=true`

The accepted runtime-enriched manifest SHA-256 is
`62c615a95cbbc97a682dcc5ecda059e7a8cf40c4d2c84cc1902cda0baaf4454f`.

## Conclusion

T600.1 passes. The project now owns a relocatable single articulation with 34
measured physical DOFs, exactly 29 active channels, five O6 mimic joints kept
out of the active action interface, and a passing 2000-step stability gate.
T600.2 is the next open stage and requires a separate design and implementation
plan.

## Related Files

- [runbook](../../docs/superpowers/runbooks/2026-09-05-m1-single-panda-o6-asset-foundation.md)
- [implementation plan](../../docs/superpowers/plans/2026-09-05-m1-single-panda-o6-asset-foundation.md)
- [T600 roadmap](../todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md)
