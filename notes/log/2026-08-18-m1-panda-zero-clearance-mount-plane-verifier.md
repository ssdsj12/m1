# 2026-08-18 M1 + Panda 零间隙安装平面验证器

## Purpose

执行 T400.6 零间隙计划 Task 2，为父侧 fixed-joint 安装位置增加独立 M1-only 顶面测量和 `1e-6 m` 硬门。

## Stage and related todo

- Stage: T400.6 / zero-clearance Task 2
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Plan: [zero-clearance Teacher rebaseline](../../docs/superpowers/plans/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)

## Input conditions

- Baseline ref: `08cf683`
- Candidate ref: working tree
- Generated USD remains the old `10 mm` artifact until Task 3
- Python: `/home/xk/miniconda3/envs/go2/bin/python`

## Procedure and evidence

1. Added `_mount_plane_errors(...)` contract test; valid RED failed because the helper did not exist。
2. Added `MOUNT_PLANE_TOLERANCE_M = 1e-6` and JSON fields for actual/expected parent pose and maximum error。
3. Expected pose comes from independent `assets/m1_panda/m1_floating.usda` path `/ZJ_V3_URDF_V1_0/BASE_LINK`; it does not use combined `/M1Panda/BASE_LINK`, whose Panda descendants could contaminate a bound。
4. Mirrored the measurement in the PXR behavior check。
5. Focused GREEN: `1 passed, 17 deselected`，exit `0`。
6. Full asset static GREEN: `18 passed`，exit `0`。

## Result

The pure verifier contract passes. PXR behavior and runtime verifier are intentionally deferred until Task 3 rebuilds `m1_panda.usd`; running them against the old 10 mm artifact should fail the new zero-clearance plane gate.

## Follow-up

Rebuild the Panda and combined USD, update only generated checksums, then run PXR/topology/no-snap gates.

## Git refs and key files

- Last verified base: `08cf683`
- Current work ref: working tree
- [verifier](../../Go2Pvcnn/scripts/verify_m1_panda_asset.py)
- [PXR behavior check](../../Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py)
- [static tests](../../Go2Pvcnn/tests/test_m1_panda_asset_static.py)
