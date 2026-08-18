# 2026-08-18 M1 + Panda 零间隙构建器合同

## Purpose

执行 T400.6 零间隙计划 Task 1，把组合资产构建源的人工 clearance 从 `0.01 m` 冻结为 `0.0 m`，但本记录不声称 USD 已重建或动力学已复验。

## Stage and related todo

- Stage: T400.6 / zero-clearance Task 1
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Plan: [zero-clearance Teacher rebaseline](../../docs/superpowers/plans/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)

## Input conditions

- Baseline ref: `d6e1b60`
- Candidate ref: working tree
- Existing unrelated modification: `graphify-out/cache/last_query_stamp`，未暂存
- Python: `/home/xk/miniconda3/envs/go2/bin/python`

## Procedure and evidence

1. Baseline asset static suite: `16 passed`，exit `0`。
2. Added AST/pure helper test; RED failed because `MOUNT_CLEARANCE_M` was `0.01` instead of `0.0`。
3. Set exact `MOUNT_CLEARANCE_M = 0.0`, added finite/nonnegative `mount_offset_z(...)`, and routed RobotAssembler offset through it。
4. Focused GREEN: `1 passed`，exit `0`。
5. Full asset static GREEN: `17 passed`，exit `0`。

## Result

Task 1 source contract passes. Generated `m1_panda.usd` and checksum are intentionally unchanged until Task 2 adds an independent mount-plane verifier and Task 3 rebuilds the asset.

## Follow-up

Execute Task 2 RED→GREEN for parent-side mount position and independent M1-only top-plane measurement. Student collection remains locked.

## Git refs and key files

- Last verified base: `d6e1b60`
- Current work ref: working tree
- [builder](../../Go2Pvcnn/scripts/build_m1_panda_asset.py)
- [asset static tests](../../Go2Pvcnn/tests/test_m1_panda_asset_static.py)
