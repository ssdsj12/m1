# 2026-08-18 M1 + Panda 零间隙资产重建

## Purpose

执行 T400.6 零间隙计划 Task 3：在不改写既有 Panda 资产的前提下，使用 Isaac Sim 5.1 重建组合 USD，并刷新生成物校验和。

## Stage and related todo

- Stage: T400.6 / zero-clearance Task 3
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Plan: [zero-clearance Teacher rebaseline](../../docs/superpowers/plans/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)

## Input conditions

- Baseline ref: `251b7af`
- Python: `/home/xk/miniconda3/envs/go2/bin/python`
- GPU selection: `CUDA_VISIBLE_DEVICES=0`
- Existing Panda SHA-256: `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`
- Previous combined USD SHA-256: `6acbd32afab08dbfb8963e0f7d990d2988cdfe8ad4fec083d0c9fa1c4585c3ff`

## Procedure and evidence

1. The first rebuild exposed an Isaac Sim 5.1 compatibility failure: `RobotAssembler.assemble_articulations` no longer exists. A builder error-path regression test also proved that closing Kit in `finally` could hide failures.
2. Added valid RED tests for reliable nonzero exit, reuse of the checked-in Panda USD by default, and the Isaac Sim 5.1 rigid-body assembler path.
3. The compatibility path now uses `assemble_rigid_bodies`, restores the required Panda root-joint state, removes the second articulation-root APIs, and preserves the single-articulation contract. `--force-panda-conversion` remains available for explicit regeneration.
4. Rebuilt `assets/m1_panda/m1_panda.usd` with exact zero clearance. The source checksum manifest remained byte-identical; the Panda USD SHA-256 remained `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`.
5. New combined USD SHA-256: `8ece30fa9b59400a2eb2ede58289cbe2841b230e13e2427b2a951880ef8f26d3`.
6. `sha256sum -c generated_files.sha256`: `2/2` pass.
7. Asset static suite: `20 passed`, exit `0` (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).
8. Real PXR behavior check: exit `0`; one articulation root `/M1Panda/BASE_LINK`, fixed mount pass, cleanup pass, and mount-plane error `0.0 m` against the independently measured M1-only top plane.

## Result

Task 3 passes. The combined asset is regenerated at zero clearance, the source Panda artifact is unchanged, generated checksums are current, and the serialized PXR topology satisfies the preliminary single-root/mount contract.

## Follow-up

Run the complete Task 4 static/PXR/runtime/relocation/no-snap gates and perform the visual penetration inspection before any C0/C1a Teacher revalidation or Student collection.

## Git refs and key files

- Last verified base: `251b7af`
- Current work ref: working tree
- [builder](../../Go2Pvcnn/scripts/build_m1_panda_asset.py)
- [combined USD](../../Go2Pvcnn/assets/m1_panda/m1_panda.usd)
- [generated checksums](../../Go2Pvcnn/assets/m1_panda/generated_files.sha256)
- [static tests](../../Go2Pvcnn/tests/test_m1_panda_asset_static.py)
