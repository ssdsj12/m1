# 2026-08-18 M1 + Panda 零间隙运行时与视觉门

## Purpose

执行 T400.6 零间隙计划 Task 4，并修复视觉检查揭示的全局包围盒安装面错误。

## Stage and related todo

- Stage: T400.6 / zero-clearance Task 4
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Plan: [zero-clearance Teacher rebaseline](../../docs/superpowers/plans/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)

## Failure and correction

首次 GUI 检查由用户指出仍有可见间隙。诊断证明旧计算使用整个 M1 `BASE_LINK` 的全局最高点 `0.12899918854236603 m`，该点位于车体前部；Panda 安装中心下方的局部可见顶板为 `0.0780230313539505 m`，因此形成约 `50.976 mm` 空隙。

增加有效 RED→GREEN 合同后，构建器和独立 verifier 均只测量安装中心 `0.11 m × 0.10 m` 半范围内的可见网格顶面。验证器同时拒绝大于 `1e-6 m` 的正间隙与负穿透；不再以全局包围盒最高点代表安装面。

## Evidence

- Pure/static RED: 局部安装面 helper 和双向 surface-gap predicate 缺失，`2 failed`。
- Runtime PXR RED against previous artifact: visible surface-gap assertion failed。
- Focused GREEN: `2 passed`；asset static GREEN: `22 passed`。
- Final static triple: `37 passed`，exit `0`。
- PXR: one articulation root, fixed mount pass, parent plane error `0.0 m`。
- Local mount top: `0.0780230313539505 m`。
- Panda visible bottom: `0.07802216708660126 m`。
- Visible surface gap: `-8.642673492431641e-07 m`，within `±1e-6 m`。
- CPU PhysX: `25 DOF`, one step, relative mount delta `2.398501783318352e-05 m < 1e-4 m`, no validation errors。
- Relocated full-tree CPU verifier: exit `0`, no remote/outside/unresolved dependencies; only allowlisted `OmniPBR.mdl` resolver boundary。
- Visual gate: GPU0 GUI run; user explicitly confirmed the Panda base and M1 top are attached。
- Panda USD SHA-256 remains `1cb6d489e7cfa44ea06959b652024180ae956fe4fc2ad82c10b1b54293389b51`。
- Accepted candidate combined USD SHA-256: `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`。

## Result

Task 4 passes. Geometry, topology, relocation, no-snap and user-confirmed visual attachment gates are complete. This does not yet accept Teacher behavior or unlock Student collection; C0 and both C1a GPU0 gates remain mandatory.

## Git refs and key files

- Baseline ref: `6afc662`
- Current work ref: working tree
- [builder](../../Go2Pvcnn/scripts/build_m1_panda_asset.py)
- [verifier](../../Go2Pvcnn/scripts/verify_m1_panda_asset.py)
- [PXR behavior](../../Go2Pvcnn/tests/run_m1_panda_asset_pxr_behavior.py)
- [asset tests](../../Go2Pvcnn/tests/test_m1_panda_asset_static.py)
