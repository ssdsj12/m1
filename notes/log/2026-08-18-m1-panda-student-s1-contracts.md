# 2026-08-18 M1 + Panda Student S1 合同

## Purpose

执行 T400.9 / Student S1 Task 1，冻结可部署 Student 的观测、历史和安全残差动作边界。

## Authority

- Teacher baseline commit: `72a2700`
- Accepted combined USD SHA-256: `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`
- [Student plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md)

## Evidence

- Valid RED: missing `student_contracts` module caused import failure, exit `2`。
- GREEN: `tests/test_m1_panda_student_contracts.py` reported `9 passed`，exit `0`。
- Frozen dimensions: observation `100`, history `10`, action `23 = 12 legs + 4 wheels + 7 Panda arm`。
- Teacher labels encode leg/arm position residuals and wheel velocity residuals; safe in-range labels reconstruct the original Teacher targets exactly。
- Runtime reconstruction clamps normalized amplitude and applies physical per-step slew limits independently to legs, wheels and arm。
- Diagnostics expose the actually applied normalized action and per-channel saturation mask。
- Wrong width, non-finite value, dtype and device mismatch fail before output mutation。

## Result

Task 1 passes. This is a pure PyTorch contract only; it does not yet create observations, a Student network, replay data, DAgger rollout or GPU training.

## Key files

- [contracts](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_contracts.py)
- [tests](../../Go2Pvcnn/tests/test_m1_panda_student_contracts.py)
