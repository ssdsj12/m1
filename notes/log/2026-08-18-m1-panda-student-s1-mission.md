# 2026-08-18 M1 + Panda Student S1 部署任务层

## Purpose

执行 T400.9 / Student S1 Task 2，将 C1a 五阶段任务、末端轨迹与 nominal command 从特权 Teacher 中抽取为部署可用的独立层。

## Evidence

- Valid RED: `student_mission` module/API missing，两个测试文件 collection error，exit `2`。
- Mission-focused GREEN: `5 passed, 17 deselected`。
- Full planned regression: Student mission + rolling Teacher + roll play static，`31 passed`，exit `0`。
- `StudentS1Mission` 复用五个 800-step phase、带限末端轨迹和 `0.095 m` 轮半径映射。
- Reset 冻结 settling 后 12 个腿位置和 7 个 Panda 位置；每步生成 `[1,23]` nominal position/velocity，轮 nominal 为 shaped `vx/radius`。
- 相同 seed/reset 可复现；不同 mission 实例的 schedule/trajectory 状态隔离。
- Teacher 新增可选 `mission_sample`；注入时使用外部 phase/pose/twist，不推进私有 schedule；缺省路径保持现有 C1a 行为测试。

## Result

Task 2 passes. Mission 不读取 QP、特权 wrench 或 Teacher 内部状态，可供后续批量编排；当前仍未实现 Student history/model 或 DAgger。

## Authority and links

- Baseline: `db264e8`
- Accepted asset SHA: `643fd0616442a9c45642f81f1f9a5fb484c6e51616cc680fc27e1f8587e78f63`
- [mission](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_mission.py)
- [tests](../../Go2Pvcnn/tests/test_m1_panda_student_mission.py)
- [Student plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md)
