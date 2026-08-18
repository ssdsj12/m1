# 2026-08-18 M1 + Panda Student S1 checkpoint

## Purpose

执行 T400.9 / Student S1 Task 5，冻结可恢复 checkpoint 与相邻 manifest 的严格兼容合同。

## Evidence

- Valid RED: 缺少 `m1_panda_student_checkpoint` 模块导致 collection error，exit `2`。
- Focused GREEN: `17 passed`，exit `0`。
- Student Tasks 1–5 combined: `62 passed`，exit `0`。
- `py_compile` 与 `git diff --check` 均 exit `0`。
- checkpoint 严格往返模型、Adam optimizer state 与 `global_step`；省略 optimizer 时支持纯推理加载。
- schema、asset SHA、Teacher commit、dataset SHA、100/10/23、动作尺度、`0.005 s`、DAgger stage/probability、model config 和 loss weights 任一不匹配均拒绝。
- 发布前递归验证模型与 optimizer tensor 有限；加载前验证全部模型 key/shape/finite，损坏输入不会部分覆盖目标模型。
- checkpoint 及 canonical sorted JSON manifest 分别使用临时同级文件、`fsync` 与 `os.replace` 原子发布，测试确认无残留 `.tmp`。

## Result

Task 5 通过。Student S1 的数据集和 checkpoint 均已具备独立版本边界；尚未接入批量 Teacher/Isaac Lab runtime。

## Links

- Baseline: `03b6ea7`
- [checkpoint module](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_checkpoint.py)
- [tests](../../Go2Pvcnn/tests/test_m1_panda_student_checkpoint.py)
- [Student plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md)
