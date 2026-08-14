# M1 + Panda Teacher Disturbance Scheduler

## Purpose

实现 A0/A1 每环境独立的六维扰动课程，并把已验证 probe 中的 BASE_LINK→body-local wrench 转换和 Isaac Lab 2.1 clear shim 提升为训练可复用接口。

## Stage

T400.5b Task 1 / pure PyTorch disturbance foundation。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## RED Evidence

1. 配置测试首次收集失败：`ModuleNotFoundError: go2_pvcnn.tasks.m1_panda_teacher`。
2. scheduler 测试收集失败：缺少 `M1PandaDisturbanceScheduler`。
3. wrench/clear 测试收集失败：缺少 `base_wrench_to_body_local`。

三轮失败均发生在相应生产接口写入之前，且原因与预期一致。

## GREEN Evidence

- 配置与非法参数：`20 passed`。
- seeded scheduler、课程、duration、hold/ramp/pulse、reset：`41 passed`。
- 加入 wrench 转换和 clear shim 后 focused：`51 passed`。
- Task 1 + foundation wrench/probe 回归：`77 passed in 0.89s`。
- 生产/测试 `py_compile`：exit `0`。
- 占位符扫描：无匹配。

## Contracts Verified

- A0：每轴 `±10 N`、`±2 Nm`、`1–2 s`、50,000-step 课程。
- A1：每轴 `±20 N`、`±5 Nm`、`0.25–1 s`、75,000-step 课程。
- A1 hold/ramp/pulse 概率为 `0.50/0.30/0.20`，pulse leading fraction 为 `0.20`。
- curriculum scale 从 `0.25` 单调到 `1.0`。
- reset 只清除指定环境，不回退全局课程进度。
- live BASE_LINK/body quaternion 转换与 probe 既有旋转样例一致。
- clear shim 只吞掉已知 `[0] -> [N,3]` Isaac Lab 2.1 错误，其他异常继续抛出。

## Result

通过。probe 改为导入公共 helper，七行真实 authority 的算法、case、窗口和输出契约未改变。

## Limitations

本任务没有创建 Isaac Lab Teacher 环境或运行新的真实仿真；外力在训练 wrapper 内的实际调用顺序属于后续 Task 4/CPU smoke 验收。

## Follow-up

执行 checkpoint/manifest Task 2。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [Scheduler](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py)
  - [Tests](../../Go2Pvcnn/tests/test_m1_panda_teacher_disturbance.py)
  - [Probe](../../Go2Pvcnn/scripts/m1_panda_wrench_probe.py)
  - [Plan](../../docs/superpowers/plans/2026-08-14-m1-panda-teacher-a0-a1-training.md)
