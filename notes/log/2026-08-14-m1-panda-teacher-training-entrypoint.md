# M1 + Panda Teacher Training Entrypoint

## Purpose

实现 A0/A1 独立 PPO 配置工厂、严格 CLI/preflight、checkpoint/resume 校验、原子 manifest 和训练资源清理顺序。

## Stage

T400.5b Task 6 / PPO config and training entrypoint。

## Related Todo

- [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)

## RED Evidence

- 配置、导出、脚本和纯 CLI helper 首轮有效 RED：`13 failed`，原因分别为目标模块/脚本不存在。
- 主训练流静态契约第二轮 RED：`2 failed, 14 passed`，原因是 resume log-dir helper 和 simulator training flow 尚未实现。

## GREEN Evidence

- training static + checkpoint contract：`49 passed in 0.93s`。
- config/script `py_compile`：exit `0`。

## Contracts Verified

- PPO 工厂每次返回独立深层结构，固定 24 steps、100 save interval、256/128 actor/critic、0.01 initial std、5 epochs、4 mini-batches。
- A0 拒绝 base checkpoint；A1 强制 base checkpoint；所有 smoke overrides 必须为正数。
- 新 run 目录拒绝覆盖；resume 复用 checkpoint 父目录并禁止另给 run name。
- A1 在 wrapper 前严格加载 frozen A0 actor；resume 在 `runner.load` 前校验 stage、60/16 shape、hidden dims、base hash 和 optimizer 合同。
- manifest 在 `runner.learn` 前原子写入，成功/失败均更新状态；A1 训练后复核 frozen hash。
- `OnPolicyRunner` 接收 train cfg 深拷贝，避免其 `pop(class_name)` 污染 manifest/YAML 配置。
- 清理顺序固定为 env 先关闭、simulation app 后关闭。

## Result

通过静态/纯 Python 验收。下一步使用真实 Isaac CPU 四段 smoke 验证 A0 initial/resume 与 A1 initial/resume。

## Git Refs

- Baseline Ref: unavailable
- Candidate Ref: filesystem working copy
- Git Ref: unavailable
- Key Files:
  - [PPO config](../../Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py)
  - [Training entrypoint](../../Go2Pvcnn/scripts/m1_panda_teacher_train.py)
  - [Tests](../../Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py)
