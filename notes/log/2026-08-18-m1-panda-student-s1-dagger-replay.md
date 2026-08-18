# 2026-08-18 M1 + Panda Student S1 DAgger 与回放

## Purpose

执行 T400.9 / Student S1 Task 4，实现确定性 Teacher/Student 动作选择、监督损失和版本化 DAgger 回放。

## Evidence

- Valid RED: `dagger` 与 `m1_panda_student_dataset` 缺失导致两项 collection error，exit `2`。
- Focused GREEN: `24 passed`，exit `0`。
- Student Tasks 1–4 combined: `45 passed`，exit `0`。
- `py_compile` 与 `git diff --check` 均 exit `0`。
- DAgger 使用局部 `torch.Generator(seed + rollout_step)`；安全否决无条件执行 Teacher，概率 `0/1` 边界和固定种子复现均通过。
- 六项标量损失完整：action、wrench、safety、slew、saturation 与加权 total；hard 样本逐样本放大且梯度覆盖四个输出头。
- normal/hard 独立 reservoir 在总容量内保留 `ceil(capacity * hard_fraction)` 个 hard 槽；样本保留 env/episode/step 身份并复制 CPU tensor storage。
- shard 先原子 `torch.save + fsync + os.replace`，再以 canonical sorted JSON 同样发布相邻 manifest；加载严格拒绝 schema、asset SHA、Teacher commit、100/10/23、`0.005 s`、动作尺度和 DAgger stage 不匹配。

## Result

Task 4 通过纯 PyTorch 验证，Student 已具备可复现的在线 DAgger 数据边界；尚未接入 Isaac Lab 批量环境或启动训练。

## Links

- Baseline: `cb9bc42`
- [DAgger selection/loss](../../Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/dagger.py)
- [versioned replay](../../Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_dataset.py)
- [Student plan](../../docs/superpowers/plans/2026-08-18-m1-panda-dagger-student-s1.md)
