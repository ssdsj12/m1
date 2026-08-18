# M1 + Panda WBC Teacher C1a 运行手册

## 范围

C1a 是平地直线滚动的 deterministic Teacher play。M1 与 Panda 是同一 articulation；Teacher 同时输出轮腿/轮关节与 Panda 关节 effort，在 `TRACK` 下默认执行五段纵向速度任务和小幅、连续、带限的 Panda 六维末端轨迹。

C1a 不是 PPO 训练、Student 训练、外力域随机化课程、转向、复杂地形或抓取任务。它也不授权 C1b、C2、C3、Student、抓取或实机工作。

## GUI Play（默认开启 Panda 目标运动）

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --device cuda:0 --steps 0 --seed 42
```

`--steps 0` 持续运行到关闭 Isaac Sim，不声明 4000-step 正式验收。默认启用 Panda 末端目标运动；只有显式添加 `--disable-target-motion` 才关闭。

## Headless 验证

八步静态 smoke：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 8 --seed 42 \
  --disable-target-motion \
  --summary-json /tmp/m1_panda_wbc_c1a_static8.json
```

关闭 Panda 轨迹的 4000-step 滚动基线：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 4000 --seed 42 \
  --disable-target-motion --stats-interval 400 \
  --summary-json /tmp/m1_panda_wbc_c1a_no_arm.json
```

默认开启 Panda 轨迹的正式联合验收：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 4000 --seed 42 \
  --stats-interval 400 \
  --summary-json /tmp/m1_panda_wbc_c1a_combined.json
```

正式运行必须进程退出码为 `0`、`exit_reason="steps_complete"` 且 JSON 中 `hard_gates_passed=true`。短 smoke 的正常结果是 `exit_reason="smoke_complete"`；它不构成正式验收，因此 `hard_gates_passed=false` 是预期行为。

## 正式硬门

4000 个 mission steps 对应 20 秒，不含启动 settling。全部条件必须同时成立：

- 五个各 800 步的速度阶段全部完成：`0.00, +0.05, +0.10, 0.00, -0.05 m/s`。
- 纵向速度 RMSE `<= 0.03 m/s`；停车后 `<= 1.0 s` 达到 `|vx| <= 0.02 m/s`。
- 前进位移为正、倒车位移为负、轮速方向不匹配计数为零。
- 最大纯滚动残差和最大侧向滑移均 `<= 0.05 m/s`，四轮始终接触。
- 最大绝对 roll/pitch 均 `<= 10 deg`。
- Panda 最大末端位置误差 `<= 0.03 m`。
- QP 可行率 `>= 0.999`；`TRACK + SCALE` 占比 `>= 0.99`。
- `HOLD` 或更严重状态计数为零。
- 关节限位违反、机身触地、非有限值、Panda 目标跳变和意外 reset 均为零。
- `exit_reason == "steps_complete"`。

任一平衡、安全或接触门失败，即使速度或末端误差达标，C1a 仍失败。

## 安全状态

- `TRACK`：正常跟踪底盘速度和 Panda 轨迹。
- `SCALE`：底盘速度与 Panda twist 同时缩放到 50%。
- `HOLD`：冻结 Panda 目标，轮速按限减速度平滑制动。
- `RETRACT`：实测纵向速度低于 `0.02 m/s` 后，Panda 平滑返回安全弯曲姿态。
- `TERMINATE`：持续失稳、非有限值或无法恢复时终止，并锁存原因。

正式 C1a 验收不允许出现 `HOLD`、`RETRACT` 或 `TERMINATE`。

## C0 回归

修改 C1a 后应复跑驻停 C0：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 2000 --seed 42 \
  --stats-interval 500 \
  --summary-json /tmp/m1_panda_wbc_c0_post_c1a.json
```

实测验收记录见 [2026-08-18-m1-panda-wbc-teacher-c1a.md](../../../notes/log/2026-08-18-m1-panda-wbc-teacher-c1a.md)。
