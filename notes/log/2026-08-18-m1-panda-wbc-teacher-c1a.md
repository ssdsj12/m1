# 2026-08-18 M1 + Panda WBC Teacher C1a 验收

## 结论

C1a 平地直线滚动 deterministic Teacher play 已通过 GPU0 正式验收。关闭 Panda 轨迹和默认开启 Panda 六维小幅轨迹的两次 4000-step 运行均 exit `0`、有限、完成五个阶段且 `hard_gates_passed=true`；随后 C0 2000-step 回归也 exit `0`。

本记录只关闭 C1a。它不是 PPO 或 Student 训练，不包含外力域随机化课程、转向、复杂地形、抓取或实机验收；C1b 转向必须另行设计和批准。

## 环境与版本

- 工作目录：`/home/xk/coding/M1/Go2Pvcnn`
- 代码提交：`f582f86 fix: expand C1a longitudinal workspace`
- GPU0：`NVIDIA GeForce RTX 5070`
- 驱动：`580.159.03`
- Isaac Sim：`5.1`
- Python：`/home/xk/miniconda3/envs/go2/bin/python`

## 纯/静态回归

执行命令：

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_wbc_contracts.py \
  tests/test_m1_panda_wbc_kinematics.py \
  tests/test_m1_panda_motion_distribution.py \
  tests/test_m1_panda_qp_backend.py \
  tests/test_m1_panda_standing_wbc.py \
  tests/test_m1_panda_wbc_safety.py \
  tests/test_m1_panda_wbc_teacher.py \
  tests/test_m1_panda_wbc_play_static.py \
  tests/test_m1_panda_rolling_contact.py \
  tests/test_m1_panda_rolling_wbc.py \
  tests/test_m1_panda_rolling_teacher.py \
  tests/test_m1_panda_wbc_roll_play_static.py
```

结果：exit `0`，`184 passed in 2.04s`。

## GPU0 命令与退出码

八步静态 smoke（最终 Jacobian 修正后复验）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 8 --seed 42 \
  --disable-target-motion \
  --summary-json /tmp/m1_panda_wbc_c1a_jfix_static8.json
```

结果：exit `0`；`steps=8`、`finite=true`、四轮接触、QP 可行率 `1.0`、全程 `TRACK`、`exit_reason="smoke_complete"`。短 smoke 不做正式硬门声明，所以 `hard_gates_passed=false` 符合契约。

关闭 Panda 目标运动的正式基线：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 4000 --seed 42 \
  --disable-target-motion --stats-interval 400 \
  --summary-json /tmp/m1_panda_wbc_c1a_no_arm.json
```

结果：exit `0`，`hard_gates_passed=true`。

默认开启 Panda 目标运动的正式联合验收：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 4000 --seed 42 \
  --stats-interval 400 \
  --summary-json /tmp/m1_panda_wbc_c1a_combined.json
```

结果：exit `0`，`hard_gates_passed=true`。

C0 回归：

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 2000 --seed 42 \
  --stats-interval 500 \
  --summary-json /tmp/m1_panda_wbc_c0_post_c1a.json
```

结果：exit `0`，C0 硬门通过。

## C1a 实测指标

| 指标 | 无 Panda 轨迹 | 联合运动 | 门限/要求 | 结果 |
| --- | ---: | ---: | --- | --- |
| steps / completed phases | `4000 / 5` | `4000 / 5` | `4000 / 5` | pass |
| phase counts | `800×5` | `800×5` | 每阶段 `800` | pass |
| finite | `true` | `true` | `true` | pass |
| exit reason | `steps_complete` | `steps_complete` | `steps_complete` | pass |
| forward displacement | `0.5805771215758379 m` | `0.5806883962175199 m` | `> 0` | pass |
| reverse displacement | `-0.18923637974284427 m` | `-0.18903445061500865 m` | `< 0` | pass |
| stop settle time | `0.8 s` | `0.8 s` | `<= 1.0 s` | pass |
| vx RMSE | `0.0006835019361779574 m/s` | `0.0006960664885555469 m/s` | `<= 0.03 m/s` | pass |
| max rolling residual | `0.0016160169363058543 m/s` | `0.0016662825831554645 m/s` | `<= 0.05 m/s` | pass |
| max lateral slip | `0.0012357225318394493 m/s` | `0.0012335278538215793 m/s` | `<= 0.05 m/s` | pass |
| min wheel contacts | `4` | `4` | `4` | pass |
| wheel direction mismatches | `0` | `0` | `0` | pass |
| wheel effort saturations | `0` | `0` | `0` | pass |
| max roll | `0.0014288753736764193 rad` | `0.0017390293069183826 rad` | `<= 0.1745329252 rad` | pass |
| max pitch | `0.0004894154262728989 rad` | `0.0004616501973941922 rad` | `<= 0.1745329252 rad` | pass |
| max EE position error | `0.0016536079598437551 m` | `0.0016413305229732028 m` | `<= 0.03 m` | pass |
| min singular value | `0.18588085905541413` | `0.18345435706147165` | 诊断值 | finite |
| QP feasible rate | `1.0` | `1.0` | `>= 0.999` | pass |
| max QP equality residual | `5.115907697472721e-12` | `1.3642420526593924e-12` | 诊断值 | finite |
| max QP inequality violation | `2.2737367544323206e-13` | `3.126388037344441e-13` | 诊断值 | finite |
| TRACK + SCALE | `4000 / 4000` | `4000 / 4000` | `>= 0.99` | pass |
| HOLD or worse | `0` | `0` | `0` | pass |
| limit/base/non-finite/snap/reset | `0/0/0/0/0` | `0/0/0/0/0` | 全零 | pass |
| max arm target step | `0.00016307830810546875 rad` | `0.00018596649169921875 rad` | 无跳变 | pass |
| hard_gates_passed | `true` | `true` | `true` | pass |

两次正式运行的安全状态均为 `TRACK: 4000`，安全原因为 `safe: 4000`。

## C0 回归指标

`/tmp/m1_panda_wbc_c0_post_c1a.json`：`steps=2000`、`finite=true`、QP 可行率 `1.0`、最大 EE 误差 `0.00022890270361811214 m`、最小奇异值 `0.1861807465116152`、最大 roll/pitch `0.002420169999822974/0.000050468926929170266 rad`、最大侧滑 `0.0009863752638921142 m/s`。关节限位、机身接触、self-collision、reset 和目标跳变均为零；全程 `TRACK/safe`，`exit_reason="steps_complete"`。

## 判定

C1a 的静态 smoke、无臂运动基线、联合 4000-step 正式运行及 post-C1a C0 回归均满足各自契约。下一阶段是需单独书面设计并批准的 C1b 转向，不自动进入 C2/C3、Student、抓取或实机工作。
