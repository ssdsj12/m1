# 2026-08-14 M1 + Panda Teacher A0/A1 Play 入口

## Scope

新增独立 `scripts/m1_panda_teacher_play.py`，忠实复现 Teacher A0/A1 wrapper 和 checkpoint 边界，不复用通用 `m1_play.py`。

## RED

新静态/纯 helper 测试在入口文件缺失时为 `13 failed`，失败原因是预期的 `FileNotFoundError`。

## GREEN

- CLI 支持 A0/A1、当前 checkpoint、A1 base checkpoint、环境数、seed、steps、统计周期和显式零扰动。
- A1 在 runner load 前计算 base SHA、strict load frozen A0 actor，并校验 A1 manifest base hash。
- 当前 checkpoint 在 runner load 前校验 stage、60/16、hidden dims 和实际 tensor shape，不要求 optimizer。
- 推理使用 `torch.inference_mode()`；`--steps 0` 跟随 SimulationApp，正 steps 有界。
- 每个统计周期输出 reward/done、六轴 wrench max、历史 wrench max 和可用 reset reason。
- 退出时验证 frozen actor 未变，先 close env 再 close SimulationApp。
- source scan 确认没有 `learn`、manifest build/write。

## Verification

```text
play focused: 13 passed
play + wrapper + checkpoint + train static: 97 passed
py_compile: exit 0
forbidden-write scan: exit 0
```

测试环境说明：go2 环境的 site-packages 包含新版 `rsl_rl`，checkpoint 测试必须通过 `PYTHONPATH=/home/xk/coding/M1/Go2Pvcnn/rsl_rl` 绑定项目版本；否则会在测试 fixture 构造旧式 ActorCritic 时产生 23 个环境接口失败。入口自身也在启动时把项目 `rsl_rl` 放到 `sys.path` 首位。

## Git Refs

- Branch: `main`
- Base HEAD observed at final handoff: `8872421d02eb93b04b150d025148c8a93e78dd09`
- Current Work Ref: uncommitted working tree
