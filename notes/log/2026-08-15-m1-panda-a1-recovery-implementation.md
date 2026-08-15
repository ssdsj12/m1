# M1 + Panda A1 recovery implementation

## Outcome

T400.7 Tasks 1–6 已按单代理 TDD 实现并通过真实 GPU0 smoke。当前状态是“恢复链路可执行，正式四候选三 seed 筛选待运行”，不表示策略已通过满幅行为验收。

## Static evidence

- ActorCritic 现支持 `scalar` 直值和 `log` 指数两种明确语义；Teacher 使用 scalar、entropy `0`、有效 std floor `0.001`。
- scheduler/wrapper 支持从 checkpoint rollout 进度恢复课程，并记录六个 wrench 轴的 run-level 最大绝对值。
- strict Play 支持 `--full-scale-disturbance` 与原子 JSON summary；sweep 强制 seeds 42/43/44。
- fork 仅允许 A1、新目录、optimizer reset、学习率 `1e-4`，并记录完整 lineage。
- 最终选定静态回归：`183 passed in 1.15s`；八个 Python 文件 `py_compile` exit `0`。

## Real GPU0 evidence

### Full-scale Play smoke

- checkpoint: `model_2700.pt`
- exit: `0`
- 64 env × 500 steps；observation/action `60/16`；curriculum scale `1.0`
- axis maxima: `[19.977552, 19.988186, 19.999401, 4.995606, 4.997717, 4.992351]`
- finite: `true`
- frozen actor SHA: `a7fd58c2753130128f698097eef3159f7f007081f9937f34990a610d8a992457`
- artifact: `/tmp/m1-panda-a1-fullscale-smoke-20260815.json`

500 步 smoke 只验证真实 Isaac/CUDA/summary 链路，不用于 2000 步生存率验收。

### One-iteration recovery fork smoke

第一次 smoke 捕获 runner console 遗留的 `mean_std` NameError；新增 RED 测试后改为 `mean_action_std`，再以新目录重跑成功。

- run: `logs/m1_panda_teacher/a1/a1_recovery_fork_smoke_20260815_v2/`
- source SHA before/after: `5be0d0afe742a318cb706e9b7688a2c4846c2df2b8ad45502246eea36437e2ef`
- final: `model_2701.pt`, SHA `9d487eae56b60698704d2b69ffddad767c81a9cc07828dd8c3723c72c994709b`
- manifest status: `completed`; source/final iteration `2700/2701`
- initial curriculum: step `64800`, scale `0.898`
- optimizer reset: `true`; recovery learning rate: `0.0001`
- saved effective scalar std: min `0.0010000000474974513`, mean `0.002663311315700412`
- runtime: observation/action `60/16`, max wrench `17.69537925720215`
- frozen initial/final SHA: identical `a7fd58c2753130128f698097eef3159f7f007081f9937f34990a610d8a992457`
- stop reason: `block_completed_pending_evaluation`

Isaac 输出仍包含 Panda root joint disjoint-transform snap warning；按已批准 recovery 隔离规则，本阶段不同时修改资产。该资产问题继续留在独立的 T400.6 零间隙工作中。

## Next

后续已完成正式筛选和 20 个 recovery blocks。最佳为 `model_9700.pt`，三 seed timeout/contact/orientation 为 `0.701863/0.222360/0.075776`，未通过 contact 与 timeout 联合门；详情见 [recovery blocks](2026-08-15-m1-panda-a1-recovery-blocks.md)。当前需要新的 reward/curriculum 设计批准，不能把最佳模型声明为 A1 验收通过。
