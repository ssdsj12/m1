# M1 + Panda Folded-Load Long Curriculum Launch

## Purpose

Launch the fresh guarded folded-load curriculum only after the amended joint4 target passed the complete GPU0 probe and smoke ladder.

## Stage And Todo

- Stage: T400.10b Task 10, L0-C0 start
- Todo: [T400](../todo/T400-m1-panda-force-aware-teacher-student.md)
- Qualification: [PD retune GPU log](2026-08-25-m1-panda-folded-load-pd-retune-gpu.md)
- Runbook: [folded-load locomotion](../../docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md)

## Git Refs

- Target implementation: `f7c1e18`
- Qualification docs: `9314467`
- Current Work Ref: `codex/m1-panda-ppo-stability`

## Command

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 \
  /home/xk/coding/IsaacLab/isaaclab.sh -p \
  scripts/m1_panda_folded_load_curriculum.py \
  --start_stage L0-C0 --num_envs 4096 --max_iterations 600 \
  --device cuda:0 \
  --experiment_root logs/m1_panda_folded_load/foundation-v1 --headless
```

The command runs under `nohup setsid`; the durable curriculum Python PID is recorded in `logs/m1_panda_folded_load/foundation-v1-launch.pid`, and combined output is written to `foundation-v1-launch.log`.

## Initial Evidence

- Curriculum PID: `2660373`
- L0-C0 training PID: `2660477`
- Manifest status: `running`
- Environment count: `4096`
- Requested L0 iterations: `600`
- First update: completed; `1,048,576` timesteps, about `287,157 steps/s`
- GPU0 after startup: about `8.63/12.23 GB`, `65%` utilization
- TensorBoard at update 4:
  - fold error max `0.1850178 rad`
  - joint margin min `0.0866818 rad`
  - effort utilization max `0.2553128`
  - hard failure `0.0`
  - inactive action `0.0`
  - KL `0.0277524`, KL abort `1.0`
  - learning rate `1.5e-5`
  - active std `[0.0050000, 0.0050238]`

## Result

The long task is genuinely running on GPU0 and has passed initialization plus multiple PPO updates without OOM or a physical hard gate. This is a launch/running claim only: `accepted=false`, no eligible checkpoint exists yet, and locomotion convergence is not established.

## Monitoring Contract

Monitor the atomic L0-C0 manifest, process liveness, KL abort/LR/std, fold/margin/effort, inactive action, hard-failure rates, and checkpoint progression. The orchestrator may evaluate and advance only after L0-C0 training eligibility and all seed 42/43/44 fixed evaluations pass; otherwise it records a stopped state and rollback without creating the next stage.
