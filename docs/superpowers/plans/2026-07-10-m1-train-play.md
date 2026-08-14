# M1 Train Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add M1-specific train and play scripts that use `Isaac-M1-Smoke-v0`.

**Architecture:** Add a small M1 PPO config, a small RSL-RL wrapper for the single `policy` observation group, and two entrypoint scripts. `m1_play.py` supports open-loop rolling/wave and checkpoint policy playback.

**Tech Stack:** Python, IsaacLab, Gymnasium, RSL-RL PPO, PyTorch.

## Global Constraints

- Use conda env `go2pvcnn_ablation`.
- Do not change original Go2 train/play behavior.
- Use task id `Isaac-M1-Smoke-v0`.

---

### Task 1: M1 Train Config

- [ ] Test static config exists with `get_m1_train_cfg`, `ActorCritic`, and PPO fields.
- [ ] Add `agent/m1_train_cfg.py`.

### Task 2: M1 RSL-RL Wrapper

- [ ] Test wrapper source flattens `obs_dict["policy"]` and supplies critic observations.
- [ ] Add `go2_pvcnn/tasks/m1_rsl_rl_wrapper.py`.

### Task 3: M1 Train Script

- [ ] Test script uses `Isaac-M1-Smoke-v0`, `get_m1_train_cfg`, `M1RslRlEnvWrapper`, `OnPolicyRunner`, and saves under `logs/m1_smoke`.
- [ ] Add `scripts/m1_train.py`.

### Task 4: M1 Play Script

- [ ] Test script supports `--checkpoint`, `build_m1_smoke_action`, `get_inference_policy`, and `runner.load`.
- [ ] Add `scripts/m1_play.py`.

### Task 5: Verification

- [ ] Run M1 static tests.
- [ ] Run `py_compile`.
- [ ] Run short M1 open-loop play smoke.
