# M1 + Panda Coordinated PPO Learning Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the runner-only 67-observation/smoke-reward prerequisite with a learnable 103-observation coordinated PPO task, pass short GPU gates, and start an isolated GPU0 long run.

**Architecture:** Add one focused MDP module for coordinated targets, observations and rewards; wire it through a dedicated environment config while preserving the 23-effort action boundary. Extend the wrapper/training entrypoint only for action clamping, manifest diagnostics and monitoring.

**Tech Stack:** Python 3.11, PyTorch 2.7, Isaac Sim/Isaac Lab 5.1, Gymnasium, RSL-RL PPO, pytest, GPU0.

## Global Constraints

- Single-agent inline execution; never stage `graphify-out/` or `assets/m1_panda/m1.zip`.
- Asset, six-axis wrench ordering, physics dt and 23-action joint order remain unchanged.
- Student S1, grasping and hardware remain out of scope.
- Follow TDD for every behavior change.

---

### Task 1: Coordinated observation and reward MDP

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_coordinated.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_mdp.py`
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/__init__.py`

**Interfaces:**
- Produce `coordinated_base_target_error_b`, `coordinated_ee_pose_error_b`, `coordinated_desired_twist_b`, `coordinated_wheel_contact`, `coordinated_base_tracking_reward`, `coordinated_folded_arm_error`, and `coordinated_ee_tracking_reward`.
- Lazily freeze reset-hand pose per environment and add a reachable configured offset; validate finite tensors and exact widths.

- [ ] Write fake-env tensor tests for target frames, 3/6/6/4 widths, arrival gating, reward signs and non-finite rejection.
- [ ] Run focused pytest and confirm missing API RED.
- [ ] Implement only the listed pure/batched MDP functions.
- [ ] Run focused GREEN and compile.
- [ ] Commit `feat: add coordinated M1 Panda PPO objectives`.

### Task 2: Freeze 103-observation environment and bounded action

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_learning_env_static.py`

**Interfaces:**
- Policy observation is exactly `103`; controlled joint order is the canonical 23; action is clamped to `[-1,1]` before `env.step`.
- Rewards expose base target, folded arm, EE tracking, balance, slip, action-rate and effort terms; existing termination terms remain.

- [ ] Write failing static/config tests for exact term order, width, reward names and action clamp.
- [ ] Run RED.
- [ ] Wire the MDP terms and action clamp without changing asset or task ID.
- [ ] Run GREEN plus coordinated regressions and compile.
- [ ] Commit `feat: make coordinated M1 Panda PPO learnable`.

### Task 3: GPU smoke, sanity block and long-run launch

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_coordinated_train.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_coordinated_train_static.py`
- Create: `notes/log/2026-08-23-m1-panda-coordinated-ppo-short-gates.md`
- Modify: `notes/log/index.md`, `notes/todo.md`, `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Training writes an immutable run manifest with observation/action widths, asset SHA, seed, run name and initialization lineage.
- `--init-a1-checkpoint` remains lineage-only; incompatible 67-observation coordinated smoke checkpoints cannot resume.

- [ ] Write failing static tests for manifest, fresh-run boundary, action/observation diagnostics and periodic save.
- [ ] Implement minimal CLI/manifest diagnostics and run local GREEN.
- [ ] Run GPU0 8-env×1-iteration fresh smoke; require 103/23, finite PPO and checkpoint.
- [ ] Run GPU0 64-env×100-iteration sanity; inspect reward/reset/base/EE metrics and stop on non-finite or catastrophic failures.
- [ ] If sanity passes, launch an isolated GPU0 64-env long run with at least 5000 requested iterations and save interval 100.
- [ ] Monitor the process and TensorBoard/checkpoint artifacts; record exact PID, command, run directory and first stable metrics.
- [ ] Align notes and commit `train: start coordinated M1 Panda Teacher long run`.
