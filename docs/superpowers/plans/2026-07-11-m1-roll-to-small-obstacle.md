# M1 Roll To Small Obstacle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically validate stable M1 rolling and progress into wheel-assisted small-obstacle crossing with Go2Pvcnn perception.

**Architecture:** Preserve the M1 16-action control contract across stages. Add deterministic stage evaluation, then introduce bounded leg control, low-obstacle terrain and teacher height scans before transferring perception to PVCNN.

**Tech Stack:** Python, IsaacLab, Gymnasium, RSL-RL PPO, PyTorch, pytest.

## Global Constraints

- Use conda environment `go2pvcnn_ablation`.
- Keep original Go2 task behavior unchanged.
- Four wheel actions must remain equal and negative for robot-forward motion.
- Never promote a checkpoint without a deterministic evaluation report.

### Task 1: Stage 1 Evaluator And Promotion

- [ ] Write failing tests for metric thresholds, JSON report fields, and checkpoint discovery.
- [ ] Implement pure evaluation helpers and `m1_checkpoint_eval.py`.
- [ ] Run unit tests and a headless evaluation of the latest roll checkpoint.
- [ ] Promote the best passing checkpoint to `logs/m1_curriculum/stage1_roll/accepted.pt`.

### Task 2: Flat Wave Transition

- [ ] Write failing tests for fixed equal wheels and bounded leg residuals.
- [ ] Extend the wrapper with a wheel-assisted wave action mode.
- [ ] Add and register `Isaac-M1-Wave-Flat-v0` with the same observation shape as roll.
- [ ] Load the accepted roll checkpoint, run a short resume smoke, and verify stability.

### Task 3: Small Obstacle Teacher Environment

- [ ] Write failing tests for terrain levels, height scanner, crossing rewards, and task registration.
- [ ] Add 0.02/0.03/0.04 m obstacle terrain and local height-scan observations.
- [ ] Add progress, clearance, collision, orientation, and crossing-success metrics.
- [ ] Run environment and short-training probes before starting long training.

### Task 4: PVCNN Perception Transfer

- [ ] Add tests for teacher/PVCNN observation shape and finite-output contracts.
- [ ] Reuse the existing PVCNN wrapper and cost-map path with M1 sensor geometry.
- [ ] Train the PVCNN adapter against teacher height scans.
- [ ] Fine-tune small-obstacle locomotion and compare against teacher evaluation.

### Task 5: Autonomous Controller And Documentation

- [ ] Add the stage controller with restart, retry, promotion, and logging behavior.
- [ ] Add train/play commands and acceptance criteria to human documentation.
- [ ] Run relevant tests, syntax checks, and headless stage probes.
