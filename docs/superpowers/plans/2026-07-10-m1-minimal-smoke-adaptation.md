# M1 Minimal Smoke Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated M1 smoke adaptation of the Go2Pvcnn project.

**Architecture:** Copy the Go2Pvcnn repository into `/home/xk/coding/M1`, then add M1-specific asset and task config beside the existing Go2 modules. The M1 smoke path avoids MPC and policy checkpoint compatibility so the first verification target is only IsaacLab loading, reset, and stepping.

**Tech Stack:** Python, IsaacLab ManagerBasedRLEnv, USD articulation asset, pytest.

## Global Constraints

- Keep existing Go2 semantic MPC files usable.
- Use the M1 physics USD as the robot asset entrypoint.
- Do not use old Go2 policy checkpoints for M1 smoke.
- Do not adapt MPC in this first pass.

---

### Task 1: Copy Source Project

**Files:**
- Create: `/home/xk/coding/M1/Go2Pvcnn`
- Create: `/home/xk/coding/M1/assets`
- Create: `/home/xk/coding/M1/docs`
- Create: `/home/xk/coding/M1/notes`
- Create: `/home/xk/coding/M1/raw`

**Steps:**
- [ ] Copy the source repository contents into `/home/xk/coding/M1`.
- [ ] Preserve the M1 spec and plan files already created in `/home/xk/coding/M1/docs/superpowers`.

### Task 2: Add M1 Asset Config

**Files:**
- Modify: `/home/xk/coding/M1/Go2Pvcnn/go2_pvcnn/assets/__init__.py`
- Test: `/home/xk/coding/M1/Go2Pvcnn/tests/test_m1_asset_static.py`

**Steps:**
- [ ] Write failing static tests for `M1_CFG`.
- [ ] Add M1 constants and `M1_CFG`.
- [ ] Run tests until green.

### Task 3: Add M1 Smoke Task

**Files:**
- Create: `/home/xk/coding/M1/Go2Pvcnn/go2_pvcnn/tasks/m1_smoke_env_cfg.py`
- Modify: `/home/xk/coding/M1/Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
- Test: `/home/xk/coding/M1/Go2Pvcnn/tests/test_m1_smoke_cfg_static.py`

**Steps:**
- [ ] Write failing static tests for the M1 Gym id and core config contracts.
- [ ] Add the no-MPC M1 smoke task config.
- [ ] Register `Isaac-M1-Smoke-v0`.
- [ ] Run tests until green.

### Task 4: Verify Runtime Smoke

**Files:**
- No source edits unless the smoke exposes a config bug.

**Steps:**
- [ ] Run a short static import/registration check.
- [ ] Run an IsaacLab one-env smoke if startup succeeds in the current machine context.
