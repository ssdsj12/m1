# M1 Rolling Wave Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split M1 smoke actions into 12 leg position controls and 4 wheel velocity controls, then expose rolling and wave mode parameters.

**Architecture:** Keep the change local to the M1 smoke path. Asset constants describe the M1 joint split; the smoke env consumes those constants in two IsaacLab action terms and exposes mode parameters for later controllers.

**Tech Stack:** Python, IsaacLab `JointPositionActionCfg`, IsaacLab `JointVelocityActionCfg`, pytest static contract tests.

## Global Constraints

- Do not modify the original Go2 MPC environment behavior.
- Keep M1 as a smoke/control-interface adaptation, not a complete learned policy.
- Use the known M1 USD at `/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd`.

---

### Task 1: Asset Joint Split

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/assets/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_asset_static.py`

**Interfaces:**
- Produces: `M1_LEG_JOINT_NAMES`, `M1_WHEEL_JOINT_NAMES`, `M1_ROLLING_MODE`, `M1_WAVE_MODE`.

- [ ] **Step 1: Write the failing test**

Add assertions that the leg tuple has 12 non-foot joints, the wheel tuple has 4 foot joints, and the mode constants are `rolling` and `wave`.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/xk/miniconda3/etc/profile.d/conda.sh && conda activate env1 && PYTHONPATH=/home/xk/coding/M1/Go2Pvcnn PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Go2Pvcnn/tests/test_m1_asset_static.py -q`

- [ ] **Step 3: Write minimal implementation**

Add constants derived explicitly from the known M1 joint names.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command and expect all tests in the file to pass.

### Task 2: Hybrid Action Contract

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_smoke_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_m1_smoke_cfg_static.py`

**Interfaces:**
- Consumes: `M1_LEG_JOINT_NAMES`, `M1_WHEEL_JOINT_NAMES`, `M1_ROLLING_MODE`, `M1_WAVE_MODE`.
- Produces: `M1SmokeActionsCfg.leg_pos`, `M1SmokeActionsCfg.wheel_vel`, and M1 rolling/wave env parameters.

- [ ] **Step 1: Write the failing test**

Assert `JointPositionActionCfg` uses `M1_LEG_JOINT_NAMES`, `JointVelocityActionCfg` uses `M1_WHEEL_JOINT_NAMES`, and mode parameters are present.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/xk/miniconda3/etc/profile.d/conda.sh && conda activate env1 && PYTHONPATH=/home/xk/coding/M1/Go2Pvcnn PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest Go2Pvcnn/tests/test_m1_smoke_cfg_static.py -q`

- [ ] **Step 3: Write minimal implementation**

Replace the single `joint_pos` action with `leg_pos` and `wheel_vel`; add rolling and wave fields to `M1SmokeEnvCfg`.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command and expect all tests in the file to pass.

### Task 3: Usage Documentation

**Files:**
- Modify: `Go2Pvcnn/docs/M1_SMOKE_USAGE.md`

**Interfaces:**
- Consumes: `Isaac-M1-Smoke-v0`, `M1SmokeEnvCfg`, action split names.

- [ ] **Step 1: Document the action split**

Update the status section to say M1 smoke now exposes 12 leg position actions plus 4 wheel velocity actions.

- [ ] **Step 2: Verify docs and Python syntax**

Run static tests and `py_compile` for touched Python files.
