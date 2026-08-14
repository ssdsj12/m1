# M1 Original Small-Obstacle Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register and verify an M1 stage-two task that uses the original Go2Pvcnn semantic flat-small obstacle environment.

**Architecture:** Subclass the locally mirrored original flat-small environment and override only M1-specific robot, action, selector, reward, and reset contracts. Keep original semantic course, curriculum, scanner, commands, and MPC teacher unchanged.

**Tech Stack:** Python 3.11, Isaac Lab, Isaac Sim, Gymnasium, RSL-RL, pytest.

## Global Constraints

- Training actions remain 12 leg positions plus 4 wheel velocities.
- Stage one and stage two-A accepted checkpoints remain unchanged.
- The long-term stage-two task must not spawn the custom deterministic obstacle bar.
- Runtime uses only the `go2_pvcnn` package under `M1/Go2Pvcnn`.

---

### Task 1: Static Environment Contract

**Files:**
- Create: `Go2Pvcnn/tests/test_m1_original_small_obstacle_env.py`

**Interfaces:**
- Consumes: original `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg`
- Produces: regression contract for M1 train/play config and task IDs

- [ ] Write tests asserting inheritance, original terrain factory reuse, semantic scanner attachment to `BASE_LINK`, `M1_CFG`, hybrid actions, M1 selectors, and no `obstacle` scene member.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q Go2Pvcnn/tests/test_m1_original_small_obstacle_env.py` and confirm collection fails because the M1 semantic config does not exist.

### Task 2: M1 Semantic Flat-Small Config

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_pvcnn_small_obstacle_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`

**Interfaces:**
- Produces: `M1PvcnnFlatSmallAvoidanceEnvCfg` and `M1PvcnnFlatSmallAvoidanceEnvCfg_PLAY`
- Produces: `Isaac-M1-Pvcnn-Flat-Small-Avoidance-v0` and play task registration

- [ ] Implement a scene subclass using the original semantic terrain/importer/scanner and `M1_CFG`.
- [ ] Implement an environment subclass using `M1SmokeActionsCfg`, M1 body/joint selectors, M1-safe reset and command ranges, and existing wave wrapper flags.
- [ ] Implement a play subclass with original play-mode teacher/reward/curriculum disabling behavior.
- [ ] Register both task IDs.
- [ ] Run the focused test until it passes, then run all M1 unit tests.

### Task 3: Runtime Smoke Verification

**Files:**
- Modify only if runtime evidence reveals a concrete config mismatch.

**Interfaces:**
- Consumes: registered M1 semantic task
- Produces: evidence for environment creation, 16 actions, semantic observations, and stable stepping

- [ ] Launch the training task headlessly with 4 environments and a short step budget.
- [ ] Confirm semantic obstacle prims exist and observations contain the 16x16 elevation/semantic map.
- [ ] Confirm all four wheel actions map to equal forward velocity while wave leg actions remain bounded.
- [ ] Run the complete pytest suite relevant to M1 task registration and curriculum.

### Task 4: Entrypoints And Notes

**Files:**
- Modify: `notes/human/human-02-training-and-entrypoints.md`
- Modify: `notes/human/human-01-overall-pipeline.md`

**Interfaces:**
- Produces: reproducible train/play commands and checkpoint promotion sequence

- [ ] Document the new task IDs and initialization from `stage2a_wave_flat/accepted_cylinder.pt`.
- [ ] Record that deterministic 1/5/10 mm tasks are diagnostics only.
- [ ] Verify every documented path and CLI flag against `m1_train.py` and `m1_play.py --help`.
