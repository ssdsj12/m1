# M1 + Panda Coordinated Mission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic combined M1 + Panda mission that navigates with Panda folded, transitions to end-effector pose/trajectory tracking, and authorizes bounded M1 motion when Panda null-space margin is insufficient.

**Architecture:** Add a pure-PyTorch mission/coordinator layer above the existing `M1PandaWbcTeacher` and `distribute_motion` interfaces. The mission owns phase transitions and folded/unfolded targets; the coordinator owns hysteretic null-space assistance and base bounds; the Isaac adapter supplies combined-articulation state and records diagnostics. Student S1 remains unchanged until Teacher runtime gates pass.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab 5.1, Gymnasium, existing RSL-RL/WBC modules, pytest.

## Global Constraints

- Use the combined `M1_PANDA_CFG`; never use `Isaac-M1-Walk-v0` for this mission.
- Preserve the 16-channel M1 boundary, 7-channel Panda arm boundary, and 23-channel combined action.
- Panda end-effector tracking and safety are lexicographically higher priority than M1 motion.
- M1 assistance is planar, low-speed, acceleration-limited, and bounded around the arrived pose.
- Student S1, grasping, force control, and maximum-load hardware tests are out of scope.
- Every task follows TDD: failing focused test, minimal implementation, focused pass, then commit.

---

### Task 1: Mission contracts and phase machine

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/coordinated_mission.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_mission.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py`

**Interfaces:**
- `MissionPhase` enum values: `FOLD_AND_NAVIGATE`, `ARRIVE_HOLD`, `UNFOLD_AND_TRACK`, `COORDINATED_TRACK`.
- `CoordinatedMissionCfg` fields: `physics_dt`, `arrive_position_tolerance_m`, `arrive_yaw_tolerance_rad`, `settled_steps`, `folded_arm_target[7]`, `unfold_duration_s`, `base_assist_radius_m`.
- `CoordinatedMissionState` contains `phase`, `step`, `target_base_pose[3]`, `arm_target[7]`, `ee_target_pose[6]`, `ee_target_twist[6]`, `settled_count`.
- `CoordinatedMission.reset(base_pose, ee_pose, target_base_pose, ee_target_pose, seed)` and `step(base_pose, ee_pose, ee_target_pose, ee_target_twist) -> CoordinatedMissionState`.

- [ ] Write failing tests for deterministic reset, phase order, per-environment isolation, folded target interpolation, settled-window arrival, and rejection of non-finite/incorrect shapes.
- [ ] Run `pytest Go2Pvcnn/tests/test_m1_panda_coordinated_mission.py -q`; expected initial import/contract failures.
- [ ] Implement explicit monotonic transitions. `FOLD_AND_NAVIGATE` reaches `ARRIVE_HOLD` only after position/yaw tolerances hold for `settled_steps`; `ARRIVE_HOLD` then interpolates to the first arm target; `COORDINATED_TRACK` is entered only after interpolation completes.
- [ ] Re-run the focused test; expected all mission contract tests pass.
- [ ] Commit: `feat: add coordinated M1 Panda mission state machine`.

### Task 2: Folded navigation and bounded base-assist coordinator

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/base_assist.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_base_assist.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py`

**Interfaces:**
- `BaseAssistCfg(max_speed_xy, max_yaw_rate, max_accel_xy, max_yaw_accel, max_displacement_xy, enable_margin, disable_margin, minimum_sigma, radius_gain)`.
- `BaseAssistDecision(base_velocity[3], active, reason, arm_margin_before, arm_margin_after)`.
- `compute_base_assist(...) -> BaseAssistDecision`, using current planar pose, target planar pose, arm margin/sigma, and the existing coordinated Jacobian/distribution result.

- [ ] Write failing tests proving: folded navigation commands bounded planar motion; assist is inactive above the enable margin; hysteresis prevents chatter; displacement/rate limits are hard; an improving assist is preferred over unnecessary base motion; non-finite metrics return a zero safe command.
- [ ] Run the focused tests and record the expected failures.
- [ ] Implement the coordinator as a pure tensor function. Use `distribute_motion(..., prescribed_base_velocity=...)` for a selected base command, preserve Panda-first solving, and expose before/after margins.
- [ ] Add only the minimum `MotionDistributionCfg` option needed to keep base assistance disabled during `UNFOLD_AND_TRACK` and enabled during `COORDINATED_TRACK`.
- [ ] Run `pytest Go2Pvcnn/tests/test_m1_panda_base_assist.py -q`; expected pass with finite outputs and exact bounds.
- [ ] Commit: `feat: add bounded M1 null-space base assistance`.

### Task 3: Integrate mission and assistance into the deterministic Teacher

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_teacher.py`

**Interfaces:**
- Extend `TeacherCfg` with `coordinated_mission: bool`, `mission_cfg`, `base_assist_cfg`, and `arm_tracking_enabled` defaults that preserve existing C0/C1a behavior when false.
- Extend `TeacherState` with optional `base_pose[3]`, `target_base_pose[3]`, `mission_ee_target_pose[6]`, and `mission_ee_target_twist[6]`; reject partial mission inputs when coordinated mode is enabled.
- Extend `TeacherCommand` diagnostics with `mission_phase`, `base_assist_active`, `base_assist_reason`, `arm_margin_before`, and `arm_margin_after`.

- [ ] Write failing tests for coordinated reset, folded arm target during navigation, no base assistance before tracking, bounded assistance during near-limit tracking, safety override precedence, and unchanged legacy C0/C1a path.
- [ ] Run focused tests and capture the initial failures.
- [ ] In `step`, sample the mission first, derive folded/transition/EE targets, call base assist only in `COORDINATED_TRACK`, then pass the resulting prescribed planar velocity into the existing motion distributor/WBC path.
- [ ] Ensure safety HOLD/TERMINATE zeroes assistance and preserves the existing finite-target fallback.
- [ ] Run `pytest Go2Pvcnn/tests/test_m1_panda_coordinated_teacher.py -q` plus the existing Teacher/WBC focused tests; expected all pass.
- [ ] Commit: `feat: integrate coordinated mission with Panda Teacher`.

### Task 4: Combined Isaac environment and command adapter

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_coordinated_env_cfg.py`
- Create: `Go2Pvcnn/scripts/m1_panda_coordinated_play.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_env_static.py`

**Interfaces:**
- Gym ID: `Isaac-M1-Panda-Coordinated-v0`.
- Config class: `M1PandaCoordinatedEnvCfg`, derived from `M1PandaWbcRollTeacherEnvCfg`, using `M1_PANDA_CFG`, `episode_length_s=30.0`, and explicit mission target fields.
- CLI accepts `--num_envs`, `--seed`, `--max_steps`, `--headless`, `--device`, `--target-base-pose`, and `--ee-target-pose`; outputs JSON diagnostics with phase counts, arrival error, EE error, assistance count, safety failures, and reset count.

- [ ] Write static tests for the exact Gym registration string, combined asset config, 23-channel boundary, target argument parsing, and diagnostics keys.
- [ ] Run the static tests and verify they fail before registration/config implementation.
- [ ] Implement the config and adapter by reusing the existing combined scene, observation terms, WBC state extraction, safety supervisor, and action application. Do not duplicate the Panda asset or WBC solver.
- [ ] Run `pytest Go2Pvcnn/tests/test_m1_panda_coordinated_env_static.py -q`; expected pass.
- [ ] Commit: `feat: add combined coordinated mission Isaac entrypoint`.

### Task 5: Runtime validation and notes

**Files:**
- Create: `Go2Pvcnn/tests/test_m1_panda_coordinated_runtime_contract.py`
- Create: `notes/log/2026-08-20-m1-panda-coordinated-mission-smoke.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

- [ ] Add a CPU contract smoke with at least four independent environments, asserting phase progression, finite 23-D actions, reset isolation, and deliberate near-limit assistance activation.
- [ ] Run local focused tests and `python -m py_compile` for all new Python files.
- [ ] Run Isaac Lab with `/home/xk/miniconda3/envs/go2/bin/python`, `--task Isaac-M1-Panda-Coordinated-v0`, `--num_envs 8`, `--max_steps 2000`, `--headless`, and `--device cuda:0`.
- [ ] Require combined articulation creation, arrival/settled phase, finite EE tracking, bounded base assistance, zero unexpected resets, zero joint-limit violations, and no non-finite diagnostics.
- [ ] Record exact command, environment, checkpoint/initialization, metrics, warnings, and exit code in the log; update indexes and explicitly mark Student S1 unchanged.
- [ ] Commit: `test: verify coordinated M1 Panda mission runtime`.

## Verification Summary

Before claiming completion, run the focused pure-PyTorch suite, combined static suite, `py_compile`, and the real Isaac smoke. A successful result requires evidence for both mission stages and at least one intentional null-space base-assistance event; a scene startup alone is insufficient.
