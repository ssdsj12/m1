# M1 + Panda Prioritized WBC Teacher Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user requires single-agent inline execution; do not dispatch subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the C0 stationary deterministic Teacher: Panda follows a smooth six-dimensional end-effector target while one unified M1 + Panda articulation maintains contact and dynamic balance through a 200 Hz whole-body QP.

**Architecture:** A pure-PyTorch motion-distribution package produces bounded M1 planar and Panda joint references at 50 Hz. A project-owned float64 reference QP turns 200 Hz articulation dynamics and four-wheel contact data into 23 joint feedforward efforts, which share one impedance and safety envelope. Isaac Lab integration is isolated behind a new asset config, environment ID, and play entry point so existing A0/A1 checkpoints and 60/16 contracts remain untouched.

**Tech Stack:** Python 3.11, PyTorch, pytest, Isaac Sim 5.1, Isaac Lab manager-based environments, PhysX tensor views, Gymnasium.

## Scope and frozen contracts

- C0 only: stationary base command, four contacts expected, small smooth Panda end-effector motion, no added external wrench curriculum, no object grasping, and no Student.
- Preserve all existing A0/A1 files, Gym IDs, checkpoints, action/observation shapes, and play commands.
- Register the new environment as `Isaac-M1-Panda-Wbc-Teacher-C0-v0`.
- Copy `M1_PANDA_CFG` into `M1_PANDA_WBC_CFG`; zero stiffness and damping only in the copied 23 WBC-controlled joint actuator definitions. Keep the two fingers position-controlled and open.
- Use `sim.dt=0.005`, `decimation=1`, a 200 Hz WBC update, and a motion-distribution update every four physics steps (50 Hz).
- Coordination vector: M1 planar `x/y/yaw` plus Panda joints 1–7, exactly 10 dimensions.
- Full generalized acceleration vector: 6 floating-base plus 25 articulation joints, exactly 31 dimensions.
- WBC output order: 12 M1 leg joints, 4 wheel joints, and 7 Panda arm joints, exactly 23 dimensions.
- Balance and contact feasibility outrank end-effector tracking. Safety degradation must scale, hold, retract, then terminate; it must never snap the arm to home.
- The reference QP is project-owned and float64. Do not add a solver dependency during C0.
- Every implementation task follows RED → GREEN → focused regression → commit. Do not proceed past a failing checkpoint.

## Planned file layout

New pure control package:

```text
Go2Pvcnn/go2_pvcnn/control/__init__.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/contracts.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/kinematics.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/qp_backend.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/standing_wbc.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/impedance.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/trajectory.py
Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py
```

Isaac integration and tests:

```text
Go2Pvcnn/go2_pvcnn/assets/m1_panda.py
Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py
Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py
Go2Pvcnn/scripts/m1_panda_wbc_play.py
Go2Pvcnn/tests/test_m1_panda_wbc_contracts.py
Go2Pvcnn/tests/test_m1_panda_wbc_kinematics.py
Go2Pvcnn/tests/test_m1_panda_motion_distribution.py
Go2Pvcnn/tests/test_m1_panda_qp_backend.py
Go2Pvcnn/tests/test_m1_panda_standing_wbc.py
Go2Pvcnn/tests/test_m1_panda_wbc_safety.py
Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py
Go2Pvcnn/tests/test_m1_panda_wbc_env_static.py
Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py
```

---

### Task 1: Freeze dimensions, names, and tensor contracts

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/contracts.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wbc_contracts.py`

- [x] **Step 1: Write failing contract tests**

Test these public values and behaviors:

```python
COORD_DOF = 10
GENERALIZED_DOF = 31
CONTROLLED_DOF = 23
PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))

joint_map = WbcJointMap.resolve(actual_joint_names)
assert joint_map.controlled.numel() == 23
assert joint_map.panda_arm.numel() == 7
assert joint_map.fingers.numel() == 2
```

Cover exact ordering, duplicate names, missing names, wrong last dimension, wrong dtype/device, and non-finite tensors. The public helper must be:

```python
require_tensor(name, value, *, trailing_shape, dtype=None, device=None)
```

- [x] **Step 2: Run RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_wbc_contracts.py
```

Expected: collection fails because `go2_pvcnn.control.m1_panda_coordination` does not exist.

- [x] **Step 3: Implement the smallest contract module**

Use immutable dataclasses. Resolve indices from exact joint names at runtime; never assume USD joint order. Export the constants, `WbcJointMap`, and `require_tensor` from the package `__init__.py`. Error messages must name the rejected field and expected shape.

- [x] **Step 4: Run GREEN and regression**

Run:

```bash
pytest -q tests/test_m1_panda_wbc_contracts.py
pytest -q tests/test_m1_panda_asset_static.py tests/test_m1_panda_smoke_cfg_static.py
python -m py_compile go2_pvcnn/control/m1_panda_coordination/*.py
```

Expected: all selected tests pass and compilation exits 0.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control Go2Pvcnn/tests/test_m1_panda_wbc_contracts.py
git commit -m "feat: define M1 Panda WBC contracts"
```

---

### Task 2: Implement coordination kinematics and singularity diagnostics

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/kinematics.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wbc_kinematics.py`

- [ ] **Step 1: Write failing analytic and finite-difference tests**

Define these interfaces:

```python
def planar_base_spatial_jacobian(ee_position_base: torch.Tensor) -> torch.Tensor: ...
def coordinated_jacobian(ee_position_base, panda_spatial_jacobian) -> torch.Tensor: ...
def damped_pseudoinverse(jacobian, damping: float) -> torch.Tensor: ...
def singularity_metrics(panda_spatial_jacobian) -> tuple[torch.Tensor, torch.Tensor]: ...
```

For end-effector point `(px, py, pz)`, verify the base columns produce x/y translation and yaw twist:

```text
linear x: [1, 0, 0]
linear y: [0, 1, 0]
linear yaw: [-py, px, 0]
angular yaw: [0, 0, 1]
```

The coordinated Jacobian must be `[..., 6, 10]`. Compare the analytic base columns to central finite differences. Verify `J @ J_damped_pinv @ J` reconstructs a well-conditioned `J`, and verify minimum singular value plus product-of-singular-values manipulability on diagonal matrices.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_m1_panda_wbc_kinematics.py
```

Expected: import failure for the missing kinematics module.

- [ ] **Step 3: Implement batched float-safe operations**

Use `torch.linalg.svd`, preserve leading batch dimensions, accept float32 or float64 inputs, and reject non-finite inputs through `require_tensor`. Clamp only the product calculation against negative round-off; do not conceal a negative or non-finite singular value.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_wbc_contracts.py tests/test_m1_panda_wbc_kinematics.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/kinematics.py Go2Pvcnn/tests/test_m1_panda_wbc_kinematics.py
git commit -m "feat: add M1 Panda coordination kinematics"
```

Expected: both files pass.

---

### Task 3: Implement bounded prioritized motion distribution

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_motion_distribution.py`

- [ ] **Step 1: Write failing bound-intersection tests**

Expose:

```python
def compute_velocity_bounds(q, qd, q_min, q_max, v_max, a_max, dt): ...
```

For every coordination DOF, require:

```python
lower = torch.maximum(torch.maximum((q_min - q) / dt, -v_max), qd - a_max * dt)
upper = torch.minimum(torch.minimum((q_max - q) / dt,  v_max), qd + a_max * dt)
```

Reject `lower > upper`, non-positive `dt`, and non-finite values.

- [ ] **Step 2: Write failing priority tests**

Freeze this config and result API:

```python
@dataclass(frozen=True)
class MotionDistributionCfg:
    pose_gain: float = 10.0
    damping: float = 1.0e-4
    singularity_threshold: float = 0.1
    null_gain: float = 5.0
    null_damping: float = 0.5
    max_saturation_passes: int = 10

@dataclass(frozen=True)
class MotionDistributionResult:
    qd_coord: torch.Tensor
    base_active: torch.Tensor
    sigma_min: torch.Tensor
    phi: torch.Tensor
    psi: torch.Tensor
    saturated: torch.Tensor
```

Tests must prove:

- arm-only motion is selected when feasible;
- a saturated Panda DOF is frozen and the remaining task is redistributed;
- base `x/y/yaw` activates on rank loss, bound exhaustion, or `sigma_min < 0.1`;
- the manipulability gradient is projected into the P1 null space;
- infeasibility reduces `psi` before reducing end-effector `phi`;
- the output is finite and within all velocity bounds.

- [ ] **Step 3: Run RED**

```bash
pytest -q tests/test_m1_panda_motion_distribution.py
```

Expected: missing module failures.

- [ ] **Step 4: Implement deterministic active-set redistribution**

Start with a selection mask that disables the three base columns. Solve P1 with a damped pseudoinverse, freeze the most violated coordinate per pass, and re-solve for at most ten passes. Activate the base if the arm-only Jacobian loses task rank, violates a bound after saturation passes, or crosses the singularity threshold. Apply P3 through `N = I - J_pinv @ J`. Return explicit `phi`, `psi`, and saturation diagnostics.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_wbc_contracts.py tests/test_m1_panda_wbc_kinematics.py tests/test_m1_panda_motion_distribution.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/constraints.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py Go2Pvcnn/tests/test_m1_panda_motion_distribution.py
git commit -m "feat: add prioritized M1 Panda motion distribution"
```

Expected: all selected tests pass.

---

### Task 4: Build the project-owned float64 reference QP backend

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/qp_backend.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_qp_backend.py`

- [ ] **Step 1: Write failing solver-contract tests**

Freeze the shared problem/result structures:

```python
@dataclass(frozen=True)
class DenseQpProblem:
    hessian: torch.Tensor
    gradient: torch.Tensor
    equality_matrix: torch.Tensor
    equality_rhs: torch.Tensor
    inequality_matrix: torch.Tensor
    inequality_upper: torch.Tensor
    lower_bound: torch.Tensor
    upper_bound: torch.Tensor

@dataclass(frozen=True)
class DenseQpResult:
    solution: torch.Tensor
    success: bool
    iterations: int
    max_equality_residual: float
    max_inequality_violation: float
    active_set: tuple[int, ...]

def solve_reference_qp(problem: DenseQpProblem, *, tolerance=1.0e-9, max_iterations=128) -> DenseQpResult: ...
```

Test unconstrained quadratic minimization, equality-only KKT, box clipping, one active inequality, redundant inequalities, repeat determinism, and an infeasible case. Successful solutions require residual and violation at most `1e-8`; infeasible input must return `success=False` with a finite diagnostic solution.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_m1_panda_qp_backend.py
```

Expected: missing backend module.

- [ ] **Step 3: Implement a deterministic active-set solver**

Convert inputs to CPU float64 inside the reference backend, symmetrize the Hessian, solve equality-constrained KKT systems with `torch.linalg.lstsq`, add the most violated constraint, and remove an active inequality only when its multiplier has the wrong sign. Include bounds as inequalities in stable index order. Never throw for ordinary infeasibility; reserve exceptions for malformed/non-finite problems.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_wbc_contracts.py tests/test_m1_panda_qp_backend.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/qp_backend.py Go2Pvcnn/tests/test_m1_panda_qp_backend.py
git commit -m "feat: add deterministic reference QP backend"
```

---

### Task 5: Formulate the C0 standing whole-body controller

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/standing_wbc.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_standing_wbc.py`

- [ ] **Step 1: Write failing problem-construction tests**

Define `StandingWbcInput`, `StandingWbcCfg`, and `StandingWbcResult`. The decision vector is exactly:

```text
z = [qdd(31), contact_force(4 wheels × 3 axes)]
```

Use these default objective weights:

```python
balance_weight = 1.0e6
base_pose_weight = 1.0e5
leg_posture_weight = 1.0e4
arm_tracking_weight = 1.0e3
wheel_stop_weight = 1.0e3
force_equalization_weight = 10.0
regularization = 1.0e-6
```

Tests must inspect the assembled matrices and verify:

- floating-base rows of `M qdd + h = S.T tau + Jc.T f` are hard equalities;
- stationary contact acceleration `Jc qdd + Jdot_qd = 0` is a hard equality;
- each contact has positive normal force and a four-sided friction pyramid;
- generalized acceleration and recovered 23-joint torque limits are enforced;
- balance residual receives a larger weight than arm tracking;
- external mount wrench maps through the supplied six-dimensional Jacobian term;
- malformed shapes and batch size greater than one are rejected in C0.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_m1_panda_standing_wbc.py
```

Expected: missing standing WBC module.

- [ ] **Step 3: Implement formulation and torque recovery**

The controller calls `solve_reference_qp` and recovers controlled torque using the actuated dynamics rows:

```python
tau = (mass_matrix @ qdd + bias_force - contact_jacobian.transpose(-1, -2) @ force - external_generalized_force)[actuated_rows]
```

Track base height, roll, pitch, angular velocity, leg posture, zero wheel acceleration, and seven Panda targets. Return QP diagnostics plus task residuals. If the QP fails, return no new effort command and let the safety layer choose the fallback.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_qp_backend.py tests/test_m1_panda_standing_wbc.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/standing_wbc.py Go2Pvcnn/tests/test_m1_panda_standing_wbc.py
git commit -m "feat: add standing M1 Panda whole-body QP"
```

---

### Task 6: Add impedance output and balance-first safety supervision

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/impedance.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wbc_safety.py`

- [ ] **Step 1: Write failing impedance tests**

Expose a pure function:

```python
tau = apply_impedance(q, qd, q_des, qd_des, tau_ff, kp, kd, effort_limit)
```

Verify `tau_ff + kp * (q_des - q) + kd * (qd_des - qd)`, symmetric effort clamping, 23-channel shape, and all-or-nothing rejection of non-finite inputs.

- [ ] **Step 2: Write failing state-machine tests**

Freeze states `TRACK`, `SCALE`, `HOLD`, `RETRACT`, `TERMINATE`. Default thresholds:

- warning roll/pitch: 7 degrees;
- critical roll/pitch: 10 degrees;
- required wheel contacts: 4;
- maximum lateral slip: `0.05 m/s`;
- two consecutive unsafe samples to advance;
- twenty consecutive safe samples to recover one level.

Test QP failure, contact loss, slip, non-finite inputs, monotonic escalation, slow recovery, zero wheel command in hold/retract, smooth retract interpolation, and terminal latching until reset.

- [ ] **Step 3: Run RED**

```bash
pytest -q tests/test_m1_panda_wbc_safety.py
```

Expected: missing impedance and safety modules.

- [ ] **Step 4: Implement and export**

`SCALE` reduces end-effector twist; `HOLD` freezes the current safe arm target; `RETRACT` advances a rate-limited interpolation toward the bent Panda home pose; `TERMINATE` emits zero wheel target and an episode termination request. Reset must clear counters, latched state, interpolation, and last effort.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_wbc_safety.py tests/test_m1_panda_standing_wbc.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/impedance.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py Go2Pvcnn/tests/test_m1_panda_wbc_safety.py
git commit -m "feat: add WBC impedance and safety supervision"
```

---

### Task 7: Compose trajectory, motion distribution, WBC, and safety into Teacher

**Files:**

- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/trajectory.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py`

- [ ] **Step 1: Write failing trajectory tests**

Implement a seeded sum-of-sinusoids trajectory with position amplitude at most `0.08 m`, orientation-vector amplitude at most `0.15 rad`, and frequency range `0.05–0.25 Hz`. Return pose, twist, and acceleration analytically. Tests cover seed repeatability, continuity, bounds, and finite derivatives.

- [ ] **Step 2: Write failing scheduling and fallback tests**

Freeze:

```python
class M1PandaWbcTeacher:
    def reset(self, state, *, seed: int) -> None: ...
    def step(self, state: TeacherState) -> TeacherCommand: ...
```

The command contains `effort[23]`, target pose/twist, motion-distribution diagnostics, QP diagnostics, and safety state. Tests must show high-level updates at physics steps `0, 4, 8`, WBC updates every step, interpolation between high-level references, deterministic reset, and safety override without bypassing WBC.

- [ ] **Step 3: Run RED**

```bash
pytest -q tests/test_m1_panda_wbc_teacher.py
```

Expected: missing trajectory/Teacher modules.

- [ ] **Step 4: Implement the deterministic orchestration**

Use dependency injection for kinematics/dynamics snapshots so pure tests need no Isaac import. A failed motion-distribution or WBC cycle reuses only the last verified finite target, advances safety, and records the failure reason. Never reuse an unverified effort vector.

- [ ] **Step 5: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_wbc_contracts.py tests/test_m1_panda_wbc_kinematics.py tests/test_m1_panda_motion_distribution.py tests/test_m1_panda_qp_backend.py tests/test_m1_panda_standing_wbc.py tests/test_m1_panda_wbc_safety.py tests/test_m1_panda_wbc_teacher.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/trajectory.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py
git commit -m "feat: compose deterministic M1 Panda WBC teacher"
```

---

### Task 8: Add an isolated Isaac Lab effort-control environment

**Files:**

- Modify: `Go2Pvcnn/go2_pvcnn/assets/m1_panda.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wbc_env_static.py`

- [ ] **Step 1: Write failing static tests**

Parse/import the config and assert:

- `M1_PANDA_CFG` is unchanged;
- `M1_PANDA_WBC_CFG` is a copy with zero stiffness/damping for the 23 controlled joints;
- fingers retain their existing position control and `0.04` open defaults;
- one `JointEffortActionCfg` addresses the exact 23-joint ordered regex/list with scale 1 and preserved order;
- `sim.dt == 0.005`, `decimation == 1`, render interval is 4, and episode length is 20 seconds;
- Gym ID `Isaac-M1-Panda-Wbc-Teacher-C0-v0` resolves lazily;
- existing A0/A1 Gym registrations and 60/16 configs are byte-for-byte unaffected by imports.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_m1_panda_wbc_env_static.py
```

Expected: the new asset/config/ID assertions fail.

- [ ] **Step 3: Implement isolated integration**

Copy actuator config objects before changing gains. Reuse the local combined USD. Add only WBC-required scene sensors: contact data for the four wheels, base contact failure, and articulation state. The environment action is effort only; do not expose an RL observation/action contract or instantiate an RSL-RL runner.

- [ ] **Step 4: Run GREEN plus A0/A1 regression**

```bash
pytest -q tests/test_m1_panda_wbc_env_static.py tests/test_m1_panda_asset_static.py tests/test_m1_panda_teacher_env_cfg_static.py tests/test_m1_panda_teacher_play_static.py
python -m py_compile go2_pvcnn/assets/m1_panda.py go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py go2_pvcnn/tasks/register_m1_envs.py
```

Expected: all selected tests pass and compilation exits 0.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/assets/m1_panda.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py Go2Pvcnn/tests/test_m1_panda_wbc_env_static.py
git commit -m "feat: register isolated M1 Panda WBC environment"
```

---

### Task 9: Wire PhysX dynamics into a deterministic play entry point

**Files:**

- Create: `Go2Pvcnn/scripts/m1_panda_wbc_play.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py`

- [ ] **Step 1: Write failing entry-point tests**

Require CLI arguments `--steps` (default `0`, unlimited), `--seed`, `--summary-json`, `--headless`, `--device`, and `--disable-target-motion`. Enforce one environment in C0. Static tests must reject imports or calls for PPO runners, checkpoint loading, `learn`, optimizer steps, or manifest writes.

The runtime adapter must read these PhysX tensors every physics step:

```python
root_view.get_generalized_mass_matrices()
root_view.get_coriolis_and_centrifugal_forces()
root_view.get_generalized_gravity_forces()
root_view.get_jacobians()
```

It must combine Coriolis/centrifugal and gravity into `h`, convert tensors into the Teacher’s explicit ordering, call the Teacher, apply the 23 efforts, and then call `env.step` exactly once.

- [ ] **Step 2: Run RED**

```bash
pytest -q tests/test_m1_panda_wbc_play_static.py
```

Expected: script/adapter assertions fail because the entry point is absent.

- [ ] **Step 3: Implement runtime diagnostics and atomic summary**

Print concise periodic diagnostics for end-effector error, minimum singular value, QP feasibility, roll/pitch, contact count, maximum lateral slip, safety state, and reset cause. If `--summary-json` is supplied, write through a sibling temporary file followed by `os.replace`. The summary schema must include seed, steps, finite flag, QP feasible count/rate, maximum end-effector position error, minimum singular value, maximum roll/pitch, maximum lateral slip, joint-limit violations, base contacts, self-collisions, safety-state counts, reset count, and exit reason.

- [ ] **Step 4: Run GREEN and commit**

```bash
pytest -q tests/test_m1_panda_wbc_play_static.py tests/test_m1_panda_wbc_env_static.py tests/test_m1_panda_wbc_teacher.py
python -m py_compile scripts/m1_panda_wbc_play.py
git add Go2Pvcnn/scripts/m1_panda_wbc_play.py Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py
git commit -m "feat: add M1 Panda WBC teacher play"
```

---

### Task 10: Validate C0 on GPU0 and publish the runbook/evidence

**Files:**

- Create: `docs/superpowers/runbooks/2026-08-17-m1-panda-prioritized-wbc-teacher-c0.md`
- Create: `notes/log/2026-08-17-m1-panda-prioritized-wbc-teacher-c0.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`

- [ ] **Step 1: Run the complete pure/static regression**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q \
  tests/test_m1_panda_wbc_contracts.py \
  tests/test_m1_panda_wbc_kinematics.py \
  tests/test_m1_panda_motion_distribution.py \
  tests/test_m1_panda_qp_backend.py \
  tests/test_m1_panda_standing_wbc.py \
  tests/test_m1_panda_wbc_safety.py \
  tests/test_m1_panda_wbc_teacher.py \
  tests/test_m1_panda_wbc_env_static.py \
  tests/test_m1_panda_wbc_play_static.py \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_teacher_env_cfg_static.py \
  tests/test_m1_panda_teacher_play_static.py
python -m py_compile \
  go2_pvcnn/control/m1_panda_coordination/*.py \
  go2_pvcnn/tasks/m1_panda_wbc_teacher_env_cfg.py \
  scripts/m1_panda_wbc_play.py
git diff --check
```

Expected: all tests pass, compilation exits 0, and `git diff --check` prints nothing.

- [ ] **Step 2: Run an 8-step GPU0 no-motion smoke**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 8 --seed 42 --disable-target-motion \
  --summary-json /tmp/m1_panda_wbc_c0_static.json
```

Expected: exit 0, eight finite steps, QP feasible rate 1.0, four wheel contacts, zero reset, zero base contact, and zero joint-limit violation.

- [ ] **Step 3: Run the 2,000-step moving-target C0 acceptance**

```bash
CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 2000 --seed 42 \
  --summary-json /tmp/m1_panda_wbc_c0_motion.json
```

Read the JSON and require all gates simultaneously:

- maximum end-effector position error `<= 0.03 m`;
- minimum Panda singular value `>= 0.1`, or a recorded safety/base-activation event before threshold crossing with no unsafe continuation;
- QP feasible rate `>= 0.999`;
- maximum wheel lateral slip `<= 0.05 m/s`;
- maximum absolute roll and pitch `<= 10 deg`;
- zero joint-limit violation, self-collision, base contact, non-finite state, discontinuous arm snap, and unexpected reset.

If a hard gate fails, stop. Record the exact metric and diagnose under `systematic-debugging`; do not weaken the gate in the same commit.

- [ ] **Step 4: Write the operator runbook and evidence**

The runbook must include the exact GUI and headless commands, C0 limitations, summary fields, safety states, acceptance thresholds, and the statement that this is a deterministic Teacher play—not PPO training and not a Student. Record actual commands, exit codes, test counts, GPU identity, JSON metrics, and commit refs in the log. Update T400.8 to mark only the C0 foundation complete and name C1/C2 rolling constraints as the next design/plan.

- [ ] **Step 5: Re-run documentation and repository checks**

```bash
cd /home/xk/coding/M1
git diff --check
git status --short
```

Expected: no whitespace errors; only intended C0 implementation/docs plus pre-existing unrelated user/graph artifacts appear.

- [ ] **Step 6: Commit the acceptance evidence**

```bash
git add \
  docs/superpowers/runbooks/2026-08-17-m1-panda-prioritized-wbc-teacher-c0.md \
  notes/log/2026-08-17-m1-panda-prioritized-wbc-teacher-c0.md \
  notes/log/index.md notes/todo/T400-m1-panda-force-aware-teacher-student.md notes/todo.md
git commit -m "docs: verify M1 Panda WBC teacher C0"
```

## C0 completion gate

C0 is complete only when Tasks 1–10 are committed, all pure/static tests pass, both GPU0 commands exit 0, and every 2,000-step hard gate passes. Completion does not authorize C1/C2 rolling motion, C3 external-wrench randomization, batched GPU QP, Teacher data collection, Student training, grasping, or real-hardware maximum-load tests. Those remain separate approved-scope plans after C0 evidence is reviewed.
