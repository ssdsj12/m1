# M1 + Panda Rolling WBC Teacher C1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, balance-first C1a Teacher that drives the combined M1 + Panda articulation through a flat-ground forward/stop/reverse speed schedule while the Panda follows a small body-frame six-dimensional trajectory.

**Architecture:** Keep the accepted C0 task and entrypoint behavior intact. Add explicit rolling-contact helpers, prescribed-base motion distribution, a rolling WBC configuration, and a separate rolling Teacher/play path; reuse the C0 float64 dynamics, impedance, trajectory, QP, and safety contracts where their defaults remain unchanged.

**Tech Stack:** Python 3.11, PyTorch float64 reference control, Isaac Sim/IsaacLab, PhysX articulation Jacobians and mass matrices, Gymnasium, pytest.

## Global Constraints

- Work only in `/home/xk/coding/M1` and execute from `/home/xk/coding/M1/Go2Pvcnn` unless a command says otherwise.
- Use single-agent inline execution with `superpowers:executing-plans`; do not dispatch subagents.
- Use TDD for every behavior change: RED test, minimal GREEN implementation, focused regression, commit.
- Preserve `Isaac-M1-Panda-Wbc-Teacher-C0-v0`, `scripts/m1_panda_wbc_play.py`, and all accepted C0 defaults.
- Use a separate Gym ID: `Isaac-M1-Panda-Wbc-Teacher-C1a-v0`.
- C1a supports exactly one environment and runs WBC at `0.005 s` with motion distribution every four physics steps.
- Use wheel radius `0.095 m` and canonical wheel order `FAR, FBL, RAR, RBL`.
- Fix `v_y_cmd=0` and `yaw_rate_cmd=0`; do not add turning, lateral travel, terrain, external wrench curriculum, grasping, Student, or PPO behavior.
- Use the five-phase 4000-step schedule `0.00, +0.05, +0.10, 0.00, -0.05 m/s`, 800 mission steps per phase.
- Keep contact, dynamics, balance, posture, and actuator safety above wheel/base speed tracking and Panda tracking.
- Never stage or commit `graphify-out/` changes.

---

## File Structure

### New runtime modules

- `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_contact.py`: wheel constants, point-Jacobian construction, `v/r` mapping, and rolling/slip metrics.
- `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py`: rolling-specific WBC weights and wrapper around the accepted dynamics/QP builder.
- `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py`: five-phase rate-limited command, planar body-frame trajectory transform, rolling orchestration, and safety-aware braking.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py`: isolated C1a effort-control environment.
- `Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py`: GUI/headless C1a runtime, diagnostics, JSON summary, and hard-gate exit status.

### Modified runtime modules

- `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py`: optional prescribed planar base velocity with `None` preserving C0 behavior.
- `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py`: rolling residual and stopped-before-retract inputs with defaults preserving C0 behavior.
- `Go2Pvcnn/scripts/m1_panda_wbc_play.py`: make the existing PhysX adapter accept an injected wheel radius and expose measured rolling state; default remains C0's accepted value.
- `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`: register the independent C1a task.

### New tests

- `Go2Pvcnn/tests/test_m1_panda_rolling_contact.py`
- `Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py`
- `Go2Pvcnn/tests/test_m1_panda_rolling_wbc.py`
- `Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py`

### Modified tests

- `Go2Pvcnn/tests/test_m1_panda_motion_distribution.py`
- `Go2Pvcnn/tests/test_m1_panda_wbc_safety.py`
- `Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py`

---

### Task 1: Add wheel/contact kinematics as a pure module

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_contact.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_rolling_contact.py`

**Interfaces:**
- Consumes: generalized velocity tensors ending in `(31,)` and four body Jacobians shaped `(4, 6, 31)`.
- Produces: `RollingContactCfg`, `wheel_speed_from_base_velocity(vx, cfg)`, `contact_point_linear_jacobian(body_jacobian, point_offset_w)`, `build_wheel_contact_jacobian(body_jacobians, cfg)`, and `rolling_contact_metrics(contact_jacobian, generalized_velocity, yaw)`.

- [ ] **Step 1: Write failing wheel mapping and point-Jacobian tests**

```python
import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.rolling_contact import (
    RollingContactCfg,
    build_wheel_contact_jacobian,
    rolling_contact_metrics,
    wheel_speed_from_base_velocity,
)


def test_forward_base_velocity_maps_to_four_canonical_wheel_speeds():
    cfg = RollingContactCfg()
    vx = torch.tensor(0.095, dtype=torch.float64)
    target = wheel_speed_from_base_velocity(vx, cfg)
    assert cfg.wheel_radius_m == pytest.approx(0.095)
    assert cfg.wheel_signs == (1.0, 1.0, 1.0, 1.0)
    assert torch.equal(target, torch.ones(4, dtype=torch.float64))


def test_bottom_point_jacobian_encodes_pure_rolling_cancellation():
    cfg = RollingContactCfg()
    body = torch.zeros(4, 6, 31, dtype=torch.float64)
    body[:, 0, 0] = 1.0
    for wheel, column in enumerate((18, 19, 20, 21)):
        body[wheel, 4, column] = 1.0
    contact = build_wheel_contact_jacobian(body, cfg)
    qd = torch.zeros(31, dtype=torch.float64)
    qd[0] = 0.095
    qd[18:22] = 1.0
    metrics = rolling_contact_metrics(contact, qd, yaw=0.0)
    assert contact.shape == (12, 31)
    assert metrics.max_longitudinal_residual_mps == pytest.approx(0.0, abs=1e-12)
    assert metrics.max_lateral_slip_mps == pytest.approx(0.0, abs=1e-12)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_rolling_contact.py
```

Expected: collection fails with `ModuleNotFoundError` for `rolling_contact`.

- [ ] **Step 3: Implement the pure rolling-contact API**

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import GENERALIZED_DOF, require_tensor


@dataclass(frozen=True)
class RollingContactCfg:
    wheel_radius_m: float = 0.095
    wheel_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.wheel_radius_m) or self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be finite and positive")
        if len(self.wheel_signs) != 4 or any(sign not in (-1.0, 1.0) for sign in self.wheel_signs):
            raise ValueError("wheel_signs must contain four values in {-1.0, 1.0}")


@dataclass(frozen=True)
class RollingContactMetrics:
    contact_velocity_heading: torch.Tensor
    max_longitudinal_residual_mps: float
    max_lateral_slip_mps: float


def wheel_speed_from_base_velocity(vx: torch.Tensor, cfg: RollingContactCfg) -> torch.Tensor:
    require_tensor("vx", vx, trailing_shape=())
    if vx.ndim != 0:
        raise ValueError("vx must be one scalar tensor")
    signs = vx.new_tensor(cfg.wheel_signs)
    return signs * vx / cfg.wheel_radius_m


def contact_point_linear_jacobian(body_jacobian: torch.Tensor, point_offset_w: torch.Tensor) -> torch.Tensor:
    require_tensor("body_jacobian", body_jacobian, trailing_shape=(6, GENERALIZED_DOF))
    require_tensor("point_offset_w", point_offset_w, trailing_shape=(3,))
    x, y, z = point_offset_w
    skew = point_offset_w.new_zeros((3, 3))
    skew[0, 1], skew[0, 2] = -z, y
    skew[1, 0], skew[1, 2] = z, -x
    skew[2, 0], skew[2, 1] = -y, x
    return body_jacobian[:3] - skew @ body_jacobian[3:]


def build_wheel_contact_jacobian(body_jacobians: torch.Tensor, cfg: RollingContactCfg) -> torch.Tensor:
    require_tensor("body_jacobians", body_jacobians, trailing_shape=(4, 6, GENERALIZED_DOF))
    offset = body_jacobians.new_tensor((0.0, 0.0, -cfg.wheel_radius_m))
    return torch.stack(
        [contact_point_linear_jacobian(body_jacobians[index], offset) for index in range(4)]
    ).reshape(12, GENERALIZED_DOF)


def rolling_contact_metrics(contact_jacobian: torch.Tensor, generalized_velocity: torch.Tensor, yaw: float) -> RollingContactMetrics:
    require_tensor("contact_jacobian", contact_jacobian, trailing_shape=(12, GENERALIZED_DOF))
    require_tensor("generalized_velocity", generalized_velocity, trailing_shape=(GENERALIZED_DOF,))
    velocity_w = (contact_jacobian @ generalized_velocity).reshape(4, 3)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = velocity_w.new_tensor(((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    velocity_heading = velocity_w @ rotation.transpose(0, 1)
    return RollingContactMetrics(
        contact_velocity_heading=velocity_heading,
        max_longitudinal_residual_mps=float(velocity_heading[:, 0].abs().max().item()),
        max_lateral_slip_mps=float(velocity_heading[:, 1].abs().max().item()),
    )
```

- [ ] **Step 4: Add finite/shape/sign rejection tests and run GREEN**

Add tests that reject zero radius, non-finite `vx`, wrong Jacobian shape, and non-four-element signs. Run:

```bash
pytest -q tests/test_m1_panda_rolling_contact.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the contact module**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_contact.py Go2Pvcnn/tests/test_m1_panda_rolling_contact.py
git commit -m "feat: add M1 rolling contact kinematics"
```

### Task 2: Add the deterministic speed schedule and body-frame trajectory

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py`

**Interfaces:**
- Consumes: reset-relative mission step, optional safety scale, current planar root pose `[x,y,yaw]`, planar root velocity `[vx,vy,yaw_rate]`, and the existing `BandLimitedPoseTrajectory`.
- Produces: `LongitudinalScheduleCfg`, `LongitudinalCommand`, `LongitudinalCommandSchedule`, and `PlanarBodyFrameTrajectory`.

- [ ] **Step 1: Write failing schedule tests**

```python
import pytest

from go2_pvcnn.control.m1_panda_coordination.rolling_teacher import (
    LongitudinalCommandSchedule,
    LongitudinalScheduleCfg,
)


def test_schedule_has_five_800_step_phases_and_rate_limits_boundaries():
    schedule = LongitudinalCommandSchedule(LongitudinalScheduleCfg())
    schedule.reset()
    for step in range(800):
        command = schedule.sample(step)
    assert command.phase == 0
    assert command.raw_target_mps == pytest.approx(0.0)
    first_forward = schedule.sample(800)
    assert first_forward.phase == 1
    assert first_forward.raw_target_mps == pytest.approx(0.05)
    assert first_forward.shaped_target_mps == pytest.approx(0.0005)
    for step in range(801, 4000):
        command = schedule.sample(step)
    assert command.phase == 4


def test_hold_scale_requests_rate_limited_stop_instead_of_locking_wheels():
    schedule = LongitudinalCommandSchedule(LongitudinalScheduleCfg())
    schedule.reset()
    for step in range(1000):
        command = schedule.sample(step)
    stopped = schedule.sample(1000, safety_scale=0.0)
    assert 0.0 < stopped.shaped_target_mps < command.shaped_target_mps
```

- [ ] **Step 2: Run schedule tests and confirm RED**

Run: `pytest -q tests/test_m1_panda_rolling_teacher.py`

Expected: collection fails because `rolling_teacher.py` does not exist.

- [ ] **Step 3: Implement the rate-limited schedule**

```python
@dataclass(frozen=True)
class LongitudinalScheduleCfg:
    physics_dt: float = 0.005
    phase_steps: int = 800
    phase_targets_mps: tuple[float, ...] = (0.0, 0.05, 0.10, 0.0, -0.05)
    max_acceleration_mps2: float = 0.1


@dataclass(frozen=True)
class LongitudinalCommand:
    phase: int
    raw_target_mps: float
    shaped_target_mps: float


class LongitudinalCommandSchedule:
    def __init__(self, cfg: LongitudinalScheduleCfg | None = None):
        self.cfg = cfg or LongitudinalScheduleCfg()
        self.reset()

    def reset(self) -> None:
        self._shaped_target_mps = 0.0
        self._last_step = -1

    def sample(self, mission_step: int, safety_scale: float = 1.0) -> LongitudinalCommand:
        if mission_step != self._last_step + 1:
            raise ValueError("mission_step must advance exactly once")
        phase = min(mission_step // self.cfg.phase_steps, len(self.cfg.phase_targets_mps) - 1)
        raw = self.cfg.phase_targets_mps[phase]
        requested = float(safety_scale) * raw
        maximum_delta = self.cfg.max_acceleration_mps2 * self.cfg.physics_dt
        delta = max(-maximum_delta, min(maximum_delta, requested - self._shaped_target_mps))
        self._shaped_target_mps += delta
        self._last_step = mission_step
        return LongitudinalCommand(phase, raw, self._shaped_target_mps)
```

- [ ] **Step 4: Write the failing planar body-frame transform test**

```python
import torch

from go2_pvcnn.control.m1_panda_coordination.trajectory import BandLimitedTrajectoryCfg
from go2_pvcnn.control.m1_panda_coordination.rolling_teacher import PlanarBodyFrameTrajectory


def test_body_frame_center_advects_with_root_without_arm_extension():
    trajectory = PlanarBodyFrameTrajectory(
        BandLimitedTrajectoryCfg(position_amplitude=0.0, orientation_amplitude=0.0)
    )
    ee = torch.tensor([1.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=torch.float64)
    root = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    trajectory.reset(ee, root, seed=42)
    moved = trajectory.sample(
        1.0,
        torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
    )
    assert moved.pose[0].item() == pytest.approx(1.2)
    assert moved.twist[0].item() == pytest.approx(0.1)
```

- [ ] **Step 5: Implement the body-frame wrapper and run GREEN**

Implement `PlanarBodyFrameTrajectory.reset(center_pose, root_xy_yaw, seed)` by storing the end-effector center in the root heading frame. Implement `sample(time_s, root_xy_yaw, root_vxy_yawrate)` with:

```python
offset_world_xy = rotation_2d(yaw) @ local.pose[:2]
pose[:2] = root_xy_yaw[:2] + offset_world_xy
pose[2] = local.pose[2]
pose[3:] = local.pose[3:]
pose[5] += root_xy_yaw[2] - self._reset_root_yaw
twist[:2] = root_vxy_yawrate[:2] + rotation_2d(yaw) @ local.twist[:2]
twist[:2] += root_vxy_yawrate[2] * torch.stack((-offset_world_xy[1], offset_world_xy[0]))
twist[2] = local.twist[2]
twist[3:] = local.twist[3:]
twist[5] += root_vxy_yawrate[2]
```

Run: `pytest -q tests/test_m1_panda_rolling_teacher.py`

Expected: schedule and body-frame tests pass.

- [ ] **Step 6: Commit the command and trajectory primitives**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py
git commit -m "feat: add C1a rolling command primitives"
```

### Task 3: Prescribe base velocity in prioritized motion distribution

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_motion_distribution.py`

**Interfaces:**
- Consumes: optional `prescribed_base_velocity` shaped `(..., 3)` in `[vx, vy, yaw_rate]` order.
- Produces: the existing `MotionDistributionResult`, with `qd_coord[:3]` exactly equal to the prescribed value and Panda solving the remaining Cartesian twist.

- [ ] **Step 1: Write failing prescribed-base tests**

```python
def test_prescribed_base_velocity_is_fixed_and_arm_solves_remaining_twist():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 0.15
    inputs["prescribed_base_velocity"] = torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
    result = distribute_motion(**inputs)
    assert torch.equal(result.qd_coord[:3], inputs["prescribed_base_velocity"])
    assert result.qd_coord[3].item() == pytest.approx(0.10, abs=1e-6)
    assert result.base_active.item()


def test_omitted_prescribed_base_velocity_preserves_c0_arm_first_behavior():
    inputs = _coordination_inputs()
    inputs["desired_twist"][0] = 0.15
    result = distribute_motion(**inputs)
    assert torch.equal(result.qd_coord[:3], torch.zeros(3, dtype=torch.float64))
    assert not result.base_active.item()
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
pytest -q tests/test_m1_panda_motion_distribution.py -k prescribed
```

Expected: `TypeError: distribute_motion() got an unexpected keyword argument 'prescribed_base_velocity'`.

- [ ] **Step 3: Implement residual-space prescribed motion**

Add `prescribed_base_velocity: torch.Tensor | None = None` to `distribute_motion`. Validate its batch, dtype, device, and finite contracts. After computing normal velocity bounds:

```python
prescribed = (
    torch.zeros(batch_shape + (3,), dtype=q.dtype, device=q.device)
    if prescribed_base_velocity is None
    else prescribed_base_velocity
)
if ((prescribed < lower[..., :3]) | (prescribed > upper[..., :3])).any().item():
    raise ValueError("prescribed_base_velocity violates computed velocity bounds")
base_twist = torch.matmul(coordinated_jacobian[..., :, :3], prescribed.unsqueeze(-1)).squeeze(-1)
target = desired_twist + cfg.pose_gain * pose_error - base_twist
lower = lower.clone()
upper = upper.clone()
lower[..., :3] = 0.0
upper[..., :3] = 0.0
```

Before the solve loop, create `flat_prescribed = prescribed.reshape(count, 3)`. After each `_distribute_single`, add `flat_prescribed[index]` to the first three output coordinates and set `base_active` when any component of that row is nonzero. This prevents the arm-first solver from canceling commanded rolling motion.

- [ ] **Step 4: Run motion-distribution and C0 Teacher tests**

Run:

```bash
pytest -q tests/test_m1_panda_motion_distribution.py tests/test_m1_panda_wbc_teacher.py
```

Expected: all tests pass, including the existing C0 arm-first cases.

- [ ] **Step 5: Commit prescribed-base distribution**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py Go2Pvcnn/tests/test_m1_panda_motion_distribution.py
git commit -m "feat: prescribe C1a base motion distribution"
```

### Task 4: Add the rolling WBC configuration

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_rolling_wbc.py`

**Interfaces:**
- Consumes: existing `StandingWbcInput`, including nonzero base and wheel acceleration targets plus the hard 12-row bottom-contact Jacobian.
- Produces: `RollingWbcCfg`, `build_rolling_wbc_problem(state, cfg)`, and `solve_rolling_wbc(state, cfg)` returning existing `StandingWbcProblem`/`StandingWbcResult` contracts.

- [ ] **Step 1: Write failing WBC priority and target tests**

```python
from dataclasses import replace
import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination.rolling_wbc import (
    RollingWbcCfg,
    build_rolling_wbc_problem,
)
from test_m1_panda_standing_wbc import _input


def test_rolling_wbc_keeps_balance_above_velocity_and_arm_tracking():
    cfg = RollingWbcCfg()
    assert cfg.balance_weight > cfg.base_velocity_weight > cfg.arm_tracking_weight


def test_rolling_wbc_carries_nonzero_base_and_wheel_targets_into_qp():
    state = _input()
    base = state.base_acceleration.clone(); base[0] = 0.5
    wheels = torch.full((4,), 2.0, dtype=torch.float64)
    assembled = build_rolling_wbc_problem(replace(state, base_acceleration=base, wheel_acceleration=wheels))
    assert assembled.task_targets["base"][0].item() == pytest.approx(0.5)
    assert torch.equal(assembled.task_targets["wheels"], wheels)
    assert assembled.qp.equality_matrix.shape == (18, 43)
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `pytest -q tests/test_m1_panda_rolling_wbc.py`

Expected: collection fails because `rolling_wbc.py` does not exist.

- [ ] **Step 3: Implement the rolling wrapper without duplicating dynamics**

```python
from dataclasses import dataclass

from .standing_wbc import (
    StandingWbcCfg,
    StandingWbcInput,
    StandingWbcProblem,
    StandingWbcResult,
    build_standing_wbc_problem,
    solve_standing_wbc,
)


@dataclass(frozen=True)
class RollingWbcCfg:
    balance_weight: float = 1.0e6
    base_velocity_weight: float = 1.0e5
    leg_posture_weight: float = 2.0e4
    arm_tracking_weight: float = 1.0e4
    wheel_tracking_weight: float = 5.0e4
    force_equalization_weight: float = 10.0
    tangential_force_weight: float = 10.0
    regularization: float = 1.0e-6
    qp_tolerance: float = 1.0e-5

    def standing_cfg(self) -> StandingWbcCfg:
        return StandingWbcCfg(
            balance_weight=self.balance_weight,
            base_pose_weight=self.base_velocity_weight,
            leg_posture_weight=self.leg_posture_weight,
            arm_tracking_weight=self.arm_tracking_weight,
            wheel_stop_weight=self.wheel_tracking_weight,
            force_equalization_weight=self.force_equalization_weight,
            tangential_force_weight=self.tangential_force_weight,
            regularization=self.regularization,
            qp_tolerance=self.qp_tolerance,
        )


def build_rolling_wbc_problem(state: StandingWbcInput, cfg: RollingWbcCfg | None = None) -> StandingWbcProblem:
    cfg = cfg or RollingWbcCfg()
    return build_standing_wbc_problem(state, cfg.standing_cfg())


def solve_rolling_wbc(state: StandingWbcInput, cfg: RollingWbcCfg | None = None) -> StandingWbcResult:
    cfg = cfg or RollingWbcCfg()
    return solve_standing_wbc(state, cfg.standing_cfg())
```

- [ ] **Step 4: Run rolling and standing WBC regressions**

Run:

```bash
pytest -q tests/test_m1_panda_rolling_wbc.py tests/test_m1_panda_standing_wbc.py tests/test_m1_panda_qp_backend.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the rolling WBC**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py Go2Pvcnn/tests/test_m1_panda_rolling_wbc.py
git commit -m "feat: add M1 Panda rolling WBC"
```

### Task 5: Extend safety for rolling residual and stopped-before-retract

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_safety.py`

**Interfaces:**
- Consumes: optional `max_rolling_residual=0.0` and `base_speed=0.0` in `BalanceSafetySupervisor.update`.
- Produces: existing safety states plus `SafetyDecision.base_velocity_scale`; C0 callers that omit new values retain their current behavior.

- [ ] **Step 1: Write failing rolling safety tests**

```python
def test_rolling_residual_advances_safety_and_scale_applies_to_base():
    supervisor = _supervisor()
    _update(supervisor, max_rolling_residual=0.051)
    decision = _update(supervisor, max_rolling_residual=0.051)
    assert decision.state == SafetyState.SCALE
    assert decision.base_velocity_scale == pytest.approx(0.5)


def test_retract_waits_in_hold_until_base_is_below_stop_threshold():
    supervisor = _supervisor()
    for _ in range(8):
        decision = _update(supervisor, roll=math.radians(8.0), base_speed=0.03)
    assert decision.state == SafetyState.HOLD
    for _ in range(2):
        decision = _update(supervisor, roll=math.radians(8.0), base_speed=0.0)
    assert decision.state == SafetyState.RETRACT
```

Update `_update` test defaults to include `max_rolling_residual=0.0` and `base_speed=0.0`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
pytest -q tests/test_m1_panda_wbc_safety.py -k 'rolling_residual or retract_waits'
```

Expected: `update()` rejects the new keyword arguments.

- [ ] **Step 3: Implement backward-compatible rolling safety fields**

Add to `SafetyCfg`:

```python
max_rolling_residual: float = 0.05
retract_base_speed: float = 0.02
base_scaled_velocity_factor: float = 0.5
```

Add `base_velocity_scale: float` to `SafetyDecision`. Return `1.0` in TRACK, `0.5` in SCALE, and `0.0` from HOLD onward. Add rolling residual to the unsafe reason chain as `rolling_residual`. When `_state == HOLD`, block `_advance()` while `abs(base_speed) >= retract_base_speed`; keep `stop_wheels=True` so C1a requests rate-limited braking.

- [ ] **Step 4: Run safety, Teacher, and C0 static regressions**

Run:

```bash
pytest -q tests/test_m1_panda_wbc_safety.py tests/test_m1_panda_wbc_teacher.py tests/test_m1_panda_wbc_play_static.py
```

Expected: all tests pass and existing C0 decision values remain unchanged apart from the new reported scale field.

- [ ] **Step 5: Commit rolling safety**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py Go2Pvcnn/tests/test_m1_panda_wbc_safety.py
git commit -m "feat: extend balance safety for rolling"
```

### Task 6: Implement the deterministic rolling Teacher

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py`

**Interfaces:**
- Consumes: `RollingTeacherState`, which contains the C0 `TeacherState`, `root_xy_yaw`, `root_vxy_yawrate`, and rolling contact metrics.
- Produces: `RollingTeacherCommand`, containing the existing effort/QP/motion/safety fields plus raw/shaped `vx`, phase, and target wheel speeds.

- [ ] **Step 1: Write failing orchestration tests with recording solvers**

```python
def test_rolling_teacher_prescribes_base_and_nonzero_wheel_speed_after_phase_boundary():
    motion = _RecordingMotionDistributor()
    wbc = _RecordingWbcSolver()
    teacher = M1PandaRollingWbcTeacher(
        kp=torch.full((23,), 2.0, dtype=torch.float64),
        kd=torch.full((23,), 0.2, dtype=torch.float64),
        effort_limit=torch.full((23,), 100.0, dtype=torch.float64),
        safe_arm_target=torch.zeros(7, dtype=torch.float64),
        motion_distribution_fn=motion,
        wbc_solver_fn=wbc,
    )
    teacher.reset(_rolling_state(0), seed=42)
    for step in range(801):
        command = teacher.step(_rolling_state(step))
    assert command.phase == 1
    assert command.shaped_base_velocity_mps == pytest.approx(0.0005)
    assert torch.all(command.wheel_velocity_target > 0.0)
    assert motion.inputs[-1]["prescribed_base_velocity"][0].item() > 0.0


def test_hold_ramps_wheel_target_toward_zero_and_freezes_arm_target():
    teacher = _rolling_teacher()
    teacher.reset(_rolling_state(0), seed=42)
    commands = []
    for step in range(806):
        state = _rolling_state(step)
        if step >= 801:
            state = replace(state, teacher_state=replace(state.teacher_state, roll=math.radians(8.0)))
        commands.append(teacher.step(state))
    assert commands[-1].safety_state >= SafetyState.HOLD
    assert abs(commands[-1].shaped_base_velocity_mps) < abs(commands[-2].shaped_base_velocity_mps)
    assert torch.equal(commands[-1].q_des[-7:], commands[-2].q_des[-7:])
```

- [ ] **Step 2: Run orchestration tests and confirm RED**

Run: `pytest -q tests/test_m1_panda_rolling_teacher.py -k 'rolling_teacher or hold_ramps'`

Expected: imports or class lookup fail for `M1PandaRollingWbcTeacher`.

- [ ] **Step 3: Implement state and command contracts**

```python
@dataclass(frozen=True)
class RollingTeacherState:
    mission_step: int
    teacher_state: TeacherState
    root_xy_yaw: torch.Tensor
    root_vxy_yawrate: torch.Tensor
    max_rolling_residual_mps: float


@dataclass(frozen=True)
class RollingTeacherCommand:
    effort: torch.Tensor
    q_des: torch.Tensor
    qd_des: torch.Tensor
    target_pose: torch.Tensor
    target_twist: torch.Tensor
    motion_distribution: MotionDistributionResult
    qp_result: DenseQpResult
    safety_state: SafetyState
    safety_reason: str
    phase: int
    raw_base_velocity_mps: float
    shaped_base_velocity_mps: float
    wheel_velocity_target: torch.Tensor
    terminate: bool
```

- [ ] **Step 4: Implement one 200 Hz rolling control step**

Use `BandLimitedTrajectoryCfg(position_amplitude=0.005, orientation_amplitude=0.01)` so C1a retains the accepted C0 motion envelope. The step order must be explicit:

```python
prior_scale = self._last_safety_decision.base_velocity_scale
longitudinal = self._schedule.sample(state.mission_step, prior_scale)
base_velocity = state.teacher_state.coord_q.new_tensor((longitudinal.shaped_target_mps, 0.0, 0.0))
trajectory_sample = self._trajectory.sample(
    self._trajectory_time_s, state.root_xy_yaw, state.root_vxy_yawrate
)
distribution = self._motion_distribution_fn(
    coordinated_jacobian=state.teacher_state.coordinated_jacobian,
    pose_error=trajectory_sample.pose - state.teacher_state.ee_pose,
    desired_twist=trajectory_sample.twist * self._last_safety_decision.twist_scale,
    prescribed_base_velocity=base_velocity,
    q=state.teacher_state.coord_q,
    qd=state.teacher_state.coord_qd,
    q_min=state.teacher_state.coord_q_min,
    q_max=state.teacher_state.coord_q_max,
    v_max=state.teacher_state.coord_v_max,
    a_max=state.teacher_state.coord_a_max,
    manipulability_gradient=state.teacher_state.manipulability_gradient,
    sigma_min=state.teacher_state.sigma_min,
    dt=0.02,
)
wheel_target = wheel_speed_from_base_velocity(base_velocity[0], self._rolling_contact_cfg)
base_acceleration = state.teacher_state.wbc_input.base_acceleration.clone()
base_acceleration[0] = (base_velocity[0] - state.root_vxy_yawrate[0]) / 0.02
base_acceleration[1] = -state.root_vxy_yawrate[1] / 0.02
base_acceleration[5] = -state.root_vxy_yawrate[2] / 0.02
wheel_acceleration = torch.clamp(
    (wheel_target - state.teacher_state.controlled_qd[12:16]) / 0.02,
    min=-100.0,
    max=100.0,
)
```

Run the schedule and WBC every physics step. Recompute motion distribution only when `mission_step == 0` or `mission_step % 4 == 0`; retain the last verified distribution and continuously ramped base command between 50 Hz updates. Build a replaced `StandingWbcInput`, solve with `solve_rolling_wbc`, update `BalanceSafetySupervisor` with rolling residual and base speed, and generate 23-channel impedance commands. Keep legs anchored to the reset posture, set `qd_des[12:16]=wheel_target`, set Panda desired velocity from distribution, and use the same finite verified-command fallback as C0.

- [ ] **Step 5: Add reset determinism and failure-fallback tests**

Assert that reset clears schedule, QP history, trajectory time, wheel integral, held targets, and safety state. Assert a failed QP never reuses an unverified effort and advances safety. Assert the arm feed-forward uses current bias force as in C0.

- [ ] **Step 6: Run Teacher suites and commit**

Run:

```bash
pytest -q tests/test_m1_panda_rolling_teacher.py tests/test_m1_panda_wbc_teacher.py tests/test_m1_panda_wbc_safety.py
```

Expected: all tests pass.

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py
git commit -m "feat: add deterministic M1 Panda rolling Teacher"
```

### Task 7: Register an isolated C1a environment

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py`

**Interfaces:**
- Consumes: accepted `M1PandaWbcTeacherEnvCfg` and its ordered 23-effort action.
- Produces: `M1PandaWbcRollTeacherEnvCfg` and Gym ID `Isaac-M1-Panda-Wbc-Teacher-C1a-v0`.

- [ ] **Step 1: Write failing registration/config tests**

```python
def test_c1a_env_inherits_c0_effort_contract_and_has_runtime_margin():
    source = Path("go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py").read_text()
    assert "class M1PandaWbcRollTeacherEnvCfg(M1PandaWbcTeacherEnvCfg)" in source
    assert "self.decimation = 1" in source
    assert "self.episode_length_s = 30.0" in source


def test_c1a_has_independent_gym_registration():
    source = Path("go2_pvcnn/tasks/register_m1_envs.py").read_text()
    assert 'id="Isaac-M1-Panda-Wbc-Teacher-C1a-v0"' in source
    assert "m1_panda_wbc_roll_teacher_env_cfg:M1PandaWbcRollTeacherEnvCfg" in source
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_m1_panda_wbc_roll_play_static.py`

Expected: the new config file and registration are absent.

- [ ] **Step 3: Add the isolated environment and registration**

```python
from isaaclab.utils import configclass
from go2_pvcnn.tasks.m1_panda_wbc_teacher_env_cfg import M1PandaWbcTeacherEnvCfg


@configclass
class M1PandaWbcRollTeacherEnvCfg(M1PandaWbcTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
```

Register it with `rsl_rl_cfg_entry_point=None`. Do not change the C0 registration.

- [ ] **Step 4: Run task/static regressions and commit**

Run:

```bash
pytest -q tests/test_m1_panda_wbc_roll_play_static.py tests/test_m1_panda_wbc_play_static.py tests/test_m1_panda_assets.py
```

Expected: all tests pass.

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "feat: register M1 Panda WBC Teacher C1a"
```

### Task 8: Wire the PhysX adapter and C1a play/summary contract

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_wbc_play.py`
- Create: `Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py`

**Interfaces:**
- Consumes: one live combined articulation and `M1PandaRollingWbcTeacher`.
- Produces: GUI/headless C1a play, `C1aSummary.to_dict()`, periodic diagnostics, atomic JSON, and nonzero exit status when hard gates fail.

- [ ] **Step 1: Characterize and parameterize the C0 adapter**

Add a C0 test asserting default construction retains `WHEEL_RADIUS_M == 0.0959`. Change the adapter constructor to accept `wheel_radius_m: float = WHEEL_RADIUS_M`, store it, and replace the hardcoded contact offset with `self.wheel_radius_m`. Expose CPU float64 fields after every `build_state`:

```python
self.latest_root_xy_yaw = torch.tensor((root_position[0], root_position[1], yaw), dtype=torch.float64)
self.latest_root_vxy_yawrate = torch.tensor((root_linear_velocity[0], root_linear_velocity[1], root_angular_velocity[2]), dtype=torch.float64)
self.latest_generalized_velocity = generalized_velocity.clone()
self.latest_contact_jacobian = contact_jacobian.clone()
self.latest_wheel_velocity = controlled_qd[12:16].clone()
```

Run: `pytest -q tests/test_m1_panda_wbc_play_static.py`

Expected: all C0 static tests pass.

- [ ] **Step 2: Write failing C1a CLI and summary tests**

Require the new script to contain:

```python
TASK_ID = "Isaac-M1-Panda-Wbc-Teacher-C1a-v0"
MISSION_STEPS = 4000
WHEEL_RADIUS_M = 0.095
```

Test `validate_args` rejects `num_envs != 1`, negative steps, and a nonpositive stats interval. Test `C1aSummary.to_dict()` includes:

```text
phase_counts
vx_rmse_mps
forward_displacement_m
reverse_displacement_m
stop_settle_time_s
max_rolling_residual_mps
max_lateral_slip_mps
max_wheel_velocity_spread_radps
wheel_effort_saturation_count
wheel_direction_mismatch_count
qp_feasible_rate
max_qp_equality_residual
max_qp_inequality_violation
safety_state_counts
exit_reason
hard_gates_passed
```

- [ ] **Step 3: Implement the C1a runtime loop**

Use `RollingPhysxTeacherAdapter(PhysxTeacherAdapter)` to call the parameterized C0 adapter with `wheel_radius_m=0.095`. Make the shared adapter construct its contact Jacobian through `build_wheel_contact_jacobian(..., RollingContactCfg(wheel_radius_m=self.wheel_radius_m))`, then compute `rolling_contact_metrics` and wrap the base `TeacherState` in `RollingTeacherState`. Settle for 100 unscored steps, reset Teacher and trajectory on the physically realized state, then pass a reset-relative `mission_step` starting at zero to the rolling Teacher.

During phases with `abs(wheel_velocity_target) >= 0.1 rad/s`, increment `wheel_direction_mismatch_count` when any sign-corrected measured wheel speed is below `-0.02 rad/s`. This is the startup/runtime wheel-sign consistency gate; do not auto-flip signs.

The default GUI command keeps Panda motion enabled. `--disable-target-motion` replaces only the local Panda trajectory with a static local target; it must not disable the rolling schedule.

- [ ] **Step 4: Implement hard-gate evaluation**

The summary must set `hard_gates_passed` only when all conditions are true:

```python
gates = (
    summary.steps == 4000,
    summary.completed_phase_count == 5,
    summary.vx_rmse_mps <= 0.03,
    summary.stop_settle_time_s is not None and summary.stop_settle_time_s <= 1.0,
    summary.forward_displacement_m > 0.0,
    summary.reverse_displacement_m < 0.0,
    summary.max_rolling_residual_mps <= 0.05,
    summary.max_lateral_slip_mps <= 0.05,
    summary.min_wheel_contact_count == 4,
    summary.max_abs_roll_rad <= math.radians(10.0),
    summary.max_abs_pitch_rad <= math.radians(10.0),
    summary.max_ee_position_error_m <= 0.03,
    summary.wheel_direction_mismatch_count == 0,
    summary.qp_feasible_count / max(summary.steps, 1) >= 0.999,
    summary.track_scale_count / max(summary.steps, 1) >= 0.99,
    summary.hold_or_worse_count == 0,
    summary.joint_limit_violations == 0,
    summary.base_contacts == 0,
    summary.non_finite_count == 0,
    summary.arm_snap_count == 0,
    summary.reset_count == 0,
    summary.exit_reason == "steps_complete",
)
```

For `--steps 4000`, return exit code `0` only when all formal gates pass. For `0 < --steps < 4000`, use `exit_reason="smoke_complete"` and return `0` only when state, command, QP, contact count, and reset checks are healthy; do not set `hard_gates_passed`. `--steps 0` GUI mode exits normally when the application closes without claiming formal acceptance.

- [ ] **Step 5: Run static and pure controller suites**

Run:

```bash
pytest -q \
  tests/test_m1_panda_wbc_roll_play_static.py \
  tests/test_m1_panda_wbc_play_static.py \
  tests/test_m1_panda_rolling_contact.py \
  tests/test_m1_panda_rolling_teacher.py \
  tests/test_m1_panda_rolling_wbc.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the play path**

```bash
git add Go2Pvcnn/scripts/m1_panda_wbc_play.py Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "feat: add M1 Panda C1a rolling play"
```

### Task 9: Run the complete pure/static regression gate

**Files:**
- Modify only files implicated by a failing test; preserve each prior task's public interfaces.

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces: a clean C0+C1a pure/static baseline before launching Isaac Sim.

- [ ] **Step 1: Run formatting-independent syntax checks**

```bash
python -m py_compile \
  go2_pvcnn/control/m1_panda_coordination/rolling_contact.py \
  go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py \
  go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py \
  go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py \
  scripts/m1_panda_wbc_roll_play.py
```

Expected: exit `0` with no output.

- [ ] **Step 2: Run all M1 + Panda WBC tests**

```bash
pytest -q \
  tests/test_m1_panda_wbc_contracts.py \
  tests/test_m1_panda_wbc_kinematics.py \
  tests/test_m1_panda_motion_distribution.py \
  tests/test_m1_panda_qp_backend.py \
  tests/test_m1_panda_standing_wbc.py \
  tests/test_m1_panda_wbc_safety.py \
  tests/test_m1_panda_wbc_teacher.py \
  tests/test_m1_panda_wbc_play_static.py \
  tests/test_m1_panda_rolling_contact.py \
  tests/test_m1_panda_rolling_wbc.py \
  tests/test_m1_panda_rolling_teacher.py \
  tests/test_m1_panda_wbc_roll_play_static.py
```

Expected: every selected test passes.

- [ ] **Step 3: Inspect scope and commit any regression-only correction**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only C1a files or intentionally modified shared C0 files are changed. If a correction was necessary, commit it with:

```bash
git add \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_contact.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_wbc_roll_teacher_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py \
  Go2Pvcnn/scripts/m1_panda_wbc_play.py \
  Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py \
  Go2Pvcnn/tests/test_m1_panda_motion_distribution.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_safety.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_contact.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_wbc.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "fix: preserve C0 through C1a regression"
```

Do not include `graphify-out/`.

### Task 10: Validate GPU0 startup and rolling without Panda motion

**Files:**
- Modify: C1a configuration/controller constants only when evidence shows a gate failure.
- Create evidence later in Task 12; use `/tmp` for transient JSON now.

**Interfaces:**
- Consumes: the C1a play path.
- Produces: verified articulation startup, correct wheel signs, stable rolling/stop/reverse dynamics without arm trajectory excitation.

- [ ] **Step 1: Run an eight-step static GPU0 smoke**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 8 --seed 42 \
  --disable-target-motion \
  --summary-json /tmp/m1_panda_wbc_c1a_static8.json
```

Expected: process exits without traceback; JSON is finite, has four wheel contacts, and reports phase 0. A short smoke is not formal hard-gate acceptance, so the script must distinguish `smoke_complete` from `steps_complete` rather than falsely passing 4000-step gates.

- [ ] **Step 2: Run the 4000-step no-arm baseline**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 4000 --seed 42 \
  --disable-target-motion --stats-interval 400 \
  --summary-json /tmp/m1_panda_wbc_c1a_no_arm.json
```

Expected: all five phases complete; speed signs, stop convergence, QP, contact, roll/pitch, slip, limits, reset, and finite-state gates pass.

- [ ] **Step 3: Diagnose before tuning if a gate fails**

Use the JSON and periodic fields to assign failure to exactly one layer:

- wrong sign or `v/r`: `rolling_contact.py`;
- slow/overshooting command: schedule acceleration limit or base/wheel acceleration feedback in `rolling_teacher.py`;
- QP infeasible: rolling WBC weights, bounds, equality residual, or torque limit;
- lateral slip/contact loss: contact Jacobian/radius first, then WBC weights;
- posture drift: balance/base/leg targets;
- reset or non-finite: adapter/state construction.

Add a failing test reproducing the diagnosed condition before changing runtime code. Re-run Task 9 after each correction.

- [ ] **Step 4: Commit only evidence-driven corrections**

```bash
git diff --check
git add \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_contact.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py \
  Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_contact.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_wbc.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "fix: stabilize M1 Panda C1a rolling baseline"
```

Skip the commit when no runtime correction was needed.

### Task 11: Validate combined rolling and Panda motion on GPU0

**Files:**
- Modify: C1a-only trajectory/controller configuration when a reproduced failure requires it.

**Interfaces:**
- Consumes: accepted no-arm rolling baseline.
- Produces: the formal 4000-step combined C1a acceptance JSON.

- [ ] **Step 1: Run the formal combined acceptance**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --headless --device cuda:0 --steps 4000 --seed 42 \
  --stats-interval 400 \
  --summary-json /tmp/m1_panda_wbc_c1a_motion.json
```

Expected: exit `0`, `hard_gates_passed=true`, `exit_reason="steps_complete"`, and every threshold in the approved specification passes.

- [ ] **Step 2: Compare against the no-arm baseline**

Read both JSON files and confirm Panda motion did not cause contact loss, HOLD-or-worse safety states, QP feasibility below `0.999`, roll/pitch above `10 deg`, rolling/slip residual above `0.05 m/s`, or arm error above `0.03 m`.

- [ ] **Step 3: Re-run accepted C0 GPU0 motion regression**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_play.py \
  --headless --device cuda:0 --steps 2000 --seed 42 \
  --stats-interval 500 \
  --summary-json /tmp/m1_panda_wbc_c0_after_c1a.json
```

Expected: the existing C0 hard gates pass and the run exits `0`.

- [ ] **Step 4: Commit any test-driven combined-motion correction**

If correction was required, run Task 9 again and commit exact files with:

```bash
git add \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_wbc.py \
  Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py \
  Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_wbc.py \
  Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "fix: validate combined rolling and Panda tracking"
```

Skip this commit when the first combined run passed.

### Task 12: Document play commands and acceptance evidence

**Files:**
- Create: `docs/superpowers/runbooks/2026-08-18-m1-panda-wbc-teacher-c1a.md`
- Create: `notes/log/2026-08-18-m1-panda-wbc-teacher-c1a.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: actual Task 9–11 commands, exit codes, commit IDs, GPU identity, and JSON values.
- Produces: reproducible GUI/headless instructions and an auditable C1a acceptance record.

- [ ] **Step 1: Write the runbook with exact commands**

Include the GUI command:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_wbc_roll_play.py \
  --device cuda:0 --steps 0 --seed 42
```

Include the static smoke, no-arm baseline, formal 4000-step combined acceptance, all hard gates, safety-state meanings, and the statement that C1a is deterministic Teacher play—not PPO, Student training, external-force curriculum, turning, terrain, or grasping.

- [ ] **Step 2: Record actual evidence**

Copy measured values from `/tmp/m1_panda_wbc_c1a_no_arm.json`, `/tmp/m1_panda_wbc_c1a_motion.json`, and `/tmp/m1_panda_wbc_c0_after_c1a.json` into the log. Record commands exactly, exit codes, selected test count, GPU name, driver, code commit, and whether each gate passed. Do not invent or round a failed gate into a pass.

- [ ] **Step 3: Update task and log indexes**

Mark only C1a complete in T400 when formal combined acceptance passed. Name C1b turning as the next separately approved design. Add one dated row to `notes/log/index.md` pointing to the evidence log and T400.

- [ ] **Step 4: Verify documentation and repository state**

```bash
git diff --check
rg -n "C1a|4000|m1_panda_wbc_roll_play|hard_gates_passed|deterministic Teacher" \
  docs/superpowers/runbooks/2026-08-18-m1-panda-wbc-teacher-c1a.md \
  notes/log/2026-08-18-m1-panda-wbc-teacher-c1a.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md
git status --short
```

Expected: no whitespace errors; all required commands and evidence fields exist; `graphify-out/` remains unstaged.

- [ ] **Step 5: Commit documentation**

```bash
git add \
  docs/superpowers/runbooks/2026-08-18-m1-panda-wbc-teacher-c1a.md \
  notes/log/2026-08-18-m1-panda-wbc-teacher-c1a.md \
  notes/log/index.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: verify M1 Panda WBC Teacher C1a"
```

## Final Verification

- [ ] Run the complete Task 9 pure/static suite once more at the final code commit.
- [ ] Confirm the no-arm and combined C1a JSON files both contain finite metrics and all required fields.
- [ ] Confirm the combined 4000-step run passes every approved hard gate.
- [ ] Confirm the post-C1a C0 2000-step regression still passes.
- [ ] Run `git diff --check` and `git status --short`.
- [ ] Confirm only the pre-existing `graphify-out/` cache modification remains outside committed work.

C1a completion does not authorize C1b, C2, C3, Student, grasping, or real-hardware work.
