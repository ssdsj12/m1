# M1 + Panda 8D Residual WBC First-Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the PDF-recommended Phase 1–4 8D residual controller over the accepted M1 + Panda WBC/QP while preserving every existing 23D, C0, C1a, and folded-load default.

**Architecture:** A new pure PyTorch residual module owns normalized-to-physical conversion, per-environment slew state, mount-wrench filtering, feedback, and posture corrections. Existing standing and rolling Teachers accept an optional residual command; omitted or zero commands preserve their current WBC problems and efforts. A thin 8D controller wrapper and dedicated play entrypoint expose manual validation without introducing PPO or Arm MPC.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab 2.x / Isaac Sim 5.1, the repository float64 reference QP, pytest, Gymnasium.

## Global Constraints

- Work as one inline agent; do not dispatch subagents.
- Preserve `Isaac-M1-Panda-Coordinated-v0`, its exact 103/23 contract, existing checkpoints, folded-load tasks, C0, and C1a defaults.
- Preserve 200 Hz physics/control (`dt=0.005 s`, `decimation=1`) and the accepted zero-clearance asset.
- Add no Arm MPC, PPO actor, long training, grasping, Student, or real-robot claim.
- New normalized action order is exactly `[Fx,Fy,Fz,Mx,My,Mz,delta_height,delta_stance]`.
- Physical limits are exactly `[30,30,50,15,15,8,0.04,0.08]` in N, Nm, m, and rad.
- No production behavior is written before its focused test has failed for the expected reason.
- Keep the unrelated tracked `Go2Pvcnn/graphify-out/cache/last_query_stamp` change out of all commits.

---

### Task 1: Stateful 8D residual and mount-wrench feedback contracts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/whole_body_residual.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_whole_body_residual.py`

**Interfaces:**
- Produces `WholeBodyResidualCfg`, `WholeBodyResidualCommand`, `WholeBodyResidualDiagnostics`, `WholeBodyResidualComposer`, `MountWrenchFeedbackCfg`, and `MountWrenchFeedback`.
- `WholeBodyResidualComposer.step(normalized: torch.Tensor) -> tuple[WholeBodyResidualCommand, WholeBodyResidualDiagnostics]` accepts `(num_envs,8)`.
- `WholeBodyResidualComposer.reset(env_ids: torch.Tensor | Sequence[int] | None = None) -> None` selectively clears state.
- `MountWrenchFeedback.update(measured_wrench_b, residual_wrench) -> torch.Tensor` returns the clipped virtual correction wrench.

- [ ] **Step 1: Write failing action-contract tests**

```python
def test_residual_channel_order_and_physical_limits():
    composer = WholeBodyResidualComposer(2, "cpu", torch.float64)
    command, diagnostics = composer.step(torch.tensor([
        [1., -1., 1., -1., 1., -1., 1., -1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ], dtype=torch.float64))
    assert torch.equal(command.physical[0], torch.tensor(
        [1.5, -1.5, 2.5, -.75, .75, -.4, .002, -.004],
        dtype=torch.float64,
    ))
    assert diagnostics.slew_saturated[0].all()
```

The first-step expectation uses the exact default 5% per-control-step slew values; repeated steps must reach the exact physical limits.

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_whole_body_residual.py`

Expected: collection fails because `whole_body_residual` does not exist.

- [ ] **Step 3: Implement validated config, frozen result dataclasses, scaling, slew, diagnostics, and selective reset**

```python
RESIDUAL_DIM = 8
RESIDUAL_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz", "delta_height", "delta_stance")

@dataclass(frozen=True)
class WholeBodyResidualCfg:
    physical_limits: tuple[float, ...] = (30., 30., 50., 15., 15., 8., .04, .08)
    slew_fraction_per_step: float = .05

@dataclass(frozen=True)
class WholeBodyResidualCommand:
    physical: torch.Tensor
    wrench_b: torch.Tensor
    delta_height: torch.Tensor
    delta_stance: torch.Tensor
```

Validate config lengths/positivity/finiteness, exact batch/device/dtype, and finite input before mutating state. Clone all returned tensors so callers cannot mutate composer state.

- [ ] **Step 4: Run GREEN and regression**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_whole_body_residual.py tests/test_m1_residual_action.py`

Expected: all pass.

- [ ] **Step 5: Add failing wrench-filter tests**

```python
def test_mount_feedback_initializes_filter_without_zero_state_impulse():
    feedback = MountWrenchFeedback(2, "cpu", torch.float64)
    measured = torch.tensor([[10., 0., 0., 0., 0., 0.], [0., 2., 0., 0., 0., 0.]], dtype=torch.float64)
    output = feedback.update(measured, torch.zeros_like(measured))
    assert torch.allclose(output[:, 0:3], -.15 * measured[:, 0:3])

def test_zero_gain_and_zero_residual_are_exact_zero():
    cfg = MountWrenchFeedbackCfg(force_gain=0., moment_gain=0.)
    feedback = MountWrenchFeedback(1, "cpu", torch.float64, cfg)
    assert torch.equal(feedback.update(torch.ones(1, 6), torch.zeros(1, 6)), torch.zeros(1, 6))
```

- [ ] **Step 6: Run RED, implement feedback, then run GREEN**

Implement a first-sample initialized EMA, optional warm-up bias accumulator, explicit zero reference, gains `(0.15,0.10)`, final physical clipping, finite-before-state-mutation, and selective reset.

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_whole_body_residual.py`

Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/whole_body_residual.py Go2Pvcnn/tests/test_m1_panda_whole_body_residual.py
git commit -m "feat: add 8d whole-body residual contracts"
```

### Task 2: Height/stance mapping and zero-equivalent WBC application

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/whole_body_residual.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_residual_wbc.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_residual_qp.py`

**Interfaces:**
- Produces `ResidualWbcCfg` and `apply_residual_to_wbc(state, command, *, nominal_leg_target, leg_soft_limits, safety_scale, cfg) -> tuple[StandingWbcInput, torch.Tensor]`.
- Returned second tensor is the clipped 12D desired leg target used by Teacher impedance.

- [ ] **Step 1: Write failing zero-equivalence and posture tests**

```python
def test_zero_residual_keeps_every_wbc_tensor_equal():
    state = make_wbc_input()
    command = zero_command()
    result, leg_target = apply_residual_to_wbc(
        state, command,
        nominal_leg_target=torch.zeros(12, dtype=torch.float64),
        leg_soft_limits=torch.tensor([[-1., 1.]] * 12, dtype=torch.float64),
        safety_scale=1.,
    )
    for field in dataclasses.fields(StandingWbcInput):
        before, after = getattr(state, field.name), getattr(result, field.name)
        assert torch.equal(before, after) if isinstance(before, torch.Tensor) else before == after

def test_height_and_stance_modify_only_approved_targets():
    # height gain 40 and leg gain 80 are inherited from runtime_adapter.
    assert result.base_acceleration[2] == pytest.approx(state.base_acceleration[2] + 40. * .04)
    assert torch.equal(result.leg_acceleration[::3] - state.leg_acceleration[::3], 80. * expected_abad_offset)
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_wbc.py`

Expected: import fails for `ResidualWbcCfg`/`apply_residual_to_wbc`.

- [ ] **Step 3: Implement immutable WBC transform**

Use `dataclasses.replace`, exact ABAD indices `(0,3,6,9)`, signs `(1,-1,1,-1)`, base gain `40`, leg gain `80`, soft-limit clipping, and `safety_scale` in `[0,1]`. Reject invalid input without changing `StandingWbcInput` or command.

- [ ] **Step 4: Run GREEN**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_wbc.py tests/test_m1_panda_standing_wbc.py tests/test_m1_panda_rolling_wbc.py`

Expected: all pass.

- [ ] **Step 5: Write and verify QP response tests**

```python
def test_six_axis_residual_enters_generalized_force_with_expected_sign():
    problem = build_standing_wbc_problem(transformed)
    assert torch.equal(problem.external_generalized_force, transformed.mount_wrench_jacobian.T @ command.wrench_b)

def test_zero_residual_qp_problem_is_exactly_equal_to_baseline():
    assert_problem_tensors_equal(build_standing_wbc_problem(original), build_standing_wbc_problem(transformed_zero))
```

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_qp.py tests/test_m1_panda_qp_backend.py`

Expected: all pass with finite positive/negative axis cases.

- [ ] **Step 6: Commit Task 2**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/whole_body_residual.py Go2Pvcnn/tests/test_m1_panda_residual_wbc.py Go2Pvcnn/tests/test_m1_panda_residual_qp.py
git commit -m "feat: map residual wrench and posture into wbc"
```

### Task 3: Continuous base participation

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_motion_distribution.py`

**Interfaces:**
- `MotionDistributionCfg` adds `sigma_safe=0.20` and `sigma_critical=0.08`.
- `MotionDistributionResult` adds `base_participation: torch.Tensor` with the same batch shape as `sigma_min`.
- Existing `base_active` remains and equals final participation greater than zero.

- [ ] **Step 1: Add failing boundary/interpolation tests**

```python
@pytest.mark.parametrize((sigma, expected), [(0.20, 0.), (0.14, .5), (0.08, 1.)])
def test_base_participation_follows_singularity_margin(sigma, expected):
    inputs = _coordination_inputs()
    inputs["sigma_min"] = torch.tensor(sigma, dtype=torch.float64)
    result = distribute_motion(**inputs)
    assert result.base_participation.item() == pytest.approx(expected)
```

Also update local test doubles constructing `MotionDistributionResult` to provide `base_participation`.

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_motion_distribution.py tests/test_m1_panda_wbc_teacher.py tests/test_m1_panda_rolling_teacher.py`

Expected: missing config/result field failures.

- [ ] **Step 3: Implement participation and scaled planar bounds**

Compute `alpha=clamp((safe-sigma)/(safe-critical),0,1)`. For unprescribed base motion, scale the first three lower/upper bounds by alpha; alpha zero retains the exact arm-only solve. If scaled authority cannot solve the task, retain the existing full-authority fallback and return participation one with `base_active=True`.

- [ ] **Step 4: Run GREEN and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_motion_distribution.py tests/test_m1_panda_wbc_teacher.py tests/test_m1_panda_rolling_teacher.py tests/test_m1_panda_base_assist.py`

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/motion_distribution.py Go2Pvcnn/tests/test_m1_panda_motion_distribution.py Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py
git commit -m "feat: add continuous base participation"
```

### Task 4: Residual safety projection and Teacher integration

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_safety.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py`

**Interfaces:**
- Produces `residual_scale_for_safety(state, scale_factor) -> float` returning `1`, configured SCALE factor, or `0` for HOLD and above.
- `M1PandaWbcTeacher.step(state, *, residual_command=None, leg_soft_limits=None)` and `M1PandaRollingTeacher.step(..., residual_command=None, leg_soft_limits=None)` remain backward compatible.

- [ ] **Step 1: Write failing safety projection tests**

```python
def test_residual_scale_matches_safety_state():
    assert residual_scale_for_safety(SafetyState.TRACK, .5) == 1.
    assert residual_scale_for_safety(SafetyState.SCALE, .5) == .5
    assert residual_scale_for_safety(SafetyState.HOLD, .5) == 0.
```

- [ ] **Step 2: Run RED, implement pure projection, run GREEN**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_wbc_safety.py`

- [ ] **Step 3: Write failing Teacher injection tests**

Record WBC solver inputs and assert optional residual changes only `external_wrench`, base z acceleration, and ABAD leg acceleration. Assert omitted residual gives byte-for-byte equal inputs and identical command effort. Force HOLD and assert the second/fallback solve receives zero residual.

- [ ] **Step 4: Run RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_wbc_teacher.py tests/test_m1_panda_rolling_teacher.py`

Expected: `step` rejects the new keyword or recorded inputs remain unchanged.

- [ ] **Step 5: Implement optional injection without changing defaults**

Apply the residual after building the nominal WBC input and before solving. Use prior safety state for the first solve; rebuild with zero scale for HOLD/RETRACT/TERMINATE. Use the transformed clipped leg target when assigning `q_des[:12]`. Expose a read-only `safety_state` property instead of reaching into private supervisor state.

- [ ] **Step 6: Run regression and commit**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_wbc_safety.py tests/test_m1_panda_wbc_teacher.py tests/test_m1_panda_rolling_teacher.py tests/test_m1_panda_standing_wbc.py tests/test_m1_panda_rolling_wbc.py`

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/safety.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/teacher.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py Go2Pvcnn/tests/test_m1_panda_wbc_safety.py Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py
git commit -m "feat: inject safe residuals into wbc teachers"
```

### Task 5: Runtime mount/posture state and residual observations

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/residual_observation.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_residual_observation.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py`

**Interfaces:**
- `PhysxTeacherAdapter.read_mount_wrench_b() -> torch.Tensor` reuses the canonical MDP conversion and returns one CPU float64 6-vector.
- `PhysxTeacherAdapter.leg_soft_limits() -> torch.Tensor` returns one CPU float64 `(12,2)` clone.
- Produces `ResidualObservationParts`, `ResidualObservation`, and `build_residual_observation(parts) -> ResidualObservation`.

- [ ] **Step 1: Write failing observation tests**

Construct batched named groups and assert exact group widths, flatten order, previous physical residual position, clone isolation, and finite/dtype/device validation. Include support margin, `sigma_min`, joint-limit min/mean margins, and filtered mount wrench.

- [ ] **Step 2: Run RED, implement, run GREEN**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_observation.py`

- [ ] **Step 3: Add static/runtime-adapter contract tests**

Assert the adapter resolves `panda_link0` and `BASE_LINK`, calls canonical shift/rotate semantics, returns CPU float64 clones, and obtains leg limits in canonical `WbcJointMap.legs` order.

- [ ] **Step 4: Run RED, implement adapter methods, run GREEN**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_observation.py tests/test_m1_panda_wbc_play_static.py tests/test_m1_panda_wbc_roll_play_static.py tests/test_m1_panda_wrench.py`

- [ ] **Step 5: Commit Task 5**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/residual_observation.py Go2Pvcnn/tests/test_m1_panda_residual_observation.py Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "feat: expose residual observations and mount state"
```

### Task 6: Independent 8D controller wrapper and action boundary

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_residual_wbc_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_residual_action_contract.py`

**Interfaces:**
- Produces `M1PandaResidualWbcController` and `ResidualWbcStep`.
- `reset(env_ids=None)` resets composer/filter and the selected Teacher state.
- `step(state, normalized_residual, measured_mount_wrench_b, leg_soft_limits, **teacher_kwargs) -> ResidualWbcStep` composes, filters, invokes the Teacher, and returns command plus diagnostics.

- [ ] **Step 1: Write failing wrapper tests**

Test exact `(N,8)` validation, output WBC command effort shape `(23,)` per environment, all-zero equivalence to direct Teacher, selective reset, feedback sign, no mutation of caller action, and no import of PPO/Arm MPC.

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_action_contract.py`

Expected: wrapper module missing.

- [ ] **Step 3: Implement one-controller-per-environment orchestration**

Follow `BatchedRollingTeacherBank`: no mutable Teacher/adapter sharing, loop only at the reference correctness boundary, preserve per-env composer/filter state, and return frozen diagnostics. Do not modify the 23D coordinated wrapper.

- [ ] **Step 4: Run GREEN and baseline action regressions**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_action_contract.py tests/test_m1_panda_batched_rolling_teacher.py tests/test_m1_panda_coordinated_env_static.py tests/test_m1_panda_folded_load_wrapper.py`

- [ ] **Step 5: Commit Task 6**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_residual_wbc_wrapper.py Go2Pvcnn/tests/test_m1_panda_residual_action_contract.py
git commit -m "feat: add independent 8d residual wbc controller"
```

### Task 7: Dedicated registration, play/probe, and CPU verification

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/scripts/m1_panda_residual_wbc_play.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_residual_wbc_play_static.py`
- Create: `Go2Pvcnn/docs/m1_panda_residual_wbc_runbook.md`

**Interfaces:**
- Registers `Isaac-M1-Panda-Residual-Wbc-v0` with the accepted C1a environment config while the script owns the external 8D controller boundary.
- Play defaults to zero residual and supports `--residual-axis`, `--residual-value`, `--warmup-steps`, `--steps`, `--headless`, and `--device`.
- JSON summary contains action/wrench maxima, QP feasibility, contacts, base contact, orientation, joint limits, resets, EE error, safety counts, and `exit_reason`.

- [ ] **Step 1: Write failing static entrypoint tests**

Assert unique Gym ID, no import-time Isaac app launch, exact default zero 8D action, explicit axis validation, no learning call, and required JSON keys.

- [ ] **Step 2: Run RED, implement entrypoint/runbook, run GREEN**

Run: `cd Go2Pvcnn && pytest -q tests/test_m1_panda_residual_wbc_play_static.py tests/test_m1_panda_wbc_env_static.py`

- [ ] **Step 3: Run CPU reference suite**

Run:

```bash
cd Go2Pvcnn
pytest -q \
  tests/test_m1_panda_whole_body_residual.py \
  tests/test_m1_panda_residual_observation.py \
  tests/test_m1_panda_residual_wbc.py \
  tests/test_m1_panda_residual_qp.py \
  tests/test_m1_panda_residual_action_contract.py \
  tests/test_m1_panda_residual_wbc_play_static.py \
  tests/test_m1_panda_motion_distribution.py \
  tests/test_m1_panda_standing_wbc.py \
  tests/test_m1_panda_rolling_wbc.py \
  tests/test_m1_panda_wbc_safety.py \
  tests/test_m1_panda_wbc_teacher.py \
  tests/test_m1_panda_rolling_teacher.py
```

Expected: all pass.

- [ ] **Step 4: Commit Task 7**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py Go2Pvcnn/scripts/m1_panda_residual_wbc_play.py Go2Pvcnn/tests/test_m1_panda_residual_wbc_play_static.py Go2Pvcnn/docs/m1_panda_residual_wbc_runbook.md
git commit -m "feat: add residual wbc play and registration"
```

### Task 8: GPU0 smoke, full regression, and project records

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-08-26-m1-panda-8d-residual-wbc-implementation.md`
- Create: `notes/log/2026-08-26-m1-panda-8d-residual-wbc-gpu0-smoke.md`

**Interfaces:**
- No new runtime interface; records exact commands, exits, commit, GPU, asset SHA, and JSON metrics.

- [ ] **Step 1: Run compile and focused/full relevant regression**

```bash
cd Go2Pvcnn
python -m compileall -q go2_pvcnn/control/m1_panda_coordination go2_pvcnn/tasks scripts/m1_panda_residual_wbc_play.py
pytest -q tests/test_m1_panda_*wbc*.py tests/test_m1_panda_motion_distribution.py tests/test_m1_panda_coordinated_env_static.py tests/test_m1_panda_folded_load_wrapper.py
git diff --check
```

Expected: exit `0` for every command.

- [ ] **Step 2: Run GPU0 zero-residual smoke**

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 ./go2pvcnn.sh -p scripts/m1_panda_residual_wbc_play.py \
  --task Isaac-M1-Panda-Residual-Wbc-v0 --headless --device cuda:0 \
  --warmup-steps 64 --steps 256 --residual-value 0.0
```

Expected: `exit_reason=steps_complete`, QP feasibility `1.0`, finite diagnostics, four contacts, zero base contact/orientation/limit/reset failures.

- [ ] **Step 3: Run GPU0 six-axis and posture smoke**

Run the same command for axes `0..7` at `+0.1` and `-0.1`, 128 mission steps each. Stop on the first hard-gate failure. Compare the zero segment EE error and snap/reset diagnostics with the existing C1a short baseline.

- [ ] **Step 4: Update records with observed evidence**

Record facts only. If GPU execution is unavailable or a gate fails, mark T400.12 partial/open and include the exact blocker; do not claim completion.

- [ ] **Step 5: Verify records and commit**

```bash
git diff --check
git status --short
git add notes/todo.md notes/todo/T400-m1-panda-force-aware-teacher-student.md notes/log/index.md notes/log/2026-08-26-m1-panda-8d-residual-wbc-implementation.md notes/log/2026-08-26-m1-panda-8d-residual-wbc-gpu0-smoke.md
git commit -m "docs: record 8d residual wbc verification"
```

## Self-Review

- Spec coverage: Tasks 1–7 cover all Phase 1–4 code boundaries; Task 8 covers CPU/GPU evidence and project records.
- Compatibility: old 23D wrappers, PPO configs, folded-load modules, QP backend, USD, and vendored RSL-RL are not modified.
- Type consistency: the one `WholeBodyResidualCommand` produced in Task 1 is consumed by Tasks 2, 4, and 6; safety produces a scalar consumed by the immutable WBC transform.
- Scope control: no Arm MPC file, PPO network, long train command, grasp task, or real deployment is included.
- Placeholder scan: every behavior and verification command is explicit.
