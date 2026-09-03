# M1 Bimanual Latent-Action Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed-base M1 + dual-Panda + dual-O6 controller whose 25 Hz full 43-action MPC teacher is compressed into a 16-dimensional latent action and reproduced by a 200 Hz body controller behind a hard safety projection.

**Architecture:** Isaac state is converted atomically from world coordinates to `BASE_LINK`, augmented with full 59-DOF constrained dynamics, and consumed by a one-second/25-node teacher that emits a `(25, 43)` active-effort trajectory. A conditional encoder refreshes `z[16]` at 25 Hz; a 200 Hz body controller maps current state, `z`, teacher phase, and prior effort to a 43-dimensional candidate, which a deterministic safety layer projects before one atomic environment write.

**Tech Stack:** Python 3.10, PyTorch CPU `float64` control contracts, PyTorch `float32` learned models, Isaac Sim/Isaac Lab, Gymnasium, NumPy NPZ datasets, pytest, existing dense QP backend.

## Global Constraints

- Generalized state is exactly 59 DOF: floating base 6 plus 53 physical joints; it is never exposed as an action.
- Active action is exactly 43 DOF in this order: M1 legs/wheels 16, platform 1, left Panda 7, right Panda 7, left O6 6, right O6 6.
- The ten O6 mimic joints remain passive physical DOFs and influence dynamics/Jacobians through the URDF mapping.
- Teacher and encoder run at exactly 25 Hz; body controller, safety projection, and physics run at exactly 200 Hz.
- Teacher horizon is exactly 25 nodes at 0.04 s, totaling 1.0 s.
- Latent action dimension is exactly 16 with groups `[0:4]`, `[4:10]`, `[10:14]`, and `[14:16]` defined by the approved design.
- All names ending in `_b` are expressed in `BASE_LINK`; no world-frame approximation may be assigned that suffix.
- Fixed-base phase enforces zero wheel speed and stationary wheel-ground contacts. Wheel rolling is outside this plan.
- Play accepts no training checkpoint argument and closes only after genuine `DONE`, safe `TERMINATED`, user window close, or `--max-steps`.
- Missing/incompatible normalization or model artifacts reject latent execution before applying any learned action.
- Run pure tests from `Go2Pvcnn` with `/home/xk/miniconda3/envs/go2/bin/python`, `PYTHONPATH="$PWD"`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Run Isaac commands from `Go2Pvcnn` with the existing M1 environment and `--num_envs 1` where that option is available.

## File Map

- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/frame_kinematics.py`: pure quaternion, pose, twist, force, and spatial-Jacobian transforms into `BASE_LINK`.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_dynamics.py`: validated 59-state/43-effort constrained forward-dynamics condensation.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py`: fingertip Jacobian extraction, mimic folding, and per-finger precontact closure.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/full_action_teacher.py`: 25 Hz, 25-node complete-action teacher.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_contracts.py`: 111-state feature packing, normalization metadata, model compatibility contract.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_model.py`: conditional encoder, trajectory decoder, and 200 Hz body controller.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/safety_projection.py`: hard 43-action projection and diagnostics.
- Create `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_runtime.py`: 25/200 Hz scheduling, valid-`z` lifetime, fallback, and safe termination.
- Modify `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py`: fingertip Jacobians plus full-dynamics snapshot contracts.
- Modify `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py`: real contact Jacobians and precontact references.
- Modify `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`: teacher baseline orchestration and explicit reset.
- Modify `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/state_machine.py`: per-finger precontact/contact dwell gates.
- Modify `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/whole_body_qp.py`: consume reduced coupled dynamics rather than zero base-arm coupling.
- Modify `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`: export stable public interfaces.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`: one adapter, common deterministic reset, teacher/latent mode selection, one atomic action write.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py`: fixed-base contact tolerances and canonical latent model path.
- Create `Go2Pvcnn/scripts/m1_dual_panda_o6_collect_teacher.py`: successful and failed trajectory collection.
- Create `Go2Pvcnn/scripts/m1_dual_panda_o6_train_latent.py`: deterministic split, normalization, training, evaluation, and artifact writing.
- Modify `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`: shared reset and formal teacher-versus-latent gates.
- Modify `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`: shared reset, lifecycle, and diagnostics.
- Add focused tests under `Go2Pvcnn/tests/` named in each task.

---

### Task 1: Audit and checkpoint the current integration baseline

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/whole_body_qp.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_object_mpc.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_runtime.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_whole_body_qp.py`
- Test: `Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py`
- Test: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`

**Interfaces:**
- Consumes: existing 43-action asset and approved design commit `cbe2544`.
- Produces: a reviewed baseline with inclusive/legacy PhysX Jacobian row handling, phase-aware force closure, preserved palm orientation targets, leg posture effort, layer diagnostics, and a Play entry point.

- [ ] **Step 1: Inventory the existing exploratory delta before editing**

Run:

```bash
git status --short
git diff --check
git diff -- Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination Go2Pvcnn/go2_pvcnn/tasks Go2Pvcnn/scripts Go2Pvcnn/tests
```

Expected: only the known bimanual files are changed; `git diff --check` prints nothing. Treat `Go2Pvcnn/tests/artifacts/m1_dual_panda_o6_progressive_200.json` as diagnostic output and leave it untracked until Task 11 regenerates formal evidence.

- [ ] **Step 2: Lock the discovered regressions into tests**

Ensure these exact assertions exist:

```python
def test_force_closure_is_not_required_before_grasp_loaded(request):
    approach = replace(request, phase=BimanualPhase.APPROACH)
    grasp = replace(request, phase=BimanualPhase.GRASP)
    assert subsolution_failure_reason(approach) is None
    assert subsolution_failure_reason(grasp) == "force_closure_margin"


def test_runtime_exposes_latest_attempted_fallback(runtime, snapshot):
    runtime.compute(snapshot)
    assert set(runtime.latest_solutions) == {
        "object", "arm", "left_hand", "right_hand", "wbc"
    }


def test_play_uses_wrapper_and_has_no_training_checkpoint():
    source = PLAY.read_text(encoding="utf-8")
    assert "M1DualPandaO6BimanualWrapper" in source
    assert "--checkpoint" not in source
    assert ".learn(" not in source
```

- [ ] **Step 3: Run the focused baseline suite**

Run:

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_whole_body_qp.py \
  tests/test_m1_dual_panda_o6_env_static.py \
  tests/test_m1_dual_panda_o6_entrypoints_static.py -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Record the known physical baseline without claiming mission success**

Run:

```bash
cd Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --seeds 42 --trials-per-seed 1 --steps 200 \
  --report tests/artifacts/m1_dual_panda_o6_progressive_200.json --headless
```

Expected: process exits normally, both palm Jacobian norms are finite and non-zero, and the report may remain unaccepted because coordinate, contact, and coupled-dynamics work belongs to later tasks.

- [ ] **Step 5: Commit the reviewed baseline, excluding the generated report**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/tests/test_m1_bimanual_object_mpc.py \
  Go2Pvcnn/tests/test_m1_bimanual_runtime.py \
  Go2Pvcnn/tests/test_m1_bimanual_whole_body_qp.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py
git commit -m "fix: stabilize bimanual integration baseline"
```

Expected: commit succeeds and the only remaining untracked path is the non-formal progressive report.

---

### Task 2: Convert every `_b` signal into the true moving `BASE_LINK` frame

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/frame_kinematics.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_frame_kinematics.py`

**Interfaces:**
- Consumes: world pose/twist/force/Jacobian tensors in Isaac quaternion `wxyz` convention.
- Produces: `pose_in_base(p_wb, q_wb, p_wt, q_wt) -> Tensor[6]`, `twist_in_base(p_wb, q_wb, v_wb, omega_wb, p_wt, v_wt, omega_wt) -> Tensor[6]`, `vectors_in_base(q_wb, vectors_w) -> Tensor[K,3]`, `spatial_jacobian_in_base(q_wb, jacobian_w) -> Tensor[6,N]`, and `physx_jacobian_body_row(body_id, body_count, jacobian_body_count) -> int`.

- [ ] **Step 1: Write invariance and moving-frame tests**

Create tests containing:

```python
def test_pose_is_invariant_to_common_world_transform():
    q_wb = axis_angle_to_quat(torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64))
    q_bt = axis_angle_to_quat(torch.tensor([-0.3, 0.1, 0.2], dtype=torch.float64))
    p_wb = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    p_bt = torch.tensor([0.4, 0.2, -0.1], dtype=torch.float64)
    p_wt = p_wb + quat_rotate(q_wb, p_bt)
    q_wt = quat_multiply(q_wb, q_bt)
    expected = torch.cat((p_bt, torch.tensor([-0.3, 0.1, 0.2], dtype=torch.float64)))
    assert torch.allclose(pose_in_base(p_wb, q_wb, p_wt, q_wt), expected, atol=1e-10)


def test_twist_removes_base_transport_velocity():
    p_wb = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    p_wt = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    v_base = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    v_target = v_base + torch.linalg.cross(omega, p_wt - p_wb)
    result = twist_in_base(
        p_wb, identity_quat(), v_base, omega, p_wt, v_target, omega
    )
    assert torch.allclose(result, torch.zeros(6, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("shape,expected", [((1, 60, 6, 59), 12), ((1, 59, 6, 59), 11)])
def test_physx_body_row_supports_both_layouts(shape, expected):
    assert physx_jacobian_body_row(12, 60, shape[1]) == expected
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_frame_kinematics.py -q
```

Expected: collection fails because `frame_kinematics` does not exist.

- [ ] **Step 3: Implement the pure transforms**

Use these equations in the new module:

```python
def pose_in_base(p_wb, q_wb, p_wt, q_wt):
    q_bw = quat_conjugate(normalize_quat(q_wb))
    return torch.cat((quat_rotate(q_bw, p_wt - p_wb), quat_to_rotvec(quat_multiply(q_bw, q_wt))))


def twist_in_base(p_wb, q_wb, v_wb, omega_wb, p_wt, v_wt, omega_wt):
    q_bw = quat_conjugate(normalize_quat(q_wb))
    transported = v_wt - v_wb - torch.linalg.cross(omega_wb, p_wt - p_wb)
    return torch.cat((quat_rotate(q_bw, transported), quat_rotate(q_bw, omega_wt - omega_wb)))


def vectors_in_base(q_wb, vectors_w):
    q_bw = quat_conjugate(normalize_quat(q_wb))
    return quat_rotate(q_bw.expand(vectors_w.shape[:-1] + (4,)), vectors_w)


def spatial_jacobian_in_base(q_wb, jacobian_w):
    rotation_bw = quat_to_matrix(quat_conjugate(normalize_quat(q_wb)))
    result = jacobian_w.clone()
    result[:3] = rotation_bw @ jacobian_w[:3]
    result[3:] = rotation_bw @ jacobian_w[3:]
    return result


def physx_jacobian_body_row(body_id, body_count, jacobian_body_count):
    if jacobian_body_count == body_count:
        return body_id
    if jacobian_body_count == body_count - 1 and body_id > 0:
        return body_id - 1
    raise ValueError("PhysX Jacobian rows do not match articulation body layout")
```

In the adapter, replace all world subtraction/orientation/twist/force/Jacobian code with these functions. Keep `base_state` as the documented Isaac world root state because it has no `_b` suffix.

- [ ] **Step 4: Run transform and adapter tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_frame_kinematics.py tests/test_m1_dual_panda_o6_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/frame_kinematics.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/tests/test_m1_bimanual_frame_kinematics.py
git commit -m "fix: express bimanual snapshots in base frame"
```

---

### Task 3: Move deterministic physics reset into the common Wrapper

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_wrapper_reset.py`
- Test: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`

**Interfaces:**
- Consumes: Isaac `env.reset(seed=seed)`, robot/box default state buffers, and `BimanualRuntime.reset()`.
- Produces: `M1DualPandaO6BimanualWrapper.reset(seed: int) -> BimanualSnapshot`, `startup_complete: bool`, and identical Probe/Play startup semantics.

- [ ] **Step 1: Write fake-environment reset ordering tests**

```python
def test_reset_writes_defaults_zeros_velocities_and_syncs_before_snapshot(fake_env):
    wrapper = M1DualPandaO6BimanualWrapper(fake_env, runtime=FakeRuntime())
    snapshot = wrapper.reset(seed=42)
    assert fake_env.calls == [
        "env.reset:42", "robot.root", "robot.joints", "box.root",
        "scene.reset", "sim.forward", "runtime.reset", "snapshot",
    ]
    assert wrapper.startup_complete is True
    assert torch.count_nonzero(snapshot.m1_qd) == 0


def test_step_requires_common_reset(fake_env):
    wrapper = M1DualPandaO6BimanualWrapper(fake_env, runtime=FakeRuntime())
    with pytest.raises(RuntimeError, match="reset"):
        wrapper.step()
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_wrapper_reset.py -q
```

Expected: tests fail because Wrapper has no reset contract.

- [ ] **Step 3: Implement reset as the only physical-reset path**

Add the following public behavior:

```python
def reset(self, seed: int) -> BimanualSnapshot:
    self.startup_complete = False
    self.env.reset(seed=int(seed))
    raw = self.env.unwrapped
    robot = raw.scene["robot"]
    box = raw.scene["box"]
    env_ids = torch.arange(raw.num_envs, device=raw.device)
    root = robot.data.default_root_state[env_ids].clone()
    root[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    joint_q = robot.data.default_joint_pos[env_ids].clone()
    joint_qd = torch.zeros_like(joint_q)
    robot.write_joint_state_to_sim(joint_q, joint_qd, env_ids=env_ids)
    box_root = box.data.default_root_state[env_ids].clone()
    box_root[:, 7:13] = 0.0
    box.write_root_state_to_sim(box_root, env_ids=env_ids)
    raw.scene.reset(env_ids)
    raw.sim.forward()
    self.adapter = M1DualPandaO6SnapshotAdapter(self.env)
    self.runtime.reset()
    self.last_command = None
    self.last_snapshot = self.adapter.snapshot()
    self._base_reference = root[0, :7].detach().cpu().clone()
    self.startup_complete = True
    return self.last_snapshot
```

`runtime.reset()` must clear counters, all cached solutions, previous command/snapshot, initial box pose, and mission state. Remove `_reset_physical_scene` from Probe and replace both entry-point startup sequences with `snapshot = wrapper.reset(seed=args.seed)`.

- [ ] **Step 4: Verify shared reset and three-step startup**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_wrapper_reset.py tests/test_m1_dual_panda_o6_entrypoints_static.py -q
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --seeds 42 --trials-per-seed 1 --steps 3 \
  --report tests/artifacts/m1_dual_panda_o6_reset_3.json --headless
```

Expected: pure tests pass; Probe reports zero initial velocities and neither `DONE` nor `TERMINATED` in the first three counted steps.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/tests/test_m1_bimanual_wrapper_reset.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py
git commit -m "fix: centralize deterministic bimanual reset"
```

---

### Task 4: Condense 59-DOF constrained dynamics into a 43-effort map

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_dynamics.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_reduced_dynamics.py`

**Interfaces:**
- Consumes: `FullDynamicsState(mass_matrix[59,59], bias[59], actuation_matrix[59,43], wheel_contact_jacobian[12,59], wheel_contact_bias[12])`.
- Produces: `ReducedDynamicsMap(qdd_offset[59], qdd_from_effort[59,43], contact_offset[12], contact_from_effort[12,43], condition_number: float)` and `condense_constrained_dynamics(state) -> ReducedDynamicsMap`.

- [ ] **Step 1: Write KKT, passive-coupling, and 43-only action tests**

```python
def test_condensed_map_satisfies_dynamics_and_stationary_contact():
    state = coupled_fixture()
    reduced = condense_constrained_dynamics(state)
    tau = torch.linspace(-1.0, 1.0, 43, dtype=torch.float64)
    qdd = reduced.qdd_offset + reduced.qdd_from_effort @ tau
    lam = reduced.contact_offset + reduced.contact_from_effort @ tau
    assert torch.linalg.vector_norm(state.mass_matrix @ qdd + state.bias - state.actuation_matrix @ tau - state.wheel_contact_jacobian.T @ lam) < 1e-9
    assert torch.linalg.vector_norm(state.wheel_contact_jacobian @ qdd + state.wheel_contact_bias) < 1e-9


def test_passive_mimic_inertia_changes_active_acceleration():
    coupled = condense_constrained_dynamics(coupled_fixture(mimic_coupling=0.3))
    uncoupled = condense_constrained_dynamics(coupled_fixture(mimic_coupling=0.0))
    assert not torch.allclose(coupled.qdd_from_effort, uncoupled.qdd_from_effort)


def test_contract_rejects_59_dimensional_action_matrix():
    with pytest.raises(ValueError, match="59, 43"):
        FullDynamicsState(
            mass_matrix=torch.eye(59, dtype=torch.float64),
            bias=torch.zeros(59, dtype=torch.float64),
            actuation_matrix=torch.eye(59, dtype=torch.float64),
            wheel_contact_jacobian=torch.zeros(12, 59, dtype=torch.float64),
            wheel_contact_bias=torch.zeros(12, dtype=torch.float64),
        )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_reduced_dynamics.py -q
```

Expected: import fails for the new contracts.

- [ ] **Step 3: Implement validated KKT condensation**

```python
GENERALIZED_DOF = 59
ACTIVE_DOF = 43
WHEEL_CONSTRAINT_DOF = 12


def condense_constrained_dynamics(state: FullDynamicsState) -> ReducedDynamicsMap:
    zero = torch.zeros((WHEEL_CONSTRAINT_DOF, WHEEL_CONSTRAINT_DOF), dtype=torch.float64)
    kkt = torch.cat((
        torch.cat((state.mass_matrix, -state.wheel_contact_jacobian.T), dim=1),
        torch.cat((state.wheel_contact_jacobian, zero), dim=1),
    ), dim=0)
    affine_rhs = torch.cat((-state.bias, -state.wheel_contact_bias))
    effort_rhs = torch.cat((state.actuation_matrix, torch.zeros(12, 43, dtype=torch.float64)), dim=0)
    solution = torch.linalg.solve(kkt, torch.cat((affine_rhs[:, None], effort_rhs), dim=1))
    return ReducedDynamicsMap(
        qdd_offset=solution[:59, 0],
        qdd_from_effort=solution[:59, 1:],
        contact_offset=solution[59:, 0],
        contact_from_effort=solution[59:, 1:],
        condition_number=float(torch.linalg.cond(kkt).item()),
    )
```

The adapter must read PhysX generalized mass, gravity plus Coriolis, construct `S` from the exact active joint IDs plus the 6 unactuated base rows, and construct four three-axis stationary wheel contact rows. Raise a diagnostic error if any shape differs; do not substitute identity dynamics.

- [ ] **Step 4: Run tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_reduced_dynamics.py tests/test_m1_dual_panda_o6_contracts.py -q
```

Expected: all tests pass, including residual below `1e-9`.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_dynamics.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/tests/test_m1_bimanual_reduced_dynamics.py
git commit -m "feat: add coupled 59 to 43 constrained dynamics"
```

---

### Task 5: Add real O6 fingertip Jacobians and contact-before-force closure

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/state_machine.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_state_machine.py`

**Interfaces:**
- Consumes: five full spatial fingertip Jacobians per side, active/mimic generalized column IDs, `O6_MIMIC_MAP`, hand state, phase, and contact mask.
- Produces: `fold_o6_fingertip_jacobians(full[5,6,59], active_ids[6], mimic_specs[5]) -> Tensor[15,6]` and `PrecontactHandController.reference(q[6], contact_mask[5], phase) -> (q_ref[6], qd_ref[6])`.

- [ ] **Step 1: Write mimic-fold and per-finger freeze tests**

```python
def test_mimic_columns_fold_into_six_active_columns():
    full = torch.zeros((5, 6, 59), dtype=torch.float64)
    full[:, :3, 40] = 1.0
    full[:, :3, 50] = 2.0
    folded = fold_o6_fingertip_jacobians(
        full, active_ids=(40, 41, 42, 43, 44, 45),
        mimic_specs=((50, 0, 1.86), (51, 2, 0.89), (52, 3, 0.89), (53, 4, 0.89), (54, 5, 0.89)),
    )
    assert folded.shape == (15, 6)
    assert torch.allclose(folded[:, 0].reshape(5, 3), full[:, :3, 40] + 1.86 * full[:, :3, 50])
    assert torch.linalg.vector_norm(folded) > 0.0


def test_preload_closes_uncontacted_fingers_and_freezes_contacted_finger():
    controller = PrecontactHandController()
    q = torch.zeros(6, dtype=torch.float64)
    mask = torch.tensor([False, True, False, False, False])
    q_ref, qd_ref = controller.reference(q, mask, BimanualPhase.PRELOAD)
    assert qd_ref[0] == 0.0
    assert qd_ref[2] == 0.0
    assert torch.all(qd_ref[3:] > 0.0)
    assert torch.all(q_ref >= q)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_o6_contact.py -q
```

Expected: import fails for `o6_contact_kinematics`.

- [ ] **Step 3: Implement real Jacobian folding and precontact control**

```python
FINGER_TO_ACTIVE = (0, 2, 3, 4, 5)


def fold_o6_fingertip_jacobians(full, active_ids, mimic_specs):
    linear = full[:, :3, list(active_ids)].clone()
    for mimic_id, master_column, multiplier in mimic_specs:
        linear[:, :, master_column] += multiplier * full[:, :3, mimic_id]
    return linear.reshape(15, 6)


class PrecontactHandController:
    def __init__(self, close_rate=0.35, open_q=None, preload_q=None):
        self.close_rate = float(close_rate)
        self.open_q = torch.tensor([0.10, 0.15, 0.10, 0.10, 0.10, 0.10], dtype=torch.float64) if open_q is None else open_q.clone()
        self.preload_q = torch.tensor([0.42, 0.55, 0.90, 0.90, 0.90, 0.90], dtype=torch.float64) if preload_q is None else preload_q.clone()

    def reference(self, q, contact_mask, phase):
        if phase is BimanualPhase.APPROACH:
            return self.open_q.clone(), torch.zeros(6, dtype=torch.float64)
        target = self.preload_q.clone()
        rate = torch.clamp((target - q) / 0.04, -self.close_rate, self.close_rate)
        for finger, active in enumerate(FINGER_TO_ACTIVE):
            if bool(contact_mask[finger]):
                target[active] = q[active]
                rate[active] = 0.0
        return target, rate
```

Add `fingertip_jacobian_b: Tensor[15,6]` to `SideHandState`; pass it to `HandMpcInput` instead of zeros. The mission may enter `PRELOAD` after both palms meet tolerance; it may enter `GRASP` only after bilateral contacts, normal-force minimum, positive force-closure/slip margin, and configured dwell all hold.

- [ ] **Step 4: Run hand and state-machine tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_o6_contact.py tests/test_m1_bimanual_hand_mpc.py tests/test_m1_bimanual_state_machine.py -q
```

Expected: all tests pass; contact Jacobian is `(15,6)`, finite, and non-zero for a non-singular fixture.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/state_machine.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py \
  Go2Pvcnn/tests/test_m1_bimanual_state_machine.py
git commit -m "feat: establish O6 contact with real fingertip Jacobians"
```

---

### Task 6: Build the 25 Hz full 43-action teacher

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/full_action_teacher.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/whole_body_qp.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_full_action_teacher.py`

**Interfaces:**
- Consumes: `BimanualSnapshot`, `ReducedDynamicsMap`, phase, object/dual-arm/dual-hand plans, active limits.
- Produces: `TeacherSolution(action_trajectory[25,43], box_trajectory_b[25,12], left_palm_trajectory_b[25,12], right_palm_trajectory_b[25,12], left_wrench_b[25,6], right_wrench_b[25,6], platform_trajectory[25,2], feasible, diagnostics)` and `FullActionTeacher.plan(TeacherInput) -> TeacherSolution`.

- [ ] **Step 1: Write horizon, wheel-lock, cadence, and reaction tests**

```python
def test_teacher_returns_complete_one_second_active_trajectory(teacher_input):
    solution = FullActionTeacher().plan(teacher_input)
    assert solution.action_trajectory.shape == (25, 43)
    assert solution.box_trajectory_b.shape == (25, 12)
    assert torch.all(solution.action_trajectory[:, 12:16] == 0.0)
    assert solution.diagnostics.dynamics_residual_max < 1e-6


def test_teacher_compensates_arm_reaction_through_base_channels(teacher_input):
    input_with_load = replace(teacher_input, right_palm_wrench_b=torch.tensor([0., 0., 20., 0., 0., 0.], dtype=torch.float64))
    nominal = FullActionTeacher().plan(teacher_input).action_trajectory[:, :12]
    loaded = FullActionTeacher().plan(input_with_load).action_trajectory[:, :12]
    assert not torch.allclose(nominal, loaded)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_full_action_teacher.py -q
```

Expected: import fails for `FullActionTeacher`.

- [ ] **Step 3: Implement receding-horizon teacher over the reduced map**

The per-node dense QP variable is `tau[43]`; use the existing planners to build `tau_nominal`, then minimize effort deviation plus task acceleration errors under hard action bounds and zero wheel channels:

```python
def _node_problem(sample, tau_nominal):
    A = sample.task_jacobian @ sample.dynamics.qdd_from_effort
    b = sample.task_acceleration_target - sample.task_jacobian @ sample.dynamics.qdd_offset - sample.task_bias
    W = torch.diag(sample.task_weights)
    hessian = 2.0 * (A.T @ W @ A + sample.effort_weight * torch.eye(43, dtype=torch.float64))
    gradient = -2.0 * (A.T @ W @ b + sample.effort_weight * tau_nominal)
    lower = -sample.effort_limits.clone()
    upper = sample.effort_limits.clone()
    lower[12:16] = 0.0
    upper[12:16] = 0.0
    return DenseQpProblem(hessian=hessian, gradient=gradient, lower_bounds=lower, upper_bounds=upper)
```

Roll the reduced forward dynamics for 25 nodes, warm-start each node from the prior teacher solution, and declare infeasible if QP, finite-value, stationary-contact, base-reference, or forward-dynamics residual checks fail. Preserve the last feasible plan only in runtime, not inside this stateless planner.

- [ ] **Step 4: Run teacher and existing MPC tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_full_action_teacher.py \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_dual_arm_mpc.py \
  tests/test_m1_bimanual_hand_mpc.py \
  tests/test_m1_bimanual_whole_body_qp.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/full_action_teacher.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/whole_body_qp.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_bimanual_full_action_teacher.py
git commit -m "feat: predict complete bimanual action trajectories"
```

---

### Task 7: Define and trainable-implement the z16 action representation

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_contracts.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_model.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_latent_model.py`

**Interfaces:**
- Consumes: normalized `state_features[111]`, `teacher_action[25,43]`, `teacher_task_features[25,50]`, `phase_scalar[1]`, and `last_effort[43]`.
- Produces: `LatentActionModel.encode(state, teacher_action, teacher_task) -> z[16]`, `decode_trajectory(state,z) -> action[25,43]`, `body_action(state,z,phase,last_effort) -> effort[43]`, and versioned `LatentArtifactMetadata`.

- [ ] **Step 1: Write exact dimensional, deterministic, and compatibility tests**

```python
def test_latent_model_shapes_and_group_contract():
    model = LatentActionModel()
    state = torch.zeros((2, 111), dtype=torch.float32)
    teacher_action = torch.zeros((2, 25, 43), dtype=torch.float32)
    teacher_task = torch.zeros((2, 25, 50), dtype=torch.float32)
    z = model.encode(state, teacher_action, teacher_task)
    assert z.shape == (2, 16)
    assert model.decode_trajectory(state, z).shape == (2, 25, 43)
    assert model.body_action(state, z, torch.zeros(2, 1), torch.zeros(2, 43)).shape == (2, 43)
    assert LATENT_GROUPS == {"support": (0, 4), "palms": (4, 10), "hands": (10, 14), "platform": (14, 16)}


def test_artifact_rejects_wrong_action_order():
    metadata = valid_metadata()
    with pytest.raises(ValueError, match="action_order"):
        metadata.validate_runtime(action_order=tuple(reversed(metadata.action_order)), state_dim=111)
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_latent_model.py -q
```

Expected: imports fail for latent modules.

- [ ] **Step 3: Implement the fixed architecture and metadata**

```python
STATE_DIM = 111
ACTION_DIM = 43
HORIZON = 25
TASK_FEATURE_DIM = 50
LATENT_DIM = 16
MODEL_FORMAT_VERSION = 1
LATENT_GROUPS = {"support": (0, 4), "palms": (4, 10), "hands": (10, 14), "platform": (14, 16)}


class LatentActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2436, 512), nn.SiLU(), nn.Linear(512, 128), nn.SiLU(), nn.Linear(128, 16), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(127, 256), nn.SiLU(), nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 1075))
        self.body = nn.Sequential(nn.Linear(171, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 43), nn.Tanh())

    def encode(self, state, teacher_action, teacher_task):
        flat = torch.cat((state, teacher_action.flatten(1), teacher_task.flatten(1)), dim=1)
        return self.encoder(flat)

    def decode_trajectory(self, state, z):
        return self.decoder(torch.cat((state, z), dim=1)).reshape(-1, 25, 43)

    def body_action(self, state, z, phase, last_effort):
        return self.body(torch.cat((state, z, phase, last_effort), dim=1))
```

`pack_state_features(snapshot)` must concatenate exactly: base root state 13, M1 `q/qd` 32, platform `q/qd` 2, dual arm `q/qd` 28, dual hand `q/qd` 24, box pose/twist 12 = 111. Metadata stores format version, dimensions, active joint order, feature order, normalization SHA-256, training seed, and dataset SHA-256.

- [ ] **Step 4: Run latent tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_latent_model.py -q
```

Expected: all tests pass and parameter initialization is deterministic under a fixed PyTorch seed.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_contracts.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_model.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_bimanual_latent_model.py
git commit -m "feat: define z16 bimanual action model"
```

---

### Task 8: Collect teacher data and train with a fixed split

**Files:**
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_collect_teacher.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_train_latent.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_latent_training.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`
- Modify: `Go2Pvcnn/.gitignore`

**Interfaces:**
- Consumes: teacher rollouts and Task 7 model.
- Produces: successful `teacher_success.npz`, failure-only `teacher_failures.npz`, `split.json`, `normalization.pt`, `latent_action_model.pt`, `metadata.json`, and `metrics.json` under a user-selected output directory.

- [ ] **Step 1: Write synthetic collection/training CLI tests**

```python
def test_split_is_deterministic_and_failures_are_excluded_from_reconstruction():
    samples = np.arange(100)
    first = deterministic_split(samples, seed=42)
    second = deterministic_split(samples, seed=42)
    assert first == second
    assert len(first.train) == 80 and len(first.validation) == 10 and len(first.test) == 10


def test_composite_loss_contains_all_design_terms():
    losses = composite_loss(synthetic_batch(), synthetic_outputs())
    assert set(losses) == {
        "action", "box_effect", "palm_effect", "contact", "base",
        "effort_smoothness", "latent_smoothness", "safety", "total",
    }
    assert torch.isfinite(losses["total"])
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_latent_training.py -q
```

Expected: script imports fail.

- [ ] **Step 3: Implement deterministic data and loss contracts**

Collector records one row per 25 Hz teacher update with `state[111]`, `teacher_action[25,43]`, `teacher_task[25,50]`, 200 Hz realized state/effect window, phase, feasibility, fallback reason, seed, and trial. The 50 task features are box pose/twist 12, left/right palm pose/twist 24, left/right wrench 12, and platform position/velocity 2. Successful rows go only to reconstruction data; failed rows go only to safety classification data.

Implement the exact weighted sum:

```python
weights = {
    "action": 1.0, "box_effect": 2.0, "palm_effect": 2.0,
    "contact": 1.0, "base": 2.0, "effort_smoothness": 0.1,
    "latent_smoothness": 0.05, "safety": 5.0,
}
total = sum(weights[name] * losses[name] for name in weights)
```

Use a fixed `80/10/10` split by `(seed, trial)` rather than individual rows, save normalization from training rows only, fix Python/NumPy/PyTorch seed, and save model only when validation is finite and the compatibility metadata validates.

- [ ] **Step 4: Run synthetic smoke training**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_latent_training.py tests/test_m1_dual_panda_o6_entrypoints_static.py -q
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_train_latent.py \
  --synthetic-smoke --epochs 2 --seed 42 --output-dir /tmp/m1_bimanual_latent_smoke
```

Expected: tests pass; command writes the seven listed dataset/model artifacts and reports finite train/validation/test losses.

- [ ] **Step 5: Commit source and ignore generated datasets/models**

```bash
git add Go2Pvcnn/scripts/m1_dual_panda_o6_collect_teacher.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_train_latent.py \
  Go2Pvcnn/tests/test_m1_bimanual_latent_training.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py \
  Go2Pvcnn/.gitignore
git commit -m "feat: add deterministic latent teacher training pipeline"
```

---

### Task 9: Add the 200 Hz hard safety projection

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/safety_projection.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_safety_projection.py`

**Interfaces:**
- Consumes: candidate effort `[43]`, snapshot, reduced dynamics, active limits, collision distances/Jacobian, base reference, and phase.
- Produces: `SafetyProjection.project(SafetyInput) -> SafetyResult(effort[43], feasible, active_constraints, fallback_reason, dynamics_residual)`.

- [ ] **Step 1: Write hard-limit and infeasibility tests**

```python
def test_projection_locks_wheels_and_respects_effort_limits(safety_input):
    candidate = 1e6 * torch.ones(43, dtype=torch.float64)
    result = SafetyProjection().project(replace(safety_input, candidate_effort=candidate))
    assert torch.all(result.effort.abs() <= safety_input.effort_limits + 1e-12)
    assert torch.all(result.effort[12:16] == 0.0)


def test_projection_rejects_nonfinite_and_stationary_contact_failure(safety_input):
    bad = safety_input.candidate_effort.clone()
    bad[0] = torch.nan
    assert SafetyProjection().project(replace(safety_input, candidate_effort=bad)).feasible is False
    impossible = replace(safety_input, base_position_tolerance_m=-1.0)
    result = SafetyProjection().project(impossible)
    assert result.fallback_reason == "base_reference_constraint_infeasible"
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_safety_projection.py -q
```

Expected: import fails for the safety module.

- [ ] **Step 3: Implement closest-safe-effort QP**

Minimize `||tau - candidate||²` with exact effort bounds, joint one-step position/velocity barriers through `qdd_offset + qdd_from_effort @ tau`, zero wheel channels, wheel-contact acceleration tolerance, base-reference acceleration bounds, collision velocity damper, and force closure only from `GRASP` through `LOWER_SAFE`. On non-finite input, solver failure, or residual failure, return the configured gravity/support safe effort and a non-empty reason; never return the unprojected candidate.

```python
def _wheel_lock_bounds(lower, upper):
    lower = lower.clone()
    upper = upper.clone()
    lower[12:16] = 0.0
    upper[12:16] = 0.0
    return lower, upper
```

- [ ] **Step 4: Run safety and WBC tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_safety_projection.py tests/test_m1_bimanual_whole_body_qp.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/safety_projection.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_bimanual_safety_projection.py
git commit -m "feat: project latent actions through hard safety constraints"
```

---

### Task 10: Schedule the 25 Hz teacher/z and 200 Hz latent controller

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_latent_runtime.py`

**Interfaces:**
- Consumes: Task 6 teacher, Task 7 artifact, Task 9 safety projector, 200 Hz snapshot stream.
- Produces: `LatentRuntime.compute(snapshot, dynamics) -> BimanualCommand`, `LatentRuntime.reset()`, exactly one teacher/encoder update every eight physics steps, and structured fallback diagnostics.

- [ ] **Step 1: Write cadence, expiry, artifact-rejection, and safety-override tests**

```python
def test_teacher_and_encoder_run_once_per_eight_body_steps(runtime, snapshots):
    for snapshot in snapshots[:17]:
        runtime.compute(snapshot, dynamics_fixture())
    assert runtime.counts == {"teacher": 3, "encoder": 3, "body": 17, "safety": 17}


def test_last_valid_z_expires_after_sixteen_physics_steps(runtime, snapshots):
    runtime.teacher.always_fail_after_first = True
    commands = [runtime.compute(s, dynamics_fixture()) for s in snapshots[:18]]
    assert commands[16].feasible is True
    assert commands[17].feasible is False
    assert "latent_expired" in commands[17].fallback_reasons


def test_missing_model_rejects_latent_runtime():
    with pytest.raises(FileNotFoundError, match="latent model"):
        LatentRuntime.from_artifact(Path("/missing/model"))
```

- [ ] **Step 2: Run tests and verify RED**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_bimanual_latent_runtime.py -q
```

Expected: import fails for `LatentRuntime`.

- [ ] **Step 3: Implement the scheduler and fallback state**

```python
class LatentRuntime:
    TEACHER_PERIOD = 8
    LATENT_TTL_STEPS = 16

    def compute(self, snapshot, dynamics):
        if self._step % self.TEACHER_PERIOD == 0:
            teacher = self.teacher.plan(self.teacher_input(snapshot, dynamics))
            if teacher.feasible:
                self._z = self.encode(snapshot, teacher)
                self._z_step = self._step
                self._last_teacher = teacher
        if self._z is None or self._step - self._z_step > self.LATENT_TTL_STEPS:
            return self._safe_command(snapshot.timestamp_ns, "latent_expired")
        phase = torch.tensor([(self._step % 8) / 8.0], dtype=torch.float32)
        candidate = self.model.body_action(self.state(snapshot), self._z, phase, self._last_effort)
        safe = self.safety.project(self.safety_input(snapshot, dynamics, candidate))
        self._step += 1
        return self.command_from_safety(snapshot.timestamp_ns, safe)
```

Before the first compute, validate model/normalization metadata and an OOD bound of `max(abs(normalized_state)) <= 10`. Reject non-finite `z`, `max(abs(z)) > 1.0 + 1e-6`, incompatible action order, stale `z`, consecutive teacher failure beyond TTL, or safety infeasibility. Log layer, reason, duration, and last safe action source.

- [ ] **Step 4: Run runtime tests**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_latent_runtime.py tests/test_m1_bimanual_runtime.py -q
```

Expected: all tests pass; counts are exactly `3/3/17/17` for 17 physics steps.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_runtime.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_bimanual_latent_runtime.py
git commit -m "feat: run low frequency teacher with z16 body control"
```

---

### Task 11: Integrate teacher/latent modes and pass ordered Isaac gates

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_acceptance.py`
- Create: `docs/superpowers/reports/2026-09-03-m1-bimanual-latent-acceptance.md`

**Interfaces:**
- Consumes: validated teacher, trained latent artifact, common reset, and formal Probe metrics.
- Produces: `mode="teacher" | "latent"` Wrapper execution, Play visualization command, 30-trial comparison report, and evidence for all ten approved acceptance gates.

- [ ] **Step 1: Write acceptance aggregation and lifecycle tests**

```python
def test_latent_acceptance_requires_teacher_safety_and_all_30_trials():
    teacher = [passing_trial(seed, i, mode="teacher") for seed in (42, 43, 44) for i in range(10)]
    latent = [passing_trial(seed, i, mode="latent") for seed in (42, 43, 44) for i in range(10)]
    result = aggregate_teacher_latent_acceptance(teacher, latent)
    assert result["accepted"] is True
    latent[0]["wheel_speed_max_rad_s"] = 0.01
    assert aggregate_teacher_latent_acceptance(teacher, latent)["accepted"] is False


def test_play_closes_only_on_terminal_or_window_close():
    source = PLAY.read_text(encoding="utf-8")
    assert 'phase in {"DONE", "TERMINATED"}' in source
    assert "simulation_app.is_running()" in source
    assert "wrapper.reset(" in source
    assert "env.reset(" not in source
```

- [ ] **Step 2: Run pure full suite before Isaac**

```bash
cd Go2Pvcnn
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests -q
```

Expected: complete pure/static suite passes.

- [ ] **Step 3: Run physical gates 1 through 7 in order using teacher mode**

```bash
cd Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --mode teacher --seeds 42 --trials-per-seed 1 --steps 3 \
  --report tests/artifacts/reset_gate.json --headless
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --mode teacher --seeds 42 --trials-per-seed 1 --steps 2000 --hold-only \
  --report tests/artifacts/fixed_base_2000.json --headless
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --mode teacher --seeds 42 --trials-per-seed 1 --steps 4000 \
  --report tests/artifacts/teacher_mission_1.json --headless
```

Expected in sequence: identical Probe/Play reset and no first-three-step false terminal; fixed base remains within configured position/orientation/wheel/contact bounds for 2000 steps; palms converge below `0.02 m`; bilateral O6 contact forms with non-zero real Jacobians and no penetration/limit failure; box lifts at least `0.10 m`, holds at least `3 s`, lowers, releases supported, and reaches `DONE`.

- [ ] **Step 4: Collect/train only after teacher gate passes**

```bash
cd Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_collect_teacher.py \
  --seeds 42 43 44 --trials-per-seed 10 --output-dir outputs/m1_bimanual_teacher --headless
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_train_latent.py \
  --dataset-dir outputs/m1_bimanual_teacher --epochs 200 --seed 42 \
  --output-dir outputs/m1_bimanual_latent
```

Expected: collector contains 30 complete trials and segregated failures; training writes a compatible artifact and finite held-out action/task-effect metrics with no recorded hard-safety violation.

- [ ] **Step 5: Run the paired 30-trial latent gate**

```bash
cd Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --mode teacher --seeds 42 43 44 --trials-per-seed 10 --steps 4000 \
  --report tests/artifacts/teacher_30.json --headless
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py \
  --mode latent --latent-artifact outputs/m1_bimanual_latent \
  --seeds 42 43 44 --trials-per-seed 10 --steps 4000 \
  --teacher-baseline tests/artifacts/teacher_30.json \
  --report tests/artifacts/latent_30.json --headless
```

Expected: all 30 paired latent trials satisfy the teacher's hard safety gates; report records action error, box/palm/contact effect error, success difference, teacher failure counts, latent expiry counts, projection interventions, and fallback duration.

- [ ] **Step 6: Verify GUI Play from deterministic reset**

Run teacher visualization first:

```bash
cd Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_play.py \
  --mode teacher --seed 42 --max-steps 4000 --diagnostics
```

Then run latent visualization with the canonical model path configured in the environment; Play itself still has no checkpoint/model CLI surface:

```bash
cd Go2Pvcnn
M1_BIMANUAL_LATENT_ARTIFACT="$PWD/outputs/m1_bimanual_latent" \
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_play.py \
  --mode latent --seed 42 --max-steps 4000 --diagnostics
```

Expected: the window stays open while the mission is active and closes automatically only at real `DONE` or safe `TERMINATED`; neither run exits after the reset synchronization period.

- [ ] **Step 7: Write the evidence report and commit final integration**

The report must list commands, git commit, asset/source SHA-256, Isaac version, model/dataset metadata, each acceptance gate result, and links to the four JSON reports. Then run:

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py \
  Go2Pvcnn/tests/test_m1_bimanual_acceptance.py \
  Go2Pvcnn/tests/artifacts/reset_gate.json \
  Go2Pvcnn/tests/artifacts/fixed_base_2000.json \
  Go2Pvcnn/tests/artifacts/teacher_mission_1.json \
  Go2Pvcnn/tests/artifacts/teacher_30.json \
  Go2Pvcnn/tests/artifacts/latent_30.json \
  docs/superpowers/reports/2026-09-03-m1-bimanual-latent-acceptance.md
git commit -m "feat: integrate and verify latent bimanual MPC"
```

Expected: commit succeeds only if all ten gates are evidenced. Do not mark the implementation complete if a physical or latent gate is absent or failed.

---

## Final Verification

- [ ] Run `git diff --check` and expect no output.
- [ ] Run the entire `Go2Pvcnn/tests` suite and retain the passing count.
- [ ] Confirm no action tensor with trailing dimension 59 is created anywhere under `m1_bimanual_coordination`.
- [ ] Confirm all `_b` assignments call Task 2 transforms or are constructed from already transformed values.
- [ ] Confirm Probe and Play contain no direct physical reset implementation.
- [ ] Confirm wheel channels `12:16` are zero in teacher output, body output after safety, and recorded execution.
- [ ] Confirm both left and right O6 contact Jacobians are `(15,6)`, finite, and non-zero during PRELOAD/GRASP.
- [ ] Confirm latent startup rejects missing model, metadata mismatch, normalization mismatch, non-finite `z`, out-of-range `z`, and out-of-distribution state.
- [ ] Confirm every fallback report contains layer, reason, duration, and last safe action source.
- [ ] Confirm rolling/sliding behavior, reinforcement learning, visual perception, randomized objects, and hardware control were not added.
