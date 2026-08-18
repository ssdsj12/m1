# M1 + Panda Online DAgger Student S1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and accept a deployable 100-observation, 10-frame, 23-action Student that imitates the accepted zero-clearance C1a WBC Teacher through Teacher pretraining and online DAgger.

**Architecture:** Keep C1a WBC as a privileged side-labeler, factor deployable mission/nominal commands out of it, and give each simulated environment isolated Teacher and history state. Train a GRU estimator/actor with supervised DAgger losses and strict versioned replay/checkpoint manifests; formal evaluation executes Student actions only.

**Tech Stack:** Python 3.11, PyTorch 2.7, pytest, Isaac Sim 5.1, Isaac Lab, PhysX, custom WBC/QP, NumPy/JSON, GPU0.

## Global Constraints

- This plan may start only after `2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md` passes and records the accepted USD SHA.
- Single-agent inline execution only; never stage `graphify-out/`.
- Use independent Gym ID, logs, dataset, manifest, checkpoint and Play paths; never overwrite A0/A1/C0/C1a.
- Exact contracts: observation `100`, history `10`, action `23 = 12 legs + 4 wheels + 7 Panda arm`; fingers remain outside the action.
- Student outputs normalized position/velocity residuals, never bare torque.
- S1 contains flat ground, the five C1a longitudinal phases and small Panda motion only: no random external wrench, turning, terrain, PPO, grasping or real hardware.
- Formal evaluation permits Teacher side labels but forbids Teacher action execution.
- Every environment owns independent settling center, mission state, QP warm-start, safety supervisor, Student history and replay metadata.

---

### Task 1: Define Student observation and action contracts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_contracts.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_contracts.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py`

**Interfaces:**
- Produces constants `STUDENT_OBSERVATION_DIM=100`, `STUDENT_HISTORY_LENGTH=10`, `STUDENT_ACTION_DIM=23`.
- Produces `StudentActionScaleCfg`, `StudentNominalCommand`, `StudentActionCommand`, `teacher_residual_label(...)`, and `apply_student_residual(...)`.

- [ ] **Step 1: Write failing dimension, reconstruction, limit, and dtype tests**

```python
def test_student_dimensions_are_frozen():
    assert STUDENT_OBSERVATION_DIM == 100
    assert STUDENT_HISTORY_LENGTH == 10
    assert STUDENT_ACTION_DIM == 23


def test_teacher_label_reconstructs_safe_teacher_targets():
    nominal = _nominal()
    q_des = nominal.position.clone(); q_des[:, :12] += 0.01; q_des[:, 16:] -= 0.02
    qd_des = nominal.velocity.clone(); qd_des[:, 12:16] += 0.5
    label = teacher_residual_label(q_des, qd_des, nominal, _scale())
    reconstructed = apply_student_residual(label, nominal, _scale(), previous_action=torch.zeros_like(label))
    torch.testing.assert_close(reconstructed.position, q_des)
    torch.testing.assert_close(reconstructed.velocity[:, 12:16], qd_des[:, 12:16])
```

Also test non-finite, wrong width/device/dtype, clipping diagnostics, and per-step slew limits.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_contracts.py
```

Expected: import failure.

- [ ] **Step 3: Implement frozen dataclasses and mappings**

```python
@dataclass(frozen=True)
class StudentActionScaleCfg:
    leg_position_rad: float = 0.25
    wheel_velocity_radps: float = 8.0
    arm_position_rad: float = 0.20
    leg_slew_per_step: float = 0.02
    wheel_slew_per_step: float = 0.50
    arm_slew_per_step: float = 0.01


@dataclass(frozen=True)
class StudentNominalCommand:
    position: torch.Tensor  # [E,23]
    velocity: torch.Tensor  # [E,23]


@dataclass(frozen=True)
class StudentActionCommand:
    normalized_action: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    saturated: torch.Tensor
```

Leg/arm labels use `(q_des-nominal.position)/scale`; wheel labels use `(qd_des-nominal.velocity)/scale`. Clamp normalized actions to `[-1,1]`, then apply group-specific slew. Validate finite float tensors on one device.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_contracts.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_contracts.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/__init__.py Go2Pvcnn/tests/test_m1_panda_student_contracts.py
git commit -m "feat: add M1 Panda Student S1 contracts"
```

### Task 2: Extract a deployable C1a mission and nominal command

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_mission.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_mission.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py`

**Interfaces:**
- Produces `StudentMissionSample(phase, shaped_vx, target_pose, target_twist, nominal)` and `StudentS1Mission.reset(...)`, `.sample(mission_step, root_xy_yaw, root_vxy_yawrate)`.
- Teacher accepts optional keyword `mission_sample`; absent preserves current C1a behavior.

- [ ] **Step 1: Write failing mission parity/reset tests**

Test five exact 800-step phases, wheel nominal `shaped_vx/0.095`, frozen settled leg/Panda nominals, deterministic seed, and independent reset of two missions. Add a Teacher recording test proving injected `mission_sample` drives the same phase/targets without a second schedule advance.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_mission.py tests/test_m1_panda_rolling_teacher.py -k mission
```

Expected: module/API missing.

- [ ] **Step 3: Implement the deployable mission**

Reuse `LongitudinalCommandSchedule`, `PlanarBodyFrameTrajectory`, and `wheel_speed_from_base_velocity`. At reset clone settled controlled `q`; freeze legs `[:12]` and arm `[16:23]`. At sample construct `[E,23]` nominal position/velocity with wheel velocity `[12:16]` and no Teacher/QP inputs.

Add `mission_sample: StudentMissionSample | None = None` to Teacher `step`; when supplied, skip its private schedule/trajectory sample and consume the injected values. Keep the old path byte-for-byte behavior through existing tests.

- [ ] **Step 4: Run regressions and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_mission.py tests/test_m1_panda_rolling_teacher.py tests/test_m1_panda_wbc_roll_play_static.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_mission.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/rolling_teacher.py Go2Pvcnn/tests/test_m1_panda_student_mission.py Go2Pvcnn/tests/test_m1_panda_rolling_teacher.py
git commit -m "feat: share deployable C1a Student mission"
```

### Task 3: Add history buffer and temporal Student model

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_model.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_model.py`

**Interfaces:**
- Produces `StudentHistoryBuffer(num_envs, device)`, `StudentNetworkCfg`, `StudentOutput`, and `M1PandaStudent.forward(history)`.

- [ ] **Step 1: Write failing history/model tests**

```python
def test_history_reset_only_clears_selected_environments():
    history = StudentHistoryBuffer(3, "cpu")
    history.append(torch.ones(3, 100))
    history.reset(torch.tensor([1]))
    assert torch.count_nonzero(history.value[0]) > 0
    assert torch.count_nonzero(history.value[1]) == 0


def test_student_network_outputs_finite_wrench_latent_and_action():
    model = M1PandaStudent(StudentNetworkCfg())
    out = model(torch.zeros(4, 10, 100))
    assert out.wrench_hat.shape == (4, 6)
    assert out.latent.shape == (4, 32)
    assert out.safety_logit.shape == (4, 1)
    assert out.raw_action.shape == (4, 23)
    assert out.action.shape == (4, 23)
    assert torch.all(out.action.abs() <= 1.0)
```

Also test wrong width/history, non-finite rejection, gradient flow and state-dict round trip.

- [ ] **Step 2: Run RED, implement, and run GREEN**

Use `nn.GRU(input_size=100, hidden_size=128, batch_first=True)`, a `Linear(128,38)` estimator head split `6+32`, a separate `Linear(128,1)` safety-boundary head, and actor `Linear(266,256)->ELU->Linear(256,128)->ELU->Linear(128,23)`. The actor input is exact `100 current observation + 128 GRU encoding + 6 W_hat + 32 latent = 266`; expose its logits as `raw_action` and `tanh(raw_action)` as `action`. History is an explicit `[E,10,100]` rolling tensor, not hidden global state.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_model.py
```

- [ ] **Step 3: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/student_model.py Go2Pvcnn/tests/test_m1_panda_student_model.py
git commit -m "feat: add temporal M1 Panda Student model"
```

### Task 4: Implement deterministic DAgger selection, replay, and losses

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/dagger.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_dataset.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_dagger.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_dataset.py`

**Interfaces:**
- Consumes `StudentOutput.action` and normalized Teacher residuals shaped `[E,23]`.
- Produces `DaggerStageCfg`, `DaggerSelection`, `select_dagger_action(...)`, `StudentLossCfg`, `student_dagger_loss(...)`, `DaggerRecord`, and `VersionedDaggerReplay`.

- [ ] **Step 1: Write failing action-selection tests**

```python
def test_safety_override_always_executes_teacher():
    cfg = DaggerStageCfg(name="mix-50", teacher_probability=0.5, seed=7)
    selected = select_dagger_action(
        student=torch.zeros(4, 23),
        teacher=torch.ones(4, 23),
        safe_to_execute_student=torch.tensor([True, False, True, False]),
        cfg=cfg,
        rollout_step=3,
    )
    assert torch.equal(selected.executed[~selected.safe_to_execute_student], torch.ones(2, 23))
    assert selected.teacher_executed[1] and selected.teacher_executed[3]


def test_fixed_seed_and_rollout_step_are_reproducible():
    args = dict(student=torch.zeros(64, 23), teacher=torch.ones(64, 23),
                safe_to_execute_student=torch.ones(64, dtype=torch.bool),
                cfg=DaggerStageCfg("mix", 0.25, 11), rollout_step=19)
    assert torch.equal(select_dagger_action(**args).executed,
                       select_dagger_action(**args).executed)
```

Also reject probabilities outside `[0,1]`, non-finite actions, unequal shapes and non-boolean masks. Assert `teacher_probability=1.0` executes only Teacher and `0.0` executes Student exactly where the safety mask permits.

- [ ] **Step 2: Write failing loss and replay tests**

```python
def test_dagger_loss_contains_all_s1_terms():
    losses = student_dagger_loss(
        output=_student_output(), target_action=torch.zeros(8, 23),
        target_wrench=torch.zeros(8, 6),
        target_safety=torch.tensor([0,0,0,0,1,1,1,1], dtype=torch.float32),
        hard_mask=torch.tensor([0,0,0,0,1,1,1,1], dtype=torch.bool),
        previous_action=torch.zeros(8, 23), cfg=StudentLossCfg(),
    )
    assert set(losses) == {"total", "action", "wrench", "safety", "slew", "saturation"}
    assert all(torch.isfinite(value) for value in losses.values())


def test_replay_keeps_environment_episode_identity_and_hard_samples(tmp_path):
    replay = VersionedDaggerReplay(capacity=4, hard_fraction=0.5, seed=5)
    replay.extend([_record(env_id=i % 2, episode_id=10 + i // 2, hard=i >= 2) for i in range(6)])
    assert len(replay) == 4
    assert sum(record.hard for record in replay.records) >= 2
    replay.save(tmp_path / "shard-00000.pt", _dataset_manifest())
    loaded = VersionedDaggerReplay.load(tmp_path / "shard-00000.pt", expected_manifest=_dataset_manifest())
    assert [(r.env_id, r.episode_id) for r in loaded.records] == [(r.env_id, r.episode_id) for r in replay.records]
```

Also test atomic save leaves neither `.tmp` nor partial manifest, sampling weights favor hard samples, and load rejects schema, asset SHA, Teacher commit, dimensions, control period, action scales or DAgger-stage mismatches.

- [ ] **Step 3: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_dagger.py tests/test_m1_panda_student_dataset.py
```

Expected: both new modules fail to import.

- [ ] **Step 4: Implement deterministic selection and normalized losses**

```python
@dataclass(frozen=True)
class DaggerStageCfg:
    name: str
    teacher_probability: float
    seed: int


@dataclass(frozen=True)
class StudentLossCfg:
    action: float = 1.0
    wrench: float = 0.25
    safety: float = 0.25
    slew: float = 0.05
    saturation: float = 0.05
    hard_sample_multiplier: float = 2.0
```

Derive the Bernoulli mask from a local `torch.Generator` seeded with `cfg.seed + rollout_step`; OR it with `~safe_to_execute_student`. Compute action MSE after the 12/4/7 channels have already been normalized by Task 1. Use wrench MSE, `output.safety_logit` BCE-with-logits against `target_safety`, adjacent-action smooth-L1 and `relu(abs(output.raw_action)-1)` saturation penalty; multiply per-sample losses by `hard_sample_multiplier` for takeover/degraded/contact-loss samples.

- [ ] **Step 5: Implement versioned replay with atomic shards**

```python
@dataclass(frozen=True)
class DaggerRecord:
    env_id: int
    episode_id: int
    step: int
    history: torch.Tensor       # [10,100]
    teacher_action: torch.Tensor # [23]
    executed_action: torch.Tensor # [23]
    wrench_target: torch.Tensor  # [6]
    safety_target: float
    hard: bool
    metadata: dict[str, object]
```

Maintain separate normal/hard reservoirs under a single capacity and reserve at least `ceil(capacity*hard_fraction)` slots for hard records. Save tensors to a temporary sibling file, `fsync`, `os.replace`, then write the canonical JSON manifest with sorted keys through the same atomic sequence.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_dagger.py tests/test_m1_panda_student_dataset.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/dagger.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_dataset.py Go2Pvcnn/tests/test_m1_panda_dagger.py Go2Pvcnn/tests/test_m1_panda_student_dataset.py
git commit -m "feat: add versioned M1 Panda DAgger replay"
```

### Task 5: Freeze Student checkpoint and manifest compatibility

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_checkpoint.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_checkpoint.py`

**Interfaces:**
- Produces `StudentCheckpointManifest`, `save_student_checkpoint(...)`, and `load_student_checkpoint(...)`.
- Consumes `StudentNetworkCfg`, `StudentActionScaleCfg`, optimizer state and the accepted asset/dataset identifiers.

- [ ] **Step 1: Write failing strict-load tests**

```python
def test_checkpoint_round_trip_is_strict(tmp_path):
    manifest = _manifest(asset_sha="zero-clearance-sha", teacher_commit="teacher-sha")
    path = tmp_path / "student-s1.pt"
    save_student_checkpoint(path, _model(), _optimizer(), manifest, global_step=123)
    loaded = load_student_checkpoint(path, _model(), _optimizer(), expected=manifest)
    assert loaded.global_step == 123


@pytest.mark.parametrize("field,value", [
    ("asset_sha", "old-10mm-sha"), ("teacher_commit", "wrong"),
    ("observation_dim", 99), ("history_length", 9), ("action_dim", 16),
    ("control_dt", 0.01), ("dagger_stage", "wrong-stage"),
])
def test_checkpoint_rejects_incompatible_manifest(tmp_path, field, value):
    path, accepted = _saved_checkpoint(tmp_path)
    rejected = dataclasses.replace(accepted, **{field: value})
    with pytest.raises(ValueError, match=field):
        load_student_checkpoint(path, _model(), expected=rejected)
```

Also test non-finite weights, missing optimizer state during resume, model-shape mismatch, action-scale mismatch and dataset hash mismatch.

- [ ] **Step 2: Run RED and implement schema version 1**

```python
@dataclass(frozen=True)
class StudentCheckpointManifest:
    schema_version: int
    asset_sha: str
    teacher_commit: str
    dataset_sha: str
    observation_dim: int
    history_length: int
    action_dim: int
    action_scales: dict[str, float]
    control_dt: float
    dagger_stage: str
    teacher_probability: float
    model_config: dict[str, int]
    loss_weights: dict[str, float]
```

Require schema `1`, dimensions `100/10/23`, `control_dt=0.005`, all finite model/optimizer tensors, and exact equality for every compatibility field. Write checkpoint and adjacent `.manifest.json` atomically.

- [ ] **Step 3: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_checkpoint.py
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_checkpoint.py Go2Pvcnn/tests/test_m1_panda_student_checkpoint.py
git commit -m "feat: version M1 Panda Student checkpoints"
```

### Task 6: Factor the runtime adapter and isolate batched Teachers

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/batched_rolling_teacher.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_batched_rolling_teacher.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_wbc_play.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py`

**Interfaces:**
- Produces `PhysxTeacherAdapter(env, env_index)` and `BatchedRollingTeacherBank(teachers, adapters)` with `reset(env_ids, state)` and `step(states, mission_samples)`.
- Preserves the current C0/C1a CLI, single-environment guard and numerical behavior.

- [ ] **Step 1: Write failing adapter extraction and isolation tests**

```python
def test_adapter_selects_only_requested_environment():
    adapter = PhysxTeacherAdapter(_fake_env(3), env_index=2)
    np.testing.assert_allclose(adapter.read_state().q, _expected_q_for_env(2))


def test_reset_and_warm_start_are_not_shared_between_teachers():
    bank = _bank(num_envs=2)
    bank.step(_states(2), _missions(2))
    qp0 = bank.teachers[0].qp_backend.warm_start.copy()
    bank.reset(torch.tensor([1]), _states(2))
    np.testing.assert_allclose(bank.teachers[0].qp_backend.warm_start, qp0)
    assert bank.teachers[1].safety.state.name == "TRACK"
```

Also assert two environments have distinct schedule, trajectory, motion distributor, QP backend, safety supervisor and settling-center objects; resetting env 1 must not change env 0 command phase, history or first-failure latch.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_batched_rolling_teacher.py tests/test_m1_panda_wbc_play_static.py tests/test_m1_panda_wbc_roll_play_static.py
```

Expected: runtime adapter and batched bank imports fail.

- [ ] **Step 3: Move the adapter without changing legacy behavior**

Move `PhysxTeacherAdapter` from `scripts/m1_panda_wbc_play.py` into `runtime_adapter.py`, add explicit `env_index`, and index every articulation/contact tensor with that value. Re-export/import it from the old script so existing imports remain valid. Do not remove the C1a `num_envs == 1` assertion.

- [ ] **Step 4: Implement one-Teacher-per-environment batching**

```python
class BatchedRollingTeacherBank:
    def __init__(self, teachers, adapters):
        if len(teachers) != len(adapters) or not teachers:
            raise ValueError("one adapter is required for every Teacher")
        self.teachers = list(teachers)
        self.adapters = list(adapters)

    def reset(self, env_ids, states):
        for env_id in env_ids.tolist():
            self.teachers[env_id].reset(states[env_id])

    def step(self, states, mission_samples):
        return [teacher.step(state, mission_sample=mission)
                for teacher, state, mission in zip(self.teachers, states, mission_samples, strict=True)]
```

Validate unique mutable subobjects at construction. Stack only the returned immutable command tensors; never vectorize by sharing controller state.

- [ ] **Step 5: Run GREEN, legacy regressions, and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_batched_rolling_teacher.py tests/test_m1_panda_wbc_play_static.py tests/test_m1_panda_wbc_roll_play_static.py tests/test_m1_panda_rolling_teacher.py
git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/runtime_adapter.py Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/batched_rolling_teacher.py Go2Pvcnn/scripts/m1_panda_wbc_play.py Go2Pvcnn/scripts/m1_panda_wbc_roll_play.py Go2Pvcnn/tests/test_m1_panda_batched_rolling_teacher.py Go2Pvcnn/tests/test_m1_panda_wbc_play_static.py Go2Pvcnn/tests/test_m1_panda_wbc_roll_play_static.py
git commit -m "refactor: isolate batched M1 Panda Teachers"
```

### Task 7: Register the 100-observation Student S1 environment

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_observation.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_s1_env_cfg.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_observation.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_s1_env_static.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/__init__.py`

**Interfaces:**
- Produces `StudentObservationParts`, `build_student_observation(parts)`, `M1PandaStudentS1EnvCfg`, and Gym ID `Isaac-M1-Panda-Student-S1-v0`.
- Consumes `m1_panda_mount_wrench_b`, the deployable mission sample, current articulation/contact state and previous normalized Student action.

- [ ] **Step 1: Write failing exact-layout tests**

```python
def test_observation_layout_is_exactly_100_float32_values():
    parts = _parts(batch=3)
    observation = build_student_observation(parts)
    assert observation.shape == (3, 100)
    assert observation.dtype == torch.float32
    torch.testing.assert_close(observation[:, 71:77], parts.mount_wrench_b)
    torch.testing.assert_close(observation[:, 77:100], parts.previous_action)
```

Assert every frozen slice independently:

```text
0:3 root linear velocity; 3:6 root angular velocity; 6:9 projected gravity
9:25 M1 q; 25:41 M1 qd; 41:48 Panda arm q; 48:55 Panda arm qd
55:61 end-effector pose error; 61:67 desired twist; 67:71 wheel contact
71:77 mount wrench; 77:100 previous action
```

Also reject wrong batch, width, device, dtype and non-finite components; verify wrench order `[Fx,Fy,Fz,Tx,Ty,Tz]` in `BASE_LINK` frame and sensor-on-robot sign.

- [ ] **Step 2: Write failing environment registration tests**

Assert the new Gym ID resolves to `M1PandaStudentS1EnvCfg`, physics timestep is `0.005`, decimation is `1`, flat terrain is selected, `num_envs` is configurable, finger position holding remains configured, and the A0/A1/C0/C1a IDs/config classes are unchanged.

- [ ] **Step 3: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_observation.py tests/test_m1_panda_student_s1_env_static.py
```

Expected: Student modules/Gym ID are missing.

- [ ] **Step 4: Implement observation assembly and S1 config**

```python
@dataclass(frozen=True)
class StudentObservationParts:
    root_linear_velocity_b: torch.Tensor
    root_angular_velocity_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    m1_joint_position: torch.Tensor
    m1_joint_velocity: torch.Tensor
    panda_arm_position: torch.Tensor
    panda_arm_velocity: torch.Tensor
    ee_pose_error_b: torch.Tensor
    desired_ee_twist_b: torch.Tensor
    wheel_contact: torch.Tensor
    mount_wrench_b: torch.Tensor
    previous_action: torch.Tensor
```

Concatenate in the frozen order, convert contact booleans to float32, and assert the final width equals `STUDENT_OBSERVATION_DIM`. Subclass the accepted rolling WBC scene/config only for assets, actuators and sensors; define a distinct Student environment config with configurable cloned environments and no random wrench event.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_observation.py tests/test_m1_panda_student_s1_env_static.py tests/test_m1_panda_wbc_env_static.py
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_observation.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_s1_env_cfg.py Go2Pvcnn/go2_pvcnn/tasks/__init__.py Go2Pvcnn/tests/test_m1_panda_student_observation.py Go2Pvcnn/tests/test_m1_panda_student_s1_env_static.py
git commit -m "feat: register M1 Panda Student S1 environment"
```

### Task 8: Add Teacher-only warm-start and online collection

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_wbc_collect.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_wbc_collect.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_dataset.py`

**Interfaces:**
- CLI requires `--accepted-asset-sha`, `--teacher-commit`, `--output-dir`, `--num-envs`, `--steps`, `--seed`, `--stage`, `--teacher-probability`, and optional `--student-checkpoint`.
- Produces immutable replay shards plus `dataset.manifest.json`; stage `teacher-warmup` requires probability `1.0` and no Student checkpoint.

- [ ] **Step 1: Write failing CLI and rollout tests**

```python
def test_teacher_warmup_rejects_student_execution():
    args = parse_args(["--stage", "teacher-warmup", "--teacher-probability", "0.5", *_required_args()])
    with pytest.raises(ValueError, match="teacher-warmup requires 1.0"):
        validate_args(args)


def test_rollout_record_contains_deployable_and_privileged_targets():
    record = build_record(_rollout_step())
    assert record.history.shape == (10, 100)
    assert record.teacher_action.shape == (23,)
    assert record.wrench_target.shape == (6,)
    assert {"qp_status", "safety_state", "takeover_reason", "nominal", "prelimit_action", "executed_action"} <= record.metadata.keys()
```

Also assert output directories must be fresh, old asset SHA is rejected before simulation starts, per-environment episode/step counters reset independently, and degraded/contact-loss/Student-error records set `hard=True`.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_wbc_collect.py tests/test_m1_panda_student_dataset.py
```

Expected: collection entry point is missing.

- [ ] **Step 3: Implement collection as an explicit staged state machine**

At each `0.005 s` step: sample one deployable mission per environment, build current observation, append its environment-owned history, infer Student when configured, compute one Teacher label per environment, apply deterministic DAgger selection and the balance-first safety override, execute the selected 23-channel command through the existing impedance layer, then append a complete `DaggerRecord`. On reset clear only that environment's mission, Teacher, history, previous action, slew state and episode counter.

The manifest records schema, accepted asset SHA, actual repository Teacher commit, seed, environment count, step count, stage, probability, control period, dimensions, action scales, loss weights, hard-sample counts, safety/takeover counts and SHA-256 for every shard.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_wbc_collect.py tests/test_m1_panda_student_dataset.py tests/test_m1_panda_batched_rolling_teacher.py
git add Go2Pvcnn/scripts/m1_panda_wbc_collect.py Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_dataset.py Go2Pvcnn/tests/test_m1_panda_wbc_collect.py
git commit -m "feat: collect M1 Panda online DAgger data"
```

- [ ] **Step 5: Prove GPU0 batch isolation before collecting training data**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TASK_ASSET_SHA="$(sha256sum assets/m1_panda/m1_panda.usd | cut -d' ' -f1)"
TASK_TEACHER_COMMIT="$(git log -1 --format=%H -- ../docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)"
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/isaaclab/bin/python scripts/m1_panda_wbc_collect.py --task Isaac-M1-Panda-Student-S1-v0 --headless --accepted-asset-sha "$TASK_ASSET_SHA" --teacher-commit "$TASK_TEACHER_COMMIT" --output-dir logs/m1_panda_student_s1/smoke-2env --num-envs 2 --steps 200 --seed 17 --stage teacher-warmup --teacher-probability 1.0
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/isaaclab/bin/python scripts/m1_panda_wbc_collect.py --task Isaac-M1-Panda-Student-S1-v0 --headless --accepted-asset-sha "$TASK_ASSET_SHA" --teacher-commit "$TASK_TEACHER_COMMIT" --output-dir logs/m1_panda_student_s1/smoke-8env --num-envs 8 --steps 400 --seed 17 --stage teacher-warmup --teacher-probability 1.0
```

Expected: exit `0`; manifest reports exactly 400 and 3200 records respectively, no cross-environment reset/state mutation, no NaN, and no unexpected reset.

### Task 9: Train preheat and explicit DAgger stages without PPO

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_train_cfg.py`
- Create: `Go2Pvcnn/scripts/m1_panda_student_train.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_train.py`

**Interfaces:**
- Produces `StudentTrainCfg`, `train_epoch(...)`, strict `--dataset-manifest` offline preheat, and strict `--resume` online-stage continuation.
- Consumes Tasks 3–5 model, losses, replay and checkpoint contracts; never imports RSL-RL/PPO runners.

- [ ] **Step 1: Write failing optimizer/resume/static tests**

```python
def test_one_epoch_reduces_supervised_loss():
    model = M1PandaStudent(StudentNetworkCfg())
    before = evaluate_loss(model, _learnable_batch())
    train_epoch(model, _loader(), torch.optim.Adam(model.parameters(), lr=3e-4), StudentLossCfg(), "cpu")
    after = evaluate_loss(model, _learnable_batch())
    assert after < before


def test_every_dagger_stage_requires_an_explicit_probability():
    with pytest.raises(ValueError, match="teacher_probability"):
        StudentTrainCfg(stage="dagger-1", teacher_probability=None)
```

Static tests parse imports and reject `rsl_rl`, `OnPolicyRunner`, PPO loss and privileged critic. Resume tests reject mismatched asset/dataset/Teacher/scales/dimensions/control period/stage and verify optimizer/global-step continuity.

- [ ] **Step 2: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_train.py tests/test_m1_panda_student_checkpoint.py
```

Expected: training module/script are missing.

- [ ] **Step 3: Implement the supervised trainer**

```python
@dataclass(frozen=True)
class StudentTrainCfg:
    stage: str
    teacher_probability: float | None
    learning_rate: float = 3e-4
    batch_size: int = 512
    epochs: int = 20
    seed: int = 17
    gradient_clip_norm: float = 1.0
```

Seed Python/NumPy/PyTorch, load and validate all shards before constructing Adam, sample hard records using their replay weights, report every named loss plus action/wrench validation error, clip gradients, reject non-finite gradients, and atomically save `best.pt` and `last.pt`. A stage advances only after its separately collected validation manifest passes the current C1a safety gate; the script never invents or automatically lowers Teacher probability.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_train.py tests/test_m1_panda_student_checkpoint.py tests/test_m1_panda_student_model.py tests/test_m1_panda_dagger.py
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_train_cfg.py Go2Pvcnn/scripts/m1_panda_student_train.py Go2Pvcnn/tests/test_m1_panda_student_train.py
git commit -m "feat: train M1 Panda DAgger Student S1"
```

- [ ] **Step 5: Run Teacher preheat and stop at its evidence gate**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
TASK_ASSET_SHA="$(sha256sum assets/m1_panda/m1_panda.usd | cut -d' ' -f1)"
TASK_TEACHER_COMMIT="$(git log -1 --format=%H -- ../docs/superpowers/runbooks/2026-08-18-m1-panda-zero-clearance-teacher-rebaseline.md)"
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/isaaclab/bin/python scripts/m1_panda_wbc_collect.py --task Isaac-M1-Panda-Student-S1-v0 --headless --accepted-asset-sha "$TASK_ASSET_SHA" --teacher-commit "$TASK_TEACHER_COMMIT" --output-dir logs/m1_panda_student_s1/teacher-warmup-seed17 --num-envs 64 --steps 4000 --seed 17 --stage teacher-warmup --teacher-probability 1.0
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_student_train.py --dataset-manifest logs/m1_panda_student_s1/teacher-warmup-seed17/dataset.manifest.json --output-dir logs/m1_panda_student_s1/preheat-seed17 --stage preheat --teacher-probability 1.0 --epochs 20 --seed 17
```

Expected: collection contains `64*4000=256000` records; training exits `0`, all validation losses are finite, and checkpoint manifest exactly matches the dataset. Record measured validation metrics before choosing the first mixed-stage probability.

- [ ] **Step 6: Run measured DAgger stages one at a time**

For each approved stage, use one explicit measured probability `P` and a fresh directory; never reuse a directory or decrease `P` automatically:

```bash
TASK_STAGE=dagger-1
TASK_PREVIOUS_STAGE=preheat-seed17
TASK_TEACHER_PROBABILITY=0.50
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/isaaclab/bin/python scripts/m1_panda_wbc_collect.py --task Isaac-M1-Panda-Student-S1-v0 --headless --accepted-asset-sha "$TASK_ASSET_SHA" --teacher-commit "$TASK_TEACHER_COMMIT" --output-dir "logs/m1_panda_student_s1/${TASK_STAGE}-seed17" --num-envs 64 --steps 4000 --seed 17 --stage "$TASK_STAGE" --teacher-probability "$TASK_TEACHER_PROBABILITY" --student-checkpoint "logs/m1_panda_student_s1/${TASK_PREVIOUS_STAGE}/best.pt"
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_student_train.py --dataset-manifest "logs/m1_panda_student_s1/${TASK_STAGE}-seed17/dataset.manifest.json" --output-dir "logs/m1_panda_student_s1/${TASK_STAGE}-train-seed17" --stage "$TASK_STAGE" --teacher-probability "$TASK_TEACHER_PROBABILITY" --resume "logs/m1_panda_student_s1/${TASK_PREVIOUS_STAGE}/best.pt" --epochs 20 --seed 17
```

The first measured trial uses `0.50`; change `TASK_STAGE`, `TASK_PREVIOUS_STAGE` and `TASK_TEACHER_PROBABILITY` only in a reviewed evidence record after the preceding stage passes. Expected for each stage: the validation gate passes before approving another lower probability; otherwise preserve its checkpoint/evidence, diagnose at the same stage and do not expand the curriculum.

### Task 10: Add strict Student-only Play and three-seed evaluation

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_evaluation.py`
- Create: `Go2Pvcnn/scripts/m1_panda_student_play.py`
- Create: `Go2Pvcnn/scripts/m1_panda_student_eval.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_evaluation.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_play.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_student_eval.py`

**Interfaces:**
- Produces `StudentEvaluationAccumulator`, `validate_student_only_summary(...)`, a visual/headless Play entry and an exact seeds `42,43,44` evaluation orchestrator.
- Student-only mode executes only `apply_student_residual(model(...).action, ...)`; `--teacher-labels` may compute comparison labels but cannot enter the execution selector.

- [ ] **Step 1: Write failing Student-only authority tests**

```python
def test_student_only_rejects_any_teacher_executed_action():
    summary = _passing_summary()
    summary["teacher_execution_count"] = 1
    with pytest.raises(ValueError, match="teacher_execution_count"):
        validate_student_only_summary(summary, teacher_success_rate=0.98)


def test_acceptance_requires_95_percent_completion_and_teacher_parity():
    summary = _passing_summary(completion_rate=0.949)
    with pytest.raises(ValueError, match="completion_rate"):
        validate_student_only_summary(summary, teacher_success_rate=0.98)
    summary = _passing_summary(success_rate=0.930)
    with pytest.raises(ValueError, match="Teacher"):
        validate_student_only_summary(summary, teacher_success_rate=0.98)
```

Validate all inherited C1a hard gates: five phase lengths, speed RMSE, stop time, displacement sign, four-wheel contact, rolling residual, lateral slip, roll/pitch, end-effector error, QP feasibility, action slew, joint limits, body contact, finite values, target continuity and unexpected resets. Require zero `HOLD`, `RETRACT`, `TERMINATE` in successful episodes and record six-axis `W_hat` error, 23-axis action error and training takeover rate without using them to waive a hard gate.

- [ ] **Step 2: Write failing CLI/static tests**

Parse `m1_panda_student_play.py` and assert default target motion is enabled, checkpoint is required, `--teacher-labels` defaults false, Student-only execution contains no `select_dagger_action`, and summary records `teacher_execution_count=0`. Assert `m1_panda_student_eval.py` accepts exactly seeds `42,43,44`, at least 64 environments, exactly 4000 steps and a fresh output directory.

- [ ] **Step 3: Run RED**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_evaluation.py tests/test_m1_panda_student_play.py tests/test_m1_panda_student_eval.py
```

Expected: evaluation module and both scripts are missing.

- [ ] **Step 4: Implement Student-only Play**

Load the strict checkpoint before launching simulation; verify its asset SHA against the current USD, allocate one mission/history/previous-action state per environment, clear each on its reset, infer the Student and execute its limited residual directly. If `--teacher-labels` is set, run isolated Teachers after the Student command has already been fixed for that step and use their result only for error metrics.

Reuse C1a metric formulas and thresholds from `m1_panda_wbc_roll_play.py`; add episode completion/success, Student safety states, teacher execution count, estimator/action errors, checkpoint/dataset/asset SHA and first failure root cause to the atomic summary JSON.

- [ ] **Step 5: Implement strict aggregation**

```python
EXPECTED_STUDENT_SEEDS = (42, 43, 44)
MIN_EVALUATION_ENVS = 64
EVALUATION_STEPS = 4000
MIN_COMPLETION_RATE = 0.95
MIN_TEACHER_RELATIVE_SUCCESS = 0.95
```

The evaluation script launches one child per seed, validates each row, aggregates episode counts rather than averaging rates, compares aggregate Student success with the accepted zero-clearance Teacher summary, and writes `ranking.json` only when all three children completed. A failed child or hard gate returns nonzero and preserves completed rows.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_evaluation.py tests/test_m1_panda_student_play.py tests/test_m1_panda_student_eval.py tests/test_m1_panda_wbc_roll_play_static.py
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_student_evaluation.py Go2Pvcnn/scripts/m1_panda_student_play.py Go2Pvcnn/scripts/m1_panda_student_eval.py Go2Pvcnn/tests/test_m1_panda_student_evaluation.py Go2Pvcnn/tests/test_m1_panda_student_play.py Go2Pvcnn/tests/test_m1_panda_student_eval.py
git commit -m "feat: evaluate M1 Panda Student-only S1"
```

### Task 11: Run the complete static, GPU smoke, and Student-only gates

**Files:** Produce immutable artifacts under `Go2Pvcnn/logs/m1_panda_student_s1/`; implementation changes require a reproduced RED test in the owning task.

- [ ] **Step 1: Run all Student and legacy coordination tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_panda_student_contracts.py tests/test_m1_panda_student_mission.py tests/test_m1_panda_student_model.py tests/test_m1_panda_dagger.py tests/test_m1_panda_student_dataset.py tests/test_m1_panda_student_checkpoint.py tests/test_m1_panda_batched_rolling_teacher.py tests/test_m1_panda_student_observation.py tests/test_m1_panda_student_s1_env_static.py tests/test_m1_panda_wbc_collect.py tests/test_m1_panda_student_train.py tests/test_m1_panda_student_evaluation.py tests/test_m1_panda_student_play.py tests/test_m1_panda_student_eval.py tests/test_m1_panda_rolling_teacher.py tests/test_m1_panda_rolling_wbc.py tests/test_m1_panda_wbc_safety.py tests/test_m1_panda_wbc_roll_play_static.py
```

Expected: all selected tests pass with no skipped Student contract test.

- [ ] **Step 2: Run 2- and 8-environment mixed GPU smoke**

Use the Task 8 collection command with the accepted SHA/Teacher variables, `--student-checkpoint` pointing to the current candidate, stage `dagger-smoke`, probability `0.50`, then `num_envs=2, steps=200` and `num_envs=8, steps=400` in fresh directories. Expected: no shared reset/warm-start/safety state, finite observations/actions/labels and no unexpected resets.

- [ ] **Step 3: Run visual Student Play with default Panda motion**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/isaaclab/bin/python scripts/m1_panda_student_play.py --task Isaac-M1-Panda-Student-S1-v0 --device cuda:0 --checkpoint logs/m1_panda_student_s1/dagger-1-train-seed17/best.pt --num-envs 1 --steps 4000 --seed 42 --summary-json logs/m1_panda_student_s1/student-visual-seed42.json
```

Expected: Panda follows a small continuous bent-arm target, M1 rolls through all five phases, the mount remains attached/quiet, and `teacher_execution_count=0`. Reject a visibly straight locked arm, penetration, jitter, contact loss or unsafe degradation.

- [ ] **Step 4: Run formal 3-seed, 64-environment Student-only evaluation**

```bash
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/isaaclab/bin/python scripts/m1_panda_student_eval.py --task Isaac-M1-Panda-Student-S1-v0 --device cuda:0 --checkpoint logs/m1_panda_student_s1/dagger-1-train-seed17/best.pt --teacher-summary /tmp/m1_panda_zero_clearance_c1a_combined.json --output-dir logs/m1_panda_student_s1/formal-dagger-1 --num-envs 64 --steps 4000 --seed 42 --seed 43 --seed 44 --teacher-labels
```

Expected: exit `0`; all three rows have `teacher_execution_count=0`; aggregate completion is at least `0.95`, aggregate Student success is at least `0.95 * Teacher success`, every inherited C1a hard gate passes, and successful episodes contain zero `HOLD/RETRACT/TERMINATE`.

- [ ] **Step 5: Stop or promote based only on evidence**

If Step 4 fails, retain its artifacts and repeat the same DAgger stage after a test-backed diagnosis; do not lower Teacher probability or add task difficulty. If it passes, freeze the exact checkpoint SHA, dataset SHA, asset SHA, Teacher authority commit, action scales, loss weights and evaluation artifact hashes.

### Task 12: Publish the S1 runbook and evidence boundary

**Files:**
- Create: `docs/superpowers/runbooks/2026-08-18-m1-panda-dagger-student-s1.md`
- Create: `notes/log/2026-08-18-m1-panda-dagger-student-s1.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes exact Task 11 commands, hashes, stdout/stderr, metrics and visual decision.
- Produces the approved train/Play commands and the explicit boundary for any later random-wrench or grasping design.

- [ ] **Step 1: Write the operational runbook**

Include literal commands for Teacher warmup, each actually executed DAgger stage, resume, single-environment visual Play and formal evaluation. Document observation slices, 10-frame reset behavior, 23-channel physical scales, expected files, manifest rejection messages, safety stop conditions and GPU0 selection.

- [ ] **Step 2: Record evidence without broadening scope**

Record the accepted asset/checkpoint/dataset SHA-256 values, Teacher authority commit, seeds/envs/steps, per-seed and aggregate metrics, Teacher execution count, safety transitions, takeovers, estimator/action errors and visual mount decision. State that S1 does not authorize random wrench, turning, terrain, PPO, grasping, fingers or real hardware; T400.3 remains the real mechanical gate.

- [ ] **Step 3: Update T400 only if formal S1 passed**

Mark the zero-clearance/Student S1 item accepted only when Task 11 Step 4 exits `0`. Otherwise record the current failing gate and leave S1 open.

- [ ] **Step 4: Verify and commit documentation**

```bash
git diff --check
rg -n "100|10-frame|23|teacher_execution_count|42|43|44|64|4000|T400\.3|not authorize" docs/superpowers/runbooks/2026-08-18-m1-panda-dagger-student-s1.md notes/log/2026-08-18-m1-panda-dagger-student-s1.md notes/todo/T400-m1-panda-force-aware-teacher-student.md
git add docs/superpowers/runbooks/2026-08-18-m1-panda-dagger-student-s1.md notes/log/2026-08-18-m1-panda-dagger-student-s1.md notes/log/index.md notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: record M1 Panda Student S1 evidence"
```

## Final Verification

- [ ] Re-run Task 11 Step 1 and verify all Student/legacy tests pass at the final commit.
- [ ] Run `sha256sum -c assets/m1_panda/generated_files.sha256` and strict-load the accepted checkpoint/dataset against the current asset and Teacher authority.
- [ ] Revalidate all three formal evaluation rows and their aggregate without rerunning Teacher actions.
- [ ] Run `git diff --check` and `git status --short`; only the pre-existing `graphify-out/cache/last_query_stamp` may remain.
