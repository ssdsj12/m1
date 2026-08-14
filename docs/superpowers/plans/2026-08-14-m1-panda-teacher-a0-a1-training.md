# M1 + Panda Teacher A0/A1 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The end user explicitly requires single-agent execution, so use `superpowers:executing-plans`; do not dispatch subagents.

**Goal:** Build and verify an executable M1 + Panda privileged-Teacher balance-training chain in which A0 learns bounded residual stabilization from zero base actions and A1 freezes an A0 60→16 checkpoint while learning a second bounded residual under stronger six-dimensional disturbances.

**Architecture:** Keep the existing combined articulation, 60-dimensional observation, mount-wrench adapter, and 16-channel residual composer as foundations. Add a pure-PyTorch disturbance scheduler and checkpoint contract, a small dedicated environment cfg, and a dedicated VecEnv wrapper; keep A0/A1 selection and manifest handling in a separate training entrypoint. A1 owns two composer instances and an immutable frozen ActorCritic; the normal RSL-RL runner owns only the trainable A1 policy.

**Tech Stack:** Python 3.11, PyTorch, Isaac Sim 5 / Isaac Lab 2.1, Gymnasium, vendored RSL-RL PPO, pytest, JSON manifests, SHA-256.

## Global Constraints

- Work only in `/home/xk/coding/M1`; this directory is not a Git worktree, so every commit step records `Git Ref: unavailable` and must not initialize Git.
- Use one agent only. Do not dispatch subagents.
- Preserve the existing combined local USD, one articulation, 25 DOF, and exactly 16 M1 actions; Panda joints remain outside the action manager.
- Policy and critic observations are exactly 60 dimensions: `3 + 3 + 16 + 16 + 16 + 6`.
- Mount wrench order is exactly `[Fx, Fy, Fz, Mx, My, Mz]`, expressed at the M1 `BASE_LINK` origin in `BASE_LINK` axes.
- A0 uses a zero base action. A1 accepts only an A0 checkpoint produced by this 60→16 training chain; never silently load the existing 572/586-dimensional PVCNN checkpoints.
- A0 full limits are `±10 N` per force axis and `±2 Nm` per torque axis, held for `1.0–2.0 s`.
- A1 full limits are `±20 N` and `±5 Nm`, with segments lasting `0.25–1.0 s`.
- Use curriculum scale `0.25 → 1.0`; default full-scale horizons are 50,000 wrapper steps for A0 and 75,000 for A1.
- A1 mode probabilities are explicit configuration defaults: hold `0.50`, linear ramp `0.30`, pulse `0.20`; pulse is nonzero for the first `0.20` of its sampled segment and zero for the remainder.
- Composer defaults stay `0.05 rad`, `1.0 rad/s`, `0.01 rad/step`, and `0.2 rad/s/step`, with existing action scales `0.25` and `8.0`.
- Frozen A0 inference uses `eval()` and `torch.no_grad()`; its parameters never enter the A1 optimizer and its SHA-256 must match before and after A1 training.
- A missing/incompatible manifest, stage, checkpoint shape, base hash, non-finite tensor, ambiguous body lookup, or stale reset state is a hard error before continued training.
- CPU one-environment A0→A1 smoke is required. Long GPU convergence is excluded because installed PyTorch supports at most `sm_90` while the RTX 5070 is `sm_120`.
- Pure/static tests use the exact file list shown in each task and this command form:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_panda_teacher_disturbance.py -q
```

- Real Isaac Sim commands use `/home/xk/miniconda3/envs/loco/bin/python` with `OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1`.

---

## File Map

- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py`: pure disturbance configuration, per-environment scheduler, quaternion wrench conversion, and finite/index validation.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py`: atomic manifest IO, file/module hashing, stage/shape/base-hash validation, and frozen ActorCritic construction.
- `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_teacher_rewards.py`: lightweight reusable balance reward helpers that consume M1 selectors and wrapper-published trainable residual state without importing the repository's large reward module.
- `Go2Pvcnn/go2_pvcnn/mdp/__init__.py`: export the Teacher-specific reward helpers through the normal `mdp.*` cfg namespace.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py`: shared A0/A1 scene, rewards, termination, exact observation/action contract, and explicit stage/disturbance fields.
- `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`: lazy registration of exactly two new Gym IDs.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py`: strict 60/16 VecEnv boundary, disturbance application, A0/A1 composition, done/reset clearing, and frozen hash exposure.
- `Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py`: non-mutating factory for the Teacher PPO dictionary.
- `Go2Pvcnn/agent/__init__.py`: export the new config factory.
- `Go2Pvcnn/scripts/m1_panda_teacher_train.py`: CLI, environment construction, checkpoint/manifest contract, RSL-RL execution, and final frozen-hash audit.
- `Go2Pvcnn/scripts/m1_panda_teacher_smoke.py`: deterministic one-process acceptance driver that runs A0, finds its checkpoint, runs A1, and validates both manifests/checkpoints.
- `Go2Pvcnn/tests/test_m1_panda_teacher_disturbance.py`: scheduler and transform behavior.
- `Go2Pvcnn/tests/test_m1_panda_teacher_checkpoint.py`: checkpoint and manifest behavior.
- `Go2Pvcnn/tests/test_m1_panda_teacher_env_cfg_static.py`: cfg, reward, observation, action, and Gym registration wiring.
- `Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py`: fake-env A0/A1 data flow, done reset, finite failures, and frozen immutability.
- `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py`: parser, stage selection, runner and manifest wiring, smoke driver commands.
- `docs/superpowers/specs/2026-08-14-m1-panda-teacher-a0-a1-training-design.md`: approved source of truth; only correct discovered ambiguity, do not expand scope.
- `docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md`: formal, resume, CPU smoke, artifact, and CUDA-limit instructions.
- `notes/todo.md`, `notes/todo/T400-m1-panda-force-aware-teacher-student.md`, `notes/log/index.md`, and one new log per verification pass: required repository memory.

---

### Task 1: Pure Per-Environment Six-Dimensional Disturbance Scheduler

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_disturbance.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_wrench_probe.py`
- Test: `Go2Pvcnn/tests/test_m1_panda_wrench_probe_static.py`

**Interfaces:**
- Consumes: `(num_envs: int, device: torch.device, step_dt: float, seed: int)` and a frozen `M1PandaDisturbanceCfg`.
- Produces: `M1PandaDisturbanceScheduler.advance() -> torch.Tensor[num_envs, 6]`, `.reset(env_ids: torch.Tensor | Sequence[int] | None = None) -> None`, `.current_wrench_b`, `.curriculum_scale`, `base_wrench_to_body_local(force_b, torque_b, base_quat_w, body_quat_w)`, and `clear_external_wrench(robot)`.
- Later tasks rely on `stage_disturbance_cfg("A0" | "A1")` returning exact approved defaults.

- [x] **Step 1: Write failing configuration and validation tests**

Add tests that assert the exact A0/A1 defaults and reject invalid ranges/probabilities:

```python
def test_stage_defaults_are_exact():
    a0 = teacher.stage_disturbance_cfg("A0")
    assert a0.force_limit_n == (10.0, 10.0, 10.0)
    assert a0.torque_limit_nm == (2.0, 2.0, 2.0)
    assert (a0.hold_time_min_s, a0.hold_time_max_s) == (1.0, 2.0)
    assert a0.mode_probabilities == (1.0, 0.0, 0.0)
    assert a0.curriculum_steps == 50_000

    a1 = teacher.stage_disturbance_cfg("A1")
    assert a1.force_limit_n == (20.0, 20.0, 20.0)
    assert a1.torque_limit_nm == (5.0, 5.0, 5.0)
    assert (a1.hold_time_min_s, a1.hold_time_max_s) == (0.25, 1.0)
    assert a1.mode_probabilities == (0.50, 0.30, 0.20)
    assert a1.pulse_on_fraction == 0.20
    assert a1.curriculum_steps == 75_000

@pytest.mark.parametrize("stage", ["", "a0", "A2"])
def test_unknown_stage_is_rejected(stage):
    with pytest.raises(ValueError, match="stage"):
        teacher.stage_disturbance_cfg(stage)
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_disturbance.py -q
```

Expected: collection fails with `ModuleNotFoundError: go2_pvcnn.tasks.m1_panda_teacher`.

- [x] **Step 3: Implement immutable configuration and stage factory**

Create:

```python
@dataclass(frozen=True)
class M1PandaDisturbanceCfg:
    force_limit_n: tuple[float, float, float]
    torque_limit_nm: tuple[float, float, float]
    hold_time_min_s: float
    hold_time_max_s: float
    curriculum_start_scale: float = 0.25
    curriculum_steps: int = 50_000
    mode_probabilities: tuple[float, float, float] = (1.0, 0.0, 0.0)
    pulse_on_fraction: float = 0.20

    def __post_init__(self) -> None:
        for name, values in (
            ("force_limit_n", self.force_limit_n),
            ("torque_limit_nm", self.torque_limit_nm),
        ):
            if len(values) != 3 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
                for value in values
            ):
                raise ValueError(f"{name} must contain three finite positive values")
        if (
            not math.isfinite(self.hold_time_min_s)
            or not math.isfinite(self.hold_time_max_s)
            or self.hold_time_min_s <= 0.0
            or self.hold_time_min_s > self.hold_time_max_s
        ):
            raise ValueError("hold times must be finite, positive, and ordered")
        if not 0.0 < self.curriculum_start_scale <= 1.0:
            raise ValueError("curriculum_start_scale must be in (0, 1]")
        if isinstance(self.curriculum_steps, bool) or self.curriculum_steps <= 0:
            raise ValueError("curriculum_steps must be a positive integer")
        if (
            len(self.mode_probabilities) != 3
            or any(not math.isfinite(value) or value < 0.0 for value in self.mode_probabilities)
            or not math.isclose(sum(self.mode_probabilities), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("mode_probabilities must be three nonnegative values summing to one")
        if not 0.0 < self.pulse_on_fraction <= 1.0:
            raise ValueError("pulse_on_fraction must be in (0, 1]")

def stage_disturbance_cfg(stage: str) -> M1PandaDisturbanceCfg:
    if stage == "A0":
        return M1PandaDisturbanceCfg(
            force_limit_n=(10.0, 10.0, 10.0),
            torque_limit_nm=(2.0, 2.0, 2.0),
            hold_time_min_s=1.0,
            hold_time_max_s=2.0,
            curriculum_steps=50_000,
        )
    if stage == "A1":
        return M1PandaDisturbanceCfg(
            force_limit_n=(20.0, 20.0, 20.0),
            torque_limit_nm=(5.0, 5.0, 5.0),
            hold_time_min_s=0.25,
            hold_time_max_s=1.0,
            curriculum_steps=75_000,
            mode_probabilities=(0.50, 0.30, 0.20),
            pulse_on_fraction=0.20,
        )
    raise ValueError(f"stage must be 'A0' or 'A1', got {stage!r}")
```

Replace the ellipsis with explicit checks and error messages; no validation may be deferred.

- [x] **Step 4: Add failing deterministic scheduling tests**

Cover independent durations, bounds, curriculum, all A1 modes, reset isolation, clone diagnostics, and atomic finite/index failures:

```python
def test_a0_scheduler_is_seeded_bounded_and_independent():
    left = teacher.M1PandaDisturbanceScheduler(
        teacher.stage_disturbance_cfg("A0"), 8, "cpu", 0.02, seed=7
    )
    right = teacher.M1PandaDisturbanceScheduler(
        teacher.stage_disturbance_cfg("A0"), 8, "cpu", 0.02, seed=7
    )
    first = left.advance()
    assert torch.equal(first, right.advance())
    assert first.shape == (8, 6)
    assert torch.all(first[:, :3].abs() <= 2.5 + 1e-6)
    assert torch.all(first[:, 3:].abs() <= 0.5 + 1e-6)
    assert left.remaining_steps.unique().numel() > 1

def test_reset_clears_only_selected_environments():
    scheduler = teacher.M1PandaDisturbanceScheduler(
        teacher.stage_disturbance_cfg("A1"), 4, "cpu", 0.02, seed=3
    )
    scheduler.advance()
    before = scheduler.current_wrench_b
    scheduler.reset([1, 3])
    assert torch.equal(scheduler.current_wrench_b[[1, 3]], torch.zeros(2, 6))
    assert torch.equal(scheduler.current_wrench_b[[0, 2]], before[[0, 2]])
```

Use a test-only config with `mode_probabilities` forced to each of hold/ramp/pulse so every envelope has exact expected values rather than relying on random mode coverage.

- [x] **Step 5: Implement the scheduler**

Implement state tensors `_current`, `_start`, `_target`, `_duration_steps`, `_elapsed_steps`, `_remaining_steps`, `_mode`, a device-matched `torch.Generator`, and a scalar `_global_step`. On `advance()`:

```python
needs_sample = self._remaining_steps == 0
if bool(needs_sample.any()):
    self._sample_segments(needs_sample.nonzero(as_tuple=False).flatten())

fraction = (self._elapsed_steps + 1).to(self.dtype) / self._duration_steps.to(self.dtype)
hold = self._target
ramp = self._start + fraction.unsqueeze(1) * (self._target - self._start)
pulse = torch.where(
    (fraction <= self.cfg.pulse_on_fraction).unsqueeze(1),
    self._target,
    torch.zeros_like(self._target),
)
self._current = torch.where(
    (self._mode == HOLD_MODE).unsqueeze(1),
    hold,
    torch.where((self._mode == RAMP_MODE).unsqueeze(1), ramp, pulse),
)
self._elapsed_steps += 1
self._remaining_steps -= 1
self._global_step += 1
return self._current.clone()
```

Sample integer durations inclusively from `ceil(min_s / step_dt)` through `ceil(max_s / step_dt)`. Scale the six target limits by:

```python
progress = min(self._global_step / self.cfg.curriculum_steps, 1.0)
scale = self.cfg.curriculum_start_scale + (1.0 - self.cfg.curriculum_start_scale) * progress
```

`reset(env_ids)` zeros segment state only; it does not rewind `_global_step`.

- [x] **Step 6: Move reusable wrench conversion and clear shim behind the new module**

Implement:

```python
def base_wrench_to_body_local(
    force_b: torch.Tensor,
    torque_b: torch.Tensor,
    base_quat_w: torch.Tensor,
    body_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Validate matching batch dimensions, last dimensions 3/4, dtype/device, and finite values.
    force_w = quat_rotate(base_quat_w, force_b)
    torque_w = quat_rotate(base_quat_w, torque_b)
    return quat_rotate_inverse(body_quat_w, force_w), quat_rotate_inverse(body_quat_w, torque_w)

def clear_external_wrench(robot) -> None:
    empty = torch.zeros(0, 3, device=robot.device)
    try:
        robot.set_external_force_and_torque(empty, empty)
    except RuntimeError as error:
        known = re.fullmatch(
            r"shape mismatch: value tensor of shape \[0\] cannot be broadcast to indexing result "
            r"of shape \[\d+, 3\]",
            str(error),
        )
        if known is None or robot.has_external_wrench is not False:
            raise
```

Update `m1_panda_wrench_probe.py` to import these as `_base_wrench_to_body_local` and `_clear_external_wrench`, preserving its existing test-visible names and behavior. Do not change the seven-row authority contract.

- [x] **Step 7: Run focused and foundation regression**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_disturbance.py \
  tests/test_m1_panda_wrench_probe_static.py \
  tests/test_m1_panda_wrench.py -q
```

Expected: all selected tests pass; no real simulator is launched.

- [x] **Step 8: Record task evidence**

Create `notes/log/2026-08-14-m1-panda-teacher-disturbance-scheduler.md` with RED/GREEN commands, counts, exact cfg, limitations, and `Git Ref: unavailable`. Update T400 and both note indexes.

---

### Task 2: Strict Checkpoint and Manifest Contract

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_checkpoint.py`

**Interfaces:**
- Produces: `file_sha256(path)`, `module_sha256(module)`, `atomic_write_manifest(path, payload)`, `load_manifest_for_checkpoint(path)`, `validate_teacher_checkpoint(path, expected_stage, expected_observation_dim, expected_action_dim, expected_actor_hidden_dims, expected_base_sha256=None, require_optimizer=False)`, `load_frozen_teacher_actor(path, device, policy_cfg)`, and `build_run_manifest(stage, task_id, seed, composer_cfg, disturbance_cfg, base_checkpoint=None, resume_checkpoint=None)`.
- `validate_teacher_checkpoint` returns `(checkpoint_dict, manifest_dict)` only after stage, schema, dims, hidden dims, tensor shapes, and optional base hash pass.
- `load_frozen_teacher_actor` returns a strictly loaded `ActorCritic` whose every parameter has `requires_grad=False`.

- [x] **Step 1: Write failing manifest and hash tests**

```python
def test_manifest_is_atomic_and_checkpoint_hash_is_stable(tmp_path):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "run_manifest.json"
    checkpoint_api.atomic_write_manifest(manifest, {"schema_version": 1, "stage": "A0"})
    assert json.loads(manifest.read_text())["stage"] == "A0"
    assert checkpoint_api.file_sha256(checkpoint) == hashlib.sha256(b"checkpoint").hexdigest()
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []
```

Also test JSON type validation and cleanup after a simulated `os.replace` failure.

- [x] **Step 2: Run RED, then implement atomic IO and hashes**

Expected RED is the missing module. Implement temporary-file write + flush + `os.fsync` + `os.replace`, `hashlib.sha256` streaming in 1 MiB blocks, and module hashing over sorted state-dict keys, dtype, shape, and contiguous CPU bytes.

- [x] **Step 3: Write failing stage/shape/base-hash tests**

Create a minimal real `ActorCritic(60, 60, 16, actor_hidden_dims=[256, 128], critic_hidden_dims=[256, 128])`, save its state dict with RSL-RL keys, and write adjacent manifests. Assert:

```python
checkpoint, manifest = checkpoint_api.validate_teacher_checkpoint(
    path,
    expected_stage="A0",
    expected_observation_dim=60,
    expected_action_dim=16,
    expected_actor_hidden_dims=(256, 128),
)
assert manifest["stage"] == "A0"
```

Mutation cases must reject: missing manifest, `stage=A1` for an A0 request, 572 observation dim, 12 action dim, wrong hidden dims, missing `model_state_dict`, wrong `actor.0.weight` shape, missing `optimizer_state_dict` for resume validation, and an A1 manifest whose `base_checkpoint_sha256` differs from the provided base.

- [x] **Step 4: Implement strict validation**

Use constants:

```python
TEACHER_SCHEMA_VERSION = 1
TEACHER_OBSERVATION_DIM = 60
TEACHER_ACTION_DIM = 16
TEACHER_HIDDEN_DIMS = (256, 128)
```

Validate exact actor shapes:

```python
{
    "actor.0.weight": (256, 60),
    "actor.0.bias": (256,),
    "actor.2.weight": (128, 256),
    "actor.2.bias": (128,),
    "actor.4.weight": (16, 128),
    "actor.4.bias": (16,),
}
```

Also validate critic input/output shapes and `std == (16,)`. Do not infer compatibility solely from the manifest.

- [x] **Step 5: Implement strict frozen actor construction**

```python
def load_frozen_teacher_actor(path, *, device, policy_cfg):
    checkpoint, _ = validate_teacher_checkpoint(
        path,
        expected_stage="A0",
        expected_observation_dim=60,
        expected_action_dim=16,
        expected_actor_hidden_dims=(256, 128),
    )
    cfg = dict(policy_cfg)
    cfg.pop("class_name", None)
    actor = ActorCritic(60, 60, 16, **cfg).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor
```

Test `not actor.training`, every parameter frozen, exact output shape, and `module_sha256` unchanged after inference.

- [x] **Step 6: Run focused tests and compile**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_checkpoint.py -q
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py
```

Expected: all focused tests and compile pass.

- [x] **Step 7: Record task evidence**

Create the required log, update T400/todo/log indexes, and record `Git Ref: unavailable`.

---

### Task 3: Teacher Reward Helpers, Environment Config, and Gym IDs

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_teacher_rewards.py`
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/register_m1_envs.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_env_cfg_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py`

**Interfaces:**
- Produces `M1PandaTeacherA0EnvCfg` and `M1PandaTeacherA1EnvCfg`, both with exact 60/16 manager contracts and `teacher_stage` field.
- Produces reward helpers `base_xy_drift_l2`, `selected_joint_velocity_l2`, `selected_joint_torques_l2`, `teacher_residual_l2`, and `teacher_residual_rate_l2`.
- Registers `Isaac-M1-Panda-Teacher-A0-v0` and `Isaac-M1-Panda-Teacher-A1-v0` lazily.

- [x] **Step 1: Write failing pure reward tests**

Use small fake env/asset objects and assert exact tensors:

```python
def test_base_xy_drift_is_relative_to_env_origin():
    env = fake_env(root_pos=[[3.0, 4.0, 0.6]], origin=[[1.0, 1.0, 0.0]])
    assert torch.equal(rewards.base_xy_drift_l2(env), torch.tensor([13.0]))

def test_teacher_residual_terms_consume_wrapper_state():
    env.m1_teacher_trainable_residual = torch.tensor([[1.0, 2.0]])
    env.m1_teacher_previous_trainable_residual = torch.tensor([[0.5, 1.0]])
    assert rewards.teacher_residual_l2(env).item() == 5.0
    assert rewards.teacher_residual_rate_l2(env).item() == 1.25
```

Test missing/mismatched/non-finite wrapper state raises a clear `RuntimeError`.

- [x] **Step 2: Implement the five helpers**

`selected_joint_velocity_l2` must index `asset.data.joint_vel[:, asset_cfg.joint_ids]`; `selected_joint_torques_l2` must index `asset.data.applied_torque[:, asset_cfg.joint_ids]`; `base_xy_drift_l2` uses `root_pos_w[:, :2] - env.scene.env_origins[:, :2]`. Residual helpers read the two named tensors published before `env.step()` and sum squares per environment.

- [x] **Step 3: Write failing AST cfg and registry tests**

Assert:

- both cfg classes inherit the shared Teacher base;
- scene uses `M1_PANDA_CFG`;
- action terms remain exactly 12 leg position then 4 wheel velocity, scales 0.25/8.0;
- policy terms remain in exact order and total 60;
- reward terms are `alive`, `base_height`, `base_linear_velocity`, `base_angular_velocity`, `flat_orientation_l2`, `base_xy_drift`, `wheel_speed`, `residual`, `residual_rate`, `joint_torques`, `feet_slide`;
- base height target is `0.60`;
- wheel and torque selectors use M1 names only;
- terminations retain timeout, base contact, and bad orientation;
- A0/A1 stage fields and disturbance values differ exactly as approved;
- registration count grows from 21 to 23 without changing prior mappings.

- [x] **Step 4: Implement the environment cfg**

Use a shared cfg derived from `M1PandaSmokeEnvCfg`; define dedicated reward cfg rather than mutating smoke defaults:

```python
@configclass
class M1PandaTeacherRewardsCfg:
    alive = RewTerm(func=isaac_mdp.is_alive, weight=2.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-12.0, params={"target_height": 0.60})
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    base_xy_drift = RewTerm(func=mdp.base_xy_drift_l2, weight=-1.0)
    wheel_speed = RewTerm(
        func=mdp.selected_joint_velocity_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_WHEEL_JOINT_NAMES))},
    )
    residual = RewTerm(func=mdp.teacher_residual_l2, weight=-0.02)
    residual_rate = RewTerm(func=mdp.teacher_residual_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.selected_joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(M1_JOINT_NAMES))},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(M1_FOOT_BODY_NAMES)),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(M1_FOOT_BODY_NAMES)),
        },
    )
```

Use `teacher_stage`, force/torque limits, time bounds, curriculum steps, mode probabilities, and pulse fraction as scalar/tuple cfg fields so `env_cfg.to_dict()` captures the full runtime contract. A1 subclasses A0 and overrides these values in `__post_init__` only after `super()`.

- [x] **Step 5: Register the two IDs lazily**

Append string entry points to `register_m1_envs.py`; do not import the cfg module at registry import time. Update the existing expected registration table and printed ID list.

- [x] **Step 6: Run static/foundation regression**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_env_cfg_static.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_asset_cfg_static.py -q
```

Expected: all pass; old 21 registrations plus two exact new mappings are preserved.

- [x] **Step 7: Record task evidence**

Create the cfg/wiring log and update repository memory with the exact static test count and Git-unavailable status.

---

### Task 4: A0 Wrapper, Disturbance Application, and Reset Semantics

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py`

**Interfaces:**
- Constructor: `M1PandaTeacherEnvWrapper(env, *, stage: str, base_actor=None, disturbance_cfg=None, seed=0)`.
- RSL-RL methods: `get_observations()`, `reset()`, `step(actions)`; returns flattened policy obs and critic mirror.
- Diagnostics: `last_final_action`, `last_trainable_residual`, `current_wrench_b`, `frozen_actor_hash`.

- [x] **Step 1: Build a minimal fake ManagerBasedRLEnv and write A0 RED tests**

The fake must expose `num_envs`, `device`, `max_episode_length`, `action_manager.total_action_dim=16`, `observation_manager.compute()`, `scene["robot"]`, `scene.env_origins`, `cfg`, `reset`, and `step`. The robot records `set_external_force_and_torque` calls and provides identity base/hand quaternions.

Test exact A0 behavior:

```python
wrapper = M1PandaTeacherEnvWrapper(fake, stage="A0", seed=5)
obs, extras = wrapper.get_observations()
assert obs.shape == (fake.num_envs, 60)
assert torch.equal(extras["observations"]["critic"], obs)

raw = torch.full((fake.num_envs, 16), 0.5)
next_obs, reward, done, extras = wrapper.step(raw)
assert fake.last_action.shape == (fake.num_envs, 16)
assert torch.allclose(fake.last_action[:, :12], torch.full((fake.num_envs, 12), 0.04))
assert torch.allclose(fake.last_action[:, 12:], torch.full((fake.num_envs, 4), 0.025))
assert fake.robot.external_wrench_calls == 1
```

The expected first-step normalized offsets follow existing physical slew divided by action scale: `0.01/0.25=0.04`, `0.2/8=0.025`.

- [x] **Step 2: Run RED and implement strict wrapper initialization/formatting**

Resolve exactly one `BASE_LINK` and `panda_hand`, require 16 actions and stage match with `env.cfg.teacher_stage`, build the scheduler from cfg fields, build the composer, reset the env, zero published residual tensors, and cache only a finite `(N, 60)` policy observation.

- [x] **Step 3: Implement A0 step order**

Use this exact order:

```python
self._validate_policy_action(actions)
self.env.m1_teacher_previous_trainable_residual.copy_(
    self.env.m1_teacher_trainable_residual
)
self.env.m1_teacher_trainable_residual.copy_(actions.clamp(-1.0, 1.0))
zero = torch.zeros_like(actions)
final_action = self._residual_composer.compose(zero, actions)
wrench_b = self._disturbance.advance()
self._apply_wrench(wrench_b)
obs_dict, rewards, terminated, truncated, extras = self.env.step(final_action)
dones = terminated | truncated
if bool(dones.any()):
    self._reset_state(dones.nonzero(as_tuple=False).flatten())
    self._apply_wrench(self._disturbance.current_wrench_b)
extras["time_outs"] = truncated
obs = self._format_and_cache(obs_dict)
return obs, rewards, dones, extras_with_critic
```

`_apply_wrench` recomputes BASE_LINK→hand-local rotation from live body quaternions every call and writes `[N, 1, 3]` force/torque to `panda_hand`.

- [x] **Step 4: Add reset/failure RED tests and implement them**

Test a done mask `[False, True, False]` clears only environment 1 in composer, scheduler, published current/previous residual, and reapplied external wrench. Test full reset clears all. Test shape, dtype, device, NaN observation, NaN action, NaN reward, ambiguous body IDs, action dim 15, and observation dim 59 fail before corrupting state.

- [x] **Step 5: Run focused wrapper tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_wrapper.py -k 'a0 or reset or finite' -q
```

Expected: all selected tests pass and fake-env recorded ordering proves force is set before `env.step`.

- [x] **Step 6: Record task evidence**

Write A0 wrapper RED/GREEN log and sync T400/todo/log indexes.

---

### Task 5: A1 Frozen A0 Actor and Two-Level Residual Composition

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py`

**Interfaces:**
- A1 requires a frozen `base_actor` with `act_inference(obs) -> Tensor[N,16]`.
- `frozen_actor_hash` is calculated at wrapper construction and `assert_frozen_actor_unchanged()` raises on drift.
- A1 owns `_base_composer` and `_residual_composer`; A0 owns only `_residual_composer`.

- [x] **Step 1: Write A1 RED tests**

Use a deterministic fake actor returning `0.5` and a trainable residual input `-0.25`. Assert the first base composer output and second composer output separately, and assert the base actor receives the cached pre-step observation.

```python
wrapper = M1PandaTeacherEnvWrapper(fake, stage="A1", base_actor=frozen)
cached = wrapper.get_observations()[0].clone()
wrapper.step(torch.full((2, 16), -0.25))
assert torch.equal(frozen.last_observation, cached)
assert wrapper.base_composer.physical_residual[:, :12].eq(0.01).all()
assert wrapper.residual_composer.physical_residual[:, :12].eq(-0.01).all()
assert wrapper.last_final_action.shape == (2, 16)
```

Test A1 rejects `base_actor=None`, a training-mode actor, any `requires_grad=True` parameter, and wrong output shape/non-finite output.

- [x] **Step 2: Implement A1 construction and inference**

At construction require the actor is in eval mode and all parameters frozen. In `step`, before mutating wrapper state:

```python
with torch.no_grad():
    base_residual = self._base_actor.act_inference(self._latest_observation)
self._validate_action_tensor("base_actor output", base_residual)
base_action = self._base_composer.compose(torch.zeros_like(actions), base_residual)
final_action = self._residual_composer.compose(base_action, actions)
```

Never place the base actor on the PPO runner or optimizer; it exists only inside the wrapper.

- [x] **Step 3: Add two-composer reset and immutability tests**

Assert done environment IDs clear both composers while non-done histories stay bit-identical. Mutate one frozen tensor under `torch.no_grad()` and assert `assert_frozen_actor_unchanged()` raises with both initial/current hashes. Verify normal A1 steps leave the hash unchanged.

- [x] **Step 4: Run the full wrapper test**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_wrapper.py -q
```

Expected: all A0 and A1 wrapper tests pass.

- [x] **Step 5: Record task evidence**

Write the A1 frozen-chain log and update branch/dashboard indexes.

---

### Task 6: Teacher PPO Config and Training Entrypoint

**Files:**
- Create: `Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py`
- Modify: `Go2Pvcnn/agent/__init__.py`
- Create: `Go2Pvcnn/scripts/m1_panda_teacher_train.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py`

**Interfaces:**
- `get_m1_panda_teacher_train_cfg() -> dict` returns a fresh deep structure on every call.
- CLI selects stage, constructs exact task ID, validates base/resume contracts, writes `run_manifest.json`, and returns nonzero on every failure.
- Required CLI: `--stage`, `--num_envs`, `--seed`, `--max_iterations`, `--run_name`, `--log-root`, `--base-checkpoint`, `--resume-checkpoint`, `--reset-optimizer`, `--save-interval`, `--num-steps-per-env`, `--learning-epochs`, `--num-mini-batches`, plus AppLauncher args.

- [x] **Step 1: Write failing train-config tests**

Assert exact base settings from the approved spec and factory independence:

```python
left = get_m1_panda_teacher_train_cfg()
right = get_m1_panda_teacher_train_cfg()
assert left["num_steps_per_env"] == 24
assert left["save_interval"] == 100
assert left["policy"]["actor_hidden_dims"] == [256, 128]
assert left["policy"]["critic_hidden_dims"] == [256, 128]
assert left["policy"]["init_noise_std"] == 0.01
assert left["algorithm"]["num_learning_epochs"] == 5
assert left["algorithm"]["num_mini_batches"] == 4
left["policy"]["actor_hidden_dims"].append(64)
assert right["policy"]["actor_hidden_dims"] == [256, 128]
```

- [x] **Step 2: Implement and export the config factory**

Copy only the plain M1 PPO values needed by this task; do not import and mutate `get_m1_train_cfg()` because RSL-RL pops class names from the dict.

- [x] **Step 3: Write failing parser/stage/static-flow tests**

Load the script without launching Isaac. Assert exact task mapping:

```python
assert TASK_IDS == {
    "A0": "Isaac-M1-Panda-Teacher-A0-v0",
    "A1": "Isaac-M1-Panda-Teacher-A1-v0",
}
```

AST/source tests must verify A1 requires `--base-checkpoint`, both resume paths call strict validation before `runner.load`, A1 loads the frozen actor before wrapper construction, manifest is written before learning, frozen hash is checked after learning, and env closes before app close.

- [x] **Step 4: Implement CLI and preflight outside simulator-dependent logic**

Provide pure helpers:

```python
def validate_cli_contract(args) -> None:
    if args.stage == "A0" and args.base_checkpoint is not None:
        raise ValueError("A0 does not accept --base-checkpoint")
    if args.stage == "A1" and args.base_checkpoint is None:
        raise ValueError("A1 requires --base-checkpoint")
    if args.max_iterations <= 0 or args.num_envs <= 0:
        raise ValueError("--max_iterations and --num_envs must be positive")

def build_log_dir(log_root: Path, stage: str, run_name: str | None) -> Path:
    name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = log_root / stage.lower() / name
    path.mkdir(parents=True, exist_ok=False)
    return path
```

Existing run directories are hard errors unless resuming, in which case the log directory is the parent of `--resume-checkpoint` and `--run_name` is forbidden.

- [x] **Step 5: Implement main training flow**

After `AppLauncher` startup:

1. resolve cfg from Gym spec and set `num_envs`, seed, and device;
2. build a fresh train cfg and apply validated smoke overrides;
3. validate A0 base checkpoint for A1 and construct the frozen actor;
4. create underlying env and `M1PandaTeacherEnvWrapper`;
5. dump env/train YAML;
6. write atomic start manifest with checkpoint pattern `model_<iteration>.pt`;
7. construct `OnPolicyRunner` from a deep copy of train cfg;
8. validate and load current-stage resume checkpoint if given;
9. run `runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)`;
10. call `wrapper.assert_frozen_actor_unchanged()` for A1;
11. atomically update manifest with `status="completed"`, final iteration, final checkpoint filename, and frozen final hash;
12. close env, then simulation app; on exception, write `status="failed"` when log dir exists and return 1.

- [x] **Step 6: Ensure manifest fields are complete**

Call `build_run_manifest` with at least:

```python
{
    "schema_version": 1,
    "stage": args.stage,
    "task_id": TASK_IDS[args.stage],
    "observation_dim": 60,
    "action_dim": 16,
    "actor_hidden_dims": [256, 128],
    "seed": args.seed,
    "composer": asdict(M1ResidualActionComposerCfg()),
    "disturbance": asdict(stage_disturbance_cfg(args.stage)),
    "checkpoint_pattern": "model_<iteration>.pt",
    "base_checkpoint": normalized_path_or_none,
    "base_checkpoint_sha256": sha_or_none,
    "frozen_actor_initial_sha256": hash_or_none,
    "resume_checkpoint": normalized_path_or_none,
    "status": "running",
}
```

- [x] **Step 7: Run config/script tests and compile**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_train_static.py \
  tests/test_m1_panda_teacher_checkpoint.py -q
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  agent/m1_panda_teacher_train_cfg.py \
  scripts/m1_panda_teacher_train.py
```

Expected: all tests and compile pass without launching Isaac.

- [x] **Step 8: Record task evidence**

Create the training-entrypoint log and update T400/todo/log indexes.

---

### Task 7: Real CPU A0→A1 Smoke, Resume Verification, and Training Runbook

**Files:**
- Create: `Go2Pvcnn/scripts/m1_panda_teacher_smoke.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py`
- Create: `docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-08-14-m1-panda-teacher-a0-a1-static-regression.md`
- Create: `notes/log/2026-08-14-m1-panda-teacher-a0-a1-cpu-smoke.md`

**Interfaces:**
- Smoke driver takes `--output-root` and invokes the real training entrypoint four times: A0 initial, A0 resume, A1 initial, A1 resume.
- It emits one final JSON object with paths, return codes, manifest stages/status, checkpoint validation, and frozen-hash equality.
- Runbook contains exact formal and resume commands with no hidden manual step.

- [x] **Step 1: Write failing smoke-driver static tests**

Assert the driver uses `subprocess.run(command, check=False, timeout=600, text=True, capture_output=True)`, parses manifests, discovers checkpoints by numeric model suffix, and runs this sequence:

```text
A0 initial -> A0 resume -> A1 initial(base=A0) -> A1 resume(base=same A0)
```

It must reject any nonzero child return, missing checkpoint/manifest, stage mismatch, changed base hash, or `frozen_actor_initial_sha256 != frozen_actor_final_sha256`.

- [x] **Step 2: Implement the acceptance driver**

Use the current interpreter (`sys.executable`) and commands equivalent to:

```bash
python scripts/m1_panda_teacher_train.py \
  --stage A0 --num_envs 1 --max_iterations 1 \
  --num-steps-per-env 4 --learning-epochs 1 --num-mini-batches 1 \
  --save-interval 1 --run_name smoke_a0 --log-root OUTPUT \
  --device cpu --headless
```

Resume each stage for one more iteration. Use a per-child timeout of 600 seconds and preserve stdout/stderr paths in the final JSON. Never claim convergence.

- [x] **Step 3: Run the full pure/static regression before Isaac**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=.:rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_disturbance.py \
  tests/test_m1_panda_teacher_checkpoint.py \
  tests/test_m1_panda_teacher_env_cfg_static.py \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_train_static.py \
  tests/test_m1_residual_action.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_panda_wrench.py \
  tests/test_m1_panda_wrench_probe_static.py -q
```

Expected: all selected tests pass. Record the exact count in the static regression log.

- [x] **Step 4: Run a real one-env environment reset/step probe**

Before PPO, run the train entrypoint with the smallest accepted one-iteration A0 command under a 600-second timeout. Inspect output for the Gym ID, observation dimension 60, action dimension 16, nonzero scheduled wrench, finite reward, and saved `model_0.pt`/manifest.

- [x] **Step 5: Run the four-stage CPU smoke driver**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 timeout 2400 \
  /home/xk/miniconda3/envs/loco/bin/python \
  scripts/m1_panda_teacher_smoke.py \
  --output-root /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher_smoke \
  --device cpu --headless
```

Expected exit `0` and final JSON with four zero return codes, A0/A1 checkpoint validation true, same A0 base SHA for initial/resumed A1, and frozen initial/final hashes equal.

- [x] **Step 6: Diagnose runtime failures systematically**

If any real command fails, invoke `systematic-debugging` before editing. Preserve the failing output in the CPU smoke log, identify the first violated contract, write a focused RED test, implement only the root-cause fix, rerun the focused test, then rerun from the first failed real stage. Do not weaken dimension, reset, finite, checkpoint, or frozen-hash gates to obtain a pass.

- [x] **Step 7: Write the exact runbook**

Document these formal templates, using the same A0 base checkpoint for A1 resume:

```bash
# A0 formal CPU-compatible command (replace cpu with a supported CUDA device only after framework upgrade)
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A0 --num_envs 64 --max_iterations 3000 \
  --run_name a0_force_balance --device cpu --headless

# A0 resume
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A0 --resume-checkpoint /ABS/A0/model_N.pt \
  --max_iterations 1000 --device cpu --headless

# A1 formal
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --num_envs 64 --max_iterations 3000 \
  --run_name a1_dynamic_force_balance --device cpu --headless

# A1 resume
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
/home/xk/miniconda3/envs/loco/bin/python \
  /home/xk/coding/M1/Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --resume-checkpoint /ABS/A1/model_M.pt \
  --max_iterations 1000 --device cpu --headless
```

The actual runbook must contain these complete commands, log paths, manifest interpretation, TensorBoard command, safe interruption expectations, and the `sm_120/sm_90` limitation.

- [x] **Step 8: Run final compile, placeholder, and command scans**

```bash
cd /home/xk/coding/M1
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_env_cfg.py \
  Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py \
  Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  Go2Pvcnn/scripts/m1_panda_teacher_smoke.py
rg -n 'T''BD|TO''DO|FI''XME|待''定|待''确认' \
  docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher*.py \
  Go2Pvcnn/scripts/m1_panda_teacher*.py
```

Expected: compile exit `0`; placeholder scan has no matches except intentional Python slicing if any, which must be reviewed manually.

- [x] **Step 9: Align all notes and perform completion audit**

Update T400.5 children with exact test/runtime evidence; add static and CPU logs to both indexes; update `notes/ai/ai-02-training-and-entrypoints.md`, `notes/ai/ai-03-environment-and-observations.md`, and `notes/ai/ai-05-ppo-and-runner.md` because the training/observation/checkpoint contract changed. Audit every spec section against current files and command output. Explicitly leave Student, Panda motion/grasping, T400.3 mechanical validation, network-denial runtime, long-policy convergence, and unsupported-GPU execution open.

- [x] **Step 10: Record Git status**

Run `git -C /home/xk/coding/M1 rev-parse --is-inside-work-tree`; expected failure confirms the established repository condition. Record `Baseline Ref`, `Candidate Ref`, `Last Feature Commit`, and `Last Verified Commit` as unavailable rather than initializing or fabricating refs.

---

## Plan Self-Review Checklist

- [x] Every approved spec section maps to a task: environment/obs (Task 3), disturbance (Task 1), action architecture/reset (Tasks 4–5), rewards/termination (Task 3), PPO/entrypoint (Task 6), checkpoint/resume (Tasks 2 and 6), failures (Tasks 1–6), CPU acceptance/commands (Task 7).
- [x] No 572/586-dimensional checkpoint can enter A1 because Task 2 validates both manifest dimensions and actual state-dict shapes.
- [x] A1 frozen actor is outside the RSL optimizer and is hashed before/after the real training call.
- [x] The two composer histories and disturbance state reset per done environment.
- [x] All new code has a RED step before implementation and exact focused commands after implementation.
- [x] Existing foundation tests remain in the final regression.
- [x] Notes and documented stage contracts are updated only after evidence exists.
- [x] No Git commit is attempted in the non-Git workspace.
