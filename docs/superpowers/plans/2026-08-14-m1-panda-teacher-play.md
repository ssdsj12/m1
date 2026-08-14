# M1 + Panda Teacher A0/A1 Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's user-selected execution mode is single-agent inline execution, so use `superpowers:executing-plans` and do not dispatch subagents.

**Goal:** Add a dedicated, strict A0/A1 Teacher playback entrypoint that defaults to GUI plus stage-matched six-dimensional disturbance and supports an explicit zero-disturbance baseline.

**Architecture:** Extend `M1PandaTeacherEnvWrapper` with a default-on disturbance gate while preserving the exact training path. Add one standalone play script that performs strict manifest/tensor/base-SHA validation, reconstructs A1's frozen-A0 plus residual chain, runs inference without writes, and emits periodic wrench/reset diagnostics. Keep training, assets, reward, PPO, observation, and action contracts unchanged.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab 2.1, Gymnasium, RSL-RL, pytest, Markdown runbooks.

## Global Constraints

- Work only in `/home/xk/coding/M1`; this directory is not a Git worktree, so each nominal commit checkpoint is recorded in T400/log notes instead of running `git commit`.
- Execute with one agent only; do not dispatch subagents.
- Preserve the exact Teacher policy contract: 60 float32 observations and 16 M1 actions.
- A0 must use zero base action; A1 must use the user-supplied frozen A0 checkpoint plus the A1 residual checkpoint.
- GUI and stage-matched disturbance are enabled by default; only `--disable-disturbance` selects the zero-wrench baseline.
- `--steps 0` means run while the SimulationApp window is open; a positive value is a hard upper bound.
- Playback must not learn, create output runs, write checkpoints/manifests, or require optimizer state.
- Strictly reject missing/incompatible manifests, model tensor shapes, stages, A1 base SHA, non-finite values, and invalid CLI combinations.
- Use `/home/xk/miniconda3/envs/go2/bin/python` and `--device cuda:0` for the requested GPU0 smoke.
- Do not claim A1 behavior acceptance; its current checkpoint remains diagnostic.

---

## File Map

- Modify `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py`: own the default-on disturbance gate and effective-wrench diagnostics.
- Modify `Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py`: fake-env RED/GREEN coverage for zero-disturbance reset, step, scheduler, and A1 composition behavior.
- Create `Go2Pvcnn/scripts/m1_panda_teacher_play.py`: own CLI, checkpoint preflight, environment/runner construction, inference loop, reset statistics, and cleanup.
- Create `Go2Pvcnn/tests/test_m1_panda_teacher_play_static.py`: import-safe CLI/helper and source-order contract tests.
- Modify `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py`: require the runbook's A0/A1 play and zero-disturbance commands.
- Modify `docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md`: publish copyable GPU0 GUI/headless commands and limitations.
- Modify `notes/human/human-02-training-and-entrypoints.md`: add the human-facing dedicated Teacher play entrypoint.
- Modify `notes/ai/ai-02-training-and-entrypoints.md`: add the agent-facing wrapper/checkpoint/play boundary.
- Modify `notes/todo.md`, `notes/todo/T400-m1-panda-force-aware-teacher-student.md`, and `notes/log/index.md`: update durable project state.
- Create `notes/log/2026-08-14-m1-panda-teacher-play-wrapper.md`, `notes/log/2026-08-14-m1-panda-teacher-play-entrypoint.md`, and `notes/log/2026-08-14-m1-panda-teacher-play-gpu0-smoke.md`: retain RED/GREEN/runtime evidence.

---

### Task 1: Default-on wrapper disturbance gate

**Files:**

- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py`
- Create: `notes/log/2026-08-14-m1-panda-teacher-play-wrapper.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`

**Interfaces:**

- Consumes: `clear_external_wrench(robot)`, `M1PandaDisturbanceScheduler.reset/advance/current_wrench_b`, existing `M1PandaTeacherEnvWrapper(..., seed: int = 0)` behavior.
- Produces: `M1PandaTeacherEnvWrapper(..., disturbance_enabled: bool = True)`, `disturbance_enabled: bool`, effective `current_wrench_b: Tensor[num_envs,6]`, and zero-preserving `max_abs_wrench_seen: float`.

- [ ] **Step 1: Make the fake robot support the existing clear shim and write failing default/disabled tests**

Update `_FakeRobot.set_external_force_and_torque` so a call with empty force/torque tensors and no `body_ids` records `"clear_wrench"`, stores zero `(num_envs, 1, 3)` force/torque, and increments the call counter. Retain the current assertions for body-specific non-empty calls.

Add these tests:

```python
def test_disturbance_is_enabled_by_default():
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(env, stage="A0", seed=5)

    wrapper.step(torch.zeros(2, 16))

    assert wrapper.disturbance_enabled is True
    assert torch.count_nonzero(wrapper.current_wrench_b) > 0
    assert wrapper.max_abs_wrench_seen > 0.0


def test_disabled_disturbance_never_advances_and_clears_external_wrench():
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(
        env, stage="A0", seed=5, disturbance_enabled=False
    )
    env.events.clear()

    wrapper.step(torch.full((2, 16), 0.5))

    assert wrapper.disturbance_enabled is False
    assert env.events == ["clear_wrench", "step"]
    assert torch.equal(wrapper.current_wrench_b, torch.zeros(2, 6))
    assert wrapper.max_abs_wrench_seen == 0.0
    assert torch.equal(env.robot.external_force, torch.zeros(2, 1, 3))
    assert torch.equal(env.robot.external_torque, torch.zeros(2, 1, 3))


def test_disabled_disturbance_stays_zero_across_done_and_explicit_reset():
    env = _FakeEnv(num_envs=2, stage="A0")
    wrapper = M1PandaTeacherEnvWrapper(
        env, stage="A0", seed=5, disturbance_enabled=False
    )
    env.next_terminated = torch.tensor([True, False])

    wrapper.step(torch.ones(2, 16))
    wrapper.reset()

    assert torch.equal(wrapper.current_wrench_b, torch.zeros(2, 6))
    assert wrapper.max_abs_wrench_seen == 0.0
    assert torch.equal(env.robot.external_force, torch.zeros(2, 1, 3))
    assert torch.equal(env.robot.external_torque, torch.zeros(2, 1, 3))


def test_disabled_disturbance_keeps_a1_frozen_base_and_residual_composition():
    env = _FakeEnv(num_envs=2, stage="A1")
    frozen = _FakeFrozenActor()
    wrapper = M1PandaTeacherEnvWrapper(
        env,
        stage="A1",
        base_actor=frozen,
        seed=5,
        disturbance_enabled=False,
    )

    wrapper.step(torch.full((2, 16), -0.25))

    assert frozen.last_observation is not None
    assert torch.equal(wrapper.last_final_action, torch.zeros(2, 16))
    assert torch.equal(wrapper.current_wrench_b, torch.zeros(2, 6))
```

Also add a constructor test that passes `disturbance_enabled=1` and expects `TypeError` containing `disturbance_enabled must be a bool`.

- [ ] **Step 2: Run the wrapper tests and confirm RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_wrapper.py -q
```

Expected: new tests fail because the constructor does not accept `disturbance_enabled` and the property does not exist; existing tests remain collectable.

- [ ] **Step 3: Implement the minimal disturbance gate**

Import the existing clear shim:

```python
from go2_pvcnn.tasks.m1_panda_teacher import (
    M1PandaDisturbanceCfg,
    M1PandaDisturbanceScheduler,
    base_wrench_to_body_local,
    clear_external_wrench,
    stage_disturbance_cfg,
)
```

Extend the constructor and validate the flag before mutating the environment:

```python
def __init__(
    self,
    env,
    *,
    stage: str,
    base_actor=None,
    disturbance_cfg: M1PandaDisturbanceCfg | None = None,
    seed: int = 0,
    disturbance_enabled: bool = True,
) -> None:
    if not isinstance(disturbance_enabled, bool):
        raise TypeError("disturbance_enabled must be a bool")
    self._disturbance_enabled = disturbance_enabled
```

Add a single helper and use it at constructor/reset, every step, and done reset:

```python
def _apply_effective_wrench(self, *, advance: bool) -> None:
    if not self._disturbance_enabled:
        clear_external_wrench(self._robot)
        return
    wrench_b = self._disturbance.advance() if advance else self._disturbance.current_wrench_b
    self._apply_wrench(wrench_b)
```

Replace direct reset/done calls to `_apply_wrench(self._disturbance.current_wrench_b)` with `_apply_effective_wrench(advance=False)`. Replace `advance()` plus `_apply_wrench()` in `step()` with `_apply_effective_wrench(advance=True)`. Do not change action composition or scheduler reset.

Expose effective diagnostics as clones/values:

```python
@property
def disturbance_enabled(self) -> bool:
    return self._disturbance_enabled

@property
def current_wrench_b(self) -> torch.Tensor:
    if not self._disturbance_enabled:
        return torch.zeros((self.num_envs, 6), device=self.device)
    return self._disturbance.current_wrench_b
```

`_apply_wrench` remains the only function that updates `_max_abs_wrench_seen`, so disabled playback keeps the maximum exactly zero.

- [ ] **Step 4: Run focused and related regression tests**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_disturbance.py \
  tests/test_m1_residual_action.py -q
```

Expected: all tests pass, with no change to default-on test behavior.

- [ ] **Step 5: Compile and record the task checkpoint**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_wrapper.py
```

Expected: exit `0`. Record the RED command/failure, GREEN counts, compile result, default-on/disabled semantics, and Git-unavailable state in `notes/log/2026-08-14-m1-panda-teacher-play-wrapper.md`; link it from the T400 branch, dashboard, and log index.

---

### Task 2: Strict A0/A1 play entrypoint

**Files:**

- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_play_static.py`
- Create: `Go2Pvcnn/scripts/m1_panda_teacher_play.py`
- Create: `notes/log/2026-08-14-m1-panda-teacher-play-entrypoint.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`

**Interfaces:**

- Consumes: `TASK_IDS`, `get_m1_panda_teacher_train_cfg()`, `validate_teacher_checkpoint(...)`, `load_frozen_teacher_actor(...)`, `file_sha256(...)`, `M1PandaTeacherEnvWrapper(..., disturbance_enabled=...)`, `OnPolicyRunner.get_inference_policy(device=...)`.
- Produces: import-safe `validate_cli_contract(args)`, `_termination_terms(env)`, `update_reset_counts(env, counts)`, `format_play_stats(...)`, `build_arg_parser()`, and executable `main() -> int`.

- [ ] **Step 1: Write import-safe failing tests for the CLI and diagnostics**

Create `tests/test_m1_panda_teacher_play_static.py` with a `_load_script()` helper using `importlib.util.spec_from_file_location`, then add:

```python
@pytest.mark.parametrize(
    ("stage", "base_checkpoint", "message"),
    [
        ("A0", Path("base.pt"), "does not accept"),
        ("A1", None, "requires"),
    ],
)
def test_play_cli_rejects_invalid_base_checkpoint_combinations(
    stage, base_checkpoint, message
):
    module = _load_script()
    args = SimpleNamespace(
        stage=stage,
        checkpoint=Path("model.pt"),
        base_checkpoint=base_checkpoint,
        num_envs=1,
        steps=0,
        stats_interval=100,
    )
    with pytest.raises(ValueError, match=message):
        module.validate_cli_contract(args)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"checkpoint": None}, "checkpoint"),
        ({"num_envs": 0}, "num_envs"),
        ({"steps": -1}, "steps"),
        ({"stats_interval": 0}, "stats_interval"),
    ],
)
def test_play_cli_rejects_missing_or_out_of_range_values(updates, message):
    module = _load_script()
    args = SimpleNamespace(
        stage="A0",
        checkpoint=Path("model.pt"),
        base_checkpoint=None,
        num_envs=1,
        steps=0,
        stats_interval=100,
    )
    for name, value in updates.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=message):
        module.validate_cli_contract(args)


def test_update_reset_counts_preserves_unavailable_terms():
    module = _load_script()
    env = SimpleNamespace(unwrapped=SimpleNamespace(termination_manager=None))
    counts = {"bad_orientation": None, "base_contact": None, "time_out": None}

    module.update_reset_counts(env, counts)

    assert counts == {
        "bad_orientation": None,
        "base_contact": None,
        "time_out": None,
    }


def test_update_reset_counts_accumulates_available_terms():
    module = _load_script()
    manager = _FakeTerminationManager(
        bad_orientation=torch.tensor([True, False]),
        base_contact=torch.tensor([False, True]),
        time_out=torch.tensor([False, False]),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(termination_manager=manager, device="cpu")
    )
    counts = {"bad_orientation": None, "base_contact": None, "time_out": None}

    module.update_reset_counts(env, counts)

    assert counts == {"bad_orientation": 1, "base_contact": 1, "time_out": 0}


def test_format_play_stats_reports_wrench_axes_and_unavailable_terms():
    module = _load_script()
    line = module.format_play_stats(
        step=100,
        mean_reward=1.25,
        done_count=3,
        wrench_b=torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]]),
        max_abs_wrench_seen=6.0,
        reset_counts={
            "bad_orientation": 1,
            "base_contact": None,
            "time_out": 2,
        },
    )
    assert "step=100" in line
    assert "wrench_axis_abs_max=[1.000,2.000,3.000,4.000,5.000,6.000]" in line
    assert "base_contact=unavailable" in line
```

Add a source contract test asserting all required flags, `validate_teacher_checkpoint` before `runner.load`, `load_optimizer=False`, `keep_std=True`, `torch.inference_mode()`, `simulation_app.is_running()`, `wrapper.assert_frozen_actor_unchanged()`, `env.close()` before `simulation_app.close()`, and absence of `runner.learn(`, `atomic_write_manifest(`, and `build_run_manifest(`.

- [ ] **Step 2: Run the new test file and confirm RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_play_static.py -q
```

Expected: collection/tests fail because `scripts/m1_panda_teacher_play.py` does not exist.

- [ ] **Step 3: Implement CLI validation and import-safe diagnostic helpers**

Create `scripts/m1_panda_teacher_play.py` with the same path bootstrap as the training script and these constants:

```python
TASK_IDS = {
    "A0": "Isaac-M1-Panda-Teacher-A0-v0",
    "A1": "Isaac-M1-Panda-Teacher-A1-v0",
}
TERMINATION_NAMES = ("bad_orientation", "base_contact", "time_out")
```

Implement exact CLI validation:

```python
def validate_cli_contract(args) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required")
    if args.stage == "A0" and args.base_checkpoint is not None:
        raise ValueError("A0 does not accept --base-checkpoint")
    if args.stage == "A1" and args.base_checkpoint is None:
        raise ValueError("A1 requires --base-checkpoint")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")
    if args.stats_interval <= 0:
        raise ValueError("--stats_interval must be positive")
```

Implement `_termination_terms` using `env.unwrapped.termination_manager.get_term(name)` with exception-to-unavailable behavior and `manager.time_outs` fallback, matching `m1_stability_probe.py`. `update_reset_counts` changes a missing count from `None` to `0` only when that term is actually available, then adds the current boolean sum. `format_play_stats` must validate `wrench_b` as finite `[N,6]`, compute `abs().amax(dim=0)`, and format missing terms as `unavailable`.

Build the parser with both canonical hyphenated spellings and repository-compatible underscore aliases:

```python
parser.add_argument("--stage", choices=tuple(TASK_IDS), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--base-checkpoint", "--base_checkpoint", dest="base_checkpoint", type=Path)
parser.add_argument("--num-envs", "--num_envs", dest="num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--stats-interval", "--stats_interval", dest="stats_interval", type=int, default=100)
parser.add_argument("--disable-disturbance", action="store_true")
AppLauncher.add_app_launcher_args(parser)
```

- [ ] **Step 4: Implement strict preflight and the inference lifecycle**

Inside `main()`, validate CLI before constructing `AppLauncher`. After launcher startup, obtain a fresh `train_cfg`, then execute this order:

```python
expected_base_hash = None
frozen_actor = None
if args.stage == "A1":
    expected_base_hash = file_sha256(args.base_checkpoint)
    frozen_actor = load_frozen_teacher_actor(
        args.base_checkpoint,
        device=args.device,
        policy_cfg=deepcopy(train_cfg["policy"]),
    )
validate_teacher_checkpoint(
    args.checkpoint,
    expected_stage=args.stage,
    expected_observation_dim=TEACHER_OBSERVATION_DIM,
    expected_action_dim=TEACHER_ACTION_DIM,
    expected_actor_hidden_dims=TEACHER_HIDDEN_DIMS,
    expected_base_sha256=expected_base_hash,
    require_optimizer=False,
)
```

Create the environment through `parse_env_cfg(TASK_IDS[args.stage], device=args.device, num_envs=args.num_envs)`, set its seed, verify `ManagerBasedRLEnv`, and wrap it with:

```python
wrapper = M1PandaTeacherEnvWrapper(
    env.unwrapped,
    stage=args.stage,
    base_actor=frozen_actor,
    seed=args.seed,
    disturbance_enabled=not args.disable_disturbance,
)
```

Create `OnPolicyRunner(wrapper, deepcopy(train_cfg), log_dir=None, device=env_cfg.sim.device)`, load the current checkpoint with `load_optimizer=False, keep_std=True`, acquire `policy = runner.get_inference_policy(device=env_cfg.sim.device)`, and obtain initial observations from `wrapper.get_observations()`.

Use this bounded/unbounded loop shape:

```python
step = 0
done_count = 0
reset_counts = {name: None for name in TERMINATION_NAMES}
while simulation_app.is_running() and (args.steps == 0 or step < args.steps):
    with torch.inference_mode():
        actions = policy(observations)
        observations, rewards, dones, _ = wrapper.step(actions)
    step += 1
    done_count += int(dones.sum().item())
    update_reset_counts(env, reset_counts)
    if step % args.stats_interval == 0 or (args.steps > 0 and step == args.steps):
        print(
            format_play_stats(
                step=step,
                mean_reward=float(rewards.mean().item()),
                done_count=done_count,
                wrench_b=wrapper.current_wrench_b,
                max_abs_wrench_seen=wrapper.max_abs_wrench_seen,
                reset_counts=reset_counts,
            ),
            flush=True,
        )
```

After the loop call `wrapper.assert_frozen_actor_unchanged()` and print a final mode line containing stage, disturbance enabled/disabled, steps, observation width, action width, and frozen hash for A1. Catch `BaseException`, print traceback, return `1`; in `finally`, close `env` and then `simulation_app`. Return `0` only on success. Never call any learning or manifest-write function.

- [ ] **Step 5: Run focused tests and repair only contract failures**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_play_static.py \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_checkpoint.py \
  tests/test_m1_panda_teacher_train_static.py -q
```

Expected: all tests pass. If failures reveal an unexpected behavior, apply `superpowers:systematic-debugging` before editing implementation.

- [ ] **Step 6: Compile, scan for forbidden writes, and record the task checkpoint**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  scripts/m1_panda_teacher_play.py \
  tests/test_m1_panda_teacher_play_static.py \
  go2_pvcnn/tasks/m1_panda_teacher_wrapper.py
! rg -n 'runner\.learn\(|atomic_write_manifest\(|build_run_manifest\(' \
  scripts/m1_panda_teacher_play.py
```

Expected: both commands exit `0`. Record RED/GREEN counts, preflight ordering, no-write scan, compile result, and Git-unavailable state in `notes/log/2026-08-14-m1-panda-teacher-play-entrypoint.md`; update T400/dashboard/log index.

---

### Task 3: Documentation, full regression, and GPU0 playback smoke

**Files:**

- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py`
- Modify: `docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md`
- Modify: `notes/human/human-02-training-and-entrypoints.md`
- Modify: `notes/ai/ai-02-training-and-entrypoints.md`
- Create: `notes/log/2026-08-14-m1-panda-teacher-play-gpu0-smoke.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`

**Interfaces:**

- Consumes: the final play CLI, the existing A0 checkpoint `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt`, and A1 checkpoint `/home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt`.
- Produces: copyable GPU0 GUI/default-disturbance commands, a zero-disturbance command, static documentation gates, and recorded runtime evidence.

- [ ] **Step 1: Add failing runbook command assertions**

Extend `test_teacher_runbook_contains_complete_formal_resume_and_monitoring_commands` with:

```python
assert source.count("scripts/m1_panda_teacher_play.py") >= 3
assert "--stage A0 --checkpoint /ABS/A0/model_N.pt" in source
assert "--stage A1 --base-checkpoint /ABS/A0/model_N.pt" in source
assert "--checkpoint /ABS/A1/model_M.pt" in source
assert "--disable-disturbance" in source
assert "--device cuda:0" in source
assert "默认开启六维扰动" in source
```

- [ ] **Step 2: Run the documentation test and confirm RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_train_static.py::test_teacher_runbook_contains_complete_formal_resume_and_monitoring_commands -q
```

Expected: fail because the runbook does not yet contain the dedicated play commands.

- [ ] **Step 3: Add exact A0/A1 GPU0 commands and operational warnings**

Append a `## Teacher Play（GPU0）` section to the runbook. Include these commands verbatim:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A0 --checkpoint /ABS/A0/model_N.pt \
  --num-envs 1 --device cuda:0
```

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --checkpoint /ABS/A1/model_M.pt \
  --num-envs 1 --device cuda:0
```

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A1 --base-checkpoint /ABS/A0/model_N.pt \
  --checkpoint /ABS/A1/model_M.pt \
  --num-envs 1 --device cuda:0 --disable-disturbance
```

State explicitly that GUI and stage-specific six-dimensional disturbance are enabled by default, `--steps 0` runs until the window is closed, `--headless --steps N` is for smoke, the zero-disturbance switch does not bypass policy/composers, and current A1 is diagnostic rather than accepted.

Add the dedicated play entrypoint and the same semantic boundary to human-02 and ai-02. Do not replace or relabel generic `m1_play.py`.

- [ ] **Step 4: Run documentation and complete static regression**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_play_static.py \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_checkpoint.py \
  tests/test_m1_panda_teacher_disturbance.py \
  tests/test_m1_panda_teacher_env_cfg_static.py \
  tests/test_m1_panda_teacher_train_static.py \
  tests/test_m1_residual_action.py -q
```

Expected: all tests pass.

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  scripts/m1_panda_teacher_play.py \
  go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_play_static.py \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_train_static.py
! rg -n 'TBD|TODO|FIXME|待定|待确认|PLACEHOLDER' \
  scripts/m1_panda_teacher_play.py \
  go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md \
  notes/human/human-02-training-and-entrypoints.md \
  notes/ai/ai-02-training-and-entrypoints.md
```

Expected: compile and placeholder scan exit `0`.

- [ ] **Step 5: Run bounded GPU0 A0 and A1 default-disturbance smoke**

Resolve the two checkpoint and adjacent manifest files first with `test -f`, then run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 timeout 600 \
  /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A0 \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --num-envs 1 --steps 8 --stats-interval 4 --device cuda:0 --headless
```

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 timeout 600 \
  /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A1 \
  --base-checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt \
  --num-envs 1 --steps 8 --stats-interval 4 --device cuda:0 --headless
```

Expected: both exit `0`; final diagnostics report observation/action `60/16`, disturbance enabled, positive historical max wrench, and A1 frozen hash unchanged.

- [ ] **Step 6: Run the explicit zero-disturbance A1 smoke**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 timeout 600 \
  /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_teacher_play.py \
  --stage A1 \
  --base-checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt \
  --num-envs 1 --steps 8 --stats-interval 4 --device cuda:0 --headless \
  --disable-disturbance
```

Expected: exit `0`; every wrench axis and `max_abs_wrench_seen` are `0`, while A1 still loads and checks the frozen A0 actor.

- [ ] **Step 7: Diagnose runtime failures before changing code**

If any real smoke fails, invoke `superpowers:systematic-debugging`, preserve stdout/stderr, reproduce the smallest failing command, and distinguish code failure from CUDA/Isaac compatibility. Do not silently fall back to CPU and report GPU0 success. After a fix, rerun the focused RED/GREEN test, full static regression, and all affected real smoke commands.

- [ ] **Step 8: Final verification and durable status update**

Invoke `superpowers:verification-before-completion`. Record exact commands, exit codes, test counts, wrench maxima, reset counts, device, checkpoint paths, base file SHA, and frozen actor hash in `notes/log/2026-08-14-m1-panda-teacher-play-gpu0-smoke.md`.

Update T400.5d to complete only if the static suite and required GPU0 smokes pass. If GPU0 is externally incompatible, leave T400.5d open with the exact blocker and verified static/CPU evidence. Update `notes/todo.md` and `notes/log/index.md` in the same patch. Since Git is unavailable, record `Current Work Ref: filesystem working copy` rather than inventing a commit hash.

---

## Plan Self-review

- Spec coverage: Tasks 1–3 cover the default disturbance, explicit zero baseline, strict A0/A1 checkpoint paths, 60/16 wrapper data flow, GUI/unbounded and headless/bounded lifecycles, reset/wrench diagnostics, no-write behavior, documentation, and GPU0 acceptance.
- Scope: no Student, Panda control, grasping, reward, PPO, asset, observation, action, or training-default changes are included.
- Type consistency: `disturbance_enabled` is a strict bool from CLI through wrapper; `current_wrench_b` remains a cloned float tensor shaped `[num_envs, 6]`; reset counts use `int | None` consistently.
- Ordering: A1 base file hash/load and current checkpoint validation occur before runner load and before the first environment step; frozen hash is asserted after playback.
- Placeholder scan: the plan contains no deferred implementation marker; runtime incompatibility has an explicit evidence-preserving branch.
- Execution choice: user already selected single-agent execution, so the implementation uses `superpowers:executing-plans` inline and does not ask again or dispatch subagents.
