# M1 + Panda Phase 6 Transition Reward Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align every 8D residual action with the physical transition reward it causes, then rerun the unchanged Phase 5 and Phase 6 fixed-condition gates through conditional long launch.

**Architecture:** Split the residual runtime into an action-preparation phase and a post-physics reward-finalization phase, connected by one immutable pending transition. The VecEnv wrapper enforces `compute_action -> env.step -> refresh -> compute_transition_reward -> reset(done)`, while all public shapes, reward equations, physical controllers and promotion criteria remain frozen.

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, Isaac Sim 5.1, Isaac Lab ManagerBasedRLEnv, vendored RSL-RL, pytest, GPU0.

## Global Constraints

- Work in `/home/xk/coding/M1` with one agent; preserve unrelated dirty-worktree changes.
- Use `apply_patch` for file edits and TDD RED before production edits.
- Preserve public observation `(103,)`, public action `(8,)`, and private effort `(23,)`.
- Preserve physics/WBC `200 Hz`, Arm MPC `50 Hz`, reward equations/weights, physical limits, safety projection, Phase 5 gates and Phase 6 promotion tolerances.
- Short training remains exactly 100 updates with candidates `u000/u025/u050/u075/u100`.
- Promotion remains nine zero-policy pairs plus fifteen candidate workers at seeds `42/43/44`, 4000 steps each.
- Do not start long training unless the fresh promotion manifest says `accepted=true` and all lineage hashes validate.
- Earlier `short_s42_fixed_condition_v1` artifacts are diagnostic-only after the wrapper SHA changes; use fresh `v2` paths.

---

### Task 1: Lock the two-phase runtime state machine with RED tests

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py`

**Interfaces:**
- Produces: `compute_action(actions: torch.Tensor, physics_step: int) -> torch.Tensor`.
- Produces: `compute_transition_reward() -> tuple[torch.Tensor, dict[str, float]]`.
- Internal pending data: current/previous normalized residuals, predicted wrench, MPC/QP counts, intervention vector, command terminate vector, and expected refreshed physics step.

- [ ] **Step 1: Write failing runtime state-machine tests**

Add tests that use the existing `_PhysicalEnv`, adapter, teacher and planner fixtures:

```python
def test_runtime_requires_refresh_before_transition_reward():
    runtime = _physical_runtime()
    runtime.reset()
    effort = runtime.compute_action(torch.zeros(1, 8), physics_step=0)
    assert effort.shape == (1, 23)
    with pytest.raises(RuntimeError, match="post-step refresh"):
        runtime.compute_transition_reward()
    with pytest.raises(RuntimeError, match="pending transition"):
        runtime.compute_action(torch.zeros(1, 8), physics_step=0)


def test_runtime_advances_residual_history_only_after_finalization():
    runtime = _physical_runtime()
    runtime.reset()
    action = torch.full((1, 8), 0.25)
    runtime.compute_action(action, physics_step=0)
    assert torch.equal(runtime._previous_normalized, torch.zeros(1, 8, dtype=torch.float64))
    runtime.refresh(1)
    reward, metrics = runtime.compute_transition_reward()
    torch.testing.assert_close(runtime._previous_normalized, action.to(torch.float64))
    assert reward.shape == (1,)
    assert metrics["mpc_feasible_rate"] == 1.0
```

Factor the existing physical-runtime fixture body into `_physical_runtime()` without changing its values.

- [ ] **Step 2: Run RED and verify the missing interface is the cause**

Run:

```bash
cd /home/xk/coding/M1
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  -k "requires_refresh or advances_residual" -q
```

Expected: FAIL because `compute_action` and `compute_transition_reward` do not exist.

- [ ] **Step 3: Add an immutable pending-transition record**

Add near the imports:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class _PendingResidualTransition:
    normalized: torch.Tensor
    previous_normalized: torch.Tensor
    predicted_wrench_b: torch.Tensor
    mpc_feasible_count: int
    qp_feasible_count: int
    intervention: torch.Tensor
    command_terminate: torch.Tensor
    mpc_replanned: bool
    expected_refresh_step: int
```

Initialize `self._pending_transition = None` and
`self._last_refresh_physics_step = None`. At the end of `refresh`, set:

```python
self._last_refresh_physics_step = int(physics_step)
```

- [ ] **Step 4: Split action preparation from reward finalization**

Rename the existing `compute` action/controller portion to `compute_action`.
Reject a second preparation before finalization. After effort and command-side
values are computed, store detached clones in `_PendingResidualTransition` and
return only `effort`:

```python
if self._pending_transition is not None:
    raise RuntimeError("cannot prepare action while a pending transition exists")
# existing MPC/controller work remains unchanged
self._pending_transition = _PendingResidualTransition(
    normalized=normalized.clone(),
    previous_normalized=self._previous_normalized.clone(),
    predicted_wrench_b=predicted.clone(),
    mpc_feasible_count=feasible,
    qp_feasible_count=qp,
    intervention=intervention.clone(),
    command_terminate=torch.tensor(
        [bool(command.terminate) for command in commands], dtype=torch.bool
    ),
    mpc_replanned=bool(physics_step % 4 == 0),
    expected_refresh_step=int(physics_step) + 1,
)
return effort
```

Move signal construction, reward calculation, training-diagnostic accumulation
and metric construction into `compute_transition_reward`. Read mount wrench
again after refresh and use the pending prediction/action values:

```python
pending = self._pending_transition
if pending is None:
    raise RuntimeError("transition reward requires a pending transition")
if self._last_refresh_physics_step != pending.expected_refresh_step:
    raise RuntimeError("transition reward requires the matching post-step refresh")
measured = torch.stack(
    [adapter.read_mount_wrench_b() for adapter in self.adapters]
)
hard_failure = torch.tensor(
    [
        float(self.base_contacts[i] or bool(pending.command_terminate[i]))
        for i in range(self.num_envs)
    ],
    dtype=torch.float64,
)
wrench_error_b = measured - pending.predicted_wrench_b
signals = ResidualRewardSignals(
    roll=torch.tensor([state.roll for state in self.states], dtype=torch.float64),
    pitch=torch.tensor([state.pitch for state in self.states], dtype=torch.float64),
    base_height_error=base_height_error,
    support_margin=torch.tensor(
        [0.02 * state.wheel_contact_count for state in self.states],
        dtype=torch.float64,
    ),
    wheel_contact_count=torch.tensor(
        [state.wheel_contact_count for state in self.states], dtype=torch.float64
    ),
    joint_margin=joint_margin.min(dim=1).values,
    hard_failure=hard_failure,
    ee_position_error=torch.linalg.vector_norm(ee_error[:, :3], dim=1),
    ee_orientation_error=torch.linalg.vector_norm(ee_error[:, 3:], dim=1),
    normalized_wrench_error=normalized_wrench_error(
        wrench_error_b, self.controller.wrench_scale
    ),
    wheel_slip=torch.tensor(
        [state.max_lateral_slip for state in self.states], dtype=torch.float64
    ),
    residual=pending.normalized,
    previous_residual=pending.previous_normalized,
    intervention=pending.intervention,
)
reward = compute_residual_reward(signals).total
# accumulate diagnostics here, then commit the transition atomically
self._last_measured = measured.clone()
self._previous_normalized = pending.normalized.clone()
self._pending_transition = None
return reward, dict(self._last_metrics)
```

Do not change reward coefficients or wrench normalization.

- [ ] **Step 5: Make reset reject unsettled transitions**

At the top of `reset`, before rebasing any adapter:

```python
if self._pending_transition is not None:
    raise RuntimeError("cannot reset while a pending transition exists")
```

- [ ] **Step 6: Run focused GREEN**

Run the Step 2 command. Expected: both new tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py
git commit -m "fix: align residual runtime transition reward"
```

---

### Task 2: Enforce post-step ordering at the VecEnv boundary

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py`

**Interfaces:**
- Consumes: Task 1 `compute_action`, `refresh`, and `compute_transition_reward`.
- Produces: RSL-compatible `step(actions) -> (observation, reward, dones, extras)` where reward belongs to the returned transition.

- [ ] **Step 1: Replace the fake runtime with an ordered event recorder and add RED test**

Extend `_FakeEnv` and `_FakeRuntime` with a shared `events` list, and add:

```python
def test_wrapper_finalizes_reward_after_physics_and_before_done_reset():
    events = []
    env = _FakeEnv(events=events)
    runtime = _FakeRuntime(events=events)
    wrapper = M1PandaArmMpcResidualEnvWrapper(env, runtime=runtime)
    wrapper.reset()
    events.clear()

    _, reward, dones, _ = wrapper.step(torch.zeros(2, 8))

    assert events == [
        "compute_action:0",
        "env.step",
        "refresh:1",
        "compute_transition_reward",
        "reset:[1]",
    ]
    assert reward.tolist() == [1.0, 2.0]
    assert dones.tolist() == [False, True]
```

The fake `compute_transition_reward` returns the current fixed reward and
metrics; `compute_action` returns only effort.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  -k "finalizes_reward_after_physics" -q
```

Expected: FAIL because wrapper still calls the old one-phase `compute`.

- [ ] **Step 3: Implement the exact wrapper order**

Replace the body after public-action validation with:

```python
effort = self.runtime.compute_action(actions, self._physics_step)
if not isinstance(effort, torch.Tensor) or effort.shape != (self.num_envs, 23):
    raise RuntimeError("runtime effort must have shape (num_envs, 23)")
if not torch.isfinite(effort).all().item():
    raise RuntimeError("runtime effort must be finite")
_, _, terminated, truncated, extras = self.env.step(
    effort.to(device=self.device, dtype=torch.float32)
)
self.runtime.refresh(self._physics_step + 1)
reward, metrics = self.runtime.compute_transition_reward()
dones = terminated | truncated
if bool(dones.any().item()):
    self.runtime.reset(dones.nonzero(as_tuple=False).flatten().cpu())
self._physics_step += 1
```

Keep observation/extras construction unchanged after this block.

- [ ] **Step 4: Run GREEN and full wrapper tests**

```bash
PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py
git commit -m "fix: settle residual reward after physics step"
```

---

### Task 3: Regression, Phase 5 revalidation, fresh Phase 6 promotion and conditional long

**Files:**
- Modify: `notes/log/2026-08-29-m1-panda-phase6-fixed-condition-implementation.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify if commands/path change: `docs/superpowers/runbooks/2026-08-28-m1-panda-arm-mpc-residual-training.md`

**Interfaces:**
- Consumes: aligned wrapper and unchanged Phase 6 train/promotion entrypoints.
- Produces: fresh `v2` short manifest, 24-worker promotion manifest, and only on acceptance a 3000-update long process.

- [ ] **Step 1: Run focused and regression suites**

```bash
export PYTHONPATH=Go2Pvcnn:Go2Pvcnn/rsl_rl
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PY=/home/xk/miniconda3/envs/go2/bin/python
$PY -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_wrapper.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_reward.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_entrypoints_static.py -q
$PY -m pytest \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_controller.py \
  Go2Pvcnn/tests/test_m1_panda_wbc_teacher.py \
  Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py -q
$PY -m compileall -q \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_arm_mpc_residual_wrapper.py
git diff --check
```

Expected: all tests and checks exit `0`.

- [ ] **Step 2: Run unchanged Phase 5 GPU0 seed-42 regression**

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_probe.py \
  --device cuda:0 --num_envs 1 --steps 4000 --seeds 42 --headless
```

Expected: summary `accepted=true`, finite, MPC/QP feasible rate `1.0`, four
wheel contacts, and all existing Phase 5 hard gates pass. Stop before training
if this fails.

- [ ] **Step 3: Run fresh exact 100-update short on GPU0**

```bash
SHORT=Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_fixed_condition_v2
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage short --device cuda:0 --num_envs 8 --max_iterations 100 --seed 42 \
  --headless --run_dir "$SHORT"
```

Expected: `run_manifest.json` has `status=safe_complete`, exactly five
candidate checkpoints, `accepted=false`, and `promotion_required=true`.

- [ ] **Step 4: Run all 24 independent fixed-condition workers**

```bash
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_promote.py \
  --short_manifest "$SHORT/run_manifest.json" --device cuda:0 --headless
```

Expected: nine zero-vs-zero calibration JSON files, fifteen candidate JSON
files, and an atomic `promotion_manifest.json`. Do not modify tolerances or
rerun only favorable seeds.

- [ ] **Step 5: Branch strictly on the promotion manifest**

Inspect with:

```bash
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["status"], d["accepted"], d.get("best_checkpoint"))' \
  "$SHORT/promotion_manifest.json"
```

If `accepted=false`, record exact rejection evidence and do not launch long.
If and only if `accepted=true`, launch:

```bash
LONG=Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_fixed_condition_v2
CUDA_VISIBLE_DEVICES=0 $PY Go2Pvcnn/scripts/m1_panda_arm_mpc_residual_train.py \
  --stage long --device cuda:0 --num_envs 8 --max_iterations 3000 --seed 42 \
  --promotion_manifest "$SHORT/promotion_manifest.json" \
  --headless --run_dir "$LONG"
```

Monitor process health, manifest status, finite PPO/KL/LR/std diagnostics,
hard safety metrics and checkpoint progression until normal completion or a
declared guard stop. Do not call a running process accepted training.

- [ ] **Step 6: Align notes and evidence**

Append exact commands, refs, hashes, test counts, Phase 5 metrics, short
manifest fields, all promotion decisions, and long status to the existing
Phase 6 implementation log. Update T400 current state and the log index using
repository-relative links. Preserve rejected `v1` evidence as diagnostic
history.

- [ ] **Step 7: Verification-before-completion and evidence commit**

```bash
git diff --check
git status --short
git add notes/log/2026-08-29-m1-panda-phase6-fixed-condition-implementation.md \
  notes/log/index.md notes/todo/T400-m1-panda-force-aware-teacher-student.md \
  docs/superpowers/runbooks/2026-08-28-m1-panda-arm-mpc-residual-training.md
git commit -m "test: record aligned phase6 promotion gates"
```

Commit only tracked code/tests/docs/evidence; never commit generated
checkpoints, TensorBoard files or runtime JSON artifacts.

## Completion rule

The alignment change is complete only after RED/GREEN and regression evidence.
The active Phase 6 objective is complete only when the fresh 100-update short
run and all 24 workers are proven complete, promotion is `accepted=true`, the
lineage-validated 3000-update long run is finished/accepted under its declared
guard contract, and notes match the authoritative manifests. If promotion is
rejected, long must remain unstarted and the objective remains open for a new
evidence-based design iteration.
