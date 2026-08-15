# M1 + Panda A1 Force-Balance Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover A1 stationary balance under full-scale `20 N / 5 Nm` six-axis disturbances without overwriting the original run, and accept only a checkpoint that passes the approved three-seed survival gates.

**Architecture:** First repair the vendored ActorCritic noise-standard-deviation contract while preserving old scalar-`std` checkpoint keys. Then add restorable disturbance progress, strict JSON evaluation, deterministic candidate ranking, and an isolated fork-training path with optimizer reset. GPU0 execution proceeds in 500-iteration blocks, each followed by the same full-scale three-seed evaluation.

**Tech Stack:** Python 3.11, PyTorch, vendored RSL-RL PPO, Isaac Lab/Isaac Sim 5.1, Gymnasium, pytest, TensorBoard JSON/event artifacts.

## Global Constraints

- Execute with one agent only; do not dispatch subagents.
- Preserve the original A0 run and `a1_dynamic_force_balance_gpu0_20260814_1848` directory byte-for-byte.
- Keep Teacher observation/action dimensions at `60/16` and the combined asset at one articulation with 25 DOF.
- Keep the frozen A0 actor SHA-256 at `a7fd58c2753130128f698097eef3159f7f007081f9937f34990a610d8a992457`.
- Keep reward weights, residual composer limits, disturbance distributions, Student, grasping, and Panda asset placement unchanged.
- Full-scale evaluation is 64 environments × 2000 steps × seeds `42,43,44`, with A1 limits `20 N / 5 Nm` and modes `0.50/0.30/0.20`.
- A valid full-scale row must use curriculum scale `1.0`, observe at least `19 N` on every force axis and `4.75 Nm` on every torque axis, and contain only finite values.
- Final acceptance requires timeout survival `>=0.80`, base-contact rate `<=0.10`, and bad-orientation rate `<=0.10` after aggregating all three seeds.
- Recovery uses scalar action std, effective std floor `0.001`, `entropy_coef=0`, reset optimizer, and initial learning rate `1e-4`.
- Train on GPU0 in 500-iteration blocks and save every 100 iterations.
- Existing dirty-worktree changes are preserved. Before every commit, inspect `git diff -- <listed files>`; never stage unrelated paths or overwrite user changes.

---

### Task 1: Correct ActorCritic noise semantics and diagnostics

**Files:**
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py:15-131`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py:268-275`
- Modify: `Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py:12-35`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py:20-70`

**Interfaces:**
- Consumes: existing `ActorCritic(num_actor_obs, num_critic_obs, num_actions, ...)` and scalar-key checkpoints containing `model_state_dict["std"]`.
- Produces: `ActorCritic.noise_parameter: Tensor`, `ActorCritic.effective_action_std: Tensor`, and mode-aware `clip_std(min, max)` where bounds are expressed in effective-standard-deviation units.

- [ ] **Step 1: Write failing scalar/log/std-floor tests**

```python
import math

import pytest
import torch

from rsl_rl.modules import ActorCritic


def _actor(mode: str, init: float = 0.01) -> ActorCritic:
    return ActorCritic(
        3,
        3,
        2,
        actor_hidden_dims=[4, 4],
        critic_hidden_dims=[4, 4],
        init_noise_std=init,
        noise_std_type=mode,
    )


def test_scalar_std_is_used_directly_and_keeps_legacy_state_key():
    actor = _actor("scalar", 0.01)
    actor.update_distribution(torch.zeros(5, 3))
    assert set(name for name in actor.state_dict() if "std" in name) == {"std"}
    assert torch.allclose(actor.action_std, torch.full((5, 2), 0.01))


def test_log_std_is_exponentiated_and_has_log_state_key():
    actor = _actor("log", 0.01)
    actor.update_distribution(torch.zeros(5, 3))
    assert set(name for name in actor.state_dict() if "std" in name) == {"log_std"}
    assert torch.allclose(actor.action_std, torch.full((5, 2), 0.01))


def test_clip_std_uses_effective_units_for_both_modes():
    for mode in ("scalar", "log"):
        actor = _actor(mode, 1.0e-8)
        actor.clip_std(min=0.001)
        assert torch.allclose(actor.effective_action_std, torch.full((2,), 0.001))


def test_unknown_noise_mode_fails():
    with pytest.raises(ValueError, match="noise_std_type"):
        _actor("softplus")
```

- [ ] **Step 2: Run the new test and confirm RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_panda_teacher_noise_std.py -q
```

Expected: failures because `noise_std_type` is ignored and scalar `0.01` currently produces an effective std near `0.698`.

- [ ] **Step 3: Implement explicit scalar/log semantics**

Replace the current noise initialization, distribution update, and clipping logic with:

```python
self.noise_std_type = noise_std_type
if noise_std_type == "scalar":
    self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
elif noise_std_type == "log":
    self.log_std = nn.Parameter(
        torch.log(init_noise_std * torch.ones(num_actions))
    )
else:
    raise ValueError(
        f"noise_std_type must be 'scalar' or 'log', got {noise_std_type!r}"
    )

@property
def noise_parameter(self):
    return self.std if self.noise_std_type == "scalar" else self.log_std

@property
def effective_action_std(self):
    if self.noise_std_type == "scalar":
        return self.std
    return torch.exp(self.log_std)

def update_distribution(self, observations):
    mean = self.actor(observations)
    std = self.effective_action_std.expand_as(mean)
    self.distribution = Normal(mean, std)

@torch.no_grad()
def clip_std(self, min=None, max=None):
    if self.noise_std_type == "scalar":
        self.std.copy_(self.std.clip(min=min, max=max))
        return
    log_min = None if min is None else math.log(float(min))
    log_max = None if max is None else math.log(float(max))
    self.log_std.copy_(self.log_std.clip(min=log_min, max=log_max))
```

Add `noise_std_type="scalar"` to the constructor signature and import `math`. Preserve actor/critic layer names so legacy `std` state dicts still strict-load in scalar mode.
Also replace the existing method assignment with the valid call `Normal.set_default_validate_args(False)`.

- [ ] **Step 4: Make Teacher config scalar with an effective floor**

Change the Teacher policy/algorithm entries to:

```python
"algorithm": {
    # existing entries stay unchanged
    "entropy_coef": 0.0,
    "clip_min_std": 0.001,
},
"policy": {
    "class_name": "ActorCritic",
    "init_noise_std": 0.01,
    "noise_std_type": "scalar",
    # existing hidden dimensions and activation stay unchanged
},
```

Remove the unsupported `state_dependent_std` key. Update the static config test to assert `noise_std_type == "scalar"`, `"state_dependent_std" not in policy`, `clip_min_std == 0.001`, and `entropy_coef == 0.0`.

- [ ] **Step 5: Log raw and effective standard deviation separately**

Replace the single runner scalar with:

```python
mean_noise_parameter = self.alg.actor_critic.noise_parameter.mean()
mean_action_std = self.alg.actor_critic.effective_action_std.mean()
self.writer.add_scalar(
    "Policy/mean_noise_parameter", mean_noise_parameter.item(), locs["it"]
)
self.writer.add_scalar(
    "Policy/mean_action_std", mean_action_std.item(), locs["it"]
)
self.writer.add_scalar(
    "Policy/mean_noise_std", mean_action_std.item(), locs["it"]
)
```

Keep the legacy `Policy/mean_noise_std` tag, but make it report the effective value.

- [ ] **Step 6: Run focused and checkpoint compatibility tests**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_noise_std.py \
  tests/test_m1_panda_teacher_checkpoint.py \
  tests/test_m1_panda_teacher_train_static.py -q
```

Expected: all selected tests pass; the existing A0/A1 scalar-`std` fixtures remain strict-compatible.

- [ ] **Step 7: Commit the isolated noise-contract change**

```bash
git diff -- Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py \
  Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py
git add Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic.py \
  Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py \
  Go2Pvcnn/agent/m1_panda_teacher_train_cfg.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_noise_std.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py
git commit -m "fix: restore teacher action noise semantics"
```

### Task 2: Restore disturbance progress and axis-wise wrench evidence

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py:218-286`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py:35-112,245-330`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_disturbance.py:140-195`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py`

**Interfaces:**
- Consumes: `M1PandaDisturbanceCfg.curriculum_steps` and wrapper stage defaults.
- Produces: scheduler keyword `initial_global_step: int = 0`, properties `global_step`, `curriculum_scale`; wrapper keyword `initial_curriculum_step: int = 0`, properties `curriculum_scale`, `global_disturbance_step`, and `axis_abs_wrench_seen: Tensor[6]`.

- [ ] **Step 1: Add RED scheduler progress tests**

```python
def test_scheduler_can_start_at_full_curriculum_without_advancing():
    cfg = stage_disturbance_cfg("A1")
    scheduler = M1PandaDisturbanceScheduler(
        cfg, 2, "cpu", 0.02, seed=3, initial_global_step=cfg.curriculum_steps
    )
    assert scheduler.global_step == 75_000
    assert scheduler.curriculum_scale == pytest.approx(1.0)
    first = scheduler.advance()
    assert torch.all(first[:, :3].abs() <= 20.0)
    assert torch.all(first[:, 3:].abs() <= 5.0)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_scheduler_rejects_invalid_initial_global_step(value):
    with pytest.raises((TypeError, ValueError), match="initial_global_step"):
        M1PandaDisturbanceScheduler(
            stage_disturbance_cfg("A1"),
            1,
            "cpu",
            0.02,
            seed=3,
            initial_global_step=value,
        )
```

- [ ] **Step 2: Run scheduler tests and confirm RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_disturbance.py -q
```

Expected: failures for the missing keyword/property.

- [ ] **Step 3: Implement validated initial progress**

Add to `M1PandaDisturbanceScheduler.__init__`:

```python
initial_global_step: int = 0,
```

Validate and assign it before creating samples:

```python
if (
    not isinstance(initial_global_step, int)
    or isinstance(initial_global_step, bool)
    or initial_global_step < 0
):
    raise ValueError("initial_global_step must be a nonnegative integer")
self._global_step = initial_global_step

@property
def global_step(self) -> int:
    return self._global_step
```

Do not clamp the stored step; `curriculum_scale` already clamps progress at `1.0`.

- [ ] **Step 4: Add RED wrapper progress and history tests**

Extend the existing fake wrapper environment tests to assert:

```python
wrapper = M1PandaTeacherEnvWrapper(
    env,
    stage="A1",
    base_actor=frozen_actor,
    seed=7,
    initial_curriculum_step=75_000,
)
assert wrapper.curriculum_scale == pytest.approx(1.0)
assert wrapper.global_disturbance_step == 75_000
wrapper.step(torch.zeros(wrapper.num_envs, 16))
assert wrapper.axis_abs_wrench_seen.shape == (6,)
assert torch.all(wrapper.axis_abs_wrench_seen >= 0)
```

- [ ] **Step 5: Implement wrapper pass-through and axis maxima**

Add the constructor keyword and scheduler argument:

```python
initial_curriculum_step: int = 0,
# ...
initial_global_step=initial_curriculum_step,
```

Initialize and update history in `_apply_wrench`:

```python
self._axis_abs_wrench_seen = torch.zeros(6, device=self.device)
# in _apply_wrench
self._axis_abs_wrench_seen = torch.maximum(
    self._axis_abs_wrench_seen, wrench_b.abs().amax(dim=0)
)
```

Expose clone/value properties:

```python
@property
def axis_abs_wrench_seen(self) -> torch.Tensor:
    return self._axis_abs_wrench_seen.clone()

@property
def curriculum_scale(self) -> float:
    return self._disturbance.curriculum_scale

@property
def global_disturbance_step(self) -> int:
    return self._disturbance.global_step
```

Whole-wrapper reset must not clear these run-level diagnostics; constructing a new wrapper starts them at zero.

- [ ] **Step 6: Run scheduler/wrapper regression**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_disturbance.py \
  tests/test_m1_panda_teacher_wrapper.py -q
```

Expected: all tests pass, including selective environment reset tests.

- [ ] **Step 7: Commit disturbance progress support**

```bash
git diff -- Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_disturbance.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_disturbance.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_wrapper.py
git commit -m "feat: restore teacher disturbance progress"
```

### Task 3: Build pure full-scale evaluation metrics and ranking

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_evaluation.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_evaluation.py`

**Interfaces:**
- Consumes: per-step reward tensor, per-step termination counts, final axis maxima, checkpoint/base hashes, and curriculum scale.
- Produces: `TeacherEvaluationAccumulator`, `validate_full_scale_summary(summary)`, and `aggregate_candidate_summaries(summaries)` returning a JSON-compatible aggregate with deterministic rank key.

- [ ] **Step 1: Write RED accumulator and gate tests**

```python
import pytest
import torch

from go2_pvcnn.tasks.m1_panda_teacher_evaluation import (
    TeacherEvaluationAccumulator,
    aggregate_candidate_summaries,
    validate_full_scale_summary,
)


def test_accumulator_reports_rates_and_reward_over_all_env_steps():
    acc = TeacherEvaluationAccumulator(num_envs=2)
    acc.update(
        rewards=torch.tensor([1.0, 3.0]),
        termination_counts={"time_out": 1, "base_contact": 0, "bad_orientation": 0},
    )
    summary = acc.finalize(
        checkpoint="model_1.pt",
        checkpoint_sha256="abc",
        base_checkpoint_sha256="base",
        seed=42,
        steps=1,
        curriculum_scale=1.0,
        axis_abs_wrench_seen=[19.5, 19.5, 19.5, 4.8, 4.8, 4.8],
        frozen_actor_sha256="frozen",
    )
    assert summary["mean_reward"] == pytest.approx(2.0)
    assert summary["termination_counts"]["time_out"] == 1
    assert summary["termination_rates"]["time_out"] == pytest.approx(1.0)


def test_full_scale_gate_rejects_an_underexcited_axis():
    summary = {
        "curriculum_scale": 1.0,
        "axis_abs_wrench_seen": [19.5, 19.5, 18.9, 4.8, 4.8, 4.8],
        "finite": True,
    }
    with pytest.raises(ValueError, match="force axis"):
        validate_full_scale_summary(summary)


def _summary(checkpoint, seed, *, timeout, contact, bad, reward):
    total = timeout + contact + bad
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": f"sha-{checkpoint}",
        "base_checkpoint_sha256": "base-sha",
        "frozen_actor_sha256": "frozen-sha",
        "seed": seed,
        "num_envs": 64,
        "steps": 2000,
        "curriculum_scale": 1.0,
        "axis_abs_wrench_seen": [19.5, 19.5, 19.5, 4.8, 4.8, 4.8],
        "termination_counts": {
            "time_out": timeout,
            "base_contact": contact,
            "bad_orientation": bad,
        },
        "termination_rates": {
            "time_out": timeout / total,
            "base_contact": contact / total,
            "bad_orientation": bad / total,
        },
        "reward_sum": reward * 64 * 2000,
        "reward_count": 64 * 2000,
        "mean_reward": reward,
        "finite": True,
    }


def test_candidate_aggregation_uses_survival_contact_orientation_reward_order():
    summaries = [
        _summary("model_a.pt", 42, timeout=80, contact=10, bad=10, reward=4.0),
        _summary("model_a.pt", 43, timeout=80, contact=10, bad=10, reward=4.0),
        _summary("model_a.pt", 44, timeout=80, contact=10, bad=10, reward=4.0),
    ]
    aggregate = aggregate_candidate_summaries(summaries)
    assert aggregate["timeout_survival_rate"] == pytest.approx(0.8)
    assert aggregate["accepted"] is True
```

- [ ] **Step 2: Run evaluation tests and confirm import RED**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_evaluation.py -q
```

Expected: collection fails because `m1_panda_teacher_evaluation.py` does not exist.

- [ ] **Step 3: Implement the pure accumulator**

Implement a small class with this public shape:

```python
TERMINATION_NAMES = ("bad_orientation", "base_contact", "time_out")


class TeacherEvaluationAccumulator:
    def __init__(self, *, num_envs: int) -> None:
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs = num_envs
        self.reward_sum = 0.0
        self.reward_count = 0
        self.termination_counts = {name: 0 for name in TERMINATION_NAMES}

    def update(self, *, rewards: torch.Tensor, termination_counts: dict[str, int]) -> None:
        if tuple(rewards.shape) != (self.num_envs,) or not bool(torch.isfinite(rewards).all()):
            raise ValueError("rewards must be finite with shape (num_envs,)")
        self.reward_sum += float(rewards.sum().item())
        self.reward_count += rewards.numel()
        for name in TERMINATION_NAMES:
            value = termination_counts[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"termination count {name} must be nonnegative")
            self.termination_counts[name] += value

    def finalize(self, **metadata) -> dict[str, object]:
        total = sum(self.termination_counts.values())
        rates = {
            name: (count / total if total else 0.0)
            for name, count in self.termination_counts.items()
        }
        return {
            **metadata,
            "num_envs": self.num_envs,
            "reward_sum": self.reward_sum,
            "reward_count": self.reward_count,
            "mean_reward": self.reward_sum / self.reward_count,
            "termination_counts": dict(self.termination_counts),
            "termination_rates": rates,
            "finite": True,
        }
```

Use strict finite/type validation in `finalize`; do not rely on JSON serialization to reject NaN.

- [ ] **Step 4: Implement full-scale validation and aggregate ranking**

`validate_full_scale_summary` must require `curriculum_scale == 1.0`, three force maxima `>=19.0`, three torque maxima `>=4.75`, `finite is True`, and nonempty hashes. `aggregate_candidate_summaries` must require exactly seeds `{42,43,44}` for one checkpoint SHA, sum termination counts, average reward weighted by each row's `reward_count`, and return:

```python
{
    "checkpoint": checkpoint,
    "checkpoint_sha256": checkpoint_sha256,
    "seeds": [42, 43, 44],
    "timeout_survival_rate": timeout / total,
    "base_contact_rate": base_contact / total,
    "bad_orientation_rate": bad_orientation / total,
    "mean_reward": weighted_reward,
    "accepted": (
        timeout / total >= 0.80
        and base_contact / total <= 0.10
        and bad_orientation / total <= 0.10
    ),
    "rank_key": [
        timeout / total,
        -(base_contact / total),
        -(bad_orientation / total),
        weighted_reward,
    ],
}
```

- [ ] **Step 5: Run pure evaluation tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_evaluation.py -q
```

Expected: all tests pass without importing Isaac Sim.

- [ ] **Step 6: Commit pure evaluation contracts**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_evaluation.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_evaluation.py
git commit -m "feat: add teacher full-scale evaluation metrics"
```

### Task 4: Add strict full-scale Play output and candidate sweep

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_panda_teacher_play.py:31-284`
- Create: `Go2Pvcnn/scripts/m1_panda_teacher_eval_sweep.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_play_static.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_teacher_eval_sweep.py`

**Interfaces:**
- Consumes: Task 2 wrapper diagnostics and Task 3 evaluation helpers.
- Produces: Play flags `--full-scale-disturbance`, `--summary-json`; sweep flags `--base-checkpoint`, repeated `--checkpoint`, `--output-dir`, `--device`, `--num-envs`, `--steps`, repeated `--seed`; artifacts `row-<checkpoint>-seed-<seed>.json` and `ranking.json`.

- [ ] **Step 1: Add RED Play CLI and JSON-output tests**

Extend `test_m1_panda_teacher_play_static.py`:

```python
def test_full_scale_play_requires_disturbance_and_finite_steps(tmp_path):
    module = _load_script()
    args = SimpleNamespace(
        stage="A1",
        checkpoint=Path("model.pt"),
        base_checkpoint=Path("base.pt"),
        num_envs=64,
        steps=2000,
        stats_interval=100,
        disable_disturbance=False,
        full_scale_disturbance=True,
        summary_json=tmp_path / "row.json",
    )
    module.validate_cli_contract(args)
    args.disable_disturbance = True
    with pytest.raises(ValueError, match="full-scale"):
        module.validate_cli_contract(args)
```

Also assert the script passes `initial_curriculum_step`, writes via `atomic_write_manifest`, and never calls `runner.learn`.

- [ ] **Step 2: Run Play tests and confirm RED**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_play_static.py -q
```

Expected: failures for missing arguments and integration strings.

- [ ] **Step 3: Implement full-scale Play mode**

Add parser arguments:

```python
parser.add_argument("--full-scale-disturbance", action="store_true")
parser.add_argument("--summary-json", type=Path, default=None)
```

Reject full-scale with disabled disturbance, zero steps, or A0. Compute initial progress only for strict A1 evaluation:

```python
disturbance_cfg = stage_disturbance_cfg(args.stage)
initial_curriculum_step = (
    disturbance_cfg.curriculum_steps if args.full_scale_disturbance else 0
)
```

Pass it to `M1PandaTeacherEnvWrapper`. Use `_termination_terms` each step to build delta counts, update `TeacherEvaluationAccumulator` with all rewards, and finalize using `wrapper.axis_abs_wrench_seen`, `wrapper.curriculum_scale`, checkpoint/base SHA, seed, steps, and frozen hash. Validate then atomically write `summary_json`.

- [ ] **Step 4: Write RED sweep parsing/ranking tests**

The new test must monkeypatch `subprocess.run` and create row JSON files, then assert:

```python
result = module.rank_completed_rows(row_paths)
assert result["winner"]["checkpoint"] == "model_2700.pt"
assert result["candidates"][0]["rank_key"] >= result["candidates"][1]["rank_key"]
```

Also test duplicate checkpoint paths, missing seeds, child nonzero exit, and pre-existing output directory rejection.

- [ ] **Step 5: Implement the sweep orchestrator**

The script must:

1. Resolve and hash all checkpoints before launching any child.
2. Create a fresh output directory with `exist_ok=False`.
3. Run one Play child per checkpoint/seed with exact flags:

```python
command = [
    sys.executable,
    str(PLAY_SCRIPT),
    "--stage", "A1",
    "--base-checkpoint", str(base_checkpoint),
    "--checkpoint", str(checkpoint),
    "--num-envs", str(num_envs),
    "--seed", str(seed),
    "--steps", str(steps),
    "--stats-interval", str(steps),
    "--full-scale-disturbance",
    "--summary-json", str(row_path),
    "--device", device,
    "--headless",
]
```

4. Require return code `0` and a valid row JSON.
5. Aggregate exactly three rows per candidate, sort descending by tuple(`rank_key`), and atomically write `ranking.json`.
6. Print one final JSON line containing `status`, `winner`, and artifact path.

- [ ] **Step 6: Run Play/sweep/evaluation tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_evaluation.py \
  tests/test_m1_panda_teacher_play_static.py \
  tests/test_m1_panda_teacher_eval_sweep.py -q
```

Expected: all tests pass and no Isaac application starts.

- [ ] **Step 7: Commit strict evaluation entrypoints**

```bash
git diff -- Go2Pvcnn/scripts/m1_panda_teacher_play.py \
  Go2Pvcnn/scripts/m1_panda_teacher_eval_sweep.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_play_static.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_eval_sweep.py
git add Go2Pvcnn/scripts/m1_panda_teacher_play.py \
  Go2Pvcnn/scripts/m1_panda_teacher_eval_sweep.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_play_static.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_eval_sweep.py
git commit -m "feat: add strict teacher checkpoint sweep"
```

### Task 5: Add isolated recovery fork and manifest lineage

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py:110-355`
- Modify: `Go2Pvcnn/scripts/m1_panda_teacher_train.py:31-297`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_checkpoint.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py`

**Interfaces:**
- Consumes: strict source checkpoint/manifest, same A0 base SHA, Task 1 scalar std, Task 2 initial curriculum step.
- Produces: `checkpoint_iteration(path) -> int`, training flag `--fork-checkpoint`, new recovery run directory, and additive manifest lineage fields.

- [ ] **Step 1: Add RED checkpoint iteration and recovery-manifest tests**

```python
def test_checkpoint_iteration_reads_nonnegative_rsl_iter(tmp_path):
    path = _write_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=False)
    payload["iter"] = 2700
    torch.save(payload, path)
    assert checkpoint_iteration(path) == 2700


def test_recovery_manifest_records_source_and_noise_contract(tmp_path):
    manifest = build_run_manifest(
        stage="A1",
        task_id="Isaac-M1-Panda-Teacher-A1-v0",
        seed=42,
        composer_cfg=M1ResidualActionComposerCfg(),
        disturbance_cfg=stage_disturbance_cfg("A1"),
        base_checkpoint=base,
        frozen_actor=frozen,
        recovery_source_checkpoint=source,
        recovery_source_iteration=2700,
        initial_curriculum_step=64_800,
        optimizer_reset=True,
        recovery_learning_rate=1.0e-4,
        noise_std_mode="scalar",
        minimum_effective_std=0.001,
    )
    assert manifest["recovery_source_iteration"] == 2700
    assert manifest["initial_curriculum_step"] == 64_800
    assert manifest["optimizer_reset"] is True
```

- [ ] **Step 2: Run checkpoint tests and confirm RED**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_checkpoint.py -q
```

Expected: failures for missing helper/manifest keywords.

- [ ] **Step 3: Implement iteration validation and additive manifest fields**

`checkpoint_iteration` must load the trusted local checkpoint through `_load_checkpoint`, require an integer `iter >= 0`, and reject bool. Extend `build_run_manifest` with optional recovery arguments; require either all recovery fields or none, require recovery only for A1, and record source absolute path/SHA without changing schema version 1 so existing strict Play validation remains compatible.

The manifest recovery block must contain:

```python
{
    "recovery_source_checkpoint": str(source_path),
    "recovery_source_checkpoint_sha256": file_sha256(source_path),
    "recovery_source_iteration": recovery_source_iteration,
    "optimizer_reset": True,
    "recovery_learning_rate": 1.0e-4,
    "noise_std_mode": "scalar",
    "minimum_effective_std": 0.001,
    "initial_curriculum_step": initial_curriculum_step,
    "initial_curriculum_scale": stage_disturbance_cfg("A1").curriculum_start_scale
        + 0.75 * min(initial_curriculum_step / 75_000, 1.0),
}
```

- [ ] **Step 4: Add RED fork CLI/directory/optimizer tests**

Extend train static tests with `SimpleNamespace` cases proving:

- `--fork-checkpoint` is A1-only;
- fork requires `--run_name` and forbids `--resume-checkpoint`/`--reset-optimizer`;
- fork creates a fresh stage-scoped directory;
- ordinary resume still reuses its own directory;
- `recovery_initial_curriculum_step(2700, 24, 75_000) == 64_800`;
- source file SHA is unchanged after directory resolution;
- source iteration is advanced before learning;
- fork calls `runner.load(..., load_optimizer=False, keep_std=True)` and clips to `0.001` before `learn`.

- [ ] **Step 5: Implement fork training flow**

Add:

```python
parser.add_argument("--fork-checkpoint", type=Path, default=None)
```

Use mutually exclusive control paths:

```python
def recovery_initial_curriculum_step(
    source_iteration: int, num_steps_per_env: int, curriculum_steps: int
) -> int:
    return min(source_iteration * num_steps_per_env, curriculum_steps)
```

For fork:

1. Validate the source as stage A1 with matching base SHA and `require_optimizer=False`.
2. Compute source iteration and initial curriculum step before constructing the wrapper.
3. Set `train_cfg["algorithm"]["learning_rate"] = 1.0e-4`.
4. Create a new `run_name` directory.
5. Pass initial curriculum progress to the wrapper.
6. Build the recovery manifest.
7. Construct the runner, load source with `load_optimizer=False, keep_std=True`, advance iteration, then call `runner.alg.actor_critic.clip_std(min=0.001)` before `runner.learn`.

For ordinary resume, restore initial curriculum progress from its checkpoint iteration as well, but keep its existing optimizer behavior. Never allow a fork child to resolve to the source parent.

- [ ] **Step 6: Record explicit nullable completion and best-evaluation fields**

At recovery start, add JSON fields with explicit meanings:

```python
manifest.update({
    "evaluation_artifacts": [],
    "best_checkpoint": None,
    "best_metrics": None,
    "stop_reason": None,
})
```

At normal block completion set `stop_reason="block_completed_pending_evaluation"`; the external evaluation step updates best fields only after valid `ranking.json` evidence exists.

- [ ] **Step 7: Run checkpoint/train/wrapper tests**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_checkpoint.py \
  tests/test_m1_panda_teacher_train_static.py \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_noise_std.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit recovery fork support**

```bash
git diff -- Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py \
  Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_checkpoint.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py \
  Go2Pvcnn/scripts/m1_panda_teacher_train.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_checkpoint.py \
  Go2Pvcnn/tests/test_m1_panda_teacher_train_static.py
git commit -m "feat: add isolated A1 recovery forks"
```

### Task 6: Complete documentation, regression, and real smoke gates

**Files:**
- Modify: `docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/todo.md`
- Create: `notes/log/2026-08-15-m1-panda-a1-recovery-implementation.md`
- Modify: `notes/log/index.md`

**Interfaces:**
- Consumes: Tasks 1-5 CLIs and artifacts.
- Produces: exact full-scale sweep/fork/block/resume commands and verified implementation record.

- [ ] **Step 1: Add RED runbook assertions**

Extend `test_teacher_runbook_contains_complete_formal_resume_and_monitoring_commands` to require:

```python
assert "--full-scale-disturbance" in source
assert "scripts/m1_panda_teacher_eval_sweep.py" in source
assert '--fork-checkpoint "$RECOVERY_WINNER"' in source
assert "--max_iterations 500" in source
assert "Policy/mean_action_std" in source
assert "timeout survival >= 0.80" in source
```

- [ ] **Step 2: Write exact recovery commands into the runbook**

Document the candidate sweep:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_teacher_eval_sweep.py \
  --base-checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_2700.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_3800.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_4500.pt \
  --checkpoint /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt \
  --seed 42 --seed 43 --seed 44 --num-envs 64 --steps 2000 \
  --output-dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/eval/a1_fullscale_candidates_20260815 \
  --device cuda:0
```

Document fork and 500-iteration resume commands, TensorBoard tags, acceptance/rollback rules, and the prohibition on changing the zero-clearance asset during recovery.

- [ ] **Step 3: Run full static regression and compilation**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=$PWD:$PWD/rsl_rl /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_panda_teacher_noise_std.py \
  tests/test_m1_panda_teacher_disturbance.py \
  tests/test_m1_panda_teacher_checkpoint.py \
  tests/test_m1_panda_teacher_wrapper.py \
  tests/test_m1_panda_teacher_train_static.py \
  tests/test_m1_panda_teacher_play_static.py \
  tests/test_m1_panda_teacher_evaluation.py \
  tests/test_m1_panda_teacher_eval_sweep.py -q
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
  rsl_rl/rsl_rl/modules/actor_critic.py \
  go2_pvcnn/tasks/m1_panda_teacher.py \
  go2_pvcnn/tasks/m1_panda_teacher_wrapper.py \
  go2_pvcnn/tasks/m1_panda_teacher_checkpoint.py \
  go2_pvcnn/tasks/m1_panda_teacher_evaluation.py \
  scripts/m1_panda_teacher_train.py \
  scripts/m1_panda_teacher_play.py \
  scripts/m1_panda_teacher_eval_sweep.py
```

Expected: all tests pass and compilation exits `0`.

- [ ] **Step 4: Run a real GPU0 full-scale Play smoke**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_teacher_play.py \
  --stage A1 \
  --base-checkpoint logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --checkpoint logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_2700.pt \
  --num-envs 64 --seed 42 --steps 500 --stats-interval 500 \
  --full-scale-disturbance \
  --summary-json /tmp/m1-panda-a1-fullscale-smoke.json \
  --device cuda:0 --headless
```

Expected: exit `0`, observation/action `60/16`, curriculum scale `1.0`, finite JSON, frozen hash unchanged. The 500-step smoke is not required to meet the 2000-step force-axis or survival acceptance gates.

- [ ] **Step 5: Run a real GPU0 one-iteration fork smoke**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_teacher_train.py \
  --stage A1 \
  --base-checkpoint logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --fork-checkpoint logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_2700.pt \
  --run_name a1_recovery_fork_smoke_20260815 \
  --num_envs 8 --max_iterations 1 --save-interval 1 \
  --device cuda:0 --headless
```

Expected: a new recovery directory, source SHA unchanged, final checkpoint one iteration after source, optimizer-reset/noise/curriculum lineage in manifest, nonzero wrench, and frozen hash unchanged.

- [ ] **Step 6: Record verified evidence and commit docs**

Write actual test counts, smoke exit codes, hashes, paths, effective std, and curriculum scale to the implementation log. Then:

```bash
git diff --check
git add docs/superpowers/runbooks/2026-08-14-m1-panda-teacher-a0-a1-training.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md \
  notes/todo.md notes/log/index.md \
  notes/log/2026-08-15-m1-panda-a1-recovery-implementation.md
git commit -m "docs: add A1 recovery operations"
```

### Task 7: Execute GPU0 selection and monitored recovery blocks

**Files:**
- Runtime artifacts: `Go2Pvcnn/logs/m1_panda_teacher/eval/a1_fullscale_candidates_20260815/`
- Runtime artifacts: `Go2Pvcnn/logs/m1_panda_teacher/a1/<new-recovery-run>/`
- Create per block: `notes/log/2026-08-15-m1-panda-a1-recovery-block-<N>.md`
- Modify per block: `notes/log/index.md`
- Modify per block: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`

**Interfaces:**
- Consumes: verified sweep and fork CLIs from Tasks 4-6.
- Produces: selected source checkpoint, recovery checkpoints, three-seed `ranking.json` artifacts, best-checkpoint manifest fields, and an accepted-or-stopped T400.7 result.

- [ ] **Step 1: Hash and snapshot protected source files**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
sha256sum \
  logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_2700.pt \
  logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_3800.pt \
  logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_4500.pt \
  logs/m1_panda_teacher/a1/a1_dynamic_force_balance_gpu0_20260814_1848/model_5999.pt \
  > /tmp/m1-panda-a1-protected-sha-before.txt
```

Expected: five hashes recorded before any sweep or fork.

- [ ] **Step 2: Run the exact four-candidate three-seed sweep**

Run the Task 6 candidate-sweep command. Expected: 12 valid row JSON files plus `ranking.json`, all children exit `0`, and a single declared winner.

- [ ] **Step 3: Verify protected hashes are unchanged**

Repeat Step 1 to `/tmp/m1-panda-a1-protected-sha-after.txt` and run:

```bash
diff -u /tmp/m1-panda-a1-protected-sha-before.txt \
  /tmp/m1-panda-a1-protected-sha-after.txt
```

Expected: no output and exit `0`.

- [ ] **Step 4: Fork the winner for the first 500-iteration block**

```bash
cd /home/xk/coding/M1/Go2Pvcnn
RECOVERY_WINNER=$(
  /home/xk/miniconda3/envs/go2/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["winner"]["checkpoint"])' \
    /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_teacher/eval/a1_fullscale_candidates_20260815/ranking.json
)
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_teacher_train.py \
  --stage A1 \
  --base-checkpoint logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --fork-checkpoint "$RECOVERY_WINNER" \
  --run_name a1_force_balance_recovery_gpu0_20260815 \
  --num_envs 64 --max_iterations 500 --save-interval 100 \
  --device cuda:0 --headless
```

The command resolves the exact absolute winner path from `ranking.json`; do not infer it from TensorBoard alone. Expected: block exit `0`, five periodic/final checkpoints as allowed by runner save semantics, effective std `>=0.001`, full/restored curriculum, and frozen hash stable.

- [ ] **Step 5: Evaluate the block's final checkpoint**

Run `m1_panda_teacher_eval_sweep.py` with only the final recovery checkpoint and seeds 42/43/44. Append the resulting ranking artifact to `evaluation_artifacts` and update `best_checkpoint`/`best_metrics` only if its rank key exceeds the prior best.

- [ ] **Step 6: Apply the condition-based continuation rule**

If accepted, set `stop_reason="accepted_full_scale_gate"` and stop. If not accepted and it is not the second consecutive block more than `0.10` below best survival, resume the recovery checkpoint in the same directory for another 500 iterations using:

```bash
/home/xk/miniconda3/envs/go2/bin/python scripts/m1_panda_teacher_train.py \
  --stage A1 \
  --base-checkpoint logs/m1_panda_teacher/a0/a0_force_balance_gpu0_20260814_1824/model_2999.pt \
  --resume-checkpoint /ABS/RECOVERY/RUN/model_N.pt \
  --max_iterations 500 --save-interval 100 \
  --device cuda:0 --headless
```

If two consecutive blocks are each more than `0.10` below best survival, set `stop_reason="two_block_survival_regression"`, stop, and retain the recorded best checkpoint.

- [ ] **Step 7: Record every block without claiming premature success**

Each block log must contain source/final checkpoint SHA, iterations, TensorBoard tail metrics, raw/effective std, curriculum start/end, force/torque axis maxima, three-seed rates, frozen hash, decision, and next command. Update T400.7 to `done` only for the accepted gate; otherwise record it as stopped-with-best or still active.

- [ ] **Step 8: Run final verification before completion**

Re-run Task 6 static regression, `git diff --check`, protected-source hash comparison, final three-seed full-scale validation, and manifest/hash inspection. Expected: all exit `0`; only then report the accepted checkpoint or the explicit stop reason.
