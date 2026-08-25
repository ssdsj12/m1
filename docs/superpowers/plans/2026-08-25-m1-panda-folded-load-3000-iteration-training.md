# M1 + Panda Folded-Load 3000-Iteration Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow every folded-load curriculum stage to train for a requested maximum of 3000 PPO updates without plateau/patience early stopping, while retaining all existing catastrophic safety stops.

**Architecture:** Make `MAX_TRAINING_ITERATIONS = 3000` and `validate_max_iterations()` shared curriculum contracts consumed by both entrypoints. Restrict `FoldedLoadTrainingGuard` to safety decisions only; the runner owns planned completion. Keep eligible-best checkpoint semantics unchanged and reduce periodic checkpoint frequency to every 100 updates.

**Tech Stack:** Python 3.11, PyTorch, pytest, Isaac Lab 5.1, RSL-RL PPO.

## Global Constraints

- Do not change assets, Panda PD parameters, action masks, observations, rewards, command ranges, domain randomization, rank definitions, or evaluation thresholds.
- Permit requested iteration counts only in the inclusive range `1..3000`.
- Remove `eligible_patience_50_updates` and guard-owned `max_iterations_600` stops.
- Retain non-finite, inactive-action leak, fold hard failure, and existing hard-failure-rate stops exactly.
- Keep `model_best.pt` eligible-only; set periodic `save_interval` to `100`.
- Use GPU 0 for the final smoke and never reuse a non-empty run directory.
- Do not delete or overwrite `foundation-v1`, `foundation-v2`, or `foundation-v3` artifacts.

---

### Task 1: Restrict the training guard to catastrophic safety stops

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py`

**Interfaces:**
- Consumes: `FoldedLoadTrainingGuard.update(iteration, episodes, *, finite, inactive_action_max, fold_hard_failure) -> GuardDecision`.
- Produces: the same public interface, but `GuardDecision.stop` is never caused by an eligible plateau or update count.

- [ ] **Step 1: Replace plateau assumptions with a failing continuation test**

Add this test after the command-level eligibility test:

```python
def test_eligible_plateau_never_stops_normal_learning():
    guard = FoldedLoadTrainingGuard(stage_spec("L0-C0"))
    first = guard.update(1, _eligible_window())
    assert first.eligible and first.save_best and not first.stop

    decision = first
    for iteration in range(2, 3001):
        decision = guard.update(iteration, [])
        assert decision.reason not in {
            "eligible_patience_50_updates",
            "max_iterations_600",
        }
        assert decision.stop is False
    assert decision.reason is None
```

This deliberately holds the rolling episode window constant, so the first eligible rank cannot improve and the old implementation stops at iteration 51.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
tests/test_m1_panda_folded_load_training_guard.py::test_eligible_plateau_never_stops_normal_learning -q
```

Expected: FAIL because `eligible_patience_50_updates` appears after 50 non-improving updates.

- [ ] **Step 3: Remove only normal-learning stop state**

In `FoldedLoadTrainingGuard.__init__`, remove:

```python
self.patience_without_improvement = 0
```

In `update()`, keep eligible-best selection and `save_best`, but remove the `improved` local, the patience update block, and both normal completion branches. The resulting decision block must be:

```python
reason = None
if self.high_failure_updates >= 2:
    reason = "hard_failure_rate_gt_0.50_for_2_updates"
elif self.medium_failure_updates >= 5:
    reason = "hard_failure_rate_gt_0.20_for_5_updates"
return GuardDecision(
    stop=reason is not None,
    eligible=bool(snapshot is not None and snapshot.eligible),
    save_best=save_best,
    reason=reason,
    snapshot=snapshot,
)
```

Do not change the four immediate safety returns or failure-rate counters.

- [ ] **Step 4: Run the guard suite and verify GREEN**

Run:

```bash
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
tests/test_m1_panda_folded_load_training_guard.py -q
```

Expected: all guard tests PASS, including catastrophe tests and the 3000-update plateau test.

- [ ] **Step 5: Commit the safety-only guard**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py \
        Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py
git commit -m "fix: let folded-load learning run to requested limit"
```

### Task 2: Share and enforce the 3000-update contract

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_curriculum.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_ppo.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_folded_load_train.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_folded_load_curriculum.py`
- Modify: `Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py`

**Interfaces:**
- Produces: `MAX_TRAINING_ITERATIONS: int = 3000`.
- Produces: `validate_max_iterations(value: int) -> int`, returning a valid integer and raising `ValueError("max_iterations must be in [1, 3000]")` otherwise.
- Consumes: both train and curriculum entrypoints use the shared constant as their parser default and shared validator in `main()`.

- [ ] **Step 1: Write failing shared-contract and checkpoint tests**

Add to `test_m1_panda_folded_load_curriculum.py`:

```python
def test_training_iteration_contract_allows_full_3000_updates():
    assert curriculum.MAX_TRAINING_ITERATIONS == 3000
    assert curriculum.validate_max_iterations(1) == 1
    assert curriculum.validate_max_iterations(3000) == 3000
    for invalid in (0, 3001, -1, True, 3.5):
        with pytest.raises((TypeError, ValueError)):
            curriculum.validate_max_iterations(invalid)
```

Update `test_m1_panda_folded_load_ppo.py`:

```python
assert cfg["save_interval"] == 100
```

Extend the operational-contract test in `test_m1_panda_folded_load_scripts.py`:

```python
curriculum_script = (ROOT / "scripts/m1_panda_folded_load_curriculum.py").read_text(
assert "default=MAX_TRAINING_ITERATIONS" in train
assert "default=MAX_TRAINING_ITERATIONS" in curriculum_script
assert "validate_max_iterations(args.max_iterations)" in train
assert "validate_max_iterations(args.max_iterations)" in curriculum_script
assert "eligible_patience_50_updates" not in train
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
tests/test_m1_panda_folded_load_curriculum.py \
tests/test_m1_panda_folded_load_ppo.py \
tests/test_m1_panda_folded_load_scripts.py -q
```

Expected: FAIL because the shared constant/validator do not exist, parser defaults remain 600, and `save_interval` remains 25.

- [ ] **Step 3: Add the shared iteration validator**

In the task curriculum module, add:

```python
MAX_TRAINING_ITERATIONS = 3000


def validate_max_iterations(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_iterations must be an integer")
    if not 1 <= value <= MAX_TRAINING_ITERATIONS:
        raise ValueError("max_iterations must be in [1, 3000]")
    return value
```

Export both names in that module's `__all__`.

- [ ] **Step 4: Make both entrypoints consume the shared contract**

Add `MAX_TRAINING_ITERATIONS` and `validate_max_iterations` to their existing curriculum imports. In both parsers use:

```python
parser.add_argument(
    "--max_iterations", type=int, default=MAX_TRAINING_ITERATIONS
)
```

In both `main()` functions replace the `<= 600` validation with:

```python
validate_max_iterations(args.max_iterations)
```

Do not change how the value reaches `runner.learn()` or `ProcessStageExecutor`.

- [ ] **Step 5: Reduce periodic checkpoint count**

In `get_m1_panda_folded_load_train_cfg()` change exactly:

```python
"save_interval": 100,
```

Do not change eligible `model_best.pt` publication.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests PASS.

- [ ] **Step 7: Commit the shared 3000-update contract**

```bash
git add Go2Pvcnn/agent/m1_panda_folded_load_train_cfg.py \
        Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py \
        Go2Pvcnn/scripts/m1_panda_folded_load_train.py \
        Go2Pvcnn/scripts/m1_panda_folded_load_curriculum.py \
        Go2Pvcnn/tests/test_m1_panda_folded_load_curriculum.py \
        Go2Pvcnn/tests/test_m1_panda_folded_load_ppo.py \
        Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py
git commit -m "feat: support 3000-update folded-load stages"
```

### Task 3: Update the runbook and verify the merged behavior

**Files:**
- Modify: `docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md`
- Test: folded-load focused suites and GPU 0 smoke artifacts.

**Interfaces:**
- Consumes: the shared 3000-update CLI contract from Task 2.
- Produces: a main-workspace command using `/home/xk/coding/M1/Go2Pvcnn`, direct Go2 Python, `--max_iterations 3000`, and a fresh `foundation-v4` experiment root.

- [ ] **Step 1: Update stale worktree paths and long-train values**

In the runbook:

- replace `/home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn` with `/home/xk/coding/M1/Go2Pvcnn`;
- document that direct `/home/xk/miniconda3/envs/go2/bin/python -u` must own the persistent PTY session;
- change the full curriculum command to `--max_iterations 3000` and `foundation-v4`;
- state that plateau early stopping is disabled and only safety stops may interrupt a stage.

- [ ] **Step 2: Run the complete focused regression**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/xk/miniconda3/envs/go2/bin/python -m pytest \
tests/test_m1_panda_folded_load_training_guard.py \
tests/test_m1_panda_folded_load_curriculum.py \
tests/test_m1_panda_folded_load_orchestrator.py \
tests/test_m1_panda_folded_load_ppo.py \
tests/test_m1_panda_folded_load_scripts.py -q
/home/xk/miniconda3/envs/go2/bin/python -m py_compile \
scripts/m1_panda_folded_load_train.py \
scripts/m1_panda_folded_load_curriculum.py \
go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py \
go2_pvcnn/tasks/m1_panda_folded_load_curriculum.py
git diff --check
```

Expected: all selected tests PASS, compilation exits 0, and `git diff --check` has no output.

- [ ] **Step 3: Run an isolated 8×1 GPU 0 smoke**

First confirm `logs/m1_panda_folded_load/smoke-3000-control-8x1` does not exist. Then run in a persistent PTY:

```bash
TERM=xterm-256color \
CONDA_PREFIX=/home/xk/miniconda3/envs/go2 \
CUDA_VISIBLE_DEVICES=0 \
/home/xk/miniconda3/envs/go2/bin/python -u \
scripts/m1_panda_folded_load_train.py \
  --stage L0-C0 \
  --run_dir logs/m1_panda_folded_load/smoke-3000-control-8x1 \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0 \
  --headless
```

Expected: one learning iteration completes, manifest records `requested_iterations=1`, `stop_reason=requested_iterations_complete`, and no plateau stop.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md
git commit -m "docs: run folded-load curriculum for 3000 updates"
```

- [ ] **Step 5: Request review before integration**

Review the complete range against the approved spec, then rerun Task 3 Step 2 before claiming completion or offering merge options.
