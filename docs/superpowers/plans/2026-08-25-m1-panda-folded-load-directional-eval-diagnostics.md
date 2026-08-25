# M1 Panda Folded-Load Directional Evaluation Diagnostics Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task. Execute in the existing isolated worktree and do not use subagents.

**Goal:** Expose the exact forward, reverse, left-turn, or right-turn evaluation gate that rejects the preserved L0-C0 checkpoint, while ensuring diagnostic evaluation can never promote or overwrite training artifacts.

**Architecture:** Keep directional scoring as a pure extension of `evaluate_records()`: each directional bucket produces a finite, JSON-safe metrics object and the aggregate decision consumes those same `passed` booleans. Add an explicit `diagnostic_only` artifact mode so three-seed GPU evaluation can write isolated reports and an aggregate without creating an accepted/final checkpoint or becoming a curriculum parent.

**Tech Stack:** Python 3.11, pytest, PyTorch/Isaac Lab runtime, existing `AtomicStageArtifacts` JSON/checkpoint protocol.

---

## Task 1: Add explicit per-direction evaluation metrics

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_folded_load_eval.py`

### Step 1: Write failing tests

Add tests that assert:

- `directional_metrics` contains exactly `forward`, `reverse`, `left`, and `right`.
- Each bucket reports `episode_count`, `tracking_metric`, `tracking_rmse`, `tracking_limit`, `base_contact_rate`, `bad_orientation_rate`, and `passed`.
- Forward/reverse use `vx_rmse` with limit `0.04`; left/right use `wz_rmse` with limit `0.12`.
- A single isolated tracking/contact/orientation defect identifies only the affected direction when global gates remain within tolerance.
- An empty bucket emits count `0`, nullable numeric fields, and `passed == False` without NaN/Inf.
- Non-finite accumulated values raise `ValueError` instead of reaching JSON.
- Top-level `directional_pass` and `passed` are derived from the same per-direction booleans.

Run:

```bash
cd /home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_panda_folded_load_scripts.py -q
```

Expected: FAIL because `directional_metrics` does not exist and non-finite values are not rejected.

### Step 2: Implement the minimal pure evaluator changes

In `m1_panda_folded_load_eval.py`:

- Introduce a direction-to-field/metric/limit mapping.
- Add a helper that builds one JSON-safe direction report.
- Preserve the existing minimum of eight episodes per direction.
- Return nullable values for empty buckets.
- Validate finite, non-negative step counts, squared-error sums, and rates before returning.
- Build `directional_pass = all(item["passed"] for item in directional_metrics.values())`.
- Add both `directional_metrics` and `directional_pass` to the report without changing any threshold.

### Step 3: Run focused tests

Run the command from Step 1. Expected: PASS.

### Step 4: Commit

```bash
git add Go2Pvcnn/scripts/m1_panda_folded_load_eval.py Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py
git commit -m "feat: report folded-load directional evaluation gates"
```

## Task 2: Make diagnostic artifacts permanently non-promotable

**Files:**
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_folded_load_train.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_folded_load_eval.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py`

### Step 1: Write failing isolation tests

Add tests proving that:

- `validate_parent()` rejects a manifest carrying `diagnostic_only: true`, even if it otherwise looks accepted.
- Finalizing three diagnostic reports writes an aggregate with `diagnostic_only: true`, retains `accepted: false`, and never creates `model_final.pt`.
- Evaluation dispatches to diagnostic finalization when the loaded manifest is diagnostic-only.

Run:

```bash
cd /home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_panda_folded_load_scripts.py tests/test_m1_panda_folded_load_training_guard.py -q
```

Expected: FAIL because diagnostic-only manifests are not yet guarded or finalized separately.

### Step 2: Implement the smallest isolation path

- Reject `diagnostic_only` at the beginning of parent validation.
- Add `AtomicStageArtifacts.finalize_diagnostics(...)` that validates the fixed seed set and report/checkpoint identity, writes only `eval_aggregate.json`, and always records `accepted: false`, `diagnostic_only: true`, and `final_checkpoint: null`.
- In evaluation `main()`, call diagnostic finalization instead of normal finalization when the manifest marker is true.
- Do not change normal training/evaluation finalization behavior.

### Step 3: Run focused regression tests

Run the command from Step 1. Expected: PASS.

### Step 4: Commit

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_training_guard.py Go2Pvcnn/scripts/m1_panda_folded_load_eval.py Go2Pvcnn/scripts/m1_panda_folded_load_train.py Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py Go2Pvcnn/tests/test_m1_panda_folded_load_training_guard.py
git commit -m "fix: isolate folded-load diagnostic evaluations"
```

## Task 3: Re-evaluate the preserved best checkpoint on GPU 0

**Files:**
- Create runtime artifacts only under: `Go2Pvcnn/logs/m1_panda_folded_load/foundation-v1/L0-C0-directional-diagnostic-v1/`
- Modify: `notes/log/2026-08-25-m1-panda-folded-load-l0-stop-diagnosis.md`

### Step 1: Verify source checkpoint identity and target absence

Confirm the source remains:

```text
Go2Pvcnn/logs/m1_panda_folded_load/foundation-v1/L0-C0/model_best.pt
sha256=f231009992ae07ae3de2560cfadb4d812fdb6cd38c8fa6deca7d4b2b8466ae8e
```

Abort rather than overwrite if the diagnostic directory already exists with incompatible content.

### Step 2: Create the isolated diagnostic directory

Copy `model_best.pt` and derive a local `run_manifest.json` from the source manifest with:

```json
{
  "diagnostic_only": true,
  "accepted": false,
  "status": "diagnostic_ready",
  "final_checkpoint": null,
  "final_checkpoint_sha256": null
}
```

Point `best_checkpoint` at the copied checkpoint. Do not modify the source run directory.

### Step 3: Run fixed-seed evaluation on GPU 0

For seeds `42`, `43`, and `44`, run:

```bash
cd /home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 TERM=xterm-256color /home/xk/coding/IsaacLab/isaaclab.sh -p scripts/m1_panda_folded_load_eval.py --stage L0-C0 --run_dir logs/m1_panda_folded_load/foundation-v1/L0-C0-directional-diagnostic-v1 --seed SEED --num_envs 64 --device cuda:0 --headless
```

Expected: three reports and one diagnostic aggregate; no `model_final.pt`.

### Step 4: Record the evidence

Append the exact per-seed directional failures and aggregate conclusion to the existing diagnosis log. Explicitly state that no retraining was started and the original run artifacts were unchanged.

### Step 5: Run final verification

```bash
cd /home/xk/coding/M1/.worktrees/m1-panda-ppo-stability/Go2Pvcnn
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_m1_panda_folded_load_scripts.py tests/test_m1_panda_folded_load_training_guard.py tests/test_m1_panda_folded_load_curriculum.py tests/test_m1_panda_folded_load_orchestrator.py -q
git diff --check
```

Expected: focused suite PASS and `git diff --check` produces no output.

### Step 6: Commit diagnosis evidence

```bash
git add notes/log/2026-08-25-m1-panda-folded-load-l0-stop-diagnosis.md
git commit -m "docs: record folded-load directional diagnosis"
```
