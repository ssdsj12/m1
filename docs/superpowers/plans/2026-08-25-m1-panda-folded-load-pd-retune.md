# M1 + Panda Folded-Load PD Retune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hold the folded Panda safely while M1 learns locomotion by applying fixed `120/8` shoulder gains only in the folded-load task and validating the unchanged safety gates on GPU0.

**Architecture:** Derive a task-local articulation configuration from `M1_PANDA_CFG`, replace only the `panda_shoulder` actuator gains, and keep the global asset configuration untouched. The wrapper writes the approved fold position and zero-velocity targets before every physics step; a standalone zero-action probe and the existing training smoke runs enforce the physical gates before long training.

**Tech Stack:** Python 3.11, Isaac Lab 2.2/Isaac Sim 5, PyTorch, pytest, rsl_rl PPO, JSON run manifests.

## Global Constraints

- Folded-load `panda_joint1–4`: fixed `Kp=120`, `Kd=8` in stationary and moving stages.
- `panda_joint5–7`: retain `Kp=80`, `Kd=4`; effort limits remain `87 Nm` and `12 Nm`.
- The global `M1_PANDA_CFG`, combined USD, asset SHA, 103-observation boundary, 23-action boundary, 200 Hz step rate, and active mask `[1]*16+[0]*7` remain unchanged.
- The fold target remains `(0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)` with zero velocity targets.
- Never weaken the finite-state, inactive-action, `0.35 rad` fold-error, `1.0` effort-utilization, or `0.01 rad` joint-margin gates.
- GPU validation uses GPU0 and must pass in order: 8×16 zero-action, 8×256 zero-action, 8×1 training, then 64×10 training.

---

### Task 1: Isolate the `120/8` Folded-Load Controller

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_env_static.py`

**Interfaces:**
- Consumes: global `M1_PANDA_CFG` with legacy `panda_shoulder` gains `80/4`.
- Produces: module constant `M1_PANDA_FOLDED_LOAD_CFG`, used only by `M1PandaFoldedLoadEnvCfg.__post_init__`.

- [ ] **Step 1: Write the failing isolation test**

```python
def test_folded_load_pd_override_is_task_local_and_preserves_global_asset():
    source = CFG.read_text(encoding="utf-8")
    asset = ASSET.read_text(encoding="utf-8")
    assert "M1_PANDA_FOLDED_LOAD_CFG = M1_PANDA_CFG.copy()" in source
    assert '"panda_shoulder": M1_PANDA_CFG.actuators["panda_shoulder"].replace(' in source
    assert "stiffness=120.0" in source
    assert "damping=8.0" in source
    assert "self.scene.robot = M1_PANDA_FOLDED_LOAD_CFG.replace" in source
    assert "stiffness=80.0" in asset
    assert "damping=4.0" in asset
```

- [ ] **Step 2: Run the test and verify the missing task-local configuration causes failure**

Run:

```bash
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_folded_load_env_static.py::test_folded_load_pd_override_is_task_local_and_preserves_global_asset -q
```

Expected: `FAIL` because `M1_PANDA_FOLDED_LOAD_CFG` does not exist.

- [ ] **Step 3: Add the minimal isolated configuration**

```python
M1_PANDA_FOLDED_LOAD_CFG = M1_PANDA_CFG.copy()
M1_PANDA_FOLDED_LOAD_CFG.actuators = {
    **M1_PANDA_CFG.actuators,
    "panda_shoulder": M1_PANDA_CFG.actuators["panda_shoulder"].replace(
        stiffness=120.0,
        damping=8.0,
    ),
}
```

Then select it only in the folded-load environment:

```python
self.scene.robot = M1_PANDA_FOLDED_LOAD_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot"
)
```

- [ ] **Step 4: Run the focused static suite**

Run the command from Step 2 without the node selector. Expected: all tests pass.

- [ ] **Step 5: Commit the controller isolation**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_env_cfg.py Go2Pvcnn/tests/test_m1_panda_folded_load_env_static.py
git commit -m "fix: strengthen folded Panda hold controller"
```

### Task 2: Make Fold Targets and Probe Exit Status Explicit

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py`
- Modify: `Go2Pvcnn/scripts/m1_panda_folded_load_train.py`
- Create: `Go2Pvcnn/scripts/m1_panda_folded_load_probe.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_wrapper.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_folded_load_probe_static.py`
- Create: `docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md`

**Interfaces:**
- Consumes: task-local controller from Task 1 and the existing `EpisodeRecord` safety metrics.
- Produces: `_apply_fold_pd_targets()`, `training_completion_exit_code(eligible: bool) -> int`, an atomic probe report, and operational GPU0 commands.

- [ ] **Step 1: Verify the existing failing-test history and run the focused tests**

The wrapper test requires every step to issue the seven approved position targets and zero velocity targets. The script test requires a finite, deliberately ineligible smoke to exit `0`, while the manifest remains `accepted=false`. Run:

```bash
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_folded_load_wrapper.py Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py Go2Pvcnn/tests/test_m1_panda_folded_load_probe_static.py -q
```

Expected: all focused tests pass with the current target-writing implementation.

- [ ] **Step 2: Guarantee the probe process returns the report status**

Add a test that asserts the entry point flushes both output streams and uses `os._exit(main())`; then implement:

```python
if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
```

This prevents SimulationApp shutdown from replacing a failed report's nonzero shell status.

- [ ] **Step 3: Correct the runbook launcher contract**

Every Isaac command must begin with:

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p
```

The runbook must preserve failed output directories, require the atomic JSON report's `passed=true`, and prohibit promotion from smoke manifests with `accepted=false`.

- [ ] **Step 4: Run the focused and full CPU suites**

```bash
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests/test_m1_panda_folded_load_wrapper.py Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py Go2Pvcnn/tests/test_m1_panda_folded_load_probe_static.py -q
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests -q
```

Expected: both commands pass.

- [ ] **Step 5: Commit controller targets, probe, and runbook**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_panda_folded_load_wrapper.py Go2Pvcnn/scripts/m1_panda_folded_load_train.py Go2Pvcnn/scripts/m1_panda_folded_load_probe.py Go2Pvcnn/tests/test_m1_panda_folded_load_wrapper.py Go2Pvcnn/tests/test_m1_panda_folded_load_scripts.py Go2Pvcnn/tests/test_m1_panda_folded_load_probe_static.py docs/superpowers/runbooks/2026-08-25-m1-panda-folded-load-locomotion.md
git commit -m "feat: guard folded-load GPU validation"
```

### Task 3: Validate GPU0 and Record the Promotion Decision

**Files:**
- Modify: `notes/todo.md`
- Modify: `notes/todo/T400-m1-panda-force-aware-teacher-student.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-08-25-m1-panda-folded-load-pd-retune-gpu.md`

**Interfaces:**
- Consumes: zero-action probe, folded-load training script, and unchanged physical gates.
- Produces: reproducible JSON/manifests plus a written pass/fail decision for long training.

- [ ] **Step 1: Preserve earlier failed diagnostics and run the 8×16 probe**

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_panda_folded_load_probe.py --num_envs 8 --steps 16 --device cuda:0 --report Go2Pvcnn/logs/m1_panda_folded_load/probe-pd120-8x16.json --headless
```

Expected: shell exit `0`, report `passed=true`, inactive action exactly zero, fold error `<=0.35`, effort utilization `<=1.0`, and joint margin `>0.01`.

- [ ] **Step 2: Run the 8×256 probe**

Use the same command with `--num_steps 256` and report `probe-pd120-8x256.json`. Expected: the same gates pass over the full PPO horizon.

- [ ] **Step 3: Run the 8×1 training smoke in a fresh directory**

```bash
TERM=xterm-256color CONDA_PREFIX=/home/xk/miniconda3/envs/go2 CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/m1_panda_folded_load_train.py --stage L0-C0 --num_envs 8 --max_iterations 1 --device cuda:0 --run_dir Go2Pvcnn/logs/m1_panda_folded_load/smoke-pd120-8x1 --headless
```

Expected: exit `0`, finite PPO diagnostics, no `fold_hard_failure`, and manifest `accepted=false` because it is only a smoke run.

- [ ] **Step 4: Run the 64×10 stability smoke in a fresh directory**

Repeat Step 3 with `--num_envs 64 --max_iterations 10` and `smoke-pd120-64x10`. Expected: exit `0`, bounded finite diagnostics, no safety stop, manifest `accepted=false`.

- [ ] **Step 5: Record exact commands, metrics, warnings, and the long-run decision**

The log must include pre-retune values (`fold_error=0.2620`, `joint_margin=-0.1503`), post-retune maxima/minima, process statuses, manifest paths, and the unresolved `Panda/root_joint` PhysX warning. Mark long training eligible only if all four GPU gates pass.

- [ ] **Step 6: Run final verification and commit evidence**

```bash
PYTHONPATH=rsl_rl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/xk/miniconda3/envs/go2/bin/python -m pytest Go2Pvcnn/tests -q
git diff --check
git status --short
git add notes/todo.md notes/todo/T400-m1-panda-force-aware-teacher-student.md notes/log/index.md notes/log/2026-08-25-m1-panda-folded-load-pd-retune-gpu.md
git commit -m "docs: record folded-load PD GPU qualification"
```

Expected: tests pass, `git diff --check` is silent, and only intentionally ignored runtime artifacts remain outside Git.
