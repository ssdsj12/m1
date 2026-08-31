# M1 + Panda Phase 6 Guarded Learning-Rate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Prevent Phase 6 residual PPO from amplifying its task-local learning rate above 1e-5, then re-run every fail-closed GPU gate through an accepted promotion and guarded 3000-update long run.

**Architecture:** Keep the shared PPO scheduler and all KL/physical acceptance thresholds unchanged. Freeze only the residual task maximum learning rate at its initial value, prove that contract with a focused static test, and use fresh v4 artifact roots so pilot, short, 24-worker promotion, and long lineage cannot reuse rejected v3 state.

**Tech Stack:** Python 3.11, pytest, PyTorch/RSL-RL PPO, Isaac Lab/Isaac Sim, CUDA GPU0, JSON/SHA-256 lineage manifests, Git.

## Global Constraints

- Work directly on /home/xk/coding/M1 branch main; do not stage the unrelated graphify-out/cache/last_query_stamp change.
- Use one agent only; do not dispatch subagents.
- Keep learning_rate=1e-5, min_learning_rate=1e-6, schedule=adaptive, desired_kl=0.01, and kl_abort_threshold=0.015.
- Set only this task max_learning_rate=1e-5; do not modify rsl_rl/rsl_rl/algorithms/ppo.py.
- Keep every pilot, physical, source-lineage, normalizer, short, promotion, and long gate unchanged and fail closed.
- Run Isaac/RSL on CUDA_VISIBLE_DEVICES=0 and logical device cuda:0 with /home/xk/miniconda3/envs/go2/bin/python.
- Preserve rejected v3 artifacts as immutable evidence. Use fresh roots ending in guarded_lr_v4.
- Do not start short unless pilot_accepted is exactly true; do not start long unless promotion accepted is exactly true.

---

### Task 1: Freeze the Residual Task Upward Learning-Rate Range

**Files:**
- Modify: Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py
- Modify: Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py

**Interfaces:**
- Consumes: get_m1_panda_arm_mpc_residual_train_cfg() -> dict and the existing PPO learning-rate keys.
- Produces: a task-local configuration where 1e-6 <= learning_rate == max_learning_rate == 1e-5, while all frozen Phase 6 settings remain unchanged.

- [ ] **Step 1: Write the failing configuration contract**

Change the existing maximum-rate assertion in test_residual_ppo_config_freezes_200_hz_stability_contract to:

~~~python
assert cfg["algorithm"]["learning_rate"] == pytest.approx(1.0e-5)
assert cfg["algorithm"]["min_learning_rate"] == pytest.approx(1.0e-6)
assert cfg["algorithm"]["max_learning_rate"] == pytest.approx(1.0e-5)
assert (
    cfg["algorithm"]["max_learning_rate"]
    == cfg["algorithm"]["learning_rate"]
)
~~~

- [ ] **Step 2: Run the focused test and verify RED**

Run:

~~~bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_train_static.py::test_residual_ppo_config_freezes_200_hz_stability_contract
~~~

Expected: FAIL because the current maximum is 1e-4, not 1e-5.

- [ ] **Step 3: Apply the minimal task-local implementation**

In get_m1_panda_arm_mpc_residual_train_cfg(), change only:

~~~python
"max_learning_rate": 1.0e-5,
~~~

Do not edit shared PPO code or any KL threshold.

- [ ] **Step 4: Run focused and relevant CPU verification**

Run:

~~~bash
cd /home/xk/coding/M1/Go2Pvcnn
/home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_panda_arm_mpc_residual_train_static.py \
  tests/test_rsl_ppo_adaptive_schedule.py \
  tests/test_m1_panda_folded_load_ppo.py \
  tests/test_m1_panda_arm_mpc_residual_pilot.py \
  tests/test_m1_panda_arm_mpc_residual_lineage.py \
  tests/test_m1_panda_arm_mpc_residual_promote.py
/home/xk/miniconda3/envs/go2/bin/python -m compileall -q \
  agent go2_pvcnn scripts rsl_rl/rsl_rl
git -C /home/xk/coding/M1 diff --check
~~~

Expected: all tests pass, compileall exits 0, and git diff --check emits no output.

- [ ] **Step 5: Commit the isolated correction**

~~~bash
cd /home/xk/coding/M1
git add \
  Go2Pvcnn/agent/m1_panda_arm_mpc_residual_train_cfg.py \
  Go2Pvcnn/tests/test_m1_panda_arm_mpc_residual_train_static.py
git commit -m "fix: guard phase6 residual learning rate"
~~~

Expected: the commit contains exactly the configuration and its test.

### Task 2: Re-run and Gate the Fresh Seed-42 Pilot

**Files:**
- Modify: notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md
- Runtime output: Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4/

**Interfaces:**
- Consumes: the committed guarded configuration and the already accepted Phase 5 seed-42 probe.
- Produces: a fresh schema-v2 pilot manifest whose source/config/runtime hashes include the corrective commit and whose pilot_accepted field decides whether short may start.

- [ ] **Step 1: Prove the fresh root does not exist and GPU0 is available**

~~~bash
test ! -e /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
~~~

Expected: the test exits 0; no conflicting training process owns GPU0.

- [ ] **Step 2: Launch the exact fresh pilot**

~~~bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage pilot \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4 \
  --num_envs 8 --max_iterations 10 --seed 42 \
  --device cuda:0 --headless
~~~

Expected: process exit 0, exactly ten completed updates, and an atomic run_manifest.json.

- [ ] **Step 3: Parse the authoritative pilot decision**

~~~bash
cd /home/xk/coding/M1
/home/xk/miniconda3/envs/go2/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4")
doc = json.loads((root / "run_manifest.json").read_text())
assert doc["schema_version"] == 2
assert doc["stage"] == "pilot"
assert doc["status"] == "safe_complete"
assert doc["completed_iterations"] == 10
assert len(doc["optimizer_summaries"]) == 10
assert doc["accepted"] is False
assert doc["promotion_required"] is False
assert not list(root.glob("candidate_u*.pt"))
assert max(x["learning_rate"] for x in doc["optimizer_summaries"]) <= 1.0e-5
print(json.dumps(doc["pilot_decision"], indent=2, sort_keys=True))
assert doc["pilot_accepted"] is True
assert doc["pilot_decision"]["accepted"] is True
PY
~~~

Expected: accepted decision, KL abort count at most 3, median completed mini-batches at least 6, median value loss below 100, and learning rate never above 1e-5. If any assertion fails, record the manifest and return to systematic diagnosis; do not launch short.

- [ ] **Step 4: Record the immutable v3 rejection and v4 pilot result**

Append exact v3/v4 manifest paths, hashes, optimizer medians, abort counts, physical metrics, command, exit status, and acceptance decision to the execution log. Do not describe v4 as accepted unless Step 3 passed.

### Task 3: Execute Short, Three-Seed Promotion, and Conditional Long

**Files:**
- Modify: notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md
- Modify: notes/log/index.md
- Modify: notes/todo/T400-m1-panda-force-aware-teacher-student.md
- Runtime output: Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4/
- Runtime output: Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_guarded_lr_v4/

**Interfaces:**
- Consumes: accepted v4 pilot manifest, five normalized short candidates, and the existing process-isolated promotion driver.
- Produces: a complete 100-update short manifest, 24 fixed-condition worker results over seeds 42/43/44, an atomic promotion decision, and only on accepted=true a monitored 3000-update long run.

- [ ] **Step 1: Launch 100-update short only from the accepted pilot**

~~~bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage short \
  --pilot_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/pilot_s42_guarded_lr_v4/run_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4 \
  --num_envs 8 --max_iterations 100 --seed 42 \
  --device cuda:0 --headless
~~~

Expected: status=safe_complete, completed_iterations=100, accepted=false, promotion_required=true, accepted pilot path/hash preserved, and exactly five candidate checkpoints containing actor and critic normalizer state dictionaries. A safety stop or non-finite diagnostic blocks promotion.

- [ ] **Step 2: Run all 24 process-isolated fixed-condition workers**

~~~bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_promote.py \
  --short_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4/run_manifest.json \
  --device cuda:0 --headless
~~~

Expected: nine zero-pair and fifteen candidate worker JSON files, seeds exactly 42/43/44, each worker in a fresh process for exactly 4000 steps, matching source/runtime/bundle/checkpoint hashes, and one atomic promotion_manifest.json. Exit 2 is an evidence-backed rejection, not permission to weaken gates.

- [ ] **Step 3: Enforce the promotion branch with parsed JSON**

~~~bash
cd /home/xk/coding/M1
/home/xk/miniconda3/envs/go2/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4")
doc = json.loads((root / "promotion_manifest.json").read_text())
print(json.dumps({k: doc.get(k) for k in ("accepted", "reasons", "best_checkpoint", "best_checkpoint_sha256")}, indent=2, sort_keys=True))
if doc["accepted"]:
    checkpoint = Path(doc["best_checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == doc["best_checkpoint_sha256"]
else:
    assert not Path("Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_guarded_lr_v4").exists()
raise SystemExit(0 if doc["accepted"] else 2)
PY
~~~

Expected: continue only on exit 0. On exit 2, document exact rejection reasons and stop with no long directory/process.

- [ ] **Step 4: Launch the 3000-update long only after accepted promotion**

~~~bash
cd /home/xk/coding/M1/Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_panda_arm_mpc_residual_train.py \
  --stage long \
  --promotion_manifest /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/short_s42_guarded_lr_v4/promotion_manifest.json \
  --run_dir /home/xk/coding/M1/Go2Pvcnn/logs/m1_panda_arm_mpc_residual/long_s42_guarded_lr_v4 \
  --num_envs 8 --max_iterations 3000 --seed 42 \
  --device cuda:0 --headless
~~~

Monitor process liveness, GPU0 ownership, atomic manifest status, completed iterations, KL/learning-rate/value-loss/std diagnostics, hard failures, MPC/QP/contact rates, stop reason, and checkpoint publication. A running process is progress, not completion; only a safe 3000/3000 manifest proves long completion.

- [ ] **Step 5: Update authoritative project records and verify the worktree**

Record commands, commit/hash lineage, pilot/short/promotion/long decisions, all worker counts, decisive metrics, and current process state. Then run:

~~~bash
cd /home/xk/coding/M1
git diff --check
git status --short
~~~

Expected: no whitespace errors; only intentional documentation changes plus the unrelated pre-existing graphify stamp.

- [ ] **Step 6: Commit the evidence ledger**

~~~bash
cd /home/xk/coding/M1
git add \
  notes/log/2026-08-30-m1-panda-phase6-ppo-scale-normalization-execution.md \
  notes/log/index.md \
  notes/todo/T400-m1-panda-force-aware-teacher-student.md
git commit -m "docs: record phase6 guarded learning rate acceptance"
~~~

Expected: the commit contains only authoritative Phase 6 execution records. Do not claim overall completion unless every named manifest and conditional gate above has been inspected and passed.

