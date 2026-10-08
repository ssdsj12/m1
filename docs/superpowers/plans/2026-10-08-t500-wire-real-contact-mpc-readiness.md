# T500 Wire Real-Contact MPC Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one deterministic dual-Panda/dual-O6 real-contact wire grasp-lift-hold pipeline, collect at least 30 complete successful teacher episodes, retrain the 15D latent artifact, and load it through MPC Play.

**Architecture:** Preserve the existing object/arm/hand MPC, WBC, safety projection, and 43D execution contract. Add explicit wire-contact observations and deterministic acceptance state around the current wrapper, then gate teacher-success collection and latent training on that acceptance record. No proximity-only contact or kinematic attachment is accepted as physical grasp evidence.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab/PhysX GPU0, DeformX/PyElastica wire backend, NumPy NPZ datasets, pytest, JSONL/SHA-256 manifests.

## Global Constraints

- Scope is the selected `wire` object only; other object classes remain unchanged.
- Each hand requires at least two selected-wire fingertip contacts above `0.2 N`.
- Bilateral contact must remain continuous for `0.5 s` before lift.
- Wire center lift must reach at least `0.05 m` and remain held for `1.0 s`.
- Endpoint slip relative to the corresponding palm must not exceed `0.02 m`.
- Contact evidence must come from target-filtered PhysX/DeformX observations, never proximity alone.
- No joint-limit violation, non-finite value, or atomic safety rejection is allowed in a success episode.
- CPU-heavy validation uses one thread/core; GPU validation uses GPU0.
- Do not change the public 43D control contract or unrelated object behavior.

---

### Task 1: Wire Contact Observation Contract

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_contact_acceptance.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_contact_acceptance.py`

**Interfaces:**
- Consumes: per-hand `contact_mask`, per-fingertip force vectors, fingertip positions, palm positions, wire endpoint positions, simulation `dt`.
- Produces: `WireContactObservation`, `WireAcceptanceConfig`, and pure `evaluate_wire_contact(observation, config) -> WireContactEvaluation`.

- [ ] **Step 1: Write failing contract tests**

Add tests that construct deterministic tensors and assert: target contacts below `0.2 N` do not count; two contacts per hand do count; endpoint slip above `0.02 m` rejects; non-finite input rejects with `non_finite_contact_observation`.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q Go2Pvcnn/tests/test_m1_wire_contact_acceptance.py
```

Expected: collection failure because `wire_contact_acceptance` does not exist.

- [ ] **Step 3: Implement immutable observations and pure evaluation**

Implement frozen dataclasses with explicit float thresholds and return fields `left_valid_contacts`, `right_valid_contacts`, `left_slip_m`, `right_slip_m`, `bilateral_contact`, `finite`, and `failure_reason`. Validate shapes and dtypes at construction.

- [ ] **Step 4: Verify GREEN and regression**

Run the Task 1 test plus `test_m1_bimanual_contact_summary.py` and `test_m1_wire_coupling.py`; expect all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_contact_acceptance.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py Go2Pvcnn/tests/test_m1_wire_contact_acceptance.py
git commit -m "feat: define wire contact acceptance observations"
```

### Task 2: Deterministic Grasp-Lift Acceptance State

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_contact_acceptance.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_contact_acceptance.py`

**Interfaces:**
- Consumes: `WireContactEvaluation`, wire reset/current center height, `dt`, safety feasibility.
- Produces: `WireAcceptanceTracker.update(...) -> WireAcceptanceState` with phase timers, lift permission, re-clamp request, success, and terminal reason.

- [ ] **Step 1: Write failing transition tests**

Cover: contact shorter than `0.5 s` cannot lift; stable bilateral contact permits lift; contact loss during lift sets `request_reclamp=True`; `0.05 m` lift followed by `1.0 s` stable hold succeeds; excessive slip and safety rejection terminate with distinct reasons.

- [ ] **Step 2: Verify RED**

Run the focused test and expect missing `WireAcceptanceTracker`.

- [ ] **Step 3: Implement the tracker**

Use accumulated simulation time only. Reset all timers deterministically. Never preserve success or latch state across reset.

- [ ] **Step 4: Verify GREEN**

Run `test_m1_wire_contact_acceptance.py` and `test_m1_bimanual_grasp_lift_pipeline.py`; expect all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_contact_acceptance.py Go2Pvcnn/tests/test_m1_wire_contact_acceptance.py
git commit -m "feat: track deterministic wire grasp-lift acceptance"
```

### Task 3: Wrapper Integration and Per-Digit Re-Clamping

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_contact_diagnostics.py`

**Interfaces:**
- Consumes: Task 1 contact evaluation and Task 2 tracker.
- Produces: wrapper property `wire_acceptance_state`, contact-aware hand targets, and diagnostics containing per-finger distance/force/normal plus palm error, wire height, slip, stage timers, feasibility, and rejection reason.

- [ ] **Step 1: Write failing wrapper and latch tests**

Assert contacted digits retain their measured joint target, uncontacted digits continue toward the Hand MPC target, loss of bilateral contact during lift suppresses upward arm command, and reset clears all contact/acceptance latches.

- [ ] **Step 2: Verify RED**

Run the two focused test files; expect failures for the missing acceptance integration and re-clamp behavior.

- [ ] **Step 3: Integrate observations and tracker**

Build observations from the existing target-filtered contact summaries, fingertip states, palm states, and wire scene geometry. Keep all values in the common M1 base frame. Apply the tracker before command submission and include its outcome in the atomic safety decision.

- [ ] **Step 4: Implement minimal per-digit re-clamping**

Extend the existing latch helper so already-contacted axes remain fixed while only uncontacted digits receive closing targets. On bilateral contact loss during lift, hold arm height and allow uncontacted fingers to close again.

- [ ] **Step 5: Verify GREEN and regression**

Run:

```bash
PYTHONPATH=Go2Pvcnn /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py \
  Go2Pvcnn/tests/test_m1_wire_contact_diagnostics.py \
  Go2Pvcnn/tests/test_m1_bimanual_grasp_lift_pipeline.py \
  Go2Pvcnn/tests/test_m1_wire_coupling.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/o6_contact_kinematics.py Go2Pvcnn/tests/test_m1_bimanual_o6_contact.py Go2Pvcnn/tests/test_m1_wire_contact_diagnostics.py
git commit -m "feat: integrate real wire contact and re-clamping"
```

### Task 4: GPU0 Contact and Lift Probe

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Create: `Go2Pvcnn/scripts/m1_wire_contact_probe.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_contact_probe_static.py`

**Interfaces:**
- Consumes: wrapper diagnostics and `wire_acceptance_state`.
- Produces: atomic JSONL episode trace and final JSON record with contact counts, forces, slip, lift, hold, success, and terminal reason.

- [ ] **Step 1: Write failing CLI/static tests**

Require `--wire-report`, fixed seed, explicit object ID, headless operation, safe terminal handling, and atomic report replacement. Assert the probe never interprets distance as contact.

- [ ] **Step 2: Verify RED**

Run the static test and expect missing probe entrypoint/fields.

- [ ] **Step 3: Implement the probe**

Use existing Play startup/binding paths. Emit a sampled record every control interval and one final record. Preserve the interactive Play window after safe termination; exit normally for headless probe runs.

- [ ] **Step 4: Run CPU/static GREEN**

Run the static test and entrypoint tests; expect all tests to pass.

- [ ] **Step 5: Run fixed GPU0 smoke**

Use CPU thread caps and GPU0:

```bash
cd Go2Pvcnn
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1
taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python scripts/m1_wire_contact_probe.py \
  --object-id wire_000 --seed 42 --max-steps 4000 \
  --wire-report artifacts/wire_contact_seed42.jsonl --headless
```

Expected: left/right contact counts reach at least `2/2`, lift reaches `0.05 m`, hold reaches `1.0 s`, and final success is true. If it fails, preserve the report and return to root-cause investigation; do not advance to data collection.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py Go2Pvcnn/scripts/m1_wire_contact_probe.py Go2Pvcnn/tests/test_m1_wire_contact_probe_static.py
git commit -m "feat: add deterministic GPU wire contact probe"
```

### Task 5: Accepted Teacher Dataset Collection

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_collect_teacher.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_teacher_collection.py`

**Interfaces:**
- Consumes: `wire_acceptance_state` and complete episode traces.
- Produces: accepted `teacher_success.npz`, rejected `teacher_failures.npz`, `collection_report.json`, per-episode JSONL, and aggregate SHA-256 manifest.

- [ ] **Step 1: Write failing dataset-gate tests**

Assert PRELOAD-only, partial-lift, excessive-slip, safety-rejected, and non-finite episodes cannot enter success. Assert a full accepted phase sequence does enter success and retains `(seed, trial)` identity.

- [ ] **Step 2: Verify RED**

Run the focused collector tests and expect the old mission-only gate to fail the new assertions.

- [ ] **Step 3: Gate collection on physical acceptance**

Require final acceptance success plus complete `APPROACH/PRELOAD/GRASP/LIFT/HOLD` evidence. Preserve specific terminal reasons. Write outputs atomically and hash every evidence file into the aggregate manifest.

- [ ] **Step 4: Verify GREEN**

Run collector, artifact, and deterministic split tests; expect all tests to pass.

- [ ] **Step 5: Collect at least 30 GPU0 successes**

Run deterministic seeds/trials with the same CPU caps. Stop and diagnose if any success violates the acceptance record. Verify `successful_episodes >= 30` before continuing.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/scripts/m1_dual_panda_o6_collect_teacher.py Go2Pvcnn/tests/test_m1_bimanual_teacher_collection.py
git commit -m "feat: collect physically accepted wire teacher episodes"
```

### Task 6: Formal 15D Retraining and MPC Play Binding

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_train_latent.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_runtime.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_latent_training.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_latent_runtime.py`
- Test: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`

**Interfaces:**
- Consumes: at least 30 accepted success episodes and their SHA-pinned manifest.
- Produces: formal 15D artifact and `--mode latent` Play path that validates and loads it without weakening the fingertip-prior contract.

- [ ] **Step 1: Write failing artifact/binding tests**

Assert formal training rejects fewer than 30 accepted episodes, rejects partial-window provenance, records the aggregate dataset SHA, preserves `latent_dim == 15`, and Play rejects missing or mismatched metadata pins.

- [ ] **Step 2: Verify RED**

Run the three focused test files and expect failures for the missing formal-dataset gate and Play latent artifact binding.

- [ ] **Step 3: Implement formal training gate and artifact validation**

Keep warm-start explicitly separate. Formal mode loads only accepted episodes, uses group-disjoint `(seed, trial)` splits, serializes model/normalizer/metadata/metrics/split/provenance, and records SHA-256 pins.

- [ ] **Step 4: Bind the latent artifact to MPC Play**

Add a distinct latent artifact CLI pair rather than reusing `--fingertip-prior-artifact`. Validate the model before Isaac scene creation and create `LatentBimanualRuntime` only after metadata, dimensions, action order, and hashes pass.

- [ ] **Step 5: Verify GREEN and run GPU0 Play smoke**

Run all latent and entrypoint tests, then a fixed-seed headless latent Play smoke. Require no traceback, finite actions, valid 15D runtime output, and preservation of the real-contact acceptance diagnostics.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/scripts/m1_dual_panda_o6_train_latent.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/latent_runtime.py Go2Pvcnn/tests/test_m1_bimanual_latent_training.py Go2Pvcnn/tests/test_m1_bimanual_latent_runtime.py Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py
git commit -m "feat: train and play accepted 15d wire latent MPC"
```

## Final Verification

- [ ] Run the complete CPU unit and pure-solver suite with thread caps.
- [ ] Run GPU0 fixed-seed teacher and latent smoke probes.
- [ ] Run the formal accepted episode set and generate per-episode JSONL plus aggregate manifest.
- [ ] Verify every declared artifact SHA-256 from a fresh process.
- [ ] Record exact commands, commit SHA, dataset SHA, artifact SHA, pass counts, and remaining limitations in the project status document.
