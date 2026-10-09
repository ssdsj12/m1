# T500 Wire Reduced-Cosserat Online Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic reduced Cosserat wire predictor, couple it to object/arm/hand MPC, and make the dual-Panda/dual-O6 pipeline capable of validated wire contact, clamping, lifting, and holding.

**Architecture:** Keep the existing 43D command contract and `DeformXDualWireBackend` as the optional physical backend. Add a pure reduced dynamics layer that consumes the latest dual endpoint state and DeformX impulses, predicts centerline/endpoint behavior over the object-MPC horizon, and exposes feasibility, slip, friction, and lift references. Add those outputs at the object-MPC boundary, then gate Arm MPC safety with next-horizon recoverability before GPU validation.

**Tech Stack:** Python 3.11, PyTorch float64 CPU solver boundary, NumPy, SciPy, Isaac Lab/PhysX, existing pytest suite, optional DeformX checkout through `DEFORMX_ROOT`.

## Global Constraints

- Preserve the public 43-channel upper-body command contract.
- Preserve existing rigid-object catalog behavior.
- DeformX imports remain lazy; CPU tests must run without DeformX installed.
- Failed predictions and commands must not overwrite the last accepted safe state.
- Only real PhysX/DeformX contacts count toward grasp acceptance.
- Do not start formal 15D latent training until accepted successful wire episodes exist.
- Do not modify T400/T500 public contracts outside the wire adapter boundary.

---

### Task 1: Define the reduced Cosserat state and deterministic predictor

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_cosserat.py`
- Test: `Go2Pvcnn/tests/test_m1_reduced_cosserat.py`

**Interfaces:**
- Consumes: `DloCenterlineState`, dual endpoint poses/twists, endpoint wrench estimates, `dt`, horizon.
- Produces: `ReducedCosseratState`, `ReducedCosseratPrediction`, and `ReducedCosseratPredictor.reset/step/predict`.

- [ ] **Step 1: Write failing tests** for finite state validation, deterministic reset/step, endpoint separation, axial stretch, tangent samples, slip/friction feasibility, prediction shape `(horizon, ...)`, and rejection without state mutation.
- [ ] **Step 2: Run the focused tests** with `pytest Go2Pvcnn/tests/test_m1_reduced_cosserat.py -q`; confirm failure because the module and types do not exist.
- [ ] **Step 3: Implement the minimal pure model**: endpoint damped translation, rest-length stretch, sampled tangent transport, bounded bending proxy, friction-cone and slip checks, immutable failure diagnostics, and CPU float64 validation.
- [ ] **Step 4: Re-run focused tests** and verify all predictor tests pass without importing DeformX or Isaac.
- [ ] **Step 5: Commit** with `git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_cosserat.py Go2Pvcnn/tests/test_m1_reduced_cosserat.py && git commit -m "feat: add reduced Cosserat wire predictor"`.

### Task 2: Couple the predictor to the dual DeformX wire runtime

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/deformx_adapter.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_wire_trajectory.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_wire_runtime.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_reduced_runtime.py`

**Interfaces:**
- Consumes: `DeformXDualWireBackend`, `dual_palm_frames`, `ReducedCosseratPredictor`.
- Produces: `ReducedWireRuntime.update(snapshot, endpoint_wrenches, dt)` returning current state, horizon prediction, endpoint targets, and a machine-readable failure reason.

- [ ] **Step 1: Add failing tests** for lazy backend operation, bind-once semantics, common M1 base-frame conversion, impulse-to-wrench mapping, deterministic update, and invalid prediction fallback.
- [ ] **Step 2: Run the focused runtime tests** and confirm they fail before the runtime facade exists.
- [ ] **Step 3: Implement the runtime facade** without changing existing backend contracts: update DeformX only when configured, convert impulses to bounded endpoint wrench estimates, feed the reduced predictor, and retain the last safe state on rejection.
- [ ] **Step 4: Update `DloWireTrajectoryPlanner`** to consume predicted endpoint targets and lift velocity rather than only the latest centerline geometry.
- [ ] **Step 5: Run runtime and existing wire coupling tests** with `pytest Go2Pvcnn/tests/test_m1_wire_reduced_runtime.py Go2Pvcnn/tests/test_m1_wire_coupling.py Go2Pvcnn/tests/test_m1_wire_scene_geometry.py -q`.
- [ ] **Step 6: Commit** with `git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/deformx_adapter.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_wire_trajectory.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/reduced_wire_runtime.py Go2Pvcnn/tests/test_m1_wire_reduced_runtime.py && git commit -m "feat: couple reduced wire dynamics to DeformX runtime"`.

### Task 3: Add wire dynamics and contact constraints to object MPC

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_object_mpc.py`

**Interfaces:**
- Consumes: `ReducedCosseratPrediction` from Task 1/2.
- Produces: extended `ObjectMpcInput` wire prediction field and `ObjectMpcDiagnostics` fields for stretch, slip, friction, and lift feasibility.

- [ ] **Step 1: Write failing tests** for wire endpoint reference tracking, finite hard constraints, force allocation, friction/slip veto, synchronized lift velocity, and deterministic fallback to the last safe object solution.
- [ ] **Step 2: Run the focused tests** and confirm the new wire fields/constraints are missing.
- [ ] **Step 3: Implement the minimal coupling**: keep rigid-object behavior unchanged, add wire-only costs and constraints, reject infeasible wire horizons, and preserve existing QP backend behavior.
- [ ] **Step 4: Run focused object-MPC and regression tests** including `test_m1_wire_coupling.py`, `test_m1_wire_contact_acceptance.py`, and the existing QP tests.
- [ ] **Step 5: Commit** with `git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contracts.py Go2Pvcnn/tests/test_m1_wire_object_mpc.py && git commit -m "feat: constrain wire object MPC with reduced dynamics"`.

### Task 4: Add Arm MPC next-horizon recoverability gating

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/safety_projection.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_arm_recoverability.py`

**Interfaces:**
- Consumes: current Panda `q/qd`, configured arm limits, candidate command, and arm MPC horizon settings.
- Produces: deterministic `recoverability_ok` result and explicit `arm_mpc_recoverability` rejection reason.

- [ ] **Step 1: Write a failing regression test** from the captured seed-42 left-arm state where safety accepts the current command but the next horizon is infeasible because joint-6 velocity/position and acceleration bounds conflict.
- [ ] **Step 2: Run the test** and confirm the current safety layer accepts the unrecoverable command.
- [ ] **Step 3: Implement the smallest predictive braking/recoverability check** at the command boundary; reject only commands that cannot brake into the configured position/velocity envelope over the next horizon.
- [ ] **Step 4: Verify the captured counterexample is rejected, safe commands remain accepted, and all Arm/WBC/QP regression tests pass.
- [ ] **Step 5: Commit** with `git add Go2Pvcnn/go2_pvcnn/control/m1_panda_coordination/arm_mpc.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/safety_projection.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py Go2Pvcnn/tests/test_m1_wire_arm_recoverability.py && git commit -m "fix: gate arm commands by horizon recoverability"`.

### Task 5: Integrate staged wire contact, clamp, lift, and hold behavior

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/grasp_lift_pipeline.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/diagnostics.py`
- Test: `Go2Pvcnn/tests/test_m1_wire_reduced_pipeline.py`

**Interfaces:**
- Consumes: runtime prediction/feasibility, real contact observations, object MPC diagnostics.
- Produces: explicit `APPROACH/PRELOAD/GRASP/LIFT/HOLD` transitions and terminal reasons for dynamics/slip/recoverability failures.

- [ ] **Step 1: Add failing transition tests** for bilateral contact gating, per-digit latch, clamp-force hold, lift pause on contact loss, slip veto, and successful hold.
- [ ] **Step 2: Run focused pipeline tests** and confirm the new dynamics gates are absent.
- [ ] **Step 3: Implement transitions and diagnostics** while preserving atomic command rejection and existing rigid-object phases.
- [ ] **Step 4: Run all wire CPU/QP tests** and verify deterministic JSONL diagnostics.
- [ ] **Step 5: Commit** with `git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/grasp_lift_pipeline.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/constraints.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/diagnostics.py Go2Pvcnn/tests/test_m1_wire_reduced_pipeline.py && git commit -m "feat: gate wire lift on reduced dynamics and contact"`.

### Task 6: GPU0 smoke and physical acceptance probe

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py` only if required for runtime wiring.
- Test/artifacts: `Go2Pvcnn/tests/test_m1_wire_contact_probe_static.py`, `Go2Pvcnn/artifacts/`.

**Interfaces:**
- Consumes: Tasks 1–5 runtime and safety contracts.
- Produces: one deterministic GPU0 JSONL probe showing bilateral contact, clamp, lift, hold, or an explicit reproducible blocker.

- [ ] **Step 1: Run CPU and pure-QP gates** before opening Isaac; do not spend GPU time if a focused regression fails.
- [ ] **Step 2: Run the fixed GPU0 wire probe** with seed 42 and diagnostics enabled.
- [ ] **Step 3: Inspect terminal phase, per-hand contact counts/forces, lift height, slip, Arm MPC feasibility, and safety reasons.
- [ ] **Step 4: If accepted, record the JSONL SHA and update the readiness report; if rejected, preserve the artifact and document the first causal blocker without claiming training readiness.
- [ ] **Step 5: Commit only owned diagnostics/report updates**; never stage unrelated assets or user changes.

## Verification Matrix

- `pytest Go2Pvcnn/tests/test_m1_reduced_cosserat.py -q`
- `pytest Go2Pvcnn/tests/test_m1_wire_reduced_runtime.py Go2Pvcnn/tests/test_m1_wire_object_mpc.py Go2Pvcnn/tests/test_m1_wire_arm_recoverability.py Go2Pvcnn/tests/test_m1_wire_reduced_pipeline.py -q`
- Existing wire/QP regression suite from `Go2Pvcnn/tests/test_m1_wire_*.py` and arm/WBC/QP tests.
- GPU0 fixed wire probe with JSONL diagnostics and SHA-256 evidence.
