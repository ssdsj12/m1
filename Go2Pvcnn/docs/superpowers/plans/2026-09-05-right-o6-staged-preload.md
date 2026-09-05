# Right O6 Staged PRELOAD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the right O6 palm base from striking the box corner by completing right-palm inward PRELOAD travel before beginning its forward travel.

**Architecture:** Keep `BimanualObjectMpc` as the sole owner of PRELOAD trajectory state. Add one right-forward scalar state, advance it only after right inward travel reaches its configured cap, and preserve the current left trajectory and fingertip-contact latch behavior. Verify the pure state transition first, then the unchanged CPU/QP contracts, and finally the GPU0 physical-contact gate.

**Tech Stack:** Python 3.11, PyTorch CPU float64 control contracts, pytest, Isaac Sim 5.1, Isaac Lab, PhysX GPU0.

## Global Constraints

- Keep the right O6 mount quaternion at WXYZ `(0, 0, 0, 1)` and the left mount at identity.
- Preserve one articulation root, 43 active controls, 53 physical DOFs, and all existing joint names/order.
- Do not change box geometry, collision filters, hand closing targets, fingertip definitions, action dimensions, solver interfaces, frequencies, dtype, or device contracts.
- Do not create a training entrypoint or implement contact MPC, perception, or Residual.
- Do not modify the T400/T500 public contracts.
- Preserve atomic last-safe fallback behavior and unrelated dirty-worktree changes.
- Do not run the formal 30-trial GPU0 layer unless the fixed-seed 1600-step physical gate passes.

---

### Task 1: Specify the right staged PRELOAD state transition

**Files:**

- Modify: `tests/test_m1_bimanual_object_mpc.py`

**Interfaces:**

- Consumes: `BimanualObjectMpc.plan(sample: ObjectMpcInput) -> ObjectMpcSolution` and `ObjectMpcCfg` PRELOAD limits.
- Produces: executable behavioral requirements for right inward-first travel, right forward travel, contact latch, APPROACH reset, and unchanged left travel.

- [ ] **Step 1: Replace the old diagonal-right assertion with an inward-only first-step assertion**

In `test_preload_moves_preclosed_noncontacting_palms_inward`, retain the left-forward calculation and change the right X assertion to:

```python
assert solution.right_palm_pose[-1, 0].item() == pytest.approx(
    snapshot.right_arm.palm_pose_b[0].item()
)
```

This is the first RED test: the existing implementation advances right X proportionally with right Y.

- [ ] **Step 2: Add a test for the inward-to-forward boundary and deterministic cap**

Add:

```python
def test_right_preload_forward_starts_only_after_inward_travel_completes() -> None:
    cfg = ObjectMpcCfg(
        preload_palm_inward_speed_m_s=0.025,
        right_preload_inward_limit_m=0.002,
        preload_forward_limit_m=0.002,
    )
    preclosed = replace(
        _hand(),
        q=torch.ones(6, dtype=DTYPE),
        contact_mask=torch.zeros(5, dtype=torch.bool),
    )
    snapshot = replace(
        _snapshot(),
        left_hand=preclosed,
        right_hand=preclosed,
        box=replace(_snapshot().box, supported=True),
    )
    planner = BimanualObjectMpc(cfg)
    entry_x = snapshot.right_arm.palm_pose_b[0].item()

    first = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.PRELOAD))
    second = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.PRELOAD))
    third = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.PRELOAD))
    fourth = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.PRELOAD))

    assert first.right_palm_pose[-1, 0].item() == pytest.approx(entry_x)
    assert second.right_palm_pose[-1, 0].item() == pytest.approx(entry_x)
    assert third.right_palm_pose[-1, 0].item() == pytest.approx(entry_x + 0.001)
    assert fourth.right_palm_pose[-1, 0].item() == pytest.approx(entry_x + 0.002)
    assert fourth.right_palm_pose[-1, 1].item() == pytest.approx(
        snapshot.right_arm.palm_pose_b[1].item() + 0.002
    )
```

At `dt=0.04`, the increment is exactly `0.001 m`. The first two calls fill the `0.002 m` inward cap; only subsequent calls advance X.

- [ ] **Step 3: Add contact-latch and APPROACH-reset coverage for forward progress**

Add one test that first fills the small inward cap and advances right X once, then supplies one true right contact bit, confirms the next no-contact call retains exactly the contacted target, sends an APPROACH sample, and confirms a following PRELOAD call starts again with zero right-forward displacement:

```python
right_contact = replace(
    preclosed,
    contact_mask=torch.tensor([False, True, False, False, False]),
)
contacted = replace(snapshot, right_hand=right_contact)
latched = planner.plan(_input(snapshot=contacted, phase=BimanualPhase.PRELOAD))
held = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.PRELOAD))
assert torch.equal(held.right_palm_pose, latched.right_palm_pose)

planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.APPROACH))
restarted = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.PRELOAD))
assert restarted.right_palm_pose[-1, 0].item() == pytest.approx(entry_x)
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_object_mpc.py -q
```

Expected: the new right-X assertions fail because current PRELOAD uses the right inward fraction to advance X diagonally; unrelated object-MPC tests pass.

---

### Task 2: Implement right inward-first, forward-second PRELOAD targets

**Files:**

- Modify: `go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py`
- Test: `tests/test_m1_bimanual_object_mpc.py`

**Interfaces:**

- Consumes: existing `_preload_inward_travel`, `_preload_contact_latched`, `preload_palm_inward_speed_m_s`, `right_preload_inward_limit_m`, and `preload_forward_limit_m`.
- Produces: internal `BimanualObjectMpc._right_preload_forward_travel: float`; no public-interface changes.

- [ ] **Step 1: Add and reset the right-forward scalar**

Initialize beside the existing PRELOAD state:

```python
self._right_preload_forward_travel = 0.0
```

Reset it in the APPROACH branch beside `_preload_inward_travel`:

```python
self._right_preload_forward_travel = 0.0
```

- [ ] **Step 2: Advance right forward travel only after inward completion**

At the start of the `fingers_preclosed` block, before the existing per-side
inward update, capture whether right inward motion was already complete:

```python
right_inward_complete_at_start = (
    self._preload_inward_travel[1]
    >= self.cfg.right_preload_inward_limit_m
)
```

After the existing per-side inward update, add:

```python
if right_inward_complete_at_start and not self._preload_contact_latched[1]:
    self._right_preload_forward_travel = min(
        self.cfg.preload_forward_limit_m,
        self._right_preload_forward_travel
        + self.cfg.preload_palm_inward_speed_m_s * self.cfg.dt,
    )
```

Capturing the condition before incrementing inward travel ensures the update
that first reaches the cap does not also advance X. This preserves the strict
phase ordering asserted by Task 1.

- [ ] **Step 3: Use the staged scalar for the right palm target**

Keep the left-forward expression unchanged and replace the right-forward expression with:

```python
right_target[0] += self._right_preload_forward_travel
```

Do not change right Y, either orientation, or either wrench.

- [ ] **Step 4: Run the object-MPC tests and verify GREEN**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest \
  tests/test_m1_bimanual_object_mpc.py -q
```

Expected: all tests pass, including unchanged left diagonal travel, right staged travel, contact latch, reset, QP feasibility, constraints, and fallback tests.

- [ ] **Step 5: Run the full CPU and pure-QP verification layers**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer cpu
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer qp
```

Expected: both commands exit zero; deterministic sampling, trajectory continuity, shape/dtype/device, mimic, fallback, atomic rejection, Arm/Hand/WBC feasibility, limits, repeatability, and last-safe-state regression checks pass.

- [ ] **Step 6: Commit the pure controller change**

Stage only the two files owned by this task:

```bash
git add tests/test_m1_bimanual_object_mpc.py \
  go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py
git commit -m "fix: stage right O6 preload approach"
```

---

### Task 3: Validate the staged path on GPU0

**Files:**

- Generate: `tests/artifacts/m1_dual_panda_o6_verification/gpu0_smoke.json`
- Generate: `tests/artifacts/m1_dual_panda_o6_right_staged_contact_diag_1600.json`
- Update: `docs/superpowers/plans/2026-09-05-right-o6-staged-preload.md`

**Interfaces:**

- Consumes: the staged right palm trajectory from Task 2 and existing diagnostic fields in `m1_dual_panda_o6_bimanual_probe.py`.
- Produces: fixed-seed physical evidence deciding whether formal GPU0 verification is allowed.

- [ ] **Step 1: Run the 200-step GPU0 smoke**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer smoke --smoke-steps 200
```

Expected: exit zero; no startup terminal, reset, nonfinite, limit, collision, or hard-safety event; object/arm/hand/WBC smoke feasibility gates pass. A short trial may remain task-incomplete because 200 steps are insufficient for bilateral contact.

- [ ] **Step 2: Run the fixed-seed 1600-step PRELOAD probe**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  --headless --device cuda:0 --steps 1600 --seed 0 \
  --report tests/artifacts/m1_dual_panda_o6_right_staged_contact_diag_1600.json
```

Expected physical gate:

```text
hard_failure_count == 0
reset_count == 0
nonfinite_count == 0
limit_violation_count == 0
contact_timing_steps.right_o6.count > 0
max_consecutive_bilateral_contact_steps >= 20
max_o6_body_contact_links.right_o6 matches right_*_(proximal|distal)
first_o6_body_contact_events.right_o6.link != right_hand_base_link
```

- [ ] **Step 3: Stop on physical failure or authorize formal verification**

If any fixed-seed physical gate fails, do not tune another rotation, collision filter, or trajectory constant. Record the exact first/max contact events and feasibility/fallback counts in this plan and stop for a new design review.

If all gates pass, mark Task 3 complete and proceed to Task 4 without changing controller parameters.

---

### Task 4: Run scoped regression and formal 30-trial verification

**Files:**

- Generate: per-trial JSONL and SHA-pinned aggregate manifest under the verifier's existing artifact directory.
- Update: `docs/superpowers/plans/2026-09-05-right-o6-staged-preload.md`

**Interfaces:**

- Consumes: fixed-seed GPU0 acceptance from Task 3 and the existing four-layer verifier CLI.
- Produces: final CPU, QP, smoke, and formal evidence without changing verifier contracts.

- [ ] **Step 1: Run the scoped bimanual regression suite**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_*.py \
  tests/test_m1_dual_panda_o6_*.py
```

Expected: all collected tests pass.

- [ ] **Step 2: Run the formal GPU0 layer**

Run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer formal
```

Expected: exactly 30 trials pass; the command writes one JSONL record per trial and an aggregate manifest pinned to source Git SHA and asset SHA256.

- [ ] **Step 3: Verify the aggregate artifact**

Read the emitted aggregate manifest and assert:

```text
expected_trial_count == 30
observed_trial_count == 30
passing_trial_count == 30
accepted == true
source Git SHA is non-empty
asset SHA256 == 69545272c2c1c6e8447d31eb2c81dea4970b16558165df9156f546928935f9c6
```

If the asset is intentionally rebuilt before this step, replace the literal SHA only after rerunning the independent 2000-step asset verifier and recording its new `hard_gates_passed: true` report.

- [ ] **Step 4: Record commands and results in the plan**

Append a dated evidence section containing command exit codes, test counts, artifact paths, trial counts, source SHA, asset SHA, contact timing, first/max right contact links, bilateral duration, and fallback counts. Do not report acceptance from console impressions alone.

- [ ] **Step 5: Commit only verification documentation explicitly owned by this task**

```bash
git add docs/superpowers/plans/2026-09-05-right-o6-staged-preload.md
git commit -m "docs: record staged O6 verification"
```
