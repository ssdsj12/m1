# Right Palm Single-Axis Orientation MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 25 Hz object-MPC hierarchy choose and smoothly plan one right-palm local rotation axis so an O6 digit reaches the box before `right_hand_base_link`, while preserving the existing 6D palm-pose and T400/T500 contracts.

**Architecture:** First pin a conservative palm-housing support model to the normalized right O6 mesh. Implement pure CPU SO(3) and geometric candidate selection in a new module, then add a scalar 25-node QP with atomic last-safe fallback. Integrate the returned rotation-vector horizon into `BimanualObjectMpc`, route reset and diagnostics through existing runtime/probe boundaries, and permit formal GPU0 verification only after the fixed-seed fingertip-contact gate passes.

**Tech Stack:** Python 3.11, PyTorch CPU float64, existing deterministic dense active-set QP backend, pytest, trimesh, USD/pxr, Isaac Sim 5.1, Isaac Lab, PhysX GPU0.

## Global Constraints

- Run only one active orientation axis (`K=1`); keep the internal basis interface compatible with `K=3`.
- Keep the left O6 mount at WXYZ `(1, 0, 0, 0)` and the right O6 mount at `(0, 0, 0, 1)`.
- Preserve one articulation root, 43 active controls, 53 physical DOFs, and all existing joint names and ordering.
- Preserve `ObjectMpcSolution.left_palm_pose` and `right_palm_pose` as CPU-float64 `(25, 6)` tensors.
- Compose orientations on SO(3); never add rotation-vector components directly.
- Keep the staged right PRELOAD translation, left palm behavior, box geometry `(0.20, 0.18, 0.10) m`, collision filters, O6 closing targets, and real fingertip Jacobians unchanged.
- Preserve atomic last-safe fallback; rejected input must not mutate selected axis, committed angle, contact latch, or last-safe trajectory.
- Do not create a training entrypoint or implement contact MPC, perception, or Residual.
- Do not modify T400/T500 public contracts.
- Preserve unrelated dirty-worktree changes and do not stage entire overlapping files without reviewing their pre-existing diff.
- Do not run the formal 30-trial GPU0 layer unless the fixed-seed 1600-step physical gate passes.

---

### Task 1: Pin the right palm-housing support geometry

**Files:**

- Modify: `tests/test_m1_dual_panda_o6_asset_static.py`
- Modify: `scripts/build_m1_dual_panda_o6_asset.py`
- Modify: `scripts/verify_m1_dual_panda_o6_asset.py`
- Regenerate: `assets/m1_dual_panda_o6/asset_manifest.json`

**Interfaces:**

- Consumes: `assets/m1_dual_panda_o6/o6_right/meshes/hand_base_link.STL` and `source_manifest.json` SHA entry.
- Produces: manifest object `right_palm_housing_support` with `mesh_sha256`, `bounds_min_m`, `bounds_max_m`, `support_points_local_m`, and `valid`.

- [ ] **Step 1: Write a failing static contract test**

Add AST/source assertions that both builder and verifier independently define:

```python
RIGHT_PALM_HOUSING_MESH_SHA256 = (
    "f7fc8dae5d375e5a33251593c46f8b882c3a1aaafe89fa9544f0ef57d657f5de"
)
RIGHT_PALM_HOUSING_BOUNDS_MIN_M = (-0.0200, -0.0392, 0.0)
RIGHT_PALM_HOUSING_BOUNDS_MAX_M = (0.0200, 0.0376, 0.1128)
```

Require the tokens `right_palm_housing_support`, `support_points_local_m`, and
`offline["right_palm_housing_support_valid"]` in the independent verifier hard
gate. Also assert the generated manifest contains exactly eight finite 3D
support points and the expected mesh SHA.

- [ ] **Step 2: Run the focused test and observe RED**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_asset_static.py
```

Expected: failure because no housing-support contract exists.

- [ ] **Step 3: Add deterministic mesh-bound extraction to the builder**

Use `trimesh.load_mesh(path, process=False)`, reject a scene/nonfinite/empty
mesh, and compute the raw bounds. Enclose them with the approved outward-rounded
bounds above, rejecting any raw coordinate outside by more than `1e-9 m`.
Generate corners without storing a dense mesh:

```python
def _aabb_corners(
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> list[list[float]]:
    return [
        [x, y, z]
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]
```

Write `right_palm_housing_support` into the manifest with the normalized source
mesh SHA read from `source_manifest.json`; require it equals the independent
constant before emitting JSON.

- [ ] **Step 4: Implement the independent offline verifier**

Recompute the source-mesh hash and trimesh bounds without importing builder
constants. Require raw bounds lie inside the expected outward bounds and that
the manifest's eight corners equal the independently generated corners. Return:

```python
{
    "right_palm_housing_support_valid": valid,
    "right_palm_housing_support": {
        "mesh_sha256": measured_sha,
        "bounds_min_m": list(measured_min),
        "bounds_max_m": list(measured_max),
        "support_points_local_m": support_points,
    },
}
```

Include `offline["right_palm_housing_support_valid"]` in
`hard_gates_passed`.

- [ ] **Step 5: Rebuild and independently verify the serialized asset**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/build_m1_dual_panda_o6_asset.py \
  --headless --device cuda:0 --asset-root assets/m1_dual_panda_o6
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_dual_panda_o6_asset.py \
  --headless --device cuda:0 --steps 2000 \
  --asset-root assets/m1_dual_panda_o6
```

Expected: both commands exit zero; the support contract is valid,
`hard_gates_passed=true`, mount calibration remains valid, and the articulation
remains 43 active / 53 physical DOFs.

- [ ] **Step 6: Run static asset regression**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_asset_static.py
```

Expected: all tests pass.

---

### Task 2: Implement pure SO(3) and support-margin geometry

**Files:**

- Create: `go2_pvcnn/control/m1_bimanual_coordination/palm_orientation_mpc.py`
- Create: `tests/test_m1_bimanual_palm_orientation_mpc.py`

**Interfaces:**

- Produces: `rotvec_to_matrix(rotvec: Tensor) -> Tensor`, `matrix_to_rotvec(matrix: Tensor) -> Tensor`, `compose_orientation_horizon(entry_rotvec_b: Tensor, basis_local: Tensor, coefficients: Tensor) -> Tensor`, `nearest_point_on_oriented_box(palm_position_b: Tensor, box_pose_b: Tensor, box_half_extents_m: Tensor) -> Tensor`, `evaluate_lead_margin(...) -> PalmLeadGeometry`, and `select_single_axis(...) -> PalmAxisSelection`.
- Tensor contract: CPU float64, `entry_rotvec_b (3,)`, `basis_local (K,3)`, `coefficients (H,K)`, fingertips `(5,3)`, housing points `(8,3)`.
- Runtime configuration embeds the independently verified eight AABB corners;
  it does not parse USD, STL, or JSON in the control loop.

- [ ] **Step 1: Write failing SO(3) and future-basis tests**

Add tests that require exact identity round-trip, a `pi/2` Z rotation, and
composition for both `K=1` and a constructible `K=3` basis:

```python
def test_compose_orientation_horizon_supports_one_and_three_axis_bases() -> None:
    entry = torch.zeros(3, dtype=DTYPE)
    one = compose_orientation_horizon(
        entry,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE),
        torch.tensor([[0.0], [math.pi / 2.0]], dtype=DTYPE),
    )
    three = compose_orientation_horizon(
        entry,
        torch.eye(3, dtype=DTYPE),
        torch.zeros((2, 3), dtype=DTYPE),
    )
    assert one.shape == (2, 3)
    assert torch.allclose(
        rotvec_to_matrix(one[-1]),
        _z_rotation(math.pi / 2.0),
        atol=1.0e-10,
    )
    assert torch.equal(three, torch.zeros((2, 3), dtype=DTYPE))
```

Also reject GPU/non-float64 tensors, zero/non-unit basis rows, non-orthogonal
multi-axis bases, and mismatched `K`. Verify equivalent axis-angle
representations at the pi boundary by comparing their rotation matrices, not
their raw rotation-vector signs.

- [ ] **Step 2: Run SO(3) tests and observe RED**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_palm_orientation_mpc.py -k "so3 or compose"
```

Expected: import/collection failure because the module is absent.

- [ ] **Step 3: Implement stable SO(3) helpers**

Use Rodrigues' formula with a small-angle series and clamp matrix-trace cosine
to `[-1, 1]`. For matrix-to-rotation-vector conversion, use the skew part away
from pi and a deterministic diagonal-based axis for the pi neighborhood. Always
return finite CPU float64 tensors and validate basis rows with:

```python
gram = basis_local @ basis_local.T
if not torch.allclose(
    gram,
    torch.eye(basis_local.shape[0], dtype=torch.float64),
    atol=1.0e-10,
    rtol=0.0,
):
    raise ValueError("basis_local rows must be orthonormal")
```

Compose each node as `R_entry @ Exp(sum_k basis[k] * coefficient[k])`; never
sum output rotation vectors.

- [ ] **Step 4: Write failing oriented-box and lead-margin tests**

Use an identity box with half extents `(0.10, 0.09, 0.05)`, a palm outside its
negative-Y face, synthetic fingertip offsets, and the eight calibrated housing
corners. Assert nearest-point clamping, deterministic lowest-index digit ties,
and selection of the canonical axis/angle with the largest margin. Include a
case where zero is optimal and a case where all three axes tie so X wins.

- [ ] **Step 5: Implement geometry and deterministic candidate selection**

Define frozen results:

```python
@dataclass(frozen=True)
class PalmLeadGeometry:
    lead_margin_m: float
    digit_support_m: float
    housing_support_m: float
    leading_digit_index: int

@dataclass(frozen=True)
class PalmAxisSelection:
    axis_local: torch.Tensor
    target_angle_rad: float
    geometry: PalmLeadGeometry
```

Candidate angles are `torch.linspace(-0.35, 0.35, 29, dtype=torch.float64)`.
Rank candidates by `(-lead_margin, abs(angle), axis_index, angle)` using Python
finite floats, which fixes tie behavior independently of tensor reduction
ordering. Use the current palm orientation to convert measured fingertip
offsets to palm-local coordinates on every call. Embed these verified defaults
as immutable CPU-float64 configuration tensors:

```python
box_half_extents_m = (0.10, 0.09, 0.05)
housing_support_points_local_m = tuple(
    (x, y, z)
    for x in (-0.0200, 0.0200)
    for y in (-0.0392, 0.0376)
    for z in (0.0, 0.1128)
)
```

The asset static test must compare this literal runtime support set with the
manifest's independently verified set, preventing calibration drift without
introducing runtime file I/O.

- [ ] **Step 6: Run the pure geometry tests and verify GREEN**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_palm_orientation_mpc.py
```

Expected: all SO(3), oriented-box, geometry, determinism, dtype/device, and
`K=3` construction tests pass.

---

### Task 3: Add the scalar 25-node orientation QP and atomic controller

**Files:**

- Modify: `go2_pvcnn/control/m1_bimanual_coordination/palm_orientation_mpc.py`
- Modify: `tests/test_m1_bimanual_palm_orientation_mpc.py`

**Interfaces:**

- Produces: `PalmOrientationMpcCfg`, `PalmOrientationInput`, `PalmOrientationDiagnostics`, `PalmOrientationSolution`, `build_palm_orientation_qp(...)`, and `RightPalmOrientationMpc.plan/reset`.
- `PalmOrientationSolution.orientation_rotvec_b` has shape `(25,3)` and `angle_rad` has shape `(25,)`.

- [ ] **Step 1: Write failing QP bound, continuity, and repeatability tests**

Freeze defaults:

```python
cfg = PalmOrientationMpcCfg()
assert cfg.dt == pytest.approx(0.04)
assert cfg.horizon_steps == 25
assert cfg.theta_max_rad == pytest.approx(0.35)
assert cfg.angular_rate_max_rad_s == pytest.approx(0.35)
assert cfg.candidate_count == 29
assert cfg.active_basis_dim == 1
assert cfg.tracking_weight == pytest.approx(20.0)
assert cfg.slew_weight == pytest.approx(2.0)
assert cfg.smoothness_weight == pytest.approx(5.0)
assert cfg.regularization == pytest.approx(1.0e-8)
assert cfg.qp_tolerance == pytest.approx(1.0e-8)
assert cfg.qp_max_iterations == 256
```

For `current_angle=0.10` and `target_angle=0.35`, solve twice and assert bitwise
equal solutions, `abs(theta)<=0.35`, and:

```python
first_delta = solution[0] - 0.10
deltas = torch.cat((first_delta.reshape(1), torch.diff(solution)))
assert torch.all(deltas.abs() <= 0.35 * 0.04 + 1.0e-10)
```

- [ ] **Step 2: Run the focused QP tests and observe RED**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_palm_orientation_mpc.py -k "qp or bound or repeat"
```

Expected: failure because the QP API is absent.

- [ ] **Step 3: Implement the exact scalar QP**

Let `D1[0,0]=1`; for `k>0`, let `D1[k,k]=1` and `D1[k,k-1]=-1`.
Let `b=[current_angle,0,...,0]`. Let `D2=D1[1:] - D1[:-1]`, giving
the 24-by-25 adjacent-increment difference matrix. Build:

```python
H = 2.0 * (
    cfg.tracking_weight * eye
    + cfg.slew_weight * D1.T @ D1
    + cfg.smoothness_weight * D2.T @ D2
    + cfg.regularization * eye
)
g = (
    -2.0 * cfg.tracking_weight * target_angle * torch.ones(horizon, dtype=DTYPE)
    -2.0 * cfg.slew_weight * D1.T @ b
    -2.0 * cfg.smoothness_weight * D2.T @ (b[1:] - b[:-1])
)
delta = cfg.angular_rate_max_rad_s * cfg.dt
A = torch.cat((D1, -D1), dim=0)
delta_vector = delta * torch.ones(horizon, dtype=DTYPE)
u = torch.cat((delta_vector + b, delta_vector - b), dim=0)
```

Use empty equality rows, bounds `[-theta_max_rad,+theta_max_rad]`, and the
existing `solve_reference_qp` with `cfg.qp_tolerance=1e-8` and
`cfg.qp_max_iterations=256`. Define the frozen configuration completely as:

```python
@dataclass(frozen=True)
class PalmOrientationMpcCfg:
    dt: float = 0.04
    horizon_steps: int = 25
    theta_max_rad: float = 0.35
    angular_rate_max_rad_s: float = 0.35
    candidate_count: int = 29
    active_basis_dim: int = 1
    tracking_weight: float = 20.0
    slew_weight: float = 2.0
    smoothness_weight: float = 5.0
    regularization: float = 1.0e-8
    qp_tolerance: float = 1.0e-8
    qp_max_iterations: int = 256
```

Reject configuration values that are nonfinite, nonpositive where required,
or have `active_basis_dim != 1`. The low-level basis validation and SO(3)
composition accept `K=3`; the current scalar controller deliberately does not.
This preserves the three-axis extension seam without pretending the scalar QP
already implements three coupled coordinates.

- [ ] **Step 4: Write failing controller lifecycle and atomicity tests**

Construct `PalmOrientationInput` with exact CPU float64 shapes:

```python
@dataclass(frozen=True)
class PalmOrientationInput:
    palm_pose_b: torch.Tensor             # (6,)
    fingertip_positions_b: torch.Tensor   # (5,3)
    box_pose_b: torch.Tensor              # (6,)
    contact_mask: torch.Tensor            # bool (5,)
    phase: BimanualPhase
```

Test initial APPROACH axis selection, PRELOAD replanning, first selected-contact
latch followed by contact loss, explicit `reset()`, invalid nonfinite input,
forced QP failure, first-cycle measured-orientation hold, and preservation of
all internal state after a rejected proposal. Contact latch must project the
currently measured relative rotation onto the selected axis at the first
selected-fingertip contact and return that constant angle over all 25 nodes; it
must not jump to a stale endpoint computed on an earlier update.

- [ ] **Step 5: Implement the stateful controller**

Store `_entry_rotation`, `_axis_local`, `_committed_angle`, `_contact_latched`,
and `_last_safe`. Compute all candidate state in local variables. Only after a
finite successful QP and SO(3) conversion assign state and clone the solution.
On failure, return a cloned last-safe solution with diagnostics
`feasible=False`, or the measured orientation repeated 25 times on the first
cycle. APPROACH and PRELOAD replan; GRASP, LIFT, HOLD, and RELEASE repeat the
latched/last contact-safe orientation. `reset()` clears every stored field.

- [ ] **Step 6: Run the full orientation module tests and verify GREEN**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_palm_orientation_mpc.py
```

Expected: all tests pass with exact shape/dtype/device, deterministic QP,
bounded rates, contact latch, reset, fallback, and atomic rejection.

- [ ] **Step 7: Commit the isolated new module and tests**

After reviewing both complete diffs, stage only the two new files:

```bash
git add \
  go2_pvcnn/control/m1_bimanual_coordination/palm_orientation_mpc.py \
  tests/test_m1_bimanual_palm_orientation_mpc.py
git diff --cached --check
git commit -m "feat: add right palm orientation mpc"
```

Do not stage any pre-existing dirty file in this checkpoint.

---

### Task 4: Integrate orientation planning atomically into Object MPC

**Files:**

- Modify: `go2_pvcnn/control/m1_bimanual_coordination/object_mpc.py`
- Modify: `go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Modify: `tests/test_m1_bimanual_object_mpc.py`
- Modify: `tests/test_m1_bimanual_runtime.py`
- Modify: `tests/test_m1_bimanual_dual_arm_mpc.py`
- Modify: `tests/test_m1_bimanual_full_action_teacher.py`

**Interfaces:**

- Consumes: `RightPalmOrientationMpc.plan(PalmOrientationInput) -> PalmOrientationSolution`.
- Produces: existing `ObjectMpcSolution.right_palm_pose[:,3:]` populated by SO(3) orientation MPC; no field or shape changes.

- [ ] **Step 1: Write failing Object-MPC integration tests**

Inject a recording orientation planner through an optional constructor argument:

```python
planner = BimanualObjectMpc(right_orientation_mpc=recording_planner)
solution = planner.plan(_input(snapshot=snapshot, phase=BimanualPhase.APPROACH))
assert torch.equal(
    solution.right_palm_pose[:, 3:],
    recording_planner.solution.orientation_rotvec_b,
)
assert torch.equal(solution.left_palm_pose[:, 3:], original_left_orientation)
```

Also assert the planner receives the latest right palm pose, five fingertip
positions, box pose, contact mask, and phase. A fake infeasible orientation
solution must make the containing object solution use reason
`right_palm_orientation_infeasible` without updating object `_last_safe`.

- [ ] **Step 2: Run integration tests and observe RED**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_runtime.py
```

Expected: failure because `BimanualObjectMpc` has no orientation dependency or
reset routing.

- [ ] **Step 3: Decorate candidate object solutions before committing**

Add `right_orientation_mpc: RightPalmOrientationMpc | None = None` to the
constructor and default it internally. Implement one helper that builds
`PalmOrientationInput`, calls the planner, clones the candidate right palm pose,
and replaces only `[:,3:]`. APPROACH, supported PRELOAD, and normal object-QP
paths must all call this helper before assigning `_last_safe`.

If orientation diagnostics are infeasible, call the existing object fallback
with `right_palm_orientation_infeasible`. Ensure the orientation planner itself
has already preserved its last-safe state, and do not commit the undecorated
object candidate.

- [ ] **Step 4: Route deterministic reset through the common runtime**

Add `BimanualObjectMpc.reset()` to clear PRELOAD translation state,
orientation planner state, and object last-safe state. In `BimanualRuntime.reset`
call it when available:

```python
reset_object = getattr(self.object_mpc, "reset", None)
if callable(reset_object):
    reset_object()
```

Test two reset/replay sequences for identical selected axis and orientation
horizon. Do not require injected legacy test doubles to implement `reset()`.

- [ ] **Step 5: Prove unchanged downstream contracts**

Extend dual-arm and teacher tests with nonzero right rotation-vector targets.
Assert 25-to-50 Hz resampling preserves `(20,6)`, teacher right palm trajectory
remains `(25,12)`, full action remains `(25,43)`, and
`ArmMpcInput.__dataclass_fields__` is unchanged.

- [ ] **Step 6: Run integration regression and verify GREEN**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_object_mpc.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_dual_arm_mpc.py \
  tests/test_m1_bimanual_full_action_teacher.py
```

Expected: all tests pass and no public dataclass field changes.

- [ ] **Step 7: Review overlapping-file diffs before committing**

Use `git diff -- <each Task 4 file>` and separate the orientation changes from
pre-existing work. Commit only when every staged hunk belongs to this task:

```bash
git diff --cached --check
git commit -m "feat: integrate palm orientation into object mpc"
```

If an overlapping hunk cannot be separated safely, leave it unstaged and note
the deferred path in this plan's evidence section; never use `git add -A`.

---

### Task 5: Expose orientation and contact-order diagnostics

**Files:**

- Modify: `scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `tests/test_m1_dual_panda_o6_entrypoints_static.py`

**Interfaces:**

- Consumes: `wrapper.runtime.object_mpc.right_orientation_mpc.last_diagnostics` and existing first/max O6 contact events.
- Produces: JSON fields `right_palm_orientation_mpc`, `first_right_palm_base_contact_orientation`, and `first_right_selected_fingertip_contact_orientation`.

- [ ] **Step 1: Write a failing static report-contract test**

Require the probe source to emit:

```python
"right_palm_orientation_mpc"
"basis_dimension"
"selected_axis_local"
"candidate_angle_rad"
"committed_angle_rad"
"lead_margin_m"
"leading_digit_index"
"housing_support_m"
"digit_support_m"
"qp_feasible"
"qp_iterations"
"fallback_reason"
"first_right_palm_base_contact_orientation"
"first_right_selected_fingertip_contact_orientation"
```

- [ ] **Step 2: Run the static test and observe RED**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_entrypoints_static.py
```

Expected: failure because orientation diagnostics are absent.

- [ ] **Step 3: Add read-only diagnostic snapshots**

At each step copy primitive values from the last diagnostics; never retain live
tensors. Attach the snapshot to the existing first/max contact event path and
separately latch the first right base-link and first selected right fingertip
events. Use `None` before an event. Diagnostic collection must not alter control
state or contact masks.

- [ ] **Step 4: Run the entrypoint contract test and verify GREEN**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_dual_panda_o6_entrypoints_static.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit diagnostics only after staged-diff review**

Review the probe and static-test staged hunks, run `git diff --cached --check`,
and commit only task-owned hunks with:

```bash
git commit -m "feat: report right palm orientation diagnostics"
```

Leave inseparable pre-existing hunks unstaged and record them as deferred.

---

### Task 6: Run CPU, pure-QP, and scoped regression gates

**Files:**

- Update: `docs/superpowers/plans/2026-09-06-right-palm-orientation-mpc.md`

**Interfaces:**

- Consumes: Tasks 1–5.
- Produces: recorded test counts and authorization for GPU0 smoke.

- [ ] **Step 1: Run CPU verification**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer cpu
```

Expected: exit zero; deterministic sampling, trajectories, shape/dtype/device,
Hand MPC constraints, mimic, fallback, and atomic rejection all pass.

- [ ] **Step 2: Run pure-QP verification**

Add `tests/test_m1_bimanual_palm_orientation_mpc.py` to the verifier's QP test
list, then run:

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer qp
```

Expected: exit zero; Object/Arm/Hand/orientation/WBC QPs are feasible, bounded,
repeatable, and preserve last-safe state on invalid input.

- [ ] **Step 3: Run all scoped bimanual tests**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python -m pytest -q \
  tests/test_m1_bimanual_*.py \
  tests/test_m1_dual_panda_o6_*.py
```

Expected: every collected test passes.

- [ ] **Step 4: Record exact commands and counts**

Append a dated evidence section to this plan with command exit codes and exact
pass counts. Stop before GPU0 if any prerequisite fails.

---

### Task 7: Run GPU0 geometry, smoke, and physical-contact gates

**Files:**

- Generate: `tests/artifacts/m1_dual_panda_o6_right_orientation_geometry_3.json`
- Generate: `tests/artifacts/m1_dual_panda_o6_verification/gpu0_smoke.json`
- Generate: `tests/artifacts/m1_dual_panda_o6_right_orientation_contact_1600.json`
- Update: `docs/superpowers/plans/2026-09-06-right-palm-orientation-mpc.md`

**Interfaces:**

- Consumes: verified orientation controller and diagnostics.
- Produces: physical acceptance decision for formal 30-trial verification.

- [ ] **Step 1: Run three-step geometry smoke**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  --headless --device cuda:0 --steps 3 --seed 0 \
  --report tests/artifacts/m1_dual_panda_o6_right_orientation_geometry_3.json
```

Expected: chosen axis has basis dimension 1, committed angle moves in the
candidate direction within the rate bound, predicted lead margin improves over
the zero-angle candidate, and no reset/nonfinite/hard failure occurs.

- [ ] **Step 2: Run 200-step GPU0 smoke**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer smoke --smoke-steps 200
```

Expected: exit zero; coordinates, table clearance, minimum spacing, frequencies,
control direction, finiteness, limits, resets, and solver feasibility pass.

- [ ] **Step 3: Run fixed-seed 1600-step PRELOAD gate**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_probe.py \
  --headless --device cuda:0 --steps 1600 --seed 0 \
  --report tests/artifacts/m1_dual_panda_o6_right_orientation_contact_1600.json
```

Require all of:

```text
hard_failure_count == 0
reset_count == 0
nonfinite_count == 0
limit_violation_count == 0
contact_timing_steps.right_o6.count > 0
max_consecutive_bilateral_contact_steps >= 20
first_o6_body_contact_events.right_o6.link matches right_*_(proximal|distal)
max_o6_body_contact_links.right_o6 matches right_*_(proximal|distal)
first_right_palm_base_contact_orientation is null or occurs after fingertip contact
```

- [ ] **Step 4: Stop or authorize formal verification**

If any physical gate fails, record the selected axis, candidate/committed angle,
lead margin, palm tracking errors, first/max contact links, contact counts, and
fallback reasons, then stop. Do not tune a second axis, collision filter,
housing bounds, or angle/rate constant without a new design review.

If every gate passes, proceed to Task 8 without changing controller constants.

---

### Task 8: Run formal 30-trial GPU0 verification

**Files:**

- Generate: `tests/artifacts/m1_dual_panda_o6_verification/formal_trials.jsonl`
- Generate: `tests/artifacts/m1_dual_panda_o6_verification/formal_report.json`
- Generate: `tests/artifacts/m1_dual_panda_o6_verification/formal_aggregate.manifest.json`
- Update: `docs/superpowers/plans/2026-09-06-right-palm-orientation-mpc.md`

**Interfaces:**

- Consumes: Task 7 physical acceptance.
- Produces: 30/30 SHA-pinned formal acceptance evidence.

- [ ] **Step 1: Run formal verification**

```bash
PYTHONPATH="$PWD" /home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_verify.py --layer formal
```

Expected: exactly 30 trials pass and the command exits zero.

- [ ] **Step 2: Validate aggregate pins**

Read the manifest and require:

```text
aggregate.expected_trial_count == 30
aggregate.observed_trial_count == 30
aggregate.passing_trial_count == 30
aggregate.accepted == true
pins.git_ref is non-empty
pins.source_sha256 is non-empty
pins.asset_sha256 equals the independently verified current asset SHA
pins.trials_jsonl.sha256 matches formal_trials.jsonl
pins.aggregate_report.sha256 matches formal_report.json
```

- [ ] **Step 3: Record final evidence**

Append exact artifact paths, trial counts, source Git SHA, source SHA256, asset
SHA256, and aggregate status to this plan. Do not claim completion from console
output without reading the artifacts.

## Inline execution evidence — 2026-09-06

- Baseline: 49 focused tests passed before production changes.
- Asset RED: 2 expected missing-contract failures; GREEN: 12 static tests passed.
- Rebuild completed and the independent GPU0 2000-step verifier exited 0.
  Manifest records 53 measured physical DOFs, 43 active controls and 2000 steps.
  Asset SHA remains `69545272c2c1c6e8447d31eb2c81dea4970b16558165df9156f546928935f9c6`.
- Implemented pure SO(3), oriented-box geometry, canonical-axis selection,
  25-node scalar QP, reset, measured-contact hold and last-safe rejection.
  Orientation module tests: 29 passed, including forced QP rejection.
- Corrected the QP linear term for second differences: the boundary offset
  must be `b[1:] - b[:-1]`. A constant-angle gradient regression verifies this.
- Object integration restores PRELOAD translation state on orientation failure.
  Nonzero orientation forwarding is tested in APPROACH, supported PRELOAD and
  the normal HOLD object QP. Combined object/orientation tests: 48 passed.
- CPU layer: 77 passed; QP layer: 69 passed at its first run (before adding
  six orientation checks and three object forwarding cases).
- GPU0 geometry: 3 steps, exit 0; local X selected, candidate 0.35 rad,
  first committed angle 0.014 rad, candidate lead margin 0.00088238 m.
- GPU0 200-step smoke: exit 0, report `passed=true`, zero hard failures,
  nonfinite values and resets; Object feasible rate 1.0, WBC rate 0.99.
- Final scoped regression: 184 passed; final QP layer: 78 passed.
- New standalone module/tests committed as `e6a012c`; integration and asset changes
  remain in the pre-existing dirty worktree.
- Physical 1600-step gate **FAILED**. Probe exit code was 0 (smoke runner completion),
  but the stricter physical acceptance predicates below failed. Formal 30 trials
  were not run.
  - Artifact: `tests/artifacts/m1_dual_panda_o6_right_orientation_contact_1600.json`.
  - Right selected fingertip contacts: 0; maximum consecutive bilateral steps: 0.
  - First right body contact: step 940, `right_hand_base_link`, 37.50675 N.
  - Maximum right contact link also `right_hand_base_link`.
  - First arm infeasibility: step 184 (right arm).
  - First limit event: step 270, `left_pinky_mcp_pitch`, position 0.066318 rad
    below minimum 0.080000 rad by 0.013682 rad.
  - First safety rejection: step 329, `safety_qp_infeasible`.
  - Hard failure count 1126; limit event count 1125; reset/nonfinite counts 0.
  - WBC feasible rate 0.8575; teacher feasible rate 0.41.
  - Final right palm orientation error 1.781095 rad.
  - At first right housing contact: local X axis; target -0.35 rad, committed
    0.021973 rad; candidate lead margin -0.021690 m.
  - Interpretation: the scalar QP remains feasible, but geometric candidate
    optimization alone has not ensured arm tracking or safe physical contact.
    The report does not establish a unique root cause. Stop at this physical
    gate as specified in Task 7; do not silently tune limits or add another axis.
- Existing overlapping files include earlier uncommitted work. Their combined
  diffs remain unstaged to preserve ownership; no broad staging was performed.
