# T500 Wire Staged Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both Panda–O6 hands approach the live `wire_000` USD, establish stable target-filtered thumb/index contact, and bind the measured grasp span to DeformX without a geometric jump.

**Architecture:** A pure scene-geometry provider converts the selected object's measured pose and catalog metadata into two grasp stations. A transactional wire-coupling state machine uses target-filtered contacts to progress from scene tracking through bilateral latch; only an accepted WBC cycle may commit state or call the explicitly bound DeformX backend. Palm-attached virtual grasp frames keep Cosserat endpoints at the thumb/index pinch centers while preserving the current object/arm/hand/WBC APIs.

**Tech Stack:** Python 3.11, PyTorch CPU float64 contracts, NumPy/SciPy rotations, Isaac Lab contact sensors on GPU0, DeformX/PyElastica, pytest.

## Global Constraints

- Preserve the public 43-dimensional action and T400/T500 contracts.
- Keep the M1 base fixed in this stage.
- The rigid USD remains the collision/perception object; DeformX is prediction-only and does not own or teleport the USD.
- Model only the wire span between stations at `center ± 0.30 * length * tangent`.
- Require both thumb and index target-filtered contacts for 5 consecutive accepted control ticks per side.
- Bind only when both pinch-center-to-station errors are at most `0.005 m`.
- No automatic rebind after post-bind contact loss in the same episode.
- Preserve the existing atomic WBC command and `latest_wire_goal` acceptance boundary.
- Run CPU-heavy commands with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`; do not exceed one CPU core.
- Do not add training, VLM, Residual control, arbitrary rod-node attachment, sliding contact, lifting, or deformable USD rendering.

---

### Task 1: Scene-Derived Wire Geometry

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_scene_geometry.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_catalog.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Modify: `Go2Pvcnn/config/m1_object_catalog.json`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Create: `Go2Pvcnn/tests/test_m1_wire_scene_geometry.py`
- Modify: `Go2Pvcnn/tests/test_m1_object_catalog.py`
- Modify: `Go2Pvcnn/tests/test_m1_object_target_contract.py`

**Interfaces:**
- Consumes: live target `BoxState.pose_b: torch.float64[6]` and `ObjectClassRecord`.
- Produces: `WireSceneGeometry(centerline, center_b, tangent_b, radius_m, left_station_b, right_station_b)` and `build_wire_scene_geometry`.

- [ ] **Step 1: Write failing catalog and geometry tests**

Add tests that pin the new metadata and pose transform:

```python
def test_wire_catalog_declares_local_longitudinal_axis():
    payload = json.loads(DEFAULT_CONFIG.read_text())
    assert payload["classes"]["wire"]["longitudinal_axis"] == [1.0, 0.0, 0.0]


def test_scene_wire_geometry_rotates_axis_and_places_distinct_stations():
    record = _wire_record(dimensions=(0.08, 0.01, 0.01), axis=(1.0, 0.0, 0.0))
    pose_b = torch.tensor([0.65, 0.0, 1.20, 0.0, 0.0, math.pi / 2], dtype=torch.float64)
    geometry = build_wire_scene_geometry(pose_b, record, sample_count=5)
    torch.testing.assert_close(geometry.center_b, pose_b[:3])
    torch.testing.assert_close(geometry.tangent_b, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=0.0)
    torch.testing.assert_close(geometry.left_station_b, torch.tensor([0.65, -0.024, 1.20], dtype=torch.float64))
    torch.testing.assert_close(geometry.right_station_b, torch.tensor([0.65, 0.024, 1.20], dtype=torch.float64))
    assert geometry.centerline.positions.shape == (5, 3)
    assert geometry.centerline.positions.dtype == np.float64
```

Also assert `default_object_instances(("wire",))[0].pose == (0.65, 0.0, 1.20)` and reject missing/zero/non-unit axes, non-positive dimensions, non-finite poses, and even `sample_count < 3`.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q \
  tests/test_m1_wire_scene_geometry.py \
  tests/test_m1_object_catalog.py \
  tests/test_m1_object_target_contract.py
```

Expected: FAIL because `longitudinal_axis`, `WireSceneGeometry`, and the explicit wire pose do not exist.

- [ ] **Step 3: Extend the immutable catalog contract**

Add `longitudinal_axis: tuple[float, float, float] | None = None` to `ObjectClassRecord`. Normalize it in `__post_init__` and require a finite unit vector when present:

```python
axis = tuple(float(value) for value in self.longitudinal_axis)
if len(axis) != 3 or not all(math.isfinite(value) for value in axis):
    raise ValueError("longitudinal_axis must contain three finite numbers")
norm = math.sqrt(sum(value * value for value in axis))
if norm <= 1.0e-12:
    raise ValueError("longitudinal_axis must be nonzero")
object.__setattr__(self, "longitudinal_axis", tuple(value / norm for value in axis))
```

Parse `payload.get("longitudinal_axis")` in `_load_class_record`, add `[1.0, 0.0, 0.0]` only to the committed wire catalog record, and add `"wire": (0.65, 0.0, 1.20)` to `_DEFAULT_OBJECT_POSES`. Preserve all existing user-edited hashes and paths in the dirty catalog.

- [ ] **Step 4: Implement the pure geometry provider**

Implement these exact public types and signature:

```python
@dataclass(frozen=True)
class WireSceneGeometry:
    centerline: DloCenterlineState
    center_b: torch.Tensor
    tangent_b: torch.Tensor
    radius_m: float
    left_station_b: torch.Tensor
    right_station_b: torch.Tensor


def build_wire_scene_geometry(
    pose_b: torch.Tensor,
    record: ObjectClassRecord,
    *,
    station_fraction: float = 0.30,
    sample_count: int = 17,
) -> WireSceneGeometry:
    if not isinstance(pose_b, torch.Tensor):
        raise TypeError("pose_b must be a torch.Tensor")
    if pose_b.dtype != torch.float64 or pose_b.device.type != "cpu" or pose_b.shape != (6,):
        raise ValueError("pose_b must be CPU float64 with shape [6]")
    if not bool(torch.isfinite(pose_b).all()):
        raise ValueError("pose_b must be finite")
    if record.dimensions is None or record.longitudinal_axis is None:
        raise ValueError("wire dimensions and longitudinal_axis are required")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 3 or sample_count % 2 == 0:
        raise ValueError("sample_count must be an odd integer >= 3")
    if not math.isfinite(station_fraction) or not 0.0 < station_fraction < 0.5:
        raise ValueError("station_fraction must be finite and in (0, 0.5)")
    center = pose_b[:3].clone()
    rotation = Rotation.from_rotvec(pose_b[3:].numpy())
    tangent = torch.from_numpy(rotation.apply(np.asarray(record.longitudinal_axis))).to(torch.float64)
    tangent = tangent / torch.linalg.vector_norm(tangent)
    length = float(record.dimensions[0])
    radius = 0.5 * max(float(record.dimensions[1]), float(record.dimensions[2]))
    left = center - station_fraction * length * tangent
    right = center + station_fraction * length * tangent
    alpha = torch.linspace(0.0, 1.0, sample_count, dtype=torch.float64)
    positions = left[None] + alpha[:, None] * (right - left)[None]
    return WireSceneGeometry(
        centerline=DloCenterlineState(positions.numpy(), radius_m=radius),
        center_b=center,
        tangent_b=tangent.clone(),
        radius_m=radius,
        left_station_b=left.clone(),
        right_station_b=right.clone(),
    )
```

Use `Rotation.from_rotvec(pose_b[3:].numpy()).apply(axis)`, choose `length = dimensions[0]`, choose `radius = 0.5 * max(dimensions[1:])`, sample only the held span from left station to right station, clone all torch outputs, and reject non-CPU/non-float64/wrong-shape inputs.

- [ ] **Step 5: Export and verify**

Export the two new symbols in `m1_bimanual_coordination/__init__.py`, rerun the Step 2 command, and expect all selected tests to PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_scene_geometry.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/object_catalog.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/config/m1_object_catalog.json \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/tests/test_m1_wire_scene_geometry.py \
  Go2Pvcnn/tests/test_m1_object_catalog.py \
  Go2Pvcnn/tests/test_m1_object_target_contract.py
git commit -m "feat: derive wire stations from scene pose"
```

---

### Task 2: Selected-Object Contact Filtering

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contact_summary.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_contact_summary.py`
- Modify: `Go2Pvcnn/tests/test_m1_object_target_contract.py`

**Interfaces:**
- Consumes: `target_object_id` resolved before `gym.make`.
- Produces: 18 target-neutral per-link filtered sensors and unchanged `SideHandState.contact_mask[5]`.

- [ ] **Step 1: Write failing target-filter tests**

Update tests to require:

```python
assert cfg.target_object_id == "wire_000"
assert cfg.target_contact_prim_path == "{ENV_REGEX_NS}/wire_000"
assert cfg.scene.left_thumb_distal_target_contact.filter_prim_paths_expr == [
    "{ENV_REGEX_NS}/wire_000"
]
```

Add a legacy assertion for `{ENV_REGEX_NS}/Box`, an unknown target rejection test, and update the raw-only consistency assertion to:

```python
assert result.consistency_reason == "raw_contact_without_target_contact"
```

- [ ] **Step 2: Run the focused tests and confirm failure**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q \
  tests/test_m1_dual_panda_o6_env_static.py \
  tests/test_m1_bimanual_contact_summary.py \
  tests/test_m1_object_target_contract.py
```

Expected: FAIL on missing target-neutral sensor/config fields.

- [ ] **Step 3: Make the environment target-aware before construction**

Rename `_box_contact_sensor` to `_target_contact_sensor` and all 18 class fields from `*_box_contact` to `*_target_contact`. Add:

```python
target_object_id: str | None = None
target_contact_prim_path: str = "{ENV_REGEX_NS}/Box"
```

In `__post_init__`, validate that a non-box target is an enabled configured instance, resolve `target_contact_prim_path = f"{{ENV_REGEX_NS}}/{target_object_id}"`, and set that exact single-element filter on every target contact sensor before the scene is constructed. Update `build_play_config` to pass `target_object_id=object_id` when constructing the cfg.

- [ ] **Step 4: Make the wrapper and summary target-neutral**

Rename internal tuples to `left_filtered_target_sensor_names` and `right_filtered_target_sensor_names`, read `*_target_contact`, and change the missing matrix message to `must filter contacts against the selected target`. Keep `SideHandState`, candidate ordering, force threshold, and five-digit shape unchanged.

Change only the consistency reason string and docstrings in `contact_summary.py`; do not change thresholding or digit selection.

- [ ] **Step 5: Verify and commit Task 2**

Rerun Step 2 and expect PASS, then:

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_env_cfg.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/contact_summary.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/tests/test_m1_dual_panda_o6_env_static.py \
  Go2Pvcnn/tests/test_m1_bimanual_contact_summary.py \
  Go2Pvcnn/tests/test_m1_object_target_contract.py
git commit -m "feat: filter O6 contacts against selected object"
```

---

### Task 3: Transactional Bilateral Latch and Palm-Attached Grasp Frames

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_coupling.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_frame_provider.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py`
- Create: `Go2Pvcnn/tests/test_m1_wire_coupling.py`
- Create: `Go2Pvcnn/tests/test_m1_dlo_grasp_frames.py`

**Interfaces:**
- Produces: `WireCouplingPhase`, immutable `WireCouplingState`, `WireCouplingProposal`, `propose_wire_coupling`, and `dual_palm_attached_grasp_frames`.
- Consumes: `BimanualSnapshot`, `WireSceneGeometry`, and previously accepted coupling state.

- [ ] **Step 1: Write failing latch tests**

Cover the exact transition contract:

```python
state = WireCouplingState.initial()
for accepted_tick in range(4):
    proposal = propose_wire_coupling(state, _contact_snapshot(left=True, right=True), geometry)
    state = proposal.next_state
    assert state.phase is WireCouplingPhase.CONTACT_LATCHING
proposal = propose_wire_coupling(state, _contact_snapshot(left=True, right=True), geometry)
assert proposal.next_state.phase is WireCouplingPhase.CONTACT_LATCHING
assert proposal.next_state.left_latched and proposal.next_state.right_latched
assert proposal.bind_requested
```

Also test: thumb-only does not count; raw inconsistency does not count; unilateral latch holds only that side; a rejected proposal leaves the original frozen state byte-equivalent; reset returns both counters to zero; post-bind contact loss proposes `SAFE_HOLD`; no proposal from `SAFE_HOLD` requests rebind.

- [ ] **Step 2: Write failing virtual-frame tests**

Use thumb/index positions whose midpoint is known, bind the palm-relative offset, rotate/translate the palm, and assert:

```python
np.testing.assert_allclose(frames["left_frame"]["position"], expected_left_pinch)
np.testing.assert_allclose(frames["right_frame"]["position"], expected_right_pinch)
assert frames["left_frame"]["director"].shape == (3, 3)
```

Verify velocity includes `v_palm + omega × offset_b`, outputs are finite float64, and reset discards captured offsets.

- [ ] **Step 3: Run tests and confirm failure**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q \
  tests/test_m1_wire_coupling.py tests/test_m1_dlo_grasp_frames.py
```

Expected: FAIL because the state machine and palm-attached frames do not exist.

- [ ] **Step 4: Implement the pure proposal state machine**

Define:

```python
class WireCouplingPhase(Enum):
    SCENE_TRACKED = auto()
    APPROACH_ENDPOINTS = auto()
    CONTACT_LATCHING = auto()
    COSSERAT_BOUND = auto()
    SAFE_HOLD = auto()


@dataclass(frozen=True)
class WireCouplingState:
    phase: WireCouplingPhase
    left_contact_ticks: int
    right_contact_ticks: int
    left_latched: bool
    right_latched: bool
    bind_count: int
    failure_reason: str | None

    @classmethod
    def initial(cls) -> "WireCouplingState":
        return cls(
            phase=WireCouplingPhase.SCENE_TRACKED,
            left_contact_ticks=0,
            right_contact_ticks=0,
            left_latched=False,
            right_latched=False,
            bind_count=0,
            failure_reason=None,
        )


@dataclass(frozen=True)
class WireCouplingProposal:
    next_state: WireCouplingState
    bind_requested: bool
    left_pinch_error_m: float
    right_pinch_error_m: float
```

`propose_wire_coupling` is pure: it checks `contact_consistent`, requires `contact_mask[:2].all()`, increments each side up to 5, computes pinch centers from `fingertip_positions_b[:2].mean(0)`, and requests bind only when both counters reach 5 and both station errors are `<= 0.005`. A bind request remains in `CONTACT_LATCHING`; only a successful backend bind may commit `COSSERAT_BOUND` and `bind_count=1`. The function never calls DeformX or mutates its input state.

- [ ] **Step 5: Implement palm-attached virtual frames**

Add a frozen `PalmGraspAttachment(left_offset_local, right_offset_local)` captured from pinch centers. Capture each offset with these exact equations:

```python
pinch_b = hand.fingertip_positions_b[:2].mean(dim=0).numpy()
palm_position_b = arm.palm_pose_b[:3].numpy()
palm_rotation_b = Rotation.from_rotvec(arm.palm_pose_b[3:].numpy()).as_matrix()
offset_local = palm_rotation_b.T @ (pinch_b - palm_position_b)
```

`dual_palm_attached_grasp_frames(snapshot, attachment)` must reconstruct `position = palm_position + palm_rotation @ offset_local`, set `director = palm_rotation`, and compute `velocity = palm_linear_velocity + np.cross(palm_omega, palm_rotation @ offset_local)`. Copy `omega` from the palm twist and return a mapping with `left_frame` and `right_frame`, each containing `position`, `director`, `velocity`, and `omega`. Keep the existing `dual_palm_frames` export for backward compatibility.

- [ ] **Step 6: Verify, export, and commit Task 3**

Rerun Step 3 and expect PASS. Export the public state/attachment types and functions, then:

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_coupling.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_frame_provider.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/__init__.py \
  Go2Pvcnn/tests/test_m1_wire_coupling.py \
  Go2Pvcnn/tests/test_m1_dlo_grasp_frames.py
git commit -m "feat: add transactional wire contact latch"
```

---

### Task 4: Explicit Continuity-Checked DeformX Binding

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/deformx_adapter.py`
- Create: `Go2Pvcnn/tests/test_m1_deformx_dual_binding.py`

**Interfaces:**
- Consumes: `DloCenterlineState` plus left/right virtual grasp frame mappings.
- Produces: `DeformXDualWireBackend.bind`, `is_bound`, `bind_count`, and existing `step` after bind.

- [ ] **Step 1: Write failing explicit-bind tests with a fake engine**

Monkeypatch `_load_deformx` with deterministic fake Config/Frame/Rod/Engine classes and assert:

```python
backend = DeformXDualWireBackend.from_config(base_radius=0.005)
assert not backend.is_bound
with pytest.raises(RuntimeError, match="bind"):
    backend.step(left_frame=left, right_frame=right)
backend.bind(state, left, right, max_endpoint_error_m=0.005)
assert backend.is_bound
assert backend.bind_count == 1
assert fake_engine.rod_initial_state.start == pytest.approx(state.positions[0])
```

Also reject non-finite frames, endpoint mismatch `> 0.005`, a second bind, reversed/degenerate centerlines, and verify `reset()` restores `is_bound=False`, `bind_count=0`.

- [ ] **Step 2: Run and confirm failure**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q tests/test_m1_deformx_dual_binding.py
```

Expected: FAIL because `step()` still implicitly initializes and `bind()` is absent.

- [ ] **Step 3: Replace implicit initialization with the explicit API**

Implement:

```python
@property
def is_bound(self) -> bool:
    return self._engine is not None

@property
def bind_count(self) -> int:
    return self._bind_count

def bind(
    self,
    initial_state: DloCenterlineState,
    left_frame: dict[str, np.ndarray],
    right_frame: dict[str, np.ndarray],
    *,
    max_endpoint_error_m: float = 0.005,
) -> None:
    """Bind once after validating both virtual frames against the centerline endpoints."""
```

The method body validates `max_endpoint_error_m`, rejects an existing engine, constructs both typed frames, computes `left_error = norm(left.position - positions[0])` and `right_error = norm(right.position - positions[-1])`, and rejects either error above the threshold. It initializes `RodInitialState` from `positions[0]`, the normalized first segment, and the same deterministic orthogonal-normal rule already used by `_initialize`; `base_length` is `np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()`. Construct the engine only after every check succeeds and then set `_bind_count = 1`. `step()` must raise before bind and must never call `_initialize`. `reset()` clears engine, frame type, and bind count.

- [ ] **Step 4: Verify and commit Task 4**

Rerun Step 2 and expect PASS, then:

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/deformx_adapter.py \
  Go2Pvcnn/tests/test_m1_deformx_dual_binding.py
git commit -m "feat: require explicit DeformX wire binding"
```

---

### Task 5: Route Scene Geometry and Commit DeformX Only After Feasible WBC

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_wire_goal.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_wire_trajectory.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/tests/test_m1_dlo_wire_goal.py`
- Modify: `Go2Pvcnn/tests/test_m1_dlo_wire_trajectory.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_runtime.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_prior_startup_binding.py`

**Interfaces:**
- Consumes: `wire_geometry_provider(snapshot) -> WireSceneGeometry`, coupling proposals, virtual frame provider, and explicit-bind backend.
- Produces: scene-derived pre-contact targets, accepted coupling state, cached post-bind `DualDloStepResult`, and `latest_wire_diagnostics`.

- [ ] **Step 1: Write failing distinct-station planner tests**

Change the wire-goal test to require left target positions centered on the first centerline point and right targets centered on the last:

```python
left_center = np.mean([t.position for t in goal.left_contact_targets], axis=0)
right_center = np.mean([t.position for t in goal.right_contact_targets], axis=0)
np.testing.assert_allclose(left_center, state.positions[0])
np.testing.assert_allclose(right_center, state.positions[-1])
assert not np.allclose(left_center, right_center)
```

Add `DloWireTrajectoryPlanner.plan_state(snapshot, object_solution, state)` tests and retain `plan(snapshot, object_solution, dlo_step)` as a compatibility wrapper for `DualDloStepResult`.

- [ ] **Step 2: Write failing transactional runtime tests**

Use recording fake backend/coupling providers to prove:

```python
assert backend.bind_calls == 0       # before bilateral accepted latch
assert backend.step_calls == 0
runtime.compute(contact_tick_5)
assert backend.bind_calls == 1       # only after feasible WBC commit
assert runtime.wire_coupling_state.phase is WireCouplingPhase.COSSERAT_BOUND
```

For a WBC rejection on tick 5 assert coupling state, backend calls,
`latest_dlo_step`, attachment, last safe command, and `latest_wire_goal` are unchanged. On the next accepted tick binding may occur once. Test post-bind contact dropout enters `SAFE_HOLD` and never increments bind count. Test `reset()` clears every wire field and resets the backend once.

- [ ] **Step 3: Run focused tests and confirm failure**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q \
  tests/test_m1_dlo_wire_goal.py \
  tests/test_m1_dlo_wire_trajectory.py \
  tests/test_m1_bimanual_runtime.py \
  tests/test_m1_bimanual_prior_startup_binding.py
```

Expected: FAIL because current goals share one midpoint and the backend advances before WBC acceptance.

- [ ] **Step 4: Add station-specific goals without breaking old callers**

Add:

```python
def build_wire_station_contact_goal(
    state: DloCenterlineState,
    *,
    clearance_m: float = 0.003,
) -> WireLineContactGoal:
    """Place left and right opposed pinch targets at the two held-span endpoints."""
```

The body validates the same inputs as `build_wire_line_contact_goal`, computes the tangent/normal/binormal once, calls a local `make_side(center, sign)` for `state.positions[0]` and `state.positions[-1]`, and returns `WireLineContactGoal` whose `center` remains `state.midpoint`. Use the existing gap and finger-offset formulas. Keep `build_wire_line_contact_goal` available for compatibility, but route the staged planner through the station-specific function.

- [ ] **Step 5: Make runtime wire state transactional**

Extend `BimanualRuntime.__init__` with:

```python
wire_geometry_provider=None,
wire_coupling_state: WireCouplingState | None = None,
wire_frame_provider=None,
```

At the beginning of `compute`, build scene geometry and a pure proposal but do not mutate accepted state and do not call the backend. Plan from `latest_dlo_step.state` only when already bound; otherwise plan from `scene_geometry.centerline`.

After `wbc_solution.feasible`:

```python
self.wire_coupling_state = proposal.next_state
if proposal.bind_requested:
    self._wire_attachment = capture_palm_grasp_attachment(snapshot)
    frames = self.wire_frame_provider(snapshot, self._wire_attachment)
    self.dlo_backend.bind(scene_geometry.centerline, **frames)
    self.wire_coupling_state = replace(
        proposal.next_state,
        phase=WireCouplingPhase.COSSERAT_BOUND,
        bind_count=1,
    )
elif proposal.next_state.phase is WireCouplingPhase.COSSERAT_BOUND:
    frames = self.wire_frame_provider(snapshot, self._wire_attachment)
    self.latest_dlo_step = self.dlo_backend.step(**frames)
```

On infeasible WBC, skip this entire commit block. Catch bind/step validation/runtime failures, retain the last accepted command/goal/centerline, and commit `SAFE_HOLD` with a concrete failure reason on the next safe boundary. Do not mutate the backend on a rejected candidate.

- [ ] **Step 6: Wire catalog metadata into each lane**

In the wrapper, create a geometry-provider closure only when the selected catalog record has `object_class == "wire"`. The closure reads `snapshot.box.pose_b` and calls `build_wire_scene_geometry`. Supply it with `dual_palm_attached_grasp_frames` and the explicit DeformX backend. Non-wire and legacy Box lanes must receive no wire provider and retain existing behavior.

- [ ] **Step 7: Verify and commit Task 5**

Rerun Step 3 and expect PASS, then:

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_wire_goal.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/dlo_wire_trajectory.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py \
  Go2Pvcnn/tests/test_m1_dlo_wire_goal.py \
  Go2Pvcnn/tests/test_m1_dlo_wire_trajectory.py \
  Go2Pvcnn/tests/test_m1_bimanual_runtime.py \
  Go2Pvcnn/tests/test_m1_bimanual_prior_startup_binding.py
git commit -m "feat: stage wire contact before Cosserat binding"
```

---

### Task 6: Deterministic Wire Diagnostics and Report Validation

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_coupling.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Create: `Go2Pvcnn/scripts/verify_m1_wire_contact_report.py`
- Create: `Go2Pvcnn/tests/test_m1_wire_contact_diagnostics.py`

**Interfaces:**
- Produces: immutable `WireCouplingDiagnostics`, `BimanualRuntime.latest_wire_diagnostics`, `--wire-report PATH`, and a pure report validator.

- [ ] **Step 1: Write failing diagnostics tests**

Require JSON-safe fields:

```python
required = {
    "object_id", "target_prim_path", "state", "transition_reason",
    "centerline_midpoint_b", "tangent_b", "radius_m",
    "left_station_b", "right_station_b", "left_pinch_error_m",
    "right_pinch_error_m", "left_contact_mask", "right_contact_mask",
    "left_contact_ticks", "right_contact_ticks", "bind_count",
    "deformx_elapsed_time", "wbc_feasible", "wire_goal_committed",
}
assert required <= report["final"].keys()
json.dumps(report, allow_nan=False)
```

Test that the validator rejects wrong object/prim, missing transition order, any non-finite number, bind count other than 1, endpoint error above 0.005, and DeformX activity before bind.

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q tests/test_m1_wire_contact_diagnostics.py
```

Expected: FAIL because the diagnostics/report contract does not exist.

- [ ] **Step 3: Implement immutable diagnostics and atomic JSON output**

Add a frozen diagnostics dataclass containing the required fields and a `to_json()` method that converts tuples/NumPy/torch values to finite Python scalars/lists. Update it once per runtime compute without influencing control.

Add `--wire-report Path` to play. Collect only state transitions plus the final record, and write atomically using a temporary file plus `os.replace`. Do not change the existing GUI shutdown/hold behavior.

- [ ] **Step 4: Implement the pure report validator**

`verify_m1_wire_contact_report.py REPORT` must return zero only when transitions contain this ordered subsequence:

```python
EXPECTED = (
    "SCENE_TRACKED",
    "APPROACH_ENDPOINTS",
    "CONTACT_LATCHING",
    "COSSERAT_BOUND",
)
```

It must also enforce `object_id == "wire_000"`, `target_prim_path.endswith("/wire_000")`, both counters `>= 5`, `bind_count == 1`, finite values, both final pinch errors `<= 0.005`, and no recorded pre-bind DeformX elapsed time.

- [ ] **Step 5: Verify and commit Task 6**

Rerun Step 2 and expect PASS, then:

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/wire_coupling.py \
  Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py \
  Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py \
  Go2Pvcnn/scripts/verify_m1_wire_contact_report.py \
  Go2Pvcnn/tests/test_m1_wire_contact_diagnostics.py
git commit -m "feat: report staged wire contact transitions"
```

---

### Task 7: Full CPU Regression and GPU0 Wire Contact Evidence

**Files:**
- Create after a passing run: `Go2Pvcnn/tests/artifacts/m1_wire_contact_gpu0.json`
- Create after a passing run: `Go2Pvcnn/tests/artifacts/m1_wire_contact_gpu0.manifest.json`
- Modify only if a test exposes a defect: files owned by Tasks 1–6 and their matching tests.

**Interfaces:**
- Consumes: completed Tasks 1–6 and prepared `wire_000` asset.
- Produces: passing CPU suite plus SHA-pinned GPU0 evidence for bilateral contact and one continuous DeformX bind.

- [ ] **Step 1: Run all bimanual CPU tests on one CPU core**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q tests/test_m1_bimanual*.py \
  tests/test_m1_dual_panda_o6*.py \
  tests/test_m1_dlo*.py \
  tests/test_m1_wire*.py \
  tests/test_m1_object_catalog.py \
  tests/test_m1_object_scene.py \
  tests/test_m1_object_target_contract.py
```

Expected: all selected tests PASS. If any fail, fix the responsible task without weakening contact count, error limits, atomicity, or existing contracts; rerun the focused test first and then this command.

- [ ] **Step 2: Run a short GPU0 scene/target smoke**

```bash
cd Go2Pvcnn
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export DEFORMX_ROOT=/home/xk/coding/DeformX
export DEFORMX_ELASTICA_ROOT=/home/xk/coding/DeformX/PyElastica-Mesh
export M1_ENABLE_DEFORMX=1
/home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_play.py \
  --mode teacher --object-id wire_000 \
  --object-catalog config/m1_object_catalog.json \
  --object-assets-root assets/m1_objects \
  --seed 42 --max-steps 16 --diagnostics \
  --wire-report /tmp/m1_wire_short.json --headless
```

Expected: no traceback/NaN; report object is `wire_000`, target prim ends in `/wire_000`, midpoint error is `<= 0.002 m`, state is pre-bind, `bind_count == 0`, and DeformX has not stepped.

- [ ] **Step 3: Run the full fixed-seed bilateral contact smoke**

```bash
cd Go2Pvcnn
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export DEFORMX_ROOT=/home/xk/coding/DeformX
export DEFORMX_ELASTICA_ROOT=/home/xk/coding/DeformX/PyElastica-Mesh
export M1_ENABLE_DEFORMX=1
/home/xk/miniconda3/envs/go2/bin/python \
  scripts/m1_dual_panda_o6_bimanual_play.py \
  --mode teacher --object-id wire_000 \
  --object-catalog config/m1_object_catalog.json \
  --object-assets-root assets/m1_objects \
  --seed 42 --max-steps 4000 --diagnostics \
  --wire-report tests/artifacts/m1_wire_contact_gpu0.json --headless
/home/xk/miniconda3/envs/go2/bin/python \
  scripts/verify_m1_wire_contact_report.py \
  tests/artifacts/m1_wire_contact_gpu0.json
```

Expected: validator exits 0 and reports the ordered transition through `COSSERAT_BOUND`, two stable 5-tick contact latches, exactly one bind, no pre-bind DeformX step, and endpoint continuity `<= 0.005 m`. This stage does not require lift.

- [ ] **Step 4: Generate a SHA-pinned evidence manifest**

Write `m1_wire_contact_gpu0.manifest.json` containing schema version, status, report path/SHA-256, current `git rev-parse HEAD`, wire USD resolved SHA-256, catalog SHA-256, DeformX commit, PyElastica commit, Isaac version, CUDA device, and the exact command. Validate with `json.load(stream, parse_constant=reject_nonfinite)` and independently recompute every digest.

- [ ] **Step 5: Run the complete project regression suite**

```bash
cd Go2Pvcnn
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH="$PWD" pytest -q
```

Expected: complete suite PASS with no CPU thread-cap violation.

- [ ] **Step 6: Commit verified evidence**

```bash
git add Go2Pvcnn/tests/artifacts/m1_wire_contact_gpu0.json \
  Go2Pvcnn/tests/artifacts/m1_wire_contact_gpu0.manifest.json
git commit -m "test: verify bilateral wire contact binding"
```

Do not mark this plan complete if the GPU report remains in `APPROACH_ENDPOINTS` or `CONTACT_LATCHING`, if only one hand contacts, or if the validator is bypassed.
