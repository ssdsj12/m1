# T500 DexManipNet Fingertip Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn a hand-agnostic five-fingertip motion distribution from the complete DexManipNet FAVOR and OakInk V2 archives, distill it into a compact frozen student, and add that student to each O6 Hand MPC as an optional, bounded, atomically rejectable soft cost.

**Architecture:** A network-free preprocessing package pins and audits external archives, evaluates source-hand URDF FK, canonicalizes left/right palm-frame fingertip geometry, and emits SHA-pinned 100 Hz/20-node shards. An offline ensemble learns the expert distribution and distills a compact four-component Gaussian-mixture student. Runtime code consumes only the frozen student and current O6 fingertip geometry; Hand MPC first solves its unchanged baseline QP, then optionally solves a prior-regularized QP and falls back to the same-cycle baseline on any prior failure.

**Tech Stack:** Python 3, NumPy, SciPy, h5py, PyTorch, `huggingface_hub`, stdlib XML/JSON/tar handling, pytest, existing dense QP backend, Isaac Lab GPU0 smoke.

## Global Constraints

- Follow [the approved design](../specs/2026-09-11-t500-dexmanipnet-fingertip-prior-design.md).
- Pin DexManipNet revision to `3933fae5fe83498fb314a0924aa21d5038fba5a5` and ManipTrans to `a3d08cfe3c3a5868a7f057533bcaf759c5af4705`.
- Process both FAVOR and OakInk V2 archives; audit every sequence and explicitly reject unsupported sequences.
- Keep `Go2Pvcnn/data/external/dexmanipnet/`, raw data, converted samples, ensembles, and student weights out of Git.
- Do not import IsaacGym, Hugging Face, h5py, source URDF parsing, or training modules from the runtime Hand MPC path.
- Do not feed task ID, object class/name, text, palm target, object target, arm state, or base state to the prior.
- Do not map source joint angles directly to O6 joint angles.
- Preserve T400 contracts, the T500 43-channel action order, 25/50/100/200 Hz scheduling, hard constraints, and existing fallback behavior.
- Keep prior disabled by default; the disabled path must bypass inference and the second QP.
- Use TDD for every production behavior: write one focused failure, observe the expected RED, implement the minimum GREEN, then run related regression.
- Every verification pass must update `notes/log/`, `notes/log/index.md`, `notes/todo.md`, and the T500 branch page.

## File Structure

- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/contracts.py`: frozen data/model/runtime constants and dataclasses.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/sources.py`: source pins and explicit supported source-hand manifests.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/download.py`: pin validation, safe extraction, hashing, and atomic directory install.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/urdf_fk.py`: Isaac-independent URDF tree and batched FK.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py`: HDF5/JSON audit and best-successful-rollout selection.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/preprocess.py`: handedness, resampling, velocity, contact, phase, and window generation.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/storage.py`: deterministic group split, shards, JSONL audit, and aggregate manifest.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/model.py`: expert/student mixture networks, NLL, temporal regularizers, and distillation loss.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/artifact.py`: student metadata, SHA validation, and load/save.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py`: frozen student inference, component selection, precision bounds, and diagnostics.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_fetch_dexmanipnet.py`: resumable download/extraction CLI.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_convert_dexmanipnet.py`: full audit/conversion CLI.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_train_fingertip_expert.py`: offline ensemble training CLI.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_distill_fingertip_prior.py`: compact-student distillation/export CLI.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_eval_fingertip_prior.py`: CPU metrics/latency and prior-off/on comparison CLI.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py`: optional prior QP terms and atomic same-cycle baseline.
- `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`: forward measured O6 fingertip geometry to each Hand MPC.
- `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`: validated student artifact construction.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`: prior diagnostics and formal reports.
- `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`: opt-in prior artifact argument.

---

### Task 1: Freeze source, sample, and artifact contracts

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/__init__.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/contracts.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/sources.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_contracts.py`

**Interfaces:**
- Produces: `DEXMANIPNET_REVISION`, `MANIPTRANS_COMMIT`, `PriorPhase`, `ExpertWindow`, `MixtureDistribution`, `StudentArtifactMetadata`, `SourceHandSpec`, `SOURCE_HANDS`.
- Consumes: no runtime or external-data dependency.

- [x] **Step 1: Write the failing contract tests**

```python
def test_expert_window_freezes_geometry_only_contract():
    window = ExpertWindow(
        fingertip_position_palm=torch.zeros(5, 3),
        fingertip_velocity_palm=torch.zeros(5, 3),
        contact_mask=torch.zeros(5, dtype=torch.bool),
        phase=PriorPhase.APPROACH,
        future_fingertip_velocity_palm=torch.zeros(20, 5, 3),
        source_group="favor/seq/rh",
        source_sha256="0" * 64,
    )
    assert window.network_input().shape == (42,)
    assert window.target.shape == (20, 5, 3)

def test_source_pins_and_five_finger_order_are_frozen():
    assert DEXMANIPNET_REVISION == "3933fae5fe83498fb314a0924aa21d5038fba5a5"
    assert MANIPTRANS_COMMIT == "a3d08cfe3c3a5868a7f057533bcaf759c5af4705"
    assert FINGER_ORDER == ("thumb", "index", "middle", "ring", "pinky")
    assert all(len(set(spec.fingertip_links)) == 5 for spec in SOURCE_HANDS.values())
```

- [x] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py`

Expected: collection fails because `expert_fingertip_prior` does not exist.

- [x] **Step 3: Implement the frozen contracts and explicit source registry**

```python
DEXMANIPNET_REVISION = "3933fae5fe83498fb314a0924aa21d5038fba5a5"
MANIPTRANS_COMMIT = "a3d08cfe3c3a5868a7f057533bcaf759c5af4705"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
MODEL_INPUT_DIM = 42
MIXTURE_COMPONENTS = 4
PRIOR_HORIZON = 20
PRIOR_DT = 0.01

class PriorPhase(IntEnum):
    APPROACH = 0
    PRELOAD = 1
    GRASP = 2
    MANIPULATE = 3
    HOLD = 4
    RELEASE = 5
    UNKNOWN = 6

@dataclass(frozen=True)
class SourceHandSpec:
    name: str
    side: str
    urdf_relpath: str
    joint_order: tuple[str, ...]
    palm_link: str
    fingertip_links: tuple[str, str, str, str, str]
```

Populate `SOURCE_HANDS` only with pinned ManipTrans hands whose URDF and five distinct fingertips exist; start with explicit right/left Inspire and Shadow entries copied from the pinned official definitions. Validate all tensors, enum values, SHA strings, unique fingertip names, and model dimensions in dataclass `__post_init__` methods.

- [x] **Step 4: Run GREEN and current contracts**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_contracts.py tests/test_m1_dual_panda_o6_contracts.py`

Expected: pass.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior Go2Pvcnn/tests/test_m1_bimanual_expert_prior_contracts.py
git commit -m "feat: freeze DexManipNet fingertip prior contracts"
```

### Task 2: Add pinned download and safe atomic extraction

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/download.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_fetch_dexmanipnet.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_download.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 1 pins.
- Produces: `sha256_file(path)`, `validate_tar_members(members)`, `atomic_extract_tar(archive, destination)`, `build_download_manifest(download_root, archive_paths, revision, source_commit)`, and CLI `--root/--download-only/--verify-only`.

- [x] **Step 1: Write path-traversal and pin tests**

```python
@pytest.mark.parametrize("name", ["/abs/file", "../escape", "safe/../../escape"])
def test_tar_validation_rejects_escape(name):
    member = tarfile.TarInfo(name)
    with pytest.raises(ValueError, match="unsafe tar member"):
        validate_tar_members([member])

def test_download_cli_contains_both_archives_and_fixed_revision():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "dexmanipnet_favor.tar.gz" in source
    assert "dexmanipnet_oakinkv2.tar.gz" in source
    assert "revision=DEXMANIPNET_REVISION" in source

def test_failed_extraction_never_installs_destination(tmp_path):
    archive = write_tar(tmp_path, {"../escape": b"bad"})
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="unsafe tar member"):
        atomic_extract_tar(archive, destination)
    assert not destination.exists()
```

- [x] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_download.py`

Expected: fails on missing download module/script.

- [x] **Step 3: Implement safe extraction and resumable fetch**

```python
def validate_tar_members(members: Iterable[tarfile.TarInfo]) -> tuple[tarfile.TarInfo, ...]:
    checked = []
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or member.isdev():
            raise ValueError(f"unsafe tar member: {member.name}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError(f"unsafe tar member link: {member.name}")
        checked.append(member)
    return tuple(checked)
```

The CLI must call the fixed download below, clone ManipTrans at the fixed commit, calculate archive SHA/size, extract into sibling temporary directories, validate expected `sequences/` roots, then atomically rename. Add an explicit `/data/external/dexmanipnet/` ignore rule even though the broader data rule already covers it.

```python
snapshot_download(
    repo_id="LiKailin/DexManipNet",
    repo_type="dataset",
    revision=DEXMANIPNET_REVISION,
    allow_patterns=("README.md", "dexmanipnet_favor.tar.gz", "dexmanipnet_oakinkv2.tar.gz"),
    local_dir=download_root,
)
```

- [x] **Step 4: Run GREEN and help smoke**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_download.py && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_fetch_dexmanipnet.py --help`

Expected: tests pass and help exits `0` without downloading.

- [x] **Step 5: Commit**

```bash
git add .gitignore Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/download.py Go2Pvcnn/scripts/m1_dual_panda_o6_fetch_dexmanipnet.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_download.py
git commit -m "feat: add pinned DexManipNet fetcher"
```

### Task 3: Implement Isaac-independent URDF FK

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/urdf_fk.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_urdf_fk.py`

**Interfaces:**
- Consumes: `SourceHandSpec` and source URDF.
- Produces: `UrdfKinematicTree.from_file(path)`, `forward_links(q, joint_order, links) -> np.ndarray`, mimic expansion, palm-relative five-tip FK.

- [x] **Step 1: Write analytical two-joint and mimic RED tests**

```python
def test_two_revolute_joint_fk_matches_hand_solution(tmp_path):
    path = write_two_link_urdf(tmp_path, mimic=False)
    tree = UrdfKinematicTree.from_file(path)
    result = tree.forward_links(np.array([[np.pi / 2, 0.0]]), ("j0", "j1"), ("tip",))
    np.testing.assert_allclose(result[0, 0, :3, 3], [0.0, 2.0, 0.0], atol=1e-8)

def test_mimic_joint_uses_master_multiplier_and_offset(tmp_path):
    path = write_two_link_urdf(tmp_path, mimic=True, multiplier=0.5, offset=0.1)
    tree = UrdfKinematicTree.from_file(path)
    assert tree.expanded_joint_positions({"j0": 0.4})["j1"] == pytest.approx(0.3)
```

- [x] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_urdf_fk.py`

Expected: missing `UrdfKinematicTree`.

- [x] **Step 3: Implement XML parsing and deterministic batched FK**

```python
def axis_angle_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    eye = np.eye(3)
    return eye + np.sin(angle)[..., None, None] * skew + (1.0 - np.cos(angle))[..., None, None] * (skew @ skew)
```

Parse link/joint name, parent/child, origin xyz/rpy, axis, type, limit, and mimic. Reject cycles, multiple parents, missing roots, unknown joint types, duplicate names, missing requested links, mismatched q columns, non-finite values, and mimic cycles. Compose root-to-link transforms in stable topological order, then compute `inv(T_palm) @ T_tip` in frozen finger order.

- [x] **Step 4: Run GREEN**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_urdf_fk.py`

Expected: pass with maximum analytical FK error `<=1e-8 m`.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/urdf_fk.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_urdf_fk.py
git commit -m "feat: add source hand URDF forward kinematics"
```

### Task 4: Audit DexManipNet sequences and select the best rollout

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_dexmanipnet.py`

**Interfaces:**
- Consumes: extracted source root and `SOURCE_HANDS`.
- Produces: `SequenceAudit`, `LoadedHandSequence`, `audit_sequence(path, source, side)`, `load_best_successful_rollout(path, source, side)`.

- [ ] **Step 1: Write minimal HDF5 RED tests**

```python
def test_loader_selects_highest_total_reward_without_exposing_task_id(tmp_path):
    sequence = write_minimal_sequence(tmp_path, rewards=[[1.0, 1.0], [0.0, 3.0]])
    loaded = load_best_successful_rollout(sequence, source="favor", side="rh")
    assert loaded.rollout_name == "rollout_1"
    assert loaded.q.shape[0] == loaded.dq.shape[0] == 2
    assert not hasattr(loaded, "description")

def test_length_mismatch_is_one_atomic_rejection(tmp_path):
    sequence = write_minimal_sequence(tmp_path, q_frames=3, dq_frames=2)
    audit = audit_sequence(sequence, source="favor", side="rh")
    assert not audit.accepted
    assert audit.reason == "length_mismatch"
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_dexmanipnet.py`

Expected: missing loader.

- [ ] **Step 3: Implement strict schema and best-rollout loading**

```python
successful = h5["rollouts/successful"]
names = sorted(successful.keys())
scores = {name: float(np.asarray(successful[name]["reward"]).sum()) for name in names}
rollout_name = max(names, key=lambda name: (scores[name], name))
```

Validate `seq_info.json`, sequence length, `q_<side>`, `dq_<side>`, root state, object pose/geometry inputs, finite arrays, supported hand-side key, and exact source joint dimension. Return only geometry-required arrays and provenance; never place primitive, object ID/name, description, or text in `LoadedHandSequence`.

- [ ] **Step 4: Run GREEN**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_dexmanipnet.py`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/dexmanipnet.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_dexmanipnet.py
git commit -m "feat: audit DexManipNet successful rollouts"
```

### Task 5: Convert fingertip geometry into deterministic shards

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/preprocess.py`
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/storage.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_convert_dexmanipnet.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_preprocess.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_storage.py`

**Interfaces:**
- Consumes: `LoadedHandSequence`, `UrdfKinematicTree`, source/archive manifests.
- Produces: `canonicalize_left`, `resample_fingertips`, `infer_contact_hysteresis`, `infer_prior_phase`, `windows_from_sequence`, `deterministic_group_split`, `write_shards`, aggregate manifest and audit JSONL.

- [ ] **Step 1: Write geometry and split RED tests**

```python
def test_left_mirror_round_trip_and_resampling_contract():
    points = np.arange(45, dtype=np.float64).reshape(3, 5, 3) / 100.0
    canonical = canonicalize_left(points)
    np.testing.assert_allclose(decanonicalize_left(canonical), points, atol=1e-6)
    out = resample_fingertips(points, source_hz=60, target_hz=100)
    assert out.shape[1:] == (5, 3)

def test_group_split_never_leaks_sequence_windows():
    split = deterministic_group_split(["a/rh"] * 3 + ["b/lh"] * 2 + ["c/rh"], seed=42)
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))

def test_contact_hysteresis_and_phase_labels_use_geometry_only():
    distance = np.array([[0.02] * 5, [0.001] * 5, [0.003] * 5, [0.02] * 5])
    normal_speed = np.array([[-0.02] * 5, [-0.01] * 5, [0.0] * 5, [0.02] * 5])
    contact = infer_contact_hysteresis(distance, normal_speed, enter_m=0.002, exit_m=0.005)
    assert contact[:, 0].tolist() == [False, True, True, False]
    phase = infer_prior_phase(contact, fingertip_speed=np.array([0.03, 0.01, 0.0, 0.02]))
    assert phase.tolist() == [PriorPhase.APPROACH, PriorPhase.PRELOAD, PriorPhase.HOLD, PriorPhase.RELEASE]
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_preprocess.py tests/test_m1_bimanual_expert_prior_storage.py`

Expected: missing preprocessing/storage functions.

- [ ] **Step 3: Implement canonicalization, 100 Hz windows, labels, and atomic shards**

```python
LEFT_REFLECTION = np.diag([1.0, -1.0, 1.0])

def canonicalize_left(values: np.ndarray) -> np.ndarray:
    return np.einsum("...j,ij->...i", values, LEFT_REFLECTION)

def deterministic_group_split(groups: Sequence[str], seed: int) -> GroupSplit:
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    n = len(unique)
    return GroupSplit(tuple(unique[: int(.8*n)]), tuple(unique[int(.8*n): int(.9*n)]), tuple(unique[int(.9*n):]))
```

Use `scipy.interpolate.CubicSpline` with source timestamps `arange(n)/60` and target timestamps that remain inside the original interval. Derive velocities from spline derivatives. Implement contact enter/exit hysteresis from signed mesh distance and relative normal speed, and seven phase labels from contact transitions plus speed thresholds. Reject missing geometry rather than filling contact false. Write `.npz` shards through a sibling temporary file, rename atomically, hash each shard, and make the aggregate manifest sort by relative shard path.

- [ ] **Step 4: Run GREEN and conversion CLI help**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_preprocess.py tests/test_m1_bimanual_expert_prior_storage.py && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_convert_dexmanipnet.py --help`

Expected: pass; help exits `0` without reading the dataset.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/preprocess.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/storage.py Go2Pvcnn/scripts/m1_dual_panda_o6_convert_dexmanipnet.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_preprocess.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_storage.py
git commit -m "feat: convert DexManipNet to fingertip prior shards"
```

### Task 6: Train and gate the offline expert ensemble

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/model.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_train_fingertip_expert.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_model.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_train_static.py`

**Interfaces:**
- Consumes: Task 5 aggregate manifest and shards.
- Produces: `FingertipMixtureNet`, `mixture_log_prob`, `mixture_nll`, `temporal_regularizer`, ensemble checkpoints/manifest, held-out metrics.

- [ ] **Step 1: Write output/NLL/CLI RED tests**

```python
def test_mixture_network_outputs_normalized_finite_distribution():
    model = FingertipMixtureNet(hidden=(128, 128, 128))
    dist = model(torch.zeros(3, 42))
    assert dist.logits.shape == (3, 4)
    assert dist.mean.shape == dist.log_std.shape == (3, 4, 20, 5, 3)
    assert torch.isfinite(dist.mean).all()
    torch.testing.assert_close(dist.logits.softmax(-1).sum(-1), torch.ones(3))

def test_training_script_does_not_import_task_or_object_features():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("description", "primitive", "object_id", "palm_target", "box_pose"):
        assert forbidden not in source
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_model.py tests/test_m1_bimanual_expert_prior_train_static.py`

Expected: missing model/training CLI.

- [ ] **Step 3: Implement mixture density and ensemble training**

```python
def mixture_nll(dist: MixtureDistribution, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None]
    inv_var = torch.exp(-2.0 * dist.log_std)
    log_component = -0.5 * (((target - dist.mean) ** 2) * inv_var + 2.0 * dist.log_std + math.log(2.0 * math.pi))
    log_component = log_component.flatten(2).sum(-1) + dist.logits.log_softmax(-1)
    return -torch.logsumexp(log_component, dim=1).mean()
```

Train multiple larger residual MLP members with fixed distinct seeds, sequence-group loaders, NLL plus finite acceleration/jerk regularization, atomic checkpoints, and best-validation selection. Compute held-out first-step velocity RMSE, integrated endpoint RMSE, NLL, and 80% interval coverage. Mark the ensemble deployable only when both RMSE values beat zero baselines by `>=10%` and coverage lies in `[0.65, 0.95]`.

- [ ] **Step 4: Run GREEN and synthetic training smoke**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_model.py tests/test_m1_bimanual_expert_prior_train_static.py && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_train_fingertip_expert.py --synthetic-smoke --output-dir /tmp/t500-expert-smoke --epochs 2`

Expected: tests pass; smoke emits finite metrics and an ensemble manifest but does not mark synthetic data as production-deployable.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/model.py Go2Pvcnn/scripts/m1_dual_panda_o6_train_fingertip_expert.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_model.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_train_static.py
git commit -m "feat: train fingertip expert ensemble"
```

### Task 7: Distill, validate, and serialize the compact student

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/artifact.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_distill_fingertip_prior.py`
- Create: `Go2Pvcnn/scripts/m1_dual_panda_o6_eval_fingertip_prior.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_distill.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_artifact.py`

**Interfaces:**
- Consumes: deployable ensemble manifest and Task 6 distribution API.
- Produces: `distribution_distillation_loss`, `save_student_artifact`, `load_student_artifact`, `accept_prior_comparison(metrics)`, SHA-pinned metadata, metrics and CPU latency report.

- [ ] **Step 1: Write distillation and tamper RED tests**

```python
def test_distillation_is_component_permutation_invariant():
    teacher = fixed_distribution()
    permuted = permute_components(teacher, torch.tensor([2, 0, 3, 1]))
    samples = sample_mixture(teacher, samples_per_state=8, seed=42)
    torch.testing.assert_close(
        distribution_distillation_loss(teacher, samples),
        distribution_distillation_loss(permuted, samples),
    )

def test_artifact_rejects_weight_tampering(tmp_path):
    artifact = write_valid_student_artifact(tmp_path)
    artifact.weights.write_bytes(artifact.weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="weight SHA"):
        load_student_artifact(artifact.root)
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_distill.py tests/test_m1_bimanual_expert_prior_artifact.py`

Expected: missing distillation/artifact APIs.

- [ ] **Step 3: Implement fixed-sample distribution distillation and strict artifact loading**

```python
def distribution_distillation_loss(student: MixtureDistribution, teacher_samples: torch.Tensor) -> torch.Tensor:
    return -mixture_log_prob(student, teacher_samples).mean()

def load_student_artifact(root: Path) -> LoadedStudent:
    metadata = StudentArtifactMetadata.from_json(root / "metadata.json")
    if sha256_file(root / "student.pt") != metadata.weight_sha256:
        raise ValueError("student weight SHA mismatch")
    model = FingertipMixtureNet(hidden=metadata.hidden)
    model.load_state_dict(torch.load(root / "student.pt", map_location="cpu", weights_only=True))
    model.eval()
    return LoadedStudent(model=model, metadata=metadata)
```

Train the compact student on real labels plus fixed Monte-Carlo teacher samples. Gate export on `student_nll - teacher_nll <= 0.05 nat/dim`, endpoint RMSE increase `<=5%`, both zero-baseline improvements `>=10%`, deterministic repeated artifact SHA, and CPU p99 `<2 ms` over 1000 measured runs after 100 warm-ups.

- [ ] **Step 4: Run GREEN and synthetic distillation smoke**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_distill.py tests/test_m1_bimanual_expert_prior_artifact.py && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_distill_fingertip_prior.py --synthetic-smoke --output-dir /tmp/t500-student-smoke --epochs 2`

Expected: tests pass; smoke artifact reloads and emits finite metrics without production approval.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/artifact.py Go2Pvcnn/scripts/m1_dual_panda_o6_distill_fingertip_prior.py Go2Pvcnn/scripts/m1_dual_panda_o6_eval_fingertip_prior.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_distill.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_artifact.py
git commit -m "feat: distill and validate fingertip prior student"
```

### Task 8: Build the runtime-only O6 prior adapter

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py`
- Test: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_runtime.py`

**Interfaces:**
- Consumes: loaded student, O6 `(5,3)` positions, `(15,6)` folded Jacobian, six-axis qd, contact mask, `BimanualPhase`, and baseline qdot.
- Produces: `FrozenO6FingertipPrior.from_artifact(path)`, `FingertipPriorTarget(mean_velocity, precision, component, probability)`, `FingertipPriorDiagnostics`, and `PriorQueryResult(target, diagnostics)`; no training imports.

- [ ] **Step 1: Write selection, variance, phase, and non-finite RED tests**

```python
def test_runtime_selects_component_nearest_baseline_tip_velocity():
    runtime = runtime_with_means([0.0, 0.1, 0.8, -0.5])
    query = runtime.target(sample_geometry(), baseline_qd=0.75 * torch.ones(6, dtype=torch.float64))
    assert query.target.component == 2
    assert torch.all(query.target.precision >= runtime.cfg.precision_min)
    assert torch.all(query.target.precision <= runtime.cfg.precision_max)

def test_safe_phase_and_nonfinite_output_disable_prior():
    safe = runtime.target(sample_geometry(phase=BimanualPhase.HOLD_SAFE), torch.zeros(6, dtype=torch.float64))
    invalid = nonfinite_runtime().target(sample_geometry(), torch.zeros(6, dtype=torch.float64))
    assert safe.target is None
    assert invalid.target is None
    assert invalid.diagnostics.reason == "nonfinite_prior"
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_runtime.py`

Expected: missing runtime adapter.

- [ ] **Step 3: Implement runtime-only input packing and Mahalanobis selection**

```python
tip_velocity = sample.contact_jacobian @ sample.qd
network_input = torch.cat((sample.fingertip_positions_b.reshape(-1), tip_velocity, sample.contact_mask.to(torch.float64), phase_one_hot(sample.phase)))
distance = (((baseline_tip_velocity[None] - means[:, 0].reshape(4, 15)) ** 2) * precision).sum(dim=1)
component = int(torch.argmin(distance - cfg.logit_weight * logits.log_softmax(-1)).item())
```

Load only model/artifact modules, bound log std and precision, zero precision rows for contacted fingertips, measure inference duration, and return an explicit disabled diagnostic on safe phases, timeout, exception, shape mismatch, or non-finite output.

- [ ] **Step 4: Run GREEN and import-boundary check**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_expert_prior_runtime.py && ! rg -n 'huggingface_hub|h5py|isaacgym|dexmanipnet.py|urdf_fk.py' go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py`

Expected: pass and no forbidden runtime import.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/expert_fingertip_prior/runtime.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_runtime.py
git commit -m "feat: add frozen O6 fingertip prior runtime"
```

### Task 9: Integrate the prior through atomic precontact and contact QPs

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py`
- Modify: `Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_hand_mpc.py`
- Modify: `Go2Pvcnn/tests/test_m1_bimanual_runtime.py`

**Interfaces:**
- Consumes: Task 8 runtime target and existing O6 positions/Jacobian.
- Produces: optional `prior_target` QP term, prior-enabled six-rate precontact projection, two-pass contact solve, same-cycle baseline fallback, extended diagnostics; default constructor semantics remain unchanged.

- [ ] **Step 1: Write disabled-equivalence and regularized-QP RED tests**

```python
def test_prior_disabled_is_exactly_existing_hand_solution():
    sample = _input(fingertip_positions_b=torch.zeros(5, 3, dtype=DTYPE))
    before = O6HandMpc().plan(sample)
    after = O6HandMpc(expert_prior=None).plan(sample)
    assert torch.equal(before.q_ref, after.q_ref)
    assert torch.equal(before.predicted_forces_b, after.predicted_forces_b)

def test_second_qp_failure_returns_same_cycle_baseline_without_pollution(monkeypatch):
    planner = O6HandMpc(expert_prior=fixed_prior())
    baseline = planner.plan(_input())
    force_second_qp_failure(monkeypatch)
    result = planner.plan(_input(q=0.2 * torch.ones(6, dtype=DTYPE)))
    assert result.diagnostics.prior_fallback_reason == "prior_qp_rejected"
    assert result.diagnostics.feasible
    assert not torch.equal(result.q_ref, baseline.q_ref)

def test_prior_precontact_projection_keeps_contact_latched_axes_and_rate_bounds():
    sample = _input(phase=BimanualPhase.PRELOAD, contact_mask=torch.tensor([True, False, False, False, False]))
    result = O6HandMpc(expert_prior=fixed_prior()).plan(sample)
    assert result.qd_ref[0].item() == 0.0
    assert result.qd_ref[1].item() == 0.0
    assert torch.all(result.qd_ref.abs() <= sample.qd_max)
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_hand_mpc.py tests/test_m1_bimanual_runtime.py`

Expected: missing position/prior fields or constructor parameter.

- [ ] **Step 3: Add the bounded quadratic term and atomic two-pass plan**

```python
def add_fingertip_prior(problem, sample, target, weight):
    selector = torch.zeros((15, problem.gradient.numel()), dtype=torch.float64)
    selector[:, :HAND_ACTIVE_DOF] = sample.contact_jacobian
    precision = torch.diag(target.precision)
    hessian = problem.hessian + 2.0 * weight * selector.T @ precision @ selector
    gradient = problem.gradient - 2.0 * weight * selector.T @ precision @ target.mean_velocity
    return replace(problem, hessian=hessian, gradient=gradient)

if sample.phase in {BimanualPhase.APPROACH, BimanualPhase.PRELOAD}:
    baseline = self._precontact_solution(sample)
    if self.expert_prior is None:
        return self._accept(baseline)
    query = self.expert_prior.target(sample, baseline.qd_ref)
    if query.target is None:
        return self._accept_same_cycle_baseline(baseline, query.diagnostics.reason)
    projected = self._solve_precontact_projection(sample, baseline, query.target)
    return self._accept(projected) if projected is not None else self._accept_same_cycle_baseline(baseline, "prior_qp_rejected")

baseline_result = solve_reference_qp(
    build_hand_contact_qp(sample, self.cfg),
    tolerance=self.cfg.qp_tolerance,
    max_iterations=self.cfg.qp_max_iterations,
)
if not baseline_result.success:
    return self._fallback(sample, "qp_infeasible")
baseline = self._solution(sample, baseline_result)
query = self.expert_prior.target(sample, baseline.qd_ref) if self.expert_prior else None
if query is None or query.target is None:
    return self._accept(baseline)
regularized_result = solve_reference_qp(
    build_hand_contact_qp(sample, self.cfg, prior_target=query.target),
    tolerance=self.cfg.qp_tolerance,
    max_iterations=self.cfg.qp_max_iterations,
)
return self._accept(self._solution(sample, regularized_result)) if regularized_result.success else self._accept_same_cycle_baseline(baseline)
```

Add `fingertip_positions_b` to `HandMpcInput` and runtime providers. `_solve_precontact_projection` uses a strong quadratic tracking term around the existing `PrecontactHandController` rate plus the bounded `J qdot` prior term, applies current position/rate bounds, and fixes latched finger axes to zero; when prior is disabled it is never called. Extend diagnostics without changing feasibility meaning. Preserve precontact closure, contact latching, friction, rate/position bounds, force equalities, and `_last_safe`; invalid prior never mutates it before baseline acceptance.

- [ ] **Step 4: Run GREEN and full Hand/QP regression**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_hand_mpc.py tests/test_m1_bimanual_runtime.py tests/test_m1_bimanual_full_action_teacher.py tests/test_m1_bimanual_o6_contact.py`

Expected: pass; disabled path exact, prior path constraints satisfied.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/hand_mpc.py Go2Pvcnn/go2_pvcnn/control/m1_bimanual_coordination/runtime.py Go2Pvcnn/tests/test_m1_bimanual_hand_mpc.py Go2Pvcnn/tests/test_m1_bimanual_runtime.py
git commit -m "feat: regularize O6 Hand MPC with fingertip prior"
```

### Task 10: Wire artifact configuration and diagnostics into Probe/Play

**Files:**
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py`
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py`
- Modify: `Go2Pvcnn/tests/test_m1_dual_panda_o6_verification.py`

**Interfaces:**
- Consumes: optional `--fingertip-prior-artifact` and Task 9 controller.
- Produces: startup SHA validation, JSON/JSONL prior diagnostics, prior-off/on CLI behavior; no default activation.

- [ ] **Step 1: Write CLI and diagnostic RED tests**

```python
def test_play_prior_is_opt_in_and_passed_to_wrapper():
    source = PLAY.read_text(encoding="utf-8")
    assert '"--fingertip-prior-artifact"' in source
    assert "default=None" in source
    assert "fingertip_prior_artifact=args.fingertip_prior_artifact" in source

def test_probe_summary_has_prior_atomic_fields():
    summary = make_probe_summary()
    assert {"prior_enabled", "prior_disabled_reason", "prior_qp_rejected_count", "prior_inference_p99_ms"} <= summary.keys()
```

- [ ] **Step 2: Run RED**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_dual_panda_o6_entrypoints_static.py tests/test_m1_dual_panda_o6_verification.py`

Expected: missing option/diagnostic fields.

- [ ] **Step 3: Implement explicit artifact loading and summaries**

```python
parser.add_argument("--fingertip-prior-artifact", type=Path, default=None)
prior = None if args.fingertip_prior_artifact is None else FrozenO6FingertipPrior.from_artifact(args.fingertip_prior_artifact)
wrapper = M1DualPandaO6BimanualWrapper(
    env,
    mode=args.mode,
    fingertip_prior_artifact=args.fingertip_prior_artifact,
)
```

Make explicit invalid artifacts fail before simulation begins. Add per-side component/probability/variance/cost/inference latency and baseline-vs-regularized tip-velocity summaries. Do not change safe termination or window persistence behavior.

- [ ] **Step 4: Run GREEN, help smoke, and focused suite**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_dual_panda_o6_entrypoints_static.py tests/test_m1_dual_panda_o6_verification.py tests/test_m1_bimanual_*.py`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/go2_pvcnn/tasks/m1_dual_panda_o6_bimanual_wrapper.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_probe.py Go2Pvcnn/scripts/m1_dual_panda_o6_bimanual_play.py Go2Pvcnn/tests/test_m1_dual_panda_o6_entrypoints_static.py Go2Pvcnn/tests/test_m1_dual_panda_o6_verification.py
git commit -m "feat: expose O6 fingertip prior in probe and play"
```

### Task 11: Fetch, convert, train, and distill the complete dataset

**Files:**
- Runtime output only: `Go2Pvcnn/data/external/dexmanipnet/`
- Create log: `notes/log/2026-09-11-t500-dexmanipnet-full-distillation.md`
- Modify: `notes/log/index.md`, `notes/todo.md`, `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`

**Interfaces:**
- Consumes: Tasks 1-10 CLIs and both fixed archives.
- Produces: archive manifest, sequence audit JSONL, deterministic shards, ensemble metrics, distilled student artifact, rejection counts, latency report.

- [ ] **Step 1: Download and verify both archives**

Run:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_fetch_dexmanipnet.py --root data/external/dexmanipnet
```

Expected: exit `0`; manifest reports the fixed HF revision, fixed ManipTrans commit, two archive sizes and two lowercase SHA-256 values; no partial extraction directories remain.

- [ ] **Step 2: Convert all auditable sequences twice**

Run:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_convert_dexmanipnet.py --root data/external/dexmanipnet --output-dir data/external/dexmanipnet/converted/run_a --seed 42
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_convert_dexmanipnet.py --root data/external/dexmanipnet --output-dir data/external/dexmanipnet/converted/run_b --seed 42
cmp data/external/dexmanipnet/converted/run_a/aggregate.manifest.json data/external/dexmanipnet/converted/run_b/aggregate.manifest.json
```

Expected: both sources appear in the audit, every discovered sequence-side has exactly one accepted/rejected row, accepted count is positive, rejection reasons are non-empty, and both aggregate SHA values match.

- [ ] **Step 3: Train the expert ensemble**

Run:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_train_fingertip_expert.py --dataset-manifest data/external/dexmanipnet/converted/run_a/aggregate.manifest.json --output-dir data/external/dexmanipnet/artifacts/expert --member-seeds 42 43 44 45 46 --epochs 200
```

Expected: held-out first-step and endpoint RMSE each improve at least `10%` over zero baselines; 80% interval coverage is within `[0.65,0.95]`; ensemble manifest is marked deployable.

- [ ] **Step 4: Distill the student twice**

Run:

```bash
cd Go2Pvcnn
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_distill_fingertip_prior.py --dataset-manifest data/external/dexmanipnet/converted/run_a/aggregate.manifest.json --ensemble-dir data/external/dexmanipnet/artifacts/expert --output-dir data/external/dexmanipnet/artifacts/student_a --seed 42 --epochs 200
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_distill_fingertip_prior.py --dataset-manifest data/external/dexmanipnet/converted/run_a/aggregate.manifest.json --ensemble-dir data/external/dexmanipnet/artifacts/expert --output-dir data/external/dexmanipnet/artifacts/student_b --seed 42 --epochs 200
cmp data/external/dexmanipnet/artifacts/student_a/metadata.json data/external/dexmanipnet/artifacts/student_b/metadata.json
cmp data/external/dexmanipnet/artifacts/student_a/metrics.json data/external/dexmanipnet/artifacts/student_b/metrics.json
```

Expected: metric JSON and artifact SHA match; student NLL increase is `<=0.05 nat/dim`, endpoint RMSE increase `<=5%`, zero-baseline improvements remain `>=10%`, and CPU p99 is `<2 ms`.

- [ ] **Step 5: Record evidence and commit notes only**

The log must include exact archive/shard/ensemble/student SHAs, counts by source/hand/phase/contact/rejection, commands, metrics, latency, Git commit, and explicit statement that no Isaac physical claim was made.

```bash
git add notes/log notes/todo.md notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md
git commit -m "docs: record DexManipNet prior distillation"
```

### Task 12: Run CPU/QP, GPU0, and unseen-operation gates

**Files:**
- Modify: `Go2Pvcnn/scripts/m1_dual_panda_o6_verify.py`
- Create: `Go2Pvcnn/tests/test_m1_bimanual_expert_prior_acceptance.py`
- Create log: `notes/log/2026-09-11-t500-dexmanipnet-prior-gpu0.md`
- Modify: `notes/log/index.md`, `notes/todo.md`, `notes/todo/T500-m1-dual-panda-o6-bimanual-mpc.md`, `README.md`

**Interfaces:**
- Consumes: SHA-pinned deployable student from Task 11.
- Produces: verification-ladder JSONL, prior-off/on aggregate, GPU0 smoke, formal T500 acceptance status.

- [ ] **Step 1: Add the expert-prior tests to the existing verification ladder**

```python
CPU_TESTS += (
    "tests/test_m1_bimanual_expert_prior_contracts.py",
    "tests/test_m1_bimanual_expert_prior_download.py",
    "tests/test_m1_bimanual_expert_prior_urdf_fk.py",
    "tests/test_m1_bimanual_expert_prior_dexmanipnet.py",
    "tests/test_m1_bimanual_expert_prior_preprocess.py",
    "tests/test_m1_bimanual_expert_prior_storage.py",
    "tests/test_m1_bimanual_expert_prior_model.py",
    "tests/test_m1_bimanual_expert_prior_train_static.py",
    "tests/test_m1_bimanual_expert_prior_distill.py",
    "tests/test_m1_bimanual_expert_prior_artifact.py",
)
PURE_QP_TESTS += (
    "tests/test_m1_bimanual_expert_prior_runtime.py",
    "tests/test_m1_bimanual_expert_prior_acceptance.py",
)
```

The acceptance test must exercise the pure gate without Isaac:

```python
def test_acceptance_rejects_any_task_or_safety_regression():
    metrics = {
        "nll_improvement_fraction": 0.20,
        "prior_off_jerk_p95": 1.0,
        "prior_on_jerk_p95": 0.9,
        "prior_off_task_success": 1.0,
        "prior_on_task_success": 0.99,
        "prior_off_safety_rejections": 0,
        "prior_on_safety_rejections": 0,
    }
    assert not accept_prior_comparison(metrics).accepted
    assert accept_prior_comparison(metrics).reason == "task_success_regression"
```

Run RED before editing and confirm the ladder omits the new tests; run GREEN after editing and require every listed test command to exit `0`.

- [ ] **Step 2: Run the complete CPU and pure-QP regression**

Run: `cd Go2Pvcnn && PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python -m pytest -q tests/test_m1_bimanual_*.py tests/test_m1_dual_panda_o6_*.py`

Expected: zero failures; include exact pass count and duration in the log. Then run `compileall` on the new package and five scripts and require exit `0`.

- [ ] **Step 3: Run fixed GPU0 prior-off/on smoke**

Use the same seed, initial state, step count, controller parameters, and GPU0 for two probe runs; the second adds `--fingertip-prior-artifact <student-dir>`.

Run:

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py --headless --num-envs 1 --steps 400 --seed 42 --report data/external/dexmanipnet/artifacts/gpu0_prior_off.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py --headless --num-envs 1 --steps 400 --seed 42 --fingertip-prior-artifact data/external/dexmanipnet/artifacts/student_a --report data/external/dexmanipnet/artifacts/gpu0_prior_on.json
```

Expected: correct left/right mirror direction, real folded O6 Jacobian, 100 Hz prior calls, no non-finite value, hard limit, unintended reset, or safety rejection; prior p99 inference remains `<2 ms`; any second-QP failure uses same-cycle baseline and increments its diagnostic.

- [ ] **Step 4: Run the controlled unseen-operation comparison**

Run prior-off/on with identical seeds `42/43/44` and the existing fixed-box task. Aggregate executed fingertip NLL, jerk p95, Hand QP feasibility, slip margin, limits, safety rejection, and task completion.

Run:

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py --headless --num-envs 1 --seeds 42 43 44 --trials-per-seed 10 --report data/external/dexmanipnet/artifacts/formal_prior_off.json --jsonl data/external/dexmanipnet/artifacts/formal_prior_off.jsonl
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_bimanual_probe.py --headless --num-envs 1 --seeds 42 43 44 --trials-per-seed 10 --fingertip-prior-artifact data/external/dexmanipnet/artifacts/student_a --report data/external/dexmanipnet/artifacts/formal_prior_on.json --jsonl data/external/dexmanipnet/artifacts/formal_prior_on.jsonl
PYTHONPATH=$PWD /home/xk/miniconda3/envs/go2/bin/python scripts/m1_dual_panda_o6_eval_fingertip_prior.py --prior-off-report data/external/dexmanipnet/artifacts/formal_prior_off.json --prior-on-report data/external/dexmanipnet/artifacts/formal_prior_on.json --output data/external/dexmanipnet/artifacts/prior_comparison.json
```

Expected: prior-on median NLL improves `>=10%`, jerk p95 does not increase, and no feasibility/safety/task metric regresses. If the existing Arm/orientation gate fails, mark whole-task acceptance blocked while retaining only the local prior smoke result.

- [ ] **Step 5: Run or explicitly defer formal 30/30 and finalize evidence**

Formal success still requires seeds `42/43/44`, ten complete trials each, all completing approach through release. If Vulkan/DRM or the existing Arm gate prevents the run, record the exact blocker and do not claim 30/30.

Update README with fetch/convert/train/distill/probe commands, update all required notes, run `git diff --check`, and commit only source/tests/docs/manifests without raw data or weights:

```bash
git add README.md Go2Pvcnn/scripts/m1_dual_panda_o6_verify.py Go2Pvcnn/tests/test_m1_bimanual_expert_prior_acceptance.py notes
git commit -m "test: verify DexManipNet fingertip prior integration"
```

## Final Verification Checklist

- [ ] `git status --short` contains no raw archive, extracted sequence, shard, ensemble, or student weight staged for Git.
- [ ] Full focused pytest exits `0` with an exact pass count.
- [ ] New package and CLIs compile and all `--help` commands exit `0`.
- [ ] Archive, aggregate, ensemble, and student manifests contain verified SHA-256 values.
- [ ] Teacher gate passes before distillation; student gate passes before runtime activation.
- [ ] Prior-disabled Hand MPC is exactly compatible with the current baseline.
- [ ] Prior failure always resolves to same-cycle baseline or existing safety fallback without state pollution.
- [ ] GPU0 evidence distinguishes local prior smoke from whole-system T500 acceptance.
- [ ] Formal 30/30 is reported only if all 30 complete physical trials pass.
