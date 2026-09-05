# M1 + Right Panda + Right O6 Asset Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the T600.1 project-owned M1 + right Panda + right O6 single articulation with 29 active control channels and an audited 2000-step Isaac Lab stability gate.

**Architecture:** Reuse the already tested O6 source-closure commit as immutable project input, but build a separate single-arm asset rooted at `/M1SinglePandaO6`. The builder keeps the existing unprefixed Panda names for T400 compatibility, prefixes only the right O6 rigid bodies and joints, and publishes a SHA-pinned manifest consumed by a separate Isaac Lab asset configuration and verifier.

**Tech Stack:** Python 3.11, pytest, Git LFS, USD/PXR, Isaac Sim 5.1, Isaac Lab, existing M1/Panda asset builders, NVIDIA GPU0.

## Global Constraints

- Execute on branch `o6_400` in an isolated worktree; do not work in the dirty main checkout.
- Preserve all T400/T500 assets, public imports, Gym IDs, checkpoint contracts, and default behavior.
- Treat `/home/xk/coding/o6asset` as read-only. Never author, normalize, or repair files in place.
- Reuse only committed O6 source-closure content from `ac95a9b`; do not consume uncommitted files from `.worktrees/t500-dual-panda-o6-mpc`.
- The T600 runtime asset lives under `Go2Pvcnn/assets/m1_single_panda_o6/` and is independent of the dual-arm runtime asset.
- The active order is exactly 12 M1 leg joints + 4 M1 wheel joints + 7 unprefixed Panda arm joints + 6 `right_` O6 joints = 29 channels.
- O6 mimic joints are physical DOFs but never independent active control channels.
- The expected physical count is 34, but the verifier must measure it and compare runtime against the manifest instead of accepting a hard-coded assumption.
- Use the arm-only `panda_arm.urdf`; `panda_finger_joint.*` must not appear in the final asset.
- The final asset has one articulation root at `/M1SinglePandaO6/BASE_LINK` and exactly two enabled assembly joints: M1-to-Panda and Panda-wrist-to-O6.
- All generated `.usd` and `.STL` files remain Git LFS objects. JSON manifests and Python source remain normal Git objects.
- T600.1 ends at asset/configuration verification. Do not add the Gym environment, MPC, perception, task state machine, reward, Residual policy, or training entrypoint.
- Every task follows RED -> minimal GREEN -> focused regression -> commit.

## Locked File Structure

```text
.gitattributes
Go2Pvcnn/assets/m1_dual_panda_o6/
  o6_left/
  o6_right/
  source_manifest.json
Go2Pvcnn/assets/m1_single_panda_o6/
  m1_single_panda_o6.usd
  panda_arm/
  prefixed/right_o6.usd
  asset_manifest.json
Go2Pvcnn/go2_pvcnn/assets/m1_single_panda_o6.py
Go2Pvcnn/scripts/normalize_o6_assets.py
Go2Pvcnn/scripts/build_m1_single_panda_o6_asset.py
Go2Pvcnn/scripts/verify_m1_single_panda_o6_asset.py
Go2Pvcnn/tests/test_m1_dual_panda_o6_sources.py
Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py
Go2Pvcnn/tests/test_m1_single_panda_o6_contracts.py
docs/superpowers/runbooks/2026-09-05-m1-single-panda-o6-asset-foundation.md
notes/log/2026-09-05-m1-single-panda-o6-asset-foundation.md
notes/log/index.md
notes/todo.md
notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md
```

---

### Task 1: Import and Reverify the O6 Source Closure

**Files:**
- Modify from audited commit: `.gitattributes`
- Create from audited commit: `Go2Pvcnn/scripts/normalize_o6_assets.py`
- Create from audited commit: `Go2Pvcnn/tests/test_m1_dual_panda_o6_sources.py`
- Create from audited commit: `Go2Pvcnn/assets/m1_dual_panda_o6/o6_left/`
- Create from audited commit: `Go2Pvcnn/assets/m1_dual_panda_o6/o6_right/`
- Create from audited commit: `Go2Pvcnn/assets/m1_dual_panda_o6/source_manifest.json`

**Interfaces:**
- Consumes: read-only vendor tree `/home/xk/coding/o6asset`.
- Produces: `normalize_o6_sources(source_root: Path, destination_root: Path) -> dict[str, object]` and a deterministic SHA256 `source_manifest.json`.
- T600 consumes only `o6_right/O6_right.usd` and its closed configuration/mesh tree. The left source remains shared project input for T500 and is not loaded by T600.

- [ ] **Step 1: Verify the clean branch does not already contain the source closure**

Run:

```bash
test ! -e Go2Pvcnn/scripts/normalize_o6_assets.py
test ! -e Go2Pvcnn/assets/m1_dual_panda_o6/source_manifest.json
```

Expected: exit `0`. If either path already exists, stop and compare it byte-for-byte with commit `ac95a9b` before continuing; do not overwrite divergent content.

- [ ] **Step 2: Run the source test path and confirm RED**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_sources.py
```

Expected: pytest exits nonzero because the test file is absent. This is the source-closure RED, not a product regression.

- [ ] **Step 3: Import the reviewed source-closure commit without taking later T500 work**

Run from the repository root:

```bash
git cherry-pick ac95a9b
```

Expected: one commit is added. Its file set is limited to `.gitattributes`, `normalize_o6_assets.py`, the source test, normalized left/right O6 source files, and `source_manifest.json`.

Audit the imported commit:

```bash
git diff-tree --no-commit-id --name-only -r HEAD
git show --check --oneline HEAD
git lfs ls-files | rg 'Go2Pvcnn/assets/m1_dual_panda_o6/(o6_left|o6_right)/'
```

Expected: no runtime dual-arm builder/config/controller files are present; all `.usd` and `.STL` source assets are listed by Git LFS.

- [ ] **Step 4: Re-run the pure source tests and rematerialize into a temporary directory**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_dual_panda_o6_sources.py
source_probe_dir=$(mktemp -d /tmp/t600-o6-source.XXXXXX)
python scripts/normalize_o6_assets.py \
  --source-root '/home/xk/coding/o6asset' \
  --destination-root "$source_probe_dir"
test -f "$source_probe_dir/o6_right/O6_right.usd"
test -f "$source_probe_dir/source_manifest.json"
```

Expected: all source tests pass; normalization exits `0`; the temporary result contains both complete sides, all four configuration layers per side, and a SHA256 entry for every copied file. Keep the temporary directory only as disposable evidence and do not stage it.

- [ ] **Step 5: Record the imported lineage before moving on**

Run:

```bash
git rev-parse HEAD
git status --short
```

Expected: the worktree is clean. The imported commit itself is the Task 1 commit, so do not create an empty follow-up commit.

### Task 2: Build the Single M1-Panda-O6 Articulation

**Files:**
- Create: `Go2Pvcnn/scripts/build_m1_single_panda_o6_asset.py`
- Create: `Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py`
- Create at runtime: `Go2Pvcnn/assets/m1_single_panda_o6/panda_arm/`
- Create at runtime: `Go2Pvcnn/assets/m1_single_panda_o6/prefixed/right_o6.usd`
- Create at runtime: `Go2Pvcnn/assets/m1_single_panda_o6/m1_single_panda_o6.usd`
- Create at runtime: `Go2Pvcnn/assets/m1_single_panda_o6/asset_manifest.json`

**Interfaces:**
- Consumes: `Go2Pvcnn/assets/m1_dual_panda_o6/source_manifest.json`, normalized right O6 source, existing `Go2Pvcnn/assets/m1_panda/m1_floating.usda`, and existing arm-only Panda URDF.
- Produces: `build_asset(asset_root: Path, o6_source_root: Path, force_panda_conversion: bool = False) -> Path`.
- Produces one articulation rooted at `/M1SinglePandaO6/BASE_LINK`, with mount joints `/M1SinglePandaO6/joints/panda_mount_joint` and `/M1SinglePandaO6/joints/right_hand_mount_joint`.

- [ ] **Step 1: Write static builder contract tests**

Create `Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py` with these first tests:

```python
from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts/build_m1_single_panda_o6_asset.py"
VERIFIER = PROJECT_ROOT / "scripts/verify_m1_single_panda_o6_asset.py"


def _builder_source() -> str:
    assert BUILDER.is_file()
    return BUILDER.read_text(encoding="utf-8")


def test_builder_freezes_single_arm_topology_and_counts():
    source = _builder_source()
    for token in (
        'ROOT_PRIM = "/M1SinglePandaO6"',
        'PANDA_PRIM = f"{ROOT_PRIM}/Panda"',
        'RIGHT_HAND_PRIM = f"{PANDA_PRIM}/right_o6"',
        'EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"',
        "EXPECTED_ACTIVE_DOF_COUNT = 29",
        "EXPECTED_ASSEMBLY_JOINT_COUNT = 2",
        'PANDA_ARM_URDF = "panda_arm.urdf"',
    ):
        assert token in source
    assert "panda_arm_hand.urdf" not in source


def test_builder_reads_only_the_normalized_right_o6_source():
    source = _builder_source()
    assert 'RIGHT_O6_ENTRY = "o6_right/O6_right.usd"' in source
    assert "o6_left/O6_left.usd" not in source
    tree = ast.parse(source)
    write_calls = {
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(word in ast.unparse(node.func).lower() for word in ("write", "save", "export"))
    }
    assert all("o6_source_root" not in call for call in write_calls)


def test_builder_has_explicit_reopen_and_manifest_phases():
    source = _builder_source()
    for name in (
        "validate_o6_source",
        "ensure_arm_only_panda",
        "write_right_prefixed_o6",
        "create_stage",
        "assemble_panda",
        "assemble_right_o6",
        "remove_child_roots_scenes_and_root_joints",
        "author_convex_collision_approximations",
        "validate_stage_contract",
        "export_reopen_validate_and_manifest",
        "build_asset",
    ):
        assert f"def {name}(" in source
```

- [ ] **Step 2: Run the builder tests and confirm RED**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_single_panda_o6_asset_static.py
```

Expected: failures because `build_m1_single_panda_o6_asset.py` is absent.

- [ ] **Step 3: Implement the builder with explicit, single-purpose phases**

Create `Go2Pvcnn/scripts/build_m1_single_panda_o6_asset.py`. Freeze these constants and top-level API:

```python
ROOT_PRIM = "/M1SinglePandaO6"
PANDA_PRIM = f"{ROOT_PRIM}/Panda"
RIGHT_HAND_PRIM = f"{PANDA_PRIM}/right_o6"
EXPECTED_ARTICULATION_ROOT = f"{ROOT_PRIM}/BASE_LINK"
PANDA_MOUNT_JOINT = f"{ROOT_PRIM}/joints/panda_mount_joint"
RIGHT_HAND_MOUNT_JOINT = f"{ROOT_PRIM}/joints/right_hand_mount_joint"
EXPECTED_ACTIVE_DOF_COUNT = 29
EXPECTED_ASSEMBLY_JOINT_COUNT = 2
PANDA_ARM_URDF = "panda_arm.urdf"
RIGHT_O6_ENTRY = "o6_right/O6_right.usd"
MOUNT_CLEARANCE_M = 0.0
MOUNT_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
BUILD_SCHEMA = 1


def build_asset(
    asset_root: Path,
    o6_source_root: Path,
    force_panda_conversion: bool = False,
) -> Path:
    asset_root = Path(asset_root).resolve()
    o6_source_root = Path(o6_source_root).resolve(strict=True)
    source = validate_o6_source(o6_source_root)
    panda_usd = ensure_arm_only_panda(asset_root, force=force_panda_conversion)
    right_o6_usd = write_right_prefixed_o6(
        o6_source_root / RIGHT_O6_ENTRY,
        asset_root / "prefixed/right_o6.usd",
    )
    stage = create_stage(asset_root)
    assemble_panda(stage, panda_usd)
    assemble_right_o6(stage, right_o6_usd)
    remove_child_roots_scenes_and_root_joints(stage)
    convex_mesh_count = author_convex_collision_approximations(stage)
    validate_stage_contract(stage, "pre-export")
    return export_reopen_validate_and_manifest(
        stage, asset_root, source, convex_mesh_count
    )
```

Implement each named phase using the proven behavior from the current single-Panda builder and audited T500 builder:

- `validate_o6_source()` reads `source_manifest.json`, requires schema `1`, verifies every SHA256 entry, and requires `manifest["right"]["entry"] == RIGHT_O6_ENTRY`.
- `ensure_arm_only_panda()` converts `assets/m1_panda/panda_source/franka_description/robots/panda_arm.urdf` with `fix_base=True`, `merge_fixed_joints=False`, position target gains `80.0/4.0`, and removes only the known empty `panda_link8/visuals` reference.
- `write_right_prefixed_o6()` opens and flattens the normalized right hand, clears flattened layer comment/documentation, and prefixes every rigid-body and joint basename with `right_` via `Usd.NamespaceEditor` before export.
- `create_stage()` references `../m1_panda/m1_floating.usda` at `ROOT_PRIM` and computes the M1 mount-patch top surface with half extents `(0.11, 0.10)`.
- `assemble_panda()` mounts unprefixed `panda_link0` to `BASE_LINK` at the measured top surface and authors one enabled fixed joint at `PANDA_MOUNT_JOINT`.
- `assemble_right_o6()` aligns `RIGHT_HAND_PRIM` to `/M1SinglePandaO6/Panda/panda_link8` and authors one enabled fixed joint to `right_hand_base_link` at `RIGHT_HAND_MOUNT_JOINT`.
- `remove_child_roots_scenes_and_root_joints()` deactivates child root joints/scenes and removes all articulation-root APIs except `EXPECTED_ARTICULATION_ROOT`.
- `author_convex_collision_approximations()` requires every O6 collision mesh to use `convexHull` and returns a positive mesh count.
- `validate_stage_contract()` requires one articulation root, exactly two enabled assembly joints, no `panda_finger_joint`, 29 active paths, and a measured PXR physical path count written to the manifest.
- `export_reopen_validate_and_manifest()` exports, reopens, reruns the same contract, and atomically writes schema, source SHA, asset SHA, active paths, physical count, assembly joints, mount transforms, O6 active order, mimic map, and collision approximation.

The CLI must require both roots:

```python
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--o6-source-root", type=Path, required=True)
parser.add_argument("--force-panda-conversion", action="store_true")
```

- [ ] **Step 4: Run static tests, build the asset, and inspect the manifest**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_single_panda_o6_asset_static.py
CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p \
  scripts/build_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --o6-source-root assets/m1_dual_panda_o6 \
  --headless
```

Expected: tests pass; build exits `0`; the exported asset reopens; the manifest reports one articulation root, two assembly joints, 29 active paths, no Panda finger joints, a positive O6 convex mesh count, and PXR physical count `34` on the accepted source versions.

- [ ] **Step 5: Commit the single-articulation builder and generated asset**

Run:

```bash
git add Go2Pvcnn/scripts/build_m1_single_panda_o6_asset.py \
  Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py \
  Go2Pvcnn/assets/m1_single_panda_o6
git commit -m "feat: build M1 single-Panda O6 articulation"
```

### Task 3: Freeze the 29-Channel Isaac Lab Asset Contract

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/assets/m1_single_panda_o6.py`
- Create: `Go2Pvcnn/tests/test_m1_single_panda_o6_contracts.py`

**Interfaces:**
- Produces: `M1_SINGLE_PANDA_O6_CFG`, `M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES`, `M1_SINGLE_PANDA_O6_ACTIVE_DOF_COUNT`, `RIGHT_O6_ACTIVE_JOINT_NAMES`, `RIGHT_O6_MIMIC_MAP`, body-name constants, and `resolve_active_joint_ids(runtime_joint_names)`.
- Downstream T600.2 consumes the exact 29-channel order and body names without inspecting USD internals.

- [ ] **Step 1: Write exact order, mimic, body-name, and actuator tests**

Create `Go2Pvcnn/tests/test_m1_single_panda_o6_contracts.py` with a self-contained Isaac Lab module stub followed by the contract tests:

```python
from __future__ import annotations

import copy
import importlib
import sys
import types

import pytest


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def replace(self, **kwargs):
        result = copy.deepcopy(self)
        result.__dict__.update(kwargs)
        return result

    def copy(self):
        return copy.deepcopy(self)


@pytest.fixture()
def contract(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    sim.UsdFileCfg = _Cfg
    sim.RigidBodyPropertiesCfg = _Cfg
    sim.ArticulationRootPropertiesCfg = _Cfg

    actuators = types.ModuleType("isaaclab.actuators")
    actuators.DCMotorCfg = _Cfg
    actuators.ImplicitActuatorCfg = _Cfg

    assets_pkg = types.ModuleType("isaaclab.assets")
    articulation = types.ModuleType("isaaclab.assets.articulation")

    class _ArticulationCfg(_Cfg):
        InitialStateCfg = _Cfg

    articulation.ArticulationCfg = _ArticulationCfg
    utils_pkg = types.ModuleType("isaaclab.utils")
    utils_assets = types.ModuleType("isaaclab.utils.assets")
    utils_assets.ISAACLAB_NUCLEUS_DIR = "/Isaac/Nucleus"
    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.sim": sim,
        "isaaclab.actuators": actuators,
        "isaaclab.assets": assets_pkg,
        "isaaclab.assets.articulation": articulation,
        "isaaclab.utils": utils_pkg,
        "isaaclab.utils.assets": utils_assets,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    for name in tuple(sys.modules):
        if name == "go2_pvcnn.assets" or name.startswith("go2_pvcnn.assets."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("go2_pvcnn.assets.m1_single_panda_o6")


def test_active_order_is_16_plus_7_plus_6(contract):
    names = contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    assert contract.M1_SINGLE_PANDA_O6_ACTIVE_DOF_COUNT == 29
    assert len(names) == len(set(names)) == 29
    assert names[:16] == contract.M1_BASE_ACTIVE_JOINT_NAMES
    assert names[16:23] == tuple(f"panda_joint{i}" for i in range(1, 8))
    assert names[23:29] == contract.RIGHT_O6_ACTIVE_JOINT_NAMES


def test_mimics_are_metadata_not_active_channels(contract):
    assert contract.RIGHT_O6_MIMIC_MAP == {
        "right_thumb_ip": ("right_thumb_cmc_pitch", 1.86, 0.0),
        "right_index_dip": ("right_index_mcp_pitch", 0.89, 0.0),
        "right_middle_dip": ("right_middle_mcp_pitch", 0.89, 0.0),
        "right_ring_dip": ("right_ring_mcp_pitch", 0.89, 0.0),
        "right_pinky_dip": ("right_pinky_mcp_pitch", 0.89, 0.0),
    }
    assert not set(contract.RIGHT_O6_MIMIC_MAP) & set(
        contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    )


def test_runtime_names_and_actuator_groups_are_isolated(contract):
    assert contract.M1_SINGLE_PANDA_O6_BASE_BODY_NAME == "BASE_LINK"
    assert contract.PANDA_WRIST_BODY_NAME == "panda_link8"
    assert contract.RIGHT_O6_PALM_BODY_NAME == "right_hand_base_link"
    assert contract.RIGHT_O6_FINGERTIP_BODY_NAMES == tuple(
        f"right_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
    )
    assert set(contract.M1_SINGLE_PANDA_O6_CFG.actuators) == {
        "legs", "wheels", "panda_shoulder", "panda_forearm", "right_o6"
    }


def test_runtime_mapping_rejects_missing_and_duplicate_names(contract):
    runtime = tuple(reversed(contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES))
    ids = contract.resolve_active_joint_ids(runtime)
    assert tuple(runtime[index] for index in ids) == contract.M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES
    with pytest.raises(ValueError, match="missing"):
        contract.resolve_active_joint_ids(runtime[1:])
    with pytest.raises(ValueError, match="duplicate"):
        contract.resolve_active_joint_ids(runtime + (runtime[-1],))
```

- [ ] **Step 2: Run the contract tests and confirm RED**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_single_panda_o6_contracts.py
```

Expected: import fails because `go2_pvcnn.assets.m1_single_panda_o6` is absent.

- [ ] **Step 3: Implement the exact asset and mapping contract**

Create `Go2Pvcnn/go2_pvcnn/assets/m1_single_panda_o6.py` with these public constants and mapping behavior:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from isaaclab.actuators import ImplicitActuatorCfg

from go2_pvcnn.assets import M1_CFG


M1_SINGLE_PANDA_O6_USD_PATH = str(
    Path(__file__).resolve().parents[2]
    / "assets/m1_single_panda_o6/m1_single_panda_o6.usd"
)
M1_SINGLE_PANDA_O6_BASE_BODY_NAME = "BASE_LINK"
PANDA_WRIST_BODY_NAME = "panda_link8"
RIGHT_O6_PALM_BODY_NAME = "right_hand_base_link"
RIGHT_O6_FINGERTIP_BODY_NAMES = tuple(
    f"right_{name}_distal" for name in ("thumb", "index", "middle", "ring", "pinky")
)
M1_BASE_ACTIVE_JOINT_NAMES = (
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT",
    "FAR_FOOT_JOINT", "FBL_FOOT_JOINT", "RAR_FOOT_JOINT", "RBL_FOOT_JOINT",
)
PANDA_ACTIVE_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
RIGHT_O6_ACTIVE_JOINT_NAMES = tuple(
    f"right_{name}" for name in (
        "thumb_cmc_pitch", "thumb_cmc_yaw", "index_mcp_pitch",
        "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch",
    )
)
RIGHT_O6_MIMIC_MAP = {
    "right_thumb_ip": ("right_thumb_cmc_pitch", 1.86, 0.0),
    "right_index_dip": ("right_index_mcp_pitch", 0.89, 0.0),
    "right_middle_dip": ("right_middle_mcp_pitch", 0.89, 0.0),
    "right_ring_dip": ("right_ring_mcp_pitch", 0.89, 0.0),
    "right_pinky_dip": ("right_pinky_mcp_pitch", 0.89, 0.0),
}
M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES = (
    *M1_BASE_ACTIVE_JOINT_NAMES,
    *PANDA_ACTIVE_JOINT_NAMES,
    *RIGHT_O6_ACTIVE_JOINT_NAMES,
)
M1_SINGLE_PANDA_O6_ACTIVE_DOF_COUNT = 29


def resolve_active_joint_ids(runtime_joint_names: Sequence[str]) -> tuple[int, ...]:
    name_to_id: dict[str, int] = {}
    duplicates: set[str] = set()
    for joint_id, name in enumerate(runtime_joint_names):
        if name in name_to_id:
            duplicates.add(name)
        else:
            name_to_id[name] = joint_id
    if duplicates:
        raise ValueError(f"duplicate runtime joint names: {sorted(duplicates)}")
    missing = [name for name in M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES if name not in name_to_id]
    if missing:
        raise ValueError(f"missing active runtime joint names: {missing}")
    return tuple(name_to_id[name] for name in M1_SINGLE_PANDA_O6_ACTIVE_JOINT_NAMES)
```

Build `M1_SINGLE_PANDA_O6_CFG` from `M1_CFG.copy()` and replace only the USD path, joint initial positions, and these task-local actuator groups:

```python
M1_SINGLE_PANDA_O6_CFG = M1_CFG.copy()
M1_SINGLE_PANDA_O6_CFG.spawn = M1_SINGLE_PANDA_O6_CFG.spawn.replace(
    usd_path=M1_SINGLE_PANDA_O6_USD_PATH
)
M1_SINGLE_PANDA_O6_CFG.init_state.joint_pos.update({
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.650,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "right_(thumb|index|middle|ring|pinky)_.*": 0.1,
})
M1_SINGLE_PANDA_O6_CFG.actuators.update({
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0,
        velocity_limit_sim=2.175, stiffness=80.0, damping=4.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0,
        velocity_limit_sim=2.61, stiffness=80.0, damping=4.0,
    ),
    "right_o6": ImplicitActuatorCfg(
        joint_names_expr=list(RIGHT_O6_ACTIVE_JOINT_NAMES),
        effort_limit_sim=10.0, velocity_limit_sim=1.0,
        stiffness=20.0, damping=1.0,
    ),
})
```

Do not add an actuator for mimic joints. Consumers import this contract from `go2_pvcnn.assets.m1_single_panda_o6`; keep `go2_pvcnn/assets/__init__.py` unchanged because the current main branch does not re-export task-specific M1/Panda asset modules there.

- [ ] **Step 4: Run focused and legacy asset contract tests**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_m1_single_panda_o6_contracts.py \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_wbc_contracts.py
```

Expected: all tests pass. Legacy `M1_PANDA_CFG`, its 25 physical DOFs, and its 23-channel WBC order remain unchanged.

- [ ] **Step 5: Commit the 29-channel public contract**

Run:

```bash
git add Go2Pvcnn/go2_pvcnn/assets/m1_single_panda_o6.py \
  Go2Pvcnn/tests/test_m1_single_panda_o6_contracts.py
git commit -m "feat: define single-Panda O6 asset contract"
```

### Task 4: Verify Offline Topology and 2000-Step Physics

**Files:**
- Create: `Go2Pvcnn/scripts/verify_m1_single_panda_o6_asset.py`
- Modify: `Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py`
- Modify on successful verification: `Go2Pvcnn/assets/m1_single_panda_o6/asset_manifest.json`

**Interfaces:**
- Consumes: combined USD, source manifest, asset manifest, and `M1_SINGLE_PANDA_O6_CFG` semantics.
- Produces: one JSON object containing dependency closure, articulation roots, measured physical DOF/body/joint names, active mapping, four-wheel contact ratio, mount drift, limit/contact/reset/nonfinite counts, and `hard_gates_passed`.
- Updates runtime evidence in `asset_manifest.json` only when every hard gate passes.

- [ ] **Step 1: Add failing verifier and gate tests**

Append to `Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py`:

```python
def test_verifier_freezes_runtime_gate_and_reports_required_metrics():
    assert VERIFIER.is_file()
    source = VERIFIER.read_text(encoding="utf-8")
    for token in (
        "EXPECTED_ACTIVE_DOF_COUNT = 29",
        "EXPECTED_PHYSICS_STEPS = 2000",
        'EXPECTED_ARTICULATION_ROOT = "/M1SinglePandaO6/BASE_LINK"',
        '"measured_physical_dof_count"',
        '"four_wheel_contact_ratio"',
        '"max_mount_position_drift_m"',
        '"max_mount_orientation_drift_rad"',
        '"nonfinite_count"',
        '"hard_joint_limit_count"',
        '"unexpected_contact_count"',
        '"unexpected_reset_count"',
        '"base_instability_count"',
        '"hard_gates_passed"',
    ):
        assert token in source


def test_runtime_physical_count_is_compared_to_manifest_not_magic_34():
    source = VERIFIER.read_text(encoding="utf-8")
    assert 'runtime["measured_physical_dof_count"] == manifest["physical_dof_count"]' in source
    assert 'runtime["measured_physical_dof_count"] == 34' not in source
```

- [ ] **Step 2: Run the verifier tests and confirm RED**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_m1_single_panda_o6_asset_static.py
```

Expected: verifier-related tests fail because `verify_m1_single_panda_o6_asset.py` is absent.

- [ ] **Step 3: Implement dependency, topology, and runtime verification**

Create `Go2Pvcnn/scripts/verify_m1_single_panda_o6_asset.py` with these frozen limits:

```python
EXPECTED_ACTIVE_DOF_COUNT = 29
EXPECTED_PHYSICS_STEPS = 2000
EXPECTED_ARTICULATION_ROOT = "/M1SinglePandaO6/BASE_LINK"
MAX_MOUNT_POSITION_DRIFT_M = 1.0e-3
MAX_MOUNT_ORIENTATION_DRIFT_RAD = 1.0e-3
CONTACT_FORCE_THRESHOLD_N = 5.0
REMOTE_PREFIXES = ("omniverse://", "http://", "https://")
BUILTIN_MDL_ALLOWLIST = {"OmniPBR.mdl"}
FOOT_BODY_NAMES = (
    "FAR_FOOT_LINK", "FBL_FOOT_LINK", "RAR_FOOT_LINK", "RBL_FOOT_LINK"
)
MOUNT_BODY_PAIRS = (
    ("BASE_LINK", "panda_link0"),
    ("panda_link8", "right_hand_base_link"),
)
```

Implement `_offline_report()` to reject unresolved non-built-in dependencies, any dependency outside project assets, more than one articulation root, missing/tampered manifests, inactive assembly joints, Panda finger joints, non-finite mass/inertia/limits/transforms, and non-convex O6 collision meshes.

Implement `_runtime_report()` with `SimulationCfg(device=device, dt=0.005)`. Initialize one articulation, hold default joint positions for exactly `steps`, and accumulate:

```python
return {
    "physics_steps": steps,
    "measured_physical_dof_count": int(robot.num_joints),
    "runtime_joint_names": list(robot.joint_names),
    "runtime_body_names": list(robot.body_names),
    "active_control_names": active_names,
    "active_control_count": len(active_names),
    "four_wheel_contact_ratio": all_four_contact_steps / steps,
    "max_mount_position_drift_m": max_position_drift,
    "max_mount_orientation_drift_rad": max_orientation_drift,
    "nonfinite_count": nonfinite_count,
    "hard_joint_limit_count": hard_joint_limit_count,
    "unexpected_contact_count": unexpected_contact_count,
    "unexpected_reset_count": unexpected_reset_count,
    "base_instability_count": base_instability_count,
    "contact_measurement_available": contact_measurement_available,
}
```

Use exact runtime joint mapping from the 29-channel contract. Count all four wheels as contacting only when each named foot body's net force exceeds `CONTACT_FORCE_THRESHOLD_N` for that step. Count contacts above the threshold on every non-foot body as unexpected.

The final gate must be explicit:

```python
hard_gates_passed = (
    not offline["offline_errors"]
    and runtime["measured_physical_dof_count"] == manifest["physical_dof_count"]
    and runtime["active_control_count"] == EXPECTED_ACTIVE_DOF_COUNT
    and runtime["physics_steps"] == EXPECTED_PHYSICS_STEPS
    and offline["articulation_roots"] == [EXPECTED_ARTICULATION_ROOT]
    and runtime["four_wheel_contact_ratio"] == 1.0
    and runtime["max_mount_position_drift_m"] <= MAX_MOUNT_POSITION_DRIFT_M
    and runtime["max_mount_orientation_drift_rad"] <= MAX_MOUNT_ORIENTATION_DRIFT_RAD
    and runtime["nonfinite_count"] == 0
    and runtime["hard_joint_limit_count"] == 0
    and runtime["unexpected_contact_count"] == 0
    and runtime["unexpected_reset_count"] == 0
    and runtime["base_instability_count"] == 0
    and runtime["contact_measurement_available"]
)
```

Atomically add `measured_physical_dof_count`, `verified_physics_steps`, `runtime_joint_names`, `runtime_body_names`, and the accepted verifier limits to `asset_manifest.json` only when `hard_gates_passed` is true. Exit `1` and leave the manifest unchanged on any failure.

- [ ] **Step 4: Run static tests and the real GPU0 physics gate**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_m1_single_panda_o6_asset_static.py \
  tests/test_m1_single_panda_o6_contracts.py
CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p \
  scripts/verify_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --steps 2000 \
  --device cuda:0 \
  --headless
```

Expected: tests pass; verifier exits `0`; JSON reports `active_control_count=29`, `measured_physical_dof_count=34` for the accepted source, `single_articulation_root=true`, `four_wheel_contact_ratio=1.0`, all failure counts `0`, and `hard_gates_passed=true`. If measured DOF differs from 34, stop and audit the manifest and runtime names instead of changing the gate to accept the difference.

- [ ] **Step 5: Commit the verifier and accepted runtime evidence**

Run:

```bash
git add Go2Pvcnn/scripts/verify_m1_single_panda_o6_asset.py \
  Go2Pvcnn/tests/test_m1_single_panda_o6_asset_static.py \
  Go2Pvcnn/assets/m1_single_panda_o6/asset_manifest.json
git commit -m "test: verify single-Panda O6 asset physics"
```

### Task 5: Run the T600.1 Final Gate and Align Documentation

**Files:**
- Create: `docs/superpowers/runbooks/2026-09-05-m1-single-panda-o6-asset-foundation.md`
- Create: `notes/log/2026-09-05-m1-single-panda-o6-asset-foundation.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md`

**Interfaces:**
- Consumes: Tasks 1-4 commits and the accepted asset manifest.
- Produces: a reproducible build/verify runbook and repository memory that marks only T600.1 complete.

- [ ] **Step 1: Run the complete local regression and source checks**

Run:

```bash
cd Go2Pvcnn
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_m1_dual_panda_o6_sources.py \
  tests/test_m1_single_panda_o6_asset_static.py \
  tests/test_m1_single_panda_o6_contracts.py \
  tests/test_m1_panda_asset_static.py \
  tests/test_m1_panda_wbc_contracts.py
python -m py_compile \
  scripts/normalize_o6_assets.py \
  scripts/build_m1_single_panda_o6_asset.py \
  scripts/verify_m1_single_panda_o6_asset.py \
  go2_pvcnn/assets/m1_single_panda_o6.py
cd ..
git diff --check
git status --short
```

Expected: all tests pass, compilation exits `0`, diff check exits `0`, and status contains only the intended T600 documentation changes at this point.

- [ ] **Step 2: Rebuild from the project-owned source and rerun the full physical gate**

Run:

```bash
cd Go2Pvcnn
CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p \
  scripts/build_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --o6-source-root assets/m1_dual_panda_o6 \
  --force-panda-conversion \
  --headless
CUDA_VISIBLE_DEVICES=0 /home/xk/coding/IsaacLab/isaaclab.sh -p \
  scripts/verify_m1_single_panda_o6_asset.py \
  --asset-root assets/m1_single_panda_o6 \
  --steps 2000 \
  --device cuda:0 \
  --headless
```

Expected: both commands exit `0`; the second JSON has `hard_gates_passed=true`; regenerated asset and manifest hashes are deterministic.

- [ ] **Step 3: Write the reproducible runbook**

Create `docs/superpowers/runbooks/2026-09-05-m1-single-panda-o6-asset-foundation.md` with:

```markdown
# M1 + Right Panda + Right O6 Asset Foundation Runbook

## Scope

This runbook builds and verifies T600.1 only. It does not register an environment or run MPC/RL.

## Inputs

- Read-only vendor O6 source: `/home/xk/coding/o6asset`
- Project O6 source manifest: `Go2Pvcnn/assets/m1_dual_panda_o6/source_manifest.json`
- Project M1/Panda source: `Go2Pvcnn/assets/m1_panda/`

## Build

Run the Task 5 Step 2 build command from the repository root.

## Verify

Run the Task 5 Step 2 verifier command. Accept only exit `0` plus `hard_gates_passed=true`.

## Failure Policy

Do not edit the vendor source, relax a hard gate, or rewrite the manifest after a failed run. Diagnose the named offline/runtime metric and rerun from a clean generated asset.
```

Replace the two prose references to Task 5 commands with the exact commands from Step 2 so the runbook is standalone.

- [ ] **Step 4: Record exact evidence and update the T600 tree**

Create `notes/log/2026-09-05-m1-single-panda-o6-asset-foundation.md` with the exact commit SHAs, commands, test count, physical DOF, active count, mount drift, contact ratio, all failure counters, source/asset SHA, device, and conclusion.

Update repository memory as follows:

- `notes/todo.md`: mark T600 active, T600.1 done, and T600.2 as the next open leaf.
- `notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md`: move T600.1 to the closed archive, record feature/verified SHAs, and point Next Step to a separate T600.2 design/plan cycle.
- `notes/log/index.md`: add the T600.1 acceptance log with exact metrics.
- Keep T600.2-T600.6 open and make no MPC, perception, or Residual acceptance claim.

- [ ] **Step 5: Verify documentation, LFS state, and final diff**

Run:

```bash
git diff --check
rg -n 'TBD|TODO|FIXME|PLACEHOLDER' \
  docs/superpowers/runbooks/2026-09-05-m1-single-panda-o6-asset-foundation.md \
  notes/log/2026-09-05-m1-single-panda-o6-asset-foundation.md \
  notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md
test $? -eq 1
git lfs fsck
git status --short
```

Expected: no whitespace errors, no placeholders, LFS fsck passes, and only intended Task 5 documentation files are uncommitted.

- [ ] **Step 6: Commit T600.1 acceptance documentation**

Run:

```bash
git add docs/superpowers/runbooks/2026-09-05-m1-single-panda-o6-asset-foundation.md \
  notes/log/2026-09-05-m1-single-panda-o6-asset-foundation.md \
  notes/log/index.md \
  notes/todo.md \
  notes/todo/T600-m1-single-panda-o6-multimodal-mpc-residual.md
git commit -m "docs: record T600 single-O6 asset acceptance"
```

- [ ] **Step 7: Verify the committed branch before any push**

Run:

```bash
git status --short
git log --oneline --decorate -6
git show --check --stat HEAD
git lfs fsck
```

Expected: the isolated worktree is clean; the five T600.1 task commits are present; the latest commit contains only runbook/notes alignment; Git/LFS checks pass.
