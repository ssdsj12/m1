### Task 1: Establish the project-owned offline asset inputs

**Files:**
- Create: `Go2Pvcnn/assets/m1_panda/source_manifest.json`
- Create: `Go2Pvcnn/assets/m1_panda/m1_floating.usda`
- Create: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`
- Copy: `/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0/` to `Go2Pvcnn/assets/m1_panda/m1/ZJ_V3_URDF_V1_0/`
- Copy: `/home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description/` to `Go2Pvcnn/assets/m1_panda/panda_source/franka_description/`

**Interfaces:**
- Consumes: the current M1 physics USD and Isaac Sim-bundled `panda_arm_hand.urdf` package.
- Produces: `M1_LOCAL_OVERLAY`, `PANDA_LOCAL_URDF`, and a project-contained source tree used by the builder.

- [ ] **Step 1: Write the failing static asset test**

```python
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "m1_panda"


def test_m1_panda_sources_are_project_owned_and_manifested():
    manifest = json.loads((ASSET_ROOT / "source_manifest.json").read_text())
    assert manifest["m1"]["entry"] == "m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd"
    assert manifest["panda"]["entry"] == "panda_source/franka_description/robots/panda_arm_hand.urdf"
    assert (ASSET_ROOT / manifest["m1"]["entry"]).is_file()
    assert (ASSET_ROOT / manifest["panda"]["entry"]).is_file()


def test_m1_overlay_has_only_a_relative_local_sublayer():
    text = (ASSET_ROOT / "m1_floating.usda").read_text()
    assert "@./m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd@" in text
    assert "/home/" not in text
    assert "omniverse://" not in text
    assert "http://" not in text
    assert "https://" not in text
```

- [ ] **Step 2: Run the test and confirm that the asset root is missing**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_asset_static.py
```

Expected: FAIL because `assets/m1_panda/source_manifest.json` does not exist.

- [ ] **Step 3: Copy the two authoritative source trees**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
mkdir -p assets/m1_panda/m1 assets/m1_panda/panda_source
cp -a /home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0 assets/m1_panda/m1/
cp -a /home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description assets/m1_panda/panda_source/
```

Expected: both destination entry files exist and `du -sh assets/m1_panda` reports a non-zero local asset tree.

- [ ] **Step 4: Add the manifest and local floating overlay**

Create `source_manifest.json` with exactly:

```json
{
  "m1": {
    "entry": "m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd",
    "source": "/home/xk/ros2_ws/src/zjs_m1_v3_description/urdf/ZJ_V3_URDF_V1_0"
  },
  "panda": {
    "entry": "panda_source/franka_description/robots/panda_arm_hand.urdf",
    "source": "/home/xk/.local/share/ov/data/exts/v2/isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/data/urdf/robots/franka_description"
  }
}
```

Create `m1_floating.usda` by copying the existing floating overlay and changing only its sublayer to:

```usda
subLayers = [
    @./m1/ZJ_V3_URDF_V1_0/configuration/ZJ_V3_URDF_V1_0_physics.usd@
]
```

- [ ] **Step 5: Run the static test and record a checksum checkpoint**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q tests/test_m1_panda_asset_static.py
find assets/m1_panda -type f -print0 | sort -z | xargs -0 sha256sum > assets/m1_panda/source_files.sha256
```

Expected: `2 passed`; `source_files.sha256` is non-empty. Record the checksum file in the T400 verification log. Git commit is unavailable in this directory.

---

