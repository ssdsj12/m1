### Task 3: Verify offline dependency closure and articulation topology

**Files:**
- Create: `Go2Pvcnn/scripts/verify_m1_panda_asset.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_asset_static.py`

**Interfaces:**
- Consumes: `m1_panda.usd`.
- Produces: JSON-compatible verification output with keys `root`, `dependencies`, `joint_names`, `body_names`, `dof_count`, and `remote_dependencies`.

- [ ] **Step 1: Add a failing static verifier-contract test**

```python
def test_verifier_checks_offline_and_topology_contracts():
    source = (ROOT / "scripts" / "verify_m1_panda_asset.py").read_text()
    for token in (
        "ComputeAllDependencies",
        "remote_dependencies",
        'EXPECTED_DOF_COUNT = 25',
        '"BASE_LINK"',
        '"panda_link0"',
        '"panda_hand"',
        '"/M1Panda/Panda/panda_link0/AssemblerFixedJoint"',
    ):
        assert token in source
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_asset_static.py::test_verifier_checks_offline_and_topology_contracts`

Expected: FAIL because the verifier file is absent.

- [ ] **Step 3: Implement dependency and runtime topology verification**

Implement a headless Isaac Lab script that:

```python
EXPECTED_DOF_COUNT = 25
REMOTE_PREFIXES = ("omniverse://", "http://", "https://")

stage = Usd.Stage.Open(str(asset_path))
layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset_path))
dependencies = [layer.identifier for layer in layers] + [str(asset) for asset in assets]
remote_dependencies = [item for item in dependencies if item.startswith(REMOTE_PREFIXES)]
assert not unresolved
assert not remote_dependencies
assert all(Path(item).resolve().is_relative_to(asset_root.resolve()) for item in dependencies if not item.startswith("anon:"))
```

Then instantiate one Isaac Lab `Articulation`, reset it, and assert:

```python
assert robot.num_joints == EXPECTED_DOF_COUNT
assert robot.find_bodies("BASE_LINK")[1] == ["BASE_LINK"]
assert robot.find_bodies("panda_link0")[1] == ["panda_link0"]
assert robot.find_bodies("panda_hand")[1] == ["panda_hand"]
mount_joint = stage.GetPrimAtPath("/M1Panda/Panda/panda_link0/AssemblerFixedJoint")
assert mount_joint.IsValid()
assert mount_joint.IsA(UsdPhysics.FixedJoint)
```

Print one JSON object and exit non-zero on every failed assertion.

- [ ] **Step 4: Run static and real verification**

Run:

```bash
cd /home/xk/coding/M1
pytest -q Go2Pvcnn/tests/test_m1_panda_asset_static.py
/home/xk/coding/IsaacLab/isaaclab.sh -p Go2Pvcnn/scripts/verify_m1_panda_asset.py \
  --asset /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda/m1_panda.usd \
  --asset-root /home/xk/coding/M1/Go2Pvcnn/assets/m1_panda \
  --headless
```

Expected: pytest passes; JSON reports `dof_count: 25`, an empty `remote_dependencies`, and all required bodies/joint exactly once.

- [ ] **Step 5: Record the topology checkpoint**

Append the verifier JSON and command exit code to `notes/log/2026-08-14-m1-panda-force-aware-teacher-student-design.md`. Git commit is unavailable.

---

