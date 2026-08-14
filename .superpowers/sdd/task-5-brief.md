### Task 5: Implement the base-frame mount-wrench observation

**Files:**
- Create: `Go2Pvcnn/go2_pvcnn/mdp/m1_panda_wrench.py`
- Modify: `Go2Pvcnn/go2_pvcnn/mdp/__init__.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/m1_panda_smoke_env_cfg.py`
- Modify: `Go2Pvcnn/tests/test_m1_panda_smoke_cfg_static.py`
- Create: `Go2Pvcnn/tests/test_m1_panda_wrench.py`

**Interfaces:**
- Consumes: world-frame incoming wrench on `panda_link0`, world positions of `panda_link0` and `BASE_LINK`, and `BASE_LINK` world quaternion.
- Produces: `shift_rotate_wrench_to_base(force_w, torque_w, sensor_pos_w, base_pos_w, base_quat_w) -> torch.Tensor[..., 6]` and `m1_panda_mount_wrench_b(env, asset_cfg, mount_body_name, base_body_name) -> torch.Tensor[num_envs, 6]`.

- [ ] **Step 1: Write tensor-level failing tests**

```python
import torch

from go2_pvcnn.mdp.m1_panda_wrench import shift_rotate_wrench_to_base


def test_identity_frame_keeps_force_and_shifts_moment_to_base_origin():
    result = shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 10.0, 0.0]]),
        torque_w=torch.tensor([[0.0, 0.0, 2.0]]),
        sensor_pos_w=torch.tensor([[1.0, 0.0, 0.0]]),
        base_pos_w=torch.zeros(1, 3),
        base_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(result, torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 12.0]]))


def test_base_yaw_rotates_world_force_into_base_frame():
    half = 2.0 ** -0.5
    result = shift_rotate_wrench_to_base(
        force_w=torch.tensor([[0.0, 1.0, 0.0]]),
        torque_w=torch.zeros(1, 3),
        sensor_pos_w=torch.zeros(1, 3),
        base_pos_w=torch.zeros(1, 3),
        base_quat_w=torch.tensor([[half, 0.0, 0.0, half]]),
    )
    assert torch.allclose(result[:, :3], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
```

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_m1_panda_wrench.py`

Expected: FAIL with `ModuleNotFoundError` for `m1_panda_wrench`.

- [ ] **Step 3: Implement the pure transform**

```python
from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils


def shift_rotate_wrench_to_base(
    force_w: torch.Tensor,
    torque_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    moment_about_base_w = torque_w + torch.linalg.cross(sensor_pos_w - base_pos_w, force_w, dim=-1)
    force_b = math_utils.quat_rotate_inverse(base_quat_w, force_w)
    moment_b = math_utils.quat_rotate_inverse(base_quat_w, moment_about_base_w)
    return torch.cat((force_b, moment_b), dim=-1)
```

- [ ] **Step 4: Implement the environment adapter**

```python
def m1_panda_mount_wrench_b(
    env,
    asset_cfg,
    mount_body_name: str = "panda_link0",
    base_body_name: str = "BASE_LINK",
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    mount_ids, mount_names = robot.find_bodies(mount_body_name, preserve_order=True)
    base_ids, base_names = robot.find_bodies(base_body_name, preserve_order=True)
    if mount_names != [mount_body_name] or base_names != [base_body_name]:
        raise RuntimeError(f"Expected one mount/base body, got {mount_names=} {base_names=}")
    incoming = robot.root_physx_view.get_link_incoming_joint_force()[:, mount_ids[0], :]
    return shift_rotate_wrench_to_base(
        incoming[:, :3],
        incoming[:, 3:],
        robot.data.body_pos_w[:, mount_ids[0]],
        robot.data.body_pos_w[:, base_ids[0]],
        robot.data.body_quat_w[:, base_ids[0]],
    )
```

Export it from `go2_pvcnn/mdp/__init__.py`, and wire the smoke term as:

```python
mount_wrench_b = ObsTerm(
    func=mdp.m1_panda_mount_wrench_b,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=M1_PANDA_MOUNT_BODY_NAME),
        "mount_body_name": M1_PANDA_MOUNT_BODY_NAME,
        "base_body_name": M1_PANDA_BASE_BODY_NAME,
    },
)
```

Add the following static assertion to `test_m1_panda_smoke_cfg_static.py` in this task:

```python
assert 'body_names=M1_PANDA_MOUNT_BODY_NAME' in env
```

- [ ] **Step 5: Run GREEN and related regression**

Run:

```bash
cd /home/xk/coding/M1/Go2Pvcnn
pytest -q \
  tests/test_m1_panda_wrench.py \
  tests/test_m1_panda_smoke_cfg_static.py \
  tests/test_m1_smoke_cfg_static.py \
  tests/test_m1_asset_static.py
```

Expected: all tests pass. Record the test count and duration; Git commit is unavailable.

---

