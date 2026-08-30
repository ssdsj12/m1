"""Mount-wrench observation helpers for the combined M1 and Panda articulation."""

from __future__ import annotations

import torch
from isaaclab.utils import math as math_utils

__all__ = ["shift_rotate_wrench_to_base", "m1_panda_mount_wrench_b"]


def shift_rotate_wrench_to_base(
    force_w: torch.Tensor,
    torque_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Express a world-frame sensor wrench at the base-link origin in base coordinates.

    The returned channel order is ``[Fx, Fy, Fz, Mx, My, Mz]``. The moment shift uses
    ``M_base = M_sensor + (p_sensor - p_base) x F`` before both vectors are rotated
    from world coordinates into the ``BASE_LINK`` actor frame.
    """
    moment_about_base_w = torque_w + torch.linalg.cross(sensor_pos_w - base_pos_w, force_w, dim=-1)
    force_b = math_utils.quat_rotate_inverse(base_quat_w, force_w)
    moment_b = math_utils.quat_rotate_inverse(base_quat_w, moment_about_base_w)
    return torch.cat((force_b, moment_b), dim=-1)


def m1_panda_mount_wrench_b(
    env,
    asset_cfg,
    mount_body_name: str = "panda_link0",
    base_body_name: str = "BASE_LINK",
) -> torch.Tensor:
    """Return the Panda-on-M1 reaction wrench about the base origin.

    PhysX reports the parent-on-child incoming wrench in the child joint frame about
    the joint origin.  The controller contract uses the equal-and-opposite
    child-on-parent reaction, so both raw force and torque are negated before the
    frame/origin conversion.  The combined asset contract makes the child joint frame
    coincide with the ``panda_link0`` actor frame.
    """
    robot = env.scene[asset_cfg.name]
    mount_ids, mount_names = robot.find_bodies(mount_body_name, preserve_order=True)
    base_ids, base_names = robot.find_bodies(base_body_name, preserve_order=True)
    if (
        len(mount_ids) != 1
        or len(base_ids) != 1
        or mount_names != [mount_body_name]
        or base_names != [base_body_name]
    ):
        raise RuntimeError(f"Expected one mount/base body, got {mount_names=} {base_names=}")

    mount_id = mount_ids[0]
    base_id = base_ids[0]
    incoming = robot.root_physx_view.get_link_incoming_joint_force()[:, mount_id, :]
    mount_quat_w = robot.data.body_quat_w[:, mount_id]
    force_w = math_utils.quat_rotate(mount_quat_w, -incoming[:, :3])
    torque_w = math_utils.quat_rotate(mount_quat_w, -incoming[:, 3:6])
    return shift_rotate_wrench_to_base(
        force_w,
        torque_w,
        robot.data.body_pos_w[:, mount_id],
        robot.data.body_pos_w[:, base_id],
        robot.data.body_quat_w[:, base_id],
    )
