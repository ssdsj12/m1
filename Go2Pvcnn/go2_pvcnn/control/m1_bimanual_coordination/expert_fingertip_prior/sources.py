"""Pinned ManipTrans source-hand manifests for offline conversion only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class SourceHandSpec:
    name: str
    side: str
    urdf_relpath: str
    joint_order: tuple[str, ...]
    palm_link: str
    fingertip_links: tuple[str, str, str, str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.side not in {"rh", "lh"}:
            raise ValueError("side must be 'rh' or 'lh'")
        if not isinstance(self.urdf_relpath, str):
            raise ValueError("urdf_relpath must be a relative URDF path")
        path = PurePosixPath(self.urdf_relpath)
        if (
            not self.urdf_relpath
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".urdf"
        ):
            raise ValueError("urdf_relpath must be a relative URDF path")
        if not self.joint_order or any(not isinstance(name, str) or not name for name in self.joint_order):
            raise ValueError("joint_order must contain non-empty names")
        if len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("joint_order must contain unique names")
        if not isinstance(self.palm_link, str) or not self.palm_link:
            raise ValueError("palm_link must be a non-empty string")
        if len(self.fingertip_links) != 5 or any(
            not isinstance(name, str) or not name for name in self.fingertip_links
        ):
            raise ValueError("fingertip_links must contain five non-empty names")
        if len(set(self.fingertip_links)) != 5:
            raise ValueError("fingertip_links must contain five unique names")


_INSPIRE_JOINT_SUFFIXES = (
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
)
_SHADOW_JOINT_ORDER = (
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
)

# Copied from ManipTrans commit a3d08cfe3c3a5868a7f057533bcaf759c5af4705:
# maniptrans_envs/lib/envs/dexhands/{inspire,shadow}.py and their referenced URDFs.
SOURCE_HANDS = {
    "inspire_rh": SourceHandSpec(
        name="inspire",
        side="rh",
        urdf_relpath="maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf",
        joint_order=tuple(f"R_{name}" for name in _INSPIRE_JOINT_SUFFIXES),
        palm_link="R_hand_base_link",
        fingertip_links=("R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"),
    ),
    "inspire_lh": SourceHandSpec(
        name="inspire",
        side="lh",
        urdf_relpath="maniptrans_envs/assets/inspire_hand/inspire_hand_left.urdf",
        joint_order=tuple(f"L_{name}" for name in _INSPIRE_JOINT_SUFFIXES),
        palm_link="L_hand_base_link",
        fingertip_links=("L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"),
    ),
    "shadow_rh": SourceHandSpec(
        name="shadow",
        side="rh",
        urdf_relpath="maniptrans_envs/assets/shadow_hand/shadow_hand_right_woarm.urdf",
        joint_order=_SHADOW_JOINT_ORDER,
        palm_link="palm",
        fingertip_links=("thtip", "fftip", "mftip", "rftip", "lftip"),
    ),
    "shadow_lh": SourceHandSpec(
        name="shadow",
        side="lh",
        urdf_relpath="maniptrans_envs/assets/shadow_hand/shadow_hand_left_woarm.urdf",
        joint_order=_SHADOW_JOINT_ORDER,
        palm_link="palm",
        fingertip_links=("thtip", "fftip", "mftip", "rftip", "lftip"),
    ),
}


__all__ = ["SOURCE_HANDS", "SourceHandSpec"]
