"""Deterministic M1 + Panda coordination and whole-body control."""

from .contracts import (
    CONTROLLED_DOF,
    COORD_DOF,
    GENERALIZED_DOF,
    M1_LEG_JOINT_NAMES,
    M1_WHEEL_JOINT_NAMES,
    PANDA_ARM_JOINT_NAMES,
    PANDA_FINGER_JOINT_NAMES,
    WbcJointMap,
    require_tensor,
)
from .student_contracts import (
    STUDENT_ACTION_DIM,
    STUDENT_HISTORY_LENGTH,
    STUDENT_OBSERVATION_DIM,
    StudentActionCommand,
    StudentActionScaleCfg,
    StudentNominalCommand,
    apply_student_residual,
    teacher_residual_label,
)

__all__ = [
    "CONTROLLED_DOF",
    "COORD_DOF",
    "GENERALIZED_DOF",
    "M1_LEG_JOINT_NAMES",
    "M1_WHEEL_JOINT_NAMES",
    "PANDA_ARM_JOINT_NAMES",
    "PANDA_FINGER_JOINT_NAMES",
    "WbcJointMap",
    "require_tensor",
    "STUDENT_ACTION_DIM",
    "STUDENT_HISTORY_LENGTH",
    "STUDENT_OBSERVATION_DIM",
    "StudentActionCommand",
    "StudentActionScaleCfg",
    "StudentNominalCommand",
    "apply_student_residual",
    "teacher_residual_label",
]
