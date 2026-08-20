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
from .student_mission import StudentMissionSample, StudentS1Mission
from .student_model import (
    M1PandaStudent,
    StudentHistoryBuffer,
    StudentNetworkCfg,
    StudentOutput,
)
from .dagger import (
    DaggerSelection,
    DaggerStageCfg,
    StudentLossCfg,
    select_dagger_action,
    student_dagger_loss,
)
from .coordinated_mission import (
    CoordinatedMission,
    CoordinatedMissionCfg,
    CoordinatedMissionState,
    MissionPhase,
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
    "StudentMissionSample",
    "StudentS1Mission",
    "M1PandaStudent",
    "StudentHistoryBuffer",
    "StudentNetworkCfg",
    "StudentOutput",
    "DaggerSelection",
    "DaggerStageCfg",
    "StudentLossCfg",
    "select_dagger_action",
    "student_dagger_loss",
    "CoordinatedMission",
    "CoordinatedMissionCfg",
    "CoordinatedMissionState",
    "MissionPhase",
]
