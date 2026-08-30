import pytest
import torch

from go2_pvcnn.control.m1_panda_coordination import (
    CONTROLLED_DOF,
    COORD_DOF,
    GENERALIZED_DOF,
    M1_LEG_JOINT_NAMES,
    M1_WHEEL_JOINT_NAMES,
    PANDA_ARM_JOINT_NAMES,
    PANDA_FINGER_JOINT_NAMES,
    PandaLinkDynamicsState,
    WbcJointMap,
    require_tensor,
)


EXPECTED_M1_LEGS = (
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
)
EXPECTED_M1_WHEELS = (
    "FAR_FOOT_JOINT",
    "FBL_FOOT_JOINT",
    "RAR_FOOT_JOINT",
    "RBL_FOOT_JOINT",
)
EXPECTED_PANDA_ARM = tuple(f"panda_joint{i}" for i in range(1, 8))
EXPECTED_FINGERS = ("panda_finger_joint1", "panda_finger_joint2")


def _link_dynamics_state():
    links = 2
    return PandaLinkDynamicsState(
        link_names=("panda_link0", "panda_link1"),
        mass=torch.ones(links, dtype=torch.float64),
        link_pos_w=torch.zeros((links, 3), dtype=torch.float64),
        link_quat_w=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * links, dtype=torch.float64
        ),
        com_pos_w=torch.zeros((links, 3), dtype=torch.float64),
        com_quat_w=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * links, dtype=torch.float64
        ),
        inertia_com_local=torch.eye(3, dtype=torch.float64).repeat(links, 1, 1),
        linear_vel_w=torch.zeros((links, 3), dtype=torch.float64),
        angular_vel_w=torch.zeros((links, 3), dtype=torch.float64),
        linear_acc_w=torch.zeros((links, 3), dtype=torch.float64),
        angular_acc_w=torch.zeros((links, 3), dtype=torch.float64),
    )


def test_panda_link_dynamics_contract_contains_pose_velocity_and_inertia():
    state = _link_dynamics_state()

    assert state.link_count == 2
    assert state.link_names[-1] == "panda_link1"
    assert state.linear_vel_w.shape == (2, 3)
    assert state.inertia_com_local.shape == (2, 3, 3)


def test_panda_link_dynamics_contract_rejects_wrong_link_velocity_shape():
    values = vars(_link_dynamics_state()).copy()
    values["linear_vel_w"] = torch.zeros((1, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match="linear_vel_w"):
        PandaLinkDynamicsState(**values)


def _actual_joint_names():
    return (
        "panda_joint4",
        "FBL_FOOT_JOINT",
        "RBL_HIP_JOINT",
        "panda_finger_joint2",
        "FAR_KNEE_JOINT",
        "panda_joint1",
        "RAR_FOOT_JOINT",
        "FBL_ABAD_JOINT",
        "panda_joint7",
        "RBL_KNEE_JOINT",
        "FAR_FOOT_JOINT",
        "panda_joint2",
        "RAR_ABAD_JOINT",
        "FBL_HIP_JOINT",
        "panda_finger_joint1",
        "FAR_ABAD_JOINT",
        "RBL_FOOT_JOINT",
        "panda_joint5",
        "RAR_HIP_JOINT",
        "FBL_KNEE_JOINT",
        "panda_joint3",
        "FAR_HIP_JOINT",
        "RBL_ABAD_JOINT",
        "panda_joint6",
        "RAR_KNEE_JOINT",
    )


def test_dimension_and_canonical_name_contracts_are_frozen():
    assert COORD_DOF == 10
    assert GENERALIZED_DOF == 31
    assert CONTROLLED_DOF == 23
    assert M1_LEG_JOINT_NAMES == EXPECTED_M1_LEGS
    assert M1_WHEEL_JOINT_NAMES == EXPECTED_M1_WHEELS
    assert PANDA_ARM_JOINT_NAMES == EXPECTED_PANDA_ARM
    assert PANDA_FINGER_JOINT_NAMES == EXPECTED_FINGERS


def test_joint_map_resolves_canonical_control_order_from_runtime_names():
    actual = _actual_joint_names()

    joint_map = WbcJointMap.resolve(actual)

    assert joint_map.controlled.dtype == torch.long
    assert joint_map.controlled.device.type == "cpu"
    assert joint_map.controlled.numel() == CONTROLLED_DOF
    assert joint_map.panda_arm.numel() == 7
    assert joint_map.fingers.numel() == 2
    assert tuple(actual[index] for index in joint_map.legs.tolist()) == EXPECTED_M1_LEGS
    assert tuple(actual[index] for index in joint_map.wheels.tolist()) == EXPECTED_M1_WHEELS
    assert tuple(actual[index] for index in joint_map.panda_arm.tolist()) == EXPECTED_PANDA_ARM
    assert tuple(actual[index] for index in joint_map.fingers.tolist()) == EXPECTED_FINGERS
    assert tuple(actual[index] for index in joint_map.controlled.tolist()) == (
        EXPECTED_M1_LEGS + EXPECTED_M1_WHEELS + EXPECTED_PANDA_ARM
    )


def test_joint_map_is_immutable():
    joint_map = WbcJointMap.resolve(_actual_joint_names())

    with pytest.raises(AttributeError):
        joint_map.controlled = torch.zeros(CONTROLLED_DOF, dtype=torch.long)


def test_joint_map_rejects_duplicate_runtime_names():
    actual = list(_actual_joint_names())
    actual[-1] = actual[0]

    with pytest.raises(ValueError, match="actual_joint_names contains duplicate name"):
        WbcJointMap.resolve(actual)


def test_joint_map_rejects_missing_required_name():
    actual = [name for name in _actual_joint_names() if name != "panda_joint6"]

    with pytest.raises(ValueError, match="missing required joints: panda_joint6"):
        WbcJointMap.resolve(actual)


def test_joint_map_rejects_string_instead_of_name_sequence():
    with pytest.raises(TypeError, match="actual_joint_names must be a sequence of strings"):
        WbcJointMap.resolve("panda_joint1")


def test_require_tensor_accepts_leading_batch_dimensions_and_returns_identity():
    value = torch.zeros(2, 3, 23, dtype=torch.float64)

    result = require_tensor(
        "effort",
        value,
        trailing_shape=(23,),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert result is value


def test_require_tensor_accepts_scalar_trailing_shape():
    value = torch.tensor(1.0)

    assert require_tensor("gain", value, trailing_shape=()) is value


def test_require_tensor_rejects_non_tensor():
    with pytest.raises(TypeError, match="effort must be a torch.Tensor"):
        require_tensor("effort", [0.0] * 23, trailing_shape=(23,))


@pytest.mark.parametrize(
    ("value", "trailing_shape", "expected"),
    [
        (torch.zeros(22), (23,), r"effort must end with shape \(23,\)"),
        (torch.zeros(2, 6, 9), (6, 10), r"jacobian must end with shape \(6, 10\)"),
        (torch.tensor(1.0), (1,), r"scalar must end with shape \(1,\)"),
    ],
)
def test_require_tensor_rejects_wrong_trailing_shape(value, trailing_shape, expected):
    with pytest.raises(ValueError, match=expected):
        require_tensor(
            "scalar" if value.ndim == 0 else "jacobian" if len(trailing_shape) == 2 else "effort",
            value,
            trailing_shape=trailing_shape,
        )


def test_require_tensor_rejects_wrong_dtype():
    with pytest.raises(TypeError, match="state must have dtype torch.float64"):
        require_tensor(
            "state",
            torch.zeros(3, dtype=torch.float32),
            trailing_shape=(3,),
            dtype=torch.float64,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_require_tensor_rejects_wrong_device():
    with pytest.raises(ValueError, match="state must be on device cuda:0"):
        require_tensor(
            "state",
            torch.zeros(3),
            trailing_shape=(3,),
            device=torch.device("cuda:0"),
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_require_tensor_rejects_non_finite_values(bad):
    with pytest.raises(ValueError, match="state must contain only finite values"):
        require_tensor("state", torch.tensor([0.0, bad]), trailing_shape=(2,))


def test_require_tensor_rejects_invalid_trailing_shape_contract():
    with pytest.raises(TypeError, match="trailing_shape must be a tuple of non-negative integers"):
        require_tensor("state", torch.zeros(2), trailing_shape=[2])
