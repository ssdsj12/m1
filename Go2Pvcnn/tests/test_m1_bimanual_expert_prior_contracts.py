from dataclasses import replace

import pytest
import torch

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.contracts import (
    DEXMANIPNET_REVISION,
    FINGER_ORDER,
    LEFT_REFLECTION,
    MANIPTRANS_COMMIT,
    MIXTURE_OUTPUT_AXIS_ORDER,
    MIXTURE_COMPONENTS,
    MODEL_INPUT_FIELD_ORDER,
    PRIOR_HORIZON,
    PHASE_ORDER,
    ExpertWindow,
    MixtureDistribution,
    PriorPhase,
    StudentArtifactMetadata,
)
from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.sources import (
    SOURCE_HANDS,
    SourceHandSpec,
)


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


def test_distribution_and_window_reject_invalid_frozen_shapes():
    window = ExpertWindow(
        fingertip_position_palm=torch.zeros(5, 3),
        fingertip_velocity_palm=torch.zeros(5, 3),
        contact_mask=torch.zeros(5, dtype=torch.bool),
        phase=PriorPhase.HOLD,
        future_fingertip_velocity_palm=torch.zeros(PRIOR_HORIZON, 5, 3),
        source_group="oakink/sequence/rh",
        source_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="shape"):
        replace(window, fingertip_position_palm=torch.zeros(4, 3))
    with pytest.raises(TypeError, match="PriorPhase"):
        replace(window, phase=PriorPhase.HOLD.value)
    with pytest.raises(TypeError, match="dtype"):
        replace(window, fingertip_velocity_palm=torch.zeros(5, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="SHA"):
        replace(window, source_sha256="A" * 64)

    with pytest.raises(ValueError, match="shape"):
        MixtureDistribution(
            logits=torch.zeros(MIXTURE_COMPONENTS),
            mean=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3),
            log_std=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 2),
        )
    with pytest.raises(TypeError, match="floating"):
        MixtureDistribution(
            logits=torch.zeros(MIXTURE_COMPONENTS, dtype=torch.int64),
            mean=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3),
            log_std=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3),
        )
    with pytest.raises(ValueError, match="finite"):
        MixtureDistribution(
            logits=torch.zeros(MIXTURE_COMPONENTS),
            mean=torch.full((MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3), float("nan")),
            log_std=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3),
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float64))
def test_distribution_requires_exact_float32_tensors(dtype):
    with pytest.raises(TypeError, match="torch.float32"):
        MixtureDistribution(
            logits=torch.zeros(MIXTURE_COMPONENTS, dtype=dtype),
            mean=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3, dtype=dtype),
            log_std=torch.zeros(MIXTURE_COMPONENTS, PRIOR_HORIZON, 5, 3, dtype=dtype),
        )


def _metadata(**overrides) -> StudentArtifactMetadata:
    values = {
        "format_version": 1,
        "input_dim": 42,
        "mixture_components": 4,
        "horizon": 20,
        "dt": 0.01,
        "finger_order": FINGER_ORDER,
        "phase_order": PHASE_ORDER,
        "input_field_order": MODEL_INPUT_FIELD_ORDER,
        "output_axis_order": MIXTURE_OUTPUT_AXIS_ORDER,
        "mirror_matrix": torch.tensor(LEFT_REFLECTION, dtype=torch.float32),
        "dataset_aggregate_sha256": "a" * 64,
        "teacher_ensemble_manifest_sha256": "b" * 64,
        "teacher_seed": 42,
        "distillation_seed": 43,
        "code_commit": "c" * 40,
        "weight_sha256": "d" * 64,
        "hidden": (128, 128),
    }
    values.update(overrides)
    return StudentArtifactMetadata(**values)


def test_student_metadata_freezes_exact_input_and_output_layouts():
    metadata = _metadata()
    assert metadata.input_field_order == (
        "fingertip_position_palm",
        "fingertip_velocity_palm",
        "contact_mask",
        "phase_one_hot",
    )
    assert metadata.output_axis_order == ("mixture_component", "horizon", "finger", "xyz")

    with pytest.raises(ValueError, match="input_field_order"):
        _metadata(input_field_order=tuple(reversed(MODEL_INPUT_FIELD_ORDER)))
    with pytest.raises(ValueError, match="output_axis_order"):
        _metadata(output_axis_order=("horizon", "mixture_component", "finger", "xyz"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", True),
        ("input_dim", 42.0),
        ("mixture_components", False),
        ("horizon", 20.0),
        ("teacher_seed", True),
        ("distillation_seed", 43.0),
        ("hidden", [128, 128]),
        ("hidden", (128, True)),
    ],
)
def test_student_metadata_rejects_non_strict_integer_schema_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        _metadata(**{field: value})


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("dataset_aggregate_sha256", "A" * 64, ValueError),
        ("weight_sha256", "d" * 63, ValueError),
        ("mirror_matrix", torch.zeros(3, 3, dtype=torch.float64), TypeError),
        ("mirror_matrix", torch.full((3, 3), float("nan"), dtype=torch.float32), ValueError),
    ],
)
def test_student_metadata_rejects_invalid_sha_and_mirror_contract(field, value, error):
    with pytest.raises(error):
        _metadata(**{field: value})


def test_source_spec_requires_tuple_names_and_registry_is_exact_and_immutable():
    spec = SOURCE_HANDS["inspire_rh"]
    with pytest.raises(TypeError, match="joint_order"):
        SourceHandSpec(
            name="inspire",
            side="rh",
            urdf_relpath=spec.urdf_relpath,
            joint_order=list(spec.joint_order),
            palm_link=spec.palm_link,
            fingertip_links=spec.fingertip_links,
        )
    with pytest.raises(TypeError, match="joint_order"):
        SourceHandSpec(
            name="inspire",
            side="rh",
            urdf_relpath=spec.urdf_relpath,
            joint_order="R_index_proximal_joint",
            palm_link=spec.palm_link,
            fingertip_links=spec.fingertip_links,
        )
    with pytest.raises(TypeError, match="fingertip_links"):
        SourceHandSpec(
            name="inspire",
            side="rh",
            urdf_relpath=spec.urdf_relpath,
            joint_order=spec.joint_order,
            palm_link=spec.palm_link,
            fingertip_links=list(spec.fingertip_links),
        )
    with pytest.raises(TypeError, match="fingertip_links"):
        SourceHandSpec(
            name="inspire",
            side="rh",
            urdf_relpath=spec.urdf_relpath,
            joint_order=spec.joint_order,
            palm_link=spec.palm_link,
            fingertip_links="R_thumb_tip",
        )
    with pytest.raises(ValueError, match="name"):
        SourceHandSpec(
            name=" ",
            side="rh",
            urdf_relpath=spec.urdf_relpath,
            joint_order=spec.joint_order,
            palm_link=spec.palm_link,
            fingertip_links=spec.fingertip_links,
        )
    for field, invalid in (("joint_order", ("valid", " ")), ("fingertip_links", ("tip", "", "a", "b", "c"))):
        values = {
            "name": "inspire",
            "side": "rh",
            "urdf_relpath": spec.urdf_relpath,
            "joint_order": spec.joint_order,
            "palm_link": spec.palm_link,
            "fingertip_links": spec.fingertip_links,
        }
        values[field] = invalid
        with pytest.raises(ValueError, match=field):
            SourceHandSpec(**values)
    for field, invalid in (("joint_order", ("valid", 1)), ("fingertip_links", ("tip", "a", "b", "c", 1))):
        values = {
            "name": "inspire",
            "side": "rh",
            "urdf_relpath": spec.urdf_relpath,
            "joint_order": spec.joint_order,
            "palm_link": spec.palm_link,
            "fingertip_links": spec.fingertip_links,
        }
        values[field] = invalid
        with pytest.raises(ValueError, match=field):
            SourceHandSpec(**values)
    for field, invalid in (("joint_order", ("duplicate", "duplicate")), ("fingertip_links", ("tip", "tip", "a", "b", "c"))):
        values = {
            "name": "inspire",
            "side": "rh",
            "urdf_relpath": spec.urdf_relpath,
            "joint_order": spec.joint_order,
            "palm_link": spec.palm_link,
            "fingertip_links": spec.fingertip_links,
        }
        values[field] = invalid
        with pytest.raises(ValueError, match="unique"):
            SourceHandSpec(**values)

    inspire_joint_suffixes = (
        "index_proximal_joint", "index_intermediate_joint",
        "middle_proximal_joint", "middle_intermediate_joint",
        "pinky_proximal_joint", "pinky_intermediate_joint",
        "ring_proximal_joint", "ring_intermediate_joint",
        "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
        "thumb_intermediate_joint", "thumb_distal_joint",
    )
    inspire_mimic_override_suffixes = (
        "index_intermediate_joint", "middle_intermediate_joint",
        "pinky_intermediate_joint", "ring_intermediate_joint",
        "thumb_intermediate_joint", "thumb_distal_joint",
    )
    shadow_joint_order = (
        "FFJ4", "FFJ3", "FFJ2", "FFJ1", "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
    )
    expected = {
        "inspire_rh": SourceHandSpec(
            name="inspire",
            side="rh",
            urdf_relpath="maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf",
            joint_order=tuple(f"R_{name}" for name in inspire_joint_suffixes),
            palm_link="R_hand_base_link",
                fingertip_links=(
                "R_thumb_tip",
                "R_index_tip",
                "R_middle_tip",
                "R_ring_tip",
                    "R_pinky_tip",
                ),
                mimic_override_joints=tuple(
                    f"R_{name}" for name in inspire_mimic_override_suffixes
                ),
        ),
        "inspire_lh": SourceHandSpec(
            name="inspire",
            side="lh",
            urdf_relpath="maniptrans_envs/assets/inspire_hand/inspire_hand_left.urdf",
            joint_order=tuple(f"L_{name}" for name in inspire_joint_suffixes),
            palm_link="L_hand_base_link",
                fingertip_links=("L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"),
                mimic_override_joints=tuple(
                    f"L_{name}" for name in inspire_mimic_override_suffixes
                ),
        ),
        "shadow_rh": SourceHandSpec(
            name="shadow",
            side="rh",
            urdf_relpath="maniptrans_envs/assets/shadow_hand/shadow_hand_right_woarm.urdf",
            joint_order=shadow_joint_order,
            palm_link="palm",
            fingertip_links=("thtip", "fftip", "mftip", "rftip", "lftip"),
        ),
        "shadow_lh": SourceHandSpec(
            name="shadow",
            side="lh",
            urdf_relpath="maniptrans_envs/assets/shadow_hand/shadow_hand_left_woarm.urdf",
            joint_order=shadow_joint_order,
            palm_link="palm",
            fingertip_links=("thtip", "fftip", "mftip", "rftip", "lftip"),
        ),
    }
    assert tuple(SOURCE_HANDS) == tuple(expected)
    assert dict(SOURCE_HANDS) == expected
    assert LEFT_REFLECTION == ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    assert type(LEFT_REFLECTION) is tuple
    assert all(type(row) is tuple for row in LEFT_REFLECTION)
    try:
        with pytest.raises(TypeError):
            SOURCE_HANDS["unexpected"] = spec
    finally:
        if type(SOURCE_HANDS) is dict:
            SOURCE_HANDS.pop("unexpected", None)
