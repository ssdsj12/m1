"""Deterministic 200 Hz scene for M1 + dual Panda + dual O6 MPC."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from go2_pvcnn.assets.m1_dual_panda_o6 import (
    M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES,
    M1_DUAL_PANDA_O6_CFG,
)
from go2_pvcnn.control.m1_bimanual_coordination.object_catalog import (
    ObjectCatalog,
    ObjectInstance,
)
from go2_pvcnn.tasks.m1_object_scene import (
    build_object_scene_cfg,
    validate_scene_object_ids,
)
from go2_pvcnn.tasks.m1_smoke_env_cfg import M1SmokeEnvCfg, M1SmokeSceneCfg


BOX_SIZE_M = (0.20, 0.18, 0.10)
SUPPORT_SIZE_M = (0.08, 0.12, 0.10)
BOX_MASS_KG = 0.5
PHYSICS_DT = 0.005
PRIVATE_ACTION_DIM = 43

_BIMANUAL_ROBOT_CFG = M1_DUAL_PANDA_O6_CFG.copy()
_BIMANUAL_ROBOT_CFG.spawn = _BIMANUAL_ROBOT_CFG.spawn.replace(
    activate_contact_sensors=True
)
# Stage one intentionally validates manipulation with a physically fixed M1
# chassis.  The mobile/sliding stage can switch this off without changing the
# 43-channel controller contract.
_BIMANUAL_ROBOT_CFG.spawn.articulation_props.fix_root_link = True
# JointEffortAction supplies the complete impedance/WBC effort.  Disable the
# actuator-side gains so the same feedback is not applied a second time.
_BIMANUAL_ROBOT_CFG.actuators = {
    name: actuator.replace(stiffness=0.0, damping=0.0)
    for name, actuator in _BIMANUAL_ROBOT_CFG.actuators.items()
}
for name in ("left_o6", "right_o6"):
    _BIMANUAL_ROBOT_CFG.actuators[name] = (
        _BIMANUAL_ROBOT_CFG.actuators[name].replace(
            stiffness=80.0,
            damping=5.0,
        )
    )


def _box_contact_sensor(body_name: str) -> ContactSensorCfg:
    """Create an Isaac Lab-supported one-body-to-Box filtered sensor."""

    side = body_name.split("_", 1)[0]
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_arm/{side}_o6/{body_name}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
        history_length=3,
        track_air_time=False,
    )


@configclass
class M1DualPandaO6BimanualSceneCfg(M1SmokeSceneCfg):
    """Combined robot, fixed support, dynamic box, and explicit contact groups."""

    robot = _BIMANUAL_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    support_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.65, 0.0, 1.10)),
        spawn=sim_utils.CuboidCfg(
            size=SUPPORT_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32)),
        ),
    )
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 1.20)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE_M,
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS_KG),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.42, 0.16)),
            activate_contact_sensors=True,
        ),
    )

    o6_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm/left_o6/.*",
        filter_prim_paths_expr=[],
        history_length=3,
        track_air_time=False,
    )
    right_o6_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm/right_o6/.*",
        filter_prim_paths_expr=[],
        history_length=3,
        track_air_time=False,
    )
    left_thumb_distal_box_contact = _box_contact_sensor("left_thumb_distal")
    left_index_proximal_box_contact = _box_contact_sensor("left_index_proximal")
    left_index_distal_box_contact = _box_contact_sensor("left_index_distal")
    left_middle_proximal_box_contact = _box_contact_sensor("left_middle_proximal")
    left_middle_distal_box_contact = _box_contact_sensor("left_middle_distal")
    left_ring_proximal_box_contact = _box_contact_sensor("left_ring_proximal")
    left_ring_distal_box_contact = _box_contact_sensor("left_ring_distal")
    left_pinky_proximal_box_contact = _box_contact_sensor("left_pinky_proximal")
    left_pinky_distal_box_contact = _box_contact_sensor("left_pinky_distal")
    right_thumb_distal_box_contact = _box_contact_sensor("right_thumb_distal")
    right_index_proximal_box_contact = _box_contact_sensor("right_index_proximal")
    right_index_distal_box_contact = _box_contact_sensor("right_index_distal")
    right_middle_proximal_box_contact = _box_contact_sensor("right_middle_proximal")
    right_middle_distal_box_contact = _box_contact_sensor("right_middle_distal")
    right_ring_proximal_box_contact = _box_contact_sensor("right_ring_proximal")
    right_ring_distal_box_contact = _box_contact_sensor("right_ring_distal")
    right_pinky_proximal_box_contact = _box_contact_sensor("right_pinky_proximal")
    right_pinky_distal_box_contact = _box_contact_sensor("right_pinky_distal")
    palm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm/left_o6/left_hand_base_link", history_length=3
    )
    right_palm_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm/right_o6/right_hand_base_link", history_length=3
    )
    wrist_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm/left_panda_link8", history_length=3
    )
    right_wrist_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm/right_panda_link8", history_length=3
    )
    platform_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/platform", history_length=3
    )
    base_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK", history_length=3
    )
    box_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Box", history_length=3)


@configclass
class M1DualPandaO6BimanualActionsCfg:
    """Private canonical 43-channel effort action consumed by the MPC wrapper."""

    joint_effort = isaac_mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=list(M1_DUAL_PANDA_O6_ACTIVE_JOINT_NAMES),
        scale=1.0,
        preserve_order=True,
    )


@configclass
class M1DualPandaO6BimanualObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class M1DualPandaO6BimanualEventsCfg:
    """Deliberately empty: the fixed-condition probe performs no randomization."""

    pass


@configclass
class M1DualPandaO6BimanualTerminationsCfg:
    """Only the episode horizon terminates the deterministic integration scene."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class M1DualPandaO6BimanualEnvCfg(M1SmokeEnvCfg):
    """Fixed-condition inline-execution environment; it has no RL runner."""

    private_action_dim = PRIVATE_ACTION_DIM
    scene: M1DualPandaO6BimanualSceneCfg = M1DualPandaO6BimanualSceneCfg(
        num_envs=1, env_spacing=3.0, replicate_physics=True
    )
    observations: M1DualPandaO6BimanualObservationsCfg = (
        M1DualPandaO6BimanualObservationsCfg()
    )
    actions: M1DualPandaO6BimanualActionsCfg = M1DualPandaO6BimanualActionsCfg()
    events: M1DualPandaO6BimanualEventsCfg = M1DualPandaO6BimanualEventsCfg()
    terminations: M1DualPandaO6BimanualTerminationsCfg = (
        M1DualPandaO6BimanualTerminationsCfg()
    )
    # ``None`` deliberately retains the historical single ``scene.box`` at
    # ``{ENV_REGEX_NS}/Box``.  A catalog is opt-in and adds object-ID-named
    # scene entries without changing that legacy asset.
    object_catalog: ObjectCatalog | None = None
    object_instances: tuple[ObjectInstance, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.object_catalog is not None:
            object_configs = build_object_scene_cfg(
                self.object_catalog,
                self.object_instances,
                "{ENV_REGEX_NS}",
            )
            validate_scene_object_ids(self.scene, object_configs)
            for object_id, object_cfg in object_configs.items():
                setattr(self.scene, object_id, object_cfg)
        self.sim.dt = PHYSICS_DT
        self.decimation = 1
        self.sim.render_interval = 4
        self.episode_length_s = 20.0
        # IsaacLab 2.3 constructs an EventManager unconditionally, so
        # ``self.events = None`` is represented by an empty config object.
        self.events = M1DualPandaO6BimanualEventsCfg()
        for sensor_name in (
            "contact_forces",
            "o6_contacts",
            "right_o6_contacts",
            "palm_contacts",
            "right_palm_contacts",
            "wrist_contacts",
            "right_wrist_contacts",
            "platform_contacts",
            "base_contacts",
            "box_contacts",
            "left_thumb_distal_box_contact",
            "left_index_proximal_box_contact",
            "left_index_distal_box_contact",
            "left_middle_proximal_box_contact",
            "left_middle_distal_box_contact",
            "left_ring_proximal_box_contact",
            "left_ring_distal_box_contact",
            "left_pinky_proximal_box_contact",
            "left_pinky_distal_box_contact",
            "right_thumb_distal_box_contact",
            "right_index_proximal_box_contact",
            "right_index_distal_box_contact",
            "right_middle_proximal_box_contact",
            "right_middle_distal_box_contact",
            "right_ring_proximal_box_contact",
            "right_ring_distal_box_contact",
            "right_pinky_proximal_box_contact",
            "right_pinky_distal_box_contact",
        ):
            getattr(self.scene, sensor_name).update_period = PHYSICS_DT
