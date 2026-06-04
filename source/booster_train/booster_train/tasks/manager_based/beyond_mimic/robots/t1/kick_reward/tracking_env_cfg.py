from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_train.tasks.manager_based.beyond_mimic.mdp as mdp


T1_SINGLE_CLIP_VELOCITY_RANGE = {
    "x": (-0.35, 0.35),
    "y": (-0.25, 0.25),
    "z": (-0.15, 0.15),
    "roll": (-0.35, 0.35),
    "pitch": (-0.35, 0.35),
    "yaw": (-0.52, 0.52),
}

TRACK_HAND_BODY_NAMES = ["left_hand_link", "right_hand_link"]
TRACK_FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]
TRACK_TRUNK_BODY_NAMES = ["Trunk"]
TRACK_END_EFFECTOR_BODY_NAMES = TRACK_HAND_BODY_NAMES + TRACK_FOOT_BODY_NAMES


@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-0.12, 0.12),
        },
        velocity_range=T1_SINGLE_CLIP_VELOCITY_RANGE,
        joint_position_range=(-0.08, 0.08),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.04, n_max=0.04)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.35, n_max=0.35))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.15, n_max=0.15))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.35, n_max=0.35))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 0.8),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.008, 0.008),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.04, 0.04), "z": (-0.04, 0.04)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": T1_SINGLE_CLIP_VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.25},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.35},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.25},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.35},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.8},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 2.5},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.2)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"],
            ),
            "threshold": 2.0,
        },
    )
    motion_foot_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=12.0,
        params={"command_name": "motion", "std": 0.2, "body_names": TRACK_FOOT_BODY_NAMES},
    )
    motion_hand_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.2, "body_names": TRACK_HAND_BODY_NAMES},
    )
    motion_foot_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=12.0,
        params={"command_name": "motion", "std": 0.18, "body_names": TRACK_FOOT_BODY_NAMES},
    )
    motion_hand_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=8.0,
        params={"command_name": "motion", "std": 0.18, "body_names": TRACK_HAND_BODY_NAMES},
    )
    motion_trunk_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=12.0,
        params={"command_name": "motion", "std": 0.18, "body_names": TRACK_TRUNK_BODY_NAMES},
    )
    motion_trunk_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=18.0,
        params={"command_name": "motion", "std": 0.18, "body_names": TRACK_TRUNK_BODY_NAMES},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.35},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.0},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={"command_name": "motion", "threshold": 0.3, "body_names": TRACK_END_EFFECTOR_BODY_NAMES},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        # Shorter episodes help a single 1s clip reset more often during early tuning.
        self.episode_length_s = 6.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "world"
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
