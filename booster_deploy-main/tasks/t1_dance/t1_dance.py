from __future__ import annotations

from booster_deploy.controllers.controller_cfg import ControllerCfg, MujocoControllerCfg
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass

from tasks.kicking_mimic.kicking_mimic import (
    T1KickingMimicPolicy,
    T1KickingMimicPolicyCfg,
)


class T1DancePolicy(T1KickingMimicPolicy):
    """Thin wrapper so task-relative assets resolve inside tasks/t1_dance."""

    def reset(self) -> None:
        super().reset()
        self.current_frame = 2.3


@configclass
class T1DancePolicyCfg(T1KickingMimicPolicyCfg):
    constructor = T1DancePolicy


T1_DANCE_BODY_NAMES = [
    "Trunk",
    "H1",
    "AL1",
    "AR1",
    "Waist",
    "H2",
    "AL2",
    "AR2",
    "Hip_Pitch_Left",
    "Hip_Pitch_Right",
    "AL3",
    "AR3",
    "Hip_Roll_Left",
    "Hip_Roll_Right",
    "left_hand_link",
    "right_hand_link",
    "Hip_Yaw_Left",
    "Hip_Yaw_Right",
    "Shank_Left",
    "Shank_Right",
    "Ankle_Cross_Left",
    "Ankle_Cross_Right",
    "left_foot_link",
    "right_foot_link",
]

T1_DANCE_DEFAULT_JOINT_POS = [
    0.0,
    0.0,
    0.2,
    -1.3,
    0.0,
    -0.5,
    0.2,
    1.3,
    0.0,
    0.5,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
]

T1_DANCE_JOINT_STIFFNESS = [
    10.0,
    10.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    50.0,
    200.0,
    200.0,
    200.0,
    200.0,
    200.0,
    50.0,
    50.0,
    200.0,
    200.0,
    200.0,
    200.0,
    50.0,
    50.0,
]

T1_DANCE_JOINT_DAMPING = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    1.0,
    1.0,
    5.0,
    5.0,
    5.0,
    5.0,
    1.0,
    1.0,
]

T1_DANCE_EFFORT_LIMIT = [
    7.0,
    7.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    18.0,
    25.0,
    45.0,
    25.0,
    25.0,
    60.0,
    24.0,
    15.0,
    45.0,
    25.0,
    25.0,
    60.0,
    24.0,
    15.0,
]

T1_DANCE_ACTION_SCALE_SIM = [
    0.25
    * T1_DANCE_EFFORT_LIMIT[T1_23DOF_CFG.joint_names.index(joint_name)]
    / T1_DANCE_JOINT_STIFFNESS[T1_23DOF_CFG.joint_names.index(joint_name)]
    for joint_name in T1_23DOF_CFG.sim_joint_names
]


@configclass
class T1DanceControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG.replace(  # type: ignore
        sim_body_names=T1_DANCE_BODY_NAMES,
        default_joint_pos=T1_DANCE_DEFAULT_JOINT_POS,
        joint_stiffness=T1_DANCE_JOINT_STIFFNESS,
        joint_damping=T1_DANCE_JOINT_DAMPING,
        effort_limit=T1_DANCE_EFFORT_LIMIT,
    )
    vel_command = None
    policy: T1DancePolicyCfg = T1DancePolicyCfg(
        action_scale_sim=T1_DANCE_ACTION_SCALE_SIM,
    )
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.70],
        visualize_reference_ghost=False,
    )
