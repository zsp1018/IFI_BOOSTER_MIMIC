from __future__ import annotations

import torch

from booster_deploy.controllers.controller_cfg import ControllerCfg, MujocoControllerCfg
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass

from tasks.kick.kick import (
    T1KickingMimicPolicy,
    T1KickingMimicPolicyCfg,
)


class T1DancePolicy(T1KickingMimicPolicy):
    """Thin wrapper so task-relative assets resolve inside tasks/t1_dance."""

    def __init__(self, cfg: T1DancePolicyCfg, controller):
        super().__init__(cfg, controller)
        self.actor_obs_history_length = cfg.actor_obs_history_length
        self.obs_history = None

    def reset(self) -> None:
        super().reset()
        self.current_frame = 0
        self.obs_history = None

    def inference(self):
        obs = self.compute_observation().flatten()

        if self.obs_history is None:
            self.obs_history = torch.zeros(
                self.actor_obs_history_length,
                obs.numel(),
                dtype=torch.float32,
                device=obs.device,
            )

        self.obs_history = self.obs_history.roll(shifts=-1, dims=0)
        self.obs_history[-1] = obs.clamp(-100.0, 100.0)

        with torch.no_grad():
            action = self._model(self.obs_history.flatten()).flatten()
            # action = torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)

        if hasattr(self.controller, "set_reference_qpos"):
            joint_pos_real = self.cmd_dof_pos[self.sim2real_joint_map]
            ref_qpos = torch.cat(
                [self.cmd_root_pos_w, self.cmd_root_quat_w, joint_pos_real],
                dim=0,
            )
            self.controller.set_reference_qpos(ref_qpos)  # type: ignore[attr-defined]

        self.last_action = action.clone()
        self.current_frame = min(
            self.current_frame + self.motion_frame_step,
            self.motion.time_step_total - 1,
        )

        if self.cfg.enable_safety_fallback:
            gravity_w = torch.tensor(
                [0.0, 0.0, -1.0],
                dtype=torch.float32,
                device=self.cfg.device,
            )
            projected_gravity = lab_math.quat_apply_inverse(
                self.robot.data.root_quat_w,
                gravity_w,
            )
            if projected_gravity[2] > -0.5:
                print(
                    "\nFalling detected, stopping policy for safety. "
                    "You can disable safety fallback by setting "
                    f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                    "to False."
                )
                if hasattr(self.controller, "disable_motors"):
                    self.controller.disable_motors()  # type: ignore[attr-defined]
                else:
                    self.controller.stop()

        default_joint_pos_sim = self.default_joint_pos[self.real2sim_joint_map]
        dof_targets_sim = action * self.action_scale_sim + default_joint_pos_sim
        return dof_targets_sim[self.sim2real_joint_map]


@configclass
class T1DancePolicyCfg(T1KickingMimicPolicyCfg):
    constructor = T1DancePolicy
    actor_obs_history_length: int = 10
    # clip_actions: float = 1.0


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
