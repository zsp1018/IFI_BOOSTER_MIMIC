from __future__ import annotations

from dataclasses import MISSING
import os

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    PolicyCfg,
)
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.motion_loader import MotionLoader


class T1KickingMimicPolicy(Policy):
    """T1 单段踢球 mimic 策略。"""

    def __init__(self, cfg: T1KickingMimicPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        checkpoint_path = self.cfg.checkpoint_path
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(self.task_path, checkpoint_path)
        self._model: torch.jit.ScriptModule = torch.jit.load(
            checkpoint_path,
            map_location="cpu",
        )
        self._model.to(self.cfg.device).eval()

        motion_path = self.cfg.motion_path
        if not os.path.isabs(motion_path):
            motion_path = os.path.join(self.task_path, motion_path)
        self.motion = MotionLoader(
            motion_file=motion_path,
            track_body_names=[self.cfg.anchor_body_name],
            track_joint_names=self.robot.cfg.sim_joint_names,
            default_motion_body_names=self.robot.cfg.sim_body_names,
            default_motion_joint_names=self.robot.cfg.sim_joint_names,
            align_to_first_frame=True,
            device=self.cfg.device,
        )

        self.robot.data.to(self.cfg.device)
        self.default_joint_pos = self.robot.default_joint_pos.to(self.cfg.device)
        action_scale_sim = self.cfg.action_scale_sim or T1_VIDEO_009D_ACTION_SCALE_SIM
        self.action_scale_sim = torch.tensor(
            action_scale_sim,
            dtype=torch.float32,
            device=self.cfg.device,
        )
        self.real2sim_joint_map = torch.tensor(
            self.robot.data.real2sim_joint_indexes,
            dtype=torch.long,
            device=self.cfg.device,
        )
        self.sim2real_joint_map = torch.tensor(
            self.robot.data.sim2real_joint_indexes,
            dtype=torch.long,
            device=self.cfg.device,
        )

        self.anchor_index = self.motion.track_body_names.index(self.cfg.anchor_body_name)
        self.motion_fps = float(torch.as_tensor(self.motion.fps).reshape(-1)[0].item())
        self.motion_frame_step = self.motion_fps * self.controller.cfg.policy_dt
        self.current_frame = 0.0
        self.last_action = torch.zeros(
            self.robot.num_joints,
            dtype=torch.float32,
            device=self.cfg.device,
        )

    def reset(self) -> None:
        self.current_frame = min(self.motion_fps * 1.7, self.motion.time_step_total - 1)
        self.last_action.zero_()
        self.motion.to(self.cfg.device)
        self.init_root_yaw_quat_w_inv = lab_math.quat_inv(
            lab_math.yaw_quat(self.robot.data.root_quat_w)
        )

    def _sample_motion_frame(
        self, frame_id: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_idx = int(frame_id)
        upper_idx = min(lower_idx + 1, self.motion.time_step_total - 1)
        tau = float(frame_id - lower_idx)

        joint_pos = torch.lerp(
            self.motion.joint_pos[lower_idx],
            self.motion.joint_pos[upper_idx],
            tau,
        )
        joint_vel = torch.lerp(
            self.motion.joint_vel[lower_idx],
            self.motion.joint_vel[upper_idx],
            tau,
        )
        root_pos_w = torch.lerp(
            self.motion.body_pos_w[lower_idx, self.anchor_index],
            self.motion.body_pos_w[upper_idx, self.anchor_index],
            tau,
        )
        root_quat_w = lab_math.quat_slerp(
            self.motion.body_quat_w[lower_idx, self.anchor_index].clone(),
            self.motion.body_quat_w[upper_idx, self.anchor_index].clone(),
            tau,
        )
        return joint_pos, joint_vel, root_pos_w, root_quat_w

    def _set_command(self) -> None:
        frame_id = min(self.current_frame, self.motion.time_step_total - 1)
        (
            self.cmd_dof_pos,
            self.cmd_dof_vel,
            self.cmd_root_pos_w,
            self.cmd_root_quat_w,
        ) = self._sample_motion_frame(frame_id)

    def compute_observation(self) -> torch.Tensor:
        self._set_command()

        command = torch.cat([self.cmd_dof_pos, self.cmd_dof_vel], dim=0)
        cur_root_quat_w = lab_math.quat_mul(
            self.init_root_yaw_quat_w_inv,
            self.robot.data.root_quat_w,
        )
        _, motion_anchor_ori_b = lab_math.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            cur_root_quat_w,
            self.cmd_root_pos_w,
            self.cmd_root_quat_w,
        )
        motion_anchor_ori_b = lab_math.matrix_from_quat(
            motion_anchor_ori_b
        )[..., :2].flatten()

        joint_pos_sim = self.robot.data.joint_pos[self.real2sim_joint_map]
        joint_vel_sim = self.robot.data.joint_vel[self.real2sim_joint_map]
        default_joint_pos_sim = self.default_joint_pos[self.real2sim_joint_map]

        obs = torch.cat(
            (
                command,
                motion_anchor_ori_b,
                self.robot.data.root_ang_vel_b,
                joint_pos_sim - default_joint_pos_sim,
                joint_vel_sim,
                self.last_action,
            ),
            dim=-1,
        )
        return obs.reshape(1, -1)

    def inference(self) -> torch.Tensor:
        with torch.no_grad():
            obs = self.compute_observation()
            action = self._model(obs).flatten()

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
class T1KickingMimicPolicyCfg(PolicyCfg):
    constructor = T1KickingMimicPolicy
    checkpoint_path: str = MISSING
    motion_path: str = MISSING
    action_scale_sim: list[float] | None = None
    anchor_body_name: str = "Trunk"
    enable_safety_fallback: bool = True


T1_VIDEO_009D_BODY_NAMES = [
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

T1_VIDEO_009D_DEFAULT_JOINT_POS = [
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

T1_VIDEO_009D_JOINT_STIFFNESS = [
    7.106115,
    7.106115,
    111.537584,
    111.537584,
    111.537584,
    111.537584,
    111.537584,
    111.537584,
    111.537584,
    111.537584,
    188.756184,
    206.830588,
    188.756184,
    188.756184,
    251.087473,
    268.099513,
    268.099513,
    206.830588,
    188.756184,
    188.756184,
    251.087473,
    268.099513,
    268.099513,
]

T1_VIDEO_009D_JOINT_DAMPING = [
    0.452389,
    0.452389,
    7.100703,
    7.100703,
    7.100703,
    7.100703,
    7.100703,
    7.100703,
    7.100703,
    7.100703,
    12.016592,
    13.167244,
    12.016592,
    12.016592,
    15.984725,
    17.067745,
    17.067745,
    13.167244,
    12.016592,
    12.016592,
    15.984725,
    17.067745,
    17.067745,
]

T1_VIDEO_009D_EFFORT_LIMIT = [
    7.0,
    7.0,
    38.3,
    38.3,
    38.3,
    38.3,
    38.3,
    38.3,
    38.3,
    38.3,
    68.0,
    96.0,
    68.0,
    68.0,
    130.0,
    76.0,
    76.0,
    96.0,
    68.0,
    68.0,
    130.0,
    76.0,
    76.0,
]

T1_VIDEO_009D_ACTION_SCALE_SIM = [
    0.25
    * T1_VIDEO_009D_EFFORT_LIMIT[T1_23DOF_CFG.joint_names.index(joint_name)]
    / T1_VIDEO_009D_JOINT_STIFFNESS[T1_23DOF_CFG.joint_names.index(joint_name)]
    for joint_name in T1_23DOF_CFG.sim_joint_names
]


@configclass
class T1KickingMimicControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG.replace(  # type: ignore
        sim_body_names=T1_VIDEO_009D_BODY_NAMES,
        default_joint_pos=T1_VIDEO_009D_DEFAULT_JOINT_POS,
        joint_stiffness=T1_VIDEO_009D_JOINT_STIFFNESS,
        joint_damping=T1_VIDEO_009D_JOINT_DAMPING,
        effort_limit=T1_VIDEO_009D_EFFORT_LIMIT,
    )
    vel_command = None
    policy: T1KickingMimicPolicyCfg = T1KickingMimicPolicyCfg()
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.70],
        visualize_reference_ghost=True,
    )
