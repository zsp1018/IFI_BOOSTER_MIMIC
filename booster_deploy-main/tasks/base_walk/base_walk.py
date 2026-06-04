from __future__ import annotations

from dataclasses import MISSING
import math
import os
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import ControllerCfg, PolicyCfg
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math
from tasks.locomotion.locomotion import T1WalkControllerCfg


class BaseWalkPolicy(Policy):
    """HTWK BaseWalk policy running on the locomotion control stack."""

    def __init__(self, cfg: BaseWalkPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)

        self._model: torch.jit.ScriptModule = torch.jit.load(
            policy_path, map_location="cpu"
        )
        self._model.eval()

        self.default_joint_pos = torch.tensor(
            self.cfg.default_joint_pos, dtype=torch.float32
        )
        self.last_action = torch.zeros(
            len(self.cfg.policy_joint_names), dtype=torch.float32
        )
        self.smoothed_commands = torch.zeros(3, dtype=torch.float32)
        self.gait_process = 0.0

        self.real2sim_joint_map = torch.tensor(
            [
                self.robot.cfg.joint_names.index(name)
                for name in self.cfg.policy_joint_names
            ],
            dtype=torch.long,
        )

    def reset(self) -> None:
        self.last_action.zero_()
        self.smoothed_commands.zero_()
        self.gait_process = 0.0

    def compute_observation(self) -> torch.Tensor:
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b

        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback and projected_gravity[2] > -0.5:
            print(
                "\nFalling detected, stopping policy for safety. "
                "You can disable safety fallback by setting "
                f"{self.cfg.__class__.__name__}.enable_safety_fallback to False."
            )
            self.controller.stop()

        command = torch.tensor(
            [
                self.controller.vel_command.lin_vel_x,
                self.controller.vel_command.lin_vel_y,
                self.controller.vel_command.ang_vel_yaw,
            ],
            dtype=torch.float32,
        )
        slew = self.controller.cfg.policy_dt
        self.smoothed_commands += torch.clamp(
            command - self.smoothed_commands, -slew, slew
        )

        if torch.linalg.vector_norm(self.smoothed_commands) < 1.0e-5:
            gait_frequency = 0.0
            self.gait_process = 0.0
        else:
            gait_frequency = self.cfg.gait_frequency
            self.gait_process = math.fmod(
                self.controller._elapsed_s * gait_frequency, 1.0
            )

        phase = torch.tensor(
            [
                math.cos(2.0 * math.pi * self.gait_process),
                math.sin(2.0 * math.pi * self.gait_process),
            ],
            dtype=torch.float32,
        )
        if gait_frequency <= 1.0e-8:
            phase.zero_()

        mapped_default_pos = self.default_joint_pos[self.real2sim_joint_map]
        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        command_scale = torch.tensor(
            [
                self.cfg.lin_vel_scale,
                self.cfg.lin_vel_scale,
                self.cfg.ang_vel_scale,
            ],
            dtype=torch.float32,
        )
        active = 1.0 if gait_frequency > 1.0e-8 else 0.0

        obs = torch.cat(
            [
                projected_gravity * self.cfg.gravity_scale,
                base_ang_vel * self.cfg.ang_vel_scale,
                self.smoothed_commands * command_scale * active,
                phase * active,
                (mapped_dof_pos - mapped_default_pos) * self.cfg.dof_pos_scale,
                mapped_dof_vel * self.cfg.dof_vel_scale,
                self.last_action,
            ],
            dim=0,
        )
        return obs

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()

        with torch.no_grad():
            action = self._model(obs.unsqueeze(0)).squeeze(0)
            action = torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)

        self.last_action = action.clone()

        dof_targets = self.default_joint_pos.clone()
        dof_targets.scatter_reduce_(
            0,
            self.real2sim_joint_map,
            action * self.cfg.action_scale,
            reduce="sum",
        )
        return dof_targets


@configclass
class BaseWalkPolicyCfg(PolicyCfg):
    constructor = BaseWalkPolicy
    checkpoint_path: str = MISSING  # type: ignore
    action_scale: float = 1.0
    clip_actions: float = 1.0
    gait_frequency: float = 1.0

    gravity_scale: float = 1.0
    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.1

    default_joint_pos: list[float] = MISSING  # type: ignore
    policy_joint_names: list[str] = MISSING  # type: ignore


@configclass
class T1BaseWalkControllerCfg(T1WalkControllerCfg):
    """T1 BaseWalk task using the locomotion startup/control pipeline."""

    robot = T1_23DOF_CFG.replace(  # type: ignore
        default_joint_pos=[
            0.0,
            0.0,
            0.2,
            -1.35,
            0.0,
            -0.5,
            0.2,
            1.35,
            0.0,
            0.5,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,
        ],
        joint_stiffness=[
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
            20.0,
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
        ],
        joint_damping=[
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
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
        ],
    )

    policy: BaseWalkPolicyCfg = BaseWalkPolicyCfg(
        checkpoint_path="models/base_walk.pt",
        default_joint_pos=[
            0.0,
            0.0,
            0.2,
            -1.35,
            0.0,
            -0.5,
            0.2,
            1.35,
            0.0,
            0.5,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,
            -0.2,
            0.0,
            0.0,
            0.4,
            -0.25,
            0.0,
        ],
        policy_joint_names=[
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ],
    )
