from __future__ import annotations

import json
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str,
                 track_body_names: Sequence[str],
                 track_joint_names: Sequence[str],
                 *,
                 default_motion_body_names: Sequence[str] | None = None,
                 default_motion_joint_names: Sequence[str] | None = None,
        tail_len: int = 0, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if "body_names" in data:
            self._body_names = data["body_names"].tolist()
        else:
            assert default_motion_body_names is not None, "Motion file missing body_names, and no default_body_names provided."
            self._body_names = default_motion_body_names
        if "joint_names" in data:
            self._joint_names = data["joint_names"].tolist()
        else:
            assert default_motion_joint_names is not None, "Motion file missing joint_names, and no default_joint_names provided."
            self._joint_names = default_motion_joint_names
        self._body_indexes = torch.tensor(
            [self._body_names.index(name) for name in track_body_names], dtype=torch.long, device=device
        )
        self._joint_indexes = torch.tensor(
            [self._joint_names.index(name) for name in track_joint_names], dtype=torch.long, device=device
        )
        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device)[:, self._joint_indexes]
        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device)[:, self._joint_indexes]
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]
        self.tail_len = tail_len

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def max_reset_frame(self) -> int:
        return self.time_step_total - self.tail_len


def _resolve_motion_files(
    motion_file: str | None,
    motion_files: Sequence[str] | None,
    motion_manifest: str | None,
) -> list[str]:
    resolved_files: list[str] = []

    if motion_manifest is not None:
        assert os.path.isfile(motion_manifest), f"Invalid manifest path: {motion_manifest}"
        with open(motion_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        if isinstance(manifest_data, dict):
            manifest_entries = manifest_data.get("motions", manifest_data.get("files", []))
        elif isinstance(manifest_data, list):
            manifest_entries = manifest_data
        else:
            raise TypeError(f"Unsupported manifest format in {motion_manifest}")

        manifest_dir = os.path.dirname(os.path.abspath(motion_manifest))
        for entry in manifest_entries:
            if isinstance(entry, str):
                path = entry
            elif isinstance(entry, dict):
                path = entry.get("output_file") or entry.get("motion_file") or entry.get("file")
                if path is None:
                    raise KeyError(f"Manifest entry missing output_file/motion_file/file: {entry}")
            else:
                raise TypeError(f"Unsupported manifest entry: {entry}")
            if not os.path.isabs(path):
                path = os.path.join(manifest_dir, path)
            resolved_files.append(path)

    if motion_files is not None:
        resolved_files.extend(motion_files)

    if motion_file is not None:
        resolved_files.append(motion_file)

    # Preserve order while deduplicating.
    deduplicated_files = list(dict.fromkeys(resolved_files))
    if not deduplicated_files:
        raise ValueError("At least one motion source must be provided.")

    for path in deduplicated_files:
        assert os.path.isfile(path), f"Invalid motion file path: {path}"

    return deduplicated_files


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        default_motion_body_names = self.cfg.default_motion_body_names or self.robot.body_names
        default_motion_joint_names = self.cfg.default_motion_joint_names or self.robot.joint_names

        self.motion_files = _resolve_motion_files(
            self.cfg.motion_file,
            self.cfg.motion_files,
            self.cfg.motion_manifest,
        )
        self.motions = [
            MotionLoader(
                motion_file,
                self.cfg.body_names,
                self.robot.joint_names,
                default_motion_body_names=default_motion_body_names,
                default_motion_joint_names=default_motion_joint_names,
                tail_len=self.cfg.tail_len,
                device=self.device,
            )
            for motion_file in self.motion_files
        ]
        self.motion_count = len(self.motions)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.motion_lengths = torch.tensor(
            [motion.time_step_total for motion in self.motions], dtype=torch.long, device=self.device
        )
        self.motion_max_reset_frames = torch.tensor(
            [motion.max_reset_frame for motion in self.motions], dtype=torch.long, device=self.device
        )
        self.motion_fps = torch.tensor([motion.fps for motion in self.motions], dtype=torch.float32, device=self.device)
        if not torch.allclose(self.motion_fps, self.motion_fps[:1]):
            raise ValueError(f"All motions must share the same fps. Got: {self.motion_fps.tolist()}")

        frame_span = 1 / (env.cfg.decimation * env.cfg.sim.dt)
        self.motion_bin_counts = [
            int(max(int(motion.max_reset_frame), 1) // frame_span) + 1 for motion in self.motions
        ]
        self.bin_failed_count = [
            torch.zeros(bin_count, dtype=torch.float32, device=self.device) for bin_count in self.motion_bin_counts
        ]
        self._current_bin_failed = [
            torch.zeros(bin_count, dtype=torch.float32, device=self.device) for bin_count in self.motion_bin_counts
        ]
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        self._joint_pos = torch.zeros(self.num_envs, len(self.robot.joint_names), dtype=torch.float32, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, len(self.robot.joint_names), dtype=torch.float32, device=self.device)
        self._body_pos_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, dtype=torch.float32, device=self.device)
        self._body_quat_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, dtype=torch.float32, device=self.device)
        self._body_quat_w[:, :, 0] = 1.0
        self._body_lin_vel_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, dtype=torch.float32, device=self.device)
        self._body_ang_vel_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, dtype=torch.float32, device=self.device)
        self._refresh_motion_state_cache()

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _gather_motion_state(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        joint_pos = torch.zeros((len(env_ids), len(self.robot.joint_names)), dtype=torch.float32, device=self.device)
        joint_vel = torch.zeros((len(env_ids), len(self.robot.joint_names)), dtype=torch.float32, device=self.device)
        body_pos_w = torch.zeros((len(env_ids), len(self.cfg.body_names), 3), dtype=torch.float32, device=self.device)
        body_quat_w = torch.zeros((len(env_ids), len(self.cfg.body_names), 4), dtype=torch.float32, device=self.device)
        body_quat_w[:, :, 0] = 1.0
        body_lin_vel_w = torch.zeros((len(env_ids), len(self.cfg.body_names), 3), dtype=torch.float32, device=self.device)
        body_ang_vel_w = torch.zeros((len(env_ids), len(self.cfg.body_names), 3), dtype=torch.float32, device=self.device)

        selected_motion_ids = self.motion_ids[env_ids]
        selected_time_steps = self.time_steps[env_ids]
        unique_motion_ids = torch.unique(selected_motion_ids).tolist()
        for motion_id in unique_motion_ids:
            mask = selected_motion_ids == motion_id
            local_env_ids = env_ids[mask]
            local_time_steps = selected_time_steps[mask]
            motion = self.motions[motion_id]
            joint_pos[mask] = motion.joint_pos[local_time_steps]
            joint_vel[mask] = motion.joint_vel[local_time_steps]
            body_pos_w[mask] = motion.body_pos_w[local_time_steps] + self._env.scene.env_origins[local_env_ids, None, :]
            body_quat_w[mask] = motion.body_quat_w[local_time_steps]
            body_lin_vel_w[mask] = motion.body_lin_vel_w[local_time_steps]
            body_ang_vel_w[mask] = motion.body_ang_vel_w[local_time_steps]

        return joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    def _refresh_motion_state_cache(self):
        (
            self._joint_pos,
            self._joint_vel,
            self._body_pos_w,
            self._body_quat_w,
            self._body_lin_vel_w,
            self._body_ang_vel_w,
        ) = self._gather_motion_state()

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)[episode_failed]
            failed_motion_ids = self.motion_ids[failed_env_ids]
            for motion_id in torch.unique(failed_motion_ids).tolist():
                motion_mask = failed_motion_ids == motion_id
                motion_env_ids = failed_env_ids[motion_mask]
                bin_count = self.motion_bin_counts[motion_id]
                max_reset_frame = max(int(self.motion_max_reset_frames[motion_id].item()), 1)
                current_bin_index = torch.clamp(
                    (self.time_steps[motion_env_ids] * bin_count) // max_reset_frame, 0, bin_count - 1
                )
                self._current_bin_failed[motion_id] += torch.bincount(
                    current_bin_index, minlength=bin_count
                ).to(torch.float32)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if self.cfg.play:
            self.motion_ids[env_ids] = 0
            self.time_steps[env_ids] = 0
            self.metrics["sampling_entropy"][env_ids] = 1.0
            self.metrics["sampling_top1_prob"][env_ids] = 1.0
            self.metrics["sampling_top1_bin"][env_ids] = 0.0
            return

        if self.motion_count > 1:
            self.motion_ids[env_ids] = torch.randint(self.motion_count, (len(env_ids),), device=self.device)

        selected_motion_ids = self.motion_ids[env_ids]
        for motion_id in torch.unique(selected_motion_ids).tolist():
            motion_mask = selected_motion_ids == motion_id
            motion_env_ids = env_ids[motion_mask]
            bin_count = self.motion_bin_counts[motion_id]
            max_reset_frame = max(int(self.motion_max_reset_frames[motion_id].item()), 1)

            sampling_probabilities = self.bin_failed_count[motion_id] + self.cfg.adaptive_uniform_ratio / float(bin_count)
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.kernel.view(1, 1, -1)
            ).view(-1)
            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

            sampled_bins = torch.multinomial(sampling_probabilities, len(motion_env_ids), replacement=True)
            if max_reset_frame <= 1:
                self.time_steps[motion_env_ids] = 0
            else:
                self.time_steps[motion_env_ids] = (
                    sampled_bins.float() / bin_count * float(max_reset_frame - 1)
                ).long()

            H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
            H_norm = H / math.log(bin_count) if bin_count > 1 else torch.tensor(1.0, device=self.device)
            pmax, imax = sampling_probabilities.max(dim=0)
            self.metrics["sampling_entropy"][motion_env_ids] = H_norm
            self.metrics["sampling_top1_prob"][motion_env_ids] = pmax
            self.metrics["sampling_top1_bin"][motion_env_ids] = imax.float() / bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        (
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = self._gather_motion_state(env_ids)

        root_pos = body_pos_w[:, 0].clone()
        root_ori = body_quat_w[:, 0].clone()
        root_lin_vel = body_lin_vel_w[:, 0].clone()
        root_ang_vel = body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion_lengths[self.motion_ids])[0]
        self._resample_command(env_ids)
        self._refresh_motion_state_cache()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        for motion_id in range(self.motion_count):
            self.bin_failed_count[motion_id] = (
                self.cfg.adaptive_alpha * self._current_bin_failed[motion_id]
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count[motion_id]
            )
            self._current_bin_failed[motion_id].zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    play: bool = False

    asset_name: str = MISSING

    motion_file: str | None = None
    motion_files: list[str] | None = None
    motion_manifest: str | None = None
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    default_motion_body_names: list[str] | None = None
    default_motion_joint_names: list[str] | None = None
    tail_len: int = 0

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
