"""Convert video2robot robot_motion.pkl to Booster tracking csv + npz."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert video2robot robot_motion.pkl to Booster tracking csv + npz.")
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input_file", type=str, help="Path to one video2robot robot_motion.pkl")
input_group.add_argument("--input_dir", type=str, help="Directory containing multiple robot_motion.pkl files")
parser.add_argument("--output_name", type=str, help="Output motion npz path for single-file mode")
parser.add_argument("--output_dir", type=str, help="Output directory for batch mode")
parser.add_argument("--csv_output_name", type=str, help="Output motion csv path for single-file mode")
parser.add_argument("--csv_output_dir", type=str, help="Output directory for csv files in batch mode")
parser.add_argument("--output_fps", type=float, default=None, help="Output fps. Default: use pkl fps")
parser.add_argument("--pattern", type=str, default="*.pkl", help="Glob pattern for batch mode")
parser.add_argument("--recursive", action="store_true", help="Recursively search input_dir in batch mode")
parser.add_argument("--manifest_path", type=str, default=None, help="Optional JSON manifest path for batch mode")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files in batch mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.input_file and not args_cli.output_name:
    parser.error("--output_name is required when using --input_file")
if args_cli.input_dir and not args_cli.output_dir:
    parser.error("--output_dir is required when using --input_dir")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from booster_train.assets.robots.booster import BOOSTER_K1_CFG, BOOSTER_T1_CFG
from booster_assets.motions import K1_JOINT_NAMES, T1_JOINT_NAMES
from _motion_io_common import load_motion_csv, load_motion_pkl, save_motion_csv
from _motion_to_npz_common import MotionSequence, make_replay_scene_cfg, save_motion_npz


ROBOT_MAP = {
    "booster_k1": (BOOSTER_K1_CFG, K1_JOINT_NAMES),
    "booster_t1": (BOOSTER_T1_CFG, T1_JOINT_NAMES),
}

def _resolve_jobs():
    if args_cli.input_file:
        output_path = Path(args_cli.output_name)
        csv_path = Path(args_cli.csv_output_name) if args_cli.csv_output_name else output_path.with_suffix(".csv")
        return [{"input_path": Path(args_cli.input_file), "output_path": output_path, "csv_path": csv_path}]

    input_dir = Path(args_cli.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid input_dir: {input_dir}")

    output_dir = Path(args_cli.output_dir)
    csv_output_dir = Path(args_cli.csv_output_dir) if args_cli.csv_output_dir else output_dir / "csv"
    iterator = input_dir.rglob(args_cli.pattern) if args_cli.recursive else input_dir.glob(args_cli.pattern)
    input_paths = sorted(path for path in iterator if path.is_file())
    if not input_paths:
        raise FileNotFoundError(f"No pkl files found in {input_dir} matching pattern {args_cli.pattern!r}")

    jobs = []
    for input_path in input_paths:
        relative_output = input_path.relative_to(input_dir).with_suffix(".npz")
        relative_csv = input_path.relative_to(input_dir).with_suffix(".csv")
        output_path = output_dir / relative_output
        csv_path = csv_output_dir / relative_csv
        if output_path.exists() and not args_cli.overwrite:
            print(f"[INFO]: Skipping existing output: {output_path}")
            continue
        jobs.append({"input_path": input_path, "output_path": output_path, "csv_path": csv_path})

    if not jobs:
        raise FileExistsError("All batch outputs already exist. Use --overwrite to regenerate them.")
    return jobs


def _prepare_jobs():
    jobs = _resolve_jobs()
    prepared = []
    for job in jobs:
        motion_data = load_motion_pkl(str(job["input_path"]))
        required = {"fps", "robot_type", "root_pos", "root_rot", "dof_pos"}
        missing = sorted(required - set(motion_data.keys()))
        if missing:
            raise KeyError(f"Missing keys in pkl: {missing}")
        prepared.append(
            {
                **job,
                "motion_data": motion_data,
                "robot_type": str(motion_data["robot_type"]),
                "input_fps": float(motion_data["fps"]),
            }
        )

    robot_types = sorted({job["robot_type"] for job in prepared})
    if len(robot_types) != 1:
        raise ValueError(
            "Batch conversion currently requires all pkls in one run to have the same robot_type. "
            f"Found: {robot_types}"
        )

    if args_cli.output_fps is None:
        fps_values = sorted({job["input_fps"] for job in prepared})
        if len(fps_values) != 1:
            raise ValueError(
                "Batch conversion with mixed input fps requires --output_fps. "
                f"Found input fps values: {fps_values}"
            )
        output_fps = fps_values[0]
    else:
        output_fps = float(args_cli.output_fps)

    return prepared, robot_types[0], output_fps


def main():
    jobs, robot_type, output_fps = _prepare_jobs()
    robot_cfg, joint_names = ROBOT_MAP[robot_type]

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(output_fps)
    sim = SimulationContext(sim_cfg)
    scene_cfg = make_replay_scene_cfg(robot_cfg)(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print(f"[INFO]: Setup complete for {robot_type} at output_fps={output_fps}...")

    manifest = []
    for index, job in enumerate(jobs, start=1):
        motion_data = job["motion_data"]
        root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
        input_fps = job["input_fps"]

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"Invalid root_pos shape: {root_pos.shape} for {job['input_path']}")
        if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
            raise ValueError(f"Invalid root_rot shape: {root_rot_xyzw.shape} for {job['input_path']}")
        if dof_pos.ndim != 2:
            raise ValueError(f"Invalid dof_pos shape: {dof_pos.shape} for {job['input_path']}")
        if dof_pos.shape[1] != len(joint_names):
            raise ValueError(
                f"DOF count mismatch in {job['input_path']}: "
                f"pkl has {dof_pos.shape[1]}, expected {len(joint_names)} for {robot_type}"
            )

        print(
            f"[INFO]: Converting {index}/{len(jobs)}: {job['input_path']} "
            f"(frames={root_pos.shape[0]}, input_fps={input_fps}, output_fps={output_fps})"
        )

        csv_path = save_motion_csv(
            job["csv_path"],
            root_pos=root_pos,
            root_rot_xyzw=root_rot_xyzw,
            dof_pos=dof_pos,
        )

        motion = torch.from_numpy(load_motion_csv(csv_path)).to(torch.float32).to(sim.device)
        motion_sequence = MotionSequence(
            base_pos_input=motion[:, :3],
            base_rot_wxyz_input=motion[:, 3:7][:, [3, 0, 1, 2]],
            dof_pos_input=motion[:, 7:],
            input_fps=input_fps,
            output_fps=output_fps,
            device=sim.device,
        )

        save_motion_npz(
            simulation_app=simulation_app,
            sim=sim,
            scene=scene,
            motion=motion_sequence,
            joint_names=joint_names,
            output_name=str(job["output_path"]),
        )

        manifest.append(
            {
                "input_file": str(job["input_path"]),
                "csv_file": str(csv_path),
                "output_file": str(job["output_path"]),
                "robot_type": robot_type,
                "frames": int(root_pos.shape[0]),
                "input_fps": input_fps,
                "output_fps": float(output_fps),
            }
        )

    if args_cli.input_dir:
        manifest_path = Path(args_cli.manifest_path) if args_cli.manifest_path else Path(args_cli.output_dir) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO]: Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
    # Isaac Sim teardown can hang after outputs have already been written.
    # This script is a one-shot converter, so exit the process directly on success.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
