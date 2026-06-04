"""Convert video2robot robot_motion.pkl to Booster tracking csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from booster_assets.motions import K1_JOINT_NAMES, T1_JOINT_NAMES

from _motion_io_common import load_motion_pkl, save_motion_csv


ROBOT_JOINT_NAMES = {
    "booster_k1": K1_JOINT_NAMES,
    "booster_t1": T1_JOINT_NAMES,
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Convert video2robot robot_motion.pkl to Booster tracking csv.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=str, help="Path to one video2robot robot_motion.pkl")
    input_group.add_argument("--input_dir", type=str, help="Directory containing multiple robot_motion.pkl files")
    parser.add_argument("--output_name", type=str, help="Output motion csv path for single-file mode")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch mode")
    parser.add_argument("--pattern", type=str, default="*.pkl", help="Glob pattern for batch mode")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input_dir in batch mode")
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional JSON manifest path for batch mode")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files in batch mode")
    args = parser.parse_args()

    if args.input_file and not args.output_name:
        parser.error("--output_name is required when using --input_file")
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    return args


def _resolve_jobs(args) -> list[dict]:
    if args.input_file:
        output_path = Path(args.output_name)
        return [{"input_path": Path(args.input_file), "output_path": output_path, "relative_path": output_path.name}]

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid input_dir: {input_dir}")

    output_dir = Path(args.output_dir)
    iterator = input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern)
    input_paths = sorted(path for path in iterator if path.is_file())
    if not input_paths:
        raise FileNotFoundError(f"No pkl files found in {input_dir} matching pattern {args.pattern!r}")

    jobs = []
    for input_path in input_paths:
        relative_output = input_path.relative_to(input_dir).with_suffix(".csv")
        output_path = output_dir / relative_output
        if output_path.exists() and not args.overwrite:
            print(f"[INFO]: Skipping existing output: {output_path}")
            continue
        jobs.append(
            {
                "input_path": input_path,
                "output_path": output_path,
                "relative_path": relative_output.as_posix(),
            }
        )

    if not jobs:
        raise FileExistsError("All batch outputs already exist. Use --overwrite to regenerate them.")
    return jobs


def main():
    args = _parse_args()
    jobs = _resolve_jobs(args)

    prepared_jobs = []
    for job in jobs:
        motion_data = load_motion_pkl(job["input_path"])

        required = {"fps", "robot_type", "root_pos", "root_rot", "dof_pos"}
        missing = sorted(required - set(motion_data.keys()))
        if missing:
            raise KeyError(f"Missing keys in pkl {job['input_path']}: {missing}")

        robot_type = str(motion_data["robot_type"])
        if robot_type not in ROBOT_JOINT_NAMES:
            raise ValueError(
                f"Unsupported robot_type={robot_type} in {job['input_path']}. "
                f"Supported types: {sorted(ROBOT_JOINT_NAMES.keys())}"
            )

        root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
        input_fps = float(motion_data["fps"])
        joint_names = ROBOT_JOINT_NAMES[robot_type]

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"Invalid root_pos shape: {root_pos.shape} for {job['input_path']}")
        if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
            raise ValueError(f"Invalid root_rot shape: {root_rot_xyzw.shape} for {job['input_path']}")
        if dof_pos.ndim != 2 or dof_pos.shape[1] != len(joint_names):
            raise ValueError(
                f"DOF count mismatch in {job['input_path']}: "
                f"pkl has {dof_pos.shape[1]}, expected {len(joint_names)} for {robot_type}"
            )

        prepared_jobs.append(
            {
                **job,
                "robot_type": robot_type,
                "root_pos": root_pos,
                "root_rot_xyzw": root_rot_xyzw,
                "dof_pos": dof_pos,
                "input_fps": input_fps,
            }
        )

    robot_types = {job["robot_type"] for job in prepared_jobs}
    if len(robot_types) > 1:
        raise ValueError(
            "Batch conversion currently requires all pkls in one run to have the same robot_type. "
            f"Found: {sorted(robot_types)}"
        )

    manifest = []
    for index, job in enumerate(prepared_jobs, start=1):
        print(
            f"[INFO]: Converting {index}/{len(jobs)}: {job['input_path']} "
            f"(frames={job['root_pos'].shape[0]}, input_fps={job['input_fps']})"
        )
        csv_path = save_motion_csv(
            job["output_path"],
            root_pos=job["root_pos"],
            root_rot_xyzw=job["root_rot_xyzw"],
            dof_pos=job["dof_pos"],
        )

        manifest.append(
            {
                "input_file": str(job["input_path"]),
                "output_file": str(csv_path),
                "csv_file": str(csv_path),
                "relative_path": job["relative_path"],
                "robot_type": job["robot_type"],
                "frames": int(job["root_pos"].shape[0]),
                "input_fps": job["input_fps"],
            }
        )

    if args.input_dir:
        manifest_path = Path(args.manifest_path) if args.manifest_path else Path(args.output_dir) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[INFO]: Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
