from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def load_motion_pkl(path: str | Path) -> dict:
    """Load a video2robot/GMR motion pickle with NumPy 1.x compatibility aliases."""

    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_motion_csv(
    input_file: str | Path,
    *,
    line_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a motion CSV in the common beyondmimic layout.

    CSV format:
    - columns 0:3   -> root position xyz
    - columns 3:7   -> root quaternion xyzw
    - columns 7:end -> joint positions
    """

    input_path = Path(input_file)
    if line_range is None:
        motion_np = np.loadtxt(input_path, delimiter=",")
    else:
        motion_np = np.loadtxt(
            input_path,
            delimiter=",",
            skiprows=line_range[0] - 1,
            max_rows=line_range[1] - line_range[0] + 1,
        )

    motion_np = np.atleast_2d(np.asarray(motion_np, dtype=np.float32))
    if motion_np.ndim != 2 or motion_np.shape[1] < 8:
        raise ValueError(f"Invalid motion CSV shape {motion_np.shape} for {input_path}")

    return motion_np


def save_motion_csv(
    output_name: str | Path,
    *,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
) -> Path:
    """Save motion arrays into the common CSV layout expected by csv_to_npz.py."""

    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Invalid root_pos shape for CSV export: {root_pos.shape}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"Invalid root_rot shape for CSV export: {root_rot_xyzw.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"Invalid dof_pos shape for CSV export: {dof_pos.shape}")
    if not (root_pos.shape[0] == root_rot_xyzw.shape[0] == dof_pos.shape[0]):
        raise ValueError(
            "CSV export requires matching frame counts for root_pos/root_rot/dof_pos. "
            f"Got {root_pos.shape[0]}, {root_rot_xyzw.shape[0]}, {dof_pos.shape[0]}"
        )

    motion = np.zeros((root_pos.shape[0], 7 + dof_pos.shape[1]), dtype=np.float32)
    motion[:, :3] = root_pos
    motion[:, 3:7] = root_rot_xyzw
    motion[:, 7:] = dof_pos

    output_path = Path(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, motion, delimiter=",")
    print(f"[INFO]: Motion CSV saved to {output_path}")
    return output_path
