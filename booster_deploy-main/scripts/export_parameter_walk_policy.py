from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re

import torch
import yaml


class ActorCritic(torch.nn.Module):
    def __init__(self, num_act: int, num_obs: int, num_privileged_obs: int):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(
            torch.full((1, num_act), fill_value=-2.0),
            requires_grad=True,
        )


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"model_(\d+)\s*\.pth$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def find_checkpoint(gym_root: Path, latest_mode: str) -> Path:
    patterns = [
        str(gym_root / "logs" / "**" / "Parameter_Walk" / "**" / "*.pth"),
        str(gym_root / "logs" / "**" / "*.pth"),
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found under {gym_root / 'logs'}")

    if latest_mode == "mtime":
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_gym_root = Path("/home/zbc/booster_t1_real/htwk-gym-zsp/htwk-gym-main")
    default_output = repo_root / "tasks" / "parameter_walk" / "models" / "parameter_walk.pt"

    parser = argparse.ArgumentParser(
        description="Export a htwk-gym T1 Parameter_Walk .pth checkpoint to a TorchScript policy for booster_deploy."
    )
    parser.add_argument("--gym-root", type=Path, default=default_gym_root)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--latest-mode",
        choices=["step", "mtime"],
        default="step",
        help="How to choose a checkpoint when --checkpoint is not given.",
    )
    args = parser.parse_args()

    gym_root = args.gym_root.resolve()
    cfg_path = args.config or (gym_root / "envs" / "T1" / "Parameter_Walk.yaml")
    checkpoint_path = args.checkpoint or find_checkpoint(gym_root, args.latest_mode)
    output_path = args.output.resolve()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    model = ActorCritic(
        num_act=int(cfg["env"]["num_actions"]),
        num_obs=int(cfg["env"]["num_observations"]),
        num_privileged_obs=int(cfg["env"]["num_privileged_obs"]),
    )
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    scripted_actor = torch.jit.script(model.actor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_actor.save(str(output_path))

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded config:     {cfg_path}")
    print(f"Saved policy:      {output_path}")
    print("Deploy with:       python3 scripts/deploy.py --task parameter_walk")


if __name__ == "__main__":
    main()
