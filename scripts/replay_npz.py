"""Replay converted robot motions from npz files."""

import argparse
import datetime
import os
import pathlib
import subprocess

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted robot motions from npz files.")
parser.add_argument("--motion", type=str, default=None, help="Path to the motion npz file.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wand registry.")
parser.add_argument(
    "--robot",
    type=str,
    default="auto",
    choices=("auto", "k1", "t1"),
    help="Robot type to replay. Use 'auto' to infer from the motion file.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record replay video to mp4.")
parser.add_argument("--video_length", type=int, default=300, help="Number of replay steps to record.")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save replay video outputs.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import omni.replicator.core as rep
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from booster_assets.motions import K1_JOINT_NAMES, T1_JOINT_NAMES
from booster_train.assets.robots.booster import BOOSTER_K1_CFG, BOOSTER_T1_CFG
from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionLoader


ROBOT_CONFIGS = {
    "k1": {
        "cfg": BOOSTER_K1_CFG,
        "joint_names": K1_JOINT_NAMES,
        "anchor_body_name": "Trunk",
    },
    "t1": {
        "cfg": BOOSTER_T1_CFG,
        "joint_names": T1_JOINT_NAMES,
        "anchor_body_name": "Trunk",
    },
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class ReplayVideoRecorder:
    def __init__(self, output_dir: str, motion_name: str, fps: float, resolution: tuple[int, int] = (1280, 720)):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.stem = f"{motion_name}_{timestamp}"
        self.frame_dir = self.output_dir / f"{self.stem}_frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / f"{self.stem}.mp4"
        self.fps = max(float(fps), 1.0)
        self.resolution = resolution
        self._render_product = rep.create.render_product("/OmniverseKit_Persp", self.resolution)
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._rgb_annotator.attach([self._render_product])
        self.frame_count = 0

    def capture_frame(self):
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        if rgb_data.size == 0:
            rgb_data = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        else:
            rgb_data = rgb_data[:, :, :3]
        frame_path = self.frame_dir / f"{self.frame_count:06d}.png"
        from PIL import Image

        Image.fromarray(rgb_data).save(frame_path)
        self.frame_count += 1

    def finalize(self) -> str:
        if self.frame_count == 0:
            raise RuntimeError("No frames were captured for the replay video.")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{self.fps:.6f}",
            "-i",
            str(self.frame_dir / "%06d.png"),
            "-pix_fmt",
            "yuv420p",
            str(self.video_path),
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(self.video_path)


def _resolve_motion_file() -> str:
    if args_cli.motion:
        motion_file = args_cli.motion
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        return motion_file

    if args_cli.registry_name:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required when using --registry_name. Install it with: pip install wandb"
            ) from exc

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        return str(pathlib.Path(artifact.download()) / "motion.npz")

    raise ValueError("Either --motion or --registry_name must be provided.")


def _infer_robot_type(motion_file: str) -> str:
    with np.load(motion_file, allow_pickle=True) as data:
        if "joint_names" not in data:
            raise KeyError("Motion file missing 'joint_names'; pass --robot explicitly.")
        joint_names = [str(name) for name in data["joint_names"].tolist()]

    joint_name_set = set(joint_names)
    k1_joint_name_set = set(K1_JOINT_NAMES)
    t1_joint_name_set = set(T1_JOINT_NAMES)

    if joint_name_set == t1_joint_name_set:
        return "t1"
    if joint_name_set == k1_joint_name_set:
        return "k1"

    if len(joint_names) == len(T1_JOINT_NAMES) and "Waist" in joint_name_set:
        return "t1"
    if len(joint_names) == len(K1_JOINT_NAMES) and "Waist" not in joint_name_set:
        return "k1"

    raise ValueError(
        "Unable to infer robot type from motion file. "
        f"joint_names={joint_names}. Pass --robot explicitly."
    )


def _read_motion_fps(motion_file: str) -> float:
    with np.load(motion_file, allow_pickle=True) as data:
        if "fps" not in data:
            return 50.0
        return float(np.asarray(data["fps"]).reshape(-1)[0])


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    motion_file: str,
    anchor_body_name: str,
    video_recorder: ReplayVideoRecorder | None = None,
):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        motion_file,
        robot.body_names,
        robot.joint_names,
        default_motion_body_names=robot.body_names,
        default_motion_joint_names=robot.joint_names,
        tail_len=0,
        device=str(sim.device),
    )
    anchor_body_index = robot.body_names.index(anchor_body_name)
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    recorded_steps = 0

    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, anchor_body_index]
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, anchor_body_index]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, anchor_body_index]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, anchor_body_index]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        sim.render()
        scene.update(sim_dt)

        if video_recorder is not None:
            video_recorder.capture_frame()
            recorded_steps += 1
            if recorded_steps >= args_cli.video_length:
                break


def main():
    motion_file = _resolve_motion_file()
    robot_type = args_cli.robot if args_cli.robot != "auto" else _infer_robot_type(motion_file)
    robot_info = ROBOT_CONFIGS[robot_type]

    print(f"[INFO] Replaying motion: {motion_file}")
    print(f"[INFO] Robot type: {robot_type}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_info["cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    video_recorder = None
    if args_cli.video:
        motion_name = pathlib.Path(motion_file).stem
        motion_fps = _read_motion_fps(motion_file)
        video_dir = (
            pathlib.Path(args_cli.video_dir)
            if args_cli.video_dir is not None
            else pathlib.Path(motion_file).resolve().parent / "videos" / "replay"
        )
        video_recorder = ReplayVideoRecorder(
            output_dir=str(video_dir),
            motion_name=motion_name,
            fps=motion_fps,
        )
        print(f"[INFO] Recording replay video to: {video_dir}")

    run_simulator(
        sim,
        scene,
        motion_file,
        anchor_body_name=robot_info["anchor_body_name"],
        video_recorder=video_recorder,
    )

    if video_recorder is not None:
        video_path = video_recorder.finalize()
        print(f"[INFO] Replay video saved to: {video_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
