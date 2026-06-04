from pathlib import Path

from isaaclab.utils import configclass

try:
    from booster_assets import BOOSTER_ASSETS_DIR
except ImportError:
    BOOSTER_ASSETS_DIR = None

from booster_train.assets.robots.booster import BOOSTER_T1_CFG as ROBOT_CFG, T1_ACTION_SCALE
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .tracking_env_cfg import TrackingEnvCfg


T1_VIDEO_012_BODY_NAMES = [
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


def _resolve_motion_file() -> str:
    if BOOSTER_ASSETS_DIR is not None:
        return f"{BOOSTER_ASSETS_DIR}/motions/T1/video_012.npz"

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "booster_assets-main" / "motions" / "T1" / "video_012.npz"
        if candidate.exists():
            return str(candidate)

    # Keep the fallback path stable even if the file is not present yet.
    return str(current.parents[9] / "booster_assets-main" / "motions" / "T1" / "video_012.npz")


T1_VIDEO_012_MOTION_FILE = _resolve_motion_file()


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        self.commands.motion.motion_manifest = None
        self.commands.motion.motion_files = None
        self.commands.motion.motion_file = T1_VIDEO_012_MOTION_FILE
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.tail_len = 0
        self.commands.motion.adaptive_uniform_ratio = 0.1
        self.commands.motion.body_names = T1_VIDEO_012_BODY_NAMES


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=5,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),
                    noise_step=0.005,
                    border_width=0.25,
                ),
            },
        )


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None


@configclass
class FlatLowFreqWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
