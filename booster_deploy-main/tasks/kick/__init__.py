from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .kick import T1KickingMimicControllerCfg


@configclass
class T1KickingMimicControllerCfg1(T1KickingMimicControllerCfg):
    """T1 kick mimic policy trained from beyond_mimic/video_009d."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/model_loser.pt"
        self.policy.motion_path = "motions/video_010.npz"


register_task("kicking_mimic", T1KickingMimicControllerCfg1())
register_task("t1_kicking_mimic", T1KickingMimicControllerCfg1())
