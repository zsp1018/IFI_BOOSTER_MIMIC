from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .t1_dance import T1DanceControllerCfg


@configclass
class T1DanceControllerCfg1(T1DanceControllerCfg):
    """T1 dance mimic policy aligned with booster_train ya T1 dance training."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/model.pt"
        self.policy.motion_path = "motions/video_012.npz"


register_task("t1_dance", T1DanceControllerCfg1())
register_task("T1_dance", T1DanceControllerCfg1())
