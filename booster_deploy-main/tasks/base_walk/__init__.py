from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .base_walk import T1BaseWalkControllerCfg


@configclass
class T1BaseWalkControllerCfg1(T1BaseWalkControllerCfg):
    """HTWK BaseWalk policy on the Booster locomotion control stack."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/base_walk.pt"


register_task("base_walk", T1BaseWalkControllerCfg1())
register_task("t1_base_walk", T1BaseWalkControllerCfg1())
