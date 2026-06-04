from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass
from .beyond_mimic import K1BeyondMimicControllerCfg


@configclass
class K1MJ2ControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_mj2_seg1.npz"
        self.policy.checkpoint_path = "models/k1_mj_dance_002_2025-12-03_00-10-28.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            4., 4., 4., 4.,
            4., 4., 4., 4.,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            1., 1., 1., 1.,
            1., 1., 1., 1.,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.
        ]
        self.robot.effort_limit = [
            6, 6,
            14, 14, 14, 14,
            14, 14, 14, 14,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


@configclass
class K1FightControllerCfg(K1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "motions/k1_fight_final_deploy.npz"
        self.policy.checkpoint_path = "models/k1_fight_001.pt"
        self.robot.joint_stiffness = [
            10.0, 10.0,
            3.95, 3.95, 3.95, 3.95,
            3.95, 3.95, 3.95, 3.95,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ]
        self.robot.joint_damping = [
            2., 2.,
            0.3, 0.3, 0.3, 0.3,
            0.3, 0.3, 0.3, 0.3,
            2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2.
        ]
        self.robot.effort_limit = [
            4, 4,
            12, 12, 12, 12,
            12, 12, 12, 12,
            30, 35, 20, 40, 20, 20,
            30, 35, 20, 40, 20, 20,
        ]


register_task("k1_mj2", K1MJ2ControllerCfg())
register_task("k1_fight", K1FightControllerCfg())
