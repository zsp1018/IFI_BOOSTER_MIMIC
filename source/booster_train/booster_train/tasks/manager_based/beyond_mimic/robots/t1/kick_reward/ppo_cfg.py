from isaaclab.utils import configclass

from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BaseLowFreqPPORunnerCfg,
    BasePPORunnerCfg,
)


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    save_interval = 500
    experiment_name = "t1_kick"


@configclass
class LowFreqPPORunnerCfg(BaseLowFreqPPORunnerCfg):
    max_iterations = 50000
    save_interval = 500
    experiment_name = "t1_kick_low_freq"
