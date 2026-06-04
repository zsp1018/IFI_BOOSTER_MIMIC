import gymnasium as gym


gym.register(
    id="Booster-T1-Kick-v0",
    entry_point=f"{__name__}.history_env:HistoryPolicyObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-T1-Kick-Flat-v0",
    entry_point=f"{__name__}.history_env:HistoryPolicyObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-T1-Kick-LowFreq-v0",
    entry_point=f"{__name__}.history_env:HistoryPolicyObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatLowFreqWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:LowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Booster-T1-Kick-Play-v0",
    entry_point=f"{__name__}.history_env:HistoryPolicyObsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
