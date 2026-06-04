from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class HistoryPolicyObsEnv(ManagerBasedRLEnv):
    """Wrap policy observations into a fixed-length history stack.

    This keeps the critic / extras untouched and replaces ``obs["policy"]``
    with a flattened history buffer so training matches deploy-side
    ``obs_history.flatten()`` semantics.
    """

    policy_obs_history_length: int = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy_obs_history = None
        self._policy_obs_dim = None
        self._stacked_policy_obs_dim = None
        self._raw_compute_policy_obs = self.observation_manager.compute

        import gymnasium as gym
        import numpy as np

        old_space = self.single_observation_space["policy"]
        old_dim = old_space.shape[0]
        new_dim = old_dim * self.policy_obs_history_length
        self._stacked_policy_obs_dim = new_dim

        new_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(new_dim,),
            dtype=old_space.dtype,
        )

        self.single_observation_space["policy"] = new_space
        self.observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, new_dim),
            dtype=old_space.dtype,
        )
        if hasattr(self.observation_manager, "group_obs_dim"):
            self.observation_manager.group_obs_dim["policy"] = (new_dim,)
        self.observation_manager.compute = self._compute_with_policy_history

        print(f"[HistoryPolicyObsEnv] policy obs dim: {old_dim} -> {new_dim}")

    @property
    def num_obs(self) -> int:
        if self._stacked_policy_obs_dim is not None:
            return self._stacked_policy_obs_dim
        return super().num_obs

    @property
    def num_observations(self) -> int:
        return self.num_obs

    def _ensure_policy_history(self, obs: torch.Tensor) -> None:
        obs_dim = obs.shape[-1]
        if self._policy_obs_history is None or self._policy_obs_dim != obs_dim:
            self._policy_obs_dim = obs_dim
            self._policy_obs_history = torch.zeros(
                self.num_envs,
                self.policy_obs_history_length,
                obs_dim,
                dtype=obs.dtype,
                device=obs.device,
            )

    def _stack_policy_obs(self, policy_obs: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        self._ensure_policy_history(policy_obs)
        assert self._policy_obs_history is not None

        if reset_mask is not None and torch.any(reset_mask):
            self._policy_obs_history[reset_mask] = 0.0

        self._policy_obs_history = self._policy_obs_history.roll(shifts=-1, dims=1)
        self._policy_obs_history[:, -1] = policy_obs
        return self._policy_obs_history.flatten(start_dim=1)

    def _compute_with_policy_history(self, *args, **kwargs):
        obs_dict = self._raw_compute_policy_obs(*args, **kwargs)
        reset_mask = self.episode_length_buf == 0
        obs_dict["policy"] = self._stack_policy_obs(obs_dict["policy"], reset_mask=reset_mask)
        return obs_dict
