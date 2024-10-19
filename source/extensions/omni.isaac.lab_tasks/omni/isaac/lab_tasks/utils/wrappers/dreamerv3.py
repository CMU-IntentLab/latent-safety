# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RL-Games vectorized environment.

The following example shows how to wrap an environment for RL-Games and register the environment construction
for RL-Games :class:`Runner` class:

.. code-block:: python

    from rl_games.common import env_configurations, vecenv

    from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    # configuration parameters
    rl_device = "cuda:0"
    clip_obs = 10.0
    clip_actions = 1.0

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

"""

# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import gymnasium
import torch
import numpy as np
from rl_games.common import env_configurations
from rl_games.common.vecenv import IVecEnv
import matplotlib.pyplot as plt
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv, VecEnvObs

"""
Vectorized environment wrapper.
"""


class DreamerVecEnvWrapper(IVecEnv):
    """Wraps around Isaac Lab environment for RL-Games.

    This class wraps around the Isaac Lab environment. Since RL-Games works directly on
    GPU buffers, the wrapper handles moving of buffers from the simulation environment
    to the same device as the learning agent. Additionally, it performs clipping of
    observations and actions.

    For algorithms like asymmetric actor-critic, RL-Games expects a dictionary for
    observations. This dictionary contains "obs" and "states" which typically correspond
    to the actor and critic observations respectively.

    To use asymmetric actor-critic, the environment observations from :class:`ManagerBasedRLEnv`
    must have the key or group name "critic". The observation group is used to set the
    :attr:`num_states` (int) and :attr:`state_space` (:obj:`gym.spaces.Box`). These are
    used by the learning agent in RL-Games to allocate buffers in the trajectory memory.
    Since this is optional for some environments, the wrapper checks if these attributes exist.
    If they don't then the wrapper defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.


    Reference:
        https://github.com/Denys88/rl_games/blob/master/rl_games/common/ivecenv.py
        https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
    """

    def __init__(self, env: ManagerBasedRLEnv, device: str):
        """Initializes the wrapper instance.

        Args:
            env: The environment to wrap around.
            rl_device: The device on which agent computations are performed.
            clip_obs: The clipping value for observations.
            clip_actions: The clipping value for actions.

        Raises:
            ValueError: The environment is not inherited from :class:`ManagerBasedRLEnv`.
            ValueError: If specified, the privileged observations (critic) are not of type :obj:`gym.spaces.Box`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")
        # initialize the wrapper
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self.size = (128,128)
        self.ac_lim = 0.15
        self.device = device

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
            pol_space = gym.spaces.Box(spaces['policy'].low, spaces['policy'].high, spaces['policy'].shape, dtype=spaces['policy'].dtype)
            img_space = gym.spaces.Box(np.zeros_like(spaces['image'].low), 255*np.ones_like(spaces['image'].high), spaces['image'].shape, dtype='uint8')
            spaces = {"policy": pol_space, "image": img_space}

        else:
            spaces = {self._obs_key: self._env.observation_space}
            

        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )
    @property
    def single_observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
            pol_space = gym.spaces.Box(spaces['policy'].low[0], spaces['policy'].high[0], spaces['policy'].shape[1:], dtype=spaces['policy'].dtype)
            new_spaces = {"policy": pol_space}
            for k in spaces["image"].keys():
                img_shape = [*spaces['image'][k].shape[1:]]
                img_shape[-1] = 3
                img_space = gym.spaces.Box(np.zeros_like(spaces['image'][k].low)[0, :, :, :3], 255*np.ones_like(spaces['image'][k].high)[0, :, :, :3], img_shape, dtype='uint8')
                
                new_spaces[k] = img_space

        else:
            new_spaces = {self._obs_key: self._env.observation_space}
            

        return gym.spaces.Dict(
            {
                **new_spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )
        '''return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )'''

    @property
    def action_space(self):
        space = self._env.action_space
        space.low = -self.ac_lim*np.ones_like(space.low)
        space.high = self.ac_lim*np.ones_like(space.high)
        #space.discrete = True
        return space
    
    @property
    def single_action_space(self):
        env_low = -self.ac_lim*np.ones_like(self.action_space.low[0,:])
        env_high = self.ac_lim*np.ones_like(self.action_space.high[0,:])
        space =  gym.spaces.Box(env_low, env_high, dtype=np.float32)
        #acts = env.unwrapped.single_action_space #train_envs[0].action_space
        #acts.low = np.ones_like(acts.low) * -1
        #acts.high = np.ones_like(acts.high) # need to normalize actions 
        return space
    

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self._env.unwrapped

    @property
    def num_envs(self) -> int:
        """Returns the number of sub-environment instances."""
        return self.unwrapped.num_envs

    def step(self, action):
        action = torch.where(action>self.ac_lim, self.ac_lim, action)
        action = torch.where(action<-self.ac_lim, -self.ac_lim, action)
        obs, reward, terminated, truncated, info = self._env.step(action)
        for k in obs["image"].keys():
            obs[k] = obs["image"][k][..., :3] #(255*(obs["image"][k][..., :3]+1)/2).int() #obs["image"][k][..., :3]
            '''for env in range(self.num_envs):
                if 'wrist' in k:
                    plt.imshow(obs[k][env].cpu().numpy())
                    plt.savefig('test_wrist'+str(env)+'.png')
                    plt.close()
                else:
                    plt.imshow(obs[k][env].cpu().numpy()) 
                    plt.savefig('test_front'+str(env)+'.png')
                    plt.close()'''
        obs.pop("image")
        done = terminated | truncated
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs["is_first"] = torch.tensor([0] * self.num_envs).to(done.device)
        obs["is_last"] = done.int()
        #obs["is_terminal"] = info.get("is_terminal", False)
        obs["is_terminal"] = terminated.int()
        return obs, reward, done, info

    def reset(self):
        obs, info = self._env.reset()
        
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        for k in obs["image"].keys():
            obs[k] = (255*(obs["image"][k][..., :3]+1)/2).int()

        obs.pop("image")
        obs["is_first"] = torch.tensor([1] * self.num_envs).to(obs['policy'].device)
        obs["is_last"] = torch.tensor([0] * self.num_envs).to(obs['policy'].device)
        obs["is_terminal"] = torch.tensor([0] * self.num_envs).to(obs['policy'].device)
        return obs

   