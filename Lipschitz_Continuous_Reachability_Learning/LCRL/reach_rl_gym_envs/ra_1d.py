from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
class LQR_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.01
        self.high = np.array([
            10.,
        ])
        self.low = np.array([
            -10.,
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action1_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # control action space
        self.action2_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # disturbance action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # joint action space

    def step(self, action):
        # assert len(action) == 2, "action should be a 1D array"
        self.state = (1+self.dt)*self.state + self.dt * (action[0]+ 0.5*action[1])
        # rew = self.state[0] - 2.0+ 2.5#-1.0
        rew = -(self.state[0]**2 - 2.0)
        if rew > 10:
            rew = 10
        if rew < -10:
            rew = -10
        constraint = self.state[0] - 1.0 +2.5
        if constraint > 10:
            constraint = 10.
        if constraint < -10:
            constraint = -10.
        
        rew = np.array([rew]).astype(np.float32)
        terminated = False
        truncated = False
        if any(self.state > self.high) or any(self.state < self.low):
            terminated = True
        info = {"constraint": np.array([10.0*constraint]).astype(np.float32)}
        return self.state.astype(np.float32), rew*10.0, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            self.state = np.random.uniform(low=-10.0, high=10.0, size=(1,))
        else:
            self.state = initial_state        
        return self.state, {}



