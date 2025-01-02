from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
class Dubins_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self):
        self.render_mode = None
        self.dt = 0.05
        self.high = np.array([
            1., 1., np.pi,
        ])
        self.low = np.array([
            -1., -1., -np.pi
        ])
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.action1_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # control action space
        #self.action2_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # disturbance action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # joint action space
        self.constraint = [0., 0., 0.5]
    def step(self, action):
        # assert len(action) == 2, "action should be a 1D array"

        self.state[0] = self.state[0] + self.dt * np.cos(self.state[2])
        self.state[1] = self.state[1] + self.dt * np.sin(self.state[2])
        self.state[2] = self.state[2] + self.dt * action[0]
        if self.state[2] > np.pi:
            self.state[2] -= 2*np.pi
        if self.state[2] < -np.pi:
            self.state[2] += 2*np.pi

        
        rew = ((self.state[0]-self.constraint[0])**2 + (self.state[1]-self.constraint[1])**2 - self.constraint[2]**2)
        
        
        terminated = False
        truncated = False
        if any(self.state[:2] > self.high[:2]) or any(self.state[:2] < self.low[:2]):
            terminated = True
        info = {}
        return self.state.astype(np.float32), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            self.state = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(3,))
        else:
            self.state = initial_state        
        return self.state, {}



