from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
# action1 is the control
# action2 is the disturbance


class Highway_10D_game_Env2(gym.Env):
    # |car 0 (disturbance)  target
    # |               car2
    # |        car1           
    # imagine that we have two three cars. The target region is the right area
    # car0: x,y,v
    # car1: x,y,v, theta, (ego agent)
    # car2: x,y,v
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    def __init__(self):
        self.high = np.array([  1, 20, 2,
                                2, 20, 3, np.pi*3/4,
                                2, 20, 2, ], dtype=np.float32)
        self.low  = np.array([  0.0,  0.0,   0.5,
                                0.0,  0.0, 0.5, np.pi*1/4,
                                1.0,  0.0, 0.5], dtype=np.float32)
        self.action1_space = spaces.Box(low = -1, high = 1, shape = (2,), dtype=np.float32) # control action space
        self.action2_space = spaces.Box(low = -1, high = 1, shape = (2,), dtype=np.float32) # disturbance action space
        self.action_space = spaces.Box(low = -1, high = 1, shape = (4,), dtype=np.float32) # joint action space

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        # self.initial_condition_high = self.high
        # self.initial_condition_low  = self.low
        self.initial_condition_high = np.array([   1.0, 20, 2,
                            2, 20, 3.0, np.pi*3/4,
                            2, 10, 1.5], dtype=np.float32)
        self.initial_condition_low  = np.array([   0.0, 10, 0.5,
                            0.0, 0.0, 0.5, np.pi*1/4,
                            1.6, 0.0, 0.5], dtype=np.float32)

    def step(self, act):
        state = self.state
        reward = 10*min(#1 - np.sqrt(((state[3]-1.5)/0.5)**2  + ((state[6]-np.pi/2)/0.1)**2),
                        # self.state[7]-self.state[4], 
                        (self.state[3]-1.),
                        (state[4]-state[8]-1),
                        state[5] - state[9]-0.2) 
        # reward = 10*min(1 - np.sqrt(((state[3]-1.5)/0.5)**2  + ((state[6]-np.pi/2)/0.1)**2),
        #                 (state[4]-state[8]-1))
        
        const = 10*min((np.sqrt((state[3]-state[0])**2 + (state[4]-state[1])**2) - 0.5), # collision avoidance
                    (np.sqrt((state[3]-state[7])**2 + (state[4]-state[8])**2) - 0.5),
                    # (state[3] - self.low[3]),         # car 1's x position > left boundary of the road
                    # (self.high[3] - state[3]),        # car 1's x position < right boundary of the road
                    # (state[6] - self.low[6]-0.1),
                    # (self.high[6]+0.1 - state[6]),         
        )   
        dt = 0.1
        epsilon_d = 0.1
        # dynamics model:
        self.state[0] = state[0]  # x, slippery road
        self.state[1] = state[1] - dt * state[2]    # y, slippery road
        self.state[2] = state[2] + dt * epsilon_d*act[2]  # vy, uncertain acceleration
        
        self.state[3] = state[3] + dt * state[5] * np.cos(state[6]) # x, ego car # just approximate it by <= dt*state[5] & >= 0
        self.state[4] = state[4] + dt * state[5] * np.sin(state[6]) # y, ego car
        self.state[5] = state[5] + dt * 2*act[0] # v, ego car
        self.state[6] = state[6] + dt * (2*act[1] ) # theta, ego car
        
        self.state[7] = state[7] 
        self.state[8] = state[8] + dt * state[9]
        self.state[9] = state[9] + dt * epsilon_d*act[3]
        
        # if self.state[2]>self.high[2] or self.state[2]<self.low[2]:
        #     self.state[2] = np.clip(self.state[2], self.low[2], self.high[2])
        Done = False
        if self.state[3]>self.high[3]+0.1 or self.state[3]<self.low[3]-0.1 or self.state[4]>self.high[4]+100 or self.state[4]<self.low[4]-10:
            Done = True
        if self.state[1]>self.high[1]+10 or self.state[1]<self.low[1]-10 or self.state[8]>self.high[8]+10 or self.state[8]<self.low[8]-10:
            Done = True
        if self.state[2]>self.high[2] + 1 or self.state[2]<0 or self.state[5]>self.high[5] + 2 or self.state[5]<0:
            Done = True
        return self.state.astype(np.float32), reward.astype(np.float32), Done, False, {"constraint":const}

    def reset(self, initial_state=np.array([]), seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # return initial_state if it is assigned a value
        if len(initial_state) == 0:
            self.state = np.random.uniform(self.initial_condition_low, 
                                            self.initial_condition_high, 
                                            (10)).astype(np.float32)
        else:
            self.state = initial_state.astype(np.float32)
        return self.state, {}

    def render(self):
        return {}




