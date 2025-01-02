from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

"""
This is the drone racing example in our paper

"""



class Double_Drones_RA_linear_Game_Env6(gym.Env):
    # we learn a single target set!
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    # lessons: keep each dimension speed same scale. Otherwise, early truncation
    
    def __init__(self):
        self.high = np.array([  1,  1,  0,  1.0,  1,  1,
                                1,  1,  0,  0.5,  1,  1,
                            ], dtype=np.float32) # x1 vx1 y1 vy1 z1 vz1          x2 vx2 y2 vy2 z2 vz2
        self.low  = np.array([  -1, -1, -3.2, 0.1, -1, -1,
                                -1, -1, -3.2, 0.1, -1, -1,
                            ], dtype=np.float32)
        self.gate_width = 0.1
        self.safe_cone_radius = 0.2
        self.action1_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32) # control action space
        self.action2_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32) # disturbance action space
        self.action_space = spaces.Box(low = -1, high = 1, shape = (6,), dtype=np.float32) # joint action space

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
    def step(self, act, evaluate_state = None):
        # local coordinate system. 
        # gate is [0,0,0]
        if evaluate_state is None:
            state = self.state
        else:
            state = evaluate_state
        

        distance = np.sqrt((state[6])**2+(state[8])**2+(state[10])**2)
        func_scale = 10.0 # previous 10
        
        
        reward = func_scale*min([
            state[2] - state[8],
            state[3] - state[9],
            (state[0] - -0.3),
            (0.3 - state[0]),
            (state[4] - -0.3),
            (0.3 - state[4])
        ])
        
        # safety cone of the ego drone:
        cone1 = np.sqrt((state[0] - state[6])**2 + (state[2] - state[8])**2) - (1+max(0,(state[10]-state[4])))*self.safe_cone_radius
        
        
        stay_with_in_left_fence = (state[0]+self.gate_width/2) - state[2]
        stay_with_in_right_fence = (-state[0]+self.gate_width/2) - state[2]
        stay_with_in_upper_fence = (-state[2]+self.gate_width/2) - state[4]
        stay_with_in_lower_fence = state[4] - (state[2]-self.gate_width/2)
        
        const = func_scale*min([
            cone1, # combined with the later one
            self.high[3] - state[3],
            stay_with_in_left_fence, 
            stay_with_in_right_fence,
            stay_with_in_upper_fence,
            stay_with_in_lower_fence
        ])
        dt = 0.1
        
        # ------- below is the closed-loop control of the other drone before adding disturbance -------
        control_gain_1 = 0.5 # scaling control for the ego drone
        control_gain_2 = 1 # scaling control for the other drone
        disturbance_gain = 0.1 # scaling disturbance for the other drone
        K1 = np.array([3.1127])
        K2 = np.array([ 9.1704,   16.8205])
        x_star = [
            0.0, 0.0, 
            0.0, 0.3, 
            0.0, 0.0
        ] # target velocity for the PID controller of the other drone
        act_other = np.array([
            K2@np.array([x_star[0]-state[0+6], x_star[1]-state[1+6]]),
            K1@np.array([x_star[3]-state[3+6]]),
            K2@np.array([x_star[4]-state[4+6], x_star[5]-state[5+6]]),
        ])
        # ------- above is the closed-loop control part -------
        # dynamics of the ego drone:
        self.state[0] = self.state[0] + dt * self.state[1]
        self.state[1] = self.state[1] + dt * control_gain_1*act[0]
        
        self.state[2] = self.state[2] + dt * self.state[3]
        self.state[3] = self.state[3] + dt * control_gain_1*act[1]
        
        self.state[4] = self.state[4] + dt * self.state[5]
        self.state[5] = self.state[5] + dt * control_gain_1*act[2]
        # dynamics of the other drone under disturbance:
        self.state[6] = self.state[6] + dt * self.state[7]
        self.state[7] = self.state[7] + dt * control_gain_2*act_other[0] + dt * disturbance_gain*act[3]
        self.state[8] = self.state[8] + dt * self.state[9]
        self.state[9] = self.state[9] + dt * control_gain_2*act_other[1] + dt * disturbance_gain*act[4] #dt * (-0.9170*self.state[8]-1.6821*self.state[9] )
        self.state[10] = self.state[10] + dt * self.state[11]
        self.state[11] = self.state[11] + dt * control_gain_2*act_other[2] + dt * disturbance_gain*act[5]
        
        
        
        # if state is outside the range, then done
        Done = True if min(self.state[[0,2,4,6,8,10]] - self.low[[0,2,4,6,8,10]]) < -0 or max(self.state[[0,2,4,6,8,10]] - self.high[[0,2,4,6,8,10]]) > 0 else False
        if abs(self.state[1]) > 30 or abs(self.state[3]) > 30 or abs(self.state[5]) > 30:
            Done = True
        
        return self.state.astype(np.float32), reward.astype(np.float32), Done, False, {"constraint":const}

    def reset(self, initial_state=np.array([]), seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # return initial_state if it is assigned a value
        if len(initial_state) == 0:
            # high = np.array([   30, 4, 0, 4, 40, 4, 
            #                     30, 4, 0, 4, 40, 4,
            #                 ], dtype=np.float32) # x1 vx1 y1 vy1 x2 vx2 y2 vy2 x3 vx3 y3 vy3 x4 vx4 y4 vy4
            # low  = np.array([   -30, -4, -80, -4, -20, -4, 
            #                     -30, -4, -80, -4, -20, -4,
            #                 ], dtype=np.float32)
            high=self.high
            low=self.low

            self.state = np.random.uniform(low, high, (12)).astype(np.float32)
            
        else:
            self.state = initial_state.astype(np.float32)

        
        return self.state, {}

    def render(self):
        return {}










