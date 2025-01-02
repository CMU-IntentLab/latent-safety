from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math
from torch.utils.data import DataLoader



class Franka_DINOWM_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self, params):
        self.render_mode = None
        self.time_step = 0.05
        self.high = np.array([
            1., 1., np.pi,
        ])
        self.low = np.array([
            -1., -1., -np.pi
        ])
        self.device = 'cuda:0'

        self.set_wm(*params)

        #self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,1,786), dtype=np.float32)
        self.action1_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32) # control action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32) # joint action space
        self.scalar = 0.15
        self.image_size=128
        self.N = 5 # number of samples to take

        self.front_hist = None
        self.wrist_hist = None
        self.state_hist = None

    def _reset_loader(self):
        self.data = iter(DataLoader(self.dataset, batch_size=1, shuffle=True))

    def set_wm(self, wm, past_data):
        self.wm = wm.to(self.device)
        self.dataset = past_data
        self._reset_loader()

    
    def step(self, action):

        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(self.device)#*self.scalar

        self.ac_hist = torch.cat([self.ac_hist[:,1:], ac_torch], dim=1)
        rew = np.inf
        latent = self.wm.forward_features(self.front_hist, self.wrist_hist, self.state_hist, self.ac_hist)


        inp1, inp2, state = self.wm.front_head(latent), self.wm.wrist_head(latent), self.wm.state_pred(latent)
        self.front_hist = torch.cat([self.front_hist[:,1:], inp1[:,[-1]]], dim=1)
        self.wrist_hist = torch.cat([self.wrist_hist[:,1:], inp2[:,[-1]]], dim=1)
        self.state_hist = torch.cat([self.state_hist[:,1:], state[:,[-1]]], dim=1)

        rew = self.safety_margin(latent)  # rew is negative if unsafe
        self.latent = latent[:,[-1]].mean(dim=2).detach().cpu().numpy() 
        terminated = False
        truncated = False
        info = {"is_first":False, "is_terminal":terminated}
        return np.copy(self.latent), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        try:
            data = next(self.data)
        except StopIteration:
            # Reset the DataLoader and reshuffle
            self._reset_loader()
            data= next(self.data)
        inputs2 = data['cam_rs_embd'][[0], :].to(self.device)
        inputs1 = data['cam_zed_right_embd'][[0], :].to(self.device)
        acs = data['action'][[0],:].to(self.device)
        states = data['state'][[0],:].to(self.device)
        
        self.latent = self.wm.forward_features(inputs1, inputs2, states, acs)[:, [0]]
        inp1, inp2, state = self.wm.front_head(self.latent), self.wm.wrist_head(self.latent), self.wm.state_pred(self.latent)
        self.front_hist = torch.cat([inputs1[:,1:], inp1[:,[-1]]], dim=1)
        self.wrist_hist = torch.cat([inputs2[:,1:], inp2[:,[-1]]], dim=1)
        self.state_hist = torch.cat([states[:,1:], state[:,[-1]]], dim=1)
        self.ac_hist = acs

        return np.copy(self.latent.mean(dim=2).detach().cpu().numpy()), {"is_first": True, "is_terminal": False}
      

    def safety_margin(self, latent):
        g_xList = []
        
        with torch.no_grad():  # Disable gradient calculation

            outputs = torch.tanh(1.5*self.wm.failure_pred(latent)[0,-1])
            g_xList.append(outputs.detach().cpu().numpy())
        
        safety_margin = np.array(g_xList).squeeze()

        return safety_margin
    
