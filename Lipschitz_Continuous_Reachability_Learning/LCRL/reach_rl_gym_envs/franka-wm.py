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
class Franka_WM_Env(gym.Env):
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,1,1536,), dtype=np.float32)
        self.action1_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # control action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # joint action space
        self.scalar = 0.15
        self.image_size=128
        self.N = 5 # number of samples to take
    def set_wm(self, wm, past_data, config):
        self.encoder = wm.encoder.to(self.device)
        self.wm = wm.to(self.device)
        self.expl = past_data[0]
        self.teleop = past_data[1]

        
        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter
    

    def step(self, action):

        init = {k: v[:, -1] for k, v in self.latent.items()}
        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(self.device)#*self.scalar
        
        rew = np.inf
        for i in range(self.N):
            latent = self.wm.dynamics.imagine_with_action(ac_torch, init)
            rew_i = self.safety_margin(latent) # rew is negative if unsafe
            if rew_i < rew: # take most pessimistic transition
                rew = rew_i
                self.latent = latent
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()

        terminated = False
        truncated = False
        info = {"is_first":False, "is_terminal":terminated}
        return np.copy(self.feat), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        coin = torch.rand(1) 
        if coin > 0.5:
            init_traj = next(self.expl)
        else:
            init_traj = next(self.teleop)

        data = self.wm.preprocess(init_traj)
        embed = self.encoder(data)
        self.latent, _ = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        for k, v in self.latent.items(): 
            self.latent[k] = v[:, [-1]]

        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy() 

        return np.copy(self.feat), {"is_first": True, "is_terminal": False}
      

    def safety_margin(self, state):
        g_xList = []
        
        feat = self.wm.dynamics.get_feat(state).detach()
        with torch.no_grad():  # Disable gradient calculation

            # head outputs zero or one
            # failure = 1, safe = 0
            outputs = torch.tanh(self.wm.heads["failure"](feat))
            #outputs = -(self.wm.heads["failure"](feat).mode()*2 - 1) # want failure set to be negative
            g_xList.append(outputs.detach().cpu().numpy())
        
        safety_margin = np.array(g_xList).squeeze()

        return safety_margin
    '''
    def get_latent(self, xs, ys, thetas, imgs):
        states = np.expand_dims(np.expand_dims(thetas,1),1)
        imgs = np.expand_dims(imgs, 1)
        dummy_acs = np.zeros((np.shape(xs)[0], 1, 1))
        firsts = np.ones((np.shape(xs)[0], 1))
        lasts = np.zeros((np.shape(xs)[0], 1))
        
        cos = np.cos(states)
        sin = np.sin(states)

        states = np.concatenate([cos, sin], axis=-1)


        chunks = 21
        if np.shape(imgs)[0] > chunks:
            bs = int(np.shape(imgs)[0]/chunks)
        else:
            bs = int(np.shape(imgs)[0]/chunks)
        
        for i in range(chunks):
            if i == chunks-1:
                data = {'obs_state': states[i*bs:], 'image': imgs[i*bs:], 'action': dummy_acs[i*bs:], 'is_first': firsts[i*bs:], 'is_terminal': lasts[i*bs:]}
            else:
                data = {'obs_state': states[i*bs:(i+1)*bs], 'image': imgs[i*bs:(i+1)*bs], 'action': dummy_acs[i*bs:(i+1)*bs], 'is_first': firsts[i*bs:(i+1)*bs], 'is_terminal': lasts[i*bs:(i+1)*bs]}

        data = self.wm.preprocess(data)
        embeds = self.encoder(data)
        if i == 0:
            embed = embeds
        else:
            embed = torch.cat([embed, embeds], dim=0)


        data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
        data = self.wm.preprocess(data)
        post, prior = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        
        g_x = self.safety_margin(post)

        feat = self.wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()
        return g_x, feat, post'''