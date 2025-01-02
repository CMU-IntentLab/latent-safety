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
class Dubins_WM_Env(gym.Env):
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
        #self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(544,), dtype=np.float32)
        self.action1_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # control action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32) # joint action space
        self.constraint = [0., 0., 0.5]
        self.speed = 1
        self.device = 'cuda:0'
        self.set_wm(*params)
        self.image_size=128
    def set_wm(self, wm, lx, config):
        self.encoder = wm.encoder.to(self.device)
        self.MLP_margin = lx.to(self.device)
        self.wm = wm.to(self.device)
        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter
    def capture_image(self, state=None):
        """Captures an image of the current state of the environment."""
        # For simplicity, we create a blank image. In practice, this should render the environment.
        fig,ax = plt.subplots()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis('off')
        fig.set_size_inches( 1, 1 )
        # Create the circle patch
        circle = patches.Circle(self.constraint[:2], self.constraint[2], edgecolor=(1,0,0), facecolor='none')
        # Add the circle patch to the axis
        dt = self.time_step
        v = self.speed
        dpi=self.image_size
        ax.add_patch(circle)
        if state is None:
            plt.quiver(self.state[0], self.state[1], dt*v*np.cos(self.state[2]), dt*v*np.sin(self.state[2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)
        else:
            plt.quiver(state[0], state[1], dt*v*np.cos(state[2]), dt*v*np.sin(state[2]), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        #plt.savefig('logs/tests/test_rarl.png', dpi=dpi)
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)

        # Load the buffer content as an RGB image
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        plt.close()
        return img_array
    

    def step(self, action):

        init = {k: v[:, -1] for k, v in self.latent.items()}
        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(self.device)
        self.latent = self.wm.dynamics.imagine_with_action(ac_torch, init)

        # step gt state
        self.state = self.integrate_forward(self.state, action)
        g_x = self.safety_margin(self.latent)

        rew = g_x
        
        image = self.capture_image()
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()

        '''self.state[0] = self.state[0] + self.dt * np.cos(self.state[2])
        self.state[1] = self.state[1] + self.dt * np.sin(self.state[2])
        self.state[2] = self.state[2] + self.dt * action[0]
        if self.state[2] > np.pi:
            self.state[2] -= 2*np.pi
        if self.state[2] < -np.pi:
            self.state[2] += 2*np.pi'''

        
        #rew = ((self.state[0]-self.constraint[0])**2 + (self.state[1]-self.constraint[1])**2 - self.constraint[2]**2)
        
        if self.state[0]>1 or self.state[0]<-1 or self.state[1]>1 or self.state[1]<-1:
            terminated = True
        else:
            terminated = False
        #terminated = not self.check_within_bounds(self.state)
        truncated = False
        info = {"obs_state": np.copy(self.state[2]), "image": image, "is_first":False, "is_terminal":terminated}
        return np.copy(self.feat), rew, terminated, truncated, info
        #return {"state": np.copy(self.feat), "obs_state": np.copy(self.state[2]), "image": image, "is_first":False, "is_terminal":terminated}, rew, terminated, truncated, info
        #return self.state.astype(np.float32), rew, terminated, truncated, info
    
    def reset(self, initial_state=None,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if initial_state is None:
            self.state = np.random.uniform(low=[-1, -1, -np.pi], high=[1, 1, np.pi], size=(3,))
        else:
            self.state = initial_state  

        self.latent, self.state = self.sample_random_state()
        image = self.capture_image()
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy() 
        #return {"state": np.copy(self.feat), "obs_state": np.copy(self.state[2]), "image": image, "is_first": True, "is_terminal": False}, {}

        return np.copy(self.feat), {"obs_state": np.copy(self.state[2]), "image": image, "is_first": True, "is_terminal": False}
      
    def sample_random_state(
        self, sample_inside_obs=False, theta=None
    ):
        """Picks the state uniformly at random.

        Args:
            sample_inside_obs (bool, optional): consider sampling the state inside
                the obstacles if True. Defaults to False.
            sample_inside_tar (bool, optional): consider sampling the state inside
                the target if True. Defaults to True.
            theta (float, optional): if provided, set the initial heading angle
                (yaw). Defaults to None.

        Returns:
            np.ndarray: the sampled initial state.
        """
        # random sample `theta`
        if theta is None:
            theta_rnd = 2.0 * np.random.uniform() * np.pi
        else:
            theta_rnd = theta

        # random sample [`x`, `y`]
        flag = True
        while flag:
            rnd_state = np.random.uniform(low=[-1, -1], high=[1, 1], size=(2,))

            state0 = np.array([rnd_state[0], rnd_state[1], theta_rnd])
            img0 = self.get_image(rnd_state[:2], theta_rnd)
            
            state1 = self.integrate_forward(state0, 0)
            
            data = {'obs_state': [[[np.cos(state0[-1]), np.sin(state0[-1])]]], 'image': [[img0]], 'action': [[0.,]], 'is_first': np.array([[[True]]]), 'is_terminal': np.array([[[False]]])}

            data = self.wm.preprocess(data)
            embed = self.encoder(data)

            post, prior = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
                )
            #post = {k: v[:, [-1]] for k, v in post.items()}
            
            g_x = self.safety_margin(post)


            if (not sample_inside_obs) and (g_x < 0):
                flag = True
            else:
                flag = False
        
        return post.copy(), np.copy(state1)



    def integrate_forward(self, state, u):
        """Integrates the dynamics forward by one step.

        Args:
            state (np.ndarray): (x, y, yaw).
            u (float): the contol input, angular speed.

        Returns:
            np.ndarray: next state.
        """
        x, y, theta = state
        
        x = x + self.time_step * self.speed * np.cos(theta)
        y = y + self.time_step * self.speed * np.sin(theta)
        theta = np.mod(theta + self.time_step * u, 2 * np.pi)
        assert theta >= 0 and theta < 2 * np.pi
        state_next = np.array([x, y, theta.item()])
        
        return state_next
    
    def safety_margin(self, state):
        g_xList = []
        
        self.MLP_margin.eval()
        feat = self.wm.dynamics.get_feat(state).detach()
        with torch.no_grad():  # Disable gradient calculation
            outputs = self.MLP_margin(feat)
            g_xList.append(outputs.detach().cpu().numpy())
        
        safety_margin = np.array(g_xList).squeeze()

        return safety_margin
    

    def get_image(self, state, theta):
        x, y = state
        fig,ax = plt.subplots()
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
        plt.axis('off')
        dpi=128
        fig.set_size_inches( 1, 1 )
        # Create the circle patch
        circle = patches.Circle((0,0), (0.5), edgecolor=(1,0,0), facecolor='none')
        # Add the circle patch to the axis
        ax.add_patch(circle)
        plt.quiver(x, y, self.time_step*self.speed*math.cos(theta), self.time_step*self.speed*math.sin(theta), angles='xy', scale_units='xy', minlength=0,width=0.05, scale=0.2,color=(0,0,1), zorder=3)
        #plt.scatter(x, y, s=10, c=(0,0,1), zorder=3)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig('test.png', dpi=dpi)
        plt.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)

        # Load the buffer content as an RGB image
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        plt.close()
        return img_array
    
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
        return g_x, feat, post