"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

This module implements the parent class for the Dubins car environments, e.g.,
one car environment and pursuit-evasion game between two Dubins cars.
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from networks.mlp import MLP
from networks.cnn import ConvEncoderMLP


import numpy as np
from .env_utils import calculate_margin_circle, calculate_margin_rect
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math 

import io
from PIL import Image

class DubinsCarDyn(object):
  """
  This base class implements a Dubins car dynamical system as well as the
  environment with concentric circles. The inner circle is the target set
  boundary, while the outer circle is the boundary of the constraint set.
  """

  def __init__(self, config, doneType='toEnd'):
    """Initializes the environment with the episode termination criterion.

    Args:
        doneType (str, optional): conditions to raise `done` flag in
            training. Defaults to 'toEnd'.
    """
    # State bounds.
    self.bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]

    self.learned_margin = False
    self.learned_dyn = False
    self.image = False
    self.debug = False
    self.use_wm = False
    self.gt_lx = False


    # Dubins car parameters.
    self.alive = True
    self.time_step = config.dt
    self.speed = config.speed  # v

    # Control parameters.
    self.max_turning_rate = config.u_max # w
    self.R_turn = self.speed / self.max_turning_rate 
    self.discrete_controls = np.array([
        -self.max_turning_rate, 0., self.max_turning_rate
    ])

    # Constraint set parameters.
    self.constraint_center = None
    self.constraint_radius = None

    # Target set parameters.
    self.target_center = None
    self.target_radius = None

    # Internal state.
    self.state = np.zeros(3)
    self.doneType = doneType

    # Set random seed.
    self.seed_val = 0
    np.random.seed(self.seed_val)

    # Cost Params
    self.targetScaling = 1.
    self.safetyScaling = 1.

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.MLP_margin = MLP(3, 1, 256).to(self.device)
    self.MLP_margin.load_state_dict(torch.load('/home/kensuke/latent-safety/logs/classifier/failure_set.pth'))
    self.MLP_margin.eval()  # Set the model to evaluation mode
    self.MLP_dyn = MLP(4, 3, 256).to(self.device)
    self.MLP_dyn.load_state_dict(torch.load('/home/kensuke/latent-safety/logs/dynamics/dynamics.pth'))
    self.MLP_dyn.eval()  # Set the model to evaluation mode

  def set_encoder(self):
    self.image = True
    act = 'SiLU'
    norm = True
    cnn_depth = 32
    kernel_size = 4 
    minres = 4
    img_size = 128
    input_shape = (img_size, img_size, 3)
    x_dim = 128 # x, y, cos(theta), sin(theta)
    
    u_dim = 1
    hidden_dim = 256
    #encoder = ConvEncoder(input_shape, cnn_depth, act, norm, kernel_size, minres)
    self.encoder = ConvEncoderMLP(input_shape, cnn_depth, act, norm, kernel_size, minres, out_dim = x_dim, in_dim=1, hidden_dim=hidden_dim, hidden_layer=2).to(self.device)
    self.encoder.load_state_dict(torch.load('/home/kensuke/latent-safety/logs/dynamics_img/encoder_img.pth'))

    self.MLP_margin = MLP(x_dim, 1, hidden_dim).to(self.device)
    self.MLP_margin.load_state_dict(torch.load('/home/kensuke/latent-safety/logs/classifier_img/failure_set_img.pth'))

    self.MLP_dyn = MLP(x_dim+u_dim, x_dim, hidden_dim).to(self.device)
    self.MLP_dyn.load_state_dict(torch.load('/home/kensuke/latent-safety/logs/dynamics_img/dynamics_img.pth'))

  def set_wm(self, wm, lx, config):
    self.encoder = wm.encoder.to(self.device)
    self.MLP_margin = lx.to(self.device)
    self.MLP_dyn = wm.dynamics.to(self.device)
    self.wm = wm.to(self.device)
    self.use_wm = True
    if config.wm:
      if config.dyn_discrete:
        self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
      else:
        self.feat_size = config.dyn_stoch + config.dyn_deter


  def reset(
      self, start=None, theta=None, sample_inside_obs=False,
      sample_inside_tar=True
  ):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): the state to reset the Dubins car to. If
            None, pick the state uniformly at random. Defaults to None.
        theta (float, optional): if provided, set the initial heading angle
            (yaw). Defaults to None.
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.

    Returns:
        np.ndarray: the state that Dubins car has been reset to.
    """
    if not self.use_wm:
      if start is None:
        x_rnd, y_rnd, theta_rnd = self.sample_random_state(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar, theta=theta
        )
        self.state = np.array([x_rnd, y_rnd, theta_rnd])
      else:
        self.state = start
      return np.copy(self.state)
    else:
      latent, gt_state = self.sample_random_state(
            sample_inside_obs=sample_inside_obs,
            sample_inside_tar=sample_inside_tar, theta=theta
        )
      self.state = gt_state
      self.latent = latent
      return latent.copy(), np.copy(gt_state)

  def get_latent(self, xs, ys, thetas, imgs):
    states = np.expand_dims(np.expand_dims(thetas,1),1)
    imgs = np.expand_dims(imgs, 1)
    dummy_acs = np.zeros((np.shape(xs)[0], 1, 3))
    rand_idx = 1 #go straight #np.random.randint(0, 3, np.shape(xs)[0])
    dummy_acs[np.arange(np.shape(xs)[0]), :, rand_idx] = 1
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

    
    #data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
#
    #data = self.wm.preprocess(data)
    #embed = self.encoder(data)
    data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
    data = self.wm.preprocess(data)
    post, prior = self.wm.dynamics.observe(
        embed, data["action"], data["is_first"]
        )
    if self.gt_lx:
      raise Exception('Not implemented')
      #g_x = self.gt_safety_margin(np.array([xs, ys]))
    else:
      g_x = self.safety_margin(post)

    feat = self.wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()
    return g_x, feat, post

  def sample_random_state(
      self, sample_inside_obs=False, sample_inside_tar=True, theta=None
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
      rnd_state = np.random.uniform(low=self.low[:2], high=self.high[:2])
      if self.image:
        img = self.get_image(rnd_state, theta_rnd)
        img = torch.tensor([img/256.]).float().to(self.device)
        embed = self.encoder(img, torch.tensor([theta_rnd]).float().to(self.device)).detach().cpu().numpy().squeeze()
        g_x = self.safety_margin(embed)
      if self.use_wm:
        state0 = np.array([rnd_state[0], rnd_state[1], theta_rnd])
        img0 = self.get_image(rnd_state, theta_rnd)
        rand_u0 = np.zeros(3)
        rand_int0 = 1 #np.random.randint(0, 3)
        rand_u0[rand_int0] = 1  
        state1 = self.integrate_forward(state0, self.discrete_controls[rand_int0])

        img1 = self.get_image(state1[:2], state1[2])
        rand_u1 = np.zeros(3)
        rand_int1 = np.random.randint(0, 3)
        rand_u1[rand_int1] = 1  

        state2 = self.integrate_forward(state1, self.discrete_controls[rand_int1])
        #img2 = self.get_image(state2[:2], state2[2])
        #rand_u2 = np.zeros(3)
        #rand_int2 = np.random.randint(0, 3)
        #rand_u2[rand_int2] = 1  

        #data = {'obs_state': [[[state0[-1]], [state1[-1]]]], 'image': [[img0, img1]], 'action': [[rand_u0, rand_u1]], 'is_first': np.array([[[True], [False]]]), 'is_terminal': np.array([[[False], [False]]])}
        #for k, v in data.items():
        #  print(np.shape(data[k]))
        data = {'obs_state': [[[np.cos(state0[-1]), np.sin(state0[-1])]]], 'image': [[img0]], 'action': [[rand_u0]], 'is_first': np.array([[[True]]]), 'is_terminal': np.array([[[False]]])}

        data = self.wm.preprocess(data)
        embed = self.encoder(data)

        post, prior = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
            )
        #post = {k: v[:, [-1]] for k, v in post.items()}
        
        if self.gt_lx:
          raise Exception('Not implemented')
          g_x = self.gt_safety_margin(state1)
        else:
          g_x = self.safety_margin(post)

      else:
        #l_x = self.target_margin(rnd_state)
        g_x = self.safety_margin(rnd_state)

      

      #if (not sample_inside_obs) and (g_x > 0):
      if (not sample_inside_obs) and (g_x < 0):
        flag = True
      #elif (not sample_inside_tar) and (l_x <= 0):
      #  flag = True
      else:
        flag = False
    x_rnd, y_rnd = rnd_state
    if self.image:
      x_rnd, y_rnd, theta_rnd = embed[0], embed[1], embed[2]
    if self.use_wm:
      return post, state1
    else:
      return x_rnd, y_rnd, theta_rnd

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

  # == Dynamics ==
  def step(self, action):
    """Evolves the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        bool: True if the episode is terminated.
    """
    if type(action) == dict:
      action = np.argmax(action[ 'action' ])

    # step latent state
    if self.use_wm:
      if self.gt_lx:
        raise Exception('Not implemented')
        g_x_cur = self.gt_safety_margin(self.state[:2])
      else:
        g_x_cur = self.safety_margin(self.latent)

      img_ac = torch.zeros(3).to(self.device)
      img_ac[action] = 1
      img_ac = img_ac.unsqueeze(0).unsqueeze(0)
      
      init = {k: v[:, -1] for k, v in self.latent.items()}
      self.latent = self.wm.dynamics.imagine_with_action(img_ac, init)
    else:
      g_x_cur = self.safety_margin(self.state[:2])

    # step gt state
    u = self.discrete_controls[action]
    state = self.integrate_forward(self.state, u)
    self.state = state

    # done
    if self.doneType == 'toEnd':
      done = not self.check_within_bounds(self.state)
    else:
      assert self.doneType == 'TF', 'invalid doneType'
      fail = g_x_cur < 0
      done = fail 
    if done:
      self.alive = False

    if self.use_wm: 
      return self.latent.copy(), np.copy(self.state), done # need latent 
    else:
      return np.copy(self.state), done

  def integrate_forward(self, state, u):
    """Integrates the dynamics forward by one step.

    Args:
        state (np.ndarray): (x, y, yaw).
        u (float): the contol input, angular speed.

    Returns:
        np.ndarray: next state.
    """
    x, y, theta = state

    if self.learned_dyn:
      inp = torch.Tensor([x,y,theta, u/self.max_turning_rate]).to(self.device)
      delta = self.MLP_dyn(inp).detach().cpu().numpy()
      delta[:2] *= self.speed*self.time_step
      delta[2] *= self.max_turning_rate*self.time_step
      state_next = state + delta
      state_next[2] = np.mod(state_next[2], 2 * np.pi)
      exit()
    else:
      x = x + self.time_step * self.speed * np.cos(theta)
      y = y + self.time_step * self.speed * np.sin(theta)
      theta = np.mod(theta + self.time_step * u, 2 * np.pi)
      assert theta >= 0 and theta < 2 * np.pi
      state_next = np.array([x, y, theta])
      
    return state_next

  # == Setting Hyper-Parameter Functions ==
  def set_bounds(self, bounds):
    """Sets the boundary of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

  def set_speed(self, speed=.5):
    """Sets speed of the car. The speed influences the angular speed and the
        discrete control set.

    Args:
        speed (float, optional): speed of the car. Defaults to .5.
    """
    self.speed = speed
    self.max_turning_rate = self.speed / self.R_turn  # w
    self.discrete_controls = np.array([
        -self.max_turning_rate, 0., self.max_turning_rate
    ])

  def set_time_step(self, time_step=.05):
    """Sets the time step for dynamics integration.

    Args:
        time_step (float, optional): time step used in the integrate_forward.
            Defaults to .05.
    """
    self.time_step = time_step

  def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
    """Sets target_radius, constraint_radius and turning radius.

    Args:
        target_radius (float, optional): the radius of the target set.
            Defaults to .3.
        constraint_radius (float, optional): the radius of the constraint set.
            Defaults to 1.0.
        R_turn (float, optional): the radius of the car's circular motion.
            Defaults to .6.
    """
    self.target_radius = target_radius
    self.constraint_radius = constraint_radius
    self.set_radius_rotation(R_turn=R_turn)

  def set_radius_rotation(self, R_turn=.6, verbose=False):
    """Sets radius of the car's circular motion. The turning radius influences
        the angular speed and the discrete control set.

    Args:
        R_turn (float, optional): the radius of the car's circular motion.
            Defaults to .6.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.R_turn = R_turn
    self.max_turning_rate = self.speed / self.R_turn  # w
    self.discrete_controls = np.array([
        -self.max_turning_rate, 0., self.max_turning_rate
    ])
    if verbose:
      print(self.discrete_controls)

  def set_constraint(self, center, radius):
    """Sets the constraint set (complement of failure set).

    Args:
        center (np.ndarray, optional): center of the constraint set.
        radius (float, optional): radius of the constraint set.
    """
    self.constraint_center = center
    self.constraint_radius = radius

  def set_target(self, center, radius):
    """Sets the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
        radius (float, optional): radius of the target set.
    """
    self.target_center = center
    self.target_radius = radius

  # == Getting Functions ==
  def check_within_bounds(self, state):
    """Checks if the agent is still in the environment.

    Args:
        state (np.ndarray): the state of the agent.

    Returns:
        bool: False if the agent is not in the environment.
    """
    for dim, bound in enumerate(self.bounds):
      flagLow = state[dim] < bound[0]
      flagHigh = state[dim] > bound[1]
      if flagLow or flagHigh:
        return False
    return True

  # == Compute Margin ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    '''x, y = (self.low + self.high)[:2] / 2.0
    w, h = (self.high - self.low)[:2]
    boundary_margin = calculate_margin_rect(
        s, [x, y, w, h], negativeInside=True
    )
    g_xList = [boundary_margin]'''
    g_xList = []
    if self.gt_lx:
      raise Exception('Not implemented')

      return self.gt_safety_margin(s)
    elif self.use_wm:
      self.MLP_margin.eval()
      feat = self.wm.dynamics.get_feat(s).detach()
      with torch.no_grad():  # Disable gradient calculation
        outputs = self.MLP_margin(feat)
        g_xList.append(outputs.detach().cpu().numpy())
    elif self.learned_margin:
      s_tensor = torch.Tensor([s[0], s[1], 0])
      with torch.no_grad():  # Disable gradient calculation
        outputs = self.MLP_margin(s_tensor.to(self.device)).item()
        g_xList.append(outputs)
    else:
      c_c_exists = (self.constraint_center is not None)
      c_r_exists = (self.constraint_radius is not None)
      if (c_c_exists and c_r_exists):
        g_x = calculate_margin_circle(
            s, [self.constraint_center, self.constraint_radius],
            negativeInside=True
        )
        g_xList.append(g_x)
    
    #safety_margin = np.max(np.array(g_xList))
    safety_margin = np.array(g_xList).squeeze()

    return self.safetyScaling * safety_margin
  
  def gt_safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    '''x, y = (self.low + self.high)[:2] / 2.0
    w, h = (self.high - self.low)[:2]
    boundary_margin = calculate_margin_rect(
        s, [x, y, w, h], negativeInside=True
    )
    g_xList = [boundary_margin]'''
    g_xList = []
    c_c_exists = (self.constraint_center is not None)
    c_r_exists = (self.constraint_radius is not None)
    if (c_c_exists and c_r_exists):
      g_x = calculate_margin_circle(
          s, [self.constraint_center, self.constraint_radius],
          negativeInside=True
      )
      g_xList.append(g_x)
    
    safety_margin = np.array(g_xList).squeeze()

    return self.safetyScaling * safety_margin

  def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    if self.target_center is not None and self.target_radius is not None:
      target_margin = calculate_margin_circle(
          s, [self.target_center, self.target_radius], negativeInside=True
      )
      return self.targetScaling * target_margin
    else:
      return None
