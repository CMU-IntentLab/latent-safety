# Author: Kensuke Nakamura, 2025
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import io
from PIL import Image
import numpy as np
import torch
import pickle
import pathlib
import ruamel.yaml as yaml
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dreamerv3-torch'))
sys.path.append(dreamer_dir)
import tools


def get_frame(states, config):
  dt = config.dt
  v = config.speed
  fig,ax = plt.subplots()
  plt.xlim([config.x_min, config.x_max])
  plt.ylim([config.y_min, config.y_max])
  plt.axis('off')
  fig.set_size_inches( 1, 1 )
  # Create the circle patch
  circle = patches.Circle([config.obs_x, config.obs_y], config.obs_r, edgecolor=(1,0,0), facecolor='none')
  # Add the circle patch to the axis
  ax.add_patch(circle)
  plt.quiver(states[0], states[1], dt*v*torch.cos(states[2]), dt*v*torch.sin(states[2]), angles='xy', scale_units='xy', minlength=0,width=0.1, scale=0.18,color=(0,0,1), zorder=3)
  plt.scatter(states[0], states[1],s=20, c=(0,0,1), zorder=3)
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  buf = io.BytesIO()
  plt.savefig(buf, format='png', dpi=config.size[0])
  buf.seek(0)

  # Load the buffer content as an RGB image
  img = Image.open(buf).convert('RGB')
  img_array = np.array(img)
  plt.close(fig)
  return img_array
   
def get_init_state(config):  
  # don't sample inside the failure set
  states = torch.zeros(3)
  while np.linalg.norm(states[:2] - np.array([config.obs_x, config.obs_y])) < config.obs_r:
    states = torch.rand(3)
    states[0] *= (config.x_max-config.buffer) - (config.x_min + config.buffer)
    states[1] *= (config.y_max-config.buffer) - (config.y_min + config.buffer)
    states[0] += config.x_min + config.buffer
    states[1] += config.y_min + config.buffer

  # so that the trajectory doesn't immediately go out of bounds
  states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
  states[2] = states[2] % (2*np.pi)
  return states

def gen_one_traj_img(config):
  states = get_init_state(config)

  state_obs = []
  img_obs = []
  state_gt = []
  dones = []
  acs = []
  u_max = final_config.turnRate
  dt = config.dt
  v = config.speed


  for t in range(config.data_length):
    # random between -u_max and u_max
    ac = torch.rand(1) * 2 * u_max - u_max

    states_next = torch.rand(3)
    states_next[0] = states[0] + v*dt*torch.cos(states[2])
    states_next[1] = states[1] + v*dt*torch.sin(states[2])
    states_next[2] = states[2] + dt*ac


    # the data is (o_t, a_t), don't observe o_t+1 yet
    state_obs.append(states[2].numpy()) # get to observe theta
    state_gt.append(states.numpy()) # gt state for debugging
    if t == config.data_length-1:
      dones.append(1)
    elif torch.abs(states[0]) > config.x_max-config.buffer or torch.abs(states[1]) > config.y_max-config.buffer: # out of bounds
      dones.append(1)
    else:
      dones.append(0)
        
    acs.append(ac)
    img_array = get_frame(states, config)
    img_obs.append(img_array)
    states = states_next
    if dones[-1] == 1:
      break
  return state_obs, acs, state_gt, img_obs, dones

def generate_trajs(config):
  demos = []
  for i in range(config.num_trajs):
    state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img(config)
    demo = {}
    demo['obs'] = {'image': img_obs, 'state': state_obs, 'priv_state': state_gt}
    demo['actions'] = acs
    demo['dones'] = dones
    demos.append(demo)
    print('demo: ', i, "timesteps: ", len(state_obs))

  
  with open('wm_demos'+str(config.size[0])+'.pkl', 'wb') as f:
    pickle.dump(demos, f)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

if __name__=='__main__':      
    parser = argparse.ArgumentParser()
   
    config, remaining = parser.parse_known_args()

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs.yaml").read_text()
    )

    name_list = ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    demos = generate_trajs(final_config)
