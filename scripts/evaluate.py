#%% preamble

import RARL_wm
import os
import math
import numpy as np
from safety_rl.RARL.utils import save_obj
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from IPython import get_ipython
the_ipython_instance = get_ipython()
if the_ipython_instance is not None:
    the_ipython_instance.magic("load_ext autoreload")
    the_ipython_instance.magic("autoreload 2")
    the_ipython_instance.magic("matplotlib inline")

#%% functions

def load_best_agent(config):
    agent = RARL_wm.construct_agent(config, environment_info)
    # TODO get this index from the training dict based on metrics rather than hard-coding
    restore_idx = 10_000
    # agent.restore(restore_idx, environment_info["outFolder"])
    agent.restore(restore_idx, os.path.join(project_root, environment_info["outFolder"]))
    return agent

def collect_rollout_data(env, agent, angle_grid_size, enable_observation_feedback=True):
    all_angle_infos = {}
    thetas = np.linspace(0, 2*math.pi, angle_grid_size, endpoint=False)
    rollout_eval_folder = os.path.join(project_root, environment_info["outFolder"], "rollout_eval")
    os.makedirs(rollout_eval_folder, exist_ok=True)
    for theta in thetas:
        theta_deg = math.degrees(theta)
        _, _, infos = env.plot_trajectories(
            q_func=agent.Q_network,
            num_rnd_traj=10,
            theta = theta,
            # convert to degrees for filename
            # save_dir=f'safe_rollouts_theta{theta_deg:.0f}deg.png',
            save_dir=os.path.join(rollout_eval_folder, f'safe_rollouts_theta{theta_deg:.0f}deg.png'),
            return_infos=True,
            enable_observation_feedback=enable_observation_feedback,
            wait_for_all_metrics_to_predict_failure=True,
        )
        all_angle_infos[theta] = infos

    save_obj(all_angle_infos, os.path.join(rollout_eval_folder, "all_angle_infos"))
    return all_angle_infos


#%% setup
config = RARL_wm.get_config(parse_args=False)
env, environment_info = RARL_wm.construct_environment(config)
agent = load_best_agent(config)

# %%
collect_rollout_data(env, agent, angle_grid_size=2)
# %%
