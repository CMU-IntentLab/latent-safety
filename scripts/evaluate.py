#%% preamble

import RARL_wm
import os
import math
import numpy as np
import pandas as pd
from safety_rl.RARL.utils import save_obj, load_obj
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


def get_grid_value_for_state(env, grid, x, y, theta):
    """
    Find the neirest value in the grid for the given state.
    """
    nx = grid.shape[0]
    ny = grid.shape[1]
    ntheta = grid.shape[2]
    dx = (env.bounds[0, 1] - env.bounds[0, 0]) / nx
    dy = (env.bounds[1, 1] - env.bounds[1, 0]) / ny
    dtheta = 2*np.pi / grid.shape[2]
    x_idx = min(int((x - env.bounds[0, 0]) / dx), nx-1)
    y_idx = min(int((y - env.bounds[1, 0]) / dy), ny-1)
    # for theta, make sure to wrap the angle to 0-2pi
    theta_idx = min(int((theta % (2*np.pi)) / dtheta), ntheta-1)
    return grid[x_idx, y_idx, theta_idx]

def collect_rollout_data(env, agent, position_gridsize, angle_gridsize, enable_observation_feedback=True):
    rollout_data = {}
    thetas = np.linspace(0, 2*math.pi, angle_gridsize, endpoint=False)
    rollout_eval_folder = os.path.join(project_root, environment_info["outFolder"], "rollout_eval")
    os.makedirs(rollout_eval_folder, exist_ok=True)
    for theta_idx, theta in enumerate(thetas):
        theta_deg = math.degrees(theta)
        print(f"Collecting rollouts for theta = {theta_deg:.0f} degrees; {theta_idx+1}/{angle_gridsize}")
        _, _, infos = env.plot_trajectories(
            q_func=agent.Q_network,
            num_rnd_traj=10,
            theta = theta,
            # convert to degrees for filename
            save_dir=os.path.join(rollout_eval_folder, f'safe_rollouts_theta{theta_deg:.0f}deg.png'),
            return_infos=True,
            enable_observation_feedback=enable_observation_feedback,
            wait_for_all_metrics_to_predict_failure=True,
            position_gridsize=position_gridsize,
        )
        rollout_data[theta_idx] = infos

    save_obj(rollout_data, os.path.join(rollout_eval_folder, "rollout_data"))
    return rollout_data

def evaluate_rollout_data(rollout_data, ground_truth_brt):
    evaluated_rollout_data = []

    for theta_data in rollout_data.values():
        for rollout_info in theta_data.values():
            # extract relevant information from the rollout
            # ground truth metrics
            ground_truth_metrics = rollout_info["groundtruth_metrics"]
            ground_truth_initial_state = ground_truth_metrics["traj"][0]
            ground_truth_initial_value = get_grid_value_for_state(env, ground_truth_brt, ground_truth_initial_state[0], ground_truth_initial_state[1], ground_truth_initial_state[2])
            ground_truth_failure_margin = ground_truth_metrics["minV"]
            ground_truth_failure_time = next(
                (idx for idx, value in enumerate(ground_truth_metrics["valueList"]) if value < 0), None
            )
            # learned metrics
            learned_metrics = rollout_info["learned_metrics"]
            learned_failure_time = next(
                (idx for idx, value in enumerate(learned_metrics["valueList"]) if value < 0), None
            )
            learned_failure_margin = learned_metrics["minV"]

            # high-level classification of the rollout according to main features
            is_feasible = ground_truth_initial_value >= 0
            if is_feasible:
                is_safe = ground_truth_failure_margin >= 0
                is_false_positive = (learned_failure_time is not None) and (learned_failure_time < ground_truth_failure_time)
            else:
                is_safe = None
                is_false_positive = None

            # aggregate all the data
            evaluated_rollout_data.append(
                {
                    "ground_truth_initial_value": ground_truth_initial_value,
                    "ground_truth_failure_margin": ground_truth_failure_margin,
                    "ground_truth_failure_time": ground_truth_failure_time,
                    "learned_failure_time": learned_failure_time,
                    "learned_failure_margin": learned_failure_margin,
                    "is_feasible": is_feasible,
                    "is_safe": is_safe,
                    "is_false_positive": is_false_positive,
                }
            )


    return pd.DataFrame(evaluated_rollout_data)

#%% setup
config = RARL_wm.get_config(parse_args=False)
env, environment_info = RARL_wm.construct_environment(config, visualize_failure_sets=False)
agent = load_best_agent(config)
ground_truth_brt = np.load(config.grid_path)

# %% 
# if you skip this cell, the cells below will just load the data from disk
rollout_data = collect_rollout_data(env, agent, position_gridsize = 51, angle_gridsize=12)

# %%
rollout_data = load_obj(os.path.join(project_root, environment_info["outFolder"], "rollout_eval", "rollout_data"))
evaluated_rollouts = evaluate_rollout_data(rollout_data, ground_truth_brt)
# %%
