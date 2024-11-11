#%% preamble

import RARL_wm
import os
import math
import numpy as np
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
    dx = (env.bounds[0, 1] - env.bounds[0, 0]) / nx
    dy = (env.bounds[1, 1] - env.bounds[1, 0]) / ny
    dtheta = 2*np.pi / grid.shape[2]
    x_idx = int((x - env.bounds[0, 0]) / dx)
    y_idx = int((y - env.bounds[1, 0]) / dy)
    # for theta, make sure to wrap the angle to 0-2pi
    theta_idx = int((theta % (2*np.pi)) / dtheta)
    return grid[x_idx, y_idx, theta_idx]

def collect_rollout_data(env, agent, position_gridsize, angle_gridsize, enable_observation_feedback=True):
    rollout_data = {}
    thetas = np.linspace(0, 2*math.pi, angle_gridsize, endpoint=False)
    rollout_eval_folder = os.path.join(project_root, environment_info["outFolder"], "rollout_eval")
    os.makedirs(rollout_eval_folder, exist_ok=True)
    for theta_idx, theta in enumerate(thetas):
        theta_deg = math.degrees(theta)
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
    evaluated_rollout_data = {}
    stats = {
        "total" : 0,
        "infeasible": 0,
        "safe": 0,
        "unsafe": 0,
        "false_positive": 0,
    }

    for theta_data in rollout_data.values():
        for rollout_info in theta_data.values():
            ground_truth_metrics = rollout_info["groundtruth_metrics"]
            x, y, theta = ground_truth_metrics["traj"][0]
            learned_metrics = rollout_info["learned_metrics"]
            # find the first time step where the value is negative
            ground_truth_failure_time = next(
                (idx for idx, value in enumerate(ground_truth_metrics["valueList"]) if value < 0), None
            )
            learned_failure_time = next(
                (idx for idx, value in enumerate(learned_metrics["valueList"]) if value < 0), None
            )
            evaluation = {
                "ground_truth_initial_value": get_grid_value_for_state(env, ground_truth_brt, x, y, theta),
                "ground_truth_failure_margin": ground_truth_metrics["minV"],
                "ground_truth_failure_time": ground_truth_failure_time,
                "learned_failure_margin": learned_metrics["minV"],
                "learned_failure_time": learned_failure_time,
            }
            evaluated_rollout_data[(x, y, theta)] = evaluation

            # book-keeping
            stats["total"] += 1
            is_infeasible = evaluation["ground_truth_initial_value"] < 0
            if is_infeasible:
                stats["infeasible"] += 1
                continue
            is_safe = evaluation["ground_truth_failure_margin"] > 0
            if is_safe:
                assert ground_truth_failure_time is None
                stats["safe"] += 1
            else:
                assert ground_truth_failure_time is not None
                stats["unsafe"] += 1
            # check if the learned classifier predicted failure despite the system being fine
            if (learned_failure_time is not None) and (learned_failure_time < ground_truth_failure_time):
                stats["false_positive"] += 1

    assert stats["total"] == stats["infeasible"] + stats["safe"] + stats["unsafe"]

    return stats, evaluated_rollout_data


#%% setup
config = RARL_wm.get_config(parse_args=False)
env, environment_info = RARL_wm.construct_environment(config, visualize_failure_sets=False)
agent = load_best_agent(config)
ground_truth_brt = np.load(config.grid_path)

# %%
# rollout_data = collect_rollout_data(env, agent, position_gridsize = 1, angle_gridsize=1)

# %%
rollout_data = load_obj(os.path.join(project_root, environment_info["outFolder"], "rollout_eval", "rollout_data"))
stats, evaluate_rollout_data = evaluate_rollout_data(rollout_data, ground_truth_brt)