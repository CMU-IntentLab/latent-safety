# %% preamble
# making sure we use GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


# %% functions
def load_best_agent(config, environment_info):
    agent = RARL_wm.construct_agent(config, environment_info)
    # TODO get this index from the training dict based on metrics rather than hard-coding
    restore_idx = 10_000
    agent.restore(
        restore_idx, os.path.join(project_root, environment_info["outFolder"])
    )
    return agent


def compute_value_funtion_metrics(env, ground_truth_brt, q_func):
    nx = ground_truth_brt.shape[0]
    ny = ground_truth_brt.shape[1]
    # sub-sample theta indices to reduce computation
    theta_indices = np.linspace(0, ground_truth_brt.shape[2] - 1, 3, dtype=int)
    slices = []
    for theta_idx in theta_indices:
        ground_truth_brt_slice = ground_truth_brt[:, :, theta_idx]
        # map index back to angle
        theta = theta_idx * 2 * np.pi / ground_truth_brt.shape[2]
        slice = env.get_value(
            q_func, theta=theta, nx=nx, ny=ny
        )
        slices.append(slice)
    v_nn = np.stack(slices, axis=2)
    v_grid = ground_truth_brt[:, :, theta_indices]
    tn,tp,fn,fp = env.confusion(v_nn, v_grid)
    print(f"True Negatives: {tn}")
    print(f"True Positives: {tp}")
    print(f"False Negatives: {fn}")
    print(f"False Positives: {fp}")
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "tn": tn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def get_grid_value_for_state(env, grid, x, y, theta):
    """
    Find the neirest value in the grid for the given state.
    """
    nx = grid.shape[0]
    ny = grid.shape[1]
    ntheta = grid.shape[2]
    dx = (env.bounds[0, 1] - env.bounds[0, 0]) / nx
    dy = (env.bounds[1, 1] - env.bounds[1, 0]) / ny
    dtheta = 2 * np.pi / grid.shape[2]
    x_idx = min(int((x - env.bounds[0, 0]) / dx), nx - 1)
    y_idx = min(int((y - env.bounds[1, 0]) / dy), ny - 1)
    # for theta, make sure to wrap the angle to 0-2pi
    theta_idx = min(int((theta % (2 * np.pi)) / dtheta), ntheta - 1)
    return grid[x_idx, y_idx, theta_idx]


def collect_rollout_data(
    env,
    agent,
    position_gridsize,
    angle_gridsize,
    output_folder,
    output_prefix,
    enable_observation_feedback=True,
):
    rollout_data = {}
    thetas = np.linspace(0, 2 * math.pi, angle_gridsize, endpoint=False)
    os.makedirs(output_folder, exist_ok=True)
    for theta_idx, theta in enumerate(thetas):
        theta_deg = math.degrees(theta)
        print(
            f"Collecting rollouts for theta = {theta_deg:.0f} degrees; {theta_idx+1}/{angle_gridsize}"
        )
        _, _, infos = env.plot_trajectories(
            q_func=agent.Q_network,
            num_rnd_traj=10,
            theta=theta,
            # convert to degrees for filename
            save_dir=os.path.join(
                output_folder,
                output_prefix + f"safe_rollouts_theta{theta_deg:.0f}deg.png",
            ),
            return_infos=True,
            enable_observation_feedback=enable_observation_feedback,
            wait_for_all_metrics_to_predict_failure=True,
            position_gridsize=position_gridsize,
        )
        rollout_data[theta_idx] = infos

    return rollout_data


def evaluate_rollout_data(rollout_data, ground_truth_brt):
    evaluated_rollout_data = []

    for theta_data in rollout_data.values():
        for rollout_info in theta_data.values():
            # extract relevant information from the rollout
            # ground truth metrics
            ground_truth_metrics = rollout_info["groundtruth_metrics"]
            ground_truth_initial_state = ground_truth_metrics["traj"][0]
            ground_truth_initial_value = get_grid_value_for_state(
                in_distribution_env, ground_truth_brt, *ground_truth_initial_state
            )
            ground_truth_failure_margin = ground_truth_metrics["minV"]
            ground_truth_failure_time = next(
                (
                    idx
                    for idx, value in enumerate(ground_truth_metrics["valueList"])
                    if value < 0
                ),
                None,
            )
            # learned metrics
            learned_metrics = rollout_info["learned_metrics"]
            learned_failure_time = next(
                (
                    idx
                    for idx, value in enumerate(learned_metrics["valueList"])
                    if value < 0
                ),
                None,
            )
            learned_failure_margin = learned_metrics["minV"]

            # high-level classification of the rollout according to main features
            is_feasible = ground_truth_initial_value >= 0
            if is_feasible:
                is_safe = ground_truth_failure_margin >= 0
                is_learning_classification_correct = (
                    is_safe and (learned_failure_time is None)
                ) or (
                    not is_safe
                    and (learned_failure_time is not None)
                    and (learned_failure_time <= ground_truth_failure_time)
                )
            else:
                is_safe = False
                # TODO: think about this case
                is_learning_classification_correct = True

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
                    "is_learning_classification_correct": is_learning_classification_correct,
                }
            )

    return pd.DataFrame(evaluated_rollout_data)


def visualize_evaluated_rollout_stats(evaluated_rollouts, title):
    # create a sunburst chart of the results
    # https://plotly.com/python/sunburst-charts/

    import plotly.express as px

    # Sample dataframe based on the provided layout
    df = evaluated_rollouts

    # Define labels for each level in the sunburst chart
    df["feasibility"] = df["is_feasible"].map({True: "feasible", False: "infeasible"})
    df["safety"] = df["is_safe"].map({True: "safe", False: "unsafe"})
    df["classification"] = df["is_learning_classification_correct"].map(
        {True: "true class.", False: "false class."}
    )

    # Generate the sunburst plot
    fig = px.sunburst(
        df,
        path=["feasibility", "safety", "classification"],
        title=title,
        color="safety",
        color_discrete_map={
            "safe": "lightgreen",
            "unsafe": "orange",
            "(?)": "lightblue",
        },
    )
    fig.update_traces(textinfo="label+value")
    # set root color to white
    fig.show()
    return fig

# %% setup
# --- base setup
position_gridsize = 21
angle_gridsize = 3
config = RARL_wm.get_config(parse_args=False)
in_distribution_env, environment_info = RARL_wm.construct_environment(
    config, visualize_failure_sets=False
)
agent = load_best_agent(config, environment_info)
ground_truth_brt = np.load(config.grid_path)

# --- in-distribution evaluation setup ---
in_distribution_output_folder = os.path.join(
    project_root, environment_info["outFolder"], "in_distribution_rollout_eval"
)
in_distribution_output_prefix = "in_distribution_"
in_distribution_data_path = os.path.join(
    in_distribution_output_folder, in_distribution_output_prefix + "rollout_data"
)

# %% in-distribution value function metrics
in_distribution_value_function_metrics = compute_value_funtion_metrics(
    in_distribution_env, ground_truth_brt, agent.Q_network
)

# --- in-distribution open-loop evaluation setup ---
in_distribution_open_loop_output_folder = os.path.join(
    project_root,
    environment_info["outFolder"],
    "in_distribution_open_loop_rollout_eval",
)
in_distribution_open_loop_output_prefix = "in_distribution_open_loop_"
in_distribution_open_loop_data_path = os.path.join(
    in_distribution_open_loop_output_folder,
    in_distribution_open_loop_output_prefix + "rollout_data",
)

# %% in_distribution rollout data collection with feedback
in_distribution_rollout_data = collect_rollout_data(
    in_distribution_env,
    agent,
    position_gridsize=position_gridsize,
    angle_gridsize=angle_gridsize,
    output_folder=in_distribution_output_folder,
    output_prefix=in_distribution_output_prefix,
)
save_obj(in_distribution_rollout_data, in_distribution_data_path)

# %% in_distribution data plotting
in_distribution_rollout_data = load_obj(in_distribution_data_path)
evaluated_rollouts = evaluate_rollout_data(
    in_distribution_rollout_data, ground_truth_brt
)
visualize_evaluated_rollout_stats(
    evaluated_rollouts, title="In-Distribution Rollout Evaluation"
)

# %% in_distribution rollout data collection in open-loop (no observation feedback)
# if you skip this cell, the cells below will just load the data from disk
in_distribution_open_loop_rollout_data = collect_rollout_data(
    in_distribution_env,
    agent,
    position_gridsize=position_gridsize,
    angle_gridsize=angle_gridsize,
    output_folder=in_distribution_open_loop_output_folder,
    output_prefix=in_distribution_open_loop_output_prefix,
    enable_observation_feedback=False,
)
save_obj(in_distribution_open_loop_rollout_data, in_distribution_open_loop_data_path)

# %% in_distribution data plotting in open-loop
in_distribution_open_loop_rollout_data = load_obj(in_distribution_open_loop_data_path)
evaluated_open_loop_rollouts = evaluate_rollout_data(
    in_distribution_open_loop_rollout_data, ground_truth_brt
)
# %%
visualize_evaluated_rollout_stats(
    evaluated_open_loop_rollouts,
    title="In-Distribution Open-Loop Rollout Evaluation",
)