# %% preamble
# making sure we use GPU1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import RARL_wm
import math
import numpy as np
import pandas as pd
import torch
from safety_rl.RARL.utils import save_obj, load_obj
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import IPython

the_ipython_instance = IPython.get_ipython()
if the_ipython_instance is not None:
    the_ipython_instance.magic("load_ext autoreload")
    the_ipython_instance.magic("autoreload 2")
    the_ipython_instance.magic("matplotlib inline")


# %% functions
def load_best_agent(config, environment_info):
    agent = RARL_wm.construct_agent(config, environment_info)
    # TODO get this index from the training dict based on metrics rather than hard-coding
    restore_idx = 150_000
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
        # map index back to angle
        theta = theta_idx * 2 * np.pi / ground_truth_brt.shape[2]
        slice = env.get_value(q_func, theta=theta, nx=nx, ny=ny)
        slices.append(slice)
    v_nn = np.stack(slices, axis=2)
    v_grid = ground_truth_brt[:, :, theta_indices]
    tn, tp, fn, fp = env.confusion(v_nn, v_grid)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp < 1e-6:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn < 1e-6:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if precision + recall < 1e-6:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "tn": tn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return metrics


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


def evaluate_rollout_data(env, rollout_data, ground_truth_brt):
    evaluated_rollout_data = []

    for theta_data in rollout_data.values():
        for rollout_info in theta_data.values():
            # extract relevant information from the rollout
            # ground truth metrics
            ground_truth_metrics = rollout_info["groundtruth_metrics"]
            ground_truth_initial_state = ground_truth_metrics["traj"][0]
            ground_truth_initial_value = get_grid_value_for_state(
                env, ground_truth_brt, *ground_truth_initial_state
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
            learned_safety_time = next(
                (
                    idx
                    for idx, value in enumerate(learned_metrics["actionValueList"])
                    if value < 0
                ),
                None,
            )

            # high-level classification of the rollout according to main features
            is_feasible = ground_truth_initial_value >= 0
            is_safe = ground_truth_failure_margin >= 0

            assert is_safe == (ground_truth_failure_time is None)

            is_learning_classification_correct = (
                is_safe and (learned_safety_time is None)
            ) or (
                (not is_safe)
                and (learned_safety_time is not None)
                and (learned_safety_time <= ground_truth_failure_time)
            )

            # aggregate all the data
            evaluated_rollout_data.append(
                {
                    "ground_truth_initial_value": ground_truth_initial_value,
                    "ground_truth_failure_time": ground_truth_failure_time,
                    "learned_failure_time": learned_safety_time,
                    "is_feasible": is_feasible,
                    "is_safe": is_safe,
                    "is_learning_classification_correct": is_learning_classification_correct,
                    "learned_metrics": learned_metrics,
                }
            )

    return pd.DataFrame(evaluated_rollout_data)

def generate_representative_rollout_videos(env, evaluated_rollouts, output_folder, output_prefix):
    # find up to 5 safe and 5 unsafe rollouts and generate a video for each of them

    safe_rollouts = evaluated_rollouts[evaluated_rollouts["is_safe"]].head(5)
    for idx, row in safe_rollouts.iterrows():
        print(f"Generating video for safe rollout {idx}")
        rollout_data = row["learned_metrics"]
        decode_video_from_feature_sequence(env, rollout_data, os.path.join(output_folder, f"{output_prefix}_safe_{idx}.mp4"))

    unsafe_rollouts = evaluated_rollouts[~evaluated_rollouts["is_safe"]].head(5)
    for idx, row in unsafe_rollouts.iterrows():
        print(f"Generating video for unsafe rollout {idx}")
        rollout_data = row["learned_metrics"]
        decode_video_from_feature_sequence(env, rollout_data, os.path.join(output_folder, f"{output_prefix}_unsafe_{idx}.mp4"))


def video_from_array(image_sequence_array, output_filename = "output_video.mp4"):
    import cv2
    # Assuming `image_sequence` is your tensor with shape (time, height, width, 3)
    # Normalize the image values to [0, 255] if they are not already in this range
    image_sequence_array = (image_sequence_array * 255).clip(0, 255).astype(np.uint8)

    # Define video properties
    height, width, _ = image_sequence_array.shape[1:]
    fps = 30  # Set frames per second

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Write frames to the video
    for frame in image_sequence_array:
        video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

    print(f"Video saved as {output_filename}")
    return output_filename


def decode_video_from_feature_sequence(env, rollout, output_filename):
    # feature_sequence_batch = torch.FloatTensor(d["learned_metrics"]["traj"]).to(env.device).squeeze(1)
    # layout: (batch, time, features)
    feature_sequence_batch = torch.FloatTensor(rollout["traj"]).to(env.device).unsqueeze(0).squeeze(2)
    image_sequence = env.car.wm.heads["decoder"](feature_sequence_batch)["image"].mode().squeeze(0).cpu().detach().numpy()
    # convert tensor of (time, npx, npy, 3) to mp4 video
    video_from_array(image_sequence, output_filename)


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
    return fig


def evaluate(
    env,
    agent,
    ground_truth_brt,
    experiment_name,
    position_gridsize,
    angle_gridsize,
    reproduce_value_function=True,
    reproduce_closed_loop_rollouts=True,
    reproduce_open_loop_rollouts=True,
    generate_videos=True,
    show_plots=True,
):
    output_folder = os.path.join(
        project_root, environment_info["outFolder"], experiment_name
    )
    os.makedirs(output_folder, exist_ok=True)

    # ------------------------------------------- value function
    value_function_metrics_path = os.path.join(
        output_folder, experiment_name + "_value_function_metrics"
    )
    if reproduce_value_function or not os.path.exists(
        f"{value_function_metrics_path}.pkl"
    ):
        value_function_metrics = compute_value_funtion_metrics(
            env, ground_truth_brt, agent.Q_network
        )
        save_obj(value_function_metrics, value_function_metrics_path)
    value_function_metrics = load_obj(value_function_metrics_path)
    # pretty print the metrics
    for key, value in value_function_metrics.items():
        print(f"{key}: {value:.3f}")

    # ----------------------- closed-loop rollout data collection
    closed_loop_rollout_data_path = os.path.join(
        output_folder, experiment_name + "closed_loop_rollout_data"
    )
    if reproduce_closed_loop_rollouts or not os.path.exists(
        f"{closed_loop_rollout_data_path}.pkl"
    ):
        rollout_data = collect_rollout_data(
            env,
            agent,
            position_gridsize=position_gridsize,
            angle_gridsize=angle_gridsize,
            output_folder=output_folder,
            output_prefix=f"closed_loop_${experiment_name}",
        )
        save_obj(rollout_data, closed_loop_rollout_data_path)
    rollout_data = load_obj(closed_loop_rollout_data_path)
    evaluated_rollouts = evaluate_rollout_data(env, rollout_data, ground_truth_brt)
    plt = visualize_evaluated_rollout_stats(
        evaluated_rollouts, title=f"{experiment_name} Closed-Loop Rollout Evaluation"
    )
    if generate_videos:
        generate_representative_rollout_videos(env, evaluated_rollouts, output_folder, experiment_name)
    if the_ipython_instance is not None and show_plots:
        IPython.display.display(plt)

    # ----------------------- open-loop rollout data collection
    open_loop_rollout_data_path = os.path.join(
        output_folder,
        experiment_name + "open_loop_rollout_data",
    )
    if reproduce_open_loop_rollouts or not os.path.exists(
        f"{open_loop_rollout_data_path}.pkl"
    ):
        open_loop_rollout_data = collect_rollout_data(
            env,
            agent,
            position_gridsize=position_gridsize,
            angle_gridsize=angle_gridsize,
            output_folder=output_folder,
            output_prefix=f"open_loop_${experiment_name}",
            enable_observation_feedback=False,
        )
        save_obj(open_loop_rollout_data, open_loop_rollout_data_path)
    open_loop_rollout_data = load_obj(open_loop_rollout_data_path)
    evaluated_open_loop_rollouts = evaluate_rollout_data(
        env, open_loop_rollout_data, ground_truth_brt
    )
    plt = visualize_evaluated_rollout_stats(
        evaluated_open_loop_rollouts,
        title=f"{experiment_name} Open-Loop Rollout Evaluation",
    )
    if generate_videos:
        generate_representative_rollout_videos(env, evaluated_open_loop_rollouts, output_folder, f"open_loop_{experiment_name}")
    if the_ipython_instance is not None and show_plots:
        IPython.display.display(plt)


# %% base setup
position_gridsize = 10
angle_gridsize = 3
config = RARL_wm.get_config(parse_args=False)
base_env, environment_info = RARL_wm.construct_environment(
    config, visualize_failure_sets=False
)
agent = load_best_agent(config, environment_info)
ground_truth_brt = np.load(config.grid_path)

# %%
experiment_setups = {}
# -------------------------------------------------------appearance ood setups
# in-distribution setup
experiment_setups["nominal"] = {
    "env": base_env,
    "ground_truth_brt": ground_truth_brt,
}

# out-of-distribution evaluation with cyan background
experiment_setups["cyanbg_ood"] = {
    "env": RARL_wm.construct_environment(
        config, visualize_failure_sets=False, ood_dict={"background": (0, 1, 1)}
    )[0],
    "ground_truth_brt": ground_truth_brt,
}
# out-of-distribution evaluation with magenta background
experiment_setups["magentabg_ood"] = {
    "env": RARL_wm.construct_environment(
        config, visualize_failure_sets=False, ood_dict={"background": (1, 0, 1)}
    )[0],
    "ground_truth_brt": ground_truth_brt,
}
# out-of-distribution evaluation with different scale
experiment_setups["scaled01_ood"] = {
    "env": RARL_wm.construct_environment(
        config, visualize_failure_sets=False, ood_dict={"scale": 0.1}
    )[0],
    "ground_truth_brt": ground_truth_brt,
}
# out-of-distribution evaluation with magenta obstacles
experiment_setups["magentaobs_ood"] = {
    "env": RARL_wm.construct_environment(
        config, visualize_failure_sets=False, ood_dict={"obstacle_color": (1, 0, 1)}
    )[0],
    "ground_truth_brt": ground_truth_brt,
}
# out-of-distribution evaluation with yellow obstacles
experiment_setups["yellowobs_ood"] = {
    "env": RARL_wm.construct_environment(
        config, visualize_failure_sets=False, ood_dict={"obstacle_color": (1, 1, 0)}
    )[0],
    "ground_truth_brt": ground_truth_brt,
}

# %%
# run all of the experiment setups in sequence
for experiment_name, experiment_setup in experiment_setups.items():
    print(f"Running evaluation for {experiment_name}")
    img = Image.fromarray(experiment_setup["env"].capture_image())
    if the_ipython_instance is not None:
        IPython.display.display(img)
    evaluate(
        env=experiment_setup["env"],
        agent=agent,
        ground_truth_brt=experiment_setup["ground_truth_brt"],
        experiment_name=experiment_name,
        position_gridsize=position_gridsize,
        angle_gridsize=angle_gridsize,
        reproduce_closed_loop_rollouts=False,
        reproduce_open_loop_rollouts=False,
        reproduce_value_function=False,
    )