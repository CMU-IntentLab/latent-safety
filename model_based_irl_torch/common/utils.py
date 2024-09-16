import sys
import os

from pathlib import Path
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from common.constants import (
    CORNELL_CLUSTER_ROBOMIMIC_DATASET_DIR,
    DOCKER_ROBOMIMIC_DATASET_DIR,
    STATE_SHAPE_META,
)
#import robomimic.utils.file_utils as FileUtils
from typing import Any, Dict


class HiddenPrints:
    """
    Suppress print output.
    """

    def __enter__(self) -> None:
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_) -> None:
        sys.stdout.close()
        sys.stdout = self._original_stdout


def to_np(x):
    return x.detach().cpu().numpy()


def get_robomimic_dataset_path_and_env_meta(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
):
    """
    Returns the path to the Robomimic dataset and environment metadata.

    Args:
        env_id (str): The ID of the environment.
        collection_type (str, optional): The type of data collection. Defaults to "ph".
        obs_type (str, optional): The type of observations. Defaults to "image".
        shaped (bool, optional): Whether the dataset is shaped or not. Defaults to False.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    assert int(done_mode) in [0, 1, 2]

    dataset_name = obs_type
    if image_size != 0:
        dataset_name += f"_{image_size}"
    if shaped:
        dataset_name += "_shaped"
    dataset_name += f"_done{done_mode}"
    dataset_path = f"{env_id.lower()}/{collection_type}/{dataset_name}_v141.hdf5"

    cwd = Path(Path.cwd())
    if "/share/portal" in str(cwd):
        # in Cornell cluster:
        root_dir = CORNELL_CLUSTER_ROBOMIMIC_DATASET_DIR
    else:
        # in Docker image
        root_dir = DOCKER_ROBOMIMIC_DATASET_DIR
    dataset_path = Path(root_dir, dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    return dataset_path, env_meta


def evaluate(agent, eval_env, video_env, num_episodes, num_episodes_to_record=1):
    """
    Evaluate the policy in environment, record video of first episode.
    """
    success = 0
    for i in trange(num_episodes, desc="Eval rollouts", ncols=0, leave=False):
        if num_episodes_to_record > 0:
            curr_env = video_env
            num_episodes_to_record -= 1
        else:
            curr_env = eval_env

        observation, done = curr_env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, reward, done, _ = curr_env.step(action)
        if float(reward) == 0:
            success += 1
    return {
        "return": np.mean(eval_env.first_env.return_queue),
        "length": np.mean(eval_env.first_env.length_queue),
        "success": success / num_episodes,
    }


def create_shape_meta(img_size, include_state):
    shape_meta = {
        "obs": {
            "agentview_image": {
                # gym expects (H, W, C)
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
            "robot0_eye_in_hand_image": {
                # gym expects (H, W, C)
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
        },
        "action": {"shape": [7]},
    }
    if include_state:
        shape_meta["obs"].update(STATE_SHAPE_META)
    return shape_meta


def get_dataset_path_and_meta_info(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
):
    """
    Returns the path to the Robomimic dataset and environment metadata.

    Args:
        env_id (str): The ID of the environment.
        collection_type (str, optional): The type of data collection. Defaults to "ph".
        obs_type (str, optional): The type of observations. Defaults to "image".
        shaped (bool, optional): Whether the dataset is shaped or not. Defaults to False.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id,
        collection_type=collection_type,
        obs_type=obs_type,
        shaped=shaped,
        image_size=image_size,
        done_mode=done_mode,
    )
    shape_meta = create_shape_meta(image_size, include_state=True)
    return dataset_path, env_meta, shape_meta


def combine_dictionaries(
    one_dict: Dict[str, Any], other_dict: Dict[str, Any], take_half: bool = False
) -> Dict[str, Any]:
    """
    Combine two dictionaries by interleaving their values.

    Args:
        one_dict (Dict[str, Any]): The first dictionary.
        other_dict (Dict[str, Any]): The second dictionary.
        take_half (bool, optional): Whether to only take the first half of the values. Defaults to False.
    """
    combined = {}
    unused_keys = set(one_dict.keys()) - set(other_dict.keys())
    assert set(unused_keys).issubset(
        {"logprob", "object_state", "privileged_state", "success"}
    ), f"Missing {unused_keys}"

    for k, v in one_dict.items():
        if k in unused_keys:
            continue
        if isinstance(v, dict):
            combined[k] = combine_dictionaries(v, other_dict[k], take_half)
        elif v is None or v.shape[0] == 0:
            combined[k] = other_dict[k]
        elif other_dict[k] is None or other_dict[k].shape[0] == 0:
            combined[k] = v
        else:
            if take_half:
                half_index = v.shape[0] // 2
                v = v[:half_index]
                other_v = other_dict[k][:half_index]
            else:
                other_v = other_dict[k]

            tmp = np.empty((v.shape[0] + other_v.shape[0], *v.shape[1:]), dtype=v.dtype)
            tmp[0::2] = v
            tmp[1::2] = other_v
            combined[k] = tmp

    return combined
