"""
Shared constants/methods across environments
"""

from pathlib import Path

# Time limit horizons for RL training
HORIZONS = {
    "lift": 100,
    "Lift": 100,
    "can": 200,
    "Can": 200,
    "PickPlaceCan": 200,
    "square": 300,
    "Square": 300,
    "NutAssemblySquare": 300,
    "dubins": 100
}

CORNELL_CLUSTER_ROBOMIMIC_DATASET_DIR = Path(
    "/share/portal/irl_squared_data/robomimic_data"
)

# TODO: need to update this with the correct path once set up.
DOCKER_ROBOMIMIC_DATASET_DIR = Path("/home/dreamerv3/robomimic_datasets")

LOW_DIM_OBS_KEYS = [
    "object",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
]

IMAGE_OBS_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
]


STATE_SHAPE_META = {
    # "robot0_joint_pos_cos": {
    #     "shape": [7],
    #     "type": "low_dim",
    # },
    # "robot0_joint_pos_sin": {
    #     "shape": [7],
    #     "type": "low_dim",
    # },
    "robot0_eef_pos": {
        "shape": [3],
        "type": "low_dim",
    },
    "robot0_eef_quat": {
        "shape": [4],
        "type": "low_dim",
    },
    "robot0_gripper_qpos": {
        "shape": [2],
        "type": "low_dim",
    },
}
