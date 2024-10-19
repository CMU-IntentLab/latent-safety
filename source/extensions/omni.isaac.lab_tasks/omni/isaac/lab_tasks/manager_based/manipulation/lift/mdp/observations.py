# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
import pytorch3d.transforms as transforms
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def object_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    return object_pos_w

def left_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("left"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def right_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("right"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def left_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("left"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def object_failures(
    env: ManagerBasedRLEnv,
    left_cfg: SceneEntityCfg = SceneEntityCfg("left"),
    right_cfg: SceneEntityCfg = SceneEntityCfg("right"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    left: RigidObject = env.scene[left_cfg.name]
    right: RigidObject = env.scene[right_cfg.name]
    left_pos_w = left.data.root_quat_w
    right_pos_w = right.data.root_quat_w
    left_mat = transforms.quaternion_to_matrix(left_pos_w)
    right_mat = transforms.quaternion_to_matrix(right_pos_w)
    left_euler = transforms.matrix_to_euler_angles(left_mat, 'XYZ')
    right_euler = transforms.matrix_to_euler_angles(right_mat, 'XYZ')   

    if torch.abs(left_euler[:, 0]) > 1.0 or torch.abs(left_euler[:, 1]) > 1.0:
        left_fail = torch.tensor([1.0])
    else:
        left_fail = torch.tensor([0.0])

    if torch.abs(right_euler[:, 0]) > 1.0 or torch.abs(right_euler[:, 1]) > 1.0:
        right_fail = torch.tensor([1.0])
    else:
        right_fail = torch.tensor([0.0])

    
    joint_quat = torch.cat([left_pos_w, right_pos_w])
    joint_fail = torch.cat([left_fail, right_fail])
    if left_fail > right_fail:
        return left_fail
    else:
        return right_fail
    #return fail#joint_fail



def object_id(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    object: RigidObject = env.scene[object_cfg.name]
    object_id = object.data.ID[:]
    
    return object_id
