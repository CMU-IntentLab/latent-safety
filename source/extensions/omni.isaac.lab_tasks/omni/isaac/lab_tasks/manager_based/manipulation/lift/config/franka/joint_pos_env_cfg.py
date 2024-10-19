# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs.mdp.observations import grab_images

from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.lab.sim import SimulationCfg, PinholeCameraCfg
from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg, ContactSensorCfg, RayCasterCfg, patterns


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        #self.commands.object_pose.body_name = "panda_hand"

        '''# Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 4.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )'''
        self.scene.left =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/left",
            spawn=sim_utils.CuboidCfg(
                size=[0.05, 0.05, 0.2],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.07, 0.18), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.object =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=[0.05, 0.05, 0.2],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, 0, 0.18], rot=[1.0, 0.0, 0.0, 0.0]),
        )
        self.scene.right =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/right",
            spawn=sim_utils.CuboidCfg(
                size=[0.05, 0.05, 0.2],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, -0.07, 0.18), rot=(1.0, 0.0, 0.0, 0.0)),
        )

        self.scene.base =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/base",
            spawn=sim_utils.CuboidCfg(
                size=[0.7, 0.7, 0.08],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.00, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.wall1 =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/wall1",
            spawn=sim_utils.CuboidCfg(
                size=[2.5, 0.1, 3.],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 1.50, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.wall2 =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/wall2",
            spawn=sim_utils.CuboidCfg(
                size=[2.5, 0.3, 3.],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., -1.50, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.wall3 =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/wall3",
            spawn=sim_utils.CuboidCfg(
                size=[0.3, 3, 3.],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4, 0.00, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.scene.wall4 =  RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/wall4",
            spawn=sim_utils.CuboidCfg(
                size=[0.3, 3, 3.],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.00, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.01,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.6848517, -0.1750107, -0.1748713, 0.6853973), convention="ros"),
        )
        
        self.scene.front_cam = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.01,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.75, 0.0, 0.75), rot=(-0.1259888, 0.6960601, 0.695506, -0.1260892 ), convention="ros"),
        )



@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
