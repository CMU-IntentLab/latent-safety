import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
odp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../optimized_dp"))
sys.path.append(odp_dir)
import yaml

import imp
import numpy as np

# Utility functions to initialize the problem
from optimized_dp.odp.Grid import Grid
from optimized_dp.odp.Shapes import *

# Specify the  file that includes dynamic systems
from optimized_dp.odp.dynamics import DubinsCapture
from optimized_dp.odp.dynamics import DubinsCar4D2
from optimized_dp.odp.dynamics import DubinsCar

# Plot options
from optimized_dp.odp.Plots import PlotOptions
from optimized_dp.odp.Plots import plot_isosurface, plot_valuefunction

# Solver core
from optimized_dp.odp.solver import HJSolver, computeSpatDerivArray
from networks.mlp import MLP
import torch
import math
import argparse

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- Initialize plotting option
- Call HJSolver function
"""


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def main(config, experiment_prefix):
    num_pts = config["num_pts"]
    x_min = config["x_min"]
    x_max = config["x_max"]
    y_min = config["y_min"]
    y_max = config["y_max"]
    u_max = config["u_max"]
    dt = config["dt"]
    v = config["speed"]

    nx = config["nx"]
    ny = config["ny"]
    nz = config["nz"]

    # Third scenario with Reach-Avoid set
    g = Grid(
        np.array([x_min, y_min, 0]),
        np.array([x_max, y_max, 2 * math.pi]),
        3,
        np.array([nx, ny, nz]),
        [2],
    )

    goal = CylinderShape(g, [2], [config["obs_x"], config["obs_y"], 0], config["obs_r"])

    lookback_length = config["numT"] * config["dt"]
    t_step = 0.05

    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    my_car = DubinsCar(
        uMode="max", dMode="min", wMax=config["u_max"], speed=config["speed"]
    )

    po2 = PlotOptions(
        do_plot=False,
        plot_type="set",
        plotDims=[0, 1, 2],
        slicesCut=[],
        save_fig=False,
        filename="/logs/grid/plot.png",
    )

    compMethods = {
        "TargetSetMode": "minVWithVTarget",
    }
    # HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
    result = HJSolver(
        my_car,
        g,
        goal,
        tau,
        compMethods,
        po2,
        saveAllTimeSteps=False,
        accuracy="medium",
    )

    path = (
        "logs/grid/LS_BRT_v"
        + ("_" + experiment_prefix if experiment_prefix else "")
        + str(config["speed"])
        + "_w"
        + str(config["u_max"])
        + ".npy"
    )
    np.save(
        path,
        result,
    )
    print("Saved to", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args = parser.parse_args()

    config_path = "/home/lassepe/worktree/latent-safety/configs/config.yaml"
    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    final_config = {}

    for name in name_list:
        recursive_update(final_config, configs[name])
    # merge all args.configs into a experiment prefix string
    nondefault_names = [name for name in name_list if name != "defaults"]
    
    experiment_prefix = "_".join(nondefault_names)
    print(experiment_prefix)
    main(final_config, experiment_prefix)
