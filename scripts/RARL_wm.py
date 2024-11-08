"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 3-dimensional Dubins car
problem. We use this script to generate Fig. 5 in the paper.

Examples:
    RA: python3 sim_car_one.py -sf -of scratch -w -wi 5000 -g 0.9999 -n 9999
    test: python3 sim_car_one.py -sf -of scratch -w -wi 50 -mu 1000 -cp 400
        -n tmp
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../safety_rl'))
sys.path.append(saferl_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
print(sys.path)
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ruamel.yaml as yaml

from safety_rl.RARL.DDQNSingle import DDQNSingle
from safety_rl.RARL.config import dqnConfig
from safety_rl.RARL.utils import save_obj
from safety_rl.gym_reachability import gym_reachability  # Custom Gym env.
import model_based_irl_torch.dreamer.models as models
import model_based_irl_torch.dreamer.tools as tools

import argparse
import collections
import copy
import warnings
import functools
import time
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from termcolor import cprint

debugging_eval = True

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

def RARL(config):
  # == ARGS ==

  # == CONFIGURATION ==
  env_name = "dubins_car_img-v1"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  maxUpdates = config.maxUpdates
  updateTimes = config.updateTimes
  updatePeriod = int(maxUpdates / updateTimes)
  updatePeriodHalf = int(updatePeriod / 2)
  maxSteps = config.numT #100

  # == Environment ==
  print("\n== Environment Information ==")
  if config.doneType == 'toEnd':
    sample_inside_obs = True
  elif config.doneType == 'TF' or config.doneType == 'fail':
    sample_inside_obs = False

  print(env_name)
  print(gym_reachability)
  env = gym.make(
      env_name, config=config, device=device, mode=config.mode, doneType=config.doneType,
      sample_inside_obs=sample_inside_obs
  )


  fn = config.name + '-' + config.doneType
  if config.showTime:
    fn = fn + '-' + timestr

  if config.learnedMargin:
    env.car.learned_margin=True
    fn = fn + '-lm'
  else:
    env.car.learned_margin=False

  if config.learnedDyn:
    env.car.learned_dyn=True
    fn = fn + '-ld'
  else:
    env.car.learned_dyn=False

  if config.image:
    env.car.image=True
    env.car.set_encoder()
    fn = fn + '-img'
  if config.debug:
    env.car.debug = True

  if config.gt_lx:
    env.car.gt_lx = True
    fn = fn + '-gtlx'


  if config.wm:
    wm = models.WorldModel(env.observation_space, env.action_space, 0, config)
    checkpoint = torch.load(config.from_ckpt)
    wm.dynamics.sample = False

    state_dict = {k[14:]:v for k,v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    lx_mlp, _ = wm._init_lx_mlp(config, 1)
    lx_ckpt= torch.load(config.lx_ckpt)
    lx_mlp.load_state_dict(lx_ckpt['agent_state_dict'])
    env.car.set_wm(wm, lx_mlp, config)


  outFolder = os.path.join(config.outFolder, 'car-DDQN', fn)
  print(outFolder)
  figureFolder = os.path.join(outFolder, 'figure')
  os.makedirs(figureFolder, exist_ok=True)


  stateDim = env.state.shape[0]
  if config.wm:
    if config.dyn_discrete:
      stateDim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      stateDim = config.dyn_stoch + config.dyn_deter
  actionNum = env.action_space.n
  actionList = np.arange(actionNum)
  print(
      "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
          stateDim, actionNum
      )
  )

  # == Setting in this Environment ==
  env.set_speed(speed=config.speed)
  env.set_constraint(radius=config.obs_r)
  env.set_radius_rotation(R_turn=config.speed/config.u_max)
  print("Dynamic parameters:")
  print("  CAR", end='\n    ')
  print(
      "Constraint: {:.1f} ".format(env.car.constraint_radius)
      + "Turn: {:.2f} ".format(env.car.R_turn)
      + "Max speed: {:.2f} ".format(env.car.speed)
      + "Max angular speed: {:.3f}".format(env.car.max_turning_rate)
  )
  print("  ENV", end='\n    ')
  print(
      "Constraint: {:.1f} ".format(env.constraint_radius)
      + "Turn: {:.2f} ".format(env.R_turn)
      + "Max speed: {:.2f} ".format(env.speed)
  )
  print(env.car.discrete_controls)
  if 2 * env.R_turn - env.constraint_radius > env.target_radius:
    print("Type II Reach-Avoid Set")
  else:
    print("Type I Reach-Avoid Set")
  env.set_seed(config.randomSeed)
  
  # == Get and Plot max{l_x, g_x} ==
  if (config.plotFigure or config.storeFigure) and not debugging_eval:
    nx, ny = 51, 51
    
    v = np.zeros((nx, ny))
    #l_x = np.zeros((nx, ny))
    g_x = np.zeros((nx, ny))
    xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
    ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

    it = np.nditer(v, flags=['multi_index'])
    ###
    idxs = []  
    imgs = []
    thetas = []
    it = np.nditer(v, flags=["multi_index"])
    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]
      theta = np.random.random()*2*np.pi
      assert theta > 0 and theta < 2*np.pi
      thetas.append(theta)
      if env.car.use_wm:
        imgs.append(env.capture_image(np.array([x, y, theta])))
        idxs.append(idx)        
      it.iternext()
    idxs = np.array(idxs)
    x_lin = xs[idxs[:,0]]
    y_lin = ys[idxs[:,1]]
    theta_lin = np.array(thetas)
    
    if config.gt_lx:
      raise Exception('Not implemented')
      g_x = env.car.gt_safety_margin(np.array([x_lin, y_lin]))
    else:   
      g_x, _, _ = env.car.get_latent(x_lin, y_lin, theta_lin, imgs)

###
    v[idxs[:, 0], idxs[:, 1]] = g_x
    g_x = v

    vmax = round(max(np.max(g_x), 0),1)
    vmin = round(min(np.min(g_x), -vmax),1)
    axStyle = env.get_axes()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    im = ax.imshow(
        g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=vmin, vmax=vmax, zorder=-1
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$g(x)$', fontsize=18)

    ax = axes[1]
    im = ax.imshow(
        v.T > 0, interpolation='none', extent=axStyle[0], origin="lower",
        cmap="seismic", vmin=-1, vmax=1, zorder=-1
    )
    cbar = fig.colorbar(
        im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
    )
    cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
    ax.set_title(r'$v(x)$', fontsize=18)

    for ax in axes:
      env.plot_target_failure_set(ax=ax)
      env.plot_formatting(ax=ax)

    fig.tight_layout()
    if config.storeFigure:
      figurePath = os.path.join(figureFolder, 'env.png')
      fig.savefig(figurePath)
    if config.plotFigure:
      plt.show()
      plt.pause(0.001)
    plt.close()
  
  # == Agent CONFIG ==
  print("\n== Agent Information ==")
  if config.annealing:
    GAMMA_END = 0.9999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = config.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

  CONFIG = dqnConfig(
      DEVICE=device, ENV_NAME=env_name, SEED=config.randomSeed,
      MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
      MEMORY_CAPACITY=config.memoryCapacity, ARCHITECTURE=config.architecture,
      ACTIVATION=config.actType, GAMMA=config.gamma, GAMMA_PERIOD=updatePeriod,
      GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
      EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=config.learningRate,
      LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=50
  )

  # == AGENT ==
  dimList = [stateDim] + list(CONFIG.ARCHITECTURE) + [actionNum]
  
  agent = DDQNSingle(
      CONFIG, actionNum, actionList, dimList=dimList, mode=config.mode,
      terminalType=config.terminalType
  )
  print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
  print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

  vmin = -1
  vmax = 1
  if config.warmup and not debugging_eval:
    print("\n== Warmup Q ==")
    lossList = agent.initQ(
        env, config.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
        vmax=vmax, plotFigure=config.plotFigure, storeFigure=config.storeFigure
    )

    if config.plotFigure or config.storeFigure:
      fig, ax = plt.subplots(1, 1, figsize=(4, 4))
      tmp = np.arange(25, config.warmupIter)
      #tmp = np.arange(config.warmupIter)
      ax.plot(tmp, lossList[tmp], 'b-')
      ax.set_xlabel('Iteration', fontsize=18)
      ax.set_ylabel('Loss', fontsize=18)
      plt.tight_layout()

      if config.storeFigure:
        figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
        fig.savefig(figurePath)
      if config.plotFigure:
        plt.show()
        plt.pause(0.001)
      plt.close()

  asdf = env.simulate_trajectories(
    agent.Q_network, T=maxSteps, num_rnd_traj=10,
    toEnd=False, enable_observation_feedback=True,
    wait_for_all_metrics_to_predict_failure=True,
    return_infos=True
  )
  assert False

  print("\n== Training Information ==")
  vmin = -1
  vmax = 1
  trainRecords, trainProgress = agent.learn(
      env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
      doneTerminate=True, vmin=vmin, vmax=vmax, showBool=True,
      checkPeriod=config.checkPeriod, outFolder=outFolder,
      plotFigure=config.plotFigure, storeFigure=config.storeFigure
  )

  trainDict = {}
  trainDict['trainRecords'] = trainRecords
  trainDict['trainProgress'] = trainProgress
  filePath = os.path.join(outFolder, 'train')

  # region: loss
  if config.plotFigure or config.storeFigure:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    data = trainRecords
    ax = axes[0]
    ax.plot(data, 'b:')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
    ax.set_title('loss_critic', fontsize=18)
    ax.set_xlim(left=0, right=maxUpdates)

    data = trainProgress[:, 0]
    ax = axes[1]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    ax.set_title('Success Rate', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])
    ax.set_ylim(0, 0.8)

    fig.tight_layout()
    if config.storeFigure:
      figurePath = os.path.join(figureFolder, 'train_loss_success.png')
      fig.savefig(figurePath)
    if config.plotFigure:
      plt.show()
      plt.pause(0.001)
    plt.close()
    # endregion

    # region: value_rollout_action
    idx = np.argmax(trainProgress[:, 0]) + 1
    successRate = np.amax(trainProgress[:, 0])
    print('We pick model with success rate-{:.3f}'.format(successRate))
    agent.restore(idx * config.checkPeriod, outFolder)

    nx = 41
    ny = nx
    xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
    ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

    resultMtx = np.empty((nx, ny), dtype=int)
    actDistMtx = np.empty((nx, ny), dtype=int)
    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
      idx = it.multi_index
      print(idx, end='\r')
      x = xs[idx[0]]
      y = ys[idx[1]]

      state = np.array([x, y, 0.])
      # TODO: this is wrong: for deraimer, the Q_network expects a latent state, not the actual state of the env
      # how to get this latent state from a regular state?
      stateTensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
      action_index = agent.Q_network(stateTensor).max(dim=1)[1].item()
      actDistMtx[idx] = action_index

      _, result, _, _ = env.simulate_one_trajectory(
          agent.Q_network, T=250, state=state, toEnd=False
      )
      resultMtx[idx] = result
      it.iternext()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axStyle = env.get_axes()

    # = Action
    ax = axes[2]
    im = ax.imshow(
        actDistMtx.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap='seismic', vmin=0, vmax=actionNum - 1, zorder=-1
    )
    ax.set_xlabel('Action', fontsize=24)

    # = Rollout
    ax = axes[1]
    im = ax.imshow(
        resultMtx.T != 1, interpolation='none', extent=axStyle[0],
        origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1
    )
    env.plot_trajectories(
        agent.Q_network, states=env.visual_initial_states, toEnd=False, ax=ax,
        c='w', lw=1.5, T=100, orientation=-np.pi / 2
    )
    ax.set_xlabel('Rollout RA', fontsize=24)

    # = Value
    ax = axes[0]
    v = env.get_value(agent.Q_network, theta=0, nx=nx, ny=ny)
    im = ax.imshow(
        v.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1
    )
    CS = ax.contour(
        xs, ys, v.T, levels=[0], colors='k', linewidths=2, linestyles='dashed'
    )
    ax.set_xlabel('Value', fontsize=24)

    for ax in axes:
      env.plot_target_failure_set(ax=ax)
      env.plot_reach_avoid_set(ax=ax)
      env.plot_formatting(ax=ax)

    fig.tight_layout()
    if config.storeFigure:
      figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
      fig.savefig(figurePath)
    if config.plotFigure:
      plt.show()
      plt.pause(0.001)
    # endregion

    trainDict['resultMtx'] = resultMtx
    trainDict['actDistMtx'] = actDistMtx

  save_obj(trainDict, filePath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    # environment parameters
    config, remaining = parser.parse_known_args()

    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    print(defaults.keys())
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    #final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    print("---------------------")
    RARL(final_config)