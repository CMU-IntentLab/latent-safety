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

import os 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../safety_rl'))
sys.path.append(saferl_dir)
print(sys.path)

from safety_rl.gym_reachability import gym_reachability  # Custom Gym env.


import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint
from torch import distributions as torchd
from tqdm import trange

import dreamer.tools as tools
import envs.wrappers as wrappers
from common.constants import HORIZONS, IMAGE_OBS_KEYS
from common.utils import (
    create_shape_meta,
    get_robomimic_dataset_path_and_env_meta,
    to_np,
    combine_dictionaries,
)
from dreamer.dreamer import Dreamer
from dreamer.parallel import Damy, Parallel
#from environments.env_make import make_env_robomimic
from termcolor import cprint
from functools import partial
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from dreamer.tools import ModelEvaluator

# os.environ["MUJOCO_GL"] = "osmesa"
from gym import spaces
from gym.spaces import Discrete 
import gym


def train_eval(config):
    # ==================== Set up training ====================
    training_setup = set_up_dreamer_training(config)
    logger = training_setup["logger"]
    agent = training_setup["agent"]
    train_envs = training_setup["train_envs"]
    eval_envs = training_setup["eval_envs"]
    eval_dataset = training_setup["eval_dataset"]
    obs_train_dataset = training_setup["obs_train_dataset"]
    obs_eval_dataset = training_setup["obs_eval_dataset"]
    logdir = training_setup["logdir"]
    train_eps = training_setup["train_eps"]
    expert_dataset = training_setup["expert_dataset"]


    # ==================== Training Fns ====================
    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        logger.image("pretrain/" + title, np.transpose(plot_arr, (2, 0, 1)))

    def eval_obs_recon():
        recon_steps = 101
        obs_mlp, obs_opt = agent._wm._init_obs_mlp(config, 3)
        train_loss = []
        eval_loss = []
        for i in range(recon_steps):
            if i % int(recon_steps/4) == 0:
                new_loss = agent.pretrain_regress_obs(
                    next(obs_eval_dataset), obs_mlp, obs_opt, eval=True
                )
                eval_loss.append(new_loss)
            else:
                new_loss = agent.pretrain_regress_obs(
                    next(obs_train_dataset), obs_mlp, obs_opt
                )
                train_loss.append(new_loss)
        log_plot("train_recon_loss", train_loss)
        log_plot("eval_recon_loss", eval_loss)
        logger.scalar("pretrain/train_recon_loss_min", np.min(train_loss))
        logger.scalar("pretrain/eval_recon_loss_min", np.min(eval_loss))
        logger.write(step=logger.step)
        del obs_mlp, obs_opt  # dont need to keep these
        return np.min(eval_loss)
    def train_lx(ckpt_name, log_dir):
        recon_steps = 2501
        best_pretrain_success_classifier = float("inf")
        lx_mlp, lx_opt = agent._wm._init_lx_mlp(config, 1)
        train_loss = []
        eval_loss = []
        for i in range(recon_steps):
            if i % 250 == 0:
                print('eval')
                new_loss, eval_plot = agent.train_lx(
                    next(obs_eval_dataset), lx_mlp, lx_opt, eval=True
                )
                eval_loss.append(new_loss)
                logger.image("classifier", np.transpose(eval_plot, (2, 0, 1)))
                logger.write(step=i+40000)
                best_pretrain_success_classifier = tools.save_checkpoint(
                    ckpt_name, i, new_loss, best_pretrain_success_classifier, lx_mlp, logdir
                )

            else:
                new_loss, _ = agent.train_lx(
                    next(obs_train_dataset), lx_mlp, lx_opt
                )
                train_loss.append(new_loss)
        log_plot("train_lx_loss", train_loss)
        log_plot("eval_lx_loss", eval_loss)
        logger.scalar("pretrain/train_lx_loss_min", np.min(train_loss))
        logger.scalar("pretrain/eval_lx_loss_min", np.min(eval_loss))
        logger.write(step=i)
        print(eval_loss)
        print('logged')
        return lx_mlp, lx_opt

    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        print(
            f"Evaluating for Seeds: {config.eval_num_seeds} and Evals per seed: {config.eval_per_seed}"
        )
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_recon/openl_agent", to_np(video_pred))

            if other_dataset:
                video_pred = agent._wm.video_pred(next(other_dataset))
                logger.video("train_recon/openl_agent", to_np(video_pred))

        logger.scalar(
            f"{eval_prefix}/eval_episodes", config.eval_num_seeds * config.eval_per_seed
        )
        logger.write(step=logger.step)
        recon_eval = eval_obs_recon()  # testing observation reconstruction

        agent.train()
        return recon_eval

    def collect_rollouts(state, num_steps):
        agent.eval()
        eval_policy = functools.partial(agent, training=False)
        state = tools.simulate(
            eval_policy,
            # agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=num_steps,
            state=state,
            batch_collect=True,
        )
        agent.train()
        return state

    # ==================== Actor Pretrain ====================
    total_pretrain_steps = config.pretrain_joint_steps + config.pretrain_actor_steps
    print(total_pretrain_steps)
    if total_pretrain_steps > 0:
        assert not (config.pretrain_on_random and config.pretrain_on_random_mixed)
        if config.pretrain_on_random or config.pretrain_on_random_mixed:
            assert (
                config.offline_traindir is not None
            ), "Need to load in random data to be trained"
        cprint(
            f"Pretraining for {config.pretrain_joint_steps=}, {config.pretrain_actor_steps=}",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = (  # noqa: E731
            lambda step: "pretrain_joint"
            if step < config.pretrain_joint_steps
            else "pretrain_actor"
        )
        best_pretrain_score = float("inf")
        for step in trange(
            total_pretrain_steps,
            desc="Encoder + Actor pretraining",
            ncols=0,
            leave=False,
        ):
            if (
                config.eval_num_seeds > 0
                and ((step + 1) % config.eval_every) == 0
                or step == 1
                # and step > 0
            ):
               
                print('eval')
                score = evaluate(
                    other_dataset=expert_dataset, eval_prefix="pretrain"
                )
                best_pretrain_score = tools.save_checkpoint(
                    ckpt_name, step, score, best_pretrain_score, agent, logdir
                )

    
            exp_data = next(expert_dataset)

            agent.pretrain_model_only(exp_data, step)


        close_envs(train_envs + eval_envs)
    
    # ==================== Training l(x) classifier ====================
    print('training l(x)')
    lx_mlp, lx_opt = train_lx('classifier', logdir)
    exit()

    '''
    # ==================== Prefill dataset ====================
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        print(type(acts))
        print(type(acts) == Discrete)
        #if hasattr(acts, "discrete"):
        if type(acts) == Discrete:
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    # ==================== "Seeding" model on exp+random ====================
    if config.initial_joint_train_steps > 0:
        if config.seed_in_batches:
            collect_rollouts(None, config.initial_joint_train_steps)
        else:
            # collect just enough for first batch
            seeding_state = collect_rollouts(
                None, config.batch_size * config.batch_length
            )
        for _ in trange(
            config.initial_joint_train_steps, ncols=0, desc="Train model on exp+random"
        ):
            expert_data = next(expert_dataset)
            random_data = next(train_dataset)
            mixed_data = combine_dictionaries(expert_data, random_data)
            agent._train_model_only(mixed_data)
            if not config.seed_in_batches:
                seeding_state = collect_rollouts(seeding_state, num_steps=1)
        tools.save_checkpoint("initial_joint_train", 0, 0, 0, agent, logdir)

    # ==================== Main Training Loop ====================
    # replace actor with ensemble
    if config.train_residuals:
        agent._task_behavior.replace_actor_with_ensemble(device=config.device)
    evaluate(other_dataset=train_dataset, eval_prefix="main")

    # setup num training steps
    if config.no_joint_steps:
        assert config.model_only_scale > 0.0 or config.actor_only_scale > 0.0
        joint_steps = 0
    else:
        joint_steps = agent._batch_train_steps
    model_only_steps = int(config.model_only_scale * agent._batch_train_steps)
    actor_only_steps = int(config.actor_only_scale * agent._batch_train_steps)
    reward_only_steps = model_only_steps if config.separate_reward_training else 0
    total_train_steps = (
        model_only_steps + reward_only_steps + actor_only_steps + joint_steps
    )
    cprint(
        f"{model_only_steps=}, {reward_only_steps=}, {actor_only_steps=}, {joint_steps=}",
        "cyan",
        attrs=["bold"],
    )

    # setup training metrics
    ckpt_name = "main"
    _iter, total_steps = 0, 0
    best_train_success, success = -float("inf"), 0
    tbar = trange(int(config.steps), ncols=0, desc="Main Training Loop")
    print("===================Start Main Training Loop===================")
    while total_steps < config.steps:
        # collect batches of data using eval_policy
        print("Batch Env Steps")
        state = collect_rollouts(state, num_steps=config.steps_per_batch)

        if config.train_residuals:
            agent._task_behavior.add_new_residual()

        for step in trange(total_train_steps, desc="Model+Actor Updates", ncols=0):
            # get data
            expert_data = possibly_get_expert_data(
                expert_dataset, config, step, model_only_steps, joint_steps
            )
            learner_data = next(train_dataset)

            # decide on train function
            if step < model_only_steps:
                train_fn = (
                    functools.partial(
                        agent._train_model_only, frozen_heads=["reward", "cont"]
                    )
                    if config.separate_reward_training
                    else agent._train_model_only
                )
                only_model_train = True
            elif step < model_only_steps + reward_only_steps:
                train_fn = agent._train_reward_only
                only_model_train = True
            elif step < model_only_steps + joint_steps:
                train_fn = agent._train
                only_model_train = False
            else:
                train_fn = agent._train_actor_only
                only_model_train = False

            # fit model/critic/actor
            if not only_model_train and (
                config.hybrid_critic_fitting or config.utd_ratio > 1
            ):
                agent._train_critic_only(
                    train_dataset, expert_dataset, utd_ratio=config.utd_ratio
                )
            train_fn(learner_data, expert_data=expert_data)

            # update metrics
            agent._step += 1
            agent._update_count += 1
            agent._logger.step += 1
            agent._metrics["update_count"] = agent._update_count
            agent._maybe_log_metrics(video_pred_log=config.video_pred_log)

            if (
                config.critic_reset_every > 0
                and agent._step % config.critic_reset_every == 0
            ):
                cprint("Resetting last three layers of critic", "cyan", attrs=["bold"])
                agent._task_behavior.reset_critics()

            if (
                # step % config.eval_every == 0 or step + 1 == total_train_steps
                agent._step % config.eval_every == 0
            ) and config.eval_num_seeds > 0:
                if not only_model_train:
                    eval_prefix = "main" if step + 1 == total_train_steps else "iter"
                    score, success = evaluate(
                        other_dataset=train_dataset, eval_prefix=eval_prefix
                    )
                best_train_success = tools.save_checkpoint(
                    ckpt_name, step, success, best_train_success, agent, logdir
                )
        if config.bc_reg:
            agent._metrics["bc_reg_weight"] = agent._task_behavior.decay_bc_weight()
        total_steps += config.steps_per_batch
        tbar.update(config.steps_per_batch)
        _iter += 1
        logger.write()
    '''
    close_envs(train_envs + eval_envs)

def set_up_dreamer_training(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # ==================== Logging ====================
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(f"{logdir}/config.yaml", "w") as f:
        yaml.dump(vars(config), f)
    # step in logger is environmental step
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    if config.debug:
        logger = tools.DebugLogger(logdir, config.action_repeat * step)
    else:
        logger = tools.Logger(logdir, config.action_repeat * step)
    logger.config(vars(config))
    logger.write()

    # ==================== Create dataset ====================
    # expert replay buffer
    expert_eps = collections.OrderedDict()
    print(expert_eps)
    tools.fill_expert_dataset_dubins(config, expert_eps)
    expert_dataset = make_dataset(expert_eps, config)
    # validation replay buffer
    expert_val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_dubins(config, expert_val_eps, is_val_set=True)

    # split expert dataset into train and eval
    obs_train_eps = collections.OrderedDict()
    obs_eval_eps = collections.OrderedDict()
    for i, (key, value) in enumerate(expert_eps.items()):
        if i < int(len(expert_eps) * 0.7):
            obs_train_eps[key] = value
        else:
            obs_eval_eps[key] = value

    obs_eval_dataset = make_dataset(obs_eval_eps, config)
    obs_train_dataset = make_dataset(obs_train_eps, config)

    # learner + eval replay buffer
    if config.offline_traindir:
        # possibly replace 'data/{dataset}/{model}" with keys in config
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.prefill)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(expert_val_eps, config)

    # ==================== Create envs ====================

    # == CONFIGURATION ==
    #env_name = "dubins_car_img_cont-v1"
    env_name = "dubins_car_img-v1" # showing Lasse the DDQN version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maxUpdates = config.maxUpdates
    updateTimes = config.updateTimes
    updatePeriod = int(maxUpdates / updateTimes)
    updatePeriodHalf = int(updatePeriod / 2)

    # == Environment ==
    print("\n== Environment Information ==")
    if config.doneType == 'toEnd':
        sample_inside_obs = True
    elif config.doneType == 'TF' or config.doneType == 'fail':
        sample_inside_obs = False

    print(env_name)
    print(gym_reachability)
    train_env = gym.make(
        env_name, config=config, device=device, mode=config.mode, doneType=config.doneType,
        sample_inside_obs=sample_inside_obs
    )

    print(train_env.observation_space)
    print(train_env.action_space)
    train_envs = [train_env]
    eval_env = gym.make(
        env_name, config=config, device=device, mode=config.mode, doneType=config.doneType,
        sample_inside_obs=sample_inside_obs
    )
    eval_envs = [eval_env]

    '''train_envs = [make_env(config) for i in range(config.envs)]
    eval_envs = [make_env(config) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space'''
    ### TODO: CLEAN ABOVE ###
    acts = train_envs[0].action_space
    bounds = np.array([[-1.1, 1.1], [-1.1, 1.1], [0, 2 * np.pi]])
    low = bounds[:, 0]
    high = bounds[:, 1]

    # Gym variables.
    midpoint = (low + high) / 2.0
    interval = high -low
    observation_space = spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )

    print(f"Action Space: {acts}.")# Low: {acts.low}. High: {acts.high}")
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # ==================== Create Agent ====================
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        expert_dataset=expert_dataset if config.hybrid_training else None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
        checkpoint = torch.load(config.from_ckpt)
        if config.critic_ensemble_size > 1 or config.reward_ensemble_size > 1:
            # only false if loading with different ensemble size
            mk, uk = agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
            for k in mk:
                assert (
                    "Value" in k
                    or "_slow_value" in k
                    or "value" in k
                    or "Reward" in k
                    or "reward" in k
                )
            for k in uk:
                assert (
                    "Value" in k
                    or "_slow_value" in k
                    or "value" in k
                    or "Reward" in k
                    or "reward" in k
                )
        else:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        try:
            tools.recursively_load_optim_state_dict(
                agent, checkpoint["optims_state_dict"]
            )
        except Exception as e:
            # likely due to mismatch in pretrain optimizers
            print("Failed to load optim state dict", e)

        pretrained_ckpt_config = Path(config.from_ckpt).parent / "config.yaml"
        if pretrained_ckpt_config.exists():
            pretrained_config = yaml.load(pretrained_ckpt_config.read_text())
            if pretrained_config["num_exp_trajs"] == config.num_exp_trajs:
                warnings.warn(
                    f"Mismatch in number of expert trajectories in pretrained config {pretrained_config['num_exp_trajs']} and actual {config.num_exp_trajs}"
                )

        agent._should_pretrain._once = False
        if config.from_ckpt and config.pretrain_ema:
            print("Using EMA weights from pretraining")
            agent.ema.load_state_dict(checkpoint["ema"])
            agent.ema.copy_to(agent._task_behavior.actor.parameters())


    training_setup = {
        "agent": agent,
        "train_envs": train_envs,
        "eval_envs": eval_envs,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "expert_dataset": expert_dataset,
        "obs_train_dataset": obs_train_dataset,
        "obs_eval_dataset": obs_eval_dataset,
        "logger": logger,
        "logdir": logdir,
        "train_eps": train_eps,
    }

    return training_setup


def close_envs(envs):
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def possibly_get_expert_data(
    expert_dataset, config, step, model_only_steps, joint_steps
):
    """Helper function to decide when to get the expert dataset,
    preventing unnecessary calls to next(expert_dataset)"""
    bc_loss_cond = (
        config.train_residuals or config.bc_reg
    ) and step >= model_only_steps

    # TODO: VERIFY CORRECTNESS
    hybrid_fitting_cond = (
        config.hybrid_training and step < model_only_steps + joint_steps
    )

    if bc_loss_cond or hybrid_fitting_cond:
        return next(expert_dataset)
    return None


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def make_env(config):
    suite, task = config.task.split("_", 1)
    assert suite == "robomimic", f"Unknown suite {suite}"
    assert task in HORIZONS.keys(), f"Unknown task {task}"
    dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id=str(task).upper(),
        shaped=config.shape_rewards,
        image_size=config.image_size,
        done_mode=config.done_mode,
    )
    shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)

    shape_rewards = config.shape_rewards
    shift_rewards = config.shift_rewards
    env = make_env_robomimic(
        env_meta,
        IMAGE_OBS_KEYS,
        shape_meta,
        add_state=True,
        reward_shift_wrapper=shift_rewards,
        reward_shaping=shape_rewards,
        offscreen_render=False,
    )
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
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

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Hybrid Training: {final_config.hybrid_training}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    cprint(f"WM Dataset Path: {final_config.wm_dataset_path}", "cyan", attrs=["bold"])
    print("---------------------")

    train_eval(final_config)