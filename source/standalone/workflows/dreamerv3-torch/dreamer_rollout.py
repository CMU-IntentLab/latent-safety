import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
'''parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=True, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)'''


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg



from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import gymnasium as gym
from omni.isaac.lab_tasks.utils.wrappers.dreamerv3 import DreamerVecEnvWrapper
import argparse
import functools
import os
import sys
import pathlib


os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np


sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import envs.wrappers as wrappers
#from parallel import Parallel, Damy


import ruamel.yaml as yaml
from ruamel.yaml import YAML
import torch 
from torch import nn
from torch import distributions as torchd
import collections
import tools
from typing import Any, Dict
from LCRL.data import Collector, VectorReplayBuffer
from LCRL.env import DummyVectorEnv
from LCRL.exploration import GaussianNoise
from LCRL.trainer import offpolicy_trainer
from LCRL.utils import TensorboardLogger
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic, ActorProb
import LCRL.reach_rl_gym_envs as reach_rl_gym_envs
to_np = lambda x: x.detach().cpu().numpy()

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
        {"logprob", "object_state", "privileged_state", "env_ids", "success"}
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


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset, expert_dataset=None):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._expert_dataset = expert_dataset
        self._hybrid_training = expert_dataset is not None
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True, filter=None):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                if self._hybrid_training:
                    learner_data, exp_data = (
                        next(self._dataset),
                        next(self._expert_dataset),
                    )
                    
                    self._train(learner_data, expert_data=exp_data)
                else:
                    self._train(next(self._dataset))
                #self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training, filter=filter)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training, filter=None):
        safety_score = None
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)

        curr_safety_score = tools.evaluate_V(feat.detach().cpu().numpy(), filter, self.mode).item()
        print('Cur safety', curr_safety_score)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            i = 0
            if filter is not None:
                prior = self._wm.dynamics.imagine_with_action(action.unsqueeze(0), latent.copy())
                lcrl_state = self._wm.dynamics.get_feat(prior).detach().cpu().numpy()
                safety_score = tools.evaluate_V(lcrl_state, filter, self.mode).item()                 
                unsafe_score = self._wm.heads["failure"](self._wm.dynamics.get_feat(prior)).mode().item()
                print('init safety score', safety_score)

                buffer = self._config.buffer_sac if self.mode == 'sac' else self._config.buffer_ddpg
                init_safety_score = safety_score-buffer
                print('proposed ac', action)

                if safety_score < buffer:
                    safe_action = torch.tensor([tools.find_a(lcrl_state, filter)]).to(self._config.device)
                    
                    
                    #action =  (9*torch.zeros_like(safe_action) + action)/10
                    #action =  (9*safe_action + action)/10
                    action =  safe_action #(9*safe_action + action)/10
                    #action =  torch.zeros_like(safe_action) #(torch.zeros_like(safe_action) + safe_action)/2

                    '''
                    while safety_score < buffer:
                        #if unsafe_score == 1.:
                        #    break
                        #if i > 5:
                        #    action = torch.zeros_like(safe_action) #torch.tensor([tools.find_a(lcrl_state, filter)]).to(self._config.device)
                        #    break
                        print('attempt', i)
                        action = (4*torch.zeros_like(safe_action) + action)/5
                        #action = (safe_action + 4*action)/5
                        print('filtered ac', action)
                        prior = self._wm.dynamics.imagine_with_action(action.unsqueeze(0), latent.copy())
                        lcrl_state = self._wm.dynamics.get_feat(prior).detach().cpu().numpy()
                        safety_score = tools.evaluate_V(lcrl_state, filter).item()
                        i += 1'''
                print('safety score', init_safety_score)
                safety_score = init_safety_score
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action, safety_score)
        return policy_output, state
    

    def _train(self, data, expert_data=None):
        metrics = {}
        # train world model
        if self._hybrid_training and expert_data:
            mixed_data = combine_dictionaries(data, expert_data, take_half=True)
        
        # train world model on safe + unsafe_data
        post, context, mets = self._wm._train(mixed_data)
        metrics.update(mets)


        # only train the actor on its own rollouts
        start = self._wm._get_post(data) #post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        # train task behavior
        metrics.update(self._task_behavior._train(start, reward)[-1])

        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):


    from gymnasium import spaces

    # check if the environment has control and disturbance actions:
    lcrl_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,1,1536,), dtype=np.float32)
    lcrl_action1_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # control action space
    lcrl_action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32) # joint action space
    
    lcrl_state_shape = lcrl_observation_space.shape
    lcrl_action_shape = lcrl_action_space.shape 
    lcrl_max_action = lcrl_action_space.high[0]

    lcrl_action1_shape = lcrl_action1_space.shape 
    #args.action2_shape = env.action2_space.shape or env.action2_space.n
    lcrl_max_action1 = lcrl_action1_space.high[0]
    #args.max_action2 = env.action2_space.high[0]

    if config.actor_activation == 'ReLU':
        actor_activation = torch.nn.ReLU
    elif config.actor_activation == 'Tanh':
        actor_activation = torch.nn.Tanh
    elif config.actor_activation == 'Sigmoid':
        actor_activation = torch.nn.Sigmoid
    elif config.actor_activation == 'SiLU':
        actor_activation = torch.nn.SiLU

    if config.critic_activation == 'ReLU':
        critic_activation = torch.nn.ReLU
    elif config.critic_activation == 'Tanh':
        critic_activation = torch.nn.Tanh
    elif config.critic_activation == 'Sigmoid':
        critic_activation = torch.nn.Sigmoid
    elif config.critic_activation == 'SiLU':
        critic_activation = torch.nn.SiLU

    if config.critic_net is not None:
        critic_net = Net(
            lcrl_state_shape,
            lcrl_action_shape,
            hidden_sizes=config.critic_net,
            activation=critic_activation,
            concat=True,
            device=config.device
        )
    else:
        critic_net = Net(
            lcrl_state_shape,
            lcrl_action_shape,
            hidden_sizes=config.hidden_sizes,
            activation=critic_activation,
            concat=True,
            device=config.device
        )

    critic = Critic(critic_net, device=config.device).to(config.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)

    critic1 = Critic(critic_net, device=config.device).to(config.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.critic_lr)
    critic2 = Critic(critic_net, device=config.device).to(config.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config.critic_lr)    
    if config.control_net is None:
        config.control_net = config.hidden_sizes
    #if args.disturbance_net is None:
    #    args.disturbance_net = args.hidden_sizes
    if config.critic_net is None:
        config.critic_net = config.hidden_sizes
    # import pdb; pdb.set_trace()
    log_path = None

    policy_mode= config.policy_mode
    epo = config.epo

    if policy_mode == 'ddpg':   
        from LCRL.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
        print("DDPG under the Avoid annealed Bellman equation has been loaded!")


        actor1_net = Net(lcrl_state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
        actor1 = Actor(
            actor1_net, lcrl_action1_shape, max_action=lcrl_max_action1, device=config.device
        ).to(config.device)
        actor1_optim = torch.optim.Adam(actor1.parameters(), lr=config.actor_lr)

        policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=config.tau,
        gamma=config.gamma_lcrl,
        exploration_noise=GaussianNoise(sigma=config.exploration_noise),
        reward_normalization=config.rew_norm,
        estimation_step=config.n_step,
        action_space=lcrl_action_space,
        actor1=actor1,
        actor1_optim=actor1_optim,
        actor_gradient_steps=config.actor_gradient_steps,
        )
        policy.load_state_dict(torch.load('/home/kensuke/IsaacLab/test_wrist_20hz_2cam1env_newrew_128_200exp_fail_test/lcrl/1019/022354/lcrl/franka_wm-v0/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_'+str(epo)+'/policy.pth'))
        #policy.load_state_dict(torch.load('/home/kensuke/IsaacLab/test_wrist_20hz_2cam1env_newrew_128_200exp_fail_test/lcrl/1011/051410/lcrl/franka_wm-v0/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_'+str(epo)+'/policy.pth'))

    else:
        from LCRL.policy import avoid_SACPolicy_annealing as SACPolicy
        actor1_net = Net(lcrl_state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
        actor1 = ActorProb(
            actor1_net, lcrl_action1_shape, device=config.device
        ).to(config.device)
        actor1_optim = torch.optim.Adam(actor1.parameters(), lr=config.actor_lr)

        
        if config.auto_alpha:
            target_entropy = -np.prod(lcrl_action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
            config.alpha = (target_entropy, log_alpha, alpha_optim)

        policy =  SACPolicy(
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=config.tau,
            gamma=config.gamma_lcrl,
            alpha = config.alpha,
            exploration_noise= None,
            deterministic_eval = True,
            estimation_step=config.n_step,
            action_space=lcrl_action_space,
            actor1=actor1,
            actor1_optim=actor1_optim,
            )
        print("SAC under the Avoid annealed Bellman equation has been loaded!")
        policy.load_state_dict(torch.load('/home/kensuke/IsaacLab/test_wrist_20hz_2cam1env_newrew_128_200exp_fail_test/lcrl/1015/172527/lcrl/franka_wm-v0/wm_sac_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_1_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_'+str(epo)+'/policy.pth'))
    policy.eval()

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    expert_eps = collections.OrderedDict()
    tools.fill_expert_dataset(config, expert_eps)




    task = 'Isaac-Lift-Cube-Franka-IK-Rel-v0'
    env_cfg = parse_env_cfg(
#        task, use_gpu=False, num_envs=1, use_fabric= False
        task, device='cuda:0', num_envs=1, #use_fabric= False
    )
    

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    env = DreamerVecEnvWrapper(env, device = env_cfg.sim.device)
    env = wrappers.NormalizeActions(env, mode='torch')
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)


    
    
    
    acts = env.unwrapped.single_action_space #train_envs[0].action_space
    acts.low = np.ones_like(acts.low) * -1
    acts.high = np.ones_like(acts.high) # need to normalize actions 
    print(acts)
    print("Action Space", acts)
    
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    
    '''
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    0.15*torch.Tensor(-torch.ones_like(torch.tensor(acts.low))).repeat(env.num_envs, 1),
                    0.15*torch.Tensor(torch.ones_like(torch.tensor(acts.low))).repeat(env.num_envs, 1), #.repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate_vectorized(
            random_agent,
            env,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")'''
    
    
    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    expert_dataset = make_dataset(expert_eps, config)

    #print(env.single_observation_space)
    agent = Dreamer(
        env.single_observation_space,#train_envs[0].observation_space,
        env.single_action_space, #single_acts,
        #train_envs[0].action_space,
        config,
        logger,
        train_dataset,
        expert_dataset=expert_dataset if config.hybrid_training else None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        #checkpoint = torch.load(logdir / "latest.pt")
        checkpoint = torch.load(logdir / "step_137500.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    agent.mode = policy_mode
    
    '''for i in range(40):
        epo = i*10
        policy.load_state_dict(torch.load(('/home/kensuke/IsaacLab/test_wrist_20hz_2cam1env_newrew_128_200exp_fail_test/lcrl/1011/051410/lcrl/franka_wm-v0/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_'+str(epo)+'/policy.pth')))

        video_pred = agent._wm.video_pred(next(eval_dataset), policy = policy)
        logger.video("eval_openl"+str(epo), to_np(video_pred))

        video_exp_pred = agent._wm.video_pred(next(expert_dataset), policy = policy)
        logger.video("expert_openl"+str(epo), to_np(video_exp_pred))
        logger.write()
    exit()'''
    
    eval_policy = functools.partial(agent, training=False, filter=policy)
    env.reset()
    test_eps = collections.OrderedDict()

    tools.simulate_vectorized(
        eval_policy,
        env,#eval_envs,
        test_eps,
        config.evaldir,
        logger,
        is_eval=True,
        episodes= 1,#config.eval_episode_num,
    )
    exit()
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False, filter=policy)
            tools.simulate_vectorized(
                eval_policy,
                env,#eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                print('logging video')
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

                video_exp_pred = agent._wm.video_pred(next(expert_dataset))
                logger.video("expert_openl", to_np(video_exp_pred))
        
        print("Start training.")
        state = tools.simulate_vectorized(
            agent,
            env,#train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        ckpt = "step_" + str(agent._step*config.action_repeat) + ".pt"
        print(ckpt)
        torch.save(items_to_save, logdir / ckpt)

   
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--livestream")
    args, remaining = parser.parse_known_args()
    yaml = YAML(typ='safe', pure=True)
    config_path = pathlib.Path(sys.argv[0]).parent / "configs.yaml"
    print(config_path)
    with open(config_path, 'r') as file:
        configs = yaml.load(file)
    #configs = yaml.load(
    #    (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    #)

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
    simulation_app.close()

