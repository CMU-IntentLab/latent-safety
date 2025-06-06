# This file is based on https://github.com/jamesjingqili/lipschitz_continuous_reachability_learning (MIT License)
# Original author: Jingqi Li
# Modified by: Kensuke Nakamura, 2025

import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic, ActorProb
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
from PyHJ.data import Batch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from PIL import Image
import io


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='dubinsRA-v0') # ra_droneracing_Game-v6, ra_highway_Game-v2, ra_1d_Game-v0
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=40000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.0)
    parser.add_argument('--total-episodes', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=20000)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--control-net', type=int, nargs='*', default=None) # for control policy
    parser.add_argument('--disturbance-net', type=int, nargs='*', default=None) # for disturbance policy
    parser.add_argument('--critic-net', type=int, nargs='*', default=None) # for critic net
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--continue-training-logdir', type=str, default=None)
    parser.add_argument('--continue-training-epoch', type=int, default=None)
    parser.add_argument('--actor-gradient-steps', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=400)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--actor-activation', type=str, default='ReLU')
    parser.add_argument('--critic-activation', type=str, default='ReLU')
    parser.add_argument('--kwargs', type=str, default='{}')
    parser.add_argument('--warm-start-path', type=str, default=None)
    # new sac params:
    parser.add_argument('--alpha', default = 0.2) 
    parser.add_argument('--auto-alpha', type=int, default=1)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    args = parser.parse_known_args()[0]
    return args



args=get_args()


env = gym.make(args.task)
# check if the environment has control and disturbance actions:
assert hasattr(env, 'action_space')
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
args.max_action = env.action_space.high[0]

args.action1_shape = env.action_space.shape or env.action_space.n
args.max_action1 = env.action_space.high[0]



train_envs = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.training_num)]
)
test_envs = DummyVectorEnv(
    [lambda: gym.make(args.task) for _ in range(args.test_num)]
)
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)
# model

if args.actor_activation == 'ReLU':
    actor_activation = torch.nn.ReLU
elif args.actor_activation == 'Tanh':
    actor_activation = torch.nn.Tanh
elif args.actor_activation == 'Sigmoid':
    actor_activation = torch.nn.Sigmoid
elif args.actor_activation == 'SiLU':
    actor_activation = torch.nn.SiLU

if args.critic_activation == 'ReLU':
    critic_activation = torch.nn.ReLU
elif args.critic_activation == 'Tanh':
    critic_activation = torch.nn.Tanh
elif args.critic_activation == 'Sigmoid':
    critic_activation = torch.nn.Sigmoid
elif args.critic_activation == 'SiLU':
    critic_activation = torch.nn.SiLU

if args.critic_net is not None:
    critic_net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.critic_net,
        activation=critic_activation,
        concat=True,
        device=args.device
    )
else:
    # report error:
    raise ValueError("Please provide critic_net!")


critic1 = Critic(critic_net, device=args.device).to(args.device)
critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
critic2 = Critic(critic_net, device=args.device).to(args.device)
critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)    
# import pdb; pdb.set_trace()
log_path = None

from PyHJ.policy import reach_avoid_SACPolicy_annealing as SACPolicy
print("SAC under the Reach-Avoid annealed Bellman equation has been loaded!")

actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
actor1 = ActorProb(
    actor1_net, 
    args.action1_shape, 
    device=args.device
).to(args.device)
actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)

if args.auto_alpha:
    target_entropy = -np.prod(env.action_space.shape)
    log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    args.alpha = (target_entropy, log_alpha, alpha_optim)

policy = SACPolicy(
critic1,
critic1_optim,
critic2,
critic2_optim,
tau=args.tau,
gamma=args.gamma,
alpha = args.alpha,
exploration_noise= None,#GaussianNoise(sigma=args.exploration_noise), # careful!
deterministic_eval = True,
estimation_step=args.n_step,
action_space=env.action_space,
actor1=actor1,
actor1_optim=actor1_optim,
)

log_path = os.path.join(args.logdir, args.task, 'baseline_sac_reach_avoid_actor_activation_{}_critic_activation_{}_game_gd_steps_{}_tau_{}_training_num_{}_buffer_size_{}_c_net_{}_{}_a1_{}_{}_gamma_{}'.format(
args.actor_activation, 
args.critic_activation, 
args.actor_gradient_steps,args.tau, 
args.training_num, 
args.buffer_size,
args.critic_net[0],
len(args.critic_net),
args.control_net[0],
len(args.control_net),
# args.alpha, always true
args.gamma)
)





# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True
)
test_collector = Collector(policy, test_envs)

if args.warm_start_path is not None:
    policy.load_state_dict(torch.load(args.warm_start_path))
    args.kwargs = args.kwargs + "warmstarted"

epoch = 0
log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
        args.exploration_noise, 
        args.actor_lr, 
        args.critic_lr, 
        args.batch_size,
        args.step_per_epoch,
        args.kwargs,
        args.seed
    )


if args.continue_training_epoch is not None:
    epoch = args.continue_training_epoch
    policy.load_state_dict(torch.load(
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "policy.pth"
        )
    ))


if args.continue_training_logdir is not None:
    policy.load_state_dict(torch.load(args.continue_training_logdir))
    # epoch = int(args.continue_training_logdir.split('_')[-9].split('_')[0])
    epoch = args.continue_training_epoch


def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(), 
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "policy.pth"
        )
    )


def stop_fn(mean_rewards):
    return False


def find_a(state):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    return act

def evaluate_V(state):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy.critic1(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()

def get_eval_plot():
    nx, ny = 51, 51
    thetas = [3*np.pi/2, 7*np.pi/4, 0, np.pi/4, np.pi/2]

    fig1, axes1 = plt.subplots(1,len(thetas), figsize=(25, 5))
    fig2, axes2 = plt.subplots(1,len(thetas), figsize=(25, 5))
    X, Y = np.meshgrid(
        np.linspace(-1., 1., nx, endpoint=True),
        np.linspace(-1., 1., ny, endpoint=True),
    )
    for i in range(len(thetas)):
        V = np.zeros_like(X)
        for ii in range(nx):
            for jj in range(ny):
                tmp_point = torch.tensor([
                            X[ii,jj],
                            Y[ii,jj],
                            math.sin(thetas[i]),
                            math.cos(thetas[i]),
                        ])
                V[ii,jj] = evaluate_V( tmp_point )
        
        axes1[i].imshow(V>0, extent=(-1., 1., -1., 1.), origin='lower')
        goal = Circle((0.5, 0.0), 0.25, color='green', fill=False)  # fill=False makes it a ring
        const = Circle((-0.5, -0.0), 0.25, color='red', fill=False)  # fill=False makes it a ring

        # Add the circle to the axes
        axes2[i].add_patch(goal)
        axes2[i].add_patch(const)
        axes2[i].imshow(V, extent=(-1., 1., -1., 1.), vmin=-1., vmax=1., origin='lower')    
        goal2 = Circle((0.5, 0.0), 0.25, color='green', fill=False)  # fill=False makes it a ring
        const2 = Circle((-0.5, -0.0), 0.25, color='red', fill=False)  # fill=False makes it a ring
        # Add the circle to the axes
        axes1[i].add_patch(goal2)
        axes1[i].add_patch(const2)  
        axes1[i].set_title('theta = {}'.format(np.round(thetas[i],2)), fontsize=12,)
        axes2[i].set_title('theta = {}'.format(np.round(thetas[i],2)), fontsize=12,)
        
            
    return fig1, fig2

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
    print("Just created the log directory!")
    # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    os.makedirs(log_path+"/epoch_id_{}".format(epoch))



gammas = np.linspace(0.99, 0.9999, endpoint=True, num=args.total_episodes)

logger = None


cont_figs =[]
binary_figs =[]
for iter in range(args.total_episodes):
    policy._gamma = gammas[iter]
    if args.continue_training_epoch is not None:
        print("episodes: {}, remaining episodes: {}".format(epoch//args.epoch, args.total_episodes - iter))
    else:
        print("episodes: {}, remaining episodes: {}".format(iter, args.total_episodes - iter))
    epoch = epoch + args.epoch
    print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
    if args.total_episodes > 1:
        writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
    else:
        if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
            print("Just created the log directory!")
            print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
            os.makedirs(log_path+"/total_epochs_{}".format(epoch))
        writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
    if logger is None:
        logger = WandbLogger()
        logger.load(writer)
    
    # import pdb; pdb.set_trace()
    result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    args.epoch,
    args.step_per_epoch,
    args.step_per_collect,
    args.test_num,
    args.batch_size,
    update_per_step=args.update_per_step,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    logger=logger
    )
    save_best_fn(policy, epoch=epoch)

    plot1, plot2 = get_eval_plot()
    wandb.log({"binary_reach_avoid_plot": wandb.Image(plot1), "continuous_plot": wandb.Image(plot2)})
    cont_figs.append(fig_to_image(plot2))
    binary_figs.append(fig_to_image(plot1))
    plt.close(plot1)
    plt.close(plot2)

# Save GIF
cont_figs[0].save(
    'cont_plot.gif',
    format='GIF',
    append_images=cont_figs[1:],
    save_all=True,
    duration=500,  # ms per frame
    loop=0
)
binary_figs[0].save(
    'binary_plot.gif',
    format='GIF',
    append_images=binary_figs[1:],
    save_all=True,
    duration=500,  # ms per frame
    loop=0
)


'''def get_eval_plot():
    grid = np.load('/home/kensuke/HJRL/new_BRT_v1_w1.25.npy')

    #thetas = [0, np.pi/6, np.pi/3, np.pi/2]
    plot_idxs = [0, 7, 14, 20]

    fig1, axes1 = plt.subplots(1,len(plot_idxs))
    fig2, axes2 = plt.subplots(1,len(plot_idxs))
    X, Y = np.meshgrid(
        np.linspace(-1.1, 1.1, grid.shape[0], endpoint=True),
        np.linspace(-1.1, 1.1, grid.shape[1], endpoint=True),
    )
    thetas_lin = np.linspace(0, 2*np.pi, grid.shape[2], endpoint=True)

    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(thetas_lin)):
        V = np.zeros_like(X)
        for ii in range(grid.shape[0]):
            for jj in range(grid.shape[1]):
                tmp_point = torch.tensor([
                            X[ii,jj],
                            Y[ii,jj],
                            thetas_lin[i],
                        ])
                V[ii,jj] = evaluate_V( tmp_point )
        
        V_grid = grid[:, :, i]
        tp_grid = np.sum((V>0) & (V_grid.T>0))
        fp_grid = np.sum((V>0) & (V_grid.T<0)) 
        fn_grid = np.sum((V<0) & (V_grid.T>0))
        tn_grid = np.sum((V<0) & (V_grid.T<0))
        tp += tp_grid
        fp += fp_grid
        fn += fn_grid
        tn += tn_grid

        prec_grid = tp_grid / (tp_grid + fp_grid) if (tp_grid + fp_grid) > 0 else 0
        rec_grid = tp_grid / (tp_grid + fn_grid) if (tp_grid + fn_grid) > 0 else 0
        f1_grid = 2 * (prec_grid * rec_grid) / (prec_grid + rec_grid) if (prec_grid + rec_grid) > 0 else 0
        
        if i in plot_idxs:
            plot_idx = plot_idxs.index(i)

            tmp_val = evaluate_V(torch.tensor([-0.8, 0, thetas_lin[i]]))
            axes1[plot_idx].imshow(V>0, extent=(-1.1, 1.1, -1.1, 1.1), origin='lower')
            axes2[plot_idx].imshow(V, extent=(-1.1, 1.1, -1.1, 1.1), vmin=-1., vmax=1., origin='lower')    
            axes1[plot_idx].set_title('theta = {}'.format(np.round(thetas_lin[i],2)), fontsize=12,)
            axes2[plot_idx].set_title('f1 = {}'.format(np.round(f1_grid,2)), fontsize=12,)
            
            axes1[plot_idx].contour(grid[:,:,i].T, levels=[0.], colors='purple', linewidths=2, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            
    print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec =  tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec+ rec) if (prec + rec) > 0 else 0
    
    return fig1, fig2, f1'''