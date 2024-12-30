import argparse
import collections
import os
import pathlib
import sys
import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint
import cv2
# add to os sys path
import sys
import matplotlib.pyplot as plt
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../real_envs'))
sys.path.append(env_dir)
print(dreamer_dir)
print(sys.path)
import model_based_irl_torch.dreamer.tools as tools
from model_based_irl_torch.dreamer.dreamer import Dreamer
from termcolor import cprint
from real_envs.env_utils import normalize_eef_and_gripper, unnormalize_eef_and_gripper, get_env_spaces
import pickle
from collections import defaultdict
from model_based_irl_torch.dreamer.tools import add_to_cache
from tqdm import tqdm, trange
from model_based_irl_torch.common.utils import to_np
import wandb
from test_loader import SplitTrajectoryDataset
from torch.utils.data import DataLoader
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Lipschitz_Continuous_Reachability_Learning'))
sys.path.append(saferl_dir)
import gymnasium #as gym
import gym
import requests
from PIL import Image
from torchvision import transforms

import torch
from torch import nn
from torch.optim import AdamW

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import imageio.v3 as iio


transform = transforms.Compose([           
                                transforms.Resize(256),                    
                                transforms.CenterCrop(224),               
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])


transform1 = transforms.Compose([           
                                transforms.Resize(520),
                                transforms.CenterCrop(518), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])





import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from train_DINO_decoder import Decoder
from train_DINO_wm import ViT
from train_DINO_wm_claude import VideoTransformer


# helpers




DINO_transform = transforms.Compose([           
                                transforms.Resize(224),
                                #transforms.CenterCrop(224), #should be multiple of model patch_size                 
                                
                                transforms.ToTensor(),])


from LCRL.data import Collector, VectorReplayBuffer
from LCRL.env import DummyVectorEnv
from LCRL.exploration import GaussianNoise
from LCRL.trainer import offpolicy_trainer
from LCRL.utils import TensorboardLogger
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic
import LCRL.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
from pathlib import Path
import collections
#from dreamer import make_dataset
# NOTE: all the reach-avoid gym environments are in reach_rl_gym, the constraint information is output as an element of the info dictionary in gym.step() function
from test_loader import SplitTrajectoryDataset
from torch.utils.data import DataLoader

"""
    Note that, we can pass arguments to the script by using
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9
    
    For learning the classical reach-avoid value function (baseline):
    python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True
    python run_training_ddpg.py --task ra_1d_Game-v0 --control-net 32 32 --disturbance-net 4 4 --critic-net 4 4 --epoch 10 --total-episodes 160 --gamma 0.9 --is-game-baseline True

"""
def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def get_args():
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

    yml = yaml.YAML(typ="safe", pure=True)
    configs = yml.load(
        #(pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
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

    final_config.logdir = f"{final_config.logdir+'/lcrl'}/{config.expt_name}"
    #final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task_lcrl}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir+'/lcrl'}", "cyan", attrs=["bold"])
    print("---------------------")
    return final_config



args=get_args()
config = args




image_size = config.size[0] #128
cam_obs_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
    )
policy_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
    )
bool_space = gym.spaces.Box(
        low=False, high=True, shape=(), dtype=bool
    )
observation_space = gym.spaces.Dict({
        'front_cam': cam_obs_space,
        'is_first': bool_space,
        'is_last': bool_space,
        'is_terminal': bool_space,
        'policy': policy_obs_space,
        'wrist_cam': cam_obs_space,
    })
action_space = gym.spaces.Box(low=-0.15, high=0.15, shape=(7,), dtype=np.float32)


config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]


wm = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=8,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=3,
        dropout=0.1
    )

wm.load_state_dict(torch.load('checkpoints/claude_zero_wfail4900.pth'))


hdf5_file = '/data/ken/ken_data/skittles_trajectories_labeled.h5'
bs = 1
bl= 10
device = 'cuda:0'
H = 3
expert_data = SplitTrajectoryDataset(hdf5_file, 3, split='train', num_test=100)

expert_loader = iter(DataLoader(expert_data, batch_size=1, shuffle=True))

env = gymnasium.make(args.task_lcrl, params = [wm, expert_loader])


# check if the environment has control and disturbance actions:
assert hasattr(env, 'action1_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
args.state_shape = env.observation_space.shape or env.observation_space.n

args.action_shape = env.action_space.shape or env.action_space.n

args.max_action = env.action_space.high[0]

args.action1_shape = env.action1_space.shape or env.action1_space.n
#args.action2_shape = env.action2_space.shape or env.action2_space.n
args.max_action1 = env.action1_space.high[0]
#args.max_action2 = env.action2_space.high[0]




from LCRL.data import Batch

#if args.wm:
from LCRL.policy import avoid_DDPGPolicy_annealing_dino as DDPGPolicy

print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

# seed
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
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

critic = Critic(critic_net, device=args.device).to(args.device)
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

# import pdb; pdb.set_trace()
log_path = None

#elif args.is_game_baseline:
#    from LCRL.policy import reach_avoid_game_DDPGPolicy_annealing as DDPGPolicy
#    print("DDPG under the Reach-Avoid annealed Bellman equation has been loaded!")
#else:
#    from LCRL.policy import reach_avoid_game_DDPGPolicy as DDPGPolicy
#    print("DDPG under the Reach-RL Bellman equation has been loaded!")
actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
actor1 = Actor(
    actor1_net, args.action1_shape, max_action=args.max_action1, device=args.device
).to(args.device)
actor1_optim = torch.optim.Adam(actor1.parameters(), lr=args.actor_lr)
#actor2_net = Net(args.state_shape, hidden_sizes=args.disturbance_net, activation=actor_activation, device=args.device)
#actor2 = Actor(
#    actor2_net, args.action2_shape, max_action=args.max_action2, device=args.device
#).to(args.device)
#actor2_optim = torch.optim.Adam(actor2.parameters(), lr=args.actor_lr)




policy = DDPGPolicy(
critic,
critic_optim,
tau=args.tau,
gamma=args.gamma_lcrl,
exploration_noise=GaussianNoise(sigma=args.exploration_noise),
reward_normalization=args.rew_norm,
estimation_step=args.n_step,
action_space=env.action_space,
actor1=actor1,
actor1_optim=actor1_optim,
actor_gradient_steps=args.actor_gradient_steps,
)

policy.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/logs/dreamer_dubins/lcrl/1227/151847/lcrl/franka_wm_DINO-v0/wm_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_1_buffer_size_40000_c_net_512_4_a1_512_4_a2_512_4_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0/epoch_id_100/policy.pth'))

def find_a(state):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy(tmp_batch, model = "actor_old").act
    act = policy.map_action(tmp).cpu().detach().numpy().flatten()
    return act

def evaluate_V(state):
    tmp_obs = np.array(state).reshape(1,-1)
    tmp_batch = Batch(obs = tmp_obs, info = Batch())
    tmp = policy.critic_old(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
    return tmp.cpu().detach().numpy().flatten()


def fill_eps_from_pkl_files(cache, cache_eval): 
    demo_path = "/home/kensuke/data/skittles/"
    # Get a list of all pickle files in the directory
    pkl_files = [os.path.join(demo_path, f) for f in os.listdir(demo_path) if f.endswith('.pkl')]
    
    pixel_keys = ["cam_rs", "cam_zed_crop"] # zed_right gets priority over zed_left
    embd_keys = ["cam_rs_embd", "cam_zed_right_embd"]
    for i, pkl_file in tqdm(
        enumerate(pkl_files),
        desc="Loading in expert data",
        ncols=0,
        leave=False,
        total=len(pkl_files),
    ):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)[0]
        
        for t, (obs, action, reward) in enumerate(data):
            transition = defaultdict(np.array)
            for obs_key in pixel_keys:
                if obs_key in obs:
                    if obs_key == "cam_zed_crop":
                        img = 255*obs[obs_key]
                        img = img.astype(np.uint8)
                    else:
                        img = obs[obs_key][0]
                    if obs_key == "cam_rs":
                        img_key = "robot0_eye_in_hand_image"
                    elif obs_key == "cam_zed_crop" or obs_key == "cam_zed_right":
                        img_key = "agentview_image"
                    # downsample img to 128x128

                    img_PIL = Image.fromarray(np.uint8(img)).convert('RGB')
                    img_obs = DINO_transform(img_PIL)
                    transition[img_key] = np.array(img_obs)
                    #img_obs = cv2.resize(img, (128, 128))
                    #transition[img_key] = np.array(img_obs, dtype=np.uint8)
            for obs_key in embd_keys:
                if obs_key in obs:
                    embd = obs[obs_key]
                    if obs_key == "cam_rs_embd":
                        emb_key = "robot0_eye_in_hand_embd"
                    elif obs_key == "cam_zed_left_embd" or obs_key == "cam_zed_right_embd":
                        emb_key = "agentview_embd"
                    transition[emb_key] = embd
            
            state = obs["state"]
            state_norm = normalize_eef_and_gripper(state)
            transition["state"] = state_norm
            transition["is_first"] = np.array(t == 0, dtype=np.bool_)
            transition["is_last"] = np.array(t == len(data) - 1, dtype=np.bool_)
            transition["is_terminal"] = np.array(t == len(data) - 1, dtype=np.bool_)
            transition["discount"] = np.array(1, dtype=np.float32)
            
            # Normalize action and insert into transition
            action = np.array(action, dtype=np.float32)
            action_norm = normalize_eef_and_gripper(action)
            transition["action"] = action_norm
            
            if i < 200:
                add_to_cache(cache, f"exp_traj_{i}", transition)
            else:
                add_to_cache(cache_eval, f"exp_traj_{i}", transition)


def make_dataset(episodes, bs, bl):
    generator = tools.sample_episodes(episodes, bl) #bl
    dataset = tools.from_generator(generator, bs) #bs
    return dataset


if __name__ == "__main__":
    #wandb.init(project="dino")

    hdf5_file = '/data/ken/ken_data/skittles_trajectories_unsafe_labeled.h5'
    bs = 1
    bl=12
    device = 'cuda:0'
    H = 3
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file, bl, split='test', num_test=28)

    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load('checkpoints/best_decoder_10m.pth'))
    decoder.eval()

    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=8,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=3,
        dropout=0.1
    ).to(device)

    transition.load_state_dict(torch.load('checkpoints/claude_zero_wfail4900.pth'))
    
    transition.eval()

    tp_ol, tn_ol, fp_ol, fn_ol = 0, 0, 0, 0
    tp_cl, tn_cl, fp_cl, fn_cl = 0, 0, 0, 0
    #data = next(expert_dataset)

    while True:
        print('loop')
        data = next(expert_loader_imagine)
        print(data["failure"][[0]])

        print((data["failure"][[0]]==0).all())

        while (data["failure"][[0]]==0).all() or (data["failure"][[0]]>0).all():
            data = next(expert_loader_imagine)

        inputs2 = data['cam_rs_embd'][[0], :H].to(device)
        inputs1 = data['cam_zed_right_embd'][[0], :H].to(device)
        all_acs = data['action'][[0]].to(device)
        all_fails = data['failure'][[0]].to(device)
        acs = data['action'][[0],:H].to(device)
        states = data['state'][[0],:H].to(device)
        im1s = (data['agentview_image'][[0], :H].squeeze().to(device)/255.).detach().cpu().numpy()
        im2s = (data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.).detach().cpu().numpy()
        
        pred_failures = []
        pred_brts = []
        for i in range(bl-H):
            latent = transition.forward_features(inputs1, inputs2, states, acs)

            pred1, pred2, pred_state, pred_fail = transition.front_head(latent), transition.wrist_head(latent), transition.state_pred(latent), transition.failure_pred(latent)
            
            latent = torch.mean(latent[:, [-1]], axis=2).detach().cpu().numpy()
            pred_brt = evaluate_V(latent)
            pred_im1 = decoder(pred1[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            pred_im2 = decoder(pred2[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            im1s = np.concatenate([im1s, pred_im1], axis=0)
            im2s = np.concatenate([im2s, pred_im2], axis=0)

            pred_failures.append(pred_fail[:,-1].item())
            pred_brts.append(pred_brt.item())
            
            
            # getting next inputs
            acs = torch.cat([acs[[0], 1:], all_acs[0,H+i].unsqueeze(0).unsqueeze(0)], dim=1)
            inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
            inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
            states = torch.cat([states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

        pred_failures = (torch.tensor(pred_failures) < 0).to(torch.int32)

        
        gt_im1 = (data['agentview_image'][[0], :bl].squeeze().to(device)/255.).detach().cpu().numpy()
        gt_im2 = (data['robot0_eye_in_hand_image'][[0], :bl].squeeze().to(device)/255.).detach().cpu().numpy()


        gt_imgs = np.concatenate([gt_im1, gt_im2], axis=-1)
        pred_imgs = np.concatenate([im1s, im2s], axis=-1)


        for i in range(len(pred_failures)):
            if pred_failures[i] == 1:
                pred_imgs[H+i, 0] *= 1.2
            if all_fails[0,H+i] == 1:
                gt_imgs[H+i, 0] *= 1.2
            if pred_failures[i] == 0 and all_fails[0,H+i] == 0:
                tn_ol+= 1
            if pred_failures[i] == 0 and all_fails[0,H+i] == 1:
                fp_ol+= 1
            if pred_failures[i] == 1 and all_fails[0,H+i] == 0:
                fn_ol+= 1
            if pred_failures[i] == 1 and all_fails[0,H+i] == 1:
                tp_ol+= 1
            
        print('open loop: tn, fp fn tp', tn_ol, fp_ol, fn_ol, tp_ol)
        print('doomed f', (torch.tensor(pred_brts)<0).to(torch.int32))
        print('failures', pred_failures)
        vid = np.concatenate([gt_imgs, pred_imgs], axis=-2)

        vid = (vid * 255).clip(0, 255).astype(np.uint8)
        frames = np.transpose(vid, (0, 2, 3, 1))
        fps = 20  # Frames per second
        iio.imwrite('output_video_dino_ol.gif', frames, duration=1/fps, loop=0)

        # Release the video writer
        print('saved!')
        inputs2 = data['cam_rs_embd'][[0], :H].to(device)
        inputs1 = data['cam_zed_right_embd'][[0], :H].to(device)
        all_acs = data['action'][[0]].to(device)
        all_states = data['state'][[0]].to(device)
        all_in2s = data['cam_rs_embd'][[0]].squeeze().to(device)
        all_in1s = data['cam_zed_right_embd'][[0]].squeeze().to(device)


        acs = data['action'][[0],:H].to(device)
        states = data['state'][[0],:H].to(device)
        im1s = (data['agentview_image'][[0], :H].squeeze().to(device)/255.).detach().cpu().numpy()
        im2s = (data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.).detach().cpu().numpy()
        pred_failures = []
        pred_brts = []
        for i in range(bl-H):
            latent = transition.forward_features(inputs1, inputs2, states, acs)

            pred1, pred2, pred_state, pred_fail = transition.front_head(latent), transition.wrist_head(latent), transition.state_pred(latent), transition.failure_pred(latent)
            
            latent = torch.mean(latent[:, [-1]], axis=2).detach().cpu().numpy()
            pred_brt = evaluate_V(latent)            
            pred_im1 = decoder(pred1[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            pred_im2 = decoder(pred2[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            im1s = np.concatenate([im1s, pred_im1], axis=0)
            im2s = np.concatenate([im2s, pred_im2], axis=0)
            pred_failures.append(pred_fail[:,-1].item())
            pred_brts.append(pred_brt.item())
            # getting next inputs
            acs = torch.cat([acs[[0], 1:], all_acs[0,H+i].unsqueeze(0).unsqueeze(0)], dim=1)
            inputs1 = torch.cat([inputs1[[0], 1:], all_in1s[H+i].unsqueeze(0).unsqueeze(0)], dim=1)
            inputs2 = torch.cat([inputs2[[0], 1:], all_in2s[H+i].unsqueeze(0).unsqueeze(0)], dim=1)
            states = torch.cat([states[[0], 1:], all_states[0, H+i].unsqueeze(0).unsqueeze(0)], dim=1)

            

        gt_im1 = (data['agentview_image'][[0], :bl].squeeze().to(device)/255.).detach().cpu().numpy()
        gt_im2 = (data['robot0_eye_in_hand_image'][[0], :bl].squeeze().to(device)/255.).detach().cpu().numpy()
            
        pred_failures = (torch.tensor(pred_failures) < 0).to(torch.int32)
              
        gt_imgs = np.concatenate([gt_im1, gt_im2], axis=-1)
        pred_imgs = np.concatenate([im1s, im2s], axis=-1)
        
        for i in range(len(pred_failures)):
            if pred_failures[i] == 1:
                pred_imgs[H+i, 0] *= 1.2
            if all_fails[0,H+i] == 1:
                gt_imgs[H+i, 0] *= 1.2
            if pred_failures[i] == 0 and all_fails[0,H+i] == 0:
                tn_cl+= 1
            if pred_failures[i] == 0 and all_fails[0,H+i] == 1:
                fp_cl+= 1
            if pred_failures[i] == 1 and all_fails[0,H+i] == 0:
                fn_cl+= 1
            if pred_failures[i] == 1 and all_fails[0,H+i] == 1:
                tp_cl+= 1    

        print('open loop: tn, fp fn tp', tn_ol, fp_ol, fn_ol, tp_ol)
        print('doomed f', (torch.tensor(pred_brts)))

        print('doomed f', (torch.tensor(pred_brts)<0.5).to(torch.int32))
        print('failures', pred_failures)
        print('closed loop: tn, fp fn tp', tn_cl, fp_cl, fn_cl, tp_cl)  
        vid = np.concatenate([gt_imgs, pred_imgs], axis=-2)

        vid = (vid * 255).clip(0, 255).astype(np.uint8)
        frames = np.transpose(vid, (0, 2, 3, 1))
        fps = 20  # Frames per second
        print(im1s.shape)
        iio.imwrite('output_video_dino_cl.gif', frames, duration=1/fps, loop=0)
        print('end loop')
        exit()

    
    




