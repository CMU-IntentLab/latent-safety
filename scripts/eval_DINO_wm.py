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
    bl=20
    device = 'cuda:0'
    H = 3
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file, 32, split='test', num_test=28)

    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load('best_decoder_10m.pth'))
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

    transition.load_state_dict(torch.load('claude_zero_wfail4900.pth'))
    
    transition.eval()

    tp_ol, tn_ol, fp_ol, fn_ol = 0, 0, 0, 0
    tp_cl, tn_cl, fp_cl, fn_cl = 0, 0, 0, 0
    #data = next(expert_dataset)

    while True:
        print('loop')
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
        for i in range(bl-H):
            pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
            pred_im1 = decoder(pred1[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            pred_im2 = decoder(pred2[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            im1s = np.concatenate([im1s, pred_im1], axis=0)
            im2s = np.concatenate([im2s, pred_im2], axis=0)

            pred_failures.append(pred_fail[:,-1].item())
            
            
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
        vid = np.concatenate([gt_imgs, pred_imgs], axis=-2)

        vid = (vid * 255).clip(0, 255).astype(np.uint8)
        frames = np.transpose(vid, (0, 2, 3, 1))
        fps = 20  # Frames per second
        #iio.imwrite('output_video_dino_ol.gif', frames, duration=1/fps, loop=0)


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
        for i in range(bl-H):
            pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
            pred_im1 = decoder(pred1[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            pred_im2 = decoder(pred2[:,-1])[0].unsqueeze(0).detach().cpu().numpy()
            im1s = np.concatenate([im1s, pred_im1], axis=0)
            im2s = np.concatenate([im2s, pred_im2], axis=0)
            pred_failures.append(pred_fail[:,-1].item())
            
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
        print('closed loop: tn, fp fn tp', tn_cl, fp_cl, fn_cl, tp_cl)  
        vid = np.concatenate([gt_imgs, pred_imgs], axis=-2)

        vid = (vid * 255).clip(0, 255).astype(np.uint8)
        frames = np.transpose(vid, (0, 2, 3, 1))
        fps = 20  # Frames per second
        print(im1s.shape)
        #iio.imwrite('output_video_dino_cl.gif', frames, duration=1/fps, loop=0)
        print('end loop')
        #exit()

    
    




