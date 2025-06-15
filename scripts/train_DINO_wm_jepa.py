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
from torch.utils.data import Dataset, DataLoader

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

import requests
from PIL import Image
from torchvision import transforms

import torch
from torch import nn
from torch.optim import AdamW

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple, Optional
from test_loader import SplitTrajectoryDataset
from dino_decoders_official import VQVAE
from dino_models import Decoder, VideoTransformer, normalize_acs, batch_quat_to_rotvec, batch_rotvec_to_quat


transform = transforms.Compose([           
                                transforms.Resize(256),                    
                                transforms.CenterCrop(224),               
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])


DINO_transform = transforms.Compose([           
                            transforms.Resize(224),
                            
                            transforms.ToTensor(),])
norm_transform = transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )

if __name__ == "__main__":
    wandb.init(project="dino-WM",
               name="WM")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


    BS = 16
    BL= 4
    EVAL_H = 16
    H = 3

    hdf5_file = '/data/ken/latent/consolidated.h5'
    hdf5_file_test = '/data/ken/latent-unsafe-test/consolidated.h5'

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split='train', num_test=0)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file_test, BL, split='test', num_test=467)
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file_test, 32, split='test', num_test=467)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    device = 'cuda:0'
   
    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load('checkpoints/testing_decoder.pth'))
    decoder.eval()

    
    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=8,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL-1,
        dropout=0.1
    ).to(device)
    transition.train()
    # Forward pass
    optimizer = AdamW([
        {'params': transition.transformer.parameters(), 'lr': 5e-5},
        {'params': transition.state_head.parameters(), 'lr': 5e-5}, 
        {'params': transition.front_head.parameters(), 'lr': 5e-5}, 
        {'params': transition.wrist_head.parameters(), 'lr': 5e-5}, 
        {'params': transition.action_encoder.parameters(), 'lr': 5e-4},
        {'params': [transition.pos_embedding], 'lr': 5e-4},
        {'params': [transition.temp_embedding], 'lr': 5e-4}
    ])

    best_eval = float('inf')
    iters = []
    train_iter = 100000

    for i in tqdm(range(train_iter), desc="Training", unit="iter"):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        if i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        data = next(expert_loader)


        data1 = data['cam_zed_embd'].to(device)
        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        data2 =  data['cam_rs_embd'].to(device)
        inputs2 = data2[:, :-1]
        output2 = data2[:, 1:]

        data_state = data['state'].to(device)
        inputs_states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data['action'].to(device)
        norm_acs = normalize_acs(data_acs, device)
        acs = norm_acs[:, :-1]

        
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, _ = transition(inputs1, inputs2, inputs_states, acs)
            im1_loss_tf = nn.MSELoss()(pred1, output1)
            im2_loss_tf = nn.MSELoss()(pred2, output2)
            state_loss_tf = nn.MSELoss()(pred_state, output_state)
            loss_tf = im1_loss_tf + im2_loss_tf + state_loss_tf

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            detach_pred1 = pred1
            detach_pred2 = pred2
            detach_pred_state = pred_state.detach()
            inputs1_ar = torch.cat([data1[:, [0]], detach_pred1[:, [0]]], dim=1)
            inputs2_ar = torch.cat([data2[:, [0]], detach_pred2[:, [0]]], dim=1)
            states_ar = torch.cat([data_state[:,[0]], detach_pred_state[:, [0]]], dim=1)
            acs_ar = norm_acs[:, [0,1]]

            pred1_ar, pred2_ar, pred_state_ar, _ = transition(inputs1_ar, inputs2_ar, states_ar, acs_ar)
            output1_ar = data1[:, 2]
            output2_ar = data2[:, 2]
            output_state_ar = data_state[:, 2]
            im1_loss_ar = nn.MSELoss()(pred1_ar[:,1], output1_ar)
            im2_loss_ar = nn.MSELoss()(pred2_ar[:,1], output2_ar)
            state_loss_ar = nn.MSELoss()(pred_state_ar[:,1], output_state_ar)
            loss_ar = im1_loss_ar + im2_loss_ar + state_loss_ar       

        loss = loss_tf + loss_ar*0.5
        #loss = loss_tf

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        print(f"\rIter {i}, TF Loss: {loss_tf:.4f}, AR loss:{loss_ar} front Loss: {im1_loss_tf.item():.4f}, wrist Loss: {im2_loss_tf.item():.4f}, state Loss: {state_loss_tf.item():.4f}", end='', flush=True)
        print(f"\rIter {i}, TF Loss: {loss_tf:.4f}, front Loss: {im1_loss_tf.item():.4f}, wrist Loss: {im2_loss_tf.item():.4f}, state Loss: {state_loss_tf.item():.4f}", end='', flush=True)
        wandb.log({'train_loss': loss_tf, "train_loss_ar": loss_ar})
        # eval
        if (i) % 1000 == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                eval_data1 = eval_data['cam_zed_embd'].to(device)
                inputs1 = eval_data1[[0], :H].to(device)

                eval_data2 =  eval_data['cam_rs_embd'].to(device)
                inputs2 = eval_data2[[0], :H].to(device)
                
                all_acs = eval_data['action'][[0]].to(device)
                all_acs = normalize_acs(all_acs, device)
                
                acs = eval_data['action'][[0],:H].to(device)
                acs = normalize_acs(acs, device)

                inputs_states = eval_data['state'][[0],:H].to(device)
                im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.
                im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                for k in range(EVAL_H-H):
                    pred1, pred2, pred_state, _ = transition(inputs1, inputs2, inputs_states, acs)

                    pred_latent = torch.cat([pred1[:,[-1]], pred2[:,[-1]]], dim=0)#.squeeze()
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
                    pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)

                    
                    im1s = torch.cat([im1s, pred_im1.squeeze(0)], dim=0)
                    im2s = torch.cat([im2s, pred_im2.squeeze(0)], dim=0)
                    
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                    inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                    states = torch.cat([inputs_states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

                    
                gt_im1 = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)
                gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)

                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-2)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-2)
                vid = torch.cat([gt_imgs, pred_imgs], dim=-3)
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)
                vid = rearrange(vid, "t h w c -> t c h w")
                wandb.log({"video": wandb.Video(vid, fps=20, format='mp4')})
                
                # done logging video

    
                eval_data = next(expert_loader_eval)
                data1 = eval_data['cam_zed_embd'].to(device)
                data2 =  eval_data['cam_rs_embd'].to(device)

                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                inputs2 = data2[:, :-1]
                output2 = data2[:, 1:]

                data_state = eval_data['state'].to(device)
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = eval_data['action'].to(device)
                data_acs = normalize_acs(data_acs, device)
                acs = data_acs[:, :-1]
                pred1, pred2, pred_state, _ = transition(inputs1, inputs2, states, acs)


                pred_latent = torch.cat([pred1[:,[H-1]], pred2[:,[H-1]]], dim=0)
                pred_ims, _ = decoder(pred_latent)
                pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)
                pred_im1 = pred_im1[0].permute(1,2,0).detach().cpu().numpy()
                pred_im2 = pred_im2[0].permute(1,2,0).detach().cpu().numpy()
                im1 = eval_data['agentview_image'][0, H].numpy()
                im2 = eval_data['robot0_eye_in_hand_image'][0, H].numpy()
                im1_loss = nn.MSELoss()(pred1, output1)
                im2_loss = nn.MSELoss()(pred2, output2)
                state_loss = nn.MSELoss()(pred_state, output_state)
                loss = im1_loss + im2_loss + state_loss
            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}, front Loss: {im1_loss.item():.4f}, wrist Loss: {im2_loss.item():.4f}, state Loss: {state_loss.item():.4f}")

            torch.save(transition.state_dict(), f'checkpoints/testing_iter{i}.pth')

            if loss < best_eval:
                best_eval = loss
                torch.save(transition.state_dict(), 'checkpoints/best_testing.pth')
            
            transition.train()
            wandb.log({'eval_loss': loss.item(), 'front_loss': im1_loss.item(), 'wrist_loss': im2_loss.item(), 'state_loss': state_loss.item(), 'pred_front': wandb.Image(pred_im1), 'pred_wrist': wandb.Image(pred_im2), 'front': wandb.Image(im1), 'wrist': wandb.Image(im2)})


    plt.legend()
    plt.savefig('training curve.png')    
      


