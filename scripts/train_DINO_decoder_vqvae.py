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

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

import requests
from PIL import Image
from torchvision import transforms

import torch
from torch import nn
from torch.optim import AdamW

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch.utils.data import DataLoader
from test_loader import SplitTrajectoryDataset
import torch
from torch import nn
from torch.functional import F

from dino_decoders_official import VQVAE

DINO_transform = transforms.Compose([           
                                transforms.Resize(224),                                
                                transforms.ToTensor(),])

from einops import rearrange, repeat
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    wandb.init(project="dino-WM",
               name="Decoder")


    hdf5_file = '/data/ken/ken_data/skittles_trajectories.h5'

    expert_data = SplitTrajectoryDataset(hdf5_file, 1, split='train', num_test=100)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, 1, split='test', num_test=100)

    expert_loader = iter(DataLoader(expert_data, batch_size=64, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=64, shuffle=True))
    device = 'cuda:1'
    H = 3

    decoder = VQVAE().to(device)
    print('decoder with parameters', count_parameters(decoder))
    
    optimizer = AdamW([
        {'params': decoder.parameters(), 'lr': 3e-4}
    ])

    best_eval = float('inf')
    iters = []
    train_losses = []
    eval_losses = []
    train_iter = 1000
    for i in range(train_iter):

        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=64, shuffle=True))
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=64, shuffle=True))
        data = next(expert_loader)

        
        inputs1 = data['cam_zed_right_embd'].to(device)
        inputs2 = data['cam_rs_embd'].to(device)
        output1 = data['agentview_image'].squeeze().to(device)/255.
        output2 = data['robot0_eye_in_hand_image'].squeeze().to(device)/255.


        inputs = torch.cat([inputs1, inputs2], dim=0)#.squeeze()

        pred, _ = decoder(inputs)
        pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
        
        pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)

        loss = nn.MSELoss()(pred1.squeeze(), output1.squeeze())
        loss += nn.MSELoss()(pred2.squeeze(), output2.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss': loss.item()})
        print(f"\rIter {i}, Train Loss: {loss.item():.4f}", end='', flush=True)
        
        if i % 100 == 0:
            train_losses.append(loss.item())
            iters.append(i)
            eval_data = next(expert_loader_eval)
            decoder.eval()
            with torch.no_grad():
                inputs1 = eval_data['cam_zed_right_embd'].to(device)
                inputs2 = eval_data['cam_rs_embd'].to(device)
                output1 = eval_data['agentview_image'].to(device)/255.
                output2 = eval_data['robot0_eye_in_hand_image'].to(device)/255.


                inputs = torch.cat([inputs1, inputs2], dim=0)
                pred, _ = decoder(inputs)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
                pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
                
                
                loss = nn.MSELoss()(pred1.squeeze(), output1.squeeze())
                loss += nn.MSELoss()(pred2.squeeze(), output2.squeeze())

            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}")
            if loss < best_eval:
                best_eval = loss
                torch.save(decoder.state_dict(), 'checkpoints/testing_decoder.pth')
            decoder.train()
            

            out_log = (output1[0,0].detach().permute(1, 2, 0).detach().cpu().numpy())
            pred_log = (pred1[0,0].detach().permute(1, 2, 0).detach().cpu().numpy())
            out_log2 = (output2[0,0].detach().permute(1, 2, 0).detach().cpu().numpy())
            pred_log2 = (pred2[0,0].detach().permute(1, 2, 0).detach().cpu().numpy())

            wandb.log({'eval_loss': loss.item(), 'ground_truth_front': wandb.Image(out_log), 'pred_front': wandb.Image(pred_log), 'ground_truth_wrist': wandb.Image(out_log2), 'pred_wrist': wandb.Image(pred_log2)})
            eval_losses.append(loss.item())


    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, eval_losses, label='eval')
    plt.legend()
    plt.savefig('training curve.png')    
      


