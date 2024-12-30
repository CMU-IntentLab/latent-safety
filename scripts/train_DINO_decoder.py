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
'''
class DINOv2Decoder(nn.Module):
    def __init__(self, token_dim=384, num_tokens=256, output_size=(224, 224), num_channels=3):
        super(DINOv2Decoder, self).__init__()
        
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.output_size = output_size
        
        # Grid size assuming a perfect square (16x16)
        self.grid_size = int(num_tokens ** 0.5)  # num_tokens = 256, so grid_size = 16
        
        # Patch size calculation based on the final image size
        self.patch_size = 224 // self.grid_size  # Patch size: 224 / 16 = 14
        
        self.num_channels = num_channels  # Number of channels in the output image

        # Linear layer to project tokens back to patch dimensions
        self.token_to_patch = nn.Linear(token_dim, self.patch_size * self.patch_size)

        # Decoder: Using ConvTranspose2d to upscale from 128x128 to 224x224
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Start with 224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Keep 224x224
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Final RGB output, keep 224x224
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, tokens):
        """
        tokens: Tensor of shape (batch_size, num_tokens, token_dim)
        Returns: Tensor of shape (batch_size, num_channels, output_size[0], output_size[1])
        """
        batch_size = tokens.size(0)

        # Step 1: Project tokens to patches
        patches = self.token_to_patch(tokens)  # (batch_size, num_tokens, patch_size * patch_size)
        patches = patches.view(batch_size, self.num_tokens, self.patch_size, self.patch_size)  # (batch_size, num_tokens, 14, 14)

        # Step 2: Rearrange patches into a grid
        grid_size = self.grid_size
        image = patches.view(batch_size, grid_size, grid_size, self.patch_size, self.patch_size)  # (batch_size, grid_h, grid_w, patch_h, patch_w)
        image = image.permute(0, 1, 3, 2, 4).contiguous()  # Rearrange to (batch_size, grid_h * patch_h, grid_w * patch_w)
        image = image.view(batch_size, 1, 224, 224)  # (batch_size, 1, 224, 224)

        # Step 3: Decode the image using ConvTranspose2d
        reconstructed = self.decoder(image)  # (batch_size, num_channels, 224, 224)
        return reconstructed'''
    

class DINOv2Decoder(nn.Module):
    def __init__(self, token_dim=384, num_tokens=256, output_size=(224, 224), num_channels=3):
        super().__init__()
        
        self.grid_size = int(num_tokens ** 0.5)
        self.patch_size = output_size[0] // self.grid_size
        
        # More sophisticated token projection with depth
        self.token_projector = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(token_dim * 2, self.patch_size * self.patch_size * num_channels)
        )

        # Deep reconstruction network
        self.decoder = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Residual blocks for deeper learning
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            
            # Refinement layers
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final reconstruction
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        batch_size = tokens.size(0)
        
        # Project tokens with more complexity
        patches = self.token_projector(tokens)
        
        # Reshape to image grid
        patches = patches.view(
            batch_size, 
            self.grid_size, 
            self.grid_size, 
            self.patch_size, 
            self.patch_size, 
            -1
        )
        
        # Rearrange patches into full image
        image = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        image = image.view(
            batch_size, 
            -1, 
            self.grid_size * self.patch_size, 
            self.grid_size * self.patch_size
        )
        
        # Deep reconstruction
        reconstructed = self.decoder(image)
        
        return reconstructed

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Convolution path
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += self.shortcut(residual)
        return F.relu(out)



DINO_transform = transforms.Compose([           
                                transforms.Resize(224),                                
                                transforms.ToTensor(),])


class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=3):
        super(Decoder, self).__init__()
        
        # Two residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels),            
            ResidualBlock2(in_channels),
            ResidualBlock2(in_channels)


        )
        
        # Three transposed convolutions to go from 16x16 to 224x224
        self.transposed_convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # New intermediate layer
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # New additional layer with no change in resolution
            nn.ConvTranspose2d(in_channels // 8, in_channels // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels // 8, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.resize_transform = transforms.Resize((224, 224))

        
    
    def forward(self, x):

        x = x.view(-1, 16, 16, 384)  # Reshape to (16, 16, 384) where 16x16 is the spatial grid
        x = x.permute(0, 3, 1, 2)        # Pass through residual blocks
        x = self.residual_blocks(x)
        # Pass through transposed convolutions
        x = self.transposed_convs(x)
        x = self.resize_transform(x)
        
        return x



'''def fill_eps_from_pkl_files(cache, cache_eval): 
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
        if i > 500:
            break
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


def make_dataset(episodes):
    generator = tools.sample_episodes(episodes, 1) #bl
    dataset = tools.from_generator(generator, 64) #bs
    return dataset'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    wandb.init(project="dino-decoder")


    hdf5_file = '/data/ken/ken_data/skittles_trajectories.h5'

    expert_data = SplitTrajectoryDataset(hdf5_file, 1, split='train', num_test=100)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, 1, split='test', num_test=100)

    expert_loader = iter(DataLoader(expert_data, batch_size=64, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=64, shuffle=True))
    device = 'cuda:1'
    H = 3

    #decoder = Decoder().to('cuda:0')
    #decoder = DINOv2Decoder().to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load('best_decoder_10m.pth'))
    
    optimizer = AdamW([
        {'params': decoder.parameters(), 'lr': 3e-4}
    ])

    best_eval = float('inf')
    iters = []
    train_losses = []
    eval_losses = []
    train_iter = 10000
    for i in range(train_iter):

        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=64, shuffle=True))
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=64, shuffle=True))
        data = next(expert_loader)

        #print(data['agentview_image'].shape)
        #print(data['agentview_embd'].shape)
        inputs1 = data['cam_zed_right_embd'].to(device)
        inputs2 = data['cam_rs_embd'].to(device)
        output1 = data['agentview_image'].squeeze().to(device)/255.
        output2 = data['robot0_eye_in_hand_image'].squeeze().to(device)/255.


        inputs = torch.cat([inputs1, inputs2], dim=0).squeeze()
        pred = decoder(inputs)

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
                pred = decoder(inputs)
                pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
                

                loss = nn.MSELoss()(pred1.squeeze(), output1.squeeze())
                loss += nn.MSELoss()(pred2.squeeze(), output2.squeeze())

            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}")
            if loss < best_eval:
                best_eval = loss
                torch.save(decoder.state_dict(), 'best_decoder_10m.pth')
            decoder.train()
            
            out_log = (output1[0,0].detach().permute(1, 2, 0).detach().cpu().numpy())
            pred_log = (pred1[0].detach().permute(1, 2, 0).detach().cpu().numpy())
            out_log2 = (output2[0,0].detach().permute(1, 2, 0).detach().cpu().numpy())
            pred_log2 = (pred2[0].detach().permute(1, 2, 0).detach().cpu().numpy())

            wandb.log({'eval_loss': loss.item(), 'ground_truth_front': wandb.Image(out_log), 'pred_front': wandb.Image(pred_log), 'ground_truth_wrist': wandb.Image(out_log2), 'pred_wrist': wandb.Image(pred_log2)})
            eval_losses.append(loss.item())


    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, eval_losses, label='eval')
    plt.legend()
    plt.savefig('training curve.png')    
      


