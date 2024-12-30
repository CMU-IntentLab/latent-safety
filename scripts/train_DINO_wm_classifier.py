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

from train_DINO_decoder import Decoder
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



DINO_transform = transforms.Compose([           
                            transforms.Resize(224),
                            #transforms.CenterCrop(224), #should be multiple of model patch_size                 
                            
                            transforms.ToTensor(),])
norm_transform = transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )

'''
def fill_eps_from_pkl_files(cache, cache_eval): 
    demo_path = "/home/kensuke/data/skittles_big/"
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

                    img_PIL = Image.fromarray(np.uint8(img)).convert('RGB')
                    img_obs = DINO_transform(img_PIL)
                    transition[img_key] = np.array(img_obs)
                    transition[img_key+'_norm'] = np.array(norm_transform(img_obs))
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
            
            if i < 1200:
                add_to_cache(cache, f"exp_traj_{i}", transition)
            else:
                add_to_cache(cache_eval, f"exp_traj_{i}", transition)


def make_dataset(episodes, BS, BL):
    generator = tools.sample_episodes(episodes, BL) #bl
    dataset = tools.from_generator(generator, BS) #bs
    return dataset
'''

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Tuple, Optional
from test_loader import SplitTrajectoryDataset
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0., 
                 num_frames: int = 2, patches_per_frame: int = 256):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register buffer instead of creating mask every forward pass
        mask = self._create_causal_mask(num_frames, patches_per_frame)
        self.register_buffer("mask", mask)
        
    def _create_causal_mask(self, num_frames: int, patches_per_frame: int) -> torch.Tensor:
        total_patches = num_frames * patches_per_frame
        mask = torch.zeros(total_patches, total_patches)
        
        for i in range(num_frames):
            start_idx = i * patches_per_frame
            end_idx = (i + 1) * patches_per_frame
            
            # Allow attention within current frame
            mask[start_idx:end_idx, start_idx:end_idx] = 1
            
            # Allow attention to previous frames
            if i > 0:
                mask[start_idx:end_idx, :start_idx] = 1
                
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Use registered mask buffer
        mask = self.mask[:seq_len, :seq_len]
        dots = dots.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0., 
                 num_frames: int = 2):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dim_head, dropout, num_frames)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class VideoTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        dim: int,
        ac_dim: int,
        state_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        num_frames: int = 2,
        dim_head: int = 64,
        dropout: float = 0.,
        emb_dropout: float = 0.,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
        
        # Improved action embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, ac_dim),
            nn.LayerNorm(ac_dim)
        ).to(device)
        
        total_dim = 2*dim + ac_dim + state_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, total_dim) * 0.02)
        self.temp_embedding = nn.Parameter(torch.randn(1, num_frames, total_dim) * 0.02)
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use TransformerBlock instead of separate components
        self.transformer = nn.ModuleList([
            TransformerBlock(total_dim, heads, dim_head, mlp_dim, dropout, num_frames)
            for _ in range(depth)
        ])
        
        # Separate prediction heads
        self.wrist_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim)
        )
                
        
        #nn.Sequential(
        #    LayerNorm(total_dim),
        #    nn.Linear(total_dim, dim)  # Predicts both video streams
        #)
        self.front_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, dim)
        )
        #nn.Sequential(
        #    LayerNorm(total_dim),
        #    nn.Linear(total_dim, dim)  # Predicts both video streams
        #)
        
        self.state_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, state_dim)
        )

        self.failure_head = nn.Sequential(
            LayerNorm(total_dim),
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, 1)
        )
        #nn.Sequential(
        #    LayerNorm(total_dim),
        #    nn.Linear(total_dim, state_dim)
        #)
        
    def forward(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        


        # Encode actions
        action_embeddings = self.action_encoder(actions).unsqueeze(2).expand(-1, -1, 256, -1)
        state_embeddings = states.unsqueeze(2).expand(-1, -1, 256, -1)
        
        # Combine features
        batch_size, num_frames, _, _ = video1.shape
    
        x = torch.cat((video1, video2, action_embeddings, state_embeddings), dim=3)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        x = x + self.temp_embedding.unsqueeze(2)
        
        # Reshape for transformer
        x = rearrange(x, 'b s n d -> b (s n) d')
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
            
        # Reshape back
        x = rearrange(x, 'b (s n) d -> b s n d', s=num_frames)
        
        # Generate predictions
        pred1 = self.front_head(x)
        pred2 = self.wrist_head(x)
        
        state_preds = self.state_head(x)
        state_preds = torch.mean(state_preds, dim=2)  # Average over patches
        
        failure_preds = self.failure_head(x)
        failure_preds = torch.mean(failure_preds, dim=2)  # Average over patches

        
        return pred1, pred2, state_preds, failure_preds

    @torch.no_grad()
    def get_dino_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract DINO features from video frames."""
        b, f, c, h, w = video.shape
        video = video.view(b * f, c, h, w)
        features = self.dino.forward_features(video)['x_norm_patchtokens']
        return features.view(b, f, -1, features.shape[-1])
    

def fail_loss(pred, fail_data):
    
    safe_data = torch.where(fail_data == 0.)
    unsafe_data = torch.where(fail_data == 1.)
    unsafe_data_weak = torch.where(fail_data == 2.)
    
    
    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]
    
    

    gamma = 0.75
    lx_loss = (1/pos.size(0))*torch.sum(torch.relu(gamma - pos)) if pos.size(0) > 0 else 0. #penalizes safe for being negative
    lx_loss +=  (1/neg.size(0))*torch.sum(torch.relu(gamma + neg)) if neg.size(0) > 0 else 0. # penalizes unsafe for being positive
    #lx_loss +=  (1/neg_weak.size(0))*torch.sum(torch.relu(-gamma + neg_weak)) if neg_weak.size(0) > 0 else 0. # penalizes unsafe for being positive
    lx_loss +=  (1/neg_weak.size(0))*torch.sum(torch.relu(neg_weak)) if neg_weak.size(0) > 0 else 0. # penalizes unsafe for being positive

    return lx_loss

if __name__ == "__main__":
    wandb.init(project="dino")


    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    #expert_eps = collections.OrderedDict()
    #expert_eps_eval = collections.OrderedDict()
    #fill_eps_from_pkl_files(expert_eps, expert_eps_eval)

    BS = 16
    BL= 4
    hdf5_file = '/data/ken/ken_data/skittles_trajectories_labeled.h5'

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split='train', num_test=100)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, BL, split='test', num_test=100)
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file, 32, split='test', num_test=100)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    #expert_dataset = make_dataset(expert_eps, BS, BL)
    #expert_dataset_eval = make_dataset(expert_eps_eval, BS, BL)
    #expert_dataset_imagine = make_dataset(expert_eps_eval, 1, 32)
    device = 'cuda:1'
    H = 3
   
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
        num_frames=BL-1,
        dropout=0.1
    ).to(device)
    transition.load_state_dict(torch.load('best_claude_nofail.pth'))

    #data = next(expert_dataset)
    data = next(expert_loader)
    

         

    #data1 = torch.tensor(data['agentview_embd']).to(device)
    data1 = data['cam_zed_right_embd'].to(device)#[transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
    data2 =  data['cam_rs_embd'].to(device)#transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

    inputs1 = data1[:, :-1]
    output1 = data1[:, 1:]

    #data2 = torch.tensor(data['robot0_eye_in_hand_embd']).to(device)
    #data2 = transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

    inputs2 = data2[:, :-1]
    output2 = data2[:, 1:]

    data_state = data['state'].to(device)
    states = data_state[:, :-1]
    output_state = data_state[:, 1:]

    #data_acs = torch.tensor(data['action']).to(device)
    data_acs = data['action'].to(device)
    acs = data_acs[:, :-1]



    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)

    # Forward pass

    optimizer = AdamW([
        {'params': transition.failure_head.parameters(), 'lr': 5e-5}, 
    ])

    best_eval = float('inf')
    best_fail= float('inf')
    iters = []
    train_iter = 100000
    for i in tqdm(range(train_iter), desc="Training", unit="iter"):
        if i % 30 == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        #data = next(expert_dataset)
        data = next(expert_loader)



        data1 = data['cam_zed_right_embd'].to(device)#[transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
        data2 =  data['cam_rs_embd'].to(device)#transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        inputs2 = data2[:, :-1]
        output2 = data2[:, 1:]

        data_state = data['state'].to(device)
        states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data['action'].to(device)
        acs = data_acs[:, :-1]

        
        optimizer.zero_grad()


        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
            #im1_loss = nn.MSELoss()(pred1, output1)
            #im2_loss = nn.MSELoss()(pred2, output2)
            #state_loss = nn.MSELoss()(pred_state, output_state)
            failure_loss = fail_loss(pred_fail, data['failure'][:, 1:])
            loss = failure_loss #im1_loss + im2_loss + state_loss #+ failure_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        wandb.log({'train_loss': train_loss})
        print(f"\rIter {i}, Train Loss: {train_loss:.4f}, failure Loss: {failure_loss.item():.4f}", end='', flush=True)
        
        if (i) % 100 == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                eval_data1 = eval_data['cam_zed_right_embd'].to(device)#[transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
                eval_data2 =  eval_data['cam_rs_embd'].to(device)#transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))


                #inputs1 = torch.tensor(eval_data['agentview_embd'][[0], :H]).to(device)
                #inputs2 = torch.tensor(eval_data['robot0_eye_in_hand_embd'][[0], :H]).to(device)
                inputs1 = eval_data1[[0], :H].to(device)
                inputs2 = eval_data2[[0], :H].to(device)
                all_acs = eval_data['action'][[0]].to(device)
                acs = eval_data['action'][[0],:H].to(device)
                states = eval_data['state'][[0],:H].to(device)
                im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.
                im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                for k in range(16-H):
                    pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
                    pred_im1 = decoder(pred1[:,-1])[0]#.detach().cpu().numpy()
                    pred_im2 = decoder(pred2[:,-1])[0]#.detach().cpu().numpy()
                    pred_fail = pred_fail[:,-1]

                    if pred_fail < 0:
                        pred_im1[0,:,:] *= 2
                        pred_im2[0,:,:] *= 2
                    
                    im1s = torch.cat([im1s, pred_im1.unsqueeze(0)], dim=0)
                    im2s = torch.cat([im2s, pred_im2.unsqueeze(0)], dim=0)
                    
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                    inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                    states = torch.cat([states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

                    
                gt_im1 = eval_data['agentview_image'][[0], :16].squeeze().to(device)
                gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :16].squeeze().to(device)
                gt_fail = eval_data['failure'][[0], :16].squeeze().to(device)

                
                for j in range(16):
                    if gt_fail[j] > 0:
                        gt_im1[j,0,:,:] *= 2
                        gt_im2[j,0,:,:] *= 2
               

                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-1)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-1)


                vid = torch.cat([gt_imgs, pred_imgs], dim=-2)
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)
                wandb.log({"video": wandb.Video(vid, fps=20)})
                
                # done logging video

    
                eval_data = next(expert_loader_eval)


            
                data1 = eval_data['cam_zed_right_embd'].to(device)#[transition.get_dino_features(torch.tensor(data['agentview_image_norm']).to(device))
                data2 =  eval_data['cam_rs_embd'].to(device)#transition.get_dino_features(torch.tensor(data['robot0_eye_in_hand_image_norm']).to(device))

                #data1 = torch.tensor(eval_data['agentview_embd']).to(device)
                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                #data2 = torch.tensor(eval_data['robot0_eye_in_hand_embd']).to(device)
                inputs2 = data2[:, :-1]
                output2 = data2[:, 1:]

                data_state = eval_data['state'].to(device)
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = eval_data['action'].to(device)
                acs = data_acs[:, :-1]
                pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
                
                pred_im1 = decoder(pred1[:,H-1])[0].permute(1, 2, 0).detach().cpu().numpy()
                pred_im2 = decoder(pred2[:,H-1])[0].permute(1, 2, 0).detach().cpu().numpy()
                im1 = eval_data['agentview_image'][0, H].permute(1, 2, 0).numpy()
                im2 = eval_data['robot0_eye_in_hand_image'][0, H].permute(1, 2, 0).numpy()
                im1_loss = nn.MSELoss()(pred1, output1)
                im2_loss = nn.MSELoss()(pred2, output2)
                state_loss = nn.MSELoss()(pred_state, output_state)
                failure_loss = fail_loss(pred_fail, eval_data['failure'][:, 1:])
                loss = failure_loss
            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}, front Loss: {im1_loss.item():.4f}, wrist Loss: {im2_loss.item():.4f}, state Loss: {state_loss.item():.4f}, failure Loss: {failure_loss.item():.4f}")

            torch.save(transition.state_dict(), f'claude_zero_wfail{i}.pth')

            if loss < best_eval:
                best_eval = loss
                torch.save(transition.state_dict(), 'best_claude_wfail.pth')
            #if failure_loss < best_fail:
            #    best_fail = failure_loss
            #    torch.save(transition.state_dict(), 'best_claude_nofail_failure.pth')
            
            transition.train()
            wandb.log({'eval_loss': loss.item(), 'front_loss': im1_loss.item(), 'wrist_loss': im2_loss.item(), 'state_loss': state_loss.item(), 'pred_front': wandb.Image(pred_im1), 'pred_wrist': wandb.Image(pred_im2), 'front': wandb.Image(im1), 'wrist': wandb.Image(im2), 'failure_loss':failure_loss.item()})


    plt.legend()
    plt.savefig('training curve.png')    
      


