#import argparse
#import collections
import os
#import pathlib
#import sys
import numpy as np
#import ruamel.yaml as yaml
import torch
#import cv2
# add to os sys path
#import sys
import matplotlib.pyplot as plt


#import model_based_irl_torch.dreamer.tools as tools
#from model_based_irl_torch.dreamer.dreamer import Dreamer
#from termcolor import cprint
#from real_envs.env_utils import normalize_eef_and_gripper, unnormalize_eef_and_gripper, get_env_spaces
import pickle
from collections import defaultdict
from tqdm import tqdm, trange
#from model_based_irl_torch.common.utils import to_np
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.transforms import functional as F

DINO_transform_wrist = transforms.Compose([           
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225])
                                ])
def crop_top_middle(image):
    top = 8
    left = 48
    height = 224
    width = 224
    return F.crop(image, top, left, height, width)

DINO_transform_front = transforms.Compose([           
                                transforms.Resize(320),
                                transforms.Lambda(crop_top_middle), #should be multiple of model patch_size
                                #transforms.CenterCrop(224), #should be multiple of model patch_size                 
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225])
                                ])


transform_front = transforms.Compose([           
                                transforms.Resize(320),
                                transforms.Lambda(crop_top_middle), #should be multiple of model patch_size                 
                                transforms.Resize(256),
                                transforms.ToTensor(),]
                                )
# Define the path to the pickle files
demo_path = "/data/ken/ken_data/skittles_big_labeled/"
dino =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to('cuda:0')
# Get a list of all pickle files in the directory
pkl_files = [os.path.join(demo_path, f) for f in os.listdir(demo_path) if f.endswith('.pkl')]

pixel_keys = ["cam_rs", "cam_zed_right"] # zed_right gets priority over zed_left


all_acs = []
transitions = 0
for i, pkl_file in tqdm(
    enumerate(pkl_files),
    desc="Loading in expert data",
    ncols=0,
    leave=False,
    total=len(pkl_files),
):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    if 'rand' in pkl_file:
        data = data[1:]
    for t in range(len(data)):
        
        #if "cam_rs_embd" not in data[t][0].keys():
        rs_img = data[t][0]["cam_rs"][0]
        img_PIL = Image.fromarray(np.uint8(rs_img)).convert('RGB')
        img_resized = DINO_transform_wrist(img_PIL).to('cuda:0')
        with torch.no_grad():
            patch_emb = dino.forward_features(img_resized.unsqueeze(0))['x_norm_patchtokens'].squeeze().detach().cpu().numpy()
        data[t][0]["cam_rs_embd"] = patch_emb

        #if "cam_zed_right_embd" not in data[t][0].keys():

        zed_img = data[t][0]["cam_zed_right"][0]
        img_PIL = Image.fromarray(np.uint8(zed_img)).convert('RGB')
        img_resized = DINO_transform_front(img_PIL).to('cuda:0')
        with torch.no_grad():
            patch_emb = dino.forward_features(img_resized.unsqueeze(0))['x_norm_patchtokens'].squeeze().detach().cpu().numpy()
        data[t][0]["cam_zed_right_embd"] = patch_emb
        data[t][0]["cam_zed_crop"] = transform_front(img_PIL).numpy().transpose(1, 2, 0)
        #plt.imshow(data[t][0]["cam_zed_crop"])
        #plt.savefig('test.png')
        all_acs.append(data[t][1])
        transitions += 1

    with open(pkl_file, "wb") as f:
        pickle.dump([data], f)

all_acs = np.array(all_acs)
print('max', np.max(all_acs, axis=0))
print('min', np.min(all_acs, axis=0))
print('total transitions:', transitions)


