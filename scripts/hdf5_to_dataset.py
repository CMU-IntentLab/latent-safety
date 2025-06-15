import os
import pickle
import h5py
import numpy as np
import torch
from torchvision import transforms

import os
import h5py
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from scipy.spatial.transform import Rotation as R
# Image transforms

def crop_top_middle(image):
    top = 30
    left = 46
    height = 180
    width = 180
    return F.crop(image, top, left, height, width)



#front cam transforms
crop_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: crop_top_middle(img)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
DINO_crop = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.1)),
    transforms.Lambda(lambda img: crop_top_middle(img)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#wrist cam transforms
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
DINO_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def resize_images_to_224(images, key):
    """Resize a batch of images to 224x224."""
    resized = []
    for i in range(len(images)):
        img = images[i]
        if key == "camera_0":  # wrist camera
            img_tensor = resize_transform(img.astype(np.uint8))
        else:
            img_tensor = crop_transform(img.astype(np.uint8))
        resized.append(img_tensor.numpy().transpose(1, 2, 0))  # back to HWC
    return np.stack(resized)

def eef_pose_to_state(T, gripper):

    # Extract translation
    x, y, z = T[:3, 3]
    
    # Extract rotation matrix and convert to quaternion
    rotation_matrix = T[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # Returns [qx, qy, qz, qw]
    
    # Concatenate into final state
    eef_state = np.concatenate(([x, y, z], quat, [gripper]))
    return eef_state


def preprocess(demo_path):
    # Load DINO model
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to('cuda:0')

    # Path to HDF5 files
    hdf5_files = [os.path.join(demo_path, f) for f in os.listdir(demo_path) if f.endswith('.hdf5')]

    pixel_keys = ["rs", "zed_right"]

    all_acs = []
    transitions = 0

    for i, hdf5_file in tqdm(enumerate(hdf5_files), desc="Loading expert data", total=len(hdf5_files), ncols=0, leave=False):
        with h5py.File(hdf5_file, "r+") as f:
            data_group = f["data"]
            if "camera_0" not in data_group or "camera_1" not in data_group:
                print(f"Skipping {hdf5_file} due to missing camera data.")
                continue
            actions = data_group["actions"][:]
            cam_rs = data_group["camera_0"][:]
            cam_zed = data_group["camera_1"][:]
            
            ee_states = data_group["ee_states"][:]
            gripper_states = data_group["gripper_states"][:]

            cam_rs_embds = []
            cam_zed_embds = []
            states = []
            for t in range(actions.shape[0]):
                if "cam_rs_embd" not in data_group:
                    # Wrist (rs) image
                    rs_img = cam_rs[t]
                    img_PIL = Image.fromarray(np.uint8(rs_img)).convert('RGB')
                    img_tensor = DINO_transform(img_PIL).to('cuda:0')
                    with torch.no_grad():
                        patch_emb = dino.forward_features(img_tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze().cpu().numpy()
                    cam_rs_embds.append(patch_emb)

                if "cam_zed_embd" not in data_group:
                    # Zed front image
                    zed_img = cam_zed[t]
                    img_PIL = Image.fromarray(np.uint8(zed_img)).convert('RGB')
                    img_tensor = DINO_crop(img_PIL).to('cuda:0')
                    with torch.no_grad():
                        patch_emb = dino.forward_features(img_tensor.unsqueeze(0))['x_norm_patchtokens'].squeeze().cpu().numpy()
                    cam_zed_embds.append(patch_emb)

                # Convert ee_states to eef_state format
                ee_state = eef_pose_to_state(ee_states[t].reshape(4, 4).T, gripper_states[t])
                states.append(ee_state)


            # Save embeddings and crops in HDF5
            if "cam_rs_embd" not in data_group:
                data_group.create_dataset("cam_rs_embd", data=np.stack(cam_rs_embds))
            if "cam_zed_embd" not in data_group:
                data_group.create_dataset("cam_zed_embd", data=np.stack(cam_zed_embds))
            if "states" not in data_group:
                data_group.create_dataset("states", data=np.stack(states))
            all_acs.extend(actions)
            transitions += len(actions)

    # Final summary
    all_acs = np.array(all_acs)
    print('max', np.max(all_acs, axis=0))
    print('min', np.min(all_acs, axis=0))
    print('total transitions:', transitions)


def convert_hdf5_to_consolidated_hdf5(hdf5_dir, output_hdf5_file):
    """
    Convert all individual HDF5 trajectory files in a directory into a single HDF5 file.
    
    Args:
        hdf5_dir (str): Directory containing trajectory_XXXX.hdf5 files.
        output_hdf5_file (str): Path to the consolidated HDF5 output file.
    """
    with h5py.File(output_hdf5_file, "w") as hf_out:
        for i, hdf5_file in enumerate(sorted(os.listdir(hdf5_dir))):
            if hdf5_file.endswith(".hdf5"):
                file_path = os.path.join(hdf5_dir, hdf5_file)
                with h5py.File(file_path, "r") as hf_in:
                    # Create a new group for this trajectory
                    group = hf_out.create_group(f"trajectory_{i}")
                    
                    # Copy config if it exists
                    if "data" in hf_in:
                        data_group = hf_in["data"]
                        if "config" in data_group.attrs:
                            group.attrs["config"] = data_group.attrs["config"]
                        
                        #if data_group["labels"].shape[0] != data_group["camera_0"].shape[0]:
                        #    print(file_path, "labels and camera_0 length mismatch")
                        #    continue
                        # Copy datasets
                        for key in data_group.keys():
                            #if key == 'labels':
                            #    assert data_group[key].shape[0] == data_group['camera_0'].shape[0] 
                            data = data_group[key][...]
                            if key in ("camera_0", "camera_1"):
                                # Resize camera images
                                data = resize_images_to_224(data, key)
                            group.create_dataset(key, data=data)
                    else:
                        # Fallback if not nested under "data"
                        for key in hf_in.keys():
                            hf_in.copy(hf_in[key], group)

                print(f"Copied {hdf5_file} → trajectory_{i}")



hdf5_dir = "/data/ken/latent"
output_hdf5_file = "/data/ken/latent/consolidated.h5"
preprocess(hdf5_dir)
convert_hdf5_to_consolidated_hdf5(hdf5_dir, output_hdf5_file)