import os
import pickle
import h5py
import numpy as np
import torch
from torchvision import transforms

DINO_transform = transforms.Resize(224)
dreamer_transform = transforms.Resize(128)
                    
def convert_pkl_to_hdf5(pkl_dir, hdf5_file,mode='DINO'):
    """
    Convert all .pkl files in a directory to a single HDF5 file.
    
    Args:
        pkl_dir (str): Path to the directory containing .pkl files.
        hdf5_file (str): Path to the output HDF5 file.
    """
    # Open the HDF5 file
    with h5py.File(hdf5_file, "w") as hf:
        # Iterate over all .pkl files in the directory
        for i, pkl_file in enumerate(sorted(os.listdir(pkl_dir))):
            if pkl_file.endswith(".pkl") and "unsafe" in pkl_file:
                # Load the pickle file
                with open(os.path.join(pkl_dir, pkl_file), "rb") as f:
                    trajectory = pickle.load(f)

                # Create a group for each trajectory
                group = hf.create_group(f"trajectory_{i}")

                # Prepare containers for actions and rewards
                actions = []
                rewards = []

                # Iterate through time steps to process observations
                obs_keys = set()  # Collect all keys across time steps for consistency
                processed_observations = []  # To store processed observation dictionaries

                for timestep, (obs_dict, action, reward) in enumerate(trajectory):
                    actions.append(action)
                    rewards.append(reward)
                    
                    # Flatten observation dictionary and collect all keys
                    processed_obs = {}
                    for key, value in obs_dict.items():
                        if key == 'cam_rs' or key == 'cam_zed_right':
                            value = value[0]
                            if mode == 'DINO':
                                value = DINO_transform(torch.tensor(value).permute(2, 0, 1)).numpy()
                                value = value.astype(np.uint8)
                            else:
                                value = dreamer_transform(torch.tensor(value).permute(2, 0, 1)).numpy()
                                value = value.astype(np.uint8)
                                value = value.transpose(1, 2, 0)
                        if key == 'cam_zed_crop':
                            value = 255*value

                            if mode == 'DINO':
                                value = DINO_transform(torch.tensor(value).permute(2, 0, 1)).numpy()
                                value = value.astype(np.uint8)
                            else:
                                value = dreamer_transform(torch.tensor(value).permute(2, 0, 1)).numpy()
                                value = value.astype(np.uint8)
                                value = value.transpose(1, 2, 0)
                        if key == 'actor':
                            continue
                        
                        obs_keys.add(key)
                        processed_obs[key] = np.array(value)
                    processed_observations.append(processed_obs)

                # Save observations: One dataset for each key
                obs_group = group.create_group("observations")
                for key in obs_keys:
                    key_data = [obs.get(key, None) for obs in processed_observations]
                    key_data = np.array(key_data)  # Convert to NumPy array
                    if key == 'label':
                        key_data = key_data[:, None]
                    print(key, key_data.shape)
                    obs_group.create_dataset(key, data=key_data)

                # Save actions and rewards
                group.create_dataset("actions", data=np.array(actions, dtype=np.float32))
                group.create_dataset("rewards", data=np.array(rewards, dtype=np.float32))

                print(f"Converted {pkl_file} to HDF5 group trajectory_{i}")


# Directory containing .pkl files
pkl_directory = "/data/ken/ken_data/skittles_big_labeled/"

# Output HDF5 file

output_hdf5 = "/data/ken/ken_data/skittles_trajectories_unsafe_labeled.h5"


convert_pkl_to_hdf5(pkl_directory, output_hdf5,mode='DINO')