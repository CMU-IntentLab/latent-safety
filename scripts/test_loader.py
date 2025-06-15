import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

class SplitTrajectoryDataset(Dataset):
    def __init__(self, hdf5_file, segment_length, split='train', num_test=100):
        """
        Custom Dataset that can load either the first 1000 trajectories or the rest.
        
        Args:
            hdf5_file (str): Path to the HDF5 file containing the trajectories.
            segment_length (int): Length of the segments to sample (H timesteps).
            split (str): 'train' for the first 1000 trajectories, 'test' for the rest.
            num_train (int): The number of trajectories to use in the training set.
        """
        self.hdf5_file = hdf5_file
        self.segment_length = segment_length
        self.split = split
        self.num_test = num_test
        
        # Open HDF5 file to get a list of trajectory groups
        with h5py.File(self.hdf5_file, 'r') as hf:
            self.trajectory_ids = list(hf.keys())
        

        # Split the dataset based on the specified split
        if self.split == 'train':
            self.trajectory_ids = self.trajectory_ids[self.num_test:]
        elif self.split == 'test':
            self.trajectory_ids = self.trajectory_ids[:self.num_test]
        else:
            raise ValueError("split must be 'train' or 'test'.")
        
        # Precompute trajectory slice indices
        self.slice_indices = []
        with h5py.File(self.hdf5_file, 'r') as hf:
            for traj_id in self.trajectory_ids:
                trajectory = hf[traj_id]
                traj_len = len(trajectory['actions'])
                for start_idx in range(0, traj_len - self.segment_length + 1, 1):
                    self.slice_indices.append((traj_id, start_idx))


    def __len__(self):
        """Returns the number of trajectories in the selected split."""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """Randomly samples a segment from a randomly selected trajectory."""

        traj_id, start_idx = self.slice_indices[idx]

        with h5py.File(self.hdf5_file, 'r') as hf:
            trajectory = hf[traj_id]

            # Get the trajectory data
            observations = trajectory['observations']
            actions = trajectory['actions'][:]
            rewards = trajectory['rewards'][:]

            # Compute end index
            end_idx = start_idx + self.segment_length

            # Extract the segment of observations, actions, and rewards
            segment_obs = {key: observations[key][start_idx:end_idx] for key in observations.keys()}
            segment_actions = actions[start_idx:end_idx]
            segment_rewards = rewards[start_idx:end_idx]
            
            # Convert the segment into torch tensors
            segment_obs_tensor = {}
            for key, value in segment_obs.items():
                if key == 'cam_zed_crop':
                    segment_obs_tensor["agentview_image"] = torch.tensor(value, dtype=torch.uint8)
                elif key == 'cam_rs':
                    segment_obs_tensor["robot0_eye_in_hand_image"] = torch.tensor(value, dtype=torch.uint8)
                else:
                    segment_obs_tensor[key] = torch.tensor(value, dtype=torch.float32)
            
            #segment_actions[:,:, 0] = segment_actions[:,:, 0] * (X_RANGE[1] - X_RANGE[0]) + X_RANGE[0]
            #segment_actions[:,:, 1] = segment_actions[:,:, 1] * (Y_RANGE[1] - Y_RANGE[0]) + Y_RANGE[0]
            #segment_actions[:,:, 2] = segment_actions[:,:, 2] * (Z_RANGE[1] - Z_RANGE[0]) + Z_RANGE[0]  
            #segment_actions[:,:, -1] = segment_actions[:,:, -1] * (GRIPPER_RANGE[1] - GRIPPER_RANGE[0]) + GRIPPER_RANGE[0]  
            segment_obs_tensor["action"] = torch.tensor(segment_actions, dtype=torch.float32)

            segment_obs_tensor["is_first"] = torch.zeros(self.segment_length)
            segment_obs_tensor["is_last"] = torch.zeros(self.segment_length)
            segment_obs_tensor["is_first"][0] = 1.
            segment_obs_tensor["is_terminal"] = segment_obs_tensor["is_last"]
            segment_obs_tensor["discount"] = torch.ones(self.segment_length, dtype=torch.float32)

        return segment_obs_tensor
    
if __name__ == '__main__':
    # Path to your HDF5 file
    hdf5_file = '/home/kensuke/data/skittles_trajectories_dreamer.h5'
    segment_length = 32  # Number of timesteps per segment
    batch_size = 32      # Number of trajectories per batch

    # Create the dataset
    train_dataset = SplitTrajectoryDataset(hdf5_file, segment_length, split='train', num_train=1000)
    test_dataset = SplitTrajectoryDataset(hdf5_file, segment_length, split='test', num_train=1000)


    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Example usage:
    for batch_idx, data in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(data.keys())
        print(data['agentview_image'].shape)
        print(data['agentview_image'].max())
        print(f"Observations: {data['cam_zed_right_embd'].shape}")
        print(f"Actions: {data['action'].shape}")

        break  # Just print one batch

    for batch_idx, data in enumerate(test_loader):
        print(f"Batch {batch_idx}:")
        print(data.keys())
        print(f"Observations: {data['cam_zed_right_embd'].shape}")
        print(f"Actions: {data['action'].shape}")


        
        break  # Just print one batch