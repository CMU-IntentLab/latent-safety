import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms import functional as F
import imageio.v3 as iio
import h5py
import shutil





# Global variables
current_idx = 0
images = []
labels = {}
current_traj = ""



def on_key_press(event):
    global current_idx

    # Label images with '0' or '1'
    if event.key in {'0', '1', '2'}:
        labels[current_idx] = int(event.key)
        print(f"Image {current_idx} labeled as {labels[current_idx]}")

        # Move to the next image
        current_idx += 1
        if current_idx < len(images):
            update_plot()
        else:
            print("All images labeled for this trajectory! Close the window to exit.")
            fig.canvas.mpl_disconnect(key_press_cid)
            plt.pause(0.5)
            plt.close(fig)


    # Rewind with spacebar
    elif event.key == ' ':
        if current_idx > 0:
            current_idx -= 1
            print(f"Rewound to image {current_idx}. Current label: {labels.get(current_idx, 'None')}")
            update_plot()
    

def update_plot():
    ax.clear()
    ax.imshow(images[current_idx])
    ax.axis('off')
    fig.canvas.draw()

def process_trajectory(traj_file):
    """Load images from a trajectory file and set up labels."""
    global images, labels, current_idx, current_traj

    current_traj = os.path.splitext(os.path.basename(traj_file))[0]
    print(f"Processing trajectory: {current_traj}")
    labels = {}

    # Load images
    images = []
    with h5py.File(traj_file, "r") as hf:
        data = hf['data']
        for i in range(data["camera_0"][:].shape[0]):
            wrist = data["camera_0"][i]
            front = data["camera_1"][i]

            #front_image = Image.fromarray(front)
            #front_image = Image.fromarray((front_image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            #front = np.array(front_image)

            joint = np.concatenate([wrist, front], axis=1)
            images.append(joint)

    # Initialize index
    current_idx = 0
    while current_idx < len(images) and current_idx in labels:
        current_idx += 1


def postprocess_trajectory(done_file, labels, traj_file):
    """Load images from a trajectory file and set up labels."""

    shutil.copy(traj_file, done_file)

    # write to done_file
    with h5py.File(done_file, "r+") as hf:
        data_group = hf['data']
        
        print(f"Assigning labels to {done_file}.")
        labels = np.array(list(labels.values()))
        print(f"Labels: {labels}")
        print(labels.shape)
        print(data_group["camera_0"][:].shape, data_group["camera_1"].shape)
        if "labels" in data_group:
            del data_group["labels"]
        data_group.create_dataset("labels", data=np.array(labels))


# Initialize the plot
plt.ion()

if __name__ == "__main__":


    #directory = "/data/ken/ken_data/skittles_big"
    #labeled_directory = "/data/ken/ken_data/skittles_big_labeled_new"

    directory = "/data/ken/ken_data/mm-unsafe"
    labeled_directory = "/data/ken/ken_data/mm-unsafe-labeled"


    directory = "/data/ken/latent-unsafe-test"
    labeled_directory = "/data/ken/latent-unsafe-test-labeled2"
    # make labeled directory if it does not exist
    if not os.path.exists(labeled_directory):
        os.makedirs(labeled_directory)

    # Get all pickle files with "unsafe" in the filename
    hdf5_files = [f for f in os.listdir(directory)]
    hdf5_files = [f for f in hdf5_files if "unsafe" in f]
    print('total files:', len(hdf5_files))
    done_files = [f for f in os.listdir(labeled_directory)]
    print('done files:', len(done_files))
    hdf5_files = list(set(hdf5_files) - set(done_files))
    print('remaining files:', len(hdf5_files))
    # Get the full paths

    tot = len(hdf5_files)
    don = 0

    for traj_file in hdf5_files:# in range(10): 
        done_file = os.path.join(labeled_directory, traj_file)
        traj_file = os.path.join(directory, traj_file)


        fig, ax = plt.subplots()

        if not os.path.exists(traj_file):
            print(f"File {traj_file} not found, skipping.")
            continue

        print(f"Processing {traj_file}...")
        process_trajectory(traj_file)
        if images:
            update_plot()
            key_press_cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
            print(f"Press '0' as safe or '1' as unsafe to label {traj_file}.")
            plt.show(block=True)

        postprocess_trajectory(done_file, labels, traj_file)
        don += 1
        print(f"Done {don}/{tot}")
        print(f"Finished labeling for {traj_file}.")
