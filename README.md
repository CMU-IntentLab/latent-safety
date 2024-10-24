# latent-safety

# making conda env
conda env create -f requirements.yaml

# generating data
mkdir datasets
python scripts/generate_data_traj.py

# training WM + Classifier
python scripts/train_wm.py


Add the best pretrained classifier and WM to the configs/config.yaml file
from_ckpt: # wm
lx_ckpt: # classifier


Run reachability computation
python scripts/RARL_wm.py


## EVERYTHING BELOW IS OLD
# generating image-based pickle files:
conda activate latent
python scripts/generate_data.py

# training models
python scripts.train_classifier.py

python scripts.train_classifier_img.py

python scripts.train_dynamics.py

python scripts.train_dynamics_img.py


# getting grid-based solution
python scripts/toy.py

# computing RL value fn
python3 scripts/RARL.py  -sf -w -wi 1000 -g 0.9999 -n RL_value 


Flags:
- -ld # use latent dynamics
- -lm # use latent margin fn
- -img # use img-based margin+dynamics
- --debug # dont compute global confusion matrix (saves time)
