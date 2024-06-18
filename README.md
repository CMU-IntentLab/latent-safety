# latent-safety

# making conda env
conda create --name latent --file requirements.txt

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
