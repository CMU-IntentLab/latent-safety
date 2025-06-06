# Latent-space Reachability in Pytorch

This is a repository for doing latent-space HJ reachability analysis using the Discounted Safety/Reach-avoid Bellman equation originally introduced in [this](https://ieeexplore.ieee.org/abstract/document/8794107) and [this](https://arxiv.org/abs/2112.12288) paper. This entire repository is heavily based on [Jingqi Li's Repo](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning) which uses Tianshou for RL. We also use the RSSM world model as implemented in [this](https://github.com/NM512/dreamerv3-torch) implementation of Dreamer-v3. 


This repository supports a DDPG implementation of the both the latent safety-only value function. To more easily promote future research, we provide an implementation of a Dubin's car with <i>continuous</i> action spaces in this repository, as opposed to the discrete action space version used in the paper. 


## Installation

We recommend Python version 3.12. 

Install instruction:

1. git clone the repo

2. cd to the root location of this repo, where you should be able to see the "setup.py". Note that if you use MacOS, then pytorch 2.4.0 is not available, and therefore you have to first change the line 22 of setup.py from "pytorch==2.4.0" to "pytorch==2.2.2", and then do the step 3. (However, Pytorch==2.4.0 is available for Ubuntu systems. So, if you use Ubuntu, then you can directly go to step 3. )

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg


## Latent reachability


To get the offline dataset for a Dubin's car model we need to rollout trajectories using randomly sampled actions:

> python scripts/generate_data_cont.py


World model training from the offline dataset. This differs from the standard training procedure from Dreamer, as this is done solely with offline data (although in-principle, you could also use online rollouts.) without task-reward information. In this implementation, we co-train the world model with the failure classifier.

> python scripts/dreamer_offline.py


Reachability analysis in the world model via RL:

> python scripts/run_training_ddpg-wm.py





