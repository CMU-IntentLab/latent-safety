## Lipschitz Continuous Reachability Learning (LCRL)

This is a repo for learning trustworthy reachability sets of high-dimensional nonlinear systems using a newly proposed Lipschitz continuous reachability value function. We also propose two efficient post-learning reach-avoid set verification methods. To the best of our knowledge, they are the first verification methods for providing deterministic guarantees for learned reach-avoid sets, against the worst-case disturbance. If you are interested in learning more about this work, please refer to the following paper: [Certifiable Deep Learning for Reachability Using a New Lipschitz Continuous Value Function](https://arxiv.org/abs/2408.07866), J. Li, D. Lee, J. Lee, K. Dong, S. Sojoudi, C. Tomlin, arxiv 2408.07866.

[Video presentation](https://drive.google.com/file/d/13jlGLcVzhqk0HtwAO3Jc8Xl0BSMKEB4T/view?usp=drive_link).

# Introduction

Inspired by prior works, we propose a new reach-avoid value function, which is Lipschitz continuous, and its Bellman equation is a contraction mapping without the need for any approximation. Our method does not need to anneal the time discount factor to 1, which is commonly used in prior works. This simplifies the training procedure and saves training time. (For example, assuming NN training quality is reasonably good, our method requires only 1/N training time of the prior methods, where N is the number of times that the time-discount factor is annealed.) 

Our two verification methods provide deterministic guarantees of whether a set of states can safely reach the target sets, under all potential disturbances within a prespecified bound. Our first method uses the learned control policy and the Lipschitz constant of the dynamics to construct a theoretical lower bound of the ground truth value function. **This lower bound becomes tight when the learned policy is optimal**. The super zero-level set of this constructed lower bounding function recovers a subset of the ground truth reach-avoid set; Moreover, our second method formulates efficient second-order cone programmings to evaluate the constraint violations and the target set reaching, against the worst-case disturbance. 

In addition, both of our methods can be computed in real-time to certify if a neighboring set of the current state is within the ground truth reach-avoid set. The computational complexity of evaluating our two certifications scales **polynomially** with respect to the state dimensions. Our method can also be used offline for a comprehensive certification, i.e., certifying if a large set of states is within the ground truth reach-avoid set.



# Implementation details and installation guide:

We adapt [minimax DDPG](http://aima.eecs.berkeley.edu/~russell/papers/aaai19-marl.pdf) and minimax SAC to learn our new reach-avoid value function. Our implementation builds upon the deep RL infrastructure of [Tianshou](https://github.com/thu-ml/tianshou) (version 0.5.1).  

We recommend Python version 3.12. 

Install instruction:

1. git clone the repo

2. cd to the root location of this repo, where you should be able to see the "setup.py"

3. run in terminal: pip install -e .

4. run in terminal: conda install -c conda-forge ffmpeg

Brief tutorial: we use `experiment_script/run_training_ddpg.py` to learn our reach-avoid value function. We visualize the learned reach-avoid sets, Lipschitz certified, and SOCP certified sets in `experiment_script/droneracing_post_training_DDPG_eval_new.ipynb`. 

# Some sample training scripts:

For drone racing: 

> python run_training_ddpg.py --task ra_droneracing_Game-v6 --control-net 512 512 512 512 --disturbance-net 512 512 512 512 --critic-net 512 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.95

For highway take-over: 

> python run_training_ddpg.py --task ra_highway_Game-v2 --control-net 512 512 512 --disturbance-net 512 512 512 --critic-net 512 512 512 --epoch 10 --total-episodes 160 --gamma 0.95

# How do we choose training parameters?

You may ask why we choose 4-layer NN for drone racing but 3-layer NN for highway driving? Part of the reason is because drone racing has more constraints. It requires a more complex NN to approximate its value function. There is no rule of thumb for choosing the best network structure because answering this question is as hard as finding conditions to ensure the convergence of deep RL, which seems challenging in practice when we have limited knowledge of the ground truth value function. From our **three-year** reachability learning project experience, we found the following heuristic seems working: 

(1) First, we over-parameterize the NN Q functions and NN policies such that we can bring the critic loss down, which measures the Bellman equation fitting error;
(2) Subsequently, we want to mitigate overfitting by shrinking the size of neural networks, but maintaining small critic loss, because otherwise, the representation power of NN is not enough to capture the complex structure of value function;
(3) We need a larger batch size if the size of NN grows. In our training, we set the batch size equal to the number of neurons in each layer. But this is just a heuristic for your reference. 

**A key thing to notice is that the initial state distribution should cover a portion of the target set. Otherwise, no state in the target set shows up in the data buffer and therefore, the policy cannot see where it should go to maximize the value function. **

In addition, we remark that the convergence of critic loss implies that the neural network value function approximates well the value function induced by the current learned policy. However, it does not mean the learning is done because we cannot tell the quality of policies by just looking at the critic loss. In minimax DDPG, it improves the learned policy by minimizing the control actor loss, and refines the disturbance policy by maximizing the disturbance actor loss. However, we observe that a small critic loss stabilizes the multi-agent reinforcement learning training, and therefore helps policy learning. 

In practice, we suggest training min-max DDPG for 160 episodes, where each episode takes 10 epochs. The precise relationship between episodes and epochs can be found at the bottom of the `run_training_ddpg.py`. We save the trained policy every 10 epochs. Due to the non-stationarity nature of the minimax DDPG training, it is hard to guarantee that the last iteration policy is the best. For now, we have not figured out a better way than just enumerating each saved policy and finding the best among them. This is, of course, conservative. Training ten episodes may be sufficient, as shown in this [jupyter notebook](https://github.com/jamesjingqili/Lipschitz_Continuous_Reachability_Learning/blob/main/experiment_script/highway_post_training_DDPG_evaluation_new.ipynb).

Finally, we recommend always setting the action space to range from -1 to 1 in the gym.env definition, but we can scale or shift the actions within the gym.step() function when defining the dynamics. For example, if we have two double integrator dynamics: the first integrator’s control is bounded by -0.1 to 0.1, and the second integrator’s control is bounded by -0.3 to 0.3. In this case, we can define self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64) and implement the dynamics in gym.step(self, u) as follows:

> x1 = x1 + dt * v1

> v1 = v1 + dt * u[0] * 0.1

> x2 = x2 + dt * v2

> v2 = v2 + dt * u[1] * 0.3

If you are familiar with natural policy gradient methods, you may know that without normalization, the policy gradient can become ill-conditioned because certain actions may dominate the actor network’s output. While the action-bound unification heuristic described above is not equivalent to natural policy gradient methods, it is inspired by them. By unifying the action bounds, we ensure that the magnitude of each dimension in the actor network’s output is balanced, which contributes to stabilizing the training process. For more examples, you can check the files in `LCRL/reach_rl_gym_envs/`.

# Future directions:

In the reachability learning part, although the sufficient condition for our new value function to be Lipschitz continuous assumes that the dynamics, reward, and constraints are Lipschitz continuous, we observe in practice that our learning framework performs well even with non-Lipschitz continuous dynamics, rewards, and constraints. It is exciting to explore this further. Another interesting future direction is to extend our framework to contact-rich multi-robot manipulation scenarios using differentiable contact simulators ([[1]](https://arxiv.org/pdf/2203.00806), [[2]](https://arxiv.org/pdf/2403.08716), [[3]](https://arxiv.org/pdf/2210.09420)). 

In the certification part of our work, we begin by exploring the new idea of using Lipschitz continuity and the learned control policy to construct a theoretical lower bound for the ground truth reach-avoid value function, which is typically unknown and difficult to compute for high-dimensional systems. However, we believe there are several avenues to improve this theoretical lower bound. 

For instance, the critic loss in minimax DDPG approximately reflects the magnitude of the value function fitting error. This raises a natural question: can we utilize critic loss information to construct a high-confidence lower bound for the ground truth value function? While this may require a detailed sample complexity analysis, the Lipschitz continuity of our new value function could be advantageous. (For example, the Lipschitz continuity of the value function is a key assumption in the sample complexity analysis and the DPG convergence proof in this [paper](https://proceedings.mlr.press/v180/xiong22a/xiong22a.pdf).) 

In addition, the quality of this lower bound depends on the learned policy. **We believe our method holds promise for developing a tighter lower bound on the ground truth value function, ultimately enabling the recovery of less conservative but trustworthy reach-avoid sets in high-dimensional problems**. For example, we can use the constructed value function lower bound to refine the policy training, and we can leverage this updated policy to improve this value function lower bound. Deriving a tight deterministic lower bound of the ground truth value function would be even more compelling, though this might necessitate substantial prior knowledge, such as the controllability and stability of the dynamical systems involved in reach-avoid problems.


Moreover, our real-time SOCP certification heavily relies on the [Clarabel](https://clarabel.org/stable/examples/py/example_socp/) QP solver, which, fortunately, is very fast. The current implementation does not exploit the structure of these SOCPs when evaluating worst-case constraint violations and target set reachability. Therefore, we see significant opportunities to enhance our SOCP trajectory-level certification methods for faster computation and reduced conservatism. For example, could agents leverage decentralized SOCP solvers to collaboratively verify their joint safety in complex traffic scenarios? Additionally, how can we harness the sparsity structure of dynamics to simplify SOCP computations? Finally, we want to emphasize that the SOCP certification can be easily extended to non-Lipschitz continuous problems, and we leave this further exploration for future work.


Overall, we are optimistic that in the near future, we will be able to compute trustworthy reach-avoid sets for high-dimensional, real-world systems. Our results suggest that this goal is reachable. If you have any questions or suggestions to improve this work, please feel free to contact the authors. Thank you!!

