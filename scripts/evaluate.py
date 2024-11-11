#%% preamble
import RARL_wm
import os
import matplotlib.pyplot as plt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#%% setup
config = RARL_wm.get_config(parse_args=False)
env, environment_info = RARL_wm.construct_environment(config)

#%% load best agent
# construct base agent:
agent = RARL_wm.construct_agent(config, environment_info)
#%%
# TODO get this index from the training dict based on metrics rather than hard-coding
restore_idx = 10_000
# agent.restore(restore_idx, environment_info["outFolder"])
agent.restore(restore_idx, os.path.join(project_root, environment_info["outFolder"]))
# %%
results, minVs, infos = env.plot_trajectories(
    q_func=agent.Q_network,
    num_rnd_traj=10,
    save_dir='safe_rollouts.png',
    return_infos=True,
    enable_observation_feedback=True,
    wait_for_all_metrics_to_predict_failure=True,
)
# %%

%matplotlib inline
plt.show()
