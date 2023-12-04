from stable_baselines3 import TD3, DDPG
import numpy as np
import matplotlib.pyplot as plt
import MDRA_Env3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
import gym
from gym import spaces
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""
本文件用于训练并画出累积奖励，如果训练或者画图出现问题，删掉log_reward中的文件
code source:
https://colab.research.google.com/github/Stable-Baselines-Team/r
l-colab-notebooks/blob/sb3/monitor_training.ipynb#scrollTo=mPXYbV39DiCj
"""

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                # if self.verbose > 0:
                #     #print(f"Num timesteps: {self.num_timesteps}")
                #     print(
                #        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                #     )
                    # print(mean_reward)
                    # with open('mean_reward-7K.txt', 'a') as f:
                    #     f.write(str(mean_reward) + '\n')
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    # if self.verbose > 0:
                    #     print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


# Create log dir
log_dir = "./log_reward"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
#env = MDRA_Env2.UAVMDRAEnv()
env = MDRA_Env3.UAVMDRAEnv()
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)
# Create action noise because TD3 and DDPG use a deterministic policy
# Create the callback: check every 1000 steps····················
callback = SaveOnBestTrainingRewardCallback(check_freq=50, log_dir=log_dir)
# Create RL model
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
#target_aciton_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, learning_rate=1e-3) #,batch_size=500
#model = DDPG('MlpPolicy', env, verbose=0, learning_rate=0.001)
# Train the agent
model.learn(total_timesteps=100000, callback=callback)


# Helper from the library
# results_plotter.plot_results(
# [log_dir], 5000, results_plotter.X_TIMESTEPS, "ppo reward")
# plt.show()

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]
    print(x.shape)
    print(y.shape)

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Episode")
    plt.ylabel("Accumulated reward")
    plt.title(title + " Smoothed")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig("Fig conver", dpi=600)
    plt.show()
    with open('reward.txt', 'w') as f:
        for reward in y:
            f.write(str(reward) + '\n')


    # 计算平均值和标准差
    # mean_value = np.mean(y)
    # std_value = np.std(y)
    # 输出结果
    # print("平均值：", mean_value)
    # print("标准差：", std_value)


plot_results(log_dir)