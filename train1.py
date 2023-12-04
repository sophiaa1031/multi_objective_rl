# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import TD3
# import matplotlib.pyplot as plt
# import MDRA_Env2
# import numpy as np
# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#
# # 创建环境
# env = MDRA_Env2.UAVMDRAEnv()
# env = DummyVecEnv([lambda: env])
#
# # 创建TD3模型
# model = TD3("MlpPolicy", env, verbose=0)
#
# # 记录训练过程中的奖励
# mean_rewards = []
# n_steps = 3000
# for i in range(n_steps):
#     model.learn(total_timesteps=1)
#     mean_rewards.append(np.mean(model.ep_info_buffer))
#
# # 绘制收敛图
# plt.plot(np.arange(len(mean_rewards)), mean_rewards)
# plt.xlabel('Training Steps')
# plt.ylabel('Mean Rewards')
# plt.title('TD3 Training Progress')
# plt.show()
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
import multi_objective_rl.QFL_Env2 as QFL_Env2
import numpy as np
from collections import deque
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 创建环境
env = QFL_Env2.UAVMDRAEnv()
env = DummyVecEnv([lambda: env])

# 创建TD3模型
model = TD3("MlpPolicy", env, verbose=0)

# 记录训练过程中的奖励
reward = []
n_steps = 3000
for i in range(n_steps):
    model.learn(total_timesteps=1)
    if len(model.ep_info_buffer) > 0:
        episode_rewards = model.ep_info_buffer[-1]['r']
        if len(episode_rewards) > 0:
            reward.append(np.mean(episode_rewards))
        else:
            print("No rewards in the latest episode. Skipping this step.")
    else:
        print("No episode information in the buffer. Skipping this step.")

# 绘制收敛图
if len(reward) > 0:
    plt.plot(np.arange(len(reward)), reward)
    plt.xlabel('Training Steps')
    plt.ylabel('Mean Rewards')
    plt.title('TD3 Training Progress')
    plt.show()
else:
    print("No rewards collected during training. Please check your environment and training configuration.")

