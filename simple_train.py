from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
import MDRA_Env2
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 创建环境
env = MDRA_Env2.UAVMDRAEnv()
env = DummyVecEnv([lambda: env])
# 创建TD3模型
model = TD3("MlpPolicy", env, verbose=0)
# 训练模型
model.learn(total_timesteps=3000)

# 通过使用 'load_results' 函数加载训练结果
results = load_results(".")

# 绘制收敛图
x, y = ts2xy(results, 'timesteps')
plt.plot(x, y, label='TD3')
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.title('TD3 Training Progress')
plt.legend()
plt.show()


# 训练模型
# mean_rewards = []
# #for i in range(100):  # 假设进行1000个训练步
# model.learn(total_timesteps=3000)  # 假设每个训练步包含1000个时间步
#
# # 获取每个训练步的平均奖励
# episode_rewards = model.ep_info_buffer
# mean_reward = np.mean([ep_info['r'] for ep_info in episode_rewards])
# mean_rewards.append(mean_reward)
#
# # 绘制收敛图
# plt.plot(mean_rewards)
# plt.xlabel('Training Steps')
# plt.ylabel('Mean Reward')
# plt.title('TD3 Training Convergence')
# plt.show()





# model.learn(total_timesteps=1000)
# # 评估模型
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
#
# # 获取训练过程中的奖曲线
# ep_rewards = [ep_info['reward'] for ep_info in model.ep_info_buffer]
#
# # 计算滑动平均奖励
# window_size = 100
# smoothed_rewards = np.convolve(ep_rewards, np.ones(window_size)/window_size, mode='valid')
#
# # 绘制收敛图
# plt.plot(np.arange(len(smoothed_rewards)), smoothed_rewards)
# plt.xlabel('Episodes')
# plt.ylabel('Average Reward')
# plt.title('TD3 Training')
# plt.show()







# 测试训练好的模型
# obs = env.reset()
# total_reward = 0.0
# x = 0.0
# step_reward = []
# for i in range(8):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     print("The reward is:", reward)
#     x += reward
#     step_reward.append(x)
#     if done:
#         break
# print("Total reward:", total_reward)
# print(step_reward)
# xx = []
# for i in range(10):
#     xx.append(i+1)
# plt.plot(step_reward, xx)
# plt.show()


# 记录奖励的变化
# episode_rewards = []
#
# # 训练模型
# total_episodes = 1000
# for episode in range(total_episodes):
#     obs = env.reset()
#     episode_reward = 0
#
#     for step in range(8):  # 假设每个episode有100个时间步
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, _ = env.step(action)
#
#         episode_reward += reward
#         if done:
#             break
#
#     episode_rewards.append(episode_reward)
#
# # 绘制奖励随episode的变化图
# plt.plot(range(total_episodes), episode_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("Reward vs. Episode")
# plt.show()