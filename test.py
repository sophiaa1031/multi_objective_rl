import numpy as np
import QFL_Env_clientnumber_latency
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from options import args_parser

def plot_reward(rewards):
    time_steps = np.arange(len(rewards))

    # 创建奖励曲线
    plt.plot(time_steps, rewards, label='Reward', marker='o', linestyle='-')

    # 添加标签和标题
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.title('Reward Curve')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

# Load the trained model
model = PPO.load('ppo_saved_weight')
args = args_parser()
args.baseline = 'rb'

# Create the environment
# Assuming 'QFL_Env_clientnumber_latency.QFLEnv' is your custom environment class
# Adjust the arguments as needed for your environment
env = QFL_Env_clientnumber_latency.QFLEnv(args,weight_lambda=0.6,lmax=0.35)
env = Monitor(env)

# # Evaluate the policy
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the agent
obs = env.reset()[0]
total_reward = []
for i in range(50):
    action, _states = model.predict(obs, deterministic=True)
    # print("round{}:, mix:{.2f}, accumulated_obj2{.2f}, max{.2f},".format(i, obj2_min, accumulated_obj2, obj2_max))
    print(i)
    obs, rewards, dones, info, _ = env.step(action)
    total_reward.append(rewards)
    env.render()

    if dones:
        obs = env.reset()
plot_reward(total_reward)
env.close()

