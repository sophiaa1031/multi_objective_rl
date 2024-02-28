import numpy as np
import QFL_Env_clientnumber_latency
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from options import args_parser

def plot_rewards(rewards):
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.show()
# Load the trained model
model = PPO.load('ppo_saved_weight')
args = args_parser()
args.baseline = 'rq'
args.lmax = 0.3

env = QFL_Env_clientnumber_latency.QFLEnv(args,weight_lambda=0.5)
env = Monitor(env)

# # Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the agent
rewards = []  # 收集每个评估回合的奖励
for i in range(10):
    obs = env.reset()[0]
    episode_reward = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        # print("round{}:, mix:{.2f}, accumulated_obj2{.2f}, max{.2f},".format(i, obj2_min, accumulated_obj2, obj2_max))
        print(i)
        obs, reward, done, info, _ = env.step(action)
        episode_reward += reward
        if done:
            rewards.append(episode_reward)
            break
plot_rewards(rewards)

