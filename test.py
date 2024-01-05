import numpy as np
import QFL_Env_clientnumber_latency
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor



# Load the trained model
model = PPO.load('ppo_saved_weight')

# Create the environment
# Assuming 'QFL_Env_clientnumber_latency.QFLEnv' is your custom environment class
# Adjust the arguments as needed for your environment
env = QFL_Env_clientnumber_latency.QFLEnv()
env = Monitor(env)

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the agent
obs = env.reset()[0]
total_reward = 0
for _ in range(50):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info, _ = env.step(action)
    total_reward.append(rewards)
    env.render()

    if dones:
        obs = env.reset()
print('the total reward is:'+total_reward)
env.close()