import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

results = load_results(".")

# 绘制收敛图
x, y = ts2xy(results, 'timesteps')
plt.plot(x, y, label='TD3')
plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.title('TD3 Training Progress')
plt.legend()
plt.show()
