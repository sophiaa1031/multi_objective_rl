from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor, get_monitor_files, LoadMonitorResultsError
import json
import pandas
import math
import ast  # 用于解析字符串中的列表

log_dir = "log_reward_weight/"
log_time = 'training_time(s).txt'
log_pf = 'pareto_front/'
file_pf = 'plot_pf.txt'


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def clear_file_contents(file_path):
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # 清空文件内容
    except Exception as e:
        print(f"Error: {e}")


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
    mean_value = np.mean(y)
    std_value = np.std(y)
    # 输出结果
    print("平均值：", mean_value)
    print("标准差：", std_value)


# plot_results(log_dir)
# 通过使用 'load_results' 函数加载训练结果

def plot_reward():
    results = load_results(".")  # load_results("./log_reward_weight/")
    # 绘制收敛图
    x, y = ts2xy(results, 'timesteps')
    y = moving_average(y, window=10)
    x = x[len(x) - len(y):]
    plt.plot(x, y, label='TD3')
    # plt.ylim(-200,0)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title('TD3 Training Progress')
    plt.legend()
    plt.show()


def plot_time(path):
    # 读取文本文件中的数据
    f = open(path, 'r')
    data = f.read().split(',')
    plt.bar(range(len(data[:-1])), list(map(float, data[:-1])))
    # plt.title('Data Plot')
    plt.xlabel('Subproblem index')
    plt.ylabel('Training time (s)')
    plt.show()


def save_pf(dir_path, path): # 存入已经是收敛后的10个数中的平均数
    clear_file_contents(path)
    file_name = os.listdir(dir_path)
    file_name_sorted = sorted(file_name, key=str)
    obj_records = []
    for i in file_name_sorted:
        with open(log_pf + '/' + i, 'r') as f2:  # 打开文件
            lines = f2.readlines()  # 读取所有行
            last_10_lines = lines[-10:]  # 获取最后10行
            avg_values = []

            for line in last_10_lines:
                parts = line.strip().split('\t')  # 拆分每行的内容
                if len(parts) > 0:  # 确保有足够的部分来解析
                    obj1_list = ast.literal_eval(parts[0])
                    obj2_list = ast.literal_eval(parts[1])
                    reward_list = ast.literal_eval(parts[2])
                    # slct_list = ast.literal_eval(parts[3])
                    # bd_list = ast.literal_eval(parts[4])
                    # quant_list =ast.literal_eval(parts[5])
                    # slct_values = [float(val) for val in slct_list]  # 转换为浮点数
                    avg_values.append([obj1_list,obj2_list])  # 添加到平均值列表

            if avg_values:
                avg_value = [sum(item) / len(item) for item in zip(*avg_values)]
                # avg_value = sum(avg_values) / len(avg_values)  # 计算平均值
                obj_records.append(avg_value)  # 添加平均值到列表

    f = open(path, 'a')
    for avg_value in obj_records:
        # f.write(f"{avg_value:.3f}\n")  # 写入平均值（保留三位小数）
        for item in avg_value:
            f.write("{:.3f}\t".format(item))
        f.write("\n")

def plot_pf(dir_path, path):
    save_pf(dir_path, path)
    f = open(path, 'r')
    x = np.array([])
    y = np.array([])
    for line in f.readlines():
        data = line.replace('\n', '').split('\t')[:2]
        # print (line)
        # print(data)
        data = list(map(float, data))
        x = np.append(x, data[0])
        y = np.append(y, data[1])
    # 使用argsort函数按照x从小到大的顺序排序索引值
    sorted_index = np.argsort(x)
    # 重新排列x和y
    x_sorted = x[sorted_index]
    y_sorted = y[sorted_index]
    # 绘制散点图
    plt.plot(x_sorted, y_sorted, '-o')
    # 添加轴标签和标题
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front')
    # 显示图形
    plt.show()

def plot_pf_original(dir_path, path):
    save_pf(dir_path, path)
    f = open(path, 'r')
    x = np.array([])
    y = np.array([])
    for line in f.readlines():
        data = line.replace('\n', '').split('\t')[:2]
        # print (line)
        # print(data)
        data = list(map(float, data))
        x = np.append(x, data[0])
        y = np.append(y, data[1])
    # 绘制散点图
    plt.plot(x, y, '-*')
    plt.scatter(x[0], y[0], color='red',marker='o', label='起始点')
    # 添加轴标签和标题
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front Original')
    plt.show()

def multi_reward():
    results = os.listdir('pareto_front/')
    sorted_results = sorted(results, key=str)
    num_rows = math.ceil(len(sorted_results) / 3)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))
    for i, df in enumerate(sorted_results):
        y = np.array([])
        with open(os.path.join('pareto_front/', df), 'r') as f:
            for line in f.readlines():
                data = line.replace('\n', '').split('\t')[2]
                data = float(data)
                y = np.append(y, data)
        x = np.arange(1, len(y)+1)
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        y = moving_average(y, window=100)
        x = x[len(x) - len(y):]
        ax.plot(x, y)  # 使用不同的标签来区分不同的DataFrame
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rewards')
        ax.set_title(df)
        # ax.legend()
    plt.tight_layout()
    plt.show()


def load_results_sorted(path: str):
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    monitor_files_sorted = sorted(monitor_files)
    if len(monitor_files_sorted) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{Monitor.EXT} found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files_sorted:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
            data_frame.sort_values("t", inplace=True)
            data_frame.reset_index(inplace=True)
        data_frames.append(data_frame)
    for data_frame in data_frames:
        data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frames, monitor_files_sorted


def multi_reward_monitor():
    # 绘制收敛图
    results, monitor_files = load_results_sorted("./log_reward_weight/")
    num_rows = math.ceil(len(results) / 3)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))
    for i, df in enumerate(results):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        x, y = ts2xy(df, 'timesteps')
        y = moving_average(y, window=100)
        x = x[len(x) - len(y):]
        ax.plot(x, y)  # 使用不同的标签来区分不同的DataFrame
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rewards')
        ax.set_title(monitor_files[i].split('/')[-1])
        # ax.legend()
    plt.tight_layout()
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

# 画那些图在这里决定
if __name__ == "__main__":
    # plot_reward()  # 一个reward
    # plot_time(log_time)  # 多目标的训练时间
    # plot_pf(log_pf, file_pf)  # 多目标的pareto面
    plot_pf_original(log_pf, file_pf)  # 多目标的pareto面
    # multi_reward()  # 多目标的reward收敛图
    # multi_reward_monitor()
