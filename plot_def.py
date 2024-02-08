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

path1 = 'paper/reward_method/RQ_1e-4.txt'
path2 = 'paper/reward_lr/RandomB.txt'
path3 = 'paper/reward_nn/VQFL_nn=(256,64,64).txt'
path4 = 'paper/obj1_obj2_lmax/0.2.txt'
path5 = 'paper/obj1_obj2_qe/06_8.txt'
reward_path = path1

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
            avg_values_norm = []

            for line in last_10_lines:
                parts = line.strip().split('\t')  # 拆分每行的内容
                if len(parts) > 0:  # 确保有足够的部分来解析
                    obj1_list = ast.literal_eval(parts[0])
                    obj2_list = ast.literal_eval(parts[1])
                    reward_list = ast.literal_eval(parts[2])
                    if len(parts) > 2:  # 确保有足够的部分来解析
                        obj1_norm_list = ast.literal_eval(parts[3])
                        obj2_norm_list = ast.literal_eval(parts[4])
                        avg_values_norm.append([obj1_norm_list, obj2_norm_list])  # 添加到平均值列表
                    # quant_list =ast.literal_eval(parts[5])
                    # slct_values = [float(val) for val in slct_list]  # 转换为浮点数
                    avg_values.append([obj1_list,obj2_list])  # 添加到平均值列表

            if avg_values:
                avg_value = [sum(item) / len(item) for item in zip(*avg_values)]
                if len(avg_values_norm)>0:  # 确保有足够的部分来解析
                    result = [sum(item) / len(item) for item in zip(*avg_values_norm)]
                    print('norm_list: {:.3f},{:.3f}'.format(result[0],result[1]))
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


def save_reward(path, data):
    try:
        with open(path, 'w') as file:
            np.savetxt(file, data)
    except Exception as e:
        print(f"保存数据到文件时出现错误：{e}")

def multi_reward(reward_path):
    results = os.listdir('pareto_front/')
    sorted_results = sorted(results, key=str)
    num_rows = math.ceil(len(sorted_results) / 3)
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))
    for i, df in enumerate(sorted_results):
        y = np.array([])
        obj = np.array([])
        with open(os.path.join('pareto_front/', df), 'r') as f:
            for line in f.readlines():
                data = line.replace('\n', '').split('\t')[2]
                obj1 = line.replace('\n', '').split('\t')[0]
                obj2 = line.replace('\n', '').split('\t')[1]
                data = float(data)
                obj1 = float(obj1)
                obj2 = float(obj2)
                y = np.append(y, data)
                obj = np.append(obj, [obj1,obj2])
        # save_data = obj.reshape(-1, 2)  # 存obj1，obj2
        save_data = y  # 存reward
        # save_reward(reward_path,save_data)
        x = np.arange(1, len(y)+1)
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        y = moving_average(y, window=10)
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
        y = moving_average(y, window=10)
        x = x[len(x) - len(y):]
        ax.plot(x, y)  # 使用不同的标签来区分不同的DataFrame
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rewards')
        ax.set_title(monitor_files[i].split('/')[-1])
        # ax.legend()
    plt.tight_layout()
    plt.show()

def plot_reward_diff_setting():
    folder_path = 'paper/reward_0114'
    plt.figure(figsize=(6.5, 4.5))  # 创建一个新的图形窗口
    # folder = os.listdir(folder_path)
    # folder = ['RQ_1e-4.txt', 'RB_1e-3.txt', 'RB_1e-5.txt', 'VQFL_256-64-64.txt', 'VQFL_1e-3.txt', 'RB_1e-4.txt', 'VQFL_1e-5.txt', 'VQFL_128-32-32.txt', 'RandomB.txt', 'VQFL_256-64.txt']
    # folder = ['RQ_1e-4.txt', 'RB_1e-4.txt', 'VQFL_256-64-64.txt']
    # folder = ['VQFL_1e-3.txt', 'VQFL_256-64-64.txt', 'VQFL_1e-5.txt','RB_1e-3.txt', 'RB_1e-4.txt', 'RB_1e-5.txt', 'RQ_1e-3.txt', 'RQ_1e-4.txt', 'RQ_1e-5.txt']
    folder = ['VQFL_256-64-64.txt','VQFL_128-32-32.txt','VQFL_256-64.txt']
    for filename in folder:
        print(filename)
        file_path = os.path.join(folder_path, filename)
        legend_label = os.path.splitext(filename)[0]  # 使用文件名作为legend标签

        # 从文件中读取数据
        data = np.loadtxt(file_path)
        smoothed_data = moving_average(data, 100)

        # 获取数据的排数（x轴）
        num_rows = len(smoothed_data)

        # 绘制原始数据的浅色线
        # plt.plot(range(len(data)), data, alpha=0.5)

        # 绘制数据
        plt.plot(range(len(data)-len(smoothed_data), len(data)), smoothed_data, label=legend_label)

    plt.xlabel('Episodes',fontsize=12)  # 设置x轴标签
    plt.ylabel('Reward',fontsize=12)  # 设置y轴标签
    plt.legend(fontsize=12)  # 显示legend标签
    # plt.legend(fontsize=8)  # 显示legend标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表



def plot_obj1_obj2_reward():
    folder_path = 'paper/obj1_obj2_lmax'
    # 创建一个新的图形窗口，并设置2个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        legend_label = os.path.splitext(filename)[0]  # 使用文件名作为legend标签
        # 从文件中读取数据
        data = np.loadtxt(file_path)
        # 获取数据的排数（x轴）
        num_rows = len(data)
        # 绘制第一个子图的数据（第一列数据）
        ax1.plot(range(1, num_rows + 1), data[:, 0], label=legend_label)
        ax1.set_xlabel('Epoch')  # 设置第一个子图的x轴标签
        ax1.set_ylabel('Reward (Obj1)')  # 设置第一个子图的y轴标签
        # 绘制第二个子图的数据（第二列数据）
        ax2.plot(range(1, num_rows + 1), data[:, 1], label=legend_label)
        ax2.set_xlabel('Epoch')  # 设置第二个子图的x轴标签
        ax2.set_ylabel('Reward (Obj2)')  # 设置第二个子图的y轴标签
    # 在每个子图中显示legend标签
    ax1.legend()
    ax2.legend()
    # 显示网格
    ax1.grid(True)
    ax2.grid(True)
    # 调整子图之间的间距
    plt.tight_layout()
    # 显示图表
    plt.show()

def plot_obj1_obj2_dot():

    # 从文件中读取数据
    file_path = 'paper/obj1_obj2_lmax.txt'
    data = np.loadtxt(file_path)

    # 分离数据列
    x = data[:, 0]
    y1 = data[:, 1]
    y2 = data[:, 2]

    # 创建一个新的图形窗口，并设置两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 4.5))


    # 绘制第一个子图
    ax1.plot(x, y1,'-o', label='Subplot 1')
    ax1.set_xlabel('L_max',fontsize=12)
    ax1.set_ylabel('obj1',fontsize=12)

    # 绘制第二个子图
    ax2.plot(x, y2,'-o', label='Subplot 2')
    ax2.set_xlabel('L_max',fontsize=12)
    ax2.set_ylabel('obj2',fontsize=12)

    # 调整子图之间的间距
    plt.tight_layout()
    # 显示图表
    plt.show()

import os

def calculate_and_save_average(input_folder):
    for filename in os.listdir(input_folder):
        # 拼接文件的完整路径
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否为普通文件
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                output_file_path = os.path.splitext(file_path)[0] + '_average.txt'
                with open(output_file_path, 'w') as output:
                    for line in lines:
                        # 将每行的文本分割成两个数，并计算平均值
                        values = line.strip().split()
                        if len(values) == 2:
                            value1 = float(values[0])
                            value2 = float(values[1])
                            average = (value1 + value2) / 2.0
                            # 将平均值写入输出文件
                            output.write(f'{average}\n')
                        else:
                            print(f"跳过无效行: {line}")
        else:
            print(f"跳过非文件: {filename}")

    print("平均值已保存到对应的文件中")


import matplotlib.pyplot as plt
import numpy as np


def plot_obj_FLround(array1, array2, array3):

    # 创建x轴，使用一个数组的长度作为x轴
    x_axis = np.arange(len(array1))

    # 创建一个新图形窗口
    plt.figure(figsize=(6.5, 4.5))

    array1_cumsum = np.cumsum(array1)
    array2_cumsum = np.cumsum(array2)
    array3_cumsum = np.cumsum(array3)

    # 绘制三个数组的曲线
    plt.plot(x_axis, array1_cumsum, '-', label='VQFL')
    plt.plot(x_axis, array2_cumsum, '-', label='RB')
    plt.plot(x_axis, array3_cumsum, '-', label='RQ')

    # 添加标签和图例
    plt.xlabel('FL iteration number',fontsize=12)
    plt.ylabel('Latency (s)',fontsize=12)
    plt.legend(fontsize=12)

    # 显示图表
    plt.show()

def multi_reward_monitor_inonefigure():
    path = "./paper2/reward_nohup/monitor"
    results, monitor_files = load_results_sorted(path) # ./paper2/reward/monitor log_reward_weight paper2/reward_nohup/monitor
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, df in enumerate(results):
        x, y = ts2xy(df, 'timesteps')
        y = moving_average(y, window=40)
        x = x[len(x) - len(y):]/50
        ax.plot(x, y, label=monitor_files[i].split('/')[-1].split(".")[0])  # 使用文件名作为标签
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.legend(loc='lower left', bbox_to_anchor=(0.15,0.02), shadow=True, ncol=3, fontsize=12)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("./paper2/fig/expr2", dpi=600)
    plt.show()


# 画那些图在这里决定
if __name__ == "__main__":
    # plot_reward()  # test里一个reward
    # plot_time(log_time)  # 多目标的训练时间
    # plot_pf(log_pf, file_pf)  # 多目标的pareto面
    # plot_pf_original(log_pf, file_pf)  # 多目标的pareto面
    # multi_reward(reward_path)  # 多目标的reward收敛图
    multi_reward_monitor() # 多目标的reward封装代码收敛图

    # plot_reward_diff_setting()


    # plot_obj1_obj2_reward()
    # plot_obj1_obj2_dot()

    # 调用函数并传递文件夹路径和输出文件名
    # calculate_and_save_average('paper/reward_method')
    # 示例用法
    # array1 = [0.1670420492356436, 0.16609383170832687, 0.16515643995151982, 0.16422967343773115, 0.16331333526566022, 0.16240723205377772, 0.16151117383612823, 0.16062497396023567, 0.15974844898699225, 0.15888141859241361, 0.15802370547114225, 0.15717513524158258, 0.15633553635254993, 0.15550473999131556, 0.15468257999292911, 0.1538688927506984, 0.15306351712770466, 0.15226629436922928, 0.15147706801596553, 0.15069568381788523, 0.14992198964862746, 0.14915583542027033, 0.1483970729983442, 0.1476455561169368, 0.14690114029373558, 0.14616368274484448, 0.1323464338393641, 0.13181928177267044, 0.1312961806469394, 0.13077705452067812, 0.13026182788176974, 0.12975042560502498, 0.12924277290860642, 0.12873879530920385, 0.12823841857582835, 0.12774156868208325, 0.1272481717567571, 0.12675815403257196, 0.12627144179290478, 0.12578796131628392, 0.1253076388184454, 0.12483040039171253, 0.12435617194144194, 0.1238848791192526, 0.12341644725272846, 0.12295080127125338, 0.12248786562760317, 0.12202756421488078, 0.12156982027833704, 0.12111455632157107, 0.12066169400654891, 0.12021115404681768, 0.11976285609322196, 0.11931671861135046, 0.11887265874985048, 0.11843059219864682, 0.1179904330359845, 0.11755209356308341, 0.11711548412504139, 0.11668051291644939, 0.11624708576998449, 0.11581510592601807, 0.11538447378101474, 0.11495508661219495, 0.11452683827558342, 0.11409961887416137, 0.11367331439236826, 0.11324780629265224, 0.11282297106912845, 0.11239867975265908, 0.11197479736079521, 0.11155118228499665, 0.11112768560634374, 0.11070415032954217, 0.11785051315419885, 0.12474064761573889, 0.12401676074212126, 0.12329233669243253, 0.12256691834989188, 0.12184001458566307, 0.12111109567064911, 0.12037958798058235, 0.11964486787693235, 0.1189062546319069, 0.11816300225569691, 0.11741429008458962, 0.11665921201210616, 0.11589676431562615, 0.1151258321931784, 0.11434517546559295, 0.11355341458528667, 0.11274901945345786, 0.11193030624142578, 0.11109545278908683, 0.11024255404511026, 0.10936976150264895, 0.1084755980242717, 0.10755964150988347, 0.10662399260810142, 0.10567641424548595]
    # array2 = [0.2540840984712872, 0.25120382880270736, 0.2483810693823516, 0.24561400424651592, 0.2429008805105623, 0.24024000542250595, 0.23762974355423389, 0.2350685141214695, 0.23255478842408, 0.23008708739875922, 0.22766397927651505, 0.22528407733774347, 0.2229460377579865, 0.22064855753774937, 0.21839037250999177, 0.21617025541911028, 0.2139870140653959, 0.2118394895090771, 0.20972655432814963, 0.20764711092424104, 0.20560008987076606, 0.2035844482975878, 0.20159916830631164, 0.19964325541019673, 0.1977157369924668, 0.19581566077653456, 0.19394209330130927, 0.19209411839432877, 0.19027083563493138, 0.18847135879904112, 0.18669481427637327, 0.18494033944994048, 0.1832070810266388, 0.18149419330637778, 0.17980083637565308, 0.17812617420959426, 0.1764693726642988, 0.17482959733861037, 0.1732060112813319, 0.17159777251606828, 0.1700040313513424, 0.16842392743815265, 0.1668565865305383, 0.1653011168967406, 0.1637566053188787, 0.16222211260731625, 0.16069666854160491, 0.15917926613247213, 0.1576688550780731, 0.15616433426181842, 0.15466454310755934, 0.15316825156972286, 0.15167414849014085, 0.1501808279991268, 0.14868677357605678, 0.14719033931667863, 0.14568972788757137, 0.1441829645991236, 0.14266786703315656, 0.14114200979585828, 0.13960268438914233, 0.13804685523332133, 0.1364711152247473, 0.13487164936030768, 0.13324422616705925, 0.1315842612941674, 0.12988705276353502, 0.12814841409404626, 0.12636623090349366, 0.12454418875896972, 0.12270064748163292, 0.12088934814411562, 0.11924301855907635, 0.11803357693687196, 0.11763278058397411, 0.11820401953567217, 0.11950309912663612, 0.12116774845826397, 0.12296305030306173, 0.12477876347940714, 0.1265706523084718, 0.1283237705279215, 0.1300354264288421, 0.13170783238282793, 0.1333449697157376, 0.134951257061046, 0.1365309992387936, 0.13808817851883326, 0.1396263963239344, 0.1411488794477131, 0.1426585117982877, 0.1441578739910454, 0.14564928297856225, 0.1471348285164738, 0.14861640541040086, 0.1500957414618418, 0.1515744214415741, 0.15305390756223633, 0.15453555694104146, 0.1560206365092437]
    # array3 = [0.3, 0.11414332736556501, 0.09694552407965845, 0.21465293932392282, 0.09659239454527649, 0.112976424778852, 0.2763264114976781, 0.11212110827531283, 0.11281435568818265, 0.20650103727294072, 0.11120028712947051, 0.2037610838045142, 0.09525420044150743, 0.14062250515028668, 0.14020792378398034, 0.13945155649120025, 0.09471331418586293, 0.09461031626599638, 0.09452090875527945, 0.13764790348486156, 0.18784520893712764, 0.1363148463816552, 0.09393894713532455, 0.12859440273251005, 0.1348371802603089, 0.11060594936978987, 0.10692168010250412, 0.29365076675219476, 0.09306180995145916, 0.1319596393366168, 0.09283687576403854, 0.1309988813976778, 0.1304671474267725, 0.22983590357475353, 0.12906082549891773, 0.1042765738753864, 0.1281385253469849, 0.09191416659089544, 0.09216349523278422, 0.17386457061059346, 0.12626899489868348, 0.12455312273899856, 0.1026640468711483, 0.10246161000046897, 0.09113603484016375, 0.12417893708611182, 0.13579873646237456, 0.16639145444657027, 0.12256878142663873, 0.11158101800882136, 0.09042465691069794, 0.12135810753453877, 0.09022510624216887, 0.12066999073904183, 0.1000655001820701, 0.12305619283113174, 0.09964434860470942, 0.15781277703898333, 0.08957988553578938, 0.19395989666952815, 0.19179837174151196, 0.08914980951400407, 0.15244518619659542, 0.11566387461674188, 0.09761468350468913, 0.0887225793653461, 0.08864504717696364, 0.1826039221001624, 0.18055690402726934, 0.18297365441011781, 0.12823029455537624, 0.1116737691047075, 0.08781442726940862, 0.3, 0.08744221785481193, 0.09471343436016173, 0.08727132578743599, 0.13742713757729924, 0.10817299089035745, 0.10774952549260569, 0.09365661895122143, 0.10693581799034527, 0.0866262102289396, 0.09307290415219469, 0.13152115970482428, 0.19045547005421792, 0.08610101690755977, 0.15206247891271937, 0.08583814529281258, 0.09147930188597989, 0.0856374233024115, 0.10213806763895898, 0.09082561901404423, 0.1307704644953322, 0.10055688474736983, 0.14010691243052117, 0.09929817906336358, 0.089391125931404, 0.08911706248988939, 0.10932742292554636]
    # array1 = [0.24184259979230316, 0.2851254359674919, 0.11932552275753477, 0.11522147464443488, 0.11379123076380787, 0.11286859712080305, 0.11262028567610165, 0.11237347239674658, 0.11212824105386424, 0.1118846647180286, 0.11164265083609477, 0.11140219916773286, 0.1111633378913328, 0.11092622556047498, 0.11069079212765165, 0.11045710824493107, 0.11022528296081188, 0.10999548757642821, 0.10976799990117596, 0.10954329257676326, 0.1093222272393945, 0.10910653386660628, 0.10890024420057412, 0.10871533394385963, 0.10860981915621193, 0.11363797507289555, 0.1637270721217401, 0.10761983577500216, 0.16206762352854148, 0.10710931883680974, 0.16043483010803125, 0.10701272205181533, 0.15882532240016345, 0.157875369207036, 0.15693531484370005, 0.15600493108445612, 0.15508399261102984, 0.1541722768342696, 0.15326956371409436, 0.15237563557715839, 0.1514902769316629, 0.1506132742787002, 0.14974441591946464, 0.14888349175761073, 0.14803029309597227, 0.14718461242678493, 0.1463462432144727, 0.14551497966996388, 0.14469061651539725, 0.14387294873795753, 0.14306177133144182, 0.14225687902400208, 0.14145806599032856, 0.14066512554633542, 0.13987784982417342, 0.13909602942512556, 0.13831945304763113, 0.13754790708732562, 0.13678117520557082, 0.13601903786247177, 0.1352612718098214, 0.13450764953876954, 0.133757938676261, 0.1330119013234087, 0.1322692933279401, 0.13152986348164816, 0.13079335263235803, 0.13005949269824618, 0.12932800557036983, 0.12859860188692152, 0.12787097965994193, 0.1271448227319205, 0.12641979903577988, 0.12569555862705992, 0.12497173145154229, 0.12424792480493328, 0.12352372043336848, 0.12279867121424871, 0.12207229734610622, 0.12134408196376528, 0.12061346608111552, 0.11987984274884572, 0.11914255029971804, 0.11840086454201018, 0.11765398975785163, 0.1169010483774545, 0.11614106925201101, 0.11537297457403582, 0.11459556576432012, 0.11380750919468252, 0.11300732370985685, 0.11219337407682217, 0.11136387879660095, 0.1105169493909878, 0.10965069607778853, 0.10876347204141261, 0.10785440827828627, 0.10692456409827264, 0.1059793915275628, 0.1050339643206453]
    # array2 = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.29830942126160537, 0.2926838954570318, 0.2871902606298712, 0.28182069848037583, 0.27656763993704514, 0.2714237265859513, 0.2663817713159018, 0.2614347174916966, 0.25657559584859196, 0.25179747814264486, 0.247093426382413, 0.242456436192463, 0.2378793724989396, 0.2333548952579994, 0.22887537234141442, 0.22443277592308364, 0.22001855776313178, 0.21562349770141098, 0.21123751863174844, 0.206849460788825, 0.20244680985409236, 0.1980153811440888, 0.19353898639770675, 0.18899917835917945, 0.184375359186715, 0.17964607861303833, 0.17479393562547746, 0.16982138634457197, 0.16480016109132622, 0.16002043582946024, 0.1563450585223633, 0.15532927760139592, 0.1575562673266141, 0.16169742253147829, 0.16647826217526956, 0.1713555472831306, 0.1761591600316384, 0.18085097109763895, 0.18543533955496877, 0.18992897438940506, 0.19435087656995395, 0.1987190984728758, 0.20304987058093232, 0.2073575469504798, 0.21165481244529938, 0.21595294938196913, 0.22026209275016922, 0.2245914527786016, 0.22894950204945302, 0.2333441306873913, 0.2377827747820705, 0.24227252309716552, 0.246820206453211, 0.25143247340626723, 0.25611585514577256, 0.26087682194985956, 0.26572183306590225, 0.27065738151348967, 0.2756900350184278, 0.28082647406307054, 0.28607352786628976, 0.2914382089744923, 0.2969277470445772, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    # array3 = [0.14963363938851487, 0.14895391506191064, 0.2848423454082493, 0.11649494161559468, 0.14659030469914391, 0.14583428574300056, 0.25500740815271244, 0.11208391118863006, 0.2071706694926015, 0.1113502720159766, 0.14564599740028927, 0.14208726349185466, 0.14101483275474413, 0.23094411171966472, 0.10985619507712865, 0.13895891123641257, 0.11263721178176223, 0.13789648591693812, 0.13731792735571305, 0.10837526928506523, 0.1363388504308572, 0.13574541225908493, 0.1076010818013925, 0.13475338813256657, 0.10710359721501216, 0.13381717227275935, 0.10662512350136165, 0.13280854321140215, 0.22018251528697125, 0.22511287469580826, 0.10528253100343503, 0.3, 0.12894809436682533, 0.20105717610867996, 0.2705746971166978, 0.25594187556527986, 0.26248208728378, 0.12462056112897485, 0.12418100867256526, 0.12368972147711962, 0.10161886707961465, 0.3, 0.1008498513985132, 0.2452872205517183, 0.26607739489370025, 0.09970630542396647, 0.15808287701733775, 0.1953603896981486, 0.11773957562912019, 0.11730112592160022, 0.1537089602891727, 0.1163004860788078, 0.2157043691950507, 0.11505961205509938, 0.14924113893410385, 0.11407395462189818, 0.12297395424466881, 0.11319614207773257, 0.14550477430194275, 0.09610856100782364, 0.11185101481391932, 0.11146167680145683, 0.11101723908051654, 0.15689769744817494, 0.1699067272051241, 0.1093845081029295, 0.1088860398218284, 0.0942350501588848, 0.10808757992206336, 0.09383139946714777, 0.13456545945340248, 0.14449340852233172, 0.1322962493675667, 0.10565007890587234, 0.10515967864232628, 0.10470787964681456, 0.10426170522265771, 0.0918996707981197, 0.09169746782429772, 0.14892964971219258, 0.09115221840118225, 0.09092902534575399, 0.14910180428217812, 0.10068556876732536, 0.09008646109250798, 0.13906874136087327, 0.098932844638878, 0.09838420516725441, 0.08913506805608738, 0.0972864954151138, 0.1467995069427664, 0.17196102997512253, 0.08763557659343862, 0.1251540202901994, 0.08767440461889929, 0.08781833997259397, 0.09609912433910599, 0.08832521796220776, 0.09717376047791729, 0.09779280121315431]
    # plot_obj_FLround(array1, array2, array3)

    #实验画图
    # multi_reward_monitor_inonefigure()
