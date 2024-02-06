from stable_baselines3 import A2C,TD3,PPO, DDPG
import matplotlib.pyplot as plt
import QFL_Env_clientnumber_latency
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import torch
import time
from plot_def import plot_time, plot_pf, multi_reward,multi_reward_monitor,plot_pf_original
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from options import args_parser

def save_result(para,csv_file_path):
    f = open(csv_file_path, 'a')
    f.write('{:.4f}'.format(para)+',')

def delete_contents_of_directory(directory_path):
    if os.path.exists(directory_path):
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)  # 删除文件

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    

log_dir = "log_reward_weight/"
# 创建环境
log_time = 'training_time(s).txt'
log_pf = 'pareto_front/'
file_pf = 'plot_pf.txt'

path1 = 'paper/reward_method/RQ_1e-4.txt'
path2 = 'paper/reward_lr/RandomB.txt'
path3 = 'paper/reward_nn/VQFL_nn=(256,64,64).txt'
path4 = 'paper/obj1_obj2_lmax/0.2.txt'
path5 = 'paper/obj1_obj2_qe/06_8.txt'
path6 = 'debug'
reward_path = path6

try:
    os.mkdir(log_pf)
except:
    1
f = open(log_time, 'w')
delete_contents_of_directory(log_dir)
delete_contents_of_directory(log_pf)

args = args_parser()
# debug
args.baseline = 'ppo'
print(args.baseline)
args.timesteps = 200000

for i in np.arange(0.35,0.42,0.5):  # i可以是weight，velocity，lmax
    # print('Now training with weight {:.2f},{:.2f}'.format(i,(1-i)))  # weight写死是0.9_0.1
    env = QFL_Env_clientnumber_latency.QFLEnv(args,weight_lambda=1,
                                              obj_file=log_pf+"/weight_"+"{:.2f}".format(i)+'.txt',lmax=0.35)
    monitor_path = log_dir+'lambda_'+"{:.2f}".format(i)+'_'
    env = Monitor(env, monitor_path)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[128,64], vf=[128,64]))
    if i >= 0:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=args.lr, policy_kwargs=policy_kwargs, batch_size=args.bs, gamma=0.99) #,batch_size=500
    else:
        model = PPO.load("ppo_saved", print_system_info=True,env=env, learning_rate=1e-4)

    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=args.timesteps)
    end_time = time.time()
    training_time = end_time - start_time
    save_result(training_time,log_time)
    model.save('ppo_saved_weight')

# 画图
# plot_time(log_time)  # 多目标的训练时间
# plot_pf_original(log_pf, file_pf)  # 多目标的pareto面
multi_reward(reward_path)
# multi_reward_monitor()