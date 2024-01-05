from stable_baselines3 import A2C,TD3,PPO
import matplotlib.pyplot as plt
import QFL_Env_clientnumber_latency
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import torch
import time
from plot_def import plot_time, plot_pf, multi_reward,multi_reward_monitor,plot_pf_original

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
try:
    os.mkdir(log_pf)
except:
    1
f = open(log_time, 'w')
delete_contents_of_directory(log_dir)
delete_contents_of_directory(log_pf)
for i in np.arange(0,1.2,0.2):
    print('Now training with weight {:.2f},{:.2f}'.format(i,(1-i)))
    env = QFL_Env_clientnumber_latency.QFLEnv(weight_lambda=i,obj_file=log_pf+"/weight_"+"{:.2f}".format(i)+'.txt')
    monitor_path = log_dir+'lambda_'+"{:.2f}".format(i)+'_'
    env = Monitor(env, monitor_path)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[256, 64], vf=[256, 64]))
    # Create RL model
    #target_aciton_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    if i >= 0:
        model = PPO("MlpPolicy", env, verbose=500, learning_rate=1e-3, policy_kwargs=policy_kwargs, batch_size=512, ) #,batch_size=500
    else:
        model = PPO.load("ppo_saved_weight", print_system_info=True, env=env, learning_rate=1e-3)
    #model = PPO.load("ppo_saved", print_system_info=True,env=env, learning_rate=1e-4)

    #model = DDPG('MlpPolicy', env, verbose=0, learning_rate=0.001)
    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=1000000)
    end_time = time.time()
    training_time = end_time - start_time
    save_result(training_time,log_time)
    model.save('ppo_saved_weight')

# plot_time(log_time)  # 多目标的训练时间
plot_pf_original(log_pf, file_pf)  # 多目标的pareto面
multi_reward()
multi_reward_monitor()