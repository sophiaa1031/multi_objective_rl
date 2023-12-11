from stable_baselines3 import A2C,TD3,PPO
import matplotlib.pyplot as plt
import QFL_Env4
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import torch
import time

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
log_time = 'training_time(s)'
log_pf = 'pareto_front/'
try:
    os.mkdir(log_pf)
except:
    1
f = open(log_time, 'w')
delete_contents_of_directory(log_dir)
delete_contents_of_directory(log_pf)
for i in np.arange(0,1.2,0.2):
    print('Now training with weight {:.1f},{:.1f}'.format(i,(1-i)))
    env = QFL_Env4.QFLEnv(weight_lambda=i,obj_file=log_pf+"/weight_"+"{:.1f}".format(i)+'.txt')
    env = Monitor(env, log_dir+'lambda_'+"{:.1f}".format(i)+'_')
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[512, 256, 64], vf=[512, 256, 64]))
    # Create RL model
    #target_aciton_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
    if i == 0:
        model = PPO("MlpPolicy", env, verbose=500, learning_rate=1e-3, policy_kwargs=policy_kwargs, batch_size=512) #,batch_size=500
    else:
        model = PPO.load("ppo_saved_weight", print_system_info=True, env=env, learning_rate=1e-3)
    #model = PPO.load("ppo_saved", print_system_info=True,env=env, learning_rate=1e-4)

    #model = DDPG('MlpPolicy', env, verbose=0, learning_rate=0.001)
    # Train the agent
    start_time = time.time()
    model.learn(total_timesteps=100000)
    end_time = time.time()
    training_time = end_time - start_time
    save_result(training_time,log_time)
    model.save('ppo_saved_weight')
