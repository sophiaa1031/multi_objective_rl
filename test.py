import numpy
import numpy as np
import IoT_Devices

# class IoTDevice:
#     def __init__(self, data_size):
#         self.data_size = data_size
#
# 创建三组每组10个设备的3*10的数组

# devices = IoT_Devices.devices
# #打印每个设备的数据大小
# for group in devices:
#     for device in group:
#         print(device.data_size)
#     print('---')
# Q_ds = []
# q_ds = 0  # 用于计算该组设备实验敏感数据之和
# for j in range(3):
#     q_ds += IoT_Devices.devices[0][j].data_size
#     print(q_ds)
# Q_ds.append(q_ds)
# print(Q_ds)




#
# from stable_baselines3 import TD3
# from stable_baselines3.common.vec_env import DummyVecEnv
# import gym
# import numpy as np
# import MDRA_Env2
#
# # 加载模型
# loaded_model = TD3.load("best_model")
# env = MDRA_Env2.UAVMDRAEnv()
# # 从环境中获取一个初始观测值
# obs = env.reset()
#
# # 获取收敛后的动作
# action, _ = loaded_model.predict(obs, deterministic=True)
# bandwidth = action[0:10]
# computing = action[10:20]
# storage = action[20:27]
# beta = np.zeros(10)
# delta = np.zeros(10)
# epsilon = np.zeros(10)
# for i in range(10):
#     beta[i] = bandwidth[i] / sum(bandwidth)  # 带宽资源,比例值之和限制<=1
#     delta[i] = computing[i] / sum(computing)  # 计算资源,比例值之和限制<=1
# for i in range(7):
#     epsilon[i + 3] = storage[i] / sum(storage)
# print("bandwith: ", beta)
# print("computing: ", delta)
# print("storage: ", epsilon)
propose_social_welfare = [70.47695850798776, 67.98965863756891, 81.13297503447426, 188.02093773907126,
                          188.02093773907126, 191.0142878194445, 202.781499740243, 284.756584258108]
# 生成新数组，第n个元素是前n个元素之和
new_array = [sum(propose_social_welfare[:i+1]) for i in range(len(propose_social_welfare))]

print(new_array)