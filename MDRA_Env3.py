import gym
import math
import numpy as np
import uav as uu
import IoT_Devices
import para
from gym import spaces

"""
此环境用于寻找sum-rate最大的带宽分配策略
该环境中，共有8组设备，每组有10个设备
"""


class UAVMDRAEnv(gym.Env):
    def __init__(self):
        super(UAVMDRAEnv, self).__init__()
        self.uav = uu.UAV()
        self.IoTd = IoT_Devices.devices
        self.observation_space = spaces.Box(low=0, high=201, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.01, high=1, shape=(10,), dtype=np.float32)
        # 共有八组设备
        self.episode_length = 8
        self.current_step = 0

    # 训练1个episode之后reset
    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        beta = np.zeros(10)
        c = np.zeros(10)  # 每个传感器的data rate
        t = np.zeros(10)  # 每个传感器的传输时间
        one_device_reward = np.zeros(10)
        """
        首先计算限制在[0,1]的给每个设备分配的资源比例值
        """
        for i in range(10):
            beta[i] = action[i] / sum(action)  # 带宽资源,比例值之和限制<=1
        #print("动作：",action)
        print("bata:", beta)
        for i in range(10):  # 每组传感器有10个，故有10组动作
            # 数据传输速率 MB/s
            c[i] = beta[i] * self.uav.B * math.log(1 + (para.IoTD_p * para.g / para.sigma)) / 8388608  #* 0.125e-6
            # 传输时间
            t[i] = self.IoTd[self.current_step][i].data_size / c[i]
            # 一个设备的reward
            one_device_reward[i] = c[i] / t[i]
            print("第",i,"个设备传输时间：",c[i])
        reward = sum(one_device_reward) / 1000
        print(reward)
        obs = self._get_obs()
        self.current_step += 1
        done = False
        if self.current_step >= self.episode_length:
            done = True
        return obs, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_obs(self):
        """
        观测空间为下一个step所有设备的数据以及UAV剩余空间
        :return:
        """
        observation_spaces = np.zeros(10, dtype=np.float32)
        for i in range(10):
            observation_spaces[i] = float(self.IoTd[self.current_step][i].data_size) / 100
        return observation_spaces
