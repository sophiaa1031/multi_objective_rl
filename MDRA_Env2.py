import gym
import math
import numpy as np
import uav as uu
import IoT_Devices
import para


from gym import spaces

"""
该环境中，共有8组设备，每组有10个设备，其中前3个设备产生时延敏感数据，后7个设备产生时延容忍数据
"""


class UAVMDRAEnv(gym.Env):
    def __init__(self):
        super(UAVMDRAEnv, self).__init__()
        self.uav = uu.UAV()
        self.IoTd = IoT_Devices.devices
        self.observation_space = spaces.Box(low=0, high=2049, shape=(11,), dtype=np.float32)
        '''
        动作空间为产生27个0-1的比例值，分为三组，前两组分别为10个带宽，计算比例值，后7个为存储比例值，
        由于每组前三个设备为实验敏感, 因此不需要分配比例值
        '''
        self.action_space = spaces.Box(low=0.01, high=1, shape=(27,), dtype=np.float32)
        # 初始化无人机状态
        self.remaining_storage = self.uav.S  # MB
        # 共有八组设备
        self.episode_length = 8
        self.current_step = 0
        # 用于存储每组时延敏感数据总大小
        self.Q_ds_step = []
        # 用于存储每组时延容忍数据总大小
        self.Q_dl_step = []

    # 训练1个episode之后reset
    def reset(self):
        self.current_step = 0
        self.Q_ds_step = []
        self.Q_dl_step = []
        self.remaining_storage = self.uav.S
        return self._get_obs()

    def step(self, action):
        # beta delta epsilon 分别为分配给每个设备的资源比例值
        beta = np.zeros(10)
        delta = np.zeros(10)
        epsilon = np.zeros(10)

        c = np.zeros(10)  # 每个传感器的data rate
        t = np.zeros(10)  # 每个传感器的传输时间
        E_tr = np.zeros(10)  # 每个传感器的传输能耗
        E_c = np.zeros(10)  # 每个传感器的计算能耗
        q_dl = np.zeros(10)  # 该组设备每个设备时延容忍数据上传大小
        one_device_reward = np.zeros(10)
        q_ds = 0  # 用于计算该组设备实验敏感数据之和
        '''
        首先计算这一组设备时延敏感数据总和 
        '''
        for j in range(3):
            q_ds += self.IoTd[self.current_step][j].data_size
        self.Q_ds_step.append(q_ds)  # 这里得到了当前组的  ”时延敏感数据总和“  并存在数组里方便后面计算剩余存储资源
        # 将产生的27个动作分组
        bandwidth = action[0:10]
        computing = action[10:20]
        storage = action[20:27]
        """
        首先计算限制在[0,1]的给每个设备分配的资源比例值
        """
        for i in range(10):
            beta[i] = bandwidth[i] / sum(bandwidth)  # 带宽资源,比例值之和限制<=1
            delta[i] = computing[i] / sum(computing)  # 计算资源,比例值之和限制<=1
        for i in range(7):
            epsilon[i + 3] = storage[i] / sum(storage)
        # print("动作：",action)
        print("bata:", beta)
        print("epsilon:", epsilon)
        print("delta:", delta)
        # 判断当前时延敏感数据是否能全部接收
        if self.remaining_storage < self.Q_ds_step[self.current_step]:
            reward = -1
        else:
            if self.current_step == 0:
                self.remaining_storage = self.uav.S - self.Q_ds_step[0]
                for i in range(10):  # 每组传感器有10个，故有10组动作
                    if i <= 2:  # 前三个设备为时延敏感数据
                        # 数据传输速率 MB/s
                        c[i] = beta[i] * self.uav.B * math.log(1 + (para.IoTD_p * para.g / para.sigma)) / 8388608  #* 0.125e-6
                        # 传输时间
                        t[i] = self.IoTd[self.current_step][i].data_size / c[i]
                        # 传输能耗 W*s 单位为J 焦耳
                        E_tr[i] = para.IoTD_p * t[i]
                        # 计算能耗 数据单位为bit 所以乘以8388608
                        E_c[i] = para.Omu * para.omega1 * self.IoTd[self.current_step][i].data_size * 8388608 * ((delta[i] * self.uav.computing) ** 2)
                        # 一个设备的reward
                        one_device_reward[i] = c[i] / (E_c[i] + E_tr[i]) + para.alpha1 * math.log(1 + self.IoTd[self.current_step][i].data_size) + para.alpha2 * math.log(1 + ((delta[i] * self.uav.computing) / (8388608 * para.omega1 * self.IoTd[self.current_step][i].data_size)))
                    else:
                        # 数据传输速率 MB/s
                        c[i] = beta[i] * self.uav.B * math.log(1 + (para.IoTD_p * para.g / para.sigma)) * 1.25e-7
                        # 单个设备时延容忍数据上传大小
                        if epsilon[i] * self.remaining_storage < self.IoTd[self.current_step][i].data_size:
                            q_dl[i] = epsilon[i] * self.remaining_storage
                        else:
                            q_dl[i] = self.IoTd[self.current_step][i].data_size
                        # 上传时间
                        t[i] = q_dl[i] / c[i]
                        # 传输能耗
                        E_tr[i] = para.IoTD_p * t[i]
                        # 计算能耗
                        E_c[i] = para.Omu * para.omega2 * q_dl[i] * 8388608 * ((delta[i] * self.uav.computing) ** 2)
                        one_device_reward[i] = (c[i] / (E_c[i] + E_tr[i])) + para.alpha1 * math.log(1 + (q_dl[i] / self.IoTd[self.current_step][i].data_size)) + para.alpha2 * math.log(1 + ((delta[i] * self.uav.computing) / (para.omega2 * q_dl[i] * 8388608)))
                self.Q_dl_step.append(sum(q_dl))
                reward = sum(one_device_reward) / para.rescale
            elif self.remaining_storage - self.Q_ds_step[self.current_step] - sum(self.Q_dl_step) < 0:
                reward = -1
            else:
                self.remaining_storage = self.uav.S - self.Q_ds_step[self.current_step] - sum(self.Q_dl_step)
                for i in range(10):  # 每组传感器有10个，故有10组动作
                    if i <= 2:  # 前三个设备为时延敏感数据
                        # 数据传输速率 MB/s
                        c[i] = beta[i] * self.uav.B * math.log(1 + (para.IoTD_p * para.g / para.sigma)) / 8388608  #* 0.125e-6
                        # 传输时间
                        t[i] = self.IoTd[self.current_step][i].data_size / c[i]
                        # 传输能耗 W*s 单位为J 焦耳
                        E_tr[i] = para.IoTD_p * t[i]
                        # 计算能耗 数据单位为bit 所以乘以8388608
                        E_c[i] = para.Omu * para.omega1 * self.IoTd[self.current_step][i].data_size * 8388608 * ((delta[i] * self.uav.computing) ** 2)
                        # 一个设备的reward
                        one_device_reward[i] = c[i] / (E_c[i] + E_tr[i]) + para.alpha1 * math.log(1 + self.IoTd[self.current_step][i].data_size) + para.alpha2 * math.log(1 + ((delta[i] * self.uav.computing) / (8388608 * para.omega1 * self.IoTd[self.current_step][i].data_size)))
                    else:
                        # 数据传输速率 MB/s
                        c[i] = beta[i] * self.uav.B * math.log(1 + (para.IoTD_p * para.g / para.sigma)) * 1.25e-7
                        # 单个设备时延容忍数据上传大小
                        if epsilon[i] * self.remaining_storage < self.IoTd[self.current_step][i].data_size:
                            q_dl[i] = epsilon[i] * self.remaining_storage
                        else:
                            q_dl[i] = self.IoTd[self.current_step][i].data_size
                        # 上传时间
                        t[i] = q_dl[i] / c[i]
                        # 传输能耗
                        E_tr[i] = para.IoTD_p * t[i]
                        # 计算能耗
                        E_c[i] = para.Omu * para.omega2 * q_dl[i] * 8388608 * ((delta[i] * self.uav.computing) ** 2)
                        one_device_reward[i] = (c[i] / (E_c[i] + E_tr[i])) + para.alpha1 * math.log(1 + (q_dl[i] / self.IoTd[self.current_step][i].data_size)) + para.alpha2 * math.log(1 + ((delta[i] * self.uav.computing) / (para.omega2 * q_dl[i] * 8388608)))
                self.Q_dl_step.append(sum(q_dl))
                reward = sum(one_device_reward) / para.rescale
                print("social_welfare:", reward)
        #print("reward:",reward)
        print("storage:", self.remaining_storage)
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
        observation_spaces = np.zeros(11, dtype=np.float32)
        for i in range(10):
            observation_spaces[i] = float(self.IoTd[self.current_step][i].data_size) / 100
        observation_spaces[10] = float(self.remaining_storage) / 1000
        return observation_spaces
