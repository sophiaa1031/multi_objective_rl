import gymnasium
import math
import numpy as np
from gymnasium import spaces
import random

"""
该环境中，共有8组设备，每组有10个设备，其中前3个设备产生时延敏感数据，后7个设备产生时延容忍数据
"""


class QFLEnv(gymnasium.Env):
    def __init__(self,cars=10,done_step = 10):
        super(QFLEnv, self).__init__()
        self.cars = 10
        self.current_step = 0
        self.done_step = 10
        # static
        self.pwr = [0.1] * self.cars
        self.fre = [2] * self.cars
        self.vel = [35] * self.cars
        self.rho = [1/self.cars] * self.cars
        self.obj1 = []
        self.obj2 = []
        
        #dynamic
        self.travel_dis = [ random.random() *500 for i in range(self.cars)]
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i]-500,2)+10**2) for i in range(self.cars)]
        self.latency_itr = [0] * self.cars
        
        
        #action1_space = spaces.Box(low=np.zeros(self.cars), high=np.ones(self.cars)*32, dtype=np.float32)

        # 动作2：0到1之间的实数
        #action2_space = spaces.Box(low=np.zeros(self.cars), high=np.ones(self.cars), dtype=np.float32)

        # 动作3：0到10之间的实数
        #action3_space = spaces.Box(low=np.zeros(self.cars), high=np.full(self.cars, 10), dtype=np.float32)

        # 将三个动作组合成一个元组空间
        #self.action_space = spaces.Tuple((action1_space, action2_space, action3_space))
        
        action_low = np.concatenate([np.zeros(self.cars), np.zeros(self.cars),np.ones(self.cars)*0.1])
        action_high = np.concatenate([np.ones(self.cars), np.ones(self.cars),np.ones(self.cars)])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        #self.action_space = action1*cars + action2*cars + action3*cars
        #需要强调的是，action1本质上是1-32的整数，action3是0.01-0.1的实数，我们这里都处理成0-1之间
        
        
        static_low = np.concatenate([np.zeros(self.cars),
                                     np.zeros(self.cars),
                                     np.zeros(1)])
        static_high = np.concatenate([np.full(self.cars, np.inf), 
                                      np.full(self.cars, np.inf),
                                      np.full(1, np.inf)])

        # 动态变量的范围
        dynamic_low = np.concatenate([np.zeros(self.cars),
                                     np.zeros(self.cars),
                                     np.zeros(1)])
        dynamic_high = np.concatenate([np.full(self.cars, np.inf), 
                                      np.full(self.cars, np.inf),
                                      np.full(1, np.inf)])

        # 定义观察空间
        self.observation_space = spaces.Box(low=np.concatenate([static_low, dynamic_low]),
                                            high=np.concatenate([static_high, dynamic_high]),
                                            dtype=np.float32)

    # 训练1个episode之后reset
    def reset(self,seed=0):
        self.current_step = 0
        self.obj1 = []
        self.obj2 = []
        # static
        # self.pwr = [0.1] * self.cars
        self.fre = [2] * self.cars
        self.vel = [35] * self.cars
        self.rho = [1/self.cars]
        
        #dynamic
        self.travel_dis = [ random.random() *500 for i in range(self.cars)]
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i]-500,2)+10**2) for i in range(self.cars)]
        self.latency_itr = [0]
        
        initial_observation = self.fre + self.vel + self.rho + self.travel_dis + self.rsu_dis + self.latency_itr
        return np.array(initial_observation),{}

    def step(self, action): #
        #observation change
        # print ('start step')
        # print ('show action',action)
        action[0:self.cars] = np.round(action[0:self.cars]*31)+1
        action[2*self.cars:3*self.cars] = action[2*self.cars:3*self.cars]*0.1
        # print ('after action',action)
        t_comp = [0.05] * self.cars
        rsu_temp = [0] * self.cars
        rate_temp = [0] * self.cars
        for i in range(self.cars):
            rsu_temp[i] = self.travel_dis[i]+t_comp[i]*self.vel[i]
            self.rsu_dis[i] = np.sqrt(np.power(rsu_temp[i]-500,2)+10**2)
            #rate_temp = action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
            rate_temp[i] = action[1*self.cars + i]*10*np.log2(1+1e7 * action[2*self.cars + i] * np.power(self.rsu_dis[i], -2))
            if rate_temp[i] == 0:
                rate_temp[i]+=0.001
            
        print ('rate_temp',rate_temp)
        # print ("action[1*self.cars + i]",action[1*self.cars + i])
        # print ("log2(1+1e7*action3/(rsu_dis^2)",np.log2(1+1e7 * action[2*self.cars + i] * np.power(self.rsu_dis[i], -2)))
        # print ("1e7*action3/(rsu_dis^2)",1e7 * action[2*self.cars + i] * np.power(self.rsu_dis[i], -2))
        # print ("action3",action[2*self.cars + i:])
        self.latency_itr[0] = np.max([t_comp[i] + (action[1]+1)/32 / rate_temp[i] for i in range(self.cars)])
        for i in range(self.cars):
            self.travel_dis[i] += self.latency_itr[0]*self.vel[i]
            
        obj1_temp = np.sum([self.rho[0]/np.power(
            np.power(2, (action[0*self.cars + i]+1)) - 1,2) for i in range(self.cars)])
        obj1_temp2 = self.latency_itr[0]
        
        self.obj1.append(obj1_temp)
        self.obj2.append(obj1_temp2)
        obs = self.fre + self.vel + self.rho + self.travel_dis + self.rsu_dis + self.latency_itr
        
        
        
        #check_termination 
        self.current_step += 1
        done = False
        if self.current_step >= self.done_step:
            done = True
        reward = self.get_reward()
        # print ('reward, done',reward, done,obj1_temp,obj1_temp2)
        return obs, reward, done, False, {}
    
    
    def get_reward(self):
        if self.current_step >= self.done_step:
            obj1 = np.sum(self.obj1)
            obj2 = np.sum(self.obj2)
            print ('obj1',obj1)
            print ('obj2',obj2)
            return -obj1 - obj2
        else:
            return 0

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
