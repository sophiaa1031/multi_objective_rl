import gymnasium
import math
import numpy as np
from gymnasium import spaces
import random

"""
该环境中，共有8组设备，每组有10个设备，其中前3个设备产生时延敏感数据，后7个设备产生时延容忍数据
"""


class QFLEnv(gymnasium.Env):
    def __init__(self,cars=10,done_step = 10,debug = False):
        super(QFLEnv, self).__init__()
        self.cars = 10
        self.current_step = 0
        self.done_step = 10 # 联邦学习全局轮数
        # static
        # self.pwr = [0.1] * self.cars
        self.fre = [2] * self.cars
        self.vel = [30] * self.cars
        self.rho = [1/self.cars] * self.cars
        self.obj1 = []
        self.obj2 = []
        self.debug = debug
        
        #dynamic
        # 每个人起始位置不一样
        # self.travel_dis = [ random.random() *500 for i in range(self.cars)]
        # 每个人都在位置为0的地方开始
        self.travel_dis = [0] * self.cars

        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i]-500,2)+10**2) for i in range(self.cars)]
        self.latency_itr = [0] * self.cars
        

        # 动作1-prob：0.5到1之间的实数

        # 动作2-bd*10：0到1之间的实数

        # 动作3-pwr*10：0.1到1之间的实数(后期会处理成0.01到0.1)
        
        # 将三个动作组合成一个元组空间
        #self.action_space = spaces.Tuple((action1_space, action2_space, action3_space))
        
        action_low = np.concatenate([np.ones(1)*0.5, np.ones(self.cars)*0.01,np.ones(self.cars)*0.1])
        action_high = np.concatenate([np.ones(1)*0.999999, np.ones(self.cars)*0.5,np.ones(self.cars)])
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
        self.vel = [30] * self.cars
        self.rho = [1/self.cars]
        
        #dynamic
        # 每个人起始位置不一样
        # self.travel_dis = [ random.random() *500 for i in range(self.cars)]
        # 每个人都在位置为0的地方开始
        self.travel_dis = [0] * self.cars
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i]-500,2)+10**2) for i in range(self.cars)]
        self.latency_itr = [0]
        
        initial_observation = [i/2 for i in self.fre] + [i/30 for i in self.vel] + self.rho + [i/500 for i in self.travel_dis] + [i/500 for i in self.rsu_dis] + self.latency_itr
        return np.array(initial_observation),{}

    def step(self, action): #
        # if np.any(np.isnan(self.travel_dis)) or np.any(np.isnan(self.rsu_dis)) or np.any(np.isnan(self.latency_itr)):
        #     print('nan')
        # if np.any(np.isnan(action)):
        #     print('nan')
        #observation change
        # print ('start step')
        # print ('show action',action)
        prob = action[0]
        bd = action[1:self.cars+1]
        pwr = action[self.cars+1:]*0.1
        # print ('prob',prob)
        # print ('bd',bd)
        # print ('pwr',pwr)
        quant = [0 for i in range(self.cars)]
        for i in range(self.cars):
            # log2(1-pw*log(prob)*1e14/4) *32 *bd * (1000-travel_dis)/vel
            quant[i] = np.log2(1 - pwr[i]*np.log(prob)*1e7/np.power(self.rsu_dis[i],2))*32*bd[i]#*(1000-self.travel_dis[i])/self.vel[i]
            # print ('np.log2(1 - pwr[i]*np.log(prob)*1e12/4/np.power(self.rsu_dis[i],2))',np.log2(1 - pwr[i]*np.log(prob)*1e12/4/np.power(self.rsu_dis[i],2)))
            # print ('np.power(self.rsu_dis[i],2)',np.power(self.rsu_dis[i],2))
            #print ('(1000-self.travel_dis[i])/self.vel[i]',(1000-self.travel_dis[i])/self.vel[i])
            if quant[i]>32:
                quant[i] = 32
            if quant[i]<1:
                quant[i] = 1
        t_comp = 0.05
        rsu_temp = [0] * self.cars
        rate_temp = [0] * self.cars
        for i in range(self.cars):
            rsu_temp[i] = self.travel_dis[i]+t_comp*self.vel[i]
            self.rsu_dis[i] = np.sqrt(np.power(rsu_temp[i]-500,2)+100)
            #rate_temp = action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
            rate_temp[i] = bd[i]*10*np.log2(1 + 1e7 * pwr[i] * np.power(self.rsu_dis[i], -2))
            # if rate_temp[i] < 0.01:
            #     rate_temp[i]=0.01
        if self.debug:
            print ('\n')
            print ('step',self.current_step)
            print ('##########################')
            print ('self.travel_dis',self.travel_dis)
            print("self.rsu_dis[i]", self.rsu_dis)
            print("bd", bd)
            print("np.log2(1 + 1e7 * pwr[i] * np.power(self.rsu_dis[i], -2)",np.log2(1 + 1e7 * pwr[i] * np.power(self.rsu_dis[i], -2)))
            print ("rsu_temp",rsu_temp)


            print ('rate_temp',rate_temp)
            print ('pwr',pwr)
            print('quant',quant)
            print('\n')
        # print ("action[1*self.cars + i]",action[1*self.cars + i])
        # print ("log2(1+1e7*action3/(rsu_dis^2)",np.log2(1+1e7 * action[2*self.cars + i] * np.power(self.rsu_dis[i], -2)))
        # print ("1e7*action3/(rsu_dis^2)",1e7 * action[2*self.cars + i] * np.power(self.rsu_dis[i], -2))
        # print ("action3",action[2*self.cars + i:])
        self.latency_itr[0] = np.max([t_comp + quant[i] / (32 * max(rate_temp[i],0.01)) for i in range(self.cars)])
        # print ("latency_itr",[t_comp + quant[i] / (32 * max(rate_temp[i],0.01)) for i in range(self.cars)])
        # print("self.latency_itr[0]",self.latency_itr[0])
        for i in range(self.cars):
            self.travel_dis[i] += self.latency_itr[0]*self.vel[i]


        # if np.any(np.isnan(self.travel_dis)) or np.any(np.isnan(self.rsu_dis)) or np.any(np.isnan(self.latency_itr)):
        #     print('nan')
        # if np.any(np.isinf(self.travel_dis)) or np.any(np.isinf(self.rsu_dis)) or np.any(np.isinf(self.latency_itr)):
        #     print('inf')
        obj1_temp = np.sum([self.rho[0]/np.power(
            np.power(2, (quant[i])) - 1,2) for i in range(self.cars)])
        # if np.isnan(obj1_temp) or np.any(np.isinf(obj1_temp)) :
        #     print('obj1_temp')
        obj1_temp2 = self.latency_itr[0]
        
        self.obj1.append(obj1_temp)
        self.obj2.append(obj1_temp2)
        obs = [i/2 for i in self.fre] + [i/30 for i in self.vel] + self.rho + [i/500 for i in self.travel_dis] + [i/500 for i in self.rsu_dis] + self.latency_itr
        # if np.any(np.isinf(obs)) or np.any(np.isnan(obs)):
        #     print ('obs')
        
        
        
        #check_termination 
        self.current_step += 1
        done = False
        if self.current_step >= self.done_step:
            done = True

        reward = self.get_reward(prob,bd,pwr,quant,rate_temp)
        # print ('reward, done',reward, done,obj1_temp,obj1_temp2)
        print ('reward',reward)
        return obs, reward, done, False, {}
    
    
    def get_reward(self,prob,bd,pwr,quant,rate_temp):
        reward = 0
        if self.current_step >= self.done_step:
            obj1 = np.sum(self.obj1)
            obj2 = np.sum(self.obj2)
            print ('obj1: ',obj1,', obj2: ',obj2)
            reward -= obj1
            reward -= obj2
        #     return reward
        #不等式约束
        if np.sum(bd) > 1:
            reward -= np.sum(bd)
        else:
            reward -= (1-np.sum(bd))
            #rate_temp >0.01
            # for i in range(len(bd)):
            #     if bd[i] < 0.01:
            #         reward -= (0.01-bd[i])*10
            # for i in range(len(rate_temp)):
            #     if rate_temp[i] < 1:
            #         reward -= (1-rate_temp[i])
            # if any(x > 32 for x in quant):
            #     reward -= 10
            # if any(x < 1 for x in quant):
            #     reward -= 10000
        return reward

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
