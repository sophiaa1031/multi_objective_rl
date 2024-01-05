import gymnasium
import math
import numpy as np
from gymnasium import spaces
import random
"""
该环境中，共有8组设备，每组有10个设备，其中前3个设备产生时延敏感数据，后7个设备产生时延容忍数据
"""


class QFLEnv(gymnasium.Env):
    def __init__(self, cars=10, done_step=4, weight_lambda=1, debug=False, obj_file=None):  # done_step需要设置成2的次方，这样才能保证total_timesteps是他的整数倍
        super(QFLEnv, self).__init__()
        self.cars = cars
        self.current_step = 0
        self.done_step = done_step  # 联邦学习全局轮数
        self.weight_lambda = weight_lambda
        self.obj_file=obj_file
        if self.obj_file:
            self.f = open(self.obj_file, 'w')
        # static 环境
        self.pwr = [0.1] * self.cars
        self.fre = [2] * self.cars
        self.vel = [30] * self.cars
        self.rho = [1 / self.cars] * self.cars
        self.obj1 = []
        self.obj2 = []
        self.debug = debug

        # dynamic 环境
        # 每个人起始位置不一样
        # self.travel_dis = [ random.random() *500 for i in range(self.cars)]
        # 每个人都在位置为0的地方开始
        self.travel_dis = [0] * self.cars
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i] - 500, 2) + 10 ** 2) for i in range(self.cars)]
        self.latency_itr = [0] * self.cars

        # 动作1-prob：0.5到1之间的实数
        # 动作2-bd*10：0.01到0.2之间的实数
        # 动作3-pwr*10：0.1到1之间的实数(后期会处理成0.01到0.1)

        action_low = np.concatenate([np.ones(1) * 0.5, np.zeros(self.cars), np.ones(self.cars) * 1])
        action_high = np.concatenate([np.ones(1) * 0.999999, np.ones(self.cars), np.ones(self.cars) * 32])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # 需要强调的是，bandwidth是0.01到0.2之间的实数，action3是0.01-0.1的实数，我们这里处理成0-1之间

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
    def reset(self, seed=0):
        self.current_step = 0
        self.obj1 = []
        self.obj2 = []
        # static
        self.pwr = [0.1] * self.cars
        self.fre = [2] * self.cars
        self.vel = [30] * self.cars
        self.rho = [1 / self.cars]

        # dynamic
        # 每个人起始位置不一样
        # self.travel_dis = [ random.random() *500 for i in range(self.cars)]
        # 每个人都在位置为0的地方开始
        self.travel_dis = [0] * self.cars
        # 每个人离rsu的勾股距离
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i] - 500, 2) + 10 ** 2) for i in range(self.cars)]
        # 每轮的时延
        self.latency_itr = [0]

        initial_observation = [i / 2 for i in self.fre] + [i / 30 for i in self.vel] + self.rho + [i / 500 for i in
                                                                                                   self.travel_dis] + [
                                  i / 500 for i in self.rsu_dis] + self.latency_itr
        return np.array(initial_observation), {}

    def step(self, action):  #

        # observation change
        prob = action[0]
        slct = action[1:self.cars + 1]
        quant = action[self.cars + 1:]

        # 根据等式约束赋值quant
        bd = [0 for i in range(self.cars)]
        # quant = [0 for i in range(self.cars)]
        for i in range(self.cars):
            bd[i] = quant[i] * np.log2(1 - self.pwr[i] * np.log(prob) * 1e7 / np.power(self.rsu_dis[i], 2)) \
                       * 32 * (1000-self.travel_dis[i])/self.vel[i]
            # if quant[i] > 32:
            #     quant[i] = 32
            # elif quant[i] < 1:
            #     quant[i] = 1
            # else:
            #     quant[i] = quant[i]

        # 更新state
        t_comp = 0.05  # 固定计算时间
        travel_dis_after_comp = [0] * self.cars  # 加上计算时延后车辆移动距离
        rate = [0] * self.cars  # 香浓公式速率
        for i in range(self.cars):
            travel_dis_after_comp[i] = self.travel_dis[i] + t_comp * self.vel[i]
            self.rsu_dis[i] = np.sqrt(np.power(travel_dis_after_comp[i] - 500, 2) + 100)
            rate[i] = bd[i] * 10 * np.log2(1 + 1e7 * self.pwr[i] * np.power(self.rsu_dis[i], -2))  # action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
        self.latency_itr[0] = np.max([t_comp + quant[i] / (32 * max(rate[i], 0.01)) for i in range(self.cars)])
        for i in range(self.cars):
            self.travel_dis[i] += self.latency_itr[0] * self.vel[i]  # 加上通信时延后车辆移动距离

        if self.debug:
            print('step {}/{}'.format(self.current_step+1, self.done_step))
            print('self.travel_dis:', self.travel_dis)
            print("self.rsu_dis:", self.rsu_dis)
            print("bandwidth:", list(bd))
            print("travel_dis_after_comp", travel_dis_after_comp)
            print('rate', rate)
            # print('pwr', list(pwr))
            print('quant', quant)

        # 计算两个目标
        obj1_temp = np.sum([slct[i]*self.rho[0] / np.power(np.power(2, (quant[i])) - 1, 2) for i in range(self.cars)])
        obj1_temp2 = np.sum(slct*prob)
        # obj1_temp2 = self.latency_itr[0]
        self.obj1.append(obj1_temp)
        self.obj2.append(obj1_temp2)
        obs = [i / 2 for i in self.fre] + [i / 30 for i in self.vel] + self.rho + [i / 500 for i in self.travel_dis] + [
            i / 500 for i in self.rsu_dis] + self.latency_itr
        # if np.any(np.isinf(obs)) or np.any(np.isnan(obs)):
        #     print ('obs')

        # check_termination
        self.current_step += 1
        done = False
        if self.current_step >= self.done_step:
            done = True

        reward = self.get_reward(prob, bd, quant, rate)
        return obs, reward, done, False, {}

    def get_reward(self, prob, bd, quant, rate):

        reward = 0

        # 不等式约束 sum(bd) < 1
        # if np.sum(bd) > 1:
        #     reward -= np.sum(bd)
        # else:
        #     reward -= (1 - np.sum(bd))

        # 变量范围 1 < quant < 32
        # if any(x > 32 for x in quant):
        #     reward -= sum(max(0, x - 32) for x in quant) / self.cars
        # if any(x < 1 for x in quant):
        #     reward -= sum(max(0, 1/x - 1) for x in quant) / self.cars

        # 帮助收敛  rate >0.01
        # if any(x < 0.01 for x in rate):
        #     reward -= sum(max(0, 0.01 - x) for x in rate) * 10

        # if self.current_step >= self.done_step:
        if self.current_step >= 0:  # 总是成立
            obj1 = np.sum(self.obj1)
            obj2 = np.sum(self.obj2)/10  # 调整两个目标的数量级
            reward -= self.weight_lambda*obj1
            reward = (1-self.weight_lambda)*obj2
            print('reward:{}, f1:{}, f2:{}'.format(reward, obj1, obj2))
        if self.current_step >= self.done_step:         # 在每个回合结束时执行，记录奖励值
            if self.obj_file:
                self.f.write(str(obj1)+"\t"+str(obj2)+"\t"+str(reward)+"\t"+str(bd)+"\t"+str(quant.tolist())+"\n")

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
