import gymnasium
import math
import numpy as np
from gymnasium import spaces
import random
"""
该环境中，共有8组设备，每组有10个设备，其中前3个设备产生时延敏感数据，后7个设备产生时延容忍数据
"""


class QFLEnv(gymnasium.Env):
    def __init__(self, cars=10, done_step=50, weight_lambda=1, debug=False, obj_file=None):  # done_step需要设置成2的次方，这样才能保证total_timesteps是他的整数倍
        super(QFLEnv, self).__init__()
        self.cars = cars
        self.current_step = 0
        self.done_step = done_step  # 联邦学习全局轮数
        self.weight_lambda = weight_lambda
        self.obj_file=obj_file
        self.qe_bit = 8
        self.latency_limit = [0.6]
        if self.obj_file:
            self.f = open(self.obj_file, 'w')
        # static 环境
        self.pwr = [0.1] * self.cars
        self.fre = [2] * self.cars
        self.vel = [30] * self.cars
        self.qe = [self.cars / np.power(np.power(2, self.qe_bit) - 1, 2)]
        self.obj1 = []
        self.obj2 = []
        self.debug = debug

        # dynamic 环境
        # self.travel_dis = [ random.random() *500 for i in range(self.cars)]  # 每个人起始位置不一样
        self.travel_dis = [0] * self.cars  # 每个人都在位置为0的地方开始
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i] - 500, 2) + 10 ** 2) for i in range(self.cars)]
        self.latency_itr = [0] * self.cars

        # 动作1-slc：0到1之间的实数
        # 动作2-bd：0.01到1之间的实数
        # 动作3-quant：1到32之间的实数

        # action_low = np.concatenate([np.ones(1) * 0.5, np.ones(self.cars) * 0.05, np.ones(self.cars) * 0.1])
        # action_high = np.concatenate([np.ones(1) * 0.999999, np.ones(self.cars), np.ones(self.cars)])
        action_low = np.concatenate([np.zeros(self.cars), np.ones(self.cars) * 0.01])
        action_high = np.concatenate([np.ones(self.cars), np.ones(self.cars)])
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
        self.qe = [1/np.power(np.power(2, self.qe_bit) - 1, 2)]

        # dynamic
        # self.travel_dis = [ random.random() *500 for i in range(self.cars)]  # 每个人起始位置不一样
        self.travel_dis = [0] * self.cars  # 每个人都在位置为0的地方开始
        self.rsu_dis = [np.sqrt(np.power(self.travel_dis[i] - 500, 2) + 10 ** 2) for i in range(self.cars)]  # 每个人离rsu的勾股距离
        self.latency_itr = [0]  # 每轮的时延

        initial_observation = [i / 0.1 for i in self.pwr] + [i / 30 for i in self.vel] + self.latency_limit + [
            i / 500 for i in self.travel_dis] + [i / 500 for i in self.rsu_dis] + self.latency_itr
        return np.array(initial_observation), {}

    def step(self, action):  #

        # observation change
        slct_before = action[:self.cars]
        bd_before = action[self.cars:self.cars *2]
        # quant_before = action[self.cars*2:self.cars *3] *32

        slct =np.where(slct_before > 0.5, 1, 0)
        bd = slct*bd_before/np.sum(slct*bd_before) if np.sum(slct) >0 else bd_before/np.sum(bd_before)
        quant = [self.qe_bit] * self.cars

        # 更新state
        t_comp = 0.05  # 固定计算时间
        travel_dis_after_comp = [0] * self.cars  # 加上计算时延后车辆移动距离
        rate = [0] * self.cars  # 香浓公式速率
        latency_actual = [0] * self.cars  # 通信时延
        for i in range(self.cars):
            travel_dis_after_comp[i] = self.travel_dis[i] + t_comp * self.vel[i]
            self.rsu_dis[i] = np.sqrt(np.power(travel_dis_after_comp[i] - 500, 2) + 100)
            rate[i] = bd[i] * 10 * np.log2(1 + 1e7 * self.pwr[i] * np.power(self.rsu_dis[i], -2))  # action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
            latency_actual[i] = quant[i] / 32 / rate[i] if bd[i] > 0 else self.latency_limit[0]+1  # 避免分母为0，如果分母为0，直接把latency标的很大
            # 用户选择，把约束作为规则放进来
            if slct[i] == 1 and \
                    (latency_actual[i] >= self.latency_limit or
                     travel_dis_after_comp[i]+latency_actual[i]*self.vel[i] >= 1000 or
                     1 / np.power(np.power(2, quant[i]) - 1, 2) > self.qe):
                slct[i] = 0
        self.latency_itr[0] = np.max([slct[i]*(t_comp + latency_actual[i]) for i in range(self.cars)]) if np.sum(slct)>0 \
            else t_comp +self.latency_limit[0]
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
        obj1_temp = np.sum(slct)
        obj1_temp2 = self.latency_itr[0]
        self.obj1.append(obj1_temp)
        self.obj2.append(obj1_temp2)
        obs = [i / 0.1 for i in self.pwr] + [i / 30 for i in self.vel] + self.latency_limit + [i / 500 for i in self.travel_dis] + [
            i / 500 for i in self.rsu_dis] + self.latency_itr

        # normalize两个目标
        obj2_min = t_comp +self.qe_bit / (32 * 10 * np.log2(1 + 1e7 * self.pwr[i] / np.power(10, 2)))
        obj2_max = t_comp +self.latency_limit[0]

        # check_termination
        self.current_step += 1
        done = False
        if self.current_step >= self.done_step:
            done = True

        reward = self.get_reward(slct, bd, quant, obj2_min, obj2_max)
        return obs, reward, done, False, {}

    def get_reward(self, slct, bd, quant, obj2_min, obj2_max):

        reward = 0
        # 调参3个权重
        a=1
        # b=50
        # c=1

        # 不等式约束 sum(bd) < 1
        # sum_bd = np.sum(slct*bd)
        # if sum_bd > 1:
        #     reward -= (sum_bd - 1)/self.cars *b
        # else:
        #     reward -= 1 -sum_bd  # 不仅要保证小于1，还尽可能靠近1

        # QE不等式约束
        # qe = np.sum(slct / np.power(np.power(2, quant) - 1, 2))
        # if qe > self.qe:
        #     reward -= qe/(self.qe*self.cars) *c
        # else:
        #     reward -= qe/(self.qe*self.cars)  # 不仅要保证小于qe ，还尽可能靠近qe


        # if self.current_step >= self.done_step:
        if self.current_step >= 0:  # 总是成立
            episode = len(self.obj1)
            accumulated_obj1 = np.sum(np.array(self.obj1))
            accumulated_obj2 = np.sum(np.array(self.obj2))
            obj1 = np.sum(np.array(self.obj1)/(self.cars * episode)) * a # normlize obj1_max = self.cars
            obj2 = np.sum((np.array(self.obj2)-obj2_min)/((obj2_max-obj2_min)*episode))  # normlize
            # obj1 = np.sum(self.obj1)
            # obj2 = np.sum(self.obj2)*20
            reward = self.weight_lambda*obj1
            reward -= (1-self.weight_lambda)*obj2
            # print('reward:{}, f1:{}, f2:{}'.format(reward, obj1, obj2))
        if self.current_step >= self.done_step:         # 在每个回合结束时执行，记录奖励值
            if self.obj_file:
                self.f.write(str(accumulated_obj1)+"\t"+str(accumulated_obj2)+"\t"+str(reward)+"\t"+str(slct.tolist())+"\t"+str(bd.tolist())+"\n")

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
